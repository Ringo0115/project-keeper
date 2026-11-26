import json
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import pytest

from keeper.approval import Decision, review
from keeper.events import TaskEvent, ToolResultEvent
from keeper.llm import LLMResponse
from keeper.prompts import TaskContext, system_prompt
from keeper.session import Session
from keeper.task_runner import TaskRunError, prepare_task_run, run_task_once
from keeper.tasks import TASKS_FILE, TaskStore, TaskStoreError
from keeper.tools.fs import ReadFileTool
from keeper.tools.registry import Registry
from keeper.tools.tasks import BlockTaskTool


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of responses")
        return LLMResponse(message=self._responses.pop(0), usage={})


class BrokenLLM:
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        raise RuntimeError("network down")


def _final(text: str = "done") -> dict:
    return {"role": "assistant", "content": text}


def _tool_call(name: str, arguments: str = "{}", call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tasks_path(tmp_path: Path) -> Path:
    return tmp_path / TASKS_FILE


def _run(
    store: TaskStore,
    task_id: str,
    root: Path,
    responses: list[dict],
    *,
    max_steps: int = 5,
    llm: object | None = None,
):
    prepared = prepare_task_run(store, task_id, root, "test-model")
    fake = llm or FakeLLM(responses)
    try:
        events = list(
            run_task_once(
                prepared,
                llm=fake,  # type: ignore[arg-type]
                registry=Registry([ReadFileTool(root)]),
                approver=None,
                max_steps=max_steps,
            )
        )
    finally:
        prepared.session.close()
    return prepared, events


# ---------------------------------------------------------------------------
# 1. 持久化 round-trip
# ---------------------------------------------------------------------------


def test_add_load_roundtrip_keeps_queue_order(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    first = store.add("第一个任务")
    second = store.add("第二个任务")

    assert first.id.startswith("t-")
    assert first.status == "queued"
    assert [task.title for task in store.tasks] == ["第一个任务", "第二个任务"]

    data = json.loads(_tasks_path(tmp_path).read_text(encoding="utf-8"))
    assert data["version"] == 1
    raw = data["tasks"][0]
    assert set(raw) == {
        "id",
        "title",
        "status",
        "created_at",
        "updated_at",
        "block_reason",
        "worker",
        "attempts",
    }
    assert raw["worker"] is None
    assert raw["attempts"] == []
    assert datetime.fromisoformat(raw["created_at"]).utcoffset() is not None

    reloaded = TaskStore.load(tmp_path)
    assert [task.id for task in reloaded.tasks] == [first.id, second.id]
    assert [task.title for task in reloaded.tasks] == ["第一个任务", "第二个任务"]


# ---------------------------------------------------------------------------
# 2. start_attempt 与单 running 约束
# ---------------------------------------------------------------------------


def test_start_attempt_opens_attempt_and_rejects_second_running(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    first = store.add("先跑")
    second = store.add("后跑")

    attempt = store.start_attempt(first.id, "session-a")

    assert store.get(first.id).status == "running"
    assert attempt.session_id == "session-a"
    assert attempt.ended_at is None
    assert attempt.outcome is None
    assert attempt.block_reason is None

    with pytest.raises(TaskStoreError):
        store.start_attempt(second.id, "session-b")

    reloaded = TaskStore.load(tmp_path)
    assert reloaded.get(first.id).status == "running"
    assert reloaded.get(second.id).status == "queued"
    assert reloaded.get(second.id).attempts == []


# ---------------------------------------------------------------------------
# 3/4. 启动恢复：第一次中断回队首；真实子进程 terminate
# ---------------------------------------------------------------------------


def test_first_recovery_requeues_interrupted_task_to_head(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    front = store.add("前排任务")
    target = store.add("目标任务")
    store.start_attempt(target.id, "session-1")

    reloaded = TaskStore.load(tmp_path)  # 模拟进程重启
    changes = reloaded.recover()

    assert len(changes) == 1
    assert changes[0].task_id == target.id
    assert changes[0].status == "queued"
    assert [task.id for task in reloaded.tasks] == [target.id, front.id]
    assert reloaded.tasks[1].id == front.id
    assert reloaded.tasks[0].status == "queued"
    assert reloaded.tasks[1].status == "queued"

    attempt = reloaded.get(target.id).attempts[-1]
    assert attempt.outcome == "interrupted"
    assert attempt.ended_at is not None
    assert attempt.block_reason is None

    assert reloaded.recover() == []


def test_real_subprocess_terminate_then_recover(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    child = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        from keeper.tasks import TaskStore

        root = Path(sys.argv[1])
        store = TaskStore.load(root)
        task = store.add("被 kill 的任务")
        store.start_attempt(task.id, "session-killed")
        print("READY", flush=True)
        time.sleep(60)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(repo_root),
    )
    try:
        line = proc.stdout.readline() if proc.stdout else ""
        assert "READY" in line, f"child did not start: {line!r}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    store = TaskStore.load(tmp_path)
    task = store.tasks[0]
    assert task.status == "running"
    assert task.attempts[-1].ended_at is None
    assert task.attempts[-1].outcome is None

    changes = store.recover()

    task = store.get(task.id)
    assert changes and changes[0].task_id == task.id
    assert task.status == "queued"
    assert task.attempts[-1].outcome == "interrupted"
    assert task.attempts[-1].ended_at is not None


# ---------------------------------------------------------------------------
# 5. 连续第二次中断 → blocked
# ---------------------------------------------------------------------------


def test_second_consecutive_interruption_blocks_task(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("总是被打断")

    store.start_attempt(task.id, "session-1")
    store = TaskStore.load(tmp_path)
    store.recover()
    assert store.get(task.id).status == "queued"

    store.start_attempt(task.id, "session-2")
    store = TaskStore.load(tmp_path)
    changes = store.recover()

    task = store.get(task.id)
    assert changes and changes[0].status == "blocked"
    assert task.status == "blocked"
    assert "连续 2 次中断" in (task.block_reason or "")
    assert task.attempts[-1].outcome == "interrupted"

    # 第三次启动：不会再被排队或运行。
    assert store.recover() == []
    assert store.next_queued() is None
    with pytest.raises(TaskRunError):
        prepare_task_run(store, task.id, tmp_path, "test-model")


# ---------------------------------------------------------------------------
# 6. 纯文本不算完成；显式 complete 才算
# ---------------------------------------------------------------------------


def test_incomplete_breaks_interrupted_streak(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("先没做完，再被 kill")
    store.start_attempt(task.id, "session-1")
    store.mark_incomplete(task.id)
    store.start_attempt(task.id, "session-2")

    store = TaskStore.load(tmp_path)
    store.recover()

    task = store.get(task.id)
    assert task.status == "queued"
    assert task.attempts[-1].outcome == "interrupted"


def test_plain_final_is_incomplete_then_explicit_complete(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("完成任务")

    prepared, events = _run(store, task.id, tmp_path, [_final("做完了")])

    task = store.get(task.id)
    assert task.status == "queued"
    assert task.attempts[-1].outcome == "incomplete"
    assert any(
        isinstance(event, TaskEvent) and event.action == "incomplete"
        for event in events
    )
    # 首次 attempt 的消息不带续跑提示。
    assert prepared.resumed is False
    assert prepared.message == f"[task {task.id}] 完成任务"

    prepared2, events2 = _run(
        store,
        task.id,
        tmp_path,
        [_tool_call("complete_task"), _final("完成了")],
    )

    task = store.get(task.id)
    assert task.status == "done"
    assert task.attempts[-1].outcome == "done"
    assert task.block_reason is None
    assert prepared2.resumed is True
    assert any(
        isinstance(event, TaskEvent) and event.action == "complete"
        for event in events2
    )


# ---------------------------------------------------------------------------
# 7/8. 空 block reason 拒绝
# ---------------------------------------------------------------------------


def test_block_task_empty_reason_keeps_file_bytes(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("不许空原因")
    tool = BlockTaskTool(store, task.id)
    before = _tasks_path(tmp_path).read_bytes()

    for reason in ("", "   ", "\t\n"):
        result = tool.execute({"reason": reason})
        assert not result.ok
        assert result.content.startswith("ERROR:")

    result = tool.execute({})
    assert not result.ok

    assert _tasks_path(tmp_path).read_bytes() == before
    assert store.get(task.id).status == "queued"
    assert store.get(task.id).block_reason is None


def test_block_task_via_loop_with_empty_reason_does_not_block(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("loop 内空原因")
    prepared = prepare_task_run(store, task.id, tmp_path, "test-model")
    llm = FakeLLM(
        [
            _tool_call("block_task", json.dumps({"reason": "   "})),
            _final("ok"),
        ]
    )
    events: list = []
    stream = run_task_once(
        prepared,
        llm=llm,
        registry=Registry([ReadFileTool(tmp_path)]),
        approver=None,
        max_steps=5,
    )
    try:
        for event in stream:
            events.append(event)
            if isinstance(event, ToolResultEvent) and event.name == "block_task":
                break
    finally:
        stream.close()
        prepared.session.close()

    result = next(
        event for event in events if isinstance(event, ToolResultEvent)
    )
    assert not result.ok
    assert result.output.startswith("ERROR:")

    task = store.get(task.id)
    assert task.status == "running"
    assert task.block_reason is None
    assert task.attempts[-1].outcome is None


def test_task_tools_are_allowed_without_asking(tmp_path: Path) -> None:
    for name in ("complete_task", "block_task"):
        verdict = review(
            tool_name=name,
            danger="task",
            args={"reason": "x"},
            root=tmp_path,
        )
        assert verdict.decision is Decision.ALLOW


# ---------------------------------------------------------------------------
# 9. 精确加载任务引用的旧 session（不是最近 session）+ 续跑消息
# ---------------------------------------------------------------------------


def test_resume_loads_referenced_session_not_latest(tmp_path: Path) -> None:
    older = Session.create(tmp_path, "m")
    older_id = older.session_id
    older_path = older.path
    older.close()
    time.sleep(0.01)
    newer = Session.create(tmp_path, "m")
    newer_path = newer.path
    newer.close()
    assert newer_path > older_path

    store = TaskStore.load(tmp_path)
    task = store.add("继续修复 login")
    store.start_attempt(task.id, older_id)
    store.mark_incomplete(task.id)

    prepared, _ = _run(store, task.id, tmp_path, [_final("还没完成")])

    assert prepared.session.session_id == older_id
    assert prepared.session.path == older_path
    assert prepared.resumed is True
    assert prepared.message.startswith(f"[task {task.id}] 继续修复 login")
    assert "上一个 attempt 没有显式完成" in prepared.message
    assert "完成时必须调用 complete_task" in prepared.message

    # 续跑消息确实写进了 session。
    assert any(
        message.get("role") == "user"
        and "上一个 attempt 没有显式完成" in (message.get("content") or "")
        for message in prepared.session.messages
    )
    assert store.get(task.id).attempts[-1].session_id == older_id
    # 更新的 session B 没有被碰。
    assert newer_path.exists()
    assert Session.load(newer_path, tmp_path).messages == []


# ---------------------------------------------------------------------------
# 10. 跑任务前修复 dangling tool_call
# ---------------------------------------------------------------------------


def test_prepare_repairs_dangling_tool_call(tmp_path: Path) -> None:
    session = Session.create(tmp_path, "m")
    session_id = session.session_id
    session.append({"role": "user", "content": "读一下"})
    session.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }
    )
    session.close()

    store = TaskStore.load(tmp_path)
    task = store.add("修复尾巴")
    store.start_attempt(task.id, session_id)
    store.mark_incomplete(task.id)

    prepared = prepare_task_run(store, task.id, tmp_path, "m")
    try:
        assert prepared.repaired == 1
        last = prepared.session.messages[-1]
        assert last["role"] == "tool"
        assert last["tool_call_id"] == "call_1"
        assert last["content"].startswith("[interrupted]")
    finally:
        prepared.session.close()


# ---------------------------------------------------------------------------
# 10b. session 文件丢失：queued → blocked，不产生假 attempt
# ---------------------------------------------------------------------------


def test_missing_session_blocks_before_start(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("引用丢失的 session")
    missing = "20200101-000000-000000-deadbe"
    store.start_attempt(task.id, missing)
    store.mark_incomplete(task.id)
    before = _tasks_path(tmp_path).read_bytes()

    with pytest.raises(TaskRunError) as excinfo:
        prepare_task_run(store, task.id, tmp_path, "m")

    task = store.get(task.id)
    assert task.status == "blocked"
    assert "session 文件丢失" in (task.block_reason or "")
    assert missing in (task.block_reason or "")
    assert len(task.attempts) == 1  # 没有新增 attempt

    # 启动前失败不改变已有的档案内容，只更新任务状态。
    assert _tasks_path(tmp_path).read_bytes() != before
    reloaded = TaskStore.load(tmp_path)
    assert reloaded.get(task.id).status == "blocked"
    assert "session 文件丢失" in str(excinfo.value)
    assert not (tmp_path / ".agent" / "sessions").exists()


# ---------------------------------------------------------------------------
# 11. 损坏文件：明确报错、不覆盖
# ---------------------------------------------------------------------------


def test_corrupt_tasks_file_errors_without_overwrite(tmp_path: Path) -> None:
    path = _tasks_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": 1, "tasks": [', encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(TaskStoreError) as excinfo:
        TaskStore.load(tmp_path)

    assert "损坏" in str(excinfo.value)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# 12. 原子写失败：原文件不变
# ---------------------------------------------------------------------------


def test_atomic_write_failure_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TaskStore.load(tmp_path)
    store.add("原来的任务")
    path = _tasks_path(tmp_path)
    before = path.read_bytes()

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("keeper.tasks.os.replace", boom)
    store.tasks[0].title = "改坏的任务"
    with pytest.raises(TaskStoreError):
        store.save()

    assert path.read_bytes() == before
    assert list(path.parent.glob("tasks.json.tmp*")) == []


# ---------------------------------------------------------------------------
# 8. LLM 错误 / max_steps 不收尾
# ---------------------------------------------------------------------------


def test_llm_error_leaves_attempt_open(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("模型错误")
    _, events = _run(store, task.id, tmp_path, [], llm=BrokenLLM())

    task = store.get(task.id)
    assert task.status == "running"
    assert task.attempts[-1].ended_at is None
    assert task.attempts[-1].outcome is None
    assert any(
        isinstance(event, TaskEvent) and event.action == "left-open"
        for event in events
    )

    changes = store.recover()
    assert changes
    assert store.get(task.id).attempts[-1].outcome == "interrupted"


def test_max_steps_leaves_attempt_open(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("停不下来")
    responses = [
        _tool_call("read_file", json.dumps({"path": "missing.txt"}), "call_1"),
        _tool_call("read_file", json.dumps({"path": "missing.txt"}), "call_2"),
    ]
    _, events = _run(store, task.id, tmp_path, responses, max_steps=1)

    task = store.get(task.id)
    assert task.status == "running"
    assert task.attempts[-1].outcome is None
    assert isinstance(events[-1], TaskEvent)
    assert events[-1].action == "left-open"


# ---------------------------------------------------------------------------
# 恢复防御：running 但没有打开中的 attempt
# ---------------------------------------------------------------------------


def test_task_block_comes_after_memory_in_prompt(tmp_path: Path) -> None:
    memory = tmp_path / ".agent" / "memory" / "PROJECT_STATE.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("# PROJECT_STATE\n\n- 当前状态。\n", encoding="utf-8")

    prompt = system_prompt(
        tmp_path,
        TaskContext(
            id="t-20261122-153012-a1b2c3",
            title="修复 login",
            attempt_number=2,
            session_id="s-1",
        ),
    )

    assert prompt.index("## 项目记忆 / PROJECT_STATE.md") < prompt.index(
        "# 当前任务"
    )
    assert "第 2 次 attempt" in prompt
    assert "优先于上面的通用规则" in prompt
    assert "先调用 complete_task" in prompt
    assert "block_task" in prompt
    # 默认 None 保持旧行为：结尾仍是 PROJECT_STATE 内容。
    assert system_prompt(tmp_path).rstrip().endswith("当前状态。")


def test_recover_running_without_open_attempt_blocks(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("坏状态")
    task.status = "running"
    store.save()

    reloaded = TaskStore.load(tmp_path)
    changes = reloaded.recover()

    task = reloaded.get(task.id)
    assert changes and changes[0].status == "blocked"
    assert task.status == "blocked"
    assert "状态损坏" in (task.block_reason or "")


# ---------------------------------------------------------------------------
# 加载不变量：明显自相矛盾的状态直接报错
# ---------------------------------------------------------------------------


def _raw_task(status: str, block_reason=None, attempts=()) -> dict:
    return {
        "id": "t-20261122-153012-a1b2c3",
        "title": "手工改坏",
        "status": status,
        "created_at": "2026-11-22T15:30:12+08:00",
        "updated_at": "2026-11-22T15:30:12+08:00",
        "block_reason": block_reason,
        "attempts": list(attempts),
    }


def _open_attempt() -> dict:
    return {
        "session_id": "s-1",
        "started_at": "2026-11-22T15:30:13+08:00",
        "ended_at": None,
        "outcome": None,
        "block_reason": None,
    }


@pytest.mark.parametrize(
    "raw",
    [
        _raw_task("blocked", None),
        _raw_task("queued", None, [_open_attempt()]),
        _raw_task("done", "不该有"),
    ],
)
def test_load_rejects_inconsistent_states(tmp_path: Path, raw: dict) -> None:
    path = _tasks_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "tasks": [raw]}, ensure_ascii=False),
        encoding="utf-8",
    )
    before = path.read_bytes()

    with pytest.raises(TaskStoreError) as excinfo:
        TaskStore.load(tmp_path)

    assert "损坏" in str(excinfo.value)
    assert path.read_bytes() == before