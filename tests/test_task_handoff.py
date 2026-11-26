"""M6 R7/R8：队列指派、attempt 记录、交接与换 worker 重试。"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from keeper.cli import main
from keeper.config import Config
from keeper.llm import LLMResponse
from keeper.projects import Project, register_project
from keeper.session import SESSIONS_DIR
from keeper.task_runner import run_task_worker
from keeper.tasks import TASKS_FILE, TaskStore, TaskStoreError


class Approver:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer

    def confirm(self, request) -> bool:
        return self.answer


class FakeLLM:
    def __init__(self, text: str = "done", *, submit: bool = True) -> None:
        self.text = text
        self.submit = submit
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None) -> LLMResponse:
        self.calls.append(messages)
        if self.submit and len(self.calls) == 1:
            return LLMResponse(message={"role": "assistant", "content": "", "tool_calls": [
                {"id": "submit-1", "type": "function", "function": {
                    "name": "submit_result", "arguments": json.dumps({"summary": "交付结果"})}}
            ]}, usage={})
        return LLMResponse(message={"role": "assistant", "content": self.text}, usage={})


def project(root: Path) -> Project:
    return Project("demo", "Demo", root, ("built-in", "codex"), "built-in")


def codex(monkeypatch, code: str) -> None:
    monkeypatch.setenv(
        "KEEPER_WORKER_CODEX_ARGV", json.dumps([sys.executable, "-c", code, "{input}"])
    )


def _block(store: TaskStore, title: str = "阻塞任务", *, worker: str | None = None) -> str:
    task = store.add(title, worker=worker)
    store.start_attempt(task.id, "old-session", worker=worker)
    store.block(task.id, "缺少接口文档")
    return task.id


def test_old_tasks_json_missing_new_fields_loads_without_rewriting(tmp_path: Path) -> None:
    legacy = {
        "version": 1,
        "tasks": [
            {
                "id": "t-old",
                "title": "旧任务",
                "status": "blocked",
                "created_at": "2026-01-01T00:00:00+08:00",
                "updated_at": "2026-01-01T00:00:00+08:00",
                "block_reason": "缺信息",
                "attempts": [
                    {
                        "session_id": "s-old",
                        "started_at": "2026-01-01T00:00:00+08:00",
                        "ended_at": "2026-01-01T00:01:00+08:00",
                        "outcome": "blocked",
                        "block_reason": "缺信息",
                    }
                ],
            }
        ],
    }
    path = tmp_path / TASKS_FILE
    path.parent.mkdir(parents=True)
    raw = json.dumps(legacy, ensure_ascii=False)
    path.write_text(raw, encoding="utf-8")

    store = TaskStore.load(tmp_path)
    task = store.get("t-old")
    assert task is not None and task.worker is None
    assert task.attempts[0].result is None
    assert task.attempts[0].worker is None
    assert task.attempts[0].summary is None
    assert task.attempts[0].transcript is None
    # 只读加载不改写原文件。
    assert path.read_text(encoding="utf-8") == raw

    store.save()
    data = json.loads(path.read_text(encoding="utf-8"))
    saved = data["tasks"][0]
    assert saved["worker"] is None
    for key in ("worker", "result", "summary", "transcript"):
        assert saved["attempts"][0][key] is None
    assert saved["block_reason"] == "缺信息"


def test_add_with_worker_persists_and_attempt_records_done(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("写测试", worker="codex")
    assert task.worker == "codex"

    store.start_attempt(task.id, "s-1", worker="codex")
    store.complete(task.id, summary="完成", transcript="/tmp/runs/1.log")
    attempt = store.get(task.id).attempts[-1]
    assert (attempt.worker, attempt.result, attempt.summary) == ("codex", "done", "完成")
    assert attempt.transcript == "/tmp/runs/1.log"
    assert store.get(task.id).status == "done"


def test_attempt_records_blocked_failed_and_interrupted(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    blocked = store.add("阻塞")
    store.start_attempt(blocked.id, "s-b")
    store.block(blocked.id, "缺参数")
    attempt = store.get(blocked.id).attempts[-1]
    assert (attempt.result, attempt.summary) == ("blocked", "缺参数")

    failed = store.add("失败")
    store.start_attempt(failed.id, "s-f", worker="codex")
    store.fail(failed.id, "退出码 7", transcript="/tmp/runs/f.log")
    attempt = store.get(failed.id).attempts[-1]
    assert (attempt.outcome, attempt.result, attempt.summary) == ("failed", "failed", "退出码 7")
    assert store.get(failed.id).status == "blocked"
    assert store.get(failed.id).block_reason == "退出码 7"

    interrupted = store.add("中断")
    store.start_attempt(interrupted.id, "s-i")
    store.fail(interrupted.id, "worker 超时", result="interrupted")
    assert store.get(interrupted.id).attempts[-1].result == "interrupted"
    assert store.get(interrupted.id).status == "blocked"


def test_retry_requires_blocked_status(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    queued = store.add("排队中")
    with pytest.raises(TaskStoreError):
        store.retry_attempt(queued.id)
    task = store.add("已完成")
    store.start_attempt(task.id, "s")
    store.complete(task.id)
    with pytest.raises(TaskStoreError):
        store.retry_attempt(task.id)


def test_retry_with_builtin_worker_handoff_and_new_session(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task_id = _block(store, "补完接口")
    llm = FakeLLM("done")

    record = run_task_worker(
        project(tmp_path),
        store,
        task_id,
        "built-in",
        retry=True,
        approver=Approver(),
        llm=llm,
    )

    task = store.get(task_id)
    assert record.result == "done" and task.status == "done"
    attempt = task.attempts[-1]
    assert attempt.worker == "built-in" and attempt.result == "done"
    assert attempt.summary and attempt.transcript and attempt.session_id
    transcript = Path(attempt.transcript)
    assert transcript.is_file()
    assert transcript.parent == (tmp_path / SESSIONS_DIR).resolve()
    assert attempt.session_id != "old-session"

    context = llm.calls[0][-1]["content"]
    assert "上一轮完成记录" in context
    assert "缺少接口文档" in context
    assert "task " + task_id in context


def test_builtin_worker_without_submit_result_fails(tmp_path: Path) -> None:
    store = TaskStore.load(tmp_path)
    task_id = _block(store, "需要交付")
    record = run_task_worker(project(tmp_path), store, task_id, "built-in", retry=True,
                             approver=Approver(), llm=FakeLLM("我做完了", submit=False))
    assert record.result == "failed"
    assert "未交付结果" in record.summary and "submit_result" in record.summary
    assert store.get(task_id).status == "blocked"


def test_retry_switches_worker_and_keeps_prior_completion_record(
    tmp_path: Path, monkeypatch
) -> None:
    store = TaskStore.load(tmp_path)
    task_id = _block(store, "实现解析器", worker="built-in")
    seen = tmp_path / "seen-context.txt"
    codex(
        monkeypatch,
        f"import sys; open({str(seen)!r}, 'w', encoding='utf-8').write(sys.stdin.read()); print('ok')",
    )

    record = run_task_worker(
        project(tmp_path), store, task_id, "codex", retry=True, approver=Approver()
    )

    task = store.get(task_id)
    assert record.result == "done" and task.status == "done"
    assert task.worker == "codex"
    attempt = task.attempts[-1]
    assert (attempt.worker, attempt.result) == ("codex", "done")
    assert Path(attempt.transcript).is_file()
    assert attempt.session_id is None  # 外部 CLI 没有 keeper 会话

    context = seen.read_text(encoding="utf-8")
    assert "上一轮完成记录" in context
    assert "worker=built-in" in context
    assert "缺少接口文档" in context
    assert "实现解析器" in context


def test_retry_recap_only_includes_messages_after_last_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    store = TaskStore.load(tmp_path)
    task_id = _block(store, "继续")
    ended = datetime.fromtimestamp(time.time() - 500).astimezone().isoformat(
        timespec="seconds"
    )
    task = store.get(task_id)
    task.attempts[-1].ended_at = ended
    # 同一个 worker 再来一次：只补上次结束之后的增量。
    task.attempts[-1].worker = "codex"
    store.save()

    sessions = tmp_path / SESSIONS_DIR
    sessions.mkdir(parents=True, exist_ok=True)
    old_ts = datetime.fromtimestamp(time.time() - 1000).timestamp()
    new_ts = datetime.fromtimestamp(time.time() - 10).timestamp()
    records = [
        {"type": "meta", "version": 1, "session_id": "s-recap", "created_at": ended},
        {"type": "message", "ts": old_ts, "message": {"role": "user", "content": "OLD-BEFORE"}},
        {"type": "message", "ts": new_ts, "message": {"role": "user", "content": "NEW-AFTER"}},
    ]
    (sessions / "s-recap.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    seen = tmp_path / "seen-recap.txt"
    codex(
        monkeypatch,
        f"import sys; open({str(seen)!r}, 'w', encoding='utf-8').write(sys.stdin.read()); print('ok')",
    )
    run_task_worker(project(tmp_path), store, task_id, "codex", retry=True, approver=Approver())

    context = seen.read_text(encoding="utf-8")
    assert "NEW-AFTER" in context
    assert "OLD-BEFORE" not in context


def test_retry_recap_uses_last_attempt_session_not_latest(
    tmp_path: Path, monkeypatch
) -> None:
    store = TaskStore.load(tmp_path)
    task_id = _block(store, "继续")
    ended = datetime.fromtimestamp(time.time() - 500).astimezone().isoformat(
        timespec="seconds"
    )
    task = store.get(task_id)
    task.attempts[-1].ended_at = ended
    store.save()

    sessions = tmp_path / SESSIONS_DIR
    sessions.mkdir(parents=True, exist_ok=True)
    # 消息时间早于上次 attempt 的 ended_at；换 worker 时仍必须给整段。
    old_ts = datetime.fromtimestamp(time.time() - 1000).timestamp()

    def write_session(session_id: str, text: str) -> None:
        records = [
            {"type": "meta", "version": 1, "session_id": session_id, "created_at": ended},
            {"type": "message", "ts": old_ts, "message": {"role": "user", "content": text}},
        ]
        (sessions / f"{session_id}.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
            encoding="utf-8",
        )

    # 上次 attempt 挂在 sess-a（另一个 worker）；项目里最新会话是 sess-b。
    write_session("sess-a", "PROGRESS-IN-A")
    write_session("sess-b", "PROGRESS-IN-B")
    task = store.get(task_id)
    task.attempts[-1].session_id = "sess-a"
    task.attempts[-1].worker = "built-in"
    store.save()

    seen = tmp_path / "seen-recap.txt"
    codex(
        monkeypatch,
        f"import sys; open({str(seen)!r}, 'w', encoding='utf-8').write(sys.stdin.read()); print('ok')",
    )
    run_task_worker(project(tmp_path), store, task_id, "codex", retry=True, approver=Approver())

    context = seen.read_text(encoding="utf-8")
    assert "PROGRESS-IN-A" in context
    assert "PROGRESS-IN-B" not in context


def test_first_dispatch_recap_includes_whole_latest_session(
    tmp_path: Path, monkeypatch
) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("首次派发")

    sessions = tmp_path / SESSIONS_DIR
    sessions.mkdir(parents=True, exist_ok=True)
    old_ts = datetime.fromtimestamp(time.time() - 1000).timestamp()
    records = [
        {"type": "meta", "version": 1, "session_id": "s-first", "created_at": "2026-01-01T00:00:00+08:00"},
        {"type": "message", "ts": old_ts, "message": {"role": "user", "content": "FIRST-CONTEXT"}},
    ]
    (sessions / "s-first.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    seen = tmp_path / "seen-first.txt"
    codex(
        monkeypatch,
        f"import sys; open({str(seen)!r}, 'w', encoding='utf-8').write(sys.stdin.read()); print('ok')",
    )
    run_task_worker(project(tmp_path), store, task.id, "codex", retry=False, approver=Approver())

    assert "FIRST-CONTEXT" in seen.read_text(encoding="utf-8")


def test_cli_add_worker_validates_roster(tmp_path: Path, capsys) -> None:
    register_project(tmp_path, project_id="demo", workers=("built-in",))
    assert main(["task", "add", "编码", "--worker", "codex", "--root", str(tmp_path)]) == 2
    assert "codex" in capsys.readouterr().err
    assert not (tmp_path / TASKS_FILE).exists()

    assert main(["task", "add", "编码", "--worker", "built-in", "--root", str(tmp_path)]) == 0
    store = TaskStore.load(tmp_path)
    assert store.tasks[0].worker == "built-in"


def test_cli_retry_rejects_non_blocked_and_missing(tmp_path: Path, capsys) -> None:
    store = TaskStore.load(tmp_path)
    assert main(["task", "retry", "nope", "--root", str(tmp_path)]) == 2
    assert "不存在" in capsys.readouterr().err

    task = store.add("排队中")
    assert main(["task", "retry", task.id, "--root", str(tmp_path)]) == 2
    assert "blocked" in capsys.readouterr().err


def test_cli_run_assigned_task_dispatches_worker(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    register_project(tmp_path, project_id="demo", workers=("built-in", "codex"))
    store = TaskStore.load(tmp_path)
    task = store.add("跑一次", worker="codex")
    codex(monkeypatch, "print('ok')")
    monkeypatch.setattr(
        "keeper.cli.load_config",
        lambda **kwargs: Config(
            base_url="http://example.invalid",
            api_key="",
            model="fake-model",
            max_steps=5,
        ),
    )
    monkeypatch.setattr("keeper.cli.StdioApprover", lambda **kwargs: Approver())

    assert main(["task", "run", task.id, "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "worker=codex" in out and "result=done" in out
    done = TaskStore.load(tmp_path).get(task.id)
    assert done.status == "done"
    assert done.attempts[-1].result == "done"


def test_cli_run_unassigned_task_still_uses_manager_loop(
    tmp_path: Path, monkeypatch
) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("普通任务")

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.responses = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "complete_task", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "assistant", "content": "完成"},
            ]

        def chat(self, messages, tools=None) -> LLMResponse:
            return LLMResponse(message=self.responses.pop(0), usage={})

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "keeper.cli.load_config",
        lambda **kwargs: Config(
            base_url="http://example.invalid",
            api_key="test-key",
            model="fake-model",
            max_steps=5,
        ),
    )
    monkeypatch.setattr("keeper.cli.LLMClient", FakeClient)

    assert main(["task", "run", task.id, "--root", str(tmp_path)]) == 0
    done = TaskStore.load(tmp_path).get(task.id)
    assert done.status == "done"
    assert done.attempts[-1].result == "done"
    assert done.attempts[-1].worker is None
