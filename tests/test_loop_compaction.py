"""M4 loop 集成：阈值、摘要调用、JSONL compaction、resume 与失败降级。

全部使用 fake LLM / 本地工具，不联网。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from keeper.compaction import SUMMARY_SECTIONS
from keeper.events import CompactionDone, CompactionFailed, CompactionStart, FinalAnswer
from keeper.llm import LLMResponse
from keeper.loop import AgentLoop
from keeper.session import SUMMARY_PREFIX, Session
from keeper.tools.base import Tool, ToolResult
from keeper.tools.fs import ReadFileTool
from keeper.tools.registry import Registry


def _valid_summary(tag: str, *, goal: str = "完成目标") -> str:
    bodies = {
        "Goal": [f"- {tag}: {goal}"],
        "Constraints & Preferences": ["- 不引入新依赖"],
        "Progress": ["- 正在处理"],
        "Key Decisions": ["- 采用最小压缩"],
        "Next Steps": ["1. 继续下一步"],
        "Critical Context": ["- /proj/app/main.py"],
    }
    lines: list[str] = []
    for heading in SUMMARY_SECTIONS:
        lines.append(f"## {heading}")
        lines.extend(bodies[heading])
    return "\n".join(lines)


VALID_SUMMARY = _valid_summary("SUMMARY")


class FakeLLM:
    """tools is None 的调用视为摘要；否则是正常 chat。"""

    def __init__(
        self,
        responses: list[tuple[dict, dict]],
        *,
        summaries: list[str] | None = None,
        summary_error: Exception | str | None = None,
    ) -> None:
        self.responses = list(responses)
        self.summaries = list(summaries) if summaries is not None else None
        self.summary_error = summary_error
        self.calls: list[list[dict]] = []
        self.summary_prompts: list[str] = []
        self.order: list[str] = []

    def chat(self, messages, tools=None, temperature=None):
        if tools is None:
            self.order.append("summary")
            self.summary_prompts.append(messages[0]["content"])
            if isinstance(self.summary_error, Exception):
                raise self.summary_error
            if isinstance(self.summary_error, str):
                return LLMResponse(
                    message={"role": "assistant", "content": self.summary_error},
                    usage={},
                )
            content = (
                self.summaries.pop(0) if self.summaries is not None else VALID_SUMMARY
            )
            # 摘要调用自身的 usage 不应参与触发判断。
            return LLMResponse(
                message={"role": "assistant", "content": content},
                usage={"prompt_tokens": 999_999},
            )

        self.order.append("chat")
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses")
        message, usage = self.responses.pop(0)
        return LLMResponse(message=message, usage=usage)


def _tool_call(
    name: str = "read_file",
    arguments: str = '{"path": "README.md"}',
    call_id: str = "call_1",
) -> dict:
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


def _final(text: str = "done") -> dict:
    return {"role": "assistant", "content": text}


def _make_loop(
    tmp_path: Path,
    llm: FakeLLM,
    *,
    context_window: int = 100_000,
    reserve_tokens: int = 1_000,
    keep_tokens: int = 5,
    max_steps: int = 10,
    tools: list[Any] | None = None,
) -> tuple[AgentLoop, Session]:
    (tmp_path / "README.md").write_text("hello keeper\n" * 20, encoding="utf-8")
    session = Session.create(tmp_path, "test")
    loop = AgentLoop(
        llm=llm,
        registry=Registry(tools or [ReadFileTool(tmp_path)]),
        session=session,
        root=tmp_path,
        max_steps=max_steps,
        context_window=context_window,
        reserve_tokens=reserve_tokens,
        keep_tokens=keep_tokens,
    )
    return loop, session


def _records(session: Session) -> list[dict]:
    lines = session.path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_usage_below_threshold_changes_nothing(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            (_tool_call(), {"prompt_tokens": 50}),
            (_final(), {}),
        ]
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=1_000,
        reserve_tokens=100,
    )
    try:
        events = list(loop.run("read it"))

        assert isinstance(events[-1], FinalAnswer)
        assert llm.summary_prompts == []
        assert not any(isinstance(event, CompactionStart) for event in events)
        assert all(record["type"] != "compaction" for record in _records(session))

        # 请求消息和 M4 之前完全一致：没有摘要消息、没有额外内容。
        assert [m["role"] for m in llm.calls[0]] == ["system", "user"]
        assert [m["role"] for m in llm.calls[1]] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]
        assert all(
            "[历史摘要]" not in str(message.get("content"))
            for call in llm.calls
            for message in call
        )
    finally:
        session.close()


def test_usage_over_threshold_summarizes_then_uses_summary(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            (_tool_call(), {"prompt_tokens": 5_000}),
            (_final(), {}),
        ]
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=1_000,
        reserve_tokens=100,
    )
    try:
        events = list(loop.run("read it"))

        assert llm.order == ["chat", "summary", "chat"]
        assert isinstance(events[-1], FinalAnswer)
        starts = [event for event in events if isinstance(event, CompactionStart)]
        dones = [event for event in events if isinstance(event, CompactionDone)]
        assert len(starts) == 1 and len(dones) == 1
        # pi 口径：usage（5000）+ 其后工具消息的估算。
        assert starts[0].tokens_before >= 5_000
        assert starts[0].cut_index == 1
        assert dones[0].tokens_after < dones[0].tokens_before

        # 摘要请求：只有一条 user prompt，不带 tools。
        assert len(llm.summary_prompts) == 1
        assert "read it" in llm.summary_prompts[0]

        # 下一次正常请求 = system + 摘要消息 + 切点之后的消息。
        second = llm.calls[1]
        assert second[0]["role"] == "system"
        assert second[1]["role"] == "user"
        assert second[1]["content"].startswith("[历史摘要]\n")
        assert second[2]["role"] == "assistant"
        assert second[3]["role"] == "tool"
        assert len(second) == 4
    finally:
        session.close()


def test_usage_missing_falls_back_to_estimate(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            (_tool_call(), {}),  # 没有 usage
            (_final(), {}),
        ]
    )
    # 阈值压到 700：第一步的估算低于阈值，追加工具结果后才超过。
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=700,
        reserve_tokens=0,
    )
    try:
        list(loop.run("read it"))
        assert len(llm.summary_prompts) == 1
        assert any(
            record["type"] == "compaction" for record in _records(session)
        )
    finally:
        session.close()


def test_jsonl_keeps_old_messages_and_resume_matches(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            (_tool_call(), {"prompt_tokens": 5_000}),
            (_final(), {}),
        ]
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=1_000,
        reserve_tokens=100,
    )
    path = session.path
    try:
        events = list(loop.run("read it"))
        effective = list(session.messages)
    finally:
        session.close()

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    types = [record["type"] for record in records]
    assert types.count("compaction") == 1

    compaction = next(r for r in records if r["type"] == "compaction")
    done = next(event for event in events if isinstance(event, CompactionDone))
    assert compaction["summary"] == VALID_SUMMARY
    assert compaction["cut_index"] == 1
    assert compaction["messages_before"] == 3
    assert compaction["tokens_before"] == done.tokens_before
    assert compaction["tokens_before"] >= 5_000
    assert compaction["tokens_after"] == done.tokens_after
    assert compaction["tokens_after"] > 0

    # 旧的 message 记录全部还在；被摘要掉的第一条 user 没有被删除/重写。
    message_records = [r["message"] for r in records if r["type"] == "message"]
    assert message_records[0] == {"role": "user", "content": "read it"}
    assert len(message_records) == 4

    loaded = Session.load(path, tmp_path)
    try:
        assert loaded.messages == effective
        assert loaded.messages[0]["role"] == "user"
        assert loaded.messages[0]["content"] == "[历史摘要]\n" + VALID_SUMMARY
        assert loaded.messages[1]["role"] == "assistant"
    finally:
        loaded.close()


def test_resume_after_compaction_still_repairs_dangling_tool_call(
    tmp_path: Path,
) -> None:
    llm = FakeLLM(
        [
            (_tool_call(), {"prompt_tokens": 5_000}),
            (_final(), {}),
        ]
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=1_000,
        reserve_tokens=100,
    )
    path = session.path
    try:
        list(loop.run("read it"))
        # 模拟压缩之后进程在工具执行前被杀。
        session.append(_tool_call(call_id="call_interrupted"))
    finally:
        session.close()

    loaded = Session.load(path, tmp_path)
    try:
        assert loaded.messages[0]["content"].startswith("[历史摘要]\n")
        assert loaded.repair_dangling_tool_calls() == 1
        assert loaded.messages[-1]["role"] == "tool"
        assert loaded.messages[-1]["tool_call_id"] == "call_interrupted"
        assert loaded.messages[-1]["content"].startswith("[interrupted]")
    finally:
        loaded.close()


def test_consecutive_compactions_use_effective_message_list(tmp_path: Path) -> None:
    first = _valid_summary("SUMMARY-ONE")
    second = _valid_summary("SUMMARY-TWO")
    llm = FakeLLM(
        [
            (_tool_call(call_id="call_1"), {"prompt_tokens": 200_000}),
            (_tool_call(call_id="call_2"), {"prompt_tokens": 200_000}),
            (_tool_call(call_id="call_3"), {}),
            (_final(), {}),
        ],
        summaries=[first, second],
    )
    loop, session = _make_loop(tmp_path, llm)
    path = session.path
    try:
        events = list(loop.run("read it"))
        effective = list(session.messages)
    finally:
        session.close()

    assert isinstance(events[-1], FinalAnswer)
    assert len(llm.summary_prompts) == 2
    # 第二次摘要覆盖的是第一次压缩后的有效消息，里面应有旧摘要。
    assert "SUMMARY-ONE" in llm.summary_prompts[1]

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    compactions = [r for r in records if r["type"] == "compaction"]
    assert len(compactions) == 2
    assert compactions[0]["cut_index"] == 1
    assert compactions[0]["messages_before"] == 3
    # 第二次的 cut_index 是压缩后有效列表的下标，不是原始 JSONL 的下标。
    assert compactions[1]["cut_index"] == 3
    assert compactions[1]["messages_before"] == 5

    loaded = Session.load(path, tmp_path)
    try:
        assert loaded.messages == effective
        assert loaded.messages[0]["content"] == "[历史摘要]\n" + second
    finally:
        loaded.close()


@pytest.mark.parametrize(
    "summary_error",
    [
        RuntimeError("summary backend down"),
        "## Goal\n只有目标",
    ],
)
def test_summary_failure_degrades_instead_of_stopping(
    tmp_path: Path, summary_error
) -> None:
    llm = FakeLLM(
        [
            (_tool_call(), {"prompt_tokens": 5_000}),
            (_final("kept going"), {}),
        ],
        summary_error=summary_error,
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=1_000,
        reserve_tokens=100,
        max_steps=2,
    )
    try:
        events = list(loop.run("read it"))

        dones = [event for event in events if isinstance(event, CompactionDone)]
        assert len(dones) == 1
        assert dones[0].degraded is True
        assert not any(
            isinstance(event, CompactionFailed) for event in events
        )
        assert isinstance(events[-1], FinalAnswer)
        assert events[-1].text == "kept going"

        records = _records(session)
        compaction = next(
            record for record in records if record["type"] == "compaction"
        )
        assert compaction["degraded"] is True
        assert any(
            record["type"] == "note" and "compaction degraded" in record["text"]
            for record in records
        )
        # 降级摘要仍然替换了上下文，任务继续。
        assert session.messages[0]["content"].startswith("[历史摘要]\n")
    finally:
        session.close()


def test_only_summary_prefix_is_not_recompacted(tmp_path: Path) -> None:
    """保留区仍然超预算、但前缀只剩旧摘要时，不再重复摘要。"""
    llm = FakeLLM([])
    loop, session = _make_loop(tmp_path, llm, context_window=1_000, reserve_tokens=100)
    try:
        session.append({"role": "user", "content": SUMMARY_PREFIX + "旧摘要"})
        session.append(_tool_call(call_id="call_1"))
        session.append(
            {"role": "tool", "tool_call_id": "call_1", "content": "x" * 2_000}
        )

        assert list(loop._compact()) == []
        assert llm.summary_prompts == []
    finally:
        session.close()


def test_summary_missing_critical_facts_is_salvaged(
    tmp_path: Path,
) -> None:
    generic = _valid_summary("SUMMARY")  # 没有 /proj/app/main.py 和 ERROR 原文
    llm = FakeLLM(
        [
            (_tool_call(name="huge", arguments="{}", call_id="call_huge"), {}),
            (
                _tool_call(call_id="call_small", arguments='{"path": "README.md"}'),
                {"prompt_tokens": 200_000},
            ),
            (_final("kept going"), {}),
        ],
        summaries=[generic],
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=10_000,
        reserve_tokens=1_000,
        tools=[HugeOutputTool(), ReadFileTool(tmp_path)],
    )
    try:
        events = list(loop.run("修复导入错误"))

        dones = [event for event in events if isinstance(event, CompactionDone)]
        assert len(dones) == 1
        # 模型摘要可用，只是漏了事实；确定性补齐，不算降级。
        assert dones[0].degraded is False
        assert not any(
            isinstance(event, CompactionFailed) for event in events
        )

        compaction = next(
            record for record in _records(session) if record["type"] == "compaction"
        )
        assert "/proj/app/main.py" in compaction["summary"]
        assert "ERROR: [Errno 2] No such file or directory" in compaction["summary"]
        assert isinstance(events[-1], FinalAnswer)
    finally:
        session.close()


HUGE_LINES = 5_000
HUGE_OUTPUT = (
    "ERROR: [Errno 2] No such file or directory: '/proj/app/main.py'\n"
    + "\n".join(f"trace line {index}" for index in range(HUGE_LINES))
)
HUGE_SUMMARY = _valid_summary(
    "SUMMARY", goal="修复 /proj/app/main.py 的导入错误"
).replace(
    "1. 继续下一步",
    "1. 运行 pytest 验证 /proj/app/main.py",
).replace(
    "- /proj/app/main.py",
    "- ERROR: [Errno 2] No such file or directory: '/proj/app/main.py'",
)


class HugeOutputTool(Tool):
    name = "huge"
    description = "返回 5000 行工具输出"
    parameters: dict = {"type": "object", "properties": {}}
    danger = "read"

    def execute(self, args: dict) -> ToolResult:
        return ToolResult(ok=True, content=HUGE_OUTPUT)


def test_tool_output_after_response_triggers_compaction(tmp_path: Path) -> None:
    """usage 只到上一次请求为止；追加的工具输出也要参与阈值判断。"""
    llm = FakeLLM(
        [
            # 第一步 usage=100，远低于阈值 900。
            (_tool_call(name="huge", arguments="{}", call_id="call_huge"), {"prompt_tokens": 100}),
            (_final("done"), {}),
        ],
        summaries=[HUGE_SUMMARY],
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=1_000,
        reserve_tokens=100,
        tools=[HugeOutputTool()],
    )
    try:
        events = list(loop.run("修复导入错误"))

        # 5000 行输出把当前上下文推过阈值，第二次请求前必须先摘要。
        assert llm.order == ["chat", "summary", "chat"]
        assert len(llm.summary_prompts) == 1
        assert "修复导入错误" in llm.summary_prompts[0]
        assert any(isinstance(event, CompactionStart) for event in events)
        assert any(isinstance(event, CompactionDone) for event in events)
        assert llm.calls[1][1]["content"].startswith("[历史摘要]\n")
        assert any(
            record["type"] == "compaction" for record in _records(session)
        )
        assert isinstance(events[-1], FinalAnswer)
    finally:
        session.close()


def test_oversized_tail_tool_output_is_dropped(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            (
                _tool_call(name="huge", arguments="{}", call_id="call_huge"),
                {"prompt_tokens": 100},
            ),
            (_final("done"), {}),
        ],
        summaries=[HUGE_SUMMARY],
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        context_window=10_000,
        reserve_tokens=1_000,
        tools=[HugeOutputTool()],
    )
    try:
        events = list(loop.run("修复导入错误"))

        # 单组输出约 22k tokens > 阈值 9k：整段交给摘要，下一请求只剩摘要。
        assert llm.order == ["chat", "summary", "chat"]
        assert [message["role"] for message in llm.calls[1]] == [
            "system",
            "user",
        ]
        request_text = json.dumps(llm.calls[1], ensure_ascii=False)
        assert f"trace line {HUGE_LINES - 1}" not in request_text
        assert "ERROR: [Errno 2] No such file or directory" in request_text
        assert "/proj/app/main.py" in request_text

        # 摘要输入按 pi 的规则只留工具结果头部。
        prompt = llm.summary_prompts[0]
        assert "more characters truncated" in prompt
        assert "ERROR: [Errno 2] No such file or directory" in prompt
        assert f"trace line {HUGE_LINES - 1}" not in prompt

        assert session.messages[0]["content"].startswith("[历史摘要]\n")
        assert len(session.messages) == 2  # 摘要 + 最终回答
        assert any(isinstance(event, CompactionDone) for event in events)
    finally:
        session.close()


def test_task_mode_wires_compaction(tmp_path: Path) -> None:
    """M3 任务模式也走同一套压缩配置。"""
    from keeper.task_runner import prepare_task_run, run_task_once
    from keeper.tasks import TaskStore

    (tmp_path / "README.md").write_text("hello keeper\n" * 20, encoding="utf-8")
    store = TaskStore.load(tmp_path)
    task = store.add("read it")
    prepared = prepare_task_run(store, task.id, tmp_path, "test-model")
    llm = FakeLLM(
        [
            (_tool_call(), {"prompt_tokens": 200_000}),
            (_final(), {}),
        ]
    )
    try:
        events = list(
            run_task_once(
                prepared,
                llm=llm,
                registry=Registry([ReadFileTool(tmp_path)]),
                approver=None,
                max_steps=5,
                context_window=100_000,
                reserve_tokens=1_000,
                keep_tokens=5,
            )
        )

        assert len(llm.summary_prompts) == 1
        assert llm.calls[1][1]["content"].startswith("[历史摘要]\n")
        assert any(isinstance(event, CompactionDone) for event in events)
        records = [
            json.loads(line)
            for line in prepared.session.path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert any(record["type"] == "compaction" for record in records)
    finally:
        prepared.session.close()


def test_huge_tool_output_is_dropped_after_compaction(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            (_tool_call(name="huge", arguments="{}", call_id="call_huge"), {}),
            (
                _tool_call(call_id="call_small", arguments='{"path": "README.md"}'),
                {"prompt_tokens": 200_000},
            ),
            (_final("done"), {}),
        ],
        summaries=[HUGE_SUMMARY],
    )
    loop, session = _make_loop(
        tmp_path,
        llm,
        tools=[HugeOutputTool(), ReadFileTool(tmp_path)],
    )
    try:
        events = list(loop.run("修复导入错误，先看日志"))

        assert isinstance(events[-1], FinalAnswer)
        assert len(llm.summary_prompts) == 1
        # 摘要输入包含前缀目标、关键路径和错误原文；工具结果只留头部。
        prompt = llm.summary_prompts[0]
        assert "修复导入错误，先看日志" in prompt
        assert "/proj/app/main.py" in prompt
        assert "ERROR: [Errno 2] No such file or directory" in prompt
        assert "more characters truncated" in prompt
        assert f"trace line {HUGE_LINES - 1}" not in prompt

        # 压缩后的下一条正常请求不再带 5000 行输出，但保留摘要里的关键信息。
        next_request = llm.calls[2]
        text = json.dumps(next_request, ensure_ascii=False)
        assert f"trace line {HUGE_LINES - 1}" not in text
        assert "ERROR: [Errno 2] No such file or directory" in text
        assert "/proj/app/main.py" in text
        assert "## Next Steps" in text
        assert next_request[1]["content"].startswith("[历史摘要]\n")

        assert any(
            record["type"] == "compaction" for record in _records(session)
        )
    finally:
        session.close()

def test_split_turn_uses_history_and_turn_prefix_summaries(
    tmp_path: Path,
) -> None:
    history_summary = _valid_summary("HISTORY")
    turn_summary = "\n".join(
        [
            "## Original Request",
            "- second",
            "## Progress So Far",
            "- 还没开始",
            "## Context Needed to Continue",
            "- 继续",
        ]
    )
    llm = FakeLLM(
        [
            (_tool_call(call_id="call_1"), {"prompt_tokens": 50}),
            (_final("first done"), {}),
            (_tool_call(call_id="call_2"), {"prompt_tokens": 200_000}),
            (_final("done"), {}),
        ],
        summaries=[history_summary, turn_summary],
    )
    loop, session = _make_loop(tmp_path, llm, max_steps=5)
    path = session.path
    try:
        list(loop.run("first"))
        events = list(loop.run("second"))
        effective = list(session.messages)
    finally:
        session.close()

    assert len(llm.summary_prompts) == 2
    # 第一段是历史（不含当前轮次），第二段只覆盖当前轮次被丢弃的前缀。
    assert "first" in llm.summary_prompts[0]
    assert "## Original Request" in llm.summary_prompts[1]
    assert "second" in llm.summary_prompts[1]

    compaction = next(
        record for record in _records(session) if record["type"] == "compaction"
    )
    # messages = [user1, a1, t1, final1, user2, a3, t3]；切在 a3。
    assert compaction["cut_index"] == 5
    assert compaction["messages_before"] == 7
    assert history_summary in compaction["summary"]
    assert turn_summary in compaction["summary"]
    assert "**Turn Context (split turn):**" in compaction["summary"]

    assert isinstance(events[-1], FinalAnswer)
    loaded = Session.load(path, tmp_path)
    try:
        assert loaded.messages == effective
        assert loaded.messages[0]["content"].startswith("[历史摘要]\n")
    finally:
        loaded.close()


def test_context_tokens_uses_usage_plus_trailing(tmp_path: Path) -> None:
    from keeper.compaction import estimate_request_tokens, estimate_tokens

    llm = FakeLLM([])
    loop, session = _make_loop(
        tmp_path, llm, context_window=100_000, reserve_tokens=1_000
    )
    try:
        session.append({"role": "user", "content": "u"})
        session.append({"role": "assistant", "content": "a"})
        loop._usage_tokens = 5_000
        loop._usage_index = 1
        session.append({"role": "tool", "tool_call_id": "c1", "content": "x" * 300})

        assert loop._context_tokens() == 5_000 + estimate_tokens(
            session.messages[2:]
        )

        # 没有 usage 时退到全量估算，并把工具 schema 算进去。
        loop._usage_tokens = None
        loop._usage_index = None
        assert loop._context_tokens() == estimate_request_tokens(
            [{"role": "system", "content": loop.system}, *session.messages],
            loop.registry.schemas(),
        )
    finally:
        session.close()


def test_record_usage_prefers_total_tokens(tmp_path: Path) -> None:
    llm = FakeLLM([])
    loop, session = _make_loop(tmp_path, llm)
    try:
        session.append({"role": "assistant", "content": "a"})

        class Response:
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        loop._record_usage(Response())
        assert loop._usage_tokens == 15
        assert loop._usage_index == 0

        # 没有 total_tokens 时用 prompt + completion。
        class Partial:
            usage = {"prompt_tokens": 10, "completion_tokens": 5}

        loop._record_usage(Partial())
        assert loop._usage_tokens == 15

        # 空 usage 不产生有效记录。
        loop._usage_tokens = None
        loop._usage_index = None

        class Empty:
            usage = {}

        loop._record_usage(Empty())
        assert loop._usage_tokens is None
        assert loop._usage_index is None
    finally:
        session.close()
