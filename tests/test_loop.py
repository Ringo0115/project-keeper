import json
from pathlib import Path

from keeper.events import (
    ApprovalRequest,
    ApprovalResult,
    FinalAnswer,
    Stop,
    ToolResultEvent,
)
from keeper.llm import LLMResponse
from keeper.loop import AgentLoop
from keeper.session import Session
from keeper.tools.fs import ReadFileTool, WriteFileTool
from keeper.tools.registry import Registry
from keeper.tools.shell import ShellTool


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of responses")
        return LLMResponse(message=self._responses.pop(0), usage={})


def _tool_call(
    arguments: str = '{"path": "README.md"}',
    call_id: str = "call_1",
    name: str = "read_file",
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


class AutoApprover:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    def confirm(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


class BoomApprover:
    def confirm(self, request: ApprovalRequest) -> bool:
        raise RuntimeError("approval terminal exploded")


def _make_loop(
    tmp_path: Path,
    responses: list[dict],
    max_steps: int = 5,
    approver: object | None = None,
    registry: Registry | None = None,
) -> tuple[AgentLoop, FakeLLM, Session]:
    (tmp_path / "README.md").write_text("# demo\nhello\n", encoding="utf-8")
    llm = FakeLLM(responses)
    session = Session.create(tmp_path, "test")
    loop = AgentLoop(
        llm=llm,
        registry=registry
        or Registry([ReadFileTool(tmp_path), WriteFileTool(tmp_path)]),
        session=session,
        root=tmp_path,
        max_steps=max_steps,
        approver=approver,  # type: ignore[arg-type]
    )
    return loop, llm, session


def test_one_tool_then_final(tmp_path: Path) -> None:
    loop, llm, session = _make_loop(
        tmp_path,
        [_tool_call(), {"role": "assistant", "content": "done"}],
    )
    try:
        events = list(loop.run("summarize readme"))
        kinds = [event.type for event in events]
        assert "tool_call" in kinds
        assert "tool_result" in kinds
        assert isinstance(events[-1], FinalAnswer)
        assert events[-1].text == "done"

        roles = [message["role"] for message in session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert "hello" in session.messages[2]["content"]
        assert session.messages[2]["tool_call_id"] == "call_1"

        # 第二次调模型时能看到工具结果
        second_call = llm.calls[1]
        assert second_call[0]["role"] == "system"
        assert any(message.get("role") == "tool" for message in second_call)
    finally:
        session.close()


def test_invalid_arguments_do_not_crash(tmp_path: Path) -> None:
    loop, _, session = _make_loop(
        tmp_path,
        [_tool_call(arguments="{not json"), {"role": "assistant", "content": "fixed"}],
    )
    try:
        events = list(loop.run("x"))
        results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(results) == 1
        assert not results[0].ok
        assert "invalid" in results[0].output

        assert session.messages[2]["role"] == "tool"
        assert session.messages[2]["content"].startswith("ERROR:")
        assert isinstance(events[-1], FinalAnswer)
    finally:
        session.close()


def test_max_steps(tmp_path: Path) -> None:
    loop, _, session = _make_loop(
        tmp_path,
        [_tool_call(), _tool_call(call_id="call_2")],
        max_steps=2,
    )
    try:
        events = list(loop.run("loop forever"))
        assert isinstance(events[-1], Stop)
        assert events[-1].reason == "max_steps"
        assert [message["role"] for message in session.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
    finally:
        session.close()


def test_llm_error_yields_stop(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("x", encoding="utf-8")

    class BrokenLLM:
        def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
            raise RuntimeError("network down")

    session = Session.create(tmp_path, "test")
    loop = AgentLoop(
        llm=BrokenLLM(),
        registry=Registry([ReadFileTool(tmp_path)]),
        session=session,
        root=tmp_path,
    )
    try:
        events = list(loop.run("x"))
        assert any(event.type == "error" for event in events)
        assert isinstance(events[-1], Stop)
        assert events[-1].reason == "llm_error"
    finally:
        session.close()

# ---------- M1 审批 ----------


def _write_call(
    path: str = "new.txt",
    content: str = "hello\n",
    call_id: str = "call_1",
) -> dict:
    return _tool_call(
        json.dumps({"path": path, "content": content}),
        call_id=call_id,
        name="write_file",
    )


def test_write_approved_by_user(tmp_path: Path) -> None:
    approver = AutoApprover(True)
    loop, _, session = _make_loop(
        tmp_path,
        [_write_call(), {"role": "assistant", "content": "done"}],
        approver=approver,
    )
    try:
        events = list(loop.run("write a file"))
        assert len(approver.requests) == 1
        assert approver.requests[0].tool == "write_file"
        assert any(isinstance(event, ApprovalResult) and event.approved for event in events)
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert result.ok
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\n"
    finally:
        session.close()


def test_write_denied_lets_model_continue(tmp_path: Path) -> None:
    approver = AutoApprover(False)
    loop, llm, session = _make_loop(
        tmp_path,
        [_write_call(), {"role": "assistant", "content": "ok, gave up"}],
        approver=approver,
    )
    try:
        events = list(loop.run("write a file"))
        assert not (tmp_path / "new.txt").exists()
        assert not next(
            event for event in events if isinstance(event, ApprovalResult)
        ).approved
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert not result.ok
        assert result.output.startswith("ERROR:")
        assert "denied" in result.output
        assert isinstance(events[-1], FinalAnswer)
        assert len(llm.calls) == 2
        assert session.messages[-2]["role"] == "tool"
        assert session.messages[-2]["content"].startswith("ERROR:")
    finally:
        session.close()


def test_write_without_approver_is_denied(tmp_path: Path) -> None:
    loop, _, session = _make_loop(
        tmp_path,
        [_write_call(), {"role": "assistant", "content": "done"}],
    )
    try:
        events = list(loop.run("write a file"))
        assert not (tmp_path / "new.txt").exists()
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert not result.ok
        assert "no approver" in result.output
    finally:
        session.close()


def test_approver_crash_is_denied(tmp_path: Path) -> None:
    loop, _, session = _make_loop(
        tmp_path,
        [_write_call(), {"role": "assistant", "content": "done"}],
        approver=BoomApprover(),
    )
    try:
        events = list(loop.run("write a file"))
        assert not (tmp_path / "new.txt").exists()
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert not result.ok
        assert "approval failed" in result.output
    finally:
        session.close()


def test_dangerous_shell_denied_without_prompt(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    approver = AutoApprover(True)
    loop, _, session = _make_loop(
        tmp_path,
        [
            _tool_call(
                json.dumps({"argv": ["rm", "-rf", "."]}),
                name="shell",
            ),
            {"role": "assistant", "content": "not allowed"},
        ],
        approver=approver,
        registry=Registry([ShellTool(tmp_path)]),
    )
    try:
        events = list(loop.run("delete everything"))
        assert approver.requests == []
        assert not any(isinstance(event, ApprovalRequest) for event in events)
        approval = next(event for event in events if isinstance(event, ApprovalResult))
        assert not approval.approved
        assert "delete" in approval.reason
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert not result.ok
        assert result.output.startswith("ERROR:")
        assert victim.exists()
    finally:
        session.close()


def test_read_never_asks_for_approval(tmp_path: Path) -> None:
    approver = AutoApprover(False)
    loop, _, session = _make_loop(
        tmp_path,
        [_tool_call(), {"role": "assistant", "content": "done"}],
        approver=approver,
    )
    try:
        events = list(loop.run("read it"))
        assert approver.requests == []
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert result.ok
    finally:
        session.close()


def test_write_preview_contains_diff(tmp_path: Path) -> None:
    approver = AutoApprover(True)
    loop, _, session = _make_loop(
        tmp_path,
        [
            _write_call(path="README.md", content="# demo\nbye\n"),
            {"role": "assistant", "content": "done"},
        ],
        approver=approver,
    )
    try:
        list(loop.run("update readme"))
        request = approver.requests[0]
        assert "-hello" in request.detail
        assert "+bye" in request.detail
        assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# demo\nbye\n"
    finally:
        session.close()
