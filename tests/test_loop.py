from pathlib import Path

from keeper.events import FinalAnswer, Stop, ToolResultEvent
from keeper.llm import LLMResponse
from keeper.loop import AgentLoop
from keeper.session import Session
from keeper.tools.fs import ReadFileTool
from keeper.tools.registry import Registry


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of responses")
        return LLMResponse(message=self._responses.pop(0), usage={})


def _tool_call(arguments: str = '{"path": "README.md"}', call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": arguments},
            }
        ],
    }


def _make_loop(
    tmp_path: Path,
    responses: list[dict],
    max_steps: int = 5,
) -> tuple[AgentLoop, FakeLLM, Session]:
    (tmp_path / "README.md").write_text("# demo\nhello\n", encoding="utf-8")
    llm = FakeLLM(responses)
    session = Session.create(tmp_path, "test")
    loop = AgentLoop(
        llm=llm,
        registry=Registry([ReadFileTool(tmp_path)]),
        session=session,
        root=tmp_path,
        max_steps=max_steps,
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