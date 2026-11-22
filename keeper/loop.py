"""Agent 主循环：组装上下文 → 调模型 → 执行工具 → 回灌 → 停机判断。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generator, Iterator, Protocol

from keeper.approval import Approver, Decision, review
from keeper.events import (
    ApprovalRequest,
    ApprovalResult,
    AssistantText,
    ErrorEvent,
    Event,
    FinalAnswer,
    StepStart,
    Stop,
    ToolCallEvent,
    ToolResultEvent,
    UserMessage,
)
from keeper.prompts import system_prompt
from keeper.session import Session
from keeper.tools.base import ToolResult, error
from keeper.tools.registry import Registry


class ChatLLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any: ...


class AgentLoop:
    def __init__(
        self,
        *,
        llm: ChatLLM,
        registry: Registry,
        session: Session,
        root: Path,
        max_steps: int = 30,
        approver: Approver | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.session = session
        self.system = system_prompt(root)
        self.max_steps = max_steps
        self.root = Path(root).resolve()
        self.approver = approver

    def _messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system}, *self.session.messages]

    def run(self, user_text: str) -> Iterator[Event]:
        self.session.append({"role": "user", "content": user_text})
        yield UserMessage(text=user_text)

        for step in range(1, self.max_steps + 1):
            yield StepStart(step=step)

            try:
                response = self.llm.chat(
                    self._messages(),
                    tools=self.registry.schemas(),
                )
            except Exception as exc:
                yield ErrorEvent(message=f"LLM call failed: {exc}")
                yield Stop(reason="llm_error")
                return

            message = response.message
            self.session.append(message)
            text = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                yield FinalAnswer(text=text)
                return

            if text.strip():
                yield AssistantText(text=text)

            for call in tool_calls:
                yield from self._run_tool_call(call)

        self.session.note(f"stopped: max_steps ({self.max_steps})")
        yield Stop(reason="max_steps")

    def _run_tool_call(self, call: dict[str, Any]) -> Iterator[Event]:
        call_id = call.get("id") or ""
        function = call.get("function") or {}
        name = function.get("name") or ""
        raw_args = function.get("arguments") or "{}"

        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if not isinstance(parsed, dict):
                raise ValueError("arguments must be a JSON object")
            args = parsed
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            yield ToolCallEvent(
                call_id=call_id,
                name=name,
                args={},
                valid_json=False,
            )
            result = ToolResult(ok=False, content=f"ERROR: invalid tool arguments: {exc}")
            duration_ms = 0
        else:
            yield ToolCallEvent(
                call_id=call_id,
                name=name,
                args=args,
                valid_json=True,
            )
            result, duration_ms = yield from self._gate_and_execute(
                call_id, name, args
            )

        self.session.append(
            {"role": "tool", "tool_call_id": call_id, "content": result.content}
        )
        yield ToolResultEvent(
            call_id=call_id,
            name=name,
            ok=result.ok,
            output=result.content,
            duration_ms=duration_ms,
        )

    def _gate_and_execute(
        self, call_id: str, name: str, args: dict[str, Any]
    ) -> Generator[Event, None, tuple[ToolResult, int]]:
        """审批门 + 执行。返回 (结果, 执行耗时)；耗时不含等待审批的时间。"""
        tool = self.registry.get(name)
        if tool is not None:
            verdict = review(
                tool_name=name,
                danger=tool.danger,
                args=args,
                root=self.root,
            )

            if verdict.decision is Decision.DENY:
                reason = verdict.reason or "not permitted by policy"
                yield ApprovalResult(
                    call_id=call_id,
                    tool=name,
                    approved=False,
                    reason=reason,
                )
                return (
                    error(f"denied: {reason}. The operation was not executed."),
                    0,
                )

            if verdict.decision is Decision.ASK:
                request = ApprovalRequest(
                    call_id=call_id,
                    tool=name,
                    summary=verdict.summary,
                    detail=verdict.detail,
                )
                yield request
                approved, reason = self._confirm(request)
                yield ApprovalResult(
                    call_id=call_id,
                    tool=name,
                    approved=approved,
                    reason=reason,
                )
                if not approved:
                    return (
                        error(
                            f"denied: {reason}. The operation was not executed; "
                            "adjust your approach."
                        ),
                        0,
                    )

        started = time.perf_counter()
        result = self.registry.dispatch(name, args)
        duration_ms = int((time.perf_counter() - started) * 1000)
        return result, duration_ms

    def _confirm(self, request: ApprovalRequest) -> tuple[bool, str]:
        if self.approver is None:
            return False, "no approver configured (fail-closed)"
        try:
            approved = bool(self.approver.confirm(request))
        except Exception as exc:  # 审批器出问题也必须拒绝
            return False, f"approval failed: {type(exc).__name__}: {exc}"
        return approved, "approved by user" if approved else "denied by user"