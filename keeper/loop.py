"""Agent 主循环：组装上下文 → 调模型 → 执行工具 → 回灌 → 停机判断。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generator, Iterator, Protocol

from keeper.approval import Approver, Decision, review
from keeper.compaction import (
    estimate_request_tokens,
    estimate_tokens,
    find_cut_point,
    merge_split_summary,
    summarize,
)
from keeper.config import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_KEEP_TOKENS,
    DEFAULT_RESERVE_TOKENS,
)
from keeper.events import (
    ApprovalRequest,
    ApprovalResult,
    AssistantText,
    CompactionDone,
    CompactionFailed,
    CompactionStart,
    ErrorEvent,
    Event,
    FinalAnswer,
    StepStart,
    Stop,
    ToolCallEvent,
    ToolResultEvent,
    UserMessage,
)
from keeper.prompts import TaskContext, system_prompt
from keeper.session import SUMMARY_PREFIX, Session
from keeper.tools.base import ToolResult, error
from keeper.tools.registry import Registry


def _is_summary_message(message: dict[str, Any]) -> bool:
    """压缩后写入的第一条消息；它不是新内容。"""
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith(SUMMARY_PREFIX)
    )


def _summary_body(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    if content.startswith(SUMMARY_PREFIX):
        return content[len(SUMMARY_PREFIX):]
    return content


class ChatLLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
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
        task: TaskContext | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
        keep_tokens: int = DEFAULT_KEEP_TOKENS,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.session = session
        self.system = system_prompt(root, task)
        self.max_steps = max_steps
        self.root = Path(root).resolve()
        self.approver = approver
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.keep_tokens = keep_tokens
        # 最近一次有效响应的整次调用 token 数（含输出），以及它对应的
        # assistant 消息下标；其后追加的消息用估算补齐。
        self._usage_tokens: int | None = None
        self._usage_index: int | None = None

    def _messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system}, *self.session.messages]

    def _should_compact(self) -> bool:
        # 口径移植自 pi-agent 的 estimateContextTokens：最近一次有效 usage
        # 加上其后的消息估算；没有 usage 时退到全量估算（含工具 schema）。
        return self._context_tokens() > self.context_window - self.reserve_tokens

    def _context_tokens(self) -> int:
        if self._usage_tokens is not None and self._usage_index is not None:
            if 0 <= self._usage_index < len(self.session.messages):
                trailing = self.session.messages[self._usage_index + 1 :]
                return self._usage_tokens + estimate_tokens(trailing)
        return estimate_request_tokens(
            self._messages(), self.registry.schemas()
        )

    @staticmethod
    def _usage_int(usage: dict[str, Any], key: str) -> int | None:
        try:
            value = int(usage.get(key))
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def _record_usage(self, response: Any) -> None:
        """记录最近一次有效 usage；优先 total_tokens，否则 prompt + completion。"""
        usage = getattr(response, "usage", None)
        if not isinstance(usage, dict):
            return

        tokens = self._usage_int(usage, "total_tokens")
        if not tokens:
            prompt = self._usage_int(usage, "prompt_tokens")
            if prompt is not None:
                completion = self._usage_int(usage, "completion_tokens") or 0
                total = prompt + completion
                tokens = total if total > 0 else None
        if not tokens:
            return

        self._usage_tokens = tokens
        self._usage_index = len(self.session.messages) - 1

    def _compact(self) -> Iterator[Event]:
        """压缩一次；摘要不可用时降级为确定性摘要，不让任务中断。"""
        threshold = self.context_window - self.reserve_tokens
        messages = self.session.messages
        point = find_cut_point(
            messages,
            self.keep_tokens,
            estimate_tokens,
            max_tail_tokens=threshold,
        )
        cut_index = point.cut_index

        if cut_index <= 0:
            return

        prefix = messages[:cut_index]
        if prefix and all(_is_summary_message(message) for message in prefix):
            # 前缀只剩旧摘要：没有新内容可总结，避免每个 step 重复摘要。
            return

        tokens_before = self._context_tokens()
        messages_before = len(messages)
        yield CompactionStart(
            tokens_before=tokens_before,
            cut_index=cut_index,
            reason="threshold",
        )

        try:
            if point.is_split_turn:
                history = messages[: point.turn_start_index]
                turn_prefix = messages[point.turn_start_index : cut_index]
                if history and all(
                    _is_summary_message(message) for message in history
                ):
                    # 历史部分只有旧摘要：直接沿用，不用再付一次摘要调用。
                    history_text = _summary_body(history[-1]) or "（没有更早的历史记录）"
                    history_degraded = False
                else:
                    history_result = summarize(
                        self.llm, history, max_input_tokens=max(1, threshold)
                    )
                    history_text = history_result.text
                    history_degraded = history_result.degraded
                turn_result = summarize(
                    self.llm,
                    turn_prefix,
                    max_input_tokens=max(1, threshold),
                    kind="turn_prefix",
                )
                summary = merge_split_summary(history_text, turn_result.text)
                degraded = history_degraded or turn_result.degraded
            else:
                result = summarize(
                    self.llm, prefix, max_input_tokens=max(1, threshold)
                )
                summary = result.text
                degraded = result.degraded

            projected = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": SUMMARY_PREFIX + summary},
                *messages[cut_index:],
            ]
            tokens_after = estimate_request_tokens(
                projected, self.registry.schemas()
            )
            self.session.compact(
                summary,
                cut_index,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                degraded=degraded,
            )
        except Exception as exc:
            self.session.note(f"compaction failed: {type(exc).__name__}: {exc}")
            yield CompactionFailed(
                tokens_before=tokens_before,
                tokens_after=self._context_tokens(),
                cut_index=cut_index,
                reason="summary_error",
                error=str(exc),
            )
            return

        # 压缩后旧 usage 只反映压缩前的上下文，作废；下次走纯估算。
        self._usage_tokens = None
        self._usage_index = None
        if degraded:
            self.session.note("compaction degraded: deterministic digest used")
        yield CompactionDone(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            cut_index=cut_index,
            messages_before=messages_before,
            messages_after=len(self.session.messages),
            degraded=degraded,
        )

    def run(self, user_text: str) -> Iterator[Event]:
        self.session.append({"role": "user", "content": user_text})
        yield UserMessage(text=user_text)

        for step in range(1, self.max_steps + 1):
            yield StepStart(step=step)

            if self._should_compact():
                yield from self._compact()

            request_messages = self._messages()
            try:
                response = self.llm.chat(
                    request_messages,
                    tools=self.registry.schemas(),
                )
            except Exception as exc:
                yield ErrorEvent(message=f"LLM call failed: {exc}")
                yield Stop(reason="llm_error")
                return

            message = response.message
            self.session.append(message)
            self._record_usage(response)
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