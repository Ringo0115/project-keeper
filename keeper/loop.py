"""Agent 主循环：组装上下文 → 调模型 → 执行工具 → 回灌 → 停机判断。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generator, Iterator, Protocol

from keeper.approval import Approver, Decision, review
from keeper.compaction import (
    SummaryError,
    estimate_tokens,
    find_cut_index,
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

# 摘要失败后，上下文至少再增长这么多 tokens 才重试，避免每个 step 重复付费。
COMPACTION_RETRY_MARGIN_TOKENS = 4096


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
        # 上一次正常 chat 的 prompt token 数；None 表示还没有可判断的响应。
        self._last_prompt_tokens: int | None = None
        # 摘要失败时的上下文大小；上下文没明显增长前不重试。
        self._compaction_failed_at: int | None = None

    def _messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system}, *self.session.messages]

    def _should_compact(self) -> bool:
        # usage 只覆盖上一次请求；之后的 assistant / tool 消息还没算进去。
        # 当前上下文估算兜住「工具输出把窗口推过阈值」的情况，
        # 也让 resume 时已经过大的会话在第一次请求前就能压缩。
        current = self._context_tokens()
        if (
            self._compaction_failed_at is not None
            and current
            < self._compaction_failed_at + COMPACTION_RETRY_MARGIN_TOKENS
        ):
            return False
        observed = self._last_prompt_tokens or 0
        threshold = self.context_window - self.reserve_tokens
        return max(observed, current) > threshold

    def _context_tokens(self) -> int:
        return estimate_tokens(self._messages())

    def _record_usage(self, response: Any, request_estimate: int) -> None:
        """记录下一次判断要用的 prompt tokens；usage 缺失时用估算兜底。"""
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            try:
                prompt_tokens = int(usage.get("prompt_tokens"))
            except (TypeError, ValueError):
                prompt_tokens = None
            if prompt_tokens is not None and prompt_tokens >= 0:
                self._last_prompt_tokens = prompt_tokens
                return
        self._last_prompt_tokens = request_estimate

    def _compact(self) -> Iterator[Event]:
        """压缩一次；失败只发事件和 note，不让任务中断。"""
        threshold = self.context_window - self.reserve_tokens
        messages = self.session.messages
        cut_index = find_cut_index(
            messages,
            self.keep_tokens,
            estimate_tokens,
            max_tail_tokens=threshold,
        )
        tokens_before = self._last_prompt_tokens or self._context_tokens()

        if cut_index <= 0:
            # 没有可丢弃的前缀：重置判断值，避免每个 step 重复检查。
            self._last_prompt_tokens = self._context_tokens()
            self._compaction_failed_at = None
            return

        yield CompactionStart(
            tokens_before=tokens_before,
            cut_index=cut_index,
            reason="threshold",
        )

        messages_before = len(messages)
        try:
            summary = summarize(
                self.llm,
                messages[:cut_index],
                max_input_tokens=max(1, threshold),
            )
            projected = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": SUMMARY_PREFIX + summary},
                *messages[cut_index:],
            ]
            tokens_after = estimate_tokens(projected)
            self.session.compact(
                summary,
                cut_index,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
            )
        except Exception as exc:
            reason = "invalid_summary" if isinstance(exc, SummaryError) else "summary_error"
            self.session.note(f"compaction failed: {type(exc).__name__}: {exc}")
            tokens_after = self._context_tokens()
            self._compaction_failed_at = tokens_after
            yield CompactionFailed(
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                cut_index=cut_index,
                reason=reason,
                error=str(exc),
            )
            return

        # 用估算值重置，避免下一步立刻按旧 usage 重复触发。
        self._last_prompt_tokens = tokens_after
        self._compaction_failed_at = None
        yield CompactionDone(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            cut_index=cut_index,
            messages_before=messages_before,
            messages_after=len(self.session.messages),
        )

    def run(self, user_text: str) -> Iterator[Event]:
        # 新的一轮用户任务：允许摘要失败后重新尝试一次。
        self._compaction_failed_at = None
        self.session.append({"role": "user", "content": user_text})
        yield UserMessage(text=user_text)

        for step in range(1, self.max_steps + 1):
            yield StepStart(step=step)

            if self._should_compact():
                yield from self._compact()

            request_messages = self._messages()
            request_estimate = estimate_tokens(request_messages)
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
            self._record_usage(response, request_estimate)
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