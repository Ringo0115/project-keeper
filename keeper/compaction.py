"""M4 最小压缩：token 估算、切点选择、六段摘要。

这里只放纯逻辑和一次摘要调用；session JSONL 的写入由 ``Session.compact`` 负责，
事件编排由 ``AgentLoop`` 负责。切点算法不拆分 assistant 的 tool_calls 与
对应的 tool 结果。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

SUMMARY_SECTIONS = (
    "Goal",
    "Constraints & Preferences",
    "Progress",
    "Key Decisions",
    "Next Steps",
    "Critical Context",
)

# 摘要只输出六段；标题必须和上面的常量一字不差。
_SUMMARY_INSTRUCTIONS = """你是 keeper 的上下文压缩器。下面是即将从会话中丢弃的历史消息（JSON 数组）。
请把它们压缩成一份结构化摘要，供另一个 LLM 继续工作。

要求：
- 只输出下面六段，标题逐字照抄，不要添加其他标题或前后说明；每段必须有内容。
- 必须保留：最初目标、全部硬约束与用户偏好、当前进度、下一步、关键文件路径（原样保留）、错误原文（错误文本不要改写或概括）。
- 不要编造历史中不存在的信息；不确定时写 (unknown)。

## Goal
[用户想完成什么；可能有多项]

## Constraints & Preferences
- [用户明确提出的约束、偏好、要求]

## Progress
- [已完成 / 进行中 / 阻塞的项目与当前状态]

## Key Decisions
- [已做出的决定和简短理由]

## Next Steps
1. [接下来应该做什么，按顺序]

## Critical Context
- [继续工作所需的关键路径、原文错误、数据、示例]
"""

_TRUNCATION_MARKER = "[...truncated {omitted} bytes...]"
_PATH_TOKEN_RE = re.compile(r"[^\s\"'`<>]+")


class SummaryError(RuntimeError):
    """摘要不可用：缺少/空段、缺少关键事实，或返回内容不是文本。"""


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """确定性 token 估算：UTF-8 JSON 字节数 // 3。

    不引入 tokenizer；同一组消息永远得到同一个数。
    """
    payload = json.dumps(messages, ensure_ascii=False)
    return len(payload.encode("utf-8")) // 3


def find_cut_index(
    messages: list[dict[str, Any]],
    keep_tokens: int,
    estimate: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
    *,
    max_tail_tokens: int | None = None,
) -> int:
    """返回保留区间的起点：``messages[cut:]`` 被保留，``messages[:cut]`` 被摘要。

    - 从最新消息往回累加，达到 ``keep_tokens`` 后确定预算切点。
    - 切点优先落在 user 消息上，其次落在 assistant 边界上。
    - 绝不返回 tool 消息下标，因此不会把 assistant 的 tool_calls 和
      tool 结果拆开。若预算切点落在工具结果上且后面没有安全边界，
      则回退到拥有该 tool_call 的 assistant（宁可多保留整组）。
    - 整段对话都在预算内时返回 0，表示无需压缩。
    - ``max_tail_tokens`` 是压缩后保留区的硬上限。如果最近的一组工具结果
      单独就超过它，保留整组仍会超窗口，此时返回 ``len(messages)``，
      把整段对话都交给摘要，保证压缩一定产生进展。
    """
    total = len(messages)
    if total == 0:
        return 0
    total_tokens = estimate(messages)
    if total_tokens <= keep_tokens and (
        max_tail_tokens is None or total_tokens <= max_tail_tokens
    ):
        return 0

    accumulated = 0
    crossing = 0
    for index in range(total - 1, -1, -1):
        accumulated += estimate([messages[index]])
        if accumulated >= keep_tokens:
            crossing = index
            break

    # 先找 user 边界，再退到 assistant 边界；两者都不会落在 tool 消息上。
    # 预算切点落在工具结果上时，这里会移到工具组之后的安全边界，
    # 丢弃整个组而不是拆开它。
    cut: int | None = None
    for role in ("user", "assistant"):
        for index in range(crossing, total):
            if messages[index].get("role") == role:
                cut = index
                break
        if cut is not None:
            break

    # 尾部只有工具结果、找不到更晚的边界：回退到最近的 assistant，
    # 让这一整组留在保留区。
    if cut is None:
        for index in range(crossing, -1, -1):
            if messages[index].get("role") == "assistant":
                cut = index
                break

    if cut is None:
        return 0

    if max_tail_tokens is not None and estimate(messages[cut:]) > max_tail_tokens:
        # 单组工具输出就超过压缩后的预算；整段摘要，避免下一 step 继续超限。
        return total
    return cut


def _truncate_text(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    keep = max(0, max_bytes - 120)
    head_bytes = keep * 2 // 3
    tail_bytes = keep - head_bytes
    head = raw[:head_bytes].decode("utf-8", "ignore")
    tail = raw[-tail_bytes:].decode("utf-8", "ignore") if tail_bytes else ""
    omitted = len(raw) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    marker = _TRUNCATION_MARKER.format(omitted=omitted)
    return f"{head}\n{marker}\n{tail}"


def _truncate_message_fields(
    message: dict[str, Any], budget_bytes: int
) -> dict[str, Any]:
    copy = dict(message)
    content = copy.get("content")
    if isinstance(content, str):
        copy["content"] = _truncate_text(content, budget_bytes)

    tool_calls = copy.get("tool_calls")
    if isinstance(tool_calls, list):
        truncated_calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                truncated_calls.append(call)
                continue
            call_copy = dict(call)
            function = call_copy.get("function")
            if isinstance(function, dict):
                function_copy = dict(function)
                arguments = function_copy.get("arguments")
                if isinstance(arguments, str):
                    function_copy["arguments"] = _truncate_text(
                        arguments, max(100, budget_bytes // 2)
                    )
                call_copy["function"] = function_copy
            truncated_calls.append(call_copy)
        copy["tool_calls"] = truncated_calls
    return copy


def _truncate_message(
    message: dict[str, Any],
    max_tokens: int,
    estimate: Callable[[list[dict[str, Any]]], int],
) -> dict[str, Any]:
    if estimate([message]) <= max_tokens:
        return message

    budget_bytes = max(200, max_tokens * 3 - 200)
    for _ in range(6):
        candidate = _truncate_message_fields(message, budget_bytes)
        if estimate([candidate]) <= max_tokens:
            return candidate
        budget_bytes //= 2
        if budget_bytes < 100:
            break
    return {
        "role": message.get("role", "user"),
        "content": "[message too large to fit the summary input]",
    }


def prepare_summary_input(
    messages: list[dict[str, Any]],
    max_input_tokens: int | None = None,
    estimate: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
) -> list[dict[str, Any]]:
    """把摘要输入压到 ``max_input_tokens`` 以内。

    - 单条超大消息保留头尾并插入截断标记（错误原文通常在头或尾）。
    - 总长度仍超限时，保留第一条消息（通常是最初目标）和最近的若干条，
      中间插入一条 ``[summary input truncated: N messages omitted]``。
    - ``max_input_tokens`` 为 None 或 <= 0 时不做限制。
    """
    if max_input_tokens is None or max_input_tokens <= 0:
        return list(messages)
    if not messages or estimate(messages) <= max_input_tokens:
        return list(messages)

    per_message_budget = max(1, max_input_tokens // 2)
    prepared = [
        _truncate_message(message, per_message_budget, estimate)
        for message in messages
    ]
    if estimate(prepared) <= max_input_tokens:
        return prepared

    first = prepared[0]
    marker = {
        "role": "user",
        "content": "[summary input truncated: messages omitted]",
    }
    remaining = (
        max_input_tokens - estimate([first]) - estimate([marker])
    )
    if remaining <= 0:
        return [first]

    tail: list[dict[str, Any]] = []
    used = 0
    for message in reversed(prepared[1:]):
        cost = estimate([message])
        if used + cost > remaining:
            break
        tail.append(message)
        used += cost

    omitted = len(prepared) - 1 - len(tail)
    if omitted <= 0 or not tail:
        return [first]

    marker["content"] = (
        f"[summary input truncated: {omitted} messages omitted]"
    )
    return [first, marker, *reversed(tail)]


def build_summary_prompt(messages: list[dict[str, Any]]) -> str:
    """把被丢弃的前缀序列化进一次独立的摘要请求。"""
    conversation = json.dumps(messages, ensure_ascii=False, indent=2)
    return (
        _SUMMARY_INSTRUCTIONS
        + "\n<conversation>\n"
        + conversation
        + "\n</conversation>\n"
    )


def missing_summary_sections(summary: str) -> list[str]:
    """返回缺失的段标题；空列表表示六段齐全。"""
    lines = {line.strip() for line in summary.splitlines()}
    return [heading for heading in SUMMARY_SECTIONS if f"## {heading}" not in lines]


def empty_summary_sections(summary: str) -> list[str]:
    """返回没有正文的段；标题存在但下一行就是下一个标题也算空。"""
    filled = {heading: False for heading in SUMMARY_SECTIONS}
    current: str | None = None
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            current = title if title in filled else None
        elif current is not None and stripped:
            filled[current] = True
    return [heading for heading in SUMMARY_SECTIONS if not filled[heading]]


def validate_summary(summary: str) -> bool:
    """六段标题齐全、每段非空（标题一字不差）。"""
    return (
        bool(summary.strip())
        and not missing_summary_sections(summary)
        and not empty_summary_sections(summary)
    )


def _path_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for token in _PATH_TOKEN_RE.findall(text):
        token = token.strip("()[]{}<>\"'`,;:")
        if not token or "://" in token or token.startswith("//"):
            continue
        is_absolute = token.startswith("/") or bool(
            re.match(r"^[A-Za-z]:[\\/]", token)
        )
        is_relative = bool(
            re.search(r"[\w.-]+[\\/][\w./-]*\.[A-Za-z0-9]+$", token)
        )
        if is_absolute or is_relative:
            candidates.append(token)
    return candidates


def critical_facts(
    messages: list[dict[str, Any]],
    *,
    max_paths: int = 5,
    max_errors: int = 3,
) -> list[str]:
    """从摘要输入里抽取必须原样保留的事实：文件路径和 ERROR 行。"""
    paths: list[str] = []
    errors: list[str] = []

    def add(bucket: list[str], value: str, limit: int) -> None:
        if value and value not in bucket and len(bucket) < limit:
            bucket.append(value)

    for message in messages:
        text_parts: list[str] = []
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif content is not None:
            text_parts.append(json.dumps(content, ensure_ascii=False))

        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str):
                text_parts.append(arguments)

        for text in text_parts:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("ERROR:"):
                    add(errors, stripped[:300], max_errors)
            for path in _path_candidates(text):
                add(paths, path, max_paths)

    return [*paths, *errors]


def missing_critical_facts(
    summary: str, messages: list[dict[str, Any]]
) -> list[str]:
    """摘要里没有原样出现的关键路径 / 错误行。"""
    return [fact for fact in critical_facts(messages) if fact not in summary]


def summarize(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    max_input_tokens: int | None = None,
) -> str:
    """用同一个 LLM client 生成摘要；不传 tools，低温。

    输入超限时先截断；摘要缺段、空段或缺少关键路径/错误原文时抛
    SummaryError，由 loop 转成 CompactionFailed 事件。
    """
    prepared = prepare_summary_input(messages, max_input_tokens)
    response = llm.chat(
        [{"role": "user", "content": build_summary_prompt(prepared)}],
        tools=None,
        temperature=0.0,
    )
    content = (response.message or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise SummaryError("summary returned no text")
    summary = content.strip()

    missing = missing_summary_sections(summary)
    empty = empty_summary_sections(summary)
    if missing or empty:
        labels = [*missing, *(f"{heading} (empty)" for heading in empty)]
        raise SummaryError("summary missing sections: " + ", ".join(labels))

    facts = missing_critical_facts(summary, prepared)
    if facts:
        raise SummaryError(
            "summary missing critical facts: " + "; ".join(facts)
        )
    return summary