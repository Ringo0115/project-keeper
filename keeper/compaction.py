"""M4 最小压缩：token 估算、切点选择、六段摘要。

切点与 split-turn 组合移植自 pi-agent（MIT，Copyright (c) 2025 Mario
Zechner）：``packages/coding-agent/src/core/compaction/compaction.ts`` 的
``findCutPoint`` / ``estimateContextTokens`` / split-turn 合并；摘要输入里
工具结果的 2000 字符截断移植自同仓库 ``compaction/utils.ts``。

失败降级（可用性检查 → 一次定向重试 → 确定性摘要）只借 Orbital
（GPL-3.0）的结构和规则，不复制其代码或文案。

这里只放纯逻辑和摘要调用；session JSONL 的写入由 ``Session.compact`` 负责，
事件编排由 ``AgentLoop`` 负责。切点算法不拆分 assistant 的 tool_calls 与
对应的 tool 结果。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from keeper.session import SUMMARY_PREFIX

SUMMARY_SECTIONS = (
    "Goal",
    "Constraints & Preferences",
    "Progress",
    "Key Decisions",
    "Next Steps",
    "Critical Context",
)

# split-turn 时只总结被丢弃的当前轮次前缀；沿用 pi-agent 的三段结构。
TURN_PREFIX_SECTIONS = (
    "Original Request",
    "Progress So Far",
    "Context Needed to Continue",
)

SPLIT_TURN_MARKER = "**Turn Context (split turn):**"

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

# pi-agent 的 turn prefix prompt 改写为中文；规则一致：只总结给出的内容，
# 不推断或复述后面保留的消息。
_TURN_PREFIX_INSTRUCTIONS = """下面是当前这一轮对话中较早、即将被丢弃的部分。后面的消息会另行保留，
不需要你重建。

请写成一份简短的检查点，放在保留消息之前，让对话能继续。

## Original Request
[用户最初要求做什么？]

## Progress So Far
- [这些消息里已经做出的决定和完成的工作]

## Context Needed to Continue
- [理解后续工作所需的信息]

只总结上面明确出现的信息，不要推断或重建后面的消息。
"""

_TRUNCATION_NOTE = "\n\n[... {omitted} more characters truncated]"

# 移植自 pi-agent ``compaction/utils.ts``：摘要输入里单条工具结果的字符上限。
SUMMARY_ITEM_CHARS = 2000

_TOOL_MARKUP_RE = re.compile(
    r"<tool_call|<function[=\s>]|<invoke\b|\"tool_calls\"\s*:|\"function_call\"\s*:|\"tool_use\"",
    re.IGNORECASE,
)
_PATH_TOKEN_RE = re.compile(r"[^\s\"'`<>]+")


@dataclass(frozen=True)
class SummaryResult:
    """一次摘要的结果；``degraded`` 表示用了确定性兜底而不是 LLM 摘要。"""

    text: str
    degraded: bool = False
    source: str = "llm"


@dataclass(frozen=True)
class CutPoint:
    """保留区起点；split-turn 时额外给出被切开的轮次起点。"""

    cut_index: int
    is_split_turn: bool = False
    turn_start_index: int = -1


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """确定性 token 估算：UTF-8 JSON 字节数 // 3。

    不引入 tokenizer；同一组消息永远得到同一个数。
    """
    payload = json.dumps(messages, ensure_ascii=False)
    return len(payload.encode("utf-8")) // 3


def estimate_request_tokens(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> int:
    """估算一次请求的上下文大小；工具 schema 与 pi 的 ``toolsAdded`` 对应。"""
    total = estimate_tokens(messages)
    if tools:
        total += len(json.dumps(tools, ensure_ascii=False).encode("utf-8")) // 3
    return total


def _is_previous_summary(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith(SUMMARY_PREFIX)
    )


def _is_turn_start(message: dict[str, Any]) -> bool:
    """轮次起点是用户消息；旧摘要不是轮次起点（pi 把 compaction 当边界）。"""
    return message.get("role") == "user" and not _is_previous_summary(message)


def find_cut_point(
    messages: list[dict[str, Any]],
    keep_tokens: int,
    estimate: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
    *,
    max_tail_tokens: int | None = None,
) -> CutPoint:
    """返回保留区间的起点：``messages[cut_index:]`` 被保留。

    - 从最新消息往回累加，达到 ``keep_tokens`` 后确定预算切点。
    - 切点优先落在 user 消息上，其次落在 assistant 边界上。
    - 绝不返回 tool 消息下标，因此不会把 assistant 的 tool_calls 和
      tool 结果拆开。若预算切点落在工具结果上且后面没有安全边界，
      则回退到拥有该 tool_call 的 assistant（宁可多保留整组）。
    - 整段对话都在预算内时返回 0，表示无需压缩。
    - ``max_tail_tokens`` 是保留区的窄窗口守卫：最近一组工具结果单独
      超过它时，保留整组会让下一次请求放不下摘要，此时返回
      ``len(messages)``，整段交给摘要。
    - 切点落在轮次中间且轮次起点不是第一条消息时，返回 split-turn，
      由调用方分别做历史摘要和轮次前缀摘要（pi-agent 的做法）。
    """
    total = len(messages)
    if total == 0:
        return CutPoint(0)
    total_tokens = estimate(messages)
    if total_tokens <= keep_tokens and (
        max_tail_tokens is None or total_tokens <= max_tail_tokens
    ):
        return CutPoint(0)

    accumulated = 0
    crossing = 0
    for index in range(total - 1, -1, -1):
        accumulated += estimate([messages[index]])
        if accumulated >= keep_tokens:
            crossing = index
            break

    # 先找 user 边界，再退到 assistant 边界；两者都不会落在 tool 消息上。
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
        return CutPoint(0)

    if max_tail_tokens is not None and estimate(messages[cut:]) > max_tail_tokens:
        # 单组工具输出就超过压缩后的预算；整段摘要，避免下一 step 继续超限。
        return CutPoint(total)

    if _is_turn_start(messages[cut]):
        return CutPoint(cut)

    turn_start = -1
    for index in range(cut - 1, -1, -1):
        if _is_turn_start(messages[index]):
            turn_start = index
            break

    # 轮次起点就是第一条消息时没有“历史”可分开总结，直接按六段总结整个前缀。
    if 0 < turn_start < cut:
        return CutPoint(cut, True, turn_start)
    return CutPoint(cut)


def _truncate_for_summary(text: str, max_chars: int = SUMMARY_ITEM_CHARS) -> str:
    """只留头部并加标记；移植自 pi-agent 的 ``truncateForSummary``。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + _TRUNCATION_NOTE.format(
        omitted=len(text) - max_chars
    )


def _summary_view(message: dict[str, Any]) -> dict[str, Any]:
    """摘要输入视图：工具结果与工具调用参数做头截断。

    pi 只截工具结果；工具调用参数是 keeper 的有意扩展——一个带巨大
    ``content`` 的 write_file 调用同样能撑爆摘要输入。
    """
    view = dict(message)
    content = view.get("content")
    if message.get("role") == "tool" and isinstance(content, str):
        view["content"] = _truncate_for_summary(content)

    tool_calls = view.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                calls.append(call)
                continue
            call_copy = dict(call)
            function = call_copy.get("function")
            if isinstance(function, dict):
                function_copy = dict(function)
                arguments = function_copy.get("arguments")
                if isinstance(arguments, str):
                    function_copy["arguments"] = _truncate_for_summary(arguments)
                call_copy["function"] = function_copy
            calls.append(call_copy)
        view["tool_calls"] = calls
    return view


def _fit_first_message(
    message: dict[str, Any],
    max_input_tokens: int,
    estimate: Callable[[list[dict[str, Any]]], int],
) -> dict[str, Any]:
    """整体兜底只剩第一条时，保证它自己也在预算内。"""
    if estimate([message]) <= max_input_tokens:
        return message
    content = message.get("content")
    if not isinstance(content, str):
        return message
    # 中文一个字符约 1 token，按字符数保守截断比按 ASCII 的 bytes//3 更安全。
    budget_chars = max(200, max_input_tokens - 200)
    copy = dict(message)
    copy["content"] = _truncate_for_summary(content, budget_chars)
    return copy


def prepare_summary_input(
    messages: list[dict[str, Any]],
    max_input_tokens: int | None = None,
    estimate: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
) -> list[dict[str, Any]]:
    """把摘要输入压到 ``max_input_tokens`` 以内。

    第一段照 pi-agent：单条工具结果 2000 字符头截断（工具调用参数同样处理）。
    第二段是 keeper 对 pi 的补充：pi 没有整体上限，而 keeper 的摘要调用
    必须自己塞进窗口；整体仍超限时保留第一条（通常是最初目标）和最近的
    若干条，中间插入 ``[summary input truncated: N messages omitted]``。
    """
    if not messages:
        return []

    # 单条截断是 pi 的固定规则，不依赖整体预算；只有第二段需要预算。
    prepared = [_summary_view(message) for message in messages]
    if max_input_tokens is None or max_input_tokens <= 0:
        return prepared
    if estimate(prepared) <= max_input_tokens:
        return prepared

    first = prepared[0]
    marker = {
        "role": "user",
        "content": "[summary input truncated: messages omitted]",
    }
    remaining = max_input_tokens - estimate([first]) - estimate([marker])
    if remaining <= 0:
        return [_fit_first_message(first, max_input_tokens, estimate)]

    tail: list[dict[str, Any]] = []
    for message in reversed(prepared[1:]):
        cost = estimate([message])
        if cost > remaining:
            break
        tail.append(message)
        remaining -= cost

    omitted = len(prepared) - 1 - len(tail)
    if not tail or omitted <= 0:
        return [_fit_first_message(first, max_input_tokens, estimate)]

    marker["content"] = f"[summary input truncated: {omitted} messages omitted]"
    return [first, marker, *reversed(tail)]


def _retry_note(note: str | None) -> str:
    if not note:
        return ""
    return (
        "\n上一次摘要不可用："
        + note
        + "。请重新输出，标题逐字照抄，每段必须有内容；"
        "不要调用工具，不要输出 JSON 或 XML，只输出摘要正文。\n"
    )


def build_summary_prompt(
    messages: list[dict[str, Any]], retry_note: str | None = None
) -> str:
    """把被丢弃的前缀序列化进一次独立的摘要请求。"""
    conversation = json.dumps(messages, ensure_ascii=False, indent=2)
    return (
        _SUMMARY_INSTRUCTIONS
        + _retry_note(retry_note)
        + "\n<conversation>\n"
        + conversation
        + "\n</conversation>\n"
    )


def build_turn_prefix_prompt(
    messages: list[dict[str, Any]], retry_note: str | None = None
) -> str:
    """split-turn 时总结当前轮次被丢弃的前缀。"""
    conversation = json.dumps(messages, ensure_ascii=False, indent=2)
    return (
        _TURN_PREFIX_INSTRUCTIONS
        + _retry_note(retry_note)
        + "\n<conversation>\n"
        + conversation
        + "\n</conversation>\n"
    )


def merge_split_summary(history_text: str, turn_text: str) -> str:
    """把历史摘要和轮次前缀摘要合成一条，与 pi-agent 的合并格式一致。"""
    return f"{history_text}\n\n---\n\n{SPLIT_TURN_MARKER}\n\n{turn_text}"


def missing_summary_sections(
    summary: str, sections: tuple[str, ...] = SUMMARY_SECTIONS
) -> list[str]:
    """返回缺失的段标题；空列表表示要求的段齐全。"""
    lines = {line.strip() for line in summary.splitlines()}
    return [heading for heading in sections if f"## {heading}" not in lines]


def empty_summary_sections(
    summary: str, sections: tuple[str, ...] = SUMMARY_SECTIONS
) -> list[str]:
    """返回没有正文的段；标题存在但下一行就是下一个标题也算空。"""
    filled = {heading: False for heading in sections}
    current: str | None = None
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            current = title if title in filled else None
        elif current is not None and stripped:
            filled[current] = True
    return [heading for heading in sections if not filled[heading]]


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


def _json_strings(value: Any):
    """递归取出 JSON 里的字符串值；用于解码工具参数，避免转义噪声。"""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)


def _argument_texts(arguments: Any) -> list[str]:
    if not isinstance(arguments, str):
        return []
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return [arguments]
    return list(_json_strings(parsed))


def critical_facts(
    messages: list[dict[str, Any]],
    *,
    max_paths: int = 5,
    max_errors: int = 3,
) -> list[str]:
    """从摘要输入里抽取必须保留的事实：文件路径和 ERROR 行。

    工具参数先解码再扫描：JSON 里的 ``C:\\\\x`` 会被还原成 ``C:\\x``，
    避免模型写单反斜杠时被误判为缺失。
    """
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
            text_parts.extend(_argument_texts(arguments))

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


def salvage_critical_facts(
    summary: str,
    messages: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> str:
    """把摘要漏掉的关键路径 / 错误原文逐字补进 Critical Context。

    新设计（两个参考都没有）：keeper D-015 要求路径和错误原文不丢，但
    硬失败会误杀；确定性地追加原文既不依赖模型，也不中止压缩。限
    ``limit`` 条、去重、换行压平，避免失控增长。
    """
    missing = [fact for fact in critical_facts(messages) if fact not in summary]
    if not missing:
        return summary

    facts: list[str] = []
    for fact in missing:
        line = " ".join(fact.split())
        if line and line not in facts:
            facts.append(line)
        if len(facts) >= limit:
            break
    if not facts:
        return summary

    block = "\n".join(f"- {fact}" for fact in facts)
    heading = "## Critical Context"
    lines = summary.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == heading
        )
    except StopIteration:
        return summary.rstrip() + f"\n\n{heading}\n{block}"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].strip().startswith("## "):
            end = index
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "\n".join([*lines[:end], block, "", *lines[end:]])


def _digest(messages: list[dict[str, Any]]) -> str:
    """确定性兜底摘要：旧摘要、用户消息原文、每条工具调用一行、关键事实。

    结构借用 Orbital 的 ``_digest``（旧摘要 / 用户消息 / 工具调用），文案
    自己写；额外保留关键事实是 keeper 的扩展，因为 D-015 的路径和错误
    原文保证不能因为摘要器失败而静默丢失。
    """
    lines = ["[摘要器返回不可用；以下是确定性转录摘要。]"]

    earlier = [message for message in messages if _is_previous_summary(message)]
    if earlier:
        body = str(earlier[-1].get("content") or "")
        if body.startswith(SUMMARY_PREFIX):
            body = body[len(SUMMARY_PREFIX):]
        lines += ["", "Earlier summary:", body]

    users = [
        message
        for message in messages
        if message.get("role") == "user" and not _is_previous_summary(message)
    ]
    if users:
        lines += ["", "User messages:"]
        # Orbital 的规则：用户消息原文保留，保证兜底摘要不丢任务描述。
        for message in users:
            lines.append(str(message.get("content") or ""))

    calls = [
        call
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]
    if calls:
        lines += ["", "Tool calls:"]
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else {}
            function = function or {}
            lines.append(
                f"- {function.get('name', '?')} {str(function.get('arguments', ''))[:200]}"
            )

    facts = critical_facts(messages)
    if facts:
        lines += ["", "Critical facts:"]
        lines += [f"- {' '.join(fact.split())}" for fact in facts[:8]]

    return "\n".join(lines)


def _looks_like_json_or_tool_markup(text: str) -> bool:
    if text.lstrip().startswith("{"):
        return True
    try:
        json.loads(text)
        return True
    except ValueError:
        pass
    return bool(_TOOL_MARKUP_RE.search(text))


def _summary_failure(
    response: Any,
    content: Any,
    sections: tuple[str, ...],
) -> str | None:
    """返回不可用原因；None 表示可用。规则借自 Orbital 的 usable 检查。"""
    if getattr(response, "finish_reason", None) == "length":
        return "响应因 token 上限被截断，摘要不完整"
    if not isinstance(content, str) or not content.strip():
        return "响应为空"
    message = getattr(response, "message", None)
    if isinstance(message, dict) and message.get("tool_calls"):
        return "响应包含工具调用"
    text = content.strip()
    if _looks_like_json_or_tool_markup(text):
        return "响应是 JSON 或工具调用文本"
    missing = missing_summary_sections(text, sections)
    empty = empty_summary_sections(text, sections)
    if missing or empty:
        labels = [*missing, *(f"{heading} (empty)" for heading in empty)]
        return "缺少段落: " + ", ".join(labels)
    return None


def summarize(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    max_input_tokens: int | None = None,
    kind: str = "history",
) -> SummaryResult:
    """生成摘要；不传 tools，低温。

    流程：可用性 + 段落检查 → 一次定向重试 → 确定性 ``_digest`` 兜底。
    传输错误不在这里二次重试（``LLMClient`` 已重试）；直接降级。
    ``kind="turn_prefix"`` 时使用三段 prompt。
    """
    prepared = prepare_summary_input(messages, max_input_tokens)
    sections = SUMMARY_SECTIONS if kind == "history" else TURN_PREFIX_SECTIONS
    build = (
        build_summary_prompt if kind == "history" else build_turn_prefix_prompt
    )

    # 新设计：keeper 要求六段，Orbital 的 plain-prose 重试会违反这个契约，
    # 所以第二次尝试是「指出失败原因 + 重申六段」，而不是换一种摘要格式。
    failure: str | None = None
    for _ in (1, 2):
        prompt = build(prepared, retry_note=failure)
        try:
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
            )
        except Exception:
            # 传输层已经由 LLMClient 重试过；重复调用只会让失败更慢。
            return SummaryResult(_digest(prepared), True, "digest")

        message = getattr(response, "message", None)
        content = message.get("content") if isinstance(message, dict) else None
        failure = _summary_failure(response, content, sections)
        if failure is None:
            summary = salvage_critical_facts(str(content).strip(), prepared)
            return SummaryResult(summary, False, "llm")

    return SummaryResult(_digest(prepared), True, "digest")