"""M4 纯函数：token 估算、切点、摘要输入、保真与降级、配置。"""

from __future__ import annotations

import json

import pytest

from keeper.compaction import (
    SUMMARY_SECTIONS,
    TURN_PREFIX_SECTIONS,
    build_summary_prompt,
    build_turn_prefix_prompt,
    critical_facts,
    empty_summary_sections,
    estimate_request_tokens,
    estimate_tokens,
    find_cut_point,
    merge_split_summary,
    missing_critical_facts,
    missing_summary_sections,
    prepare_summary_input,
    salvage_critical_facts,
    summarize,
    validate_summary,
)
from keeper.llm import LLMResponse
from keeper.session import SUMMARY_PREFIX


def _len_estimate(messages: list[dict]) -> int:
    """测试用确定性估算：只数 content 字符数。"""
    return sum(len(message.get("content") or "") for message in messages)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(
    text: str = "", call_ids: tuple[str, ...] = (), arguments: str = "{}"
) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if call_ids:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": arguments},
            }
            for call_id in call_ids
        ]
    return message


def _tool(call_id: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _valid_summary(tag: str = "ok") -> str:
    return "\n".join(
        [
            "## Goal",
            f"- {tag}",
            "## Constraints & Preferences",
            "- 不引入新依赖",
            "## Progress",
            "- 进行中",
            "## Key Decisions",
            "- 采用最小压缩",
            "## Next Steps",
            "1. 跑测试",
            "## Critical Context",
            "- /proj/keeper/loop.py",
        ]
    )


def _turn_summary(tag: str = "turn") -> str:
    return "\n".join(
        [
            "## Original Request",
            f"- {tag}",
            "## Progress So Far",
            "- 已读取文件",
            "## Context Needed to Continue",
            "- 继续处理",
        ]
    )


# ---------- token 估算 ----------


def test_estimate_tokens_is_deterministic_utf8_bytes_over_three() -> None:
    messages = [{"role": "user", "content": "中文 abc\n第二行"}]
    expected = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) // 3
    assert estimate_tokens(messages) == expected
    assert estimate_tokens(messages) == estimate_tokens(list(messages))


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens([]) == len("[]") // 3


def test_estimate_request_tokens_counts_tool_schema() -> None:
    messages = [_user("hello")]
    tools = [
        {
            "type": "function",
            "function": {"name": "read_file", "description": "读取文件"},
        }
    ]
    schema_tokens = len(json.dumps(tools, ensure_ascii=False).encode("utf-8")) // 3

    assert estimate_request_tokens(messages, None) == estimate_tokens(messages)
    assert (
        estimate_request_tokens(messages, tools)
        == estimate_tokens(messages) + schema_tokens
    )


# ---------- 切点 ----------


def test_find_cut_point_keeps_everything_under_budget() -> None:
    messages = [_user("hello"), _assistant("world")]
    assert find_cut_point(messages, 10_000, _len_estimate).cut_index == 0
    assert find_cut_point([], 1, _len_estimate).cut_index == 0


def test_find_cut_point_prefers_user_boundary() -> None:
    messages = [
        _user("a" * 10),
        _assistant("b" * 10),
        _user("c" * 10),
        _assistant("d" * 10),
    ]
    point = find_cut_point(messages, 25, _len_estimate)
    assert point.cut_index == 2
    assert messages[point.cut_index]["role"] == "user"
    # 切点本身就是轮次起点，不是 split-turn。
    assert point.is_split_turn is False


def test_find_cut_point_never_returns_tool_message() -> None:
    messages = [
        _user("u"),
        _assistant(call_ids=("c1", "c2")),
        _tool("c1", "x" * 50),
        _tool("c2", "y" * 50),
        _assistant("later"),
        _tool("c3", "z" * 50),
    ]
    for keep in range(1, 200):
        cut = find_cut_point(messages, keep, _len_estimate).cut_index
        assert messages[cut]["role"] != "tool", (keep, cut)


def test_cut_after_tool_group_discards_whole_group() -> None:
    messages = [
        _user("goal"),
        _assistant(call_ids=("c1",)),
        _tool("c1", "H" * 100),
        _assistant("next step"),
        _tool("c2", "small"),
    ]
    cut = find_cut_point(messages, 40, _len_estimate).cut_index
    assert cut == 3
    assert messages[cut]["role"] == "assistant"
    # 被丢弃的前缀里工具组是完整的；保留区不以 tool 开头。
    assert messages[cut - 1]["role"] == "tool"


def test_tail_tool_group_moves_back_to_owning_assistant() -> None:
    messages = [
        _user("goal"),
        _assistant(call_ids=("c1",)),
        _tool("c1", "H" * 100),
    ]
    cut = find_cut_point(messages, 40, _len_estimate).cut_index
    assert cut == 1
    assert messages[cut]["role"] == "assistant"
    assert messages[cut]["tool_calls"][0]["id"] == "c1"
    # assistant 和它的 tool 结果一起留在保留区。
    assert [m["role"] for m in messages[cut:]] == ["assistant", "tool"]


def test_multi_tool_results_stay_with_their_assistant() -> None:
    messages = [
        _user("goal"),
        _assistant(call_ids=("c1", "c2", "c3")),
        _tool("c1", "a" * 30),
        _tool("c2", "b" * 30),
        _tool("c3", "c" * 30),
    ]
    cut = find_cut_point(messages, 1, _len_estimate).cut_index
    assert cut == 1
    kept_ids = [
        message["tool_call_id"]
        for message in messages[cut:]
        if message["role"] == "tool"
    ]
    assert kept_ids == ["c1", "c2", "c3"]


def test_find_cut_point_drops_tail_group_when_it_exceeds_budget() -> None:
    messages = [
        _user("goal"),
        _assistant(call_ids=("c1",)),
        _tool("c1", "H" * 100),
    ]
    # 默认保留整组（宁可多保留一点）。
    assert find_cut_point(messages, 40, _len_estimate).cut_index == 1
    assert (
        find_cut_point(
            messages, 40, _len_estimate, max_tail_tokens=200
        ).cut_index
        == 1
    )
    # 整组超过压缩后的硬上限：全部交给摘要，保证有进展。
    assert (
        find_cut_point(
            messages, 40, _len_estimate, max_tail_tokens=50
        ).cut_index
        == 3
    )


def test_find_cut_point_marks_split_turn() -> None:
    messages = [
        _user("first"),
        _assistant("done"),
        _user("second"),
        _assistant(call_ids=("c1",)),
        _tool("c1", "x" * 50),
    ]
    point = find_cut_point(messages, 10, _len_estimate)
    assert point.cut_index == 3
    assert point.is_split_turn is True
    assert point.turn_start_index == 2


def test_find_cut_point_no_split_when_turn_starts_first() -> None:
    messages = [
        _user("goal"),
        _assistant(call_ids=("c1",)),
        _tool("c1", "H" * 100),
    ]
    point = find_cut_point(messages, 40, _len_estimate)
    assert point.cut_index == 1
    assert point.is_split_turn is False


def test_find_cut_point_ignores_previous_summary_as_turn_start() -> None:
    messages = [
        {"role": "user", "content": SUMMARY_PREFIX + "旧摘要"},
        _assistant(call_ids=("c1",)),
        _tool("c1", "H" * 100),
    ]
    point = find_cut_point(messages, 40, _len_estimate)
    assert point.cut_index == 1
    assert point.is_split_turn is False


# ---------- 摘要输入 ----------


def test_prepare_summary_input_truncates_huge_tool_message() -> None:
    huge = (
        "ERROR: boom at /proj/app/main.py\n"
        + "x" * 20_000
        + "\ntrace line 4999"
    )
    messages = [_user("initial goal"), _tool("c1", huge)]

    prepared = prepare_summary_input(messages, max_input_tokens=5_000)
    text = json.dumps(prepared, ensure_ascii=False)

    assert estimate_tokens(prepared) <= 5_000
    assert "initial goal" in text
    assert "ERROR: boom at /proj/app/main.py" in text
    assert "more characters truncated" in text
    # 只留头部，尾行不再进入摘要输入（pi 的规则）。
    assert "trace line 4999" not in text


def test_prepare_summary_input_truncates_tool_call_arguments() -> None:
    arguments = '{"content": "' + "a" * 5_000 + '"}'
    messages = [_assistant(call_ids=("c1",), arguments=arguments)]

    prepared = prepare_summary_input(messages)
    call = prepared[0]["tool_calls"][0]

    assert "more characters truncated" in call["function"]["arguments"]
    assert len(call["function"]["arguments"]) < 5_000


def test_prepare_summary_input_drops_middle_messages() -> None:
    messages = [_user("initial goal")] + [
        {"role": "assistant", "content": f"msg-{index}-" + "x" * 2_000}
        for index in range(50)
    ]

    prepared = prepare_summary_input(messages, max_input_tokens=3_000)
    text = json.dumps(prepared, ensure_ascii=False)

    assert estimate_tokens(prepared) <= 3_000
    assert prepared[0]["content"] == "initial goal"
    assert "messages omitted" in text
    assert "msg-49-" in text
    assert "msg-0-" not in text


def test_prepare_summary_input_fits_oversized_first_message() -> None:
    messages = [_user("goal\n" + "x" * 20_000)]

    prepared = prepare_summary_input(messages, max_input_tokens=1_000)

    assert len(prepared) == 1
    assert estimate_tokens(prepared) <= 1_000
    assert "more characters truncated" in prepared[0]["content"]


# ---------- 摘要 prompt / 校验 ----------


def test_build_summary_prompt_contains_raw_prefix_and_six_sections() -> None:
    messages = [
        _user("修复 /proj/app/main.py 的导入错误"),
        _tool("c1", "ERROR: [Errno 2] No such file or directory"),
    ]
    prompt = build_summary_prompt(messages)

    assert "修复 /proj/app/main.py 的导入错误" in prompt
    assert "ERROR: [Errno 2] No such file or directory" in prompt
    assert "错误原文" in prompt
    for heading in SUMMARY_SECTIONS:
        assert f"## {heading}" in prompt


def test_build_turn_prefix_prompt_has_three_sections() -> None:
    prompt = build_turn_prefix_prompt([_user("second")])

    for heading in TURN_PREFIX_SECTIONS:
        assert f"## {heading}" in prompt
    assert "second" in prompt


def test_validate_summary_requires_all_six_exact_headings() -> None:
    assert validate_summary(_valid_summary()) is True
    assert missing_summary_sections(_valid_summary()) == []

    missing_progress = _valid_summary().replace(
        "## Progress", "## progress"
    )
    assert validate_summary(missing_progress) is False
    assert missing_summary_sections(missing_progress) == ["Progress"]

    assert validate_summary("") is False
    assert validate_summary("## Goal\n只有一段") is False


def test_validate_summary_rejects_empty_sections() -> None:
    empty_goal = _valid_summary().replace("## Goal\n- ok", "## Goal")
    assert validate_summary(empty_goal) is False
    assert empty_summary_sections(empty_goal) == ["Goal"]


def test_critical_facts_extracts_paths_and_error_lines() -> None:
    messages = [
        _tool("c1", "ERROR: [Errno 2] No such file: '/proj/app/main.py'"),
        _user("see https://api.deepseek.com/v1 and keeper/loop.py"),
    ]
    facts = critical_facts(messages)

    assert "/proj/app/main.py" in facts
    assert (
        "ERROR: [Errno 2] No such file: '/proj/app/main.py'" in facts
    )
    assert "keeper/loop.py" in facts
    assert not any("deepseek.com" in fact for fact in facts)


def test_critical_facts_decodes_windows_paths_in_tool_arguments() -> None:
    # JSON 里是双反斜杠；解码后必须是模型会写的单反斜杠路径。
    arguments = '{"path": "C:\\\\proj\\\\app.py"}'
    messages = [_assistant(call_ids=("c1",), arguments=arguments), _tool("c1", "ok")]

    facts = critical_facts(messages)

    assert "C:\\proj\\app.py" in facts
    assert all("C:\\\\proj" not in fact for fact in facts)


# ---------- 摘要生成 / 保真 / 降级 ----------


def test_summarize_uses_no_tools_and_low_temperature() -> None:
    llm = _FakeSummaryLLM(_valid_summary())
    result = summarize(llm, [_user("hello")])

    assert result.text == _valid_summary()
    assert result.degraded is False
    assert result.source == "llm"
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None
    assert llm.calls[0]["temperature"] == 0.0
    assert "hello" in llm.calls[0]["messages"][0]["content"]


def test_summarize_salvages_missing_critical_facts() -> None:
    messages = [_tool("c1", "ERROR: boom at /proj/app/main.py")]
    llm = _FakeSummaryLLM(_valid_summary())  # 只有 /proj/keeper/loop.py

    result = summarize(llm, messages)

    assert result.degraded is False
    assert missing_critical_facts(result.text, messages) == []
    assert "/proj/app/main.py" in result.text
    assert "ERROR: boom at /proj/app/main.py" in result.text
    # 补救后仍是合法的六段摘要。
    assert validate_summary(result.text)


def test_summarize_accepts_preserved_critical_facts() -> None:
    messages = [_tool("c1", "ERROR: boom at /proj/app/main.py")]
    summary = _valid_summary().replace(
        "- /proj/keeper/loop.py",
        "- /proj/app/main.py\n- ERROR: boom at /proj/app/main.py",
    )
    llm = _FakeSummaryLLM(summary)

    result = summarize(llm, messages)

    assert result.text == summary
    assert missing_critical_facts(result.text, messages) == []


def test_summarize_truncates_oversized_input() -> None:
    huge = (
        "ERROR: boom at /proj/app/main.py\n"
        + "x" * 20_000
        + "\ntrace line 4999"
    )
    messages = [_user("initial goal"), _tool("c1", huge)]
    summary = _valid_summary().replace(
        "- /proj/keeper/loop.py",
        "- /proj/app/main.py\n- ERROR: boom at /proj/app/main.py",
    )
    llm = _FakeSummaryLLM(summary)

    result = summarize(llm, messages, max_input_tokens=5_000)

    assert result.text == summary
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "initial goal" in prompt
    assert "ERROR: boom at /proj/app/main.py" in prompt
    assert "more characters truncated" in prompt
    assert "trace line 4999" not in prompt


def test_summarize_falls_back_to_digest_after_bad_summary() -> None:
    llm = _FakeSummaryLLM("## Goal\n只有目标")
    result = summarize(llm, [_user("hello")])

    assert result.degraded is True
    assert result.source == "digest"
    assert "[摘要器返回不可用" in result.text
    # 内容失败会做一次定向重试，然后才降级。
    assert len(llm.calls) == 2
    assert "上一次摘要不可用" in llm.calls[1]["messages"][0]["content"]


def test_summarize_retries_once_then_uses_valid_summary() -> None:
    llm = _FakeSummaryLLM("not a summary", _valid_summary())

    result = summarize(llm, [_user("hello")])

    assert result.text == _valid_summary()
    assert result.degraded is False
    assert len(llm.calls) == 2


def test_summarize_treats_length_finish_reason_as_failure() -> None:
    llm = _FakeSummaryLLM(
        _valid_summary(), _valid_summary(), finish_reasons=["length", None]
    )

    result = summarize(llm, [_user("hello")])

    assert result.text == _valid_summary()
    assert result.degraded is False
    assert len(llm.calls) == 2
    assert "token 上限" in llm.calls[1]["messages"][0]["content"]


def test_summarize_transport_error_uses_digest_without_retry() -> None:
    llm = _FakeSummaryLLM(error=RuntimeError("backend down"))

    result = summarize(llm, [_user("goal")])

    assert result.degraded is True
    assert len(llm.calls) == 1
    assert "goal" in result.text


def test_summarize_rejects_tool_call_response_and_digests() -> None:
    llm = _ToolCallSummaryLLM()

    result = summarize(llm, [_user("hello")])

    assert result.degraded is True
    assert len(llm.calls) == 2


def test_summarize_turn_prefix_uses_three_sections() -> None:
    llm = _FakeSummaryLLM(_turn_summary())

    result = summarize(llm, [_user("second")], kind="turn_prefix")

    assert result.text == _turn_summary()
    assert result.degraded is False
    assert "## Original Request" in llm.calls[0]["messages"][0]["content"]


def test_merge_split_summary_contains_marker() -> None:
    merged = merge_split_summary("history", "turn")

    assert merged.startswith("history")
    assert "**Turn Context (split turn):**" in merged
    assert merged.endswith("turn")


def test_digest_keeps_previous_summary_user_messages_and_facts() -> None:
    messages = [
        {"role": "user", "content": SUMMARY_PREFIX + "old summary"},
        _user("goal A"),
        _tool("c1", "ERROR: boom at /proj/app/main.py"),
    ]
    llm = _FakeSummaryLLM("## Goal\n只有目标")

    result = summarize(llm, messages)

    assert result.degraded is True
    assert "Earlier summary:" in result.text
    assert "old summary" in result.text
    assert "User messages:" in result.text
    assert "goal A" in result.text
    assert "Critical facts:" in result.text
    assert "ERROR: boom at /proj/app/main.py" in result.text


def test_salvage_critical_facts_does_not_duplicate() -> None:
    messages = [_tool("c1", "ERROR: boom at /proj/app/main.py")]
    summary = _valid_summary().replace(
        "- /proj/keeper/loop.py", "- /proj/app/main.py"
    )

    salvaged = salvage_critical_facts(summary, messages)

    assert salvaged.count("- /proj/app/main.py") == 1
    assert "ERROR: boom at /proj/app/main.py" in salvaged
    # 再次补救是幂等的，不会重复追加。
    assert salvage_critical_facts(salvaged, messages) == salvaged


class _FakeSummaryLLM:
    def __init__(
        self,
        *contents: str,
        finish_reasons: list[str | None] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.contents = list(contents)
        self.finish_reasons = list(finish_reasons or [])
        self.error = error
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, temperature=None):
        self.calls.append(
            {"messages": messages, "tools": tools, "temperature": temperature}
        )
        if self.error is not None:
            raise self.error
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        finish = (
            self.finish_reasons.pop(0)
            if len(self.finish_reasons) > 1
            else (self.finish_reasons[0] if self.finish_reasons else None)
        )
        return LLMResponse(
            message={"role": "assistant", "content": content},
            usage={"prompt_tokens": 123},
            finish_reason=finish,
        )


class _ToolCallSummaryLLM:
    """模拟把摘要写成工具调用的模型。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, temperature=None):
        self.calls.append({"messages": messages})
        return LLMResponse(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            usage={},
        )


def test_config_compaction_defaults_and_env(monkeypatch) -> None:
    from keeper.config import load_config

    for name in (
        "KEEPER_CONTEXT_WINDOW",
        "KEEPER_RESERVE_TOKENS",
        "KEEPER_KEEP_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config()
    assert config.context_window == 65536
    assert config.reserve_tokens == 16384
    assert config.keep_tokens == 20000

    monkeypatch.setenv("KEEPER_CONTEXT_WINDOW", "128000")
    monkeypatch.setenv("KEEPER_RESERVE_TOKENS", "8000")
    monkeypatch.setenv("KEEPER_KEEP_TOKENS", "12345")
    config = load_config()
    assert config.context_window == 128000
    assert config.reserve_tokens == 8000
    assert config.keep_tokens == 12345