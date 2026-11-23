"""M4 纯函数：token 估算、切点、六段摘要、配置。"""

from __future__ import annotations

import json

import pytest

from keeper.compaction import (
    SUMMARY_SECTIONS,
    SummaryError,
    build_summary_prompt,
    critical_facts,
    empty_summary_sections,
    estimate_tokens,
    find_cut_index,
    missing_critical_facts,
    missing_summary_sections,
    prepare_summary_input,
    summarize,
    validate_summary,
)
from keeper.llm import LLMResponse


def _len_estimate(messages: list[dict]) -> int:
    """测试用确定性估算：只数 content 字符数。"""
    return sum(len(message.get("content") or "") for message in messages)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str = "", call_ids: tuple[str, ...] = ()) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if call_ids:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
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


# ---------- token 估算 ----------


def test_estimate_tokens_is_deterministic_utf8_bytes_over_three() -> None:
    messages = [{"role": "user", "content": "中文 abc\n第二行"}]
    expected = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) // 3
    assert estimate_tokens(messages) == expected
    assert estimate_tokens(messages) == estimate_tokens(list(messages))


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens([]) == len("[]") // 3


# ---------- 切点 ----------


def test_find_cut_index_keeps_everything_under_budget() -> None:
    messages = [_user("hello"), _assistant("world")]
    assert find_cut_index(messages, 10_000, _len_estimate) == 0
    assert find_cut_index([], 1, _len_estimate) == 0


def test_find_cut_index_prefers_user_boundary() -> None:
    messages = [
        _user("a" * 10),
        _assistant("b" * 10),
        _user("c" * 10),
        _assistant("d" * 10),
    ]
    # 往回累加 10 + 10 + 10 在 assistant(index=1) 处达到预算；
    # 预算切点不是 user，于是前移到最近的 user(index=2)。
    cut = find_cut_index(messages, 25, _len_estimate)
    assert cut == 2
    assert messages[cut]["role"] == "user"


def test_find_cut_index_never_returns_tool_message() -> None:
    messages = [
        _user("u"),
        _assistant(call_ids=("c1", "c2")),
        _tool("c1", "x" * 50),
        _tool("c2", "y" * 50),
        _assistant("later"),
        _tool("c3", "z" * 50),
    ]
    for keep in range(1, 200):
        cut = find_cut_index(messages, keep, _len_estimate)
        assert messages[cut]["role"] != "tool", (keep, cut)


def test_cut_after_tool_group_discards_whole_group() -> None:
    messages = [
        _user("goal"),
        _assistant(call_ids=("c1",)),
        _tool("c1", "H" * 100),
        _assistant("next step"),
        _tool("c2", "small"),
    ]
    cut = find_cut_index(messages, 40, _len_estimate)
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
    cut = find_cut_index(messages, 40, _len_estimate)
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
    cut = find_cut_index(messages, 1, _len_estimate)
    assert cut == 1
    kept_ids = [
        message["tool_call_id"]
        for message in messages[cut:]
        if message["role"] == "tool"
    ]
    assert kept_ids == ["c1", "c2", "c3"]


def test_find_cut_index_drops_tail_group_when_it_exceeds_budget() -> None:
    messages = [
        _user("goal"),
        _assistant(call_ids=("c1",)),
        _tool("c1", "H" * 100),
    ]
    # 默认保留整组（宁可多保留一点）。
    assert find_cut_index(messages, 40, _len_estimate) == 1
    assert find_cut_index(
        messages, 40, _len_estimate, max_tail_tokens=200
    ) == 1
    # 整组超过压缩后的硬上限：全部交给摘要，保证有进展。
    assert find_cut_index(
        messages, 40, _len_estimate, max_tail_tokens=50
    ) == 3


# ---------- 摘要输入截断 ----------


def test_prepare_summary_input_truncates_huge_message() -> None:
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
    assert "trace line 4999" in text
    assert "truncated" in text


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


def test_summarize_rejects_missing_critical_facts() -> None:
    messages = [_tool("c1", "ERROR: boom at /proj/app/main.py")]
    llm = _FakeSummaryLLM(_valid_summary())  # 只有 /proj/keeper/loop.py

    with pytest.raises(SummaryError):
        summarize(llm, messages)
    assert missing_critical_facts(_valid_summary(), messages)


def test_summarize_accepts_preserved_critical_facts() -> None:
    messages = [_tool("c1", "ERROR: boom at /proj/app/main.py")]
    summary = _valid_summary().replace(
        "- /proj/keeper/loop.py",
        "- /proj/app/main.py\n- ERROR: boom at /proj/app/main.py",
    )
    llm = _FakeSummaryLLM(summary)

    assert summarize(llm, messages) == summary
    assert missing_critical_facts(summary, messages) == []


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

    assert summarize(llm, messages, max_input_tokens=5_000) == summary
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "initial goal" in prompt
    assert "ERROR: boom at /proj/app/main.py" in prompt
    assert "trace line 4999" in prompt
    assert "truncated" in prompt


class _FakeSummaryLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, temperature=None):
        self.calls.append(
            {"messages": messages, "tools": tools, "temperature": temperature}
        )
        return LLMResponse(
            message={"role": "assistant", "content": self.content},
            usage={"prompt_tokens": 123},
        )


def test_summarize_uses_no_tools_and_low_temperature() -> None:
    llm = _FakeSummaryLLM(_valid_summary())
    summary = summarize(llm, [_user("hello")])

    assert summary == _valid_summary()
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None
    assert llm.calls[0]["temperature"] == 0.0
    assert "hello" in llm.calls[0]["messages"][0]["content"]


def test_summarize_rejects_missing_sections() -> None:
    llm = _FakeSummaryLLM("## Goal\n只有目标")
    with pytest.raises(SummaryError):
        summarize(llm, [_user("hello")])


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