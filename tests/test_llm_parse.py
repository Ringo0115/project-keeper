"""LLM 客户端解析：finish_reason 必须保留，摘要层用它识别被截断的响应。"""

from __future__ import annotations

import httpx

from keeper.llm import LLMClient


def _response(choice: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [choice], "usage": {"total_tokens": 5}},
    )


def test_parse_keeps_finish_reason() -> None:
    parsed = LLMClient._parse(
        _response(
            {
                "message": {"role": "assistant", "content": "partial"},
                "finish_reason": "length",
            }
        )
    )

    assert parsed.finish_reason == "length"
    assert parsed.usage == {"total_tokens": 5}


def test_parse_missing_finish_reason_is_none() -> None:
    parsed = LLMClient._parse(
        _response({"message": {"role": "assistant", "content": "ok"}})
    )

    assert parsed.finish_reason is None