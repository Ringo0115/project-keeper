"""OpenAI 兼容的模型客户端（DeepSeek 等）。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

RETRYABLE_STATUS = {408, 409, 429}


class LLMError(RuntimeError):
    """模型调用失败。"""


@dataclass
class LLMResponse:
    message: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._temperature = temperature
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(self._url, json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    return self._parse(response)

                body = response.text[:800]
                last_error = LLMError(f"HTTP {response.status_code}: {body}")
                if (
                    response.status_code not in RETRYABLE_STATUS
                    and response.status_code < 500
                ):
                    raise last_error

            if attempt < 2:
                time.sleep(2**attempt)

        raise LLMError(f"request failed after retries: {last_error}")

    @staticmethod
    def _parse(response: httpx.Response) -> LLMResponse:
        try:
            data = response.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response: {response.text[:500]}") from exc
        return LLMResponse(
            message=_normalize_message(message),
            usage=data.get("usage") or {},
        )


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """只保留 OpenAI 标准字段，丢掉 reasoning_content 之类的扩展。"""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    normalized: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        normalized["tool_calls"] = [
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": (call.get("function") or {}).get("name", ""),
                    "arguments": (call.get("function") or {}).get("arguments") or "{}",
                },
            }
            for index, call in enumerate(tool_calls)
        ]
    return normalized