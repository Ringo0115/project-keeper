"""配置加载：环境变量 + .env。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_STEPS = 30
DEFAULT_APPROVAL_TIMEOUT = 60.0
DEFAULT_CONTEXT_WINDOW = 65536
DEFAULT_RESERVE_TOKENS = 16384
DEFAULT_KEEP_TOKENS = 20000


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float = 0.2
    approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT
    context_window: int = DEFAULT_CONTEXT_WINDOW
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_tokens: int = DEFAULT_KEEP_TOKENS
    memory_auto: bool = True


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"0", "false", "off", "no"}:
        return False
    if value in {"1", "true", "on", "yes"}:
        return True
    return default


def load_config(*, model: str | None = None, max_steps: int | None = None) -> Config:
    """优先级：显式参数 > 环境变量 > 默认值。"""
    load_dotenv()
    api_key = os.getenv("KEEPER_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
    base_url = (os.getenv("KEEPER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_name = model or os.getenv("KEEPER_MODEL") or DEFAULT_MODEL
    steps = max_steps if max_steps is not None else _int_env("KEEPER_MAX_STEPS", DEFAULT_MAX_STEPS)
    approval_timeout = _float_env(
        "KEEPER_APPROVAL_TIMEOUT", DEFAULT_APPROVAL_TIMEOUT
    )
    context_window = _int_env("KEEPER_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW)
    reserve_tokens = _int_env("KEEPER_RESERVE_TOKENS", DEFAULT_RESERVE_TOKENS)
    keep_tokens = _int_env("KEEPER_KEEP_TOKENS", DEFAULT_KEEP_TOKENS)
    memory_auto = _bool_env("KEEPER_MEMORY_AUTO", True)
    return Config(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        max_steps=max(steps, 1),
        approval_timeout=max(approval_timeout, 0.0),
        context_window=max(context_window, 1),
        reserve_tokens=max(reserve_tokens, 0),
        keep_tokens=max(keep_tokens, 1),
        memory_auto=memory_auto,
    )
