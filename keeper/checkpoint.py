"""M5.1b：agent 主动 checkpoint 的请求标记。

会话中的 agent 只能“请求”整理，不能直接跑整理：工具把请求写到
``.agent/memory/.tidy-request.json``，真正整理仍在 CLI 边界、仍按 M5.0 的
due 闸门与单飞锁执行。标记不写记忆文件、不拿锁、不调用 LLM。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from keeper.memory import atomic_write, memory_path

CHECKPOINT_NAME = ".tidy-request.json"
CHECKPOINT_REASON_MAX = 200


@dataclass(frozen=True)
class CheckpointRequest:
    requested_at: str = ""
    reason: str = ""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean_reason(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    # 单行、限长：reason 会进整理 prompt 与 run record，保持为数据。
    text = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    return text[:CHECKPOINT_REASON_MAX]


def request_checkpoint(root: Path, reason: Any = "") -> CheckpointRequest:
    """写入（覆盖）请求标记；返回实际落盘的请求。只写标记文件。"""
    request = CheckpointRequest(
        requested_at=_now_iso(), reason=_clean_reason(reason)
    )
    atomic_write(
        memory_path(root, CHECKPOINT_NAME),
        json.dumps(
            {
                "version": 1,
                "requested_at": request.requested_at,
                "reason": request.reason,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    return request


def read_checkpoint_request(root: Path) -> CheckpointRequest | None:
    """读取请求标记；不存在、坏 JSON、字段类型不对都按没有请求处理。"""
    path = memory_path(root, CHECKPOINT_NAME)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    requested_at = data.get("requested_at")
    reason = data.get("reason")
    return CheckpointRequest(
        requested_at=requested_at if isinstance(requested_at, str) else "",
        reason=_clean_reason(reason),
    )


def clear_checkpoint_request(root: Path) -> bool:
    """删除请求标记；成功（或本来不存在）返回 True。"""
    try:
        memory_path(root, CHECKPOINT_NAME).unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def consume_checkpoint_request(
    root: Path, request: CheckpointRequest
) -> bool:
    """只在标记仍是这次读到的那个时才删除。

    整理 pass 可能运行几分钟；期间如果有新的 checkpoint 写入，不能把新
    请求一起删掉。这里先重读比较（时间 + reason），不同就保留。跨进程下
    重读与 unlink 仍不是原子操作，因此只承诺“尽力不丢”，不承诺严格互斥。
    """
    current = read_checkpoint_request(root)
    if current != request:
        return False
    return clear_checkpoint_request(root)