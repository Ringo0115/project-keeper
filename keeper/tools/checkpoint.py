"""M5.1b：agent 主动请求记忆整理（checkpoint）。

工具只写 ``.agent/memory/.tidy-request.json`` 请求标记；不整理、不写记忆、
不拿锁。请求在下一次 CLI 边界的自动整理里被消费，且不绕过 due 闸门。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from keeper.checkpoint import request_checkpoint
from keeper.tools.base import Tool, ToolResult, error


class RequestCheckpointTool(Tool):
    name: ClassVar[str] = "request_memory_checkpoint"
    description: ClassVar[str] = (
        "请求在本次会话结束后的 CLI 边界整理项目记忆。只写一个请求标记，"
        "不会立即整理，也不会绕过 due 闸门或备份 / 守恒；reason 可选，"
        "用一句话说明为什么现在值得检查记忆。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "可选；一句话说明请求原因（会显示在整理记录里）",
            }
        },
        "required": [],
    }
    # 只写标记文件；审批门对 request_memory_checkpoint 明确放行（见 approval）。
    danger: ClassVar[str] = "write"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            request = request_checkpoint(self.root, args.get("reason"))
        except OSError as exc:
            return error(f"cannot write checkpoint request: {exc}")
        detail = f" ({request.reason})" if request.reason else ""
        return ToolResult(
            ok=True,
            content=(
                "memory checkpoint requested for the next CLI boundary"
                f"{detail}; tidy still runs only if the due gate allows it."
            ),
        )