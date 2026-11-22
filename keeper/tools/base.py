"""工具协议。

核心约定：execute 永不抛异常。失败也返回 ToolResult，错误文本以 ERROR: 开头，
因为读它的是模型，不是 Python 的 except。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class ToolResult:
    ok: bool
    content: str


def error(message: str) -> ToolResult:
    return ToolResult(ok=False, content=f"ERROR: {message}")


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    danger: ClassVar[str] = "read"  # read | write | exec | delete，M1 审批用

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> ToolResult:
        """执行工具。不要抛异常。"""

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }