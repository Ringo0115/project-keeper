"""工具注册表：给模型 schema，按名字分发调用。"""

from __future__ import annotations

from typing import Any, Iterable

from keeper.tools.base import Tool, ToolResult, error


class Registry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if not self._tools:
            raise ValueError("registry needs at least one tool")

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return error(f"unknown tool '{name}'. Available: {', '.join(self.names())}")
        if not isinstance(args, dict):
            return error("tool arguments must be a JSON object")
        try:
            result = tool.execute(args)
        except Exception as exc:  # 兜底：工具自己的 bug 不能弄死 agent
            return error(f"tool '{name}' crashed: {type(exc).__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return error(f"tool '{name}' returned {type(result).__name__}, expected ToolResult")
        return result