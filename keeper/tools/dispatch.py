"""Manager-only worker dispatch tools (never included in worker registries)."""

from __future__ import annotations

from typing import Any, ClassVar

from keeper.context import WorkerBrief
from keeper.dispatch import CompletionRecord, dispatch_fanout, dispatch_one
from keeper.projects import Project
from keeper.tools.base import Tool, ToolResult, error
from keeper.workers import default_workers


def _record_dict(record: CompletionRecord) -> dict[str, str | None]:
    return {"worker": record.worker, "result": record.result, "summary": record.summary,
            "reply": record.reply, "transcript": record.transcript,
            "started_at": record.started_at, "ended_at": record.ended_at}


class DispatchTaskTool(Tool):
    name: ClassVar[str] = "dispatch_task"
    description: ClassVar[str] = "审批后将一个自包含 brief 派发给指定 worker。"
    # Detailed fail-closed approval happens inside dispatch_one exactly once.
    danger: ClassVar[str] = "read"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "worker": {"type": "string"},
            "label": {"type": "string"},
            "task": {"type": "string"},
            "allowed": {"type": "array", "items": {"type": "string"}},
            "forbidden": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["worker", "label", "task"],
    }

    def __init__(self, project: Project, approver: Any, *, llm: Any = None,
                 max_steps: int = 30, timeout: float = 1800) -> None:
        self.project, self.approver, self.llm = project, approver, llm
        self.max_steps, self.timeout = max_steps, timeout

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            brief = WorkerBrief(args["label"], args["task"], tuple(args.get("allowed", ())),
                                tuple(args.get("forbidden", ())))
            result = dispatch_one(self.project, brief, args["worker"], self.approver,
                                  llm=self.llm, max_steps=self.max_steps, timeout=self.timeout)
            return ToolResult(True, __import__("json").dumps(_record_dict(result), ensure_ascii=False))
        except Exception as exc:
            return error(f"dispatch failed: {type(exc).__name__}: {exc}")


class DispatchFanoutTool(Tool):
    name: ClassVar[str] = "dispatch_fanout"
    description: ClassVar[str] = "一次审批后并行派发 2–5 个互不依赖子任务，全部完成后返回记录。"
    # Detailed batch approval happens inside dispatch_fanout exactly once.
    danger: ClassVar[str] = "read"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "worker": {"type": "string"},
                        "label": {"type": "string"},
                        "task": {"type": "string"},
                        "allowed": {"type": "array", "items": {"type": "string"}},
                        "forbidden": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["worker", "label", "task"],
                },
            },
        },
        "required": ["tasks"],
    }

    def __init__(self, project: Project, approver: Any, *, llm: Any = None,
                 max_steps: int = 30, timeout: float = 1800) -> None:
        self.project, self.approver, self.llm = project, approver, llm
        self.max_steps, self.timeout = max_steps, timeout

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            items = args.get("tasks")
            if not isinstance(items, list): return error("tasks must be an array")
            briefs = [WorkerBrief(item["label"], item["task"], tuple(item.get("allowed", ())),
                                  tuple(item.get("forbidden", ()))) for item in items]
            workers = [item["worker"] for item in items]
            records = dispatch_fanout(self.project, briefs, workers, self.approver,
                                      llm=self.llm, max_steps=self.max_steps, timeout=self.timeout)
            return ToolResult(True, __import__("json").dumps([_record_dict(r) for r in records], ensure_ascii=False))
        except Exception as exc:
            return error(f"fanout failed: {type(exc).__name__}: {exc}")


def manager_dispatch_tools(project: Project, approver: Any, *, llm: Any = None,
                           max_steps: int = 30, timeout: float = 1800) -> tuple[Tool, Tool]:
    """Factory for manager registry only; workers receive separate restricted registries."""
    return (DispatchTaskTool(project, approver, llm=llm, max_steps=max_steps, timeout=timeout),
            DispatchFanoutTool(project, approver, llm=llm, max_steps=max_steps, timeout=timeout))
