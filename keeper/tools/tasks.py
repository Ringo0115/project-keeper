"""complete_task / block_task：任务簿记工具。

两个工具在构造时绑定本次 task_id，模型不能传任意 id；它们只在任务运行期间
注册进 Registry。execute 永不抛异常，失败返回 ``ERROR:`` 文本。

这两个工具不走审批：它们只做任务状态转换（终态），不写文件内容；
审批策略在 approval.review() 里显式 ALLOW。
"""

from __future__ import annotations

from typing import Any, ClassVar

from keeper.tasks import TaskStore, TaskStoreError
from keeper.tools.base import Tool, ToolResult, error


class CompleteTaskTool(Tool):
    name: ClassVar[str] = "complete_task"
    description: ClassVar[str] = (
        "把当前任务标记为 done。只有确认任务真正完成时才调用；"
        "纯文本回答“做完了”不算完成。不需要参数。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
    }
    danger: ClassVar[str] = "task"

    def __init__(self, store: TaskStore, task_id: str, *, transcript: str | None = None) -> None:
        self.store = store
        self.task_id = task_id
        self.transcript = transcript

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            self.store.complete(self.task_id, transcript=self.transcript)
        except TaskStoreError as exc:
            return error(str(exc))
        except Exception as exc:  # 兜底：工具 bug 不能弄死 loop
            return error(f"cannot complete task: {type(exc).__name__}: {exc}")
        return ToolResult(ok=True, content=f"task {self.task_id} marked done")


class BlockTaskTool(Tool):
    name: ClassVar[str] = "block_task"
    description: ClassVar[str] = (
        "把当前任务标记为 blocked，并把具体原因写入任务记录。"
        "无法完成任务时必须调用；reason 不能为空，需要写清卡在哪里、需要人处理什么。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "为什么无法完成；必须具体，不能为空",
            }
        },
        "required": ["reason"],
    }
    danger: ClassVar[str] = "task"

    def __init__(self, store: TaskStore, task_id: str, *, transcript: str | None = None) -> None:
        self.store = store
        self.task_id = task_id
        self.transcript = transcript

    def execute(self, args: dict[str, Any]) -> ToolResult:
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return error("block_task 需要非空 reason；任务状态未改变")
        try:
            self.store.block(self.task_id, reason, transcript=self.transcript)
        except TaskStoreError as exc:
            return error(str(exc))
        except Exception as exc:  # 兜底：工具 bug 不能弄死 loop
            return error(f"cannot block task: {type(exc).__name__}: {exc}")
        return ToolResult(
            ok=True,
            content=f"task {self.task_id} marked blocked: {reason.strip()}",
        )