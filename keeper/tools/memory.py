"""update_memory 工具：提议更新目标项目的 .agent/memory/。

工具本身不做审批；loop 的审批门会在执行前调用 memory.plan_update 生成
diff。execute 永不抛异常，失败返回 ``ERROR:`` 文本。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from keeper.memory import (
    MemoryOpError,
    apply_plan,
    plan_update,
)
from keeper.tools.base import Tool, ToolResult, error


def _label(path: Path) -> str:
    return f".agent/memory/{path.name}"


class UpdateMemoryTool(Tool):
    name: ClassVar[str] = "update_memory"
    description: ClassVar[str] = (
        "提议更新项目记忆（.agent/memory/）。一次只改一个文件："
        "decisions（决策）、lessons（教训）或 project_state（当前状态）。"
        "action='add' 追加一条 entry（不给 section 时追加到文件末尾，"
        "即落在最后一个分组下）；action='supersede' 用新条目替换旧条目，"
        "旧条目会原样移入对应的 *_ARCHIVE.md，不删除。"
        "supersede 时 section 只用于在重名旧条目中消歧，新条目永远跟随"
        "旧条目的分组；section 内找不到旧条目会报错。"
        "entry 是一行 bullet 正文（不要带 '- '）；detail 每条压成两格缩进的一行，"
        "不保留原缩进层级；主题分组用 section（不带 '## '）。"
        "INDEX.md 自动生成，不能手改。写入前会展示 diff 并请求用户确认。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "enum": ["decisions", "lessons", "project_state"],
                "description": "要更新的记忆文件",
            },
            "action": {
                "type": "string",
                "enum": ["add", "supersede"],
                "description": "add 追加；supersede 替换旧条目并归档",
            },
            "entry": {
                "type": "string",
                "description": "一行 bullet 正文；不要带开头的 '- '",
            },
            "detail": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "可选；每条写成 bullet 下的两格缩进细节；"
                    "原有缩进层级不保留，多行会被压平为逐行细节"
                ),
            },
            "section": {
                "type": "string",
                "description": (
                    "可选；'## ' 分组标题，不要带 '## '。"
                    "add 不给时追加到文件末尾；supersede 时只用于消歧"
                ),
            },
            "supersedes": {
                "type": "string",
                "description": (
                    "action=supersede 时必填：旧 bullet 原文（可带 '- '）；"
                    "必须是当前文件里唯一的一条"
                ),
            },
        },
        "required": ["file", "action", "entry"],
    }
    danger: ClassVar[str] = "write"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            plan = plan_update(self.root, args)
        except MemoryOpError as exc:
            return error(str(exc))
        except Exception as exc:  # 兜底：规划阶段的 bug 也不能弄死 loop
            return error(f"cannot plan memory update: {type(exc).__name__}: {exc}")

        if not plan.changes:
            return ToolResult(ok=True, content="memory unchanged")

        # 局部导入：切断 keeper.tidy -> keeper.tools 包初始化
        # -> keeper.tools.memory -> keeper.tidy 的循环导入。
        # 失败推迟到执行期是故意的：execute 永不抛异常。
        try:
            from keeper.tidy import TidyLock
        except Exception as exc:
            return error(
                f"cannot initialize memory lock: {type(exc).__name__}: {exc}"
            )
        lock = TidyLock(self.root)
        if not lock.acquire():
            return error(
                "memory is being tidied; retry after it finishes "
                "(the tidy pass holds the same lock)"
            )
        try:
            result = apply_plan(plan)
        except Exception as exc:  # 兜底：execute 永不抛异常
            return error(f"memory write crashed: {type(exc).__name__}: {exc}")
        finally:
            lock.release()
        if result.failed is not None:
            written = ", ".join(_label(c.path) for c in result.written) or "(none)"
            not_written = (
                ", ".join(_label(c.path) for c in result.not_written) or "(none)"
            )
            if result.failed.path.name == "INDEX.md":
                return error(
                    "memory 已更新，INDEX 刷新失败: "
                    f"{result.error}. 已写: {written}; 未写: {not_written}. "
                    "请重试 update_memory 刷新 INDEX。"
                )
            return error(
                f"memory write failed at {_label(result.failed.path)}: "
                f"{result.error}. 已写: {written}; 未写: {not_written}."
            )

        written = ", ".join(_label(c.path) for c in result.written)
        return ToolResult(ok=True, content=f"memory updated: {written}")
