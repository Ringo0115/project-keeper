"""整理员一轮专用的记忆工具：只读写 ``.agent/memory/`` 下的 Markdown。

和用户会话的 update_memory 不同，这里的写入不弹终端审批；由
``MemoryTidyApprover`` 做只对 memory 目录生效的独立判定。工具自身再做
一次路径校验，避免审批器被绕过。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from keeper.events import ApprovalRequest
from keeper.memory import (
    FILE_NAMES,
    MAX_MEMORY_BYTES,
    MEMORY_DIR,
    atomic_write,
    detect_newline,
    rewrap,
)
from keeper.tools.base import Tool, ToolResult, error


def resolve_memory_path(
    root: Path, raw: Any, *, allow_index: bool = False
) -> Path:
    """把模型给的路径限制到 ``.agent/memory/`` 的平铺 Markdown 文件。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path is required")
    raw = raw.strip()
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("absolute paths are not allowed")

    root = Path(root).resolve()
    memory_dir = (root / MEMORY_DIR).resolve()
    if candidate.parent == Path("."):
        path = (memory_dir / candidate.name).resolve()
    else:
        path = (root / candidate).resolve()

    if not path.is_relative_to(memory_dir):
        raise ValueError("path must stay inside .agent/memory/")
    if path.parent != memory_dir:
        raise ValueError("memory files are flat; no subdirectories")
    if path.suffix.lower() != ".md":
        raise ValueError("only .md memory files are allowed")
    if not allow_index and path.name == FILE_NAMES["index"]:
        raise ValueError("INDEX.md is generated automatically; do not write it")
    return path


class MemoryReadTool(Tool):
    name: ClassVar[str] = "memory_read"
    description: ClassVar[str] = (
        "读取 .agent/memory/ 下的一个 Markdown 文件（live 或 *_ARCHIVE.md）。"
        "path 可以是 DECISIONS.md 这样的文件名，或 .agent/memory/DECISIONS.md。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "记忆文件名，例如 DECISIONS_ARCHIVE.md",
            }
        },
        "required": ["path"],
    }
    danger: ClassVar[str] = "read"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            path = resolve_memory_path(self.root, args.get("path"), allow_index=True)
        except ValueError as exc:
            return error(str(exc))
        try:
            if not path.is_file():
                return error(f"memory file not found: {path.name}")
            if path.stat().st_size > MAX_MEMORY_BYTES:
                return error(f"{path.name} is too large to read")
            with path.open("r", encoding="utf-8", newline="") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            return error(f"cannot read {path.name}: {exc}")
        return ToolResult(ok=True, content=text)


class MemoryWriteTool(Tool):
    name: ClassVar[str] = "memory_write"
    description: ClassVar[str] = (
        "用完整内容覆盖 .agent/memory/ 下的一个 Markdown 文件。"
        "INDEX.md 自动生成，不能写；不要删除条目，只把 live 条目整块搬到 archive。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "记忆文件名，例如 DECISIONS.md",
            },
            "content": {
                "type": "string",
                "description": "文件的完整新内容（不是 diff）",
            },
        },
        "required": ["path", "content"],
    }
    danger: ClassVar[str] = "write"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            path = resolve_memory_path(self.root, args.get("path"))
        except ValueError as exc:
            return error(str(exc))

        content = args.get("content")
        if not isinstance(content, str):
            return error("content is required and must be a string")

        try:
            if path.is_file():
                with path.open("r", encoding="utf-8", newline="") as fh:
                    existing = fh.read()
            else:
                existing = None
        except (OSError, UnicodeDecodeError) as exc:
            return error(f"cannot read existing {path.name}: {exc}")

        text = content.replace("\r\n", "\n").replace("\r", "\n")
        if text and not text.endswith("\n"):
            text += "\n"
        newline = detect_newline(existing) if existing is not None else "\n"
        text = rewrap(text, newline)

        size = len(text.encode("utf-8"))
        if size > MAX_MEMORY_BYTES:
            return error(
                f"{path.name} would be {size} bytes; limit is {MAX_MEMORY_BYTES}"
            )
        try:
            atomic_write(path, text)
        except OSError as exc:
            return error(f"cannot write {path.name}: {exc}")
        return ToolResult(ok=True, content=f"wrote .agent/memory/{path.name} ({size} bytes)")


class MemoryTidyApprover:
    """独立审批器：只放行 ``.agent/memory/`` 内的 Markdown 写入。

    不读 stdin、不问用户、不弹 ``[y/N]``；自动整理的安全边界是备份 +
    id 守恒 + run record，不是逐条人审。
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def confirm(self, request: ApprovalRequest) -> bool:
        if request.tool != "memory_write":
            return False
        try:
            resolve_memory_path(self.root, request.detail)
        except (ValueError, OSError):
            return False
        return True