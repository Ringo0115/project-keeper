"""文件系统工具。M0 只读，路径限制在项目根目录内。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from keeper.tools.base import Tool, ToolResult, error

MAX_BYTES = 1_000_000
DEFAULT_MAX_LINES = 200
HARD_MAX_LINES = 1000


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ReadFileTool(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "读取项目里的一个文本文件，返回带行号的内容。"
        "路径相对于项目根目录；大文件用 start_line / max_lines 分段读。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对项目根目录的文件路径"},
            "start_line": {
                "type": "integer",
                "description": "从第几行开始读（1 起），默认 1",
            },
            "max_lines": {
                "type": "integer",
                "description": f"最多返回多少行，默认 {DEFAULT_MAX_LINES}，上限 {HARD_MAX_LINES}",
            },
        },
        "required": ["path"],
    }
    danger: ClassVar[str] = "read"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return error("path is required")

        try:
            path = self._resolve(raw_path)
        except ValueError as exc:
            return error(str(exc))

        if not path.exists():
            return error(f"file not found: {raw_path}")
        if path.is_dir():
            return error(f"'{raw_path}' is a directory, not a file")

        try:
            size = path.stat().st_size
        except OSError as exc:
            return error(f"cannot stat '{raw_path}': {exc}")
        if size > MAX_BYTES:
            return error(
                f"file too large ({size} bytes, limit {MAX_BYTES}): {raw_path}"
            )

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return error(f"cannot read '{raw_path}': {exc}")
        if "\x00" in text[:4096]:
            return error(f"'{raw_path}' looks like a binary file")

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return ToolResult(ok=True, content=f"{self._rel(path)} is empty")

        start = max(1, _as_int(args.get("start_line"), 1))
        max_lines = min(
            HARD_MAX_LINES,
            max(1, _as_int(args.get("max_lines"), DEFAULT_MAX_LINES)),
        )
        start = min(start, total)
        end = min(total, start + max_lines - 1)

        chunk = lines[start - 1 : end]
        numbered = "\n".join(
            f"{start + offset:>5}\t{line}" for offset, line in enumerate(chunk)
        )
        header = f"{self._rel(path)} (lines {start}-{end} of {total})"
        if end < total:
            header += (
                f"\n[truncated: {total - end} more lines; "
                f"continue with start_line={end + 1}]"
            )
        return ToolResult(ok=True, content=f"{header}\n{numbered}")

    # ---------- 内部 ----------

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        path = candidate if candidate.is_absolute() else self.root / candidate
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"path escapes project root: {raw}")
        return path

    def _rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()