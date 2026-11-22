"""文件系统工具：读、列目录、写、文本搜索。所有路径限制在项目根目录内。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Iterator

from keeper.tools.base import Tool, ToolResult, error

MAX_BYTES = 1_000_000
DEFAULT_MAX_LINES = 200
HARD_MAX_LINES = 1000

IGNORED_DIRS = {".git", ".agent", "__pycache__"}
PROTECTED_WRITE_DIRS = {".git", ".agent"}

DEFAULT_MAX_ENTRIES = 200
HARD_MAX_ENTRIES = 1000

DEFAULT_MAX_RESULTS = 100
HARD_MAX_RESULTS = 500
MAX_SEARCH_LINE_CHARS = 200


def resolve_path(root: Path, raw: str) -> Path:
    """把用户给的路径解析到项目根内，越界直接抛 ValueError。"""
    root = Path(root).resolve()
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes project root: {raw}")
    return path


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


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
            path = resolve_path(self.root, raw_path)
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
            return ToolResult(ok=True, content=f"{relative_path(self.root, path)} is empty")

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
        header = f"{relative_path(self.root, path)} (lines {start}-{end} of {total})"
        if end < total:
            header += (
                f"\n[truncated: {total - end} more lines; "
                f"continue with start_line={end + 1}]"
            )
        return ToolResult(ok=True, content=f"{header}\n{numbered}")


class ListDirTool(Tool):
    name: ClassVar[str] = "list_dir"
    description: ClassVar[str] = (
        "列出项目里一个目录的直接内容（不递归）。目录以 / 结尾；"
        f"自动跳过 {', '.join(sorted(IGNORED_DIRS))}。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对项目根目录的目录路径，默认 '.'",
            },
            "max_entries": {
                "type": "integer",
                "description": f"最多返回多少条，默认 {DEFAULT_MAX_ENTRIES}，上限 {HARD_MAX_ENTRIES}",
            },
        },
        "required": [],
    }
    danger: ClassVar[str] = "read"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        raw_path = _as_str(args.get("path")) or "."
        if not raw_path.strip():
            raw_path = "."

        try:
            path = resolve_path(self.root, raw_path)
        except ValueError as exc:
            return error(str(exc))
        if not path.exists():
            return error(f"directory not found: {raw_path}")
        if not path.is_dir():
            return error(f"'{raw_path}' is a file, not a directory")

        try:
            entries = [
                entry
                for entry in path.iterdir()
                if entry.name not in IGNORED_DIRS
            ]
        except OSError as exc:
            return error(f"cannot list '{raw_path}': {exc}")

        entries.sort(key=lambda entry: (not entry.is_dir(), entry.name.lower()))
        total = len(entries)
        max_entries = min(
            HARD_MAX_ENTRIES,
            max(1, _as_int(args.get("max_entries"), DEFAULT_MAX_ENTRIES)),
        )
        shown = entries[:max_entries]
        label = relative_path(self.root, path)
        if total == 0:
            return ToolResult(ok=True, content=f"{label} is empty")

        lines = [
            f"{entry.name}/" if entry.is_dir() else entry.name for entry in shown
        ]
        header = f"{label} ({len(shown)} of {total} entries)"
        if len(shown) < total:
            header += f"\n[truncated: {total - len(shown)} more entries]"
        return ToolResult(ok=True, content=header + "\n" + "\n".join(lines))


class WriteFileTool(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "新建或覆盖项目里的一个文本文件。路径相对于项目根目录；"
        "父目录会自动创建。覆盖已有文件前会请求用户确认（展示 diff）。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对项目根目录的文件路径"},
            "content": {"type": "string", "description": "要写入的完整文件内容"},
        },
        "required": ["path", "content"],
    }
    danger: ClassVar[str] = "write"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        raw_path = args.get("path")
        content = args.get("content")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return error("path is required")
        if not isinstance(content, str):
            return error("content is required (use an empty string to empty a file)")

        try:
            path = resolve_path(self.root, raw_path)
        except ValueError as exc:
            return error(str(exc))

        rel = path.relative_to(self.root)
        if rel.parts and rel.parts[0] in PROTECTED_WRITE_DIRS:
            return error(f"refusing to write inside '{rel.parts[0]}': {raw_path}")
        if path.exists() and path.is_dir():
            return error(f"'{raw_path}' is a directory, not a file")

        existed = path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        except OSError as exc:
            return error(f"cannot write '{raw_path}': {exc}")

        action = "overwrote" if existed else "created"
        size = len(content.encode("utf-8"))
        lines = len(content.splitlines())
        return ToolResult(
            ok=True,
            content=f"{action} {relative_path(self.root, path)} ({size} bytes, {lines} lines)",
        )


class SearchTool(Tool):
    name: ClassVar[str] = "search"
    description: ClassVar[str] = (
        "在项目里做纯文本搜索（区分大小写的子串匹配），返回 file:line: 匹配行。"
        "自动跳过二进制文件、超大文件和 "
        f"{', '.join(sorted(IGNORED_DIRS))} 目录。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "要查找的文本（不按正则解析）"},
            "path": {
                "type": "string",
                "description": "相对项目根目录的搜索起点（文件或目录），默认 '.'",
            },
            "max_results": {
                "type": "integer",
                "description": f"最多返回多少条，默认 {DEFAULT_MAX_RESULTS}，上限 {HARD_MAX_RESULTS}",
            },
        },
        "required": ["pattern"],
    }
    danger: ClassVar[str] = "read"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return error("pattern is required")

        raw_path = _as_str(args.get("path")) or "."
        if not raw_path.strip():
            raw_path = "."

        try:
            target = resolve_path(self.root, raw_path)
        except ValueError as exc:
            return error(str(exc))
        if not target.exists():
            return error(f"path not found: {raw_path}")

        max_results = min(
            HARD_MAX_RESULTS,
            max(1, _as_int(args.get("max_results"), DEFAULT_MAX_RESULTS)),
        )
        results: list[str] = []
        truncated = False

        for file_path in self._iter_files(target):
            if len(results) >= max_results:
                truncated = True
                break
            for match in self._search_file(file_path, pattern):
                results.append(match)
                if len(results) >= max_results:
                    truncated = True
                    break

        label = relative_path(self.root, target)
        if not results:
            return ToolResult(
                ok=True,
                content=f"no matches for {pattern!r} under {label}",
            )

        header = f"{len(results)} match(es) for {pattern!r} under {label}"
        if truncated:
            header += (
                f"\n[truncated: stopped at {max_results}; "
                "narrow the pattern or path for more]"
            )
        return ToolResult(ok=True, content=header + "\n" + "\n".join(results))

    # ---------- 内部 ----------

    def _iter_files(self, target: Path) -> Iterator[Path]:
        if target.is_file():
            yield target
            return
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(
                name for name in dirnames if name not in IGNORED_DIRS
            )
            for filename in sorted(filenames):
                yield Path(dirpath) / filename

    def _search_file(self, path: Path, pattern: str) -> Iterator[str]:
        try:
            if path.stat().st_size > MAX_BYTES:
                return
            data = path.read_bytes()
        except OSError:
            return
        if b"\x00" in data[:4096]:
            return

        text = data.decode("utf-8", errors="replace")
        rel = relative_path(self.root, path)
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern not in line:
                continue
            snippet = line.strip()
            if len(snippet) > MAX_SEARCH_LINE_CHARS:
                snippet = snippet[: MAX_SEARCH_LINE_CHARS - 1] + "…"
            yield f"{rel}:{lineno}: {snippet}"