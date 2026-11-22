"""项目运行时记忆：目标项目的 .agent/memory/ 下的 Markdown 文件。

一条 entry = 一个顶层 ``- `` bullet；缩进行是它的细节；``## `` 是主题分组。
替换旧条目时，旧条目整块原样移入 ``*_ARCHIVE.md``，新条目下写
``supersedes: <文件里匹配到的旧 bullet 原行>``，不删除。
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MEMORY_DIR = Path(".agent") / "memory"
MAX_MEMORY_BYTES = 1_000_000
PREVIEW_DIFF_LINES = 80
ARCHIVE_SECTION = "Superseded"
INDEX_NOTICE = "<!-- 自动生成，勿手改；由 keeper update_memory 维护 -->"
MEMORY_SECTION_PREFIX = "## 项目记忆 / "

FILE_NAMES: dict[str, str] = {
    "decisions": "DECISIONS.md",
    "lessons": "LESSONS.md",
    "project_state": "PROJECT_STATE.md",
    "index": "INDEX.md",
}
ARCHIVE_NAMES: dict[str, str] = {
    "decisions": "DECISIONS_ARCHIVE.md",
    "lessons": "LESSONS_ARCHIVE.md",
    "project_state": "PROJECT_STATE_ARCHIVE.md",
}
ENTRY_FILES = ("decisions", "lessons", "project_state")
INJECTION_ORDER = ("decisions", "lessons", "index", "project_state")
FILE_NOUN = {"decisions": "decision", "lessons": "lesson", "project_state": "state"}
FILE_LABEL = {
    "decisions": "durable decisions",
    "lessons": "durable lessons",
    "project_state": "currently true project state",
}


class MemoryOpError(ValueError):
    """记忆操作不合法；调用方应把它变成 ERROR 文本，而不是抛给用户。"""


@dataclass(frozen=True)
class Entry:
    """一个顶层 bullet 及其紧随的缩进细节。"""

    bullet: str
    section: str | None
    start: int
    end: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FileChange:
    path: Path
    before: str | None
    after: str


@dataclass(frozen=True)
class MemoryPlan:
    changes: tuple[FileChange, ...]


@dataclass(frozen=True)
class ApplyResult:
    written: tuple[FileChange, ...] = ()
    failed: FileChange | None = None
    error: str = ""
    not_written: tuple[FileChange, ...] = ()


# ---------------------------------------------------------------------------
# 路径与读取
# ---------------------------------------------------------------------------


def memory_path(root: Path, name: str) -> Path:
    return Path(root).resolve() / MEMORY_DIR / name


def _read_optional(path: Path) -> str | None:
    """读取原始文本；保留 CRLF / LF 原样。文件不存在返回 None。"""
    if not path.exists():
        return None
    if not path.is_file():
        raise MemoryOpError(f"{path.name} is not a file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MemoryOpError(f"cannot stat {path.name}: {exc}") from exc
    if size > MAX_MEMORY_BYTES:
        raise MemoryOpError(
            f"{path.name} is too large ({size} bytes, limit {MAX_MEMORY_BYTES})"
        )
    try:
        # newline="" 保留 \r\n，避免 universal newlines 把 CRLF 悄悄改成 LF。
        with path.open("r", encoding="utf-8", newline="") as fh:
            return fh.read()
    except UnicodeDecodeError as exc:
        raise MemoryOpError(
            f"{path.name} is not valid UTF-8; refusing to rewrite it"
        ) from exc
    except OSError as exc:
        raise MemoryOpError(f"cannot read {path.name}: {exc}") from exc


def _detect_newline(text: str | None) -> str:
    if not text:
        return "\n"
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    counts = {"\r\n": crlf, "\n": lf, "\r": cr}
    best = max(counts, key=lambda key: counts[key])
    return best if counts[best] else "\n"


def _join(lines: Iterable[str], newline: str) -> str:
    items = list(lines)
    if not items:
        return ""
    return newline.join(items) + newline


def _rewrap(text: str, newline: str) -> str:
    if newline == "\n":
        return text
    return text.replace("\n", newline)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def parse_entries(text: str | None) -> list[Entry]:
    """解析顶层 ``- `` entry；缩进行属于上一条 entry。"""
    lines = (text or "").splitlines()
    entries: list[Entry] = []
    section: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("# "):
            section = None
        elif line.startswith("- "):
            j = i + 1
            # 空行或下一个顶层行终止 entry；只有紧随的缩进行算细节。
            while (
                j < len(lines)
                and lines[j].strip()
                and lines[j][:1] in (" ", "\t")
            ):
                j += 1
            entries.append(
                Entry(
                    bullet=line,
                    section=section,
                    start=i,
                    end=j,
                    lines=tuple(lines[i:j]),
                )
            )
            i = j
            continue
        i += 1
    return entries


def _find_section(lines: list[str], name: str) -> tuple[int, int] | None:
    matches: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if line.startswith("## ") and line[3:].strip() == name:
            j = i + 1
            while j < len(lines) and not lines[j].startswith("## "):
                j += 1
            matches.append((i, j))
    if len(matches) > 1:
        raise MemoryOpError(
            f"section '{name}' appears {len(matches)} times; "
            "rename or remove the duplicate headings first"
        )
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# 构造新内容
# ---------------------------------------------------------------------------


def _ensure_header(text: str | None, key: str, *, archive: bool = False) -> str:
    if text is not None and text.strip():
        return text
    stem = (ARCHIVE_NAMES if archive else FILE_NAMES)[key].removesuffix(".md")
    return f"# {stem}\n"


def _append_block(text: str, block: list[str], section: str | None) -> str:
    newline = _detect_newline(text)
    lines = text.splitlines()

    if section is None:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            lines.append("")
        lines.extend(block)
        return _join(lines, newline)

    bounds = _find_section(lines, section)
    if bounds is None:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            lines.append("")
        lines.extend([f"## {section}", "", *block])
        return _join(lines, newline)

    start, end = bounds
    body = lines[start + 1 : end]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()

    new_lines = lines[: start + 1]
    new_lines.append("")
    new_lines.extend(body)
    if body:
        new_lines.append("")
    new_lines.extend(block)
    after = lines[end:]
    if after:
        new_lines.append("")
        new_lines.extend(after)
    return _join(new_lines, newline)


def _insert_block_at(text: str, index: int, block: list[str]) -> str:
    """把 block 插到原来的位置；用于无 ``##`` 分组的条目。"""
    newline = _detect_newline(text)
    lines = text.splitlines()
    index = max(0, min(index, len(lines)))
    if index >= len(lines):
        while lines and not lines[-1].strip():
            lines.pop()
        index = len(lines)
    insert = list(block)
    if index > 0 and lines[index - 1].strip():
        insert.insert(0, "")
    if index < len(lines) and lines[index].strip():
        insert.append("")
    lines[index:index] = insert
    return _join(lines, newline)


def _remove_entry(text: str, entry: Entry) -> str:
    newline = _detect_newline(text)
    lines = text.splitlines()
    del lines[entry.start : entry.end]
    # 只收拾切口处的双空行，不动其他 entry。
    if (
        0 < entry.start < len(lines)
        and not lines[entry.start - 1].strip()
        and not lines[entry.start].strip()
    ):
        del lines[entry.start]
    return _join(lines, newline)


def _entries_phrase(count: int) -> str:
    return f"{count} entry" if count == 1 else f"{count} entries"


def build_index(
    live: dict[str, str | None],
    archives: dict[str, str | None],
) -> str:
    """固定格式的 INDEX：三 live + 三 archive，只数顶层 bullet。"""
    lines = ["# INDEX", "", INDEX_NOTICE, "", "## Memory files", ""]
    for key in ENTRY_FILES:
        count = len(parse_entries(live.get(key)))
        lines.append(
            f"- .agent/memory/{FILE_NAMES[key]} — "
            f"{FILE_LABEL[key]} ({_entries_phrase(count)})"
        )
    lines.extend(["", "## Archives", ""])
    for key in ENTRY_FILES:
        count = len(parse_entries(archives.get(key)))
        noun = FILE_NOUN[key]
        label = f"{count} archived {noun}" if count == 1 else f"{count} archived {noun}s"
        lines.append(f"- .agent/memory/{ARCHIVE_NAMES[key]} — {label}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 参数校验与规划
# ---------------------------------------------------------------------------


def _resolve_file(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryOpError(
            "file is required: decisions, lessons, or project_state"
        )
    key = value.strip().lower()
    if key.endswith(".md"):
        key = key[:-3]
    if key == "index":
        raise MemoryOpError(
            "INDEX.md is generated automatically; "
            "choose decisions, lessons, or project_state"
        )
    if key not in ENTRY_FILES:
        raise MemoryOpError(
            f"unknown memory file '{value}'; "
            "expected decisions, lessons, or project_state"
        )
    return key


def _clean_entry(value: Any) -> str:
    if not isinstance(value, str):
        raise MemoryOpError("entry is required and must be a string")
    text = value.strip()
    if "\n" in text or "\r" in text:
        raise MemoryOpError("entry must be a single line")
    if text.startswith("- "):
        text = text[2:].strip()
    if not text:
        raise MemoryOpError("entry must not be empty")
    return text


def _clean_details(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: list[Any] = value.splitlines()
    elif isinstance(value, list):
        raw_items = value
    else:
        raise MemoryOpError("detail must be a string or an array of strings")

    details: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            raise MemoryOpError("detail items must be strings")
        for line in item.replace("\r", "").split("\n"):
            line = line.strip()
            if line:
                details.append(line)
    return details


def _clean_section(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MemoryOpError("section must be a string")
    text = value.strip()
    if "\n" in text or "\r" in text:
        raise MemoryOpError("section must be a single line")
    text = text.lstrip("#").strip()
    return text or None


def _clean_supersedes(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryOpError(
            "supersedes is required when action='supersede': "
            "pass the old bullet line exactly as it appears"
        )
    text = value.strip()
    if "\n" in text or "\r" in text:
        raise MemoryOpError("supersedes must be a single line")
    return text


def _match_supersede(text: str, raw: str, section: str | None) -> Entry:
    wanted = raw.strip()
    if not wanted.startswith("- "):
        wanted = f"- {wanted}"
    wanted = wanted.strip()

    entries = parse_entries(text)
    matches = [entry for entry in entries if entry.bullet.strip() == wanted]

    if section is not None:
        scoped = [entry for entry in matches if entry.section == section]
        if len(scoped) == 1:
            return scoped[0]
        if len(scoped) > 1:
            raise MemoryOpError(
                f"ambiguous supersedes target in section '{section}' "
                f"({len(scoped)} matches): {raw!r}"
            )
        # section 只用于消歧；新条目永远跟随旧条目的 section。
        # 指定 section 却找不到旧条目时直接失败，不做全局回退。
        available = "\n".join(f"  {entry.bullet}" for entry in entries[:10])
        if not available:
            available = "  (none)"
        raise MemoryOpError(
            f"supersedes target not found in section '{section}': {raw!r}. "
            "section is only used for disambiguation; the new entry follows "
            "the old entry's section.\n"
            f"Live entries:\n{available}"
        )

    if not matches:
        available = "\n".join(f"  {entry.bullet}" for entry in entries[:10])
        if not available:
            available = "  (none)"
        raise MemoryOpError(
            f"supersedes target not found: {raw!r}\n"
            f"Live entries:\n{available}"
        )
    if len(matches) > 1:
        raise MemoryOpError(
            f"ambiguous supersedes target ({len(matches)} matches): {raw!r}; "
            "add one unique section or make the old bullet text unique"
        )
    return matches[0]


def plan_update(root: Path, args: dict[str, Any]) -> MemoryPlan:
    """只读规划：校验参数并算出所有将变化的文件。绝不写盘。

    已知限制：审批预览与 execute 各自重算计划，不做 TOCTOU 防护；
    本地单人 CLI 下可接受。
    """
    if not isinstance(args, dict):
        raise MemoryOpError("arguments must be a JSON object")

    key = _resolve_file(args.get("file"))
    action = args.get("action")
    if action not in ("add", "supersede"):
        raise MemoryOpError("action must be 'add' or 'supersede'")

    entry_text = _clean_entry(args.get("entry"))
    details = _clean_details(args.get("detail"))
    section = _clean_section(args.get("section"))

    raw_supersedes = args.get("supersedes")
    if action == "supersede":
        supersedes = _clean_supersedes(raw_supersedes)
    else:
        if raw_supersedes not in (None, ""):
            raise MemoryOpError("supersedes is only valid with action='supersede'")
        supersedes = None

    live: dict[str, str | None] = {
        name: _read_optional(memory_path(root, FILE_NAMES[name]))
        for name in INJECTION_ORDER
    }
    archives: dict[str, str | None] = {
        name: _read_optional(memory_path(root, ARCHIVE_NAMES[name]))
        for name in ENTRY_FILES
    }

    target_before = live[key]
    target_text = _ensure_header(target_before, key)
    new_archives = dict(archives)
    archive_before: str | None = None
    archive_after: str | None = None

    if action == "supersede":
        old = _match_supersede(target_text, supersedes or "", section)
        new_block = [
            f"- {entry_text}",
            f"  supersedes: {old.bullet}",  # 永远取文件里的原行
            *[f"  {line}" for line in details],
        ]
        # section 只用于查找消歧；落点永远跟随旧条目的分组。
        target_section = old.section
        target_text = _remove_entry(target_text, old)
        if target_section is None:
            target_after = _insert_block_at(target_text, old.start, new_block)
        else:
            target_after = _append_block(target_text, new_block, target_section)

        archive_before = archives[key]
        archive_text = _ensure_header(archive_before, key, archive=True)
        archive_after = _append_block(
            archive_text,
            list(old.lines),
            ARCHIVE_SECTION,
        )
        new_archives[key] = archive_after
    else:
        new_block = [
            f"- {entry_text}",
            *[f"  {line}" for line in details],
        ]
        target_after = _append_block(target_text, new_block, section)

    new_live = dict(live)
    new_live[key] = target_after

    index_before = live.get("index")
    index_after = build_index(new_live, new_archives)
    if index_before is not None:
        index_after = _rewrap(index_after, _detect_newline(index_before))

    changes: list[FileChange] = []

    # 顺序固定为 archive -> live -> INDEX：archive 先落盘，live 失败时
    # 最多留下两份副本，不会出现「live 已删、archive 没写」的丢失。
    if action == "supersede":
        _add_change(
            changes,
            memory_path(root, ARCHIVE_NAMES[key]),
            archive_before,
            archive_after or "",
        )
    _add_change(
        changes,
        memory_path(root, FILE_NAMES[key]),
        target_before,
        target_after,
    )
    _add_change(
        changes,
        memory_path(root, FILE_NAMES["index"]),
        index_before,
        index_after,
    )
    return MemoryPlan(tuple(changes))


def _add_change(
    changes: list[FileChange],
    path: Path,
    before: str | None,
    after: str,
) -> None:
    if before != after:
        changes.append(FileChange(path=path, before=before, after=after))


# ---------------------------------------------------------------------------
# 预览与写盘
# ---------------------------------------------------------------------------


def _label(path: Path) -> str:
    return f".agent/memory/{path.name}"


def format_plan(plan: MemoryPlan) -> str:
    """审批用的 unified diff；行数上限 80，多文件依次展示。"""
    if not plan.changes:
        return "no changes"

    chunks: list[str] = []
    for change in plan.changes:
        label = _label(change.path)
        before = change.before.splitlines() if change.before is not None else []
        after = change.after.splitlines()
        fromfile = f"a/{label}" if change.before is not None else "/dev/null"
        diff = list(
            difflib.unified_diff(
                before,
                after,
                fromfile=fromfile,
                tofile=f"b/{label}",
                lineterm="",
                n=2,
            )
        )
        if not diff:
            chunks.append(
                f"--- a/{label}\n+++ b/{label}\n"
                "(content changed; line endings only)"
            )
            continue
        if len(diff) > PREVIEW_DIFF_LINES:
            omitted = len(diff) - PREVIEW_DIFF_LINES
            diff = diff[:PREVIEW_DIFF_LINES] + [
                f"... ({omitted} more diff lines omitted)"
            ]
        chunks.append("\n".join(diff))
    return "\n\n".join(chunks) if chunks else "no changes"


def _write_change(change: FileChange) -> None:
    change.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = change.path.with_name(change.path.name + ".keeper-tmp")
    try:
        # newline=""：写入时不翻译换行，CRLF 保持 CRLF。
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(change.after)
        os.replace(tmp, change.path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def apply_plan(plan: MemoryPlan) -> ApplyResult:
    """按 plan.changes 的顺序写盘；遇错停止并报告已写 / 未写。"""
    written: list[FileChange] = []
    for index, change in enumerate(plan.changes):
        try:
            _write_change(change)
        except OSError as exc:
            return ApplyResult(
                written=tuple(written),
                failed=change,
                error=str(exc),
                not_written=tuple(plan.changes[index:]),
            )
        written.append(change)
    return ApplyResult(written=tuple(written))


# ---------------------------------------------------------------------------
# 注入
# ---------------------------------------------------------------------------


def _render_memory(root: Path) -> str:
    parts: list[str] = []
    for key in INJECTION_ORDER:
        name = FILE_NAMES[key]
        try:
            text = _read_optional(memory_path(root, name))
        except Exception as exc:  # 单个文件坏了也不能让 CLI 起不来
            parts.append(
                f"{MEMORY_SECTION_PREFIX}{name}\n\n"
                f"(memory: {name} 读取失败: {exc})"
            )
            continue
        if text and text.strip():
            # 只去掉首尾换行，保留第一行可能存在的缩进。
            body = text.strip("\r\n")
            parts.append(f"{MEMORY_SECTION_PREFIX}{name}\n\n{body}")
    if not parts:
        return ""
    return "# 项目记忆（启动时注入）\n\n" + "\n\n".join(parts) + "\n"


def render_memory(root: Path) -> str:
    """启动注入文本：DECISIONS -> LESSONS -> INDEX -> PROJECT_STATE。"""
    try:
        return _render_memory(root)
    except Exception as exc:
        return (
            "# 项目记忆（启动时注入）\n\n"
            f"(memory: 读取失败: {type(exc).__name__}: {exc})\n"
        )
