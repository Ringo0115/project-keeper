"""项目运行时记忆：目标项目的 .agent/memory/ 下的 Markdown 文件。

一条 entry = 一个顶层 ``- `` bullet；缩进行是它的细节；``## `` 是主题分组。
替换旧条目时，旧条目整块原样移入 ``*_ARCHIVE.md``，新条目下写
``supersedes: <文件里匹配到的旧 bullet 原行>``，不删除。
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

MEMORY_DIR = Path(".agent") / "memory"
MAX_MEMORY_BYTES = 1_000_000
PREVIEW_DIFF_LINES = 80
ARCHIVE_SECTION = "Superseded"
INDEX_NOTICE = "<!-- 自动生成，勿手改；由 keeper update_memory 维护 -->"
MEMORY_SECTION_PREFIX = "## 项目记忆 / "

# 每条 entry 的 id 戳。单独一行缩进注释，渲染 Markdown 时不可见；
# 注入前由 inject_view 剥离，不占预算。
MEM_COMMENT_RE = re.compile(r"<!--mem\s+(?P<body>.*?)-->")
MEM_COMMENT_LINE_RE = re.compile(r"^\s*<!--mem\s+.*?-->\s*$")
MEM_FIELD_ORDER = ("id", "created", "touched", "pinned")

# pinned 是代码级保护（M5.1b），不是提示词建议：带这个字段的条目在整理
# 后被改 / 移 / 删 / 转档，postpass 会从备份回滚。手工写 `pinned:1` 即可；
# 没有该字段的旧记录一律按未 pinned 处理。
PINNED_FIELD = "pinned"
PINNED_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "pinned"})

# 预算口径：UTF-8 字节 // 3。计入 bullet 与缩进细节，不含 mem 注释和
# archive；INDEX 是导航文件，单独给一档。
FILE_BUDGETS: dict[str, dict[str, int]] = {
    "decisions": {"soft": 3000, "hard": 4500},
    "lessons": {"soft": 2500, "hard": 4000},
    "project_state": {"soft": 1200, "hard": 1800},
    "index": {"soft": 600, "hard": 900},
}

# 预算按 worker 的上下文窗口缩放（M5.1b）：默认窗口下就是上表；更小的窗口
# 收紧预算，更大的窗口最多放大 2 倍。只缩放 soft / hard，不改预算口径。
MEMORY_BUDGET_WINDOW = 65536
BUDGET_SCALE_MIN = 0.5
BUDGET_SCALE_MAX = 2.0

_LEAD_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?[:：]?\s*")

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
    """一个顶层 bullet 及其紧随的缩进细节。

    ``mem`` 是紧贴 bullet 的 ``<!--mem …-->`` 字段；``mem_index`` 是它在
    ``lines`` 里的下标（没有注释时为 None）。``details`` 不包含注释行。
    """

    bullet: str
    section: str | None
    start: int
    end: int
    lines: tuple[str, ...]
    mem: dict[str, str] | None = None
    mem_index: int | None = None

    @property
    def details(self) -> tuple[str, ...]:
        if self.mem_index is None:
            return self.lines[1:]
        return tuple(
            line
            for index, line in enumerate(self.lines)
            if index > 0 and index != self.mem_index
        )


def is_pinned(entry: Entry) -> bool:
    """``<!--mem ... pinned:1-->`` 才是 pinned；缺字段按未 pinned。"""
    if not entry.mem:
        return False
    value = entry.mem.get(PINNED_FIELD, "")
    return value.strip().lower() in PINNED_TRUE_VALUES


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


def read_memory_text(root: Path, name: str) -> str | None:
    """读取 .agent/memory/ 下的一个文件；不存在返回 None。"""
    return _read_optional(memory_path(root, name))


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


def _mem_fields(line: str) -> dict[str, str] | None:
    """解析 ``<!--mem id:… created:… touched:…-->``；没有返回 None。"""
    match = MEM_COMMENT_RE.search(line)
    if not match:
        return None
    fields: dict[str, str] = {}
    for part in match.group("body").split():
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        if key and value:
            fields[key] = value
    return fields or None


def _find_mem_comment(lines: tuple[str, ...]) -> tuple[dict[str, str] | None, int | None]:
    for index, line in enumerate(lines):
        fields = _mem_fields(line)
        if fields is not None and MEM_COMMENT_LINE_RE.match(line):
            return fields, index
    return None, None


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
            block = tuple(lines[i:j])
            mem, mem_index = _find_mem_comment(block)
            entries.append(
                Entry(
                    bullet=line,
                    section=section,
                    start=i,
                    end=j,
                    lines=block,
                    mem=mem,
                    mem_index=mem_index,
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


# ---------------------------------------------------------------------------
# id 戳与 entry 守恒
# ---------------------------------------------------------------------------


def _normalise_title(title: str) -> str:
    text = " ".join((title or "").replace("**", "").split())
    return _LEAD_DATE_RE.sub("", text).strip().lower()


def _entry_title(entry: Entry) -> str:
    text = entry.bullet[2:] if entry.bullet.startswith("- ") else entry.bullet
    return _normalise_title(text)


def _entry_content(entry: Entry) -> str:
    parts = [
        entry.bullet[2:] if entry.bullet.startswith("- ") else entry.bullet,
        *entry.details,
    ]
    return " ".join(" ".join(parts).replace("**", "").split())


def _bullet_key(entry: Entry) -> str:
    text = entry.bullet[2:] if entry.bullet.startswith("- ") else entry.bullet
    return " ".join(text.replace("**", "").split())


def new_entry_id(title: str, taken: set[str]) -> str:
    """内容派生 id：纯 ASCII 用 slug，非 ASCII 用标题 sha1 前 6 位。"""
    norm = _normalise_title(title)
    slug = re.sub(r"[^a-z0-9]+", "-", norm).strip("-")[:32].strip("-")
    if any(ord(char) > 127 for char in norm) or not slug:
        base = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:6]
    else:
        base = slug
    entry_id = base
    suffix = 2
    while entry_id in taken:
        entry_id = f"{base}-{suffix}"
        suffix += 1
    return entry_id


def entry_ids(text: str | None) -> set[str]:
    """文件里所有 mem 注释携带的 id。"""
    return {
        entry.mem["id"]
        for entry in parse_entries(text)
        if entry.mem and entry.mem.get("id")
    }


def _comment_indent(entry: Entry) -> str:
    if entry.mem_index is not None:
        line = entry.lines[entry.mem_index]
        return line[: len(line) - len(line.lstrip())] or "  "
    for line in entry.details:
        if line[:1] in (" ", "\t"):
            return line[: len(line) - len(line.lstrip())] or "  "
    return "  "


def _format_mem_comment(fields: dict[str, str], indent: str) -> str:
    ordered = [key for key in MEM_FIELD_ORDER if key in fields]
    ordered.extend(key for key in fields if key not in MEM_FIELD_ORDER)
    body = " ".join(f"{key}:{fields[key]}" for key in ordered)
    return f"{indent}<!--mem {body}-->"


def stamp_entries(
    new_text: str,
    old_text: str | Iterable[str] | None,
    key: str,
    *,
    today: str | None = None,
    archive_ids: set[str] | None = None,
) -> tuple[str, list[str]]:
    """给 entry 补 id / created / touched；已有 id 不重写。

    ``old_text`` 可以是多份旧文本（live + archive），用于把整理后丢失
    注释的条目按 bullet 原文重新绑定 id。内容变化只更新 ``touched``。
    """
    if key not in ENTRY_FILES or not new_text:
        return new_text, []
    today = today or date.today().isoformat()
    warnings: list[str] = []

    if old_text is None:
        old_texts: list[str] = []
    elif isinstance(old_text, str):
        old_texts = [old_text]
    else:
        old_texts = [text for text in old_text if text]

    old_by_id: dict[str, Entry] = {}
    old_by_key: dict[str, list[Entry]] = {}
    for text in old_texts:
        for entry in parse_entries(text):
            if entry.mem and entry.mem.get("id"):
                old_by_id.setdefault(entry.mem["id"], entry)
            old_by_key.setdefault(_bullet_key(entry), []).append(entry)

    lines = new_text.splitlines()
    entries = parse_entries(new_text)
    by_start = {entry.start: entry for entry in entries}
    taken: set[str] = set(archive_ids or ())
    taken.update(old_by_id)
    taken.update(
        entry.mem["id"]
        for entry in entries
        if entry.mem and entry.mem.get("id")
    )
    used: set[str] = set()
    claimed: set[int] = set()
    rendered: dict[int, list[str]] = {}

    for entry in entries:
        fields: dict[str, str] | None = None
        own_id = entry.mem.get("id") if entry.mem else None
        own_old = old_by_id.get(own_id) if own_id else None
        duplicate = bool(own_id and own_id in used)

        if duplicate:
            warnings.append(
                f"entry id:{own_id} appeared twice; the second got a fresh id"
            )
            own_old = None
        if own_old is not None and id(own_old) not in claimed:
            fields = {**own_old.mem, **(entry.mem or {})}
            claimed.add(id(own_old))
            if _entry_content(entry) != _entry_content(own_old):
                fields["touched"] = today
        elif own_id and not duplicate:
            fields = dict(entry.mem or {})
        else:
            for candidate in old_by_key.get(_bullet_key(entry), ()):
                candidate_id = candidate.mem.get("id") if candidate.mem else None
                if id(candidate) in claimed or (
                    candidate_id and candidate_id in used
                ):
                    continue
                # 手写的 pinned 条目可能还没有 id；保留它的 pinned 字段，
                # 下面再补一个新 id。普通的无 id 条目不用这个回退。
                if candidate_id is None and not (
                    candidate.mem or {}
                ).get(PINNED_FIELD):
                    continue
                fields = dict(candidate.mem or {})
                claimed.add(id(candidate))
                if _entry_content(entry) != _entry_content(candidate):
                    fields["touched"] = today
                break

        if fields is None:
            # 没有 id 的条目（例如手写 pinned）保留自己的字段；已有 id 但
            # 找不到旧条目时视为重复 / 失配，必须换新 id。
            fields = dict(entry.mem or {}) if not own_id else {}
        if "id" not in fields:
            fields["id"] = new_entry_id(_entry_title(entry), taken | used)
        fields.setdefault("created", today)
        fields.setdefault("touched", fields["created"])
        used.add(fields["id"])

        block = list(entry.lines)
        if entry.mem_index is not None:
            block[entry.mem_index] = _format_mem_comment(
                fields, _comment_indent(entry)
            )
        else:
            block.insert(1, _format_mem_comment(fields, _comment_indent(entry)))
        rendered[entry.start] = block

    out: list[str] = []
    index = 0
    while index < len(lines):
        entry = by_start.get(index)
        if entry is None:
            out.append(lines[index])
            index += 1
            continue
        out.extend(rendered[entry.start])
        index = entry.end
    result = _join(out, _detect_newline(new_text))
    # 原文件没有终止换行时保持原样；否则 only-touched 的 stamp 会给未编辑
    # 文件凭空加一个换行（M5.1b 的 pinned 严格字节回归会因此失败）。
    if new_text and not new_text.endswith(("\n", "\r")):
        result = result.rstrip("\r\n")
    return result, warnings


# ---------------------------------------------------------------------------
# 预算与注入视图（纯函数，M6 上下文包复用）
# ---------------------------------------------------------------------------


def strip_mem_comments(text: str | None) -> str:
    """去掉 mem 注释；CRLF 统一为 LF。"""
    if not text:
        return text or ""
    out: list[str] = []
    for line in text.splitlines():
        if MEM_COMMENT_LINE_RE.match(line):
            continue
        if "<!--mem" in line:
            line = MEM_COMMENT_RE.sub("", line).rstrip()
        out.append(line)
    result = "\n".join(out)
    if text.endswith(("\n", "\r")) and not result.endswith("\n"):
        result += "\n"
    return result


def _tokens(text: str) -> int:
    return len(text.encode("utf-8")) // 3


def memory_tokens(text: str | None) -> int:
    """预算口径：剥离 mem 注释后的 UTF-8 字节数 // 3。"""
    return _tokens(strip_mem_comments(text))


def budget_scale(context_window: int | None) -> float:
    """窗口相对默认窗口的缩放系数；None / 非法值按默认窗口（1.0）。"""
    try:
        window = int(context_window) if context_window is not None else 0
    except (TypeError, ValueError):
        return 1.0
    if window <= 0:
        return 1.0
    return min(
        max(window / MEMORY_BUDGET_WINDOW, BUDGET_SCALE_MIN),
        BUDGET_SCALE_MAX,
    )


def effective_budgets(
    context_window: int | None = None,
) -> dict[str, dict[str, int]]:
    """按 worker 窗口缩放后的 soft / hard 预算；default 窗口等于原表。"""
    scale = budget_scale(context_window)
    out: dict[str, dict[str, int]] = {}
    for key, budget in FILE_BUDGETS.items():
        if scale == 1.0:
            out[key] = dict(budget)
            continue
        soft = max(1, round(budget["soft"] * scale))
        hard = max(soft + 1, round(budget["hard"] * scale))
        out[key] = {"soft": soft, "hard": hard}
    return out


def file_budget(
    key: str, context_window: int | None = None
) -> dict[str, int] | None:
    return effective_budgets(context_window).get(key)


def over_soft_budget(
    text: str | None, key: str, context_window: int | None = None
) -> bool:
    budgets = file_budget(key, context_window)
    return bool(budgets) and memory_tokens(text) > budgets["soft"]


def over_hard_budget(
    text: str | None, key: str, context_window: int | None = None
) -> bool:
    budgets = file_budget(key, context_window)
    return bool(budgets) and memory_tokens(text) > budgets["hard"]


def _counter_note(key: str, count: int) -> str:
    name = FILE_NAMES.get(key, "the memory file")
    return (
        f"- [... {count} more entries omitted — read .agent/memory/{name} "
        "for the full text ...]"
    )


def _head_within(text: str, hard_budget: int) -> str:
    note = (
        "\n[... content trimmed from this view — read the file on disk "
        "for the full text ...]"
    )
    if _tokens(text) <= hard_budget:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        candidate = "\n".join([*kept, line]) + note
        if _tokens(candidate) > hard_budget:
            break
        kept.append(line)
    if not kept:
        return note.lstrip("\n")
    return "\n".join(kept) + note


def _compact_view(text: str) -> str:
    """只保留 bullet 行，去掉缩进细节。"""
    lines = text.splitlines()
    entries = parse_entries(text)
    by_start = {entry.start: entry for entry in entries}
    in_block: set[int] = set()
    for entry in entries:
        in_block.update(range(entry.start, entry.end))
    out: list[str] = []
    for index, line in enumerate(lines):
        entry = by_start.get(index)
        if entry is not None:
            out.append(strip_mem_comments(entry.bullet))
            continue
        if index in in_block or MEM_COMMENT_LINE_RE.match(line):
            continue
        out.append(strip_mem_comments(line))
    return "\n".join(out)


def _fold_view(text: str, key: str, hard_budget: int) -> str:
    """按 ``## `` 分组折叠：每段保留最新一条，其余只留计数提示。"""
    lines = text.splitlines()
    entries = parse_entries(text)
    by_start = {entry.start: entry for entry in entries}
    in_block: set[int] = set()
    for entry in entries:
        in_block.update(range(entry.start, entry.end))

    sections: dict[str | None, list[Entry]] = {}
    for entry in entries:
        sections.setdefault(entry.section, []).append(entry)
    kept: set[int] = set()
    for members in sections.values():
        kept.add(
            max(
                members,
                key=lambda entry: ((entry.mem or {}).get("touched", ""), entry.start),
            ).start
        )

    out: list[str] = []
    omitted = 0
    for index, line in enumerate(lines):
        entry = by_start.get(index)
        if line.startswith("## "):
            if omitted:
                out.append(_counter_note(key, omitted))
                omitted = 0
            out.append(strip_mem_comments(line))
            continue
        if entry is not None:
            if entry.start in kept:
                out.append(strip_mem_comments(entry.bullet))
            else:
                omitted += 1
            continue
        if index in in_block or MEM_COMMENT_LINE_RE.match(line):
            continue
        out.append(strip_mem_comments(line))
    if omitted:
        out.append(_counter_note(key, omitted))

    collapsed: list[str] = []
    for line in out:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    return _head_within("\n".join(collapsed), hard_budget)


def inject_view(text: str | None, key: str, hard_budget: int) -> str:
    """注入 system prompt 的视图：去 mem 注释；超 hard 时逐级折叠。

    纯函数，不碰磁盘；M6 的上下文包可以直接复用。
    """
    if not text:
        return text or ""
    visible = strip_mem_comments(text)
    if _tokens(visible) <= hard_budget:
        return text if visible == text else visible
    if key not in ENTRY_FILES:
        return _head_within(visible, hard_budget)
    compact = _compact_view(text)
    if _tokens(compact) <= hard_budget:
        return compact
    return _fold_view(text, key, hard_budget)


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
    archive_ids = entry_ids(archives[key])
    # 先给现有条目补 id：supersede 搬进 archive 的旧块也要带 id。
    stamped_before, _ = stamp_entries(
        target_text, target_before, key, archive_ids=archive_ids
    )
    new_archives = dict(archives)
    archive_before: str | None = None
    archive_after: str | None = None

    if action == "supersede":
        old = _match_supersede(stamped_before, supersedes or "", section)
        new_block = [
            f"- {entry_text}",
            f"  supersedes: {old.bullet}",  # 永远取文件里的原行
            *[f"  {line}" for line in details],
        ]
        # section 只用于查找消歧；落点永远跟随旧条目的分组。
        target_section = old.section
        target_text = _remove_entry(stamped_before, old)
        if target_section is None:
            target_after = _insert_block_at(target_text, old.start, new_block)
        else:
            target_after = _append_block(target_text, new_block, target_section)
        target_after, _ = stamp_entries(
            target_after, stamped_before, key, archive_ids=archive_ids
        )

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
        target_after = _append_block(stamped_before, new_block, section)
        target_after, _ = stamp_entries(
            target_after, stamped_before, key, archive_ids=archive_ids
        )

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


def atomic_write(path: Path, text: str) -> None:
    """同目录临时文件 + flush + os.replace；newline="" 保留 CRLF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".keeper-tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _write_change(change: FileChange) -> None:
    atomic_write(change.path, change.after)


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


def _render_memory(root: Path, context_window: int | None = None) -> str:
    parts: list[str] = []
    budgets = effective_budgets(context_window)
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
            view = inject_view(text, key, budgets[key]["hard"])
            if not view or not view.strip():
                continue
            # 只去掉首尾换行，保留第一行可能存在的缩进。
            body = view.strip("\r\n")
            parts.append(f"{MEMORY_SECTION_PREFIX}{name}\n\n{body}")
    if not parts:
        return ""
    return "# 项目记忆（启动时注入）\n\n" + "\n\n".join(parts) + "\n"


def render_memory(root: Path, context_window: int | None = None) -> str:
    """启动注入文本：DECISIONS -> LESSONS -> INDEX -> PROJECT_STATE。

    ``context_window`` 为当前 worker 的窗口；None 保持默认预算（旧行为）。
    """
    try:
        return _render_memory(root, context_window=context_window)
    except Exception as exc:
        return (
            "# 项目记忆（启动时注入）\n\n"
            f"(memory: 读取失败: {type(exc).__name__}: {exc})\n"
        )


# 供 tidy 模块复用的文本工具（名字不带下划线，避免跨模块引用私有实现）。
detect_newline = _detect_newline
ensure_header = _ensure_header
insert_block_at = _insert_block_at
remove_entry = _remove_entry
rewrap = _rewrap
