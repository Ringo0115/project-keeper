"""审批策略：只读自动放行，写入/执行需要确认，删除与高危命令直接拒绝。

策略判断（review）是纯函数，确认动作（Approver.confirm）通过参数注入，
测试不依赖 stdin。任何异常、无输入、EOF、超时都按拒绝处理（fail-closed）。
"""

from __future__ import annotations

import difflib
import os
import shlex
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TextIO

from keeper.events import ApprovalRequest
from keeper.tools.fs import resolve_path

DEFAULT_TIMEOUT = 60.0

PREVIEW_DIFF_LINES = 80
PREVIEW_NEW_LINES = 20
PREVIEW_SOURCE_LINES = 5000
PREVIEW_MAX_BYTES = 1_000_000

_DELETE_COMMANDS = {"rm", "del", "erase", "rmdir", "rd", "unlink", "shred"}
_ALWAYS_DANGEROUS = {"format", "shutdown"}


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


_DANGER_DECISION = {
    "read": Decision.ALLOW,
    "write": Decision.ASK,
    "exec": Decision.ASK,
}


@dataclass(frozen=True)
class Review:
    decision: Decision
    reason: str = ""
    summary: str = ""
    detail: str = ""


class Approver(Protocol):
    """审批器只负责回答「用户是否同意」，策略判断在 review() 里。"""

    def confirm(self, request: ApprovalRequest) -> bool: ...


def review(
    *,
    tool_name: str,
    danger: str,
    args: dict[str, Any],
    root: Path,
) -> Review:
    """执行前的静态判断。不碰 stdin，不执行工具。"""
    summary = _summary(tool_name, args)

    if tool_name == "shell":
        detail = _shell_detail(args)
        hazard = dangerous_command(args.get("argv"))
        if hazard:
            return Review(
                decision=Decision.DENY,
                reason=hazard,
                summary=summary,
                detail=detail,
            )
        return Review(decision=Decision.ASK, summary=summary, detail=detail)

    if danger == "delete":
        return Review(
            decision=Decision.DENY,
            reason="delete operations are not allowed",
            summary=summary,
        )

    decision = _DANGER_DECISION.get(danger)
    if decision is None:
        return Review(
            decision=Decision.DENY,
            reason=f"unknown danger level '{danger}' (fail-closed)",
            summary=summary,
        )

    if decision is Decision.ALLOW:
        return Review(decision=Decision.ALLOW, summary=summary)

    detail = _write_preview(args, root) if tool_name == "write_file" else ""
    return Review(decision=Decision.ASK, summary=summary, detail=detail)


def dangerous_command(argv: Any) -> str | None:
    """识别直接拒绝的命令；正常命令返回 None。"""
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        return None

    command = _basename(argv[0]).lower()
    if command.endswith(".exe"):
        command = command[:-4]
    rest = [item.lower() for item in argv[1:]]

    if command in _DELETE_COMMANDS:
        return f"delete command '{command}' is not allowed"
    if command in _ALWAYS_DANGEROUS:
        return f"high-risk command '{command}' is not allowed"

    if command == "git":
        if "reset" in rest and "--hard" in rest:
            return "git reset --hard is not allowed"
        if "push" in rest and any(
            item == "--force" or _short_flag_has(item, "f") for item in rest
        ):
            return "git push --force is not allowed"
        if "clean" in rest and any(_short_flag_has(item, "f") for item in rest):
            return "git clean -f is not allowed"

    return None


def _basename(raw: str) -> str:
    return raw.replace("\\", "/").rsplit("/", 1)[-1]


def _short_flag_has(item: str, flag: str) -> bool:
    return item.startswith("-") and not item.startswith("--") and flag in item[1:]


def _summary(tool_name: str, args: dict[str, Any]) -> str:
    path = args.get("path")
    if tool_name == "shell":
        argv = args.get("argv")
        if isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv):
            return f"shell: {shlex.join(argv)}"
        return "shell"
    if isinstance(path, str) and path:
        return f"{tool_name}: {path}"
    return tool_name


def _shell_detail(args: dict[str, Any]) -> str:
    argv = args.get("argv")
    if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
        return "(invalid argv)"
    timeout = args.get("timeout_sec")
    suffix = f"\ntimeout: {timeout}s" if timeout is not None else ""
    return f"$ {shlex.join(argv)}\ncwd: project root{suffix}"


def _write_preview(args: dict[str, Any], root: Path) -> str:
    raw_path = args.get("path")
    content = args.get("content")
    if not isinstance(raw_path, str) or not isinstance(content, str):
        return "(cannot preview: path/content missing)"

    try:
        path = resolve_path(root, raw_path)
    except ValueError as exc:
        return f"(cannot preview: {exc})"

    try:
        rel = path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        rel = raw_path

    if path.exists() and path.is_dir():
        return "(path is a directory)"

    if not path.exists():
        lines = content.splitlines()
        shown = lines[:PREVIEW_NEW_LINES]
        body = "\n".join(f"+{line}" for line in shown)
        omitted = len(lines) - len(shown)
        note = f"\n... ({omitted} more lines)" if omitted > 0 else ""
        size = len(content.encode("utf-8"))
        return f"new file ({size} bytes, {len(lines)} lines)\n{body}{note}"

    try:
        if path.stat().st_size > PREVIEW_MAX_BYTES:
            return "(existing file too large to diff; it will be replaced)"
        old = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(cannot read existing file: {exc})"

    if old == content:
        return "no changes: new content is identical to the existing file"

    old_lines = old.splitlines()
    new_lines = content.splitlines()
    diff = list(
        difflib.unified_diff(
            old_lines[:PREVIEW_SOURCE_LINES],
            new_lines[:PREVIEW_SOURCE_LINES],
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return "no line changes (content differs only in trailing newline)"
    shown = diff[:PREVIEW_DIFF_LINES]
    omitted = len(diff) - len(shown)
    note = f"\n... ({omitted} diff lines omitted)" if omitted > 0 else ""
    return "\n".join(shown) + note


class StdioApprover:
    """在终端询问 y/N。EOF、异常、超时、空回答一律视为拒绝。"""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        stream: TextIO | None = None,
    ) -> None:
        self.timeout = timeout
        self.stream = stream or sys.stderr

    def confirm(self, request: ApprovalRequest) -> bool:
        print(
            f"approve {request.summary}? [y/N] ",
            end="",
            file=self.stream,
            flush=True,
        )
        try:
            answer = _read_line(self.timeout)
        except (EOFError, KeyboardInterrupt, OSError):
            print(file=self.stream)
            return False
        print(file=self.stream)
        if answer is None:
            print(
                f"[approval] timeout after {self.timeout:g}s, denied",
                file=self.stream,
            )
            return False
        return answer.strip().lower() in {"y", "yes"}


def _read_line(timeout: float) -> str | None:
    """读一行；TTY 下支持超时，非 TTY 依赖 EOF。超时返回 None。"""
    if timeout > 0 and sys.stdin.isatty():
        if os.name == "nt":
            return _read_line_windows(timeout, sys.stderr)
        return _read_line_posix(timeout)

    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")


def _read_line_windows(timeout: float, stream: TextIO) -> str | None:
    import msvcrt

    chars: list[str] = []
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            return None
        if not msvcrt.kbhit():
            time.sleep(0.03)
            continue

        char = msvcrt.getwch()
        if char in ("\r", "\n"):
            return "".join(chars)
        if char == "\x08":
            if chars:
                chars.pop()
                stream.write("\b \b")
                stream.flush()
            continue
        if char == "\x03":
            raise KeyboardInterrupt
        if char == "\x1a":
            raise EOFError
        if char in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        chars.append(char)
        stream.write(char)
        stream.flush()


def _read_line_posix(timeout: float) -> str | None:
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")