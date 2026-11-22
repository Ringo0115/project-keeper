"""受限 shell 工具：argv 数组、命令白名单、固定 cwd、超时与输出截断。

永远不使用 shell=True，所以参数里的 `;` `&&` `|` 都只是普通字符串。
"""

from __future__ import annotations

import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

from keeper.tools.base import Tool, ToolResult, error

ALLOWED_COMMANDS = (
    "git",
    "python",
    "python3",
    "py",
    "pytest",
    "pip",
    "pip3",
)

DEFAULT_TIMEOUT = 30.0
MIN_TIMEOUT = 0.1
MAX_TIMEOUT = 120.0
MAX_OUTPUT_CHARS = 20_000


class ShellTool(Tool):
    name: ClassVar[str] = "shell"
    description: ClassVar[str] = (
        "在项目根目录执行一条受限命令。参数必须是 argv 数组，例如 "
        '["git", "status"]；第一条命令必须在白名单内 '
        f"({', '.join(ALLOWED_COMMANDS)})。不使用 shell，不支持管道和重定向。"
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string"},
                "description": '命令和参数数组，例如 ["pytest", "-q"]',
            },
            "timeout_sec": {
                "type": "number",
                "description": (
                    f"超时秒数，默认 {DEFAULT_TIMEOUT:g}，"
                    f"范围 {MIN_TIMEOUT:g}-{MAX_TIMEOUT:g}"
                ),
            },
        },
        "required": ["argv"],
    }
    danger: ClassVar[str] = "exec"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv:
            return error("argv must be a non-empty array of strings")
        if not all(isinstance(item, str) and item for item in argv):
            return error("argv must contain only non-empty strings")

        command = _command_name(argv[0])
        if command is None:
            return error(
                f"argv[0] must be a bare command name, not a path: {argv[0]!r}"
            )
        if command not in ALLOWED_COMMANDS:
            return error(
                f"command not allowed: {argv[0]!r}. "
                f"Allowed: {', '.join(ALLOWED_COMMANDS)}"
            )

        timeout = _timeout(args.get("timeout_sec"))
        spawn_argv = list(argv)
        if command in ("python", "python3") and not argv[0].lower().endswith(".exe"):
            # Windows 上 python 可能不在 PATH，统一用当前解释器。
            spawn_argv[0] = sys.executable

        try:
            completed = subprocess.run(
                spawn_argv,
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = _combine_streams(exc.stdout, exc.stderr)
            message = f"command timed out after {timeout:g}s: {shlex.join(argv)}"
            if partial:
                message += f"\n[partial output]\n{_truncate(partial)}"
            return error(message)
        except FileNotFoundError:
            return error(f"command not found: {argv[0]}")
        except (OSError, ValueError) as exc:
            return error(f"cannot run {shlex.join(argv)}: {exc}")

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        header = f"$ {shlex.join(argv)} (exit {completed.returncode})"
        body = _combine_streams(stdout, stderr) or "(no output)"

        if completed.returncode == 0:
            return ToolResult(ok=True, content=f"{header}\n{body}")
        return error(
            f"command exited with code {completed.returncode}: {shlex.join(argv)}\n{body}"
        )


def _command_name(raw: str) -> str | None:
    if "/" in raw or "\\" in raw:
        return None
    name = raw.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    if not math.isfinite(timeout):
        return DEFAULT_TIMEOUT
    return max(MIN_TIMEOUT, min(MAX_TIMEOUT, timeout))


def _combine_streams(stdout: Any, stderr: Any) -> str:
    parts: list[str] = []
    out = _as_text(stdout).rstrip()
    err = _as_text(stderr).rstrip()
    if out:
        parts.append(f"[stdout]\n{_truncate(out)}")
    if err:
        parts.append(f"[stderr]\n{_truncate(err)}")
    return "\n".join(parts)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = MAX_OUTPUT_CHARS * 3 // 5
    tail = MAX_OUTPUT_CHARS - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[truncated {omitted} chars]\n{text[-tail:]}"