# keeper — 项目级持久化 Agent（CLI）
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Claude Agent SDK 助手：原生 CLI 查找 + 工具风险分级。

块 A 原样搬自 Orbital ``agent_os/agent/transports/sdk_transport.py``
（``_is_batch_file`` / ``native_claude_cli``）；块 B 原样搬自
``agent_os/agent/transports/tool_risk.py``（风险常量、工具分类与放行策略）。

本文件不 import ``claude_agent_sdk``，也不 import ``agent_os``：它只提供
SDK 将来要用的助手函数。Orbital 的 ``Autonomy`` 定义在它自己的
``prompt_builder`` 里，keeper 没有那个模块，因此三档枚举在本文件内定义，
``should_auto_approve`` 的分支逻辑一行不改。
"""

from __future__ import annotations

import enum
import logging
import os
import shutil
import sys
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 块 A：原生 CLI 查找（来自 Orbital sdk_transport.py）
# ---------------------------------------------------------------------------
def _is_batch_file(path: str) -> bool:
    return path.lower().endswith((".cmd", ".bat"))


def native_claude_cli(
    command: str | None,
    *,
    windows: bool | None = None,
    home: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    """Windows 上把 npm 的 ``claude.cmd`` 垫片换成原生 ``claude.exe``。

    claude-agent-sdk 0.2.x 拒绝把 ``.bat``/``.cmd`` 文件当作 CLI 启动
    （cmd.exe 会重新解析 argv，无法可靠转义），而 npm 安装下对 ``claude``
    的 PATH 解析会落到 ``%APPDATA%\\npm\\claude.CMD``，于是每次 Claude Code
    派发都在适配器启动时失败。垫片本身只是去跑旁边的
    ``node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe``。
    查找顺序：该 exe、PATH 上的 ``claude.exe``、原生安装器的
    ``~\\.local\\bin\\claude.exe``。三者都没有就原样返回垫片，让 SDK 自己
    的拒绝信息说明该装什么。
    """
    if windows is None:
        windows = sys.platform == "win32"
    if not windows or not command or not _is_batch_file(command):
        return command
    candidates = [os.path.join(os.path.dirname(command), "node_modules",
                               "@anthropic-ai", "claude-code", "bin", "claude.exe")]
    on_path = which("claude.exe")
    if on_path and not _is_batch_file(on_path):
        candidates.append(on_path)
    candidates.append(os.path.join(home or os.path.expanduser("~"),
                                   ".local", "bin", "claude.exe"))
    for candidate in candidates:
        if candidate.lower().endswith(".exe") and os.path.isfile(candidate):
            logger.info("Claude Code CLI: using native %s instead of shim %s",
                        candidate, command)
            return candidate
    return command


# ---------------------------------------------------------------------------
# 块 B：工具风险分级（来自 Orbital tool_risk.py）
# ---------------------------------------------------------------------------
class Autonomy(enum.Enum):
    """等价于 Orbital ``prompt_builder.Autonomy`` 的三档自主度。"""

    HANDS_OFF = "hands_off"
    CHECK_IN = "check_in"
    SUPERVISED = "supervised"


# ---------------------------------------------------------------------------
# 风险分类
# ---------------------------------------------------------------------------
READ = "read"
WRITE = "write"
SHELL = "shell"
DELEGATE = "delegate"
REQUIRES_APPROVAL = "requires_approval"

# ---------------------------------------------------------------------------
# 工具 -> 分类映射
# ---------------------------------------------------------------------------
_TOOL_CATEGORY: dict[str, str] = {
    # 读取 / 观察类
    "Read":        READ,
    "Glob":        READ,
    "Grep":        READ,
    "LS":          READ,
    "Search":      READ,
    "Explore":     READ,
    "TaskGet":     READ,
    "TaskList":    READ,
    "TaskOutput":  READ,
    "WebSearch":   READ,
    "WebFetch":    READ,
    "AskUser":     READ,
    # 写入 / 变更类
    "Edit":         WRITE,
    "Write":        WRITE,
    "MultiEdit":    WRITE,
    "NotebookEdit": WRITE,
    "TodoWrite":    WRITE,
    "TaskCreate":   WRITE,
    "TaskUpdate":   WRITE,
    # shell 类
    "Bash":        SHELL,
    "ExecuteBash": SHELL,
    # 委派类
    "Agent":       DELEGATE,
}


# Spec 161：Orbital 自己的派发工具，经 daemon 的 MCP 端点提供给会话顶层的
# worker。check-in 下按名字放行：用户已经把这个 worker 放在这个座位上，把
# 活转交出去就是该座位要做的事。Claude Code 原生的 ``Agent``（DELEGATE 类）
# 仍然会弹审批。
_APPROVED_BY_NAME = frozenset({"mcp__orbital__dispatch"})


def classify_tool(tool_name: str) -> str:
    """返回 *tool_name* 的风险分类。

    未识别的工具名归为 ``requires_approval``。
    """
    return _TOOL_CATEGORY.get(tool_name, REQUIRES_APPROVAL)


def should_auto_approve(tool_name: str, autonomy: Autonomy) -> bool:
    """判断 *tool_name* 在 *autonomy* 下能否自动放行。

    策略：
        HANDS_OFF  — 全部自动放行。
        CHECK_IN   — 自动放行 ``read`` 类工具和 ``mcp__orbital__dispatch``。
        SUPERVISED — 一律不自动放行（每个请求都上报）。
    """
    if autonomy is Autonomy.HANDS_OFF:
        return True
    if autonomy is Autonomy.SUPERVISED:
        return False
    # CHECK_IN：read 类工具，外加按名字放行的 Orbital 派发工具
    if tool_name in _APPROVED_BY_NAME:
        return True
    return classify_tool(tool_name) == READ