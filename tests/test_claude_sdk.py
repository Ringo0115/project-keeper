from __future__ import annotations

import pytest

from keeper.claude_sdk import (
    Autonomy,
    DELEGATE,
    READ,
    REQUIRES_APPROVAL,
    SHELL,
    WRITE,
    classify_tool,
    native_claude_cli,
    should_auto_approve,
)


# ---------------------------------------------------------------------------
# A. 原生 CLI 查找（5 条）
# ---------------------------------------------------------------------------
def test_windows_false_returns_command_unchanged():
    assert native_claude_cli(r"C:\npm\claude.cmd", windows=False) == r"C:\npm\claude.cmd"


def test_prefers_exe_beside_npm_shim(tmp_path):
    shim = tmp_path / "npm" / "claude.cmd"
    shim.parent.mkdir()
    shim.write_text("@echo off\n", encoding="utf-8")
    exe = tmp_path / "npm" / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")

    got = native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                            which=lambda _name: None)

    assert got == str(exe)


def test_falls_back_to_claude_exe_on_path(tmp_path):
    shim = tmp_path / "npm" / "claude.cmd"
    shim.parent.mkdir()
    shim.write_text("@echo off\n", encoding="utf-8")
    on_path = tmp_path / "bin" / "claude.exe"
    on_path.parent.mkdir()
    on_path.write_text("", encoding="utf-8")

    got = native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                            which=lambda name: str(on_path) if name == "claude.exe" else None)

    assert got == str(on_path)


def test_falls_back_to_home_local_bin(tmp_path):
    shim = tmp_path / "npm" / "claude.cmd"
    shim.parent.mkdir()
    shim.write_text("@echo off\n", encoding="utf-8")
    home = tmp_path / "home"
    local_exe = home / ".local" / "bin" / "claude.exe"
    local_exe.parent.mkdir(parents=True)
    local_exe.write_text("", encoding="utf-8")

    got = native_claude_cli(str(shim), windows=True, home=str(home),
                            which=lambda _name: None)

    assert got == str(local_exe)


def test_no_native_exe_returns_shim_unchanged(tmp_path):
    shim = tmp_path / "npm" / "claude.cmd"
    shim.parent.mkdir()
    shim.write_text("@echo off\n", encoding="utf-8")

    got = native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                            which=lambda _name: None)

    assert got == str(shim)


# ---------------------------------------------------------------------------
# B. 工具分类（5 条）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("Read", READ),
        ("Edit", WRITE),
        ("Bash", SHELL),
        ("Agent", DELEGATE),
        ("SomeUnknownTool", REQUIRES_APPROVAL),
    ],
)
def test_classify_tool(tool_name, expected):
    assert classify_tool(tool_name) == expected


# ---------------------------------------------------------------------------
# C. 三档策略（4 条）
# ---------------------------------------------------------------------------
def test_hands_off_auto_approves_everything():
    for name in ("Read", "Write", "Bash", "Agent", "mcp__orbital__dispatch", "Unknown"):
        assert should_auto_approve(name, Autonomy.HANDS_OFF) is True


def test_check_in_auto_approves_read_tools():
    assert should_auto_approve("Read", Autonomy.CHECK_IN) is True
    assert should_auto_approve("Grep", Autonomy.CHECK_IN) is True


def test_check_in_denies_writes_shell_and_agent_but_allows_dispatch():
    for name in ("Write", "Bash", "Agent"):
        assert should_auto_approve(name, Autonomy.CHECK_IN) is False
    assert should_auto_approve("mcp__orbital__dispatch", Autonomy.CHECK_IN) is True


def test_supervised_never_auto_approves():
    for name in ("Read", "Write", "Bash", "Agent", "mcp__orbital__dispatch"):
        assert should_auto_approve(name, Autonomy.SUPERVISED) is False