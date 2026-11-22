from pathlib import Path

import pytest

from keeper.approval import (
    Decision,
    Review,
    StdioApprover,
    dangerous_command,
    review,
)
from keeper.events import ApprovalRequest


def _review(root: Path, tool_name: str, danger: str, args: dict) -> Review:
    return review(tool_name=tool_name, danger=danger, args=args, root=root)


def test_read_is_allowed(tmp_path: Path) -> None:
    result = _review(tmp_path, "read_file", "read", {"path": "a.txt"})
    assert result.decision is Decision.ALLOW


def test_write_asks_with_diff(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("old line\nkeep\n", encoding="utf-8")
    result = _review(
        tmp_path,
        "write_file",
        "write",
        {"path": "a.txt", "content": "new line\nkeep\n"},
    )
    assert result.decision is Decision.ASK
    assert result.summary == "write_file: a.txt"
    assert "-old line" in result.detail
    assert "+new line" in result.detail


def test_write_new_file_preview(tmp_path: Path) -> None:
    result = _review(
        tmp_path,
        "write_file",
        "write",
        {"path": "new.txt", "content": "hello\nworld\n"},
    )
    assert result.decision is Decision.ASK
    assert "new file" in result.detail
    assert "+hello" in result.detail
    assert "+world" in result.detail


def test_write_without_changes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("same\n", encoding="utf-8")
    result = _review(
        tmp_path,
        "write_file",
        "write",
        {"path": "a.txt", "content": "same\n"},
    )
    assert "no changes" in result.detail


def test_delete_is_denied(tmp_path: Path) -> None:
    result = _review(tmp_path, "delete_file", "delete", {"path": "a.txt"})
    assert result.decision is Decision.DENY
    assert "delete" in result.reason


def test_unknown_danger_is_denied(tmp_path: Path) -> None:
    result = _review(tmp_path, "mystery", "admin", {})
    assert result.decision is Decision.DENY
    assert "fail-closed" in result.reason


def test_shell_requires_approval(tmp_path: Path) -> None:
    result = _review(tmp_path, "shell", "exec", {"argv": ["git", "status"]})
    assert result.decision is Decision.ASK
    assert "$ git status" in result.detail
    assert "cwd: project root" in result.detail


@pytest.mark.parametrize(
    "argv",
    [
        ["rm", "-rf", "/"],
        ["rm", "file.txt"],
        ["del", "/s", "build"],
        ["rd", "/s", "/q", "build"],
        ["format", "C:"],
        ["shutdown", "/s"],
        ["git", "reset", "--hard"],
        ["git", "push", "--force", "origin", "main"],
        ["git", "push", "-f"],
        ["git", "clean", "-fdx"],
    ],
)
def test_dangerous_shell_is_denied(tmp_path: Path, argv: list[str]) -> None:
    result = _review(tmp_path, "shell", "exec", {"argv": argv})
    assert result.decision is Decision.DENY
    assert result.reason


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "status"],
        ["git", "push", "origin", "main"],
        ["git", "reset", "HEAD~1"],
        ["pytest", "-q"],
        ["python", "-c", "print(1)"],
    ],
)
def test_safe_commands_are_not_flagged(argv: list[str]) -> None:
    assert dangerous_command(argv) is None


def test_dangerous_command_ignores_bad_input() -> None:
    assert dangerous_command(None) is None
    assert dangerous_command("rm -rf /") is None
    assert dangerous_command([]) is None


def test_stdio_approver_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("keeper.approval._read_line", lambda timeout: "YES")
    approver = StdioApprover()
    assert approver.confirm(ApprovalRequest(summary="write_file: a.txt")) is True


@pytest.mark.parametrize("answer", ["", "n", "no", "yess", "1"])
def test_stdio_approver_rejects_everything_else(
    monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    monkeypatch.setattr("keeper.approval._read_line", lambda timeout: answer)
    approver = StdioApprover()
    assert approver.confirm(ApprovalRequest(summary="x")) is False


def test_stdio_approver_timeout_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("keeper.approval._read_line", lambda timeout: None)
    approver = StdioApprover(timeout=0.01)
    assert approver.confirm(ApprovalRequest(summary="x")) is False


def test_stdio_approver_eof_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_eof(timeout: float) -> str:
        raise EOFError

    monkeypatch.setattr("keeper.approval._read_line", raise_eof)
    approver = StdioApprover()
    assert approver.confirm(ApprovalRequest(summary="x")) is False