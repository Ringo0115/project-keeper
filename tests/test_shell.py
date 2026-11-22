from pathlib import Path

from keeper.tools.base import ToolResult
from keeper.tools.shell import ShellTool


def _run(
    root: Path, argv: list, timeout: float | None = None
) -> ToolResult:
    args: dict = {"argv": argv}
    if timeout is not None:
        args["timeout_sec"] = timeout
    return ShellTool(root).execute(args)


def test_runs_whitelisted_python(tmp_path: Path) -> None:
    result = _run(tmp_path, ["python", "-c", "print('hello')"])
    assert result.ok
    assert "[stdout]" in result.content
    assert "hello" in result.content
    assert "(exit 0)" in result.content


def test_rejects_non_whitelisted_command(tmp_path: Path) -> None:
    result = _run(tmp_path, ["curl", "https://example.com"])
    assert not result.ok
    assert result.content.startswith("ERROR:")
    assert "not allowed" in result.content


def test_rejects_executable_path(tmp_path: Path) -> None:
    result = _run(tmp_path, ["C:\\Windows\\System32\\cmd.exe", "/c", "dir"])
    assert not result.ok
    assert "bare command name" in result.content


def test_rejects_empty_argv(tmp_path: Path) -> None:
    result = _run(tmp_path, [])
    assert not result.ok
    assert "non-empty array" in result.content


def test_rejects_invalid_argv_items(tmp_path: Path) -> None:
    result = _run(tmp_path, ["git", 1])  # type: ignore[list-item]
    assert not result.ok
    assert "non-empty strings" in result.content


def test_no_shell_expansion(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        [
            "python",
            "-c",
            "import sys; print('ARG=' + sys.argv[1])",
            "a; echo pwned",
        ],
    )
    assert result.ok
    assert "ARG=a; echo pwned" in result.content


def test_nonzero_exit_reports_error(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        ["python", "-c", "import sys; sys.stderr.write('bad\\n'); sys.exit(3)"],
    )
    assert not result.ok
    assert result.content.startswith("ERROR:")
    assert "exited with code 3" in result.content
    assert "bad" in result.content


def test_timeout_is_interrupted(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        ["python", "-c", "import time; time.sleep(5)"],
        timeout=0.25,
    )
    assert not result.ok
    assert result.content.startswith("ERROR:")
    assert "timed out" in result.content


def test_long_output_is_truncated(tmp_path: Path) -> None:
    result = _run(tmp_path, ["python", "-c", "print('x' * 50000)"])
    assert result.ok
    assert "[truncated" in result.content
    assert len(result.content) < 22_000


def test_cwd_is_project_root(tmp_path: Path) -> None:
    result = _run(tmp_path, ["python", "-c", "import os; print(os.getcwd())"])
    assert result.ok
    stdout = result.content.split("[stdout]", 1)[1].split("[stderr]", 1)[0]
    line = stdout.strip().splitlines()[0]
    assert Path(line).resolve() == tmp_path.resolve()