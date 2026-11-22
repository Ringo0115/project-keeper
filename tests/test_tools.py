from pathlib import Path

import pytest

from keeper.tools.base import Tool, ToolResult
from keeper.tools.fs import ReadFileTool
from keeper.tools.registry import Registry


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# demo\n\nhello\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


def test_read_file_ok(root: Path) -> None:
    result = ReadFileTool(root).execute({"path": "README.md"})
    assert result.ok
    assert "# demo" in result.content
    assert "hello" in result.content
    assert "lines 1-3 of 3" in result.content
    assert "    1\t# demo" in result.content


def test_read_file_missing(root: Path) -> None:
    result = ReadFileTool(root).execute({"path": "nope.md"})
    assert not result.ok
    assert result.content.startswith("ERROR:")
    assert "not found" in result.content


def test_read_file_escape_root(root: Path) -> None:
    result = ReadFileTool(root).execute({"path": "../outside.txt"})
    assert not result.ok
    assert "escapes project root" in result.content


def test_read_file_directory(root: Path) -> None:
    result = ReadFileTool(root).execute({"path": "src"})
    assert not result.ok
    assert "directory" in result.content


def test_read_file_binary(root: Path) -> None:
    (root / "bin.dat").write_bytes(b"\x00\x01\x02")
    result = ReadFileTool(root).execute({"path": "bin.dat"})
    assert not result.ok
    assert "binary" in result.content


def test_read_file_truncates(root: Path) -> None:
    (root / "long.txt").write_text(
        "\n".join(f"line {i}" for i in range(1, 301)), encoding="utf-8"
    )
    result = ReadFileTool(root).execute({"path": "long.txt", "max_lines": 10})
    assert result.ok
    assert "lines 1-10 of 300" in result.content
    assert "line 10" in result.content
    assert "line 11" not in result.content
    assert "truncated" in result.content
    assert "start_line=11" in result.content


def test_read_file_start_line(root: Path) -> None:
    (root / "long.txt").write_text(
        "\n".join(f"line {i}" for i in range(1, 301)), encoding="utf-8"
    )
    result = ReadFileTool(root).execute(
        {"path": "long.txt", "start_line": 100, "max_lines": 5}
    )
    assert result.ok
    assert "lines 100-104 of 300" in result.content


def test_read_file_requires_path(root: Path) -> None:
    result = ReadFileTool(root).execute({})
    assert not result.ok
    assert "path is required" in result.content


def test_registry_unknown_tool(root: Path) -> None:
    registry = Registry([ReadFileTool(root)])
    result = registry.dispatch("nope", {})
    assert not result.ok
    assert "unknown tool" in result.content


def test_registry_rejects_non_object_args(root: Path) -> None:
    registry = Registry([ReadFileTool(root)])
    result = registry.dispatch("read_file", ["README.md"])  # type: ignore[arg-type]
    assert not result.ok
    assert "JSON object" in result.content


class BoomTool(Tool):
    name = "boom"
    description = "always crashes"
    parameters = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> ToolResult:
        raise RuntimeError("kaboom")


def test_registry_wraps_tool_crash() -> None:
    registry = Registry([BoomTool()])
    result = registry.dispatch("boom", {})
    assert not result.ok
    assert result.content.startswith("ERROR:")
    assert "kaboom" in result.content