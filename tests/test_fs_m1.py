from pathlib import Path

import pytest

from keeper.tools.fs import ListDirTool, SearchTool, WriteFileTool


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("todo\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\nprint('hi')\n", encoding="utf-8"
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret\n", encoding="utf-8")
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "state.json").write_text("secret\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    return tmp_path


# ---------- list_dir ----------


def test_list_dir_shows_entries_and_skips_ignored(root: Path) -> None:
    result = ListDirTool(root).execute({})
    assert result.ok
    assert "README.md" in result.content
    assert "notes.txt" in result.content
    assert "src/" in result.content
    assert ".git" not in result.content
    assert ".agent" not in result.content
    assert "__pycache__" not in result.content
    assert "of 3 entries" in result.content


def test_list_dir_limits_entries(root: Path) -> None:
    result = ListDirTool(root).execute({"max_entries": 2})
    assert result.ok
    assert "2 of 3 entries" in result.content
    assert "[truncated: 1 more entries]" in result.content


def test_list_dir_subdirectory(root: Path) -> None:
    result = ListDirTool(root).execute({"path": "src"})
    assert result.ok
    assert "src (1 of 1 entries)" in result.content
    assert "app.py" in result.content


def test_list_dir_missing(root: Path) -> None:
    result = ListDirTool(root).execute({"path": "nope"})
    assert not result.ok
    assert "not found" in result.content


def test_list_dir_rejects_file(root: Path) -> None:
    result = ListDirTool(root).execute({"path": "README.md"})
    assert not result.ok
    assert "not a directory" in result.content


def test_list_dir_escape_root(root: Path) -> None:
    result = ListDirTool(root).execute({"path": ".."})
    assert not result.ok
    assert "escapes project root" in result.content


# ---------- write_file ----------


def test_write_file_creates_parents(root: Path) -> None:
    result = WriteFileTool(root).execute(
        {"path": "new/dir/file.txt", "content": "hello\n"}
    )
    assert result.ok
    assert "created" in result.content
    assert (root / "new" / "dir" / "file.txt").read_text(encoding="utf-8") == "hello\n"


def test_write_file_overwrites(root: Path) -> None:
    result = WriteFileTool(root).execute(
        {"path": "notes.txt", "content": "done\n"}
    )
    assert result.ok
    assert "overwrote" in result.content
    assert (root / "notes.txt").read_text(encoding="utf-8") == "done\n"


def test_write_file_empty_content(root: Path) -> None:
    result = WriteFileTool(root).execute({"path": "empty.txt", "content": ""})
    assert result.ok
    assert (root / "empty.txt").read_text(encoding="utf-8") == ""


def test_write_file_escape_root(root: Path) -> None:
    result = WriteFileTool(root).execute({"path": "../out.txt", "content": "x"})
    assert not result.ok
    assert "escapes project root" in result.content
    assert not (root.parent / "out.txt").exists()


def test_write_file_rejects_git_and_agent(root: Path) -> None:
    for path in (".git/config", ".agent/notes.md"):
        result = WriteFileTool(root).execute({"path": path, "content": "x"})
        assert not result.ok
        assert "refusing to write" in result.content


def test_write_file_rejects_directory(root: Path) -> None:
    result = WriteFileTool(root).execute({"path": "src", "content": "x"})
    assert not result.ok
    assert "is a directory" in result.content


def test_write_file_requires_content(root: Path) -> None:
    result = WriteFileTool(root).execute({"path": "a.txt"})
    assert not result.ok
    assert "content is required" in result.content


# ---------- search ----------


def test_search_finds_file_line(root: Path) -> None:
    result = SearchTool(root).execute({"pattern": "print"})
    assert result.ok
    assert "src/app.py:2: print('hi')" in result.content


def test_search_is_case_sensitive(root: Path) -> None:
    result = SearchTool(root).execute({"pattern": "PRINT"})
    assert result.ok
    assert "no matches" in result.content


def test_search_skips_ignored_dirs(root: Path) -> None:
    result = SearchTool(root).execute({"pattern": "secret"})
    assert result.ok
    assert "no matches" in result.content


def test_search_skips_binary_files(root: Path) -> None:
    (root / "bin.dat").write_bytes(b"\x00needle\n")
    result = SearchTool(root).execute({"pattern": "needle"})
    assert result.ok
    assert "no matches" in result.content


def test_search_limits_results(root: Path) -> None:
    (root / "many.txt").write_text(
        "\n".join(f"needle {i}" for i in range(10)), encoding="utf-8"
    )
    result = SearchTool(root).execute({"pattern": "needle", "max_results": 3})
    assert result.ok
    assert result.content.startswith("3 match(es)")
    assert "[truncated: stopped at 3" in result.content
    assert result.content.count("many.txt:") == 3


def test_search_single_file(root: Path) -> None:
    result = SearchTool(root).execute({"path": "src/app.py", "pattern": "import"})
    assert result.ok
    assert "src/app.py:1: import os" in result.content


def test_search_missing_path(root: Path) -> None:
    result = SearchTool(root).execute({"path": "nope", "pattern": "x"})
    assert not result.ok
    assert "not found" in result.content


def test_search_escape_root(root: Path) -> None:
    result = SearchTool(root).execute({"path": "..", "pattern": "x"})
    assert not result.ok
    assert "escapes project root" in result.content


def test_search_requires_pattern(root: Path) -> None:
    result = SearchTool(root).execute({})
    assert not result.ok
    assert "pattern is required" in result.content