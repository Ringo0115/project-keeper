"""M6 R1：项目注册表、root 解析与项目隔离。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from keeper.cli import main
from keeper.projects import (
    RegistryError,
    current_project,
    keeper_home,
    load_registry,
    register_project,
    registry_path,
    use_project,
)
from keeper.tasks import TASKS_FILE, TaskStore


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "keeper-home"
    monkeypatch.setenv("KEEPER_HOME", str(path))
    return path


def _mkproject(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def test_registry_location_honors_keeper_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEEPER_HOME", str(tmp_path / "kh"))
    assert keeper_home() == tmp_path / "kh"
    assert registry_path() == tmp_path / "kh" / "projects.json"


def test_missing_registry_is_empty_and_creates_nothing(home: Path) -> None:
    registry = load_registry()
    assert registry.projects == {}
    assert registry.current is None
    assert not home.exists()


def test_register_project_records_fields(tmp_path: Path, home: Path) -> None:
    root = _mkproject(tmp_path, "alpha")
    project = register_project(
        root,
        name="Alpha",
        workers=("built-in", "claude-code"),
        default_worker="claude-code",
        instructions="先读 docs/",
    )
    assert project.root == root.resolve()
    data = json.loads(registry_path().read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["current"] is None
    record = data["projects"]["alpha"]
    assert record["id"] == "alpha"
    assert record["name"] == "Alpha"
    assert record["root"] == str(root.resolve())
    assert record["workers"] == ["built-in", "claude-code"]
    assert record["default_worker"] == "claude-code"
    assert record["instructions"] == "先读 docs/"


def test_register_defaults_workers(tmp_path: Path, home: Path) -> None:
    root = _mkproject(tmp_path, "solo")
    project = register_project(root)
    assert project.workers == ("built-in",)
    assert project.default_worker == "built-in"

    implied = register_project(
        _mkproject(tmp_path, "other"), default_worker="codex"
    )
    assert implied.workers == ("codex",)
    assert implied.default_worker == "codex"


def test_register_validation(tmp_path: Path, home: Path) -> None:
    root = _mkproject(tmp_path, "x")
    with pytest.raises(RegistryError, match="不存在"):
        register_project(tmp_path / "missing")
    with pytest.raises(RegistryError, match="不合法"):
        register_project(root, project_id="Bad Id")
    with pytest.raises(RegistryError, match="不在启用"):
        register_project(
            root, project_id="ok", workers=("built-in",), default_worker="codex"
        )


def test_register_rejects_duplicate_id_and_directory(
    tmp_path: Path, home: Path
) -> None:
    first = _mkproject(tmp_path, "first")
    second = _mkproject(tmp_path, "second")
    register_project(first, project_id="same")
    with pytest.raises(RegistryError, match="已存在"):
        register_project(second, project_id="same")
    with pytest.raises(RegistryError, match="已注册"):
        register_project(first, project_id="another")


def test_cli_project_register_list_show_use(
    tmp_path: Path, home: Path, capsys
) -> None:
    root = _mkproject(tmp_path, "demo")
    assert main(["project", "register", str(root), "--name", "演示项目"]) == 0
    assert "registered demo  演示项目" in capsys.readouterr().out

    assert main(["project", "list"]) == 0
    listed = capsys.readouterr().out
    assert " demo  演示项目 " in listed
    assert not listed.startswith("*")

    assert main(["project", "show", "demo"]) == 0
    shown = capsys.readouterr().out
    assert f"root: {root.resolve()}" in shown
    assert "workers: built-in" in shown
    assert "default_worker: built-in" in shown
    assert "instructions: " in shown

    assert main(["project", "use", "demo"]) == 0
    assert "current project: demo" in capsys.readouterr().out
    assert current_project() is not None
    assert current_project().id == "demo"

    assert main(["project", "list"]) == 0
    assert capsys.readouterr().out.startswith("* demo")


def test_cli_unknown_project_errors(tmp_path: Path, home: Path, capsys) -> None:
    assert main(["project", "show", "nope"]) == 2
    assert "未注册" in capsys.readouterr().err
    assert main(["project", "use", "nope"]) == 2
    assert "未注册" in capsys.readouterr().err


def test_selected_project_drives_task_add_and_data_stays_isolated(
    tmp_path: Path, home: Path, monkeypatch, capsys
) -> None:
    alpha = _mkproject(tmp_path, "alpha")
    beta = _mkproject(tmp_path, "beta")
    elsewhere = _mkproject(tmp_path, "elsewhere")
    register_project(alpha, name="Alpha")
    register_project(beta, name="Beta")

    use_project("alpha")
    monkeypatch.chdir(elsewhere)
    assert main(["task", "add", "alpha task"]) == 0
    capsys.readouterr()

    use_project("beta")
    assert main(["task", "add", "beta task"]) == 0
    capsys.readouterr()

    assert [task.title for task in TaskStore.load(alpha).tasks] == ["alpha task"]
    assert [task.title for task in TaskStore.load(beta).tasks] == ["beta task"]
    assert not (elsewhere / TASKS_FILE).exists()


def test_explicit_root_wins_with_notice(
    tmp_path: Path, home: Path, capsys
) -> None:
    selected = _mkproject(tmp_path, "selected")
    other = _mkproject(tmp_path, "other")
    register_project(selected, project_id="sel")
    use_project("sel")

    assert main(["task", "add", "old style", "--root", str(other)]) == 0
    captured = capsys.readouterr()
    assert "覆盖已选项目 sel" in captured.err
    assert (other / TASKS_FILE).exists()
    assert not (selected / TASKS_FILE).exists()


def test_root_defaults_to_cwd_without_registry(
    tmp_path: Path, home: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["task", "add", "plain"]) == 0
    capsys.readouterr()
    assert (tmp_path / TASKS_FILE).exists()


def test_corrupt_registry_is_not_overwritten(
    tmp_path: Path, home: Path, capsys
) -> None:
    home.mkdir(parents=True)
    path = home / "projects.json"
    original = '{"version": 1, "projects": {'
    path.write_text(original, encoding="utf-8")
    root = _mkproject(tmp_path, "p")

    assert main(["project", "register", str(root)]) == 2
    assert "损坏" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == original

    assert main(["project", "use", "x"]) == 2
    assert path.read_text(encoding="utf-8") == original

    with pytest.raises(RegistryError, match="损坏"):
        register_project(root)


def test_corrupt_registry_falls_back_to_cwd_with_warning(
    tmp_path: Path, home: Path, monkeypatch, capsys
) -> None:
    home.mkdir(parents=True)
    (home / "projects.json").write_text("not json", encoding="utf-8")
    workdir = _mkproject(tmp_path, "work")
    monkeypatch.chdir(workdir)

    assert main(["task", "add", "fallback"]) == 0
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "回退到当前目录" in captured.err
    assert (workdir / TASKS_FILE).exists()


def test_registry_version_mismatch(home: Path) -> None:
    home.mkdir(parents=True)
    (home / "projects.json").write_text(
        '{"version": 99, "projects": {}, "current": null}', encoding="utf-8"
    )
    with pytest.raises(RegistryError, match="版本"):
        load_registry()


def test_read_only_commands_do_not_create_global_state(
    home: Path, capsys
) -> None:
    assert not home.exists()
    assert main(["project", "list"]) == 0
    assert "(no projects)" in capsys.readouterr().out
    assert not home.exists()

    assert main(["project", "show", "x"]) == 2
    capsys.readouterr()
    assert not home.exists()