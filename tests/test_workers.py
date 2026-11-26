"""M6 R3 / R11：worker 名册与私有记忆路径。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from keeper.projects import DEFAULT_WORKER, Project, register_project
from keeper.workers import (
    DEFAULT_WORKER_SPECS,
    WorkerError,
    WorkerSpec,
    default_workers,
    render_roster,
    roster_for,
    worker_memory_path,
)


def _project(
    root: Path,
    workers: tuple[str, ...] = ("built-in", "claude-code"),
    default_worker: str = "claude-code",
) -> Project:
    return Project(
        id="demo",
        name="Demo",
        root=root,
        workers=workers,
        default_worker=default_worker,
    )


def test_default_workers_complete() -> None:
    workers = default_workers()
    assert workers is DEFAULT_WORKER_SPECS
    assert [worker.slug for worker in workers] == [
        "built-in",
        "claude-code",
        "codex",
        "pi",
    ]
    by_slug = {worker.slug: worker for worker in workers}
    assert by_slug["built-in"].kind == "built-in"
    for slug in ("claude-code", "codex", "pi"):
        assert by_slug[slug].kind == "cli"
    for worker in workers:
        assert worker.name
        assert worker.description
        assert worker.routing_hint
        assert worker.permissions
        assert worker.autonomy
        assert worker.available is True


def test_worker_spec_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        default_workers()[0].name = "changed"  # type: ignore[misc]


def test_worker_spec_rejects_bad_kind_and_slug() -> None:
    with pytest.raises(WorkerError, match="kind"):
        WorkerSpec(
            slug="worker",
            name="Worker",
            kind="daemon",
            description="d",
            routing_hint="r",
            permissions="p",
            autonomy="a",
        )
    with pytest.raises(WorkerError, match="slug"):
        WorkerSpec(
            slug="../escape",
            name="Worker",
            kind="cli",
            description="d",
            routing_hint="r",
            permissions="p",
            autonomy="a",
        )


def test_render_roster_contains_slugs_and_hints() -> None:
    text = render_roster(default_workers())
    assert "worker 名册" in text
    for worker in default_workers():
        assert worker.slug in text
        assert worker.name in text
        assert worker.routing_hint in text
        assert worker.description in text
    assert "(没有启用的 worker)" not in text
    assert "(没有启用的 worker)" in render_roster([])


def test_roster_for_filters_by_project(tmp_path: Path) -> None:
    project = _project(
        tmp_path, workers=("built-in", "codex"), default_worker="codex"
    )
    assert [worker.slug for worker in roster_for(project)] == ["built-in", "codex"]


def test_roster_for_empty_uses_r1_default(tmp_path: Path) -> None:
    project = Project(id="p", name="P", root=tmp_path, workers=())
    assert [worker.slug for worker in roster_for(project)] == [DEFAULT_WORKER]


def test_roster_for_rejects_unknown_worker(tmp_path: Path) -> None:
    project = _project(tmp_path, workers=("ghost",), default_worker="ghost")
    with pytest.raises(WorkerError, match="未知 worker: ghost"):
        roster_for(project)


def test_worker_memory_path_is_inside_project_and_uncreated(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    path = worker_memory_path(root, "claude-code")
    assert path == root / ".agent" / "sub_agents" / "claude-code" / "MEMORY.md"
    assert path.is_relative_to(root)
    assert not path.exists()
    assert not (root / ".agent").exists()


@pytest.mark.parametrize(
    "slug",
    [
        "",
        "..",
        "../x",
        "a/b",
        "a\\b",
        "/abs",
        "C:\\abs",
        "a b",
        ".hidden",
        "A",
        "a.b",
    ],
)
def test_worker_memory_path_rejects_illegal_slug(
    tmp_path: Path, slug: str
) -> None:
    with pytest.raises(WorkerError):
        worker_memory_path(tmp_path, slug)
    assert not (tmp_path / ".agent").exists()


def test_roster_aligns_with_r1_default_worker(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project = register_project(
        root,
        workers=("built-in", "claude-code"),
        default_worker="claude-code",
    )
    roster = roster_for(project)
    assert {worker.slug for worker in roster} == {"built-in", "claude-code"}
    assert project.default_worker in {worker.slug for worker in roster}
    assert DEFAULT_WORKER == "built-in"