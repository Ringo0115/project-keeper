from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import keeper.dispatch as dispatch_mod
from keeper.context import WorkerBrief
from keeper.dispatch import _register_live, _unregister_live, dispatch_fanout, kill_live_workers
from keeper.fanout import FanoutBatch
from keeper.projects import Project


def project(root: Path) -> Project:
    return Project("demo", "Demo", root, ("built-in", "codex", "pi"), "built-in")


class Approver:
    def confirm(self, request):  # 全部批准
        return True


def codex(monkeypatch, code: str) -> None:
    monkeypatch.setenv("KEEPER_WORKER_CODEX_ARGV", json.dumps([sys.executable, "-c", code, "{input}"]))


def pi(monkeypatch, code: str) -> None:
    monkeypatch.setenv("KEEPER_WORKER_PI_ARGV", json.dumps([sys.executable, "-c", code, "{input}"]))


def _on_disk(tmp_path: Path) -> dict:
    files = list((tmp_path / ".agent" / "fanout").glob("*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_batch_created_running_and_resolved_when_last_task_lands(tmp_path: Path) -> None:
    batch = FanoutBatch.create(tmp_path, [("codex", "a"), ("pi", "b")])
    data = _on_disk(tmp_path)
    assert [task["status"] for task in data["tasks"]] == ["running", "running"]
    assert data["status"] == "running"

    assert batch.absorb("a", status="done", summary="ok") is True
    assert batch.absorb("a", status="failed", summary="迟到的重复上报") is False  # 幂等
    data = _on_disk(tmp_path)
    assert data["tasks"][0]["status"] == "done" and data["status"] == "running"

    assert batch.absorb("b", status="blocked", summary="等审批") is True
    data = _on_disk(tmp_path)
    assert data["status"] == "resolved" and data["resolve_reason"] == "all_terminal"
    assert batch.resolve("interrupted") is False  # 已收尾，再收尾无效


def test_resolve_marks_unfinished_tasks_interrupted_once(tmp_path: Path) -> None:
    batch = FanoutBatch.create(tmp_path, [("codex", "a"), ("pi", "b")])
    batch.absorb("a", status="done", summary="ok")

    assert batch.resolve("interrupted") is True
    data = _on_disk(tmp_path)
    assert [task["status"] for task in data["tasks"]] == ["done", "interrupted"]
    assert data["status"] == "resolved" and data["resolve_reason"] == "interrupted"

    assert batch.resolve("interrupted") is False
    assert _on_disk(tmp_path)["resolve_reason"] == "interrupted"


def test_fanout_writes_batch_records_on_disk(tmp_path: Path, monkeypatch) -> None:
    codex(monkeypatch, "print('codex ok')")
    pi(monkeypatch, "print('pi ok')")
    briefs = [WorkerBrief("a", "task a", ("src",)), WorkerBrief("b", "task b", ("src",))]

    records = dispatch_fanout(project(tmp_path), briefs, ["codex", "pi"], Approver())

    assert len(records) == 2
    data = _on_disk(tmp_path)
    assert data["status"] == "resolved"
    assert data["resolve_reason"] in {"all_terminal", "finished"}
    assert {task["label"] for task in data["tasks"]} == {"a", "b"}
    assert all(task["status"] in {"done", "blocked", "failed", "interrupted"} for task in data["tasks"])
    assert all(task["ended_at"] for task in data["tasks"])


def test_interrupt_resolves_batch_before_reraising(tmp_path: Path, monkeypatch) -> None:
    codex(monkeypatch, "import time; time.sleep(1)")
    pi(monkeypatch, "import time; time.sleep(1)")

    def boom(futures):
        raise KeyboardInterrupt

    monkeypatch.setattr(dispatch_mod, "as_completed", boom)
    briefs = [WorkerBrief("a", "task a", ("src",)), WorkerBrief("b", "task b", ("src",))]

    with pytest.raises(KeyboardInterrupt):
        dispatch_fanout(project(tmp_path), briefs, ["codex", "pi"], Approver())

    data = _on_disk(tmp_path)
    assert data["status"] == "resolved" and data["resolve_reason"] == "interrupted"
    assert {task["status"] for task in data["tasks"]} == {"interrupted"}


def test_kill_live_workers_terminates_real_child_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _register_live(proc)
    try:
        killed = kill_live_workers()
    finally:
        _unregister_live(proc)

    assert proc.pid in killed
    assert proc.wait(timeout=15) is not None
