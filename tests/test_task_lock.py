from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from keeper.task_lock import TaskLock
from keeper.tasks import TaskStore


def _python_process(code: str, *args: str, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", code, *args],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_two_processes_cannot_overwrite_task_queue(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    holder_code = """
from pathlib import Path
import sys
from keeper.task_lock import TaskLock
from keeper.tasks import TaskStore
root = Path(sys.argv[1])
lock = TaskLock(root)
assert lock.acquire()
TaskStore.load(root).add('holder task')
print('READY', flush=True)
sys.stdin.readline()
lock.release()
"""
    holder = _python_process(holder_code, str(tmp_path), cwd=repo)
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "READY"

    contender = subprocess.run(
        [sys.executable, "-m", "keeper", "task", "add", "contender task", "--root", str(tmp_path)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert contender.returncode != 0
    assert "pid" in contender.stderr and str(tmp_path / ".agent" / "tasks.lock") in contender.stderr

    assert holder.stdin is not None
    holder.stdin.write("\n")
    holder.stdin.flush()
    out, err = holder.communicate(timeout=15)
    assert holder.returncode == 0, (out, err)

    store = TaskStore.load(tmp_path)
    assert [task.title for task in store.tasks] == ["holder task"]


def test_dead_pid_lock_is_reclaimed(tmp_path: Path) -> None:
    lock = TaskLock(tmp_path)
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text(json.dumps({"pid": 2_000_000_000}), encoding="utf-8")

    assert lock.acquire()
    lock.release()
    assert not lock.path.exists()


def test_aged_invalid_lock_is_reclaimed(tmp_path: Path) -> None:
    lock = TaskLock(tmp_path, stale_seconds=1)
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text("not json", encoding="utf-8")
    old = time.time() - 10
    os.utime(lock.path, (old, old))

    assert lock.acquire()
    lock.release()
    assert not lock.path.exists()


def test_released_lock_can_be_acquired_by_another_process(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    first = TaskLock(tmp_path)
    assert first.acquire()
    first.release()

    process = _python_process(
        "from pathlib import Path; import sys; from keeper.task_lock import TaskLock; "
        "lock=TaskLock(Path(sys.argv[1])); print(lock.acquire()); lock.release()",
        str(tmp_path),
        cwd=repo,
    )
    out, err = process.communicate(timeout=15)
    assert process.returncode == 0, err
    assert out.strip() == "True"
