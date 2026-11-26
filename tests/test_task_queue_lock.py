"""任务队列跨进程锁：真起 python 进程验证互斥、释放、陈旧锁与双 running。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_REL = Path(".agent") / "tasks.lock"

# 抢到锁后停在事件文件上，模拟「另一个进程正在写队列」。
HOLDER = """
import sys, time
from pathlib import Path
from keeper.file_lock import FileLock

root = Path(sys.argv[1])
ready = Path(sys.argv[2])
stop = Path(sys.argv[3])
lock = FileLock(root / ".agent" / "tasks.lock")
if not lock.acquire():
    print("BUSY", flush=True)
    sys.exit(9)
ready.write_text("ready", encoding="utf-8")
deadline = time.time() + 60
while time.time() < deadline and not stop.exists():
    time.sleep(0.02)
lock.release()
"""

# 先加载旧快照，等信号后再尝试标记 running——这正是两个终端同时起跑的形状。
LOADER = """
import sys, time
from pathlib import Path
from keeper.tasks import TaskStore, TaskStoreError

root = Path(sys.argv[1])
task_id = sys.argv[2]
loaded = Path(sys.argv[3])
go = Path(sys.argv[4])
store = TaskStore.load(root)
loaded.write_text("loaded", encoding="utf-8")
deadline = time.time() + 60
while time.time() < deadline and not go.exists():
    time.sleep(0.02)
try:
    store.start_attempt(task_id, None)
except TaskStoreError as exc:
    print(f"ERROR: {exc}", flush=True)
    sys.exit(3)
print("STARTED", flush=True)
"""

MARKER = """
import sys
from pathlib import Path
from keeper.tasks import TaskStore

root = Path(sys.argv[1])
task_id = sys.argv[2]
TaskStore.load(root).start_attempt(task_id, None)
"""


def _run_cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "keeper", *args, "--root", str(root)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=60,
    )


def _spawn(script: str, *args: Path | str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", script, *map(str, args)],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for(path: Path, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {path}")


def _titles(root: Path) -> list[str]:
    path = root / ".agent" / "tasks.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [task["title"] for task in data["tasks"]]


def _start_holder(tmp_path: Path) -> tuple[subprocess.Popen[str], Path, Path]:
    ready = tmp_path / "holder-ready"
    stop = tmp_path / "holder-stop"
    holder = _spawn(HOLDER, tmp_path, ready, stop)
    _wait_for(ready)
    return holder, ready, stop


def _stop_holder(holder: subprocess.Popen[str], stop: Path) -> None:
    stop.write_text("stop", encoding="utf-8")
    holder.communicate(timeout=30)


def test_second_add_fails_while_lock_held(tmp_path: Path) -> None:
    first = _run_cli(tmp_path, "task", "add", "任务A")
    assert first.returncode == 0, first.stderr

    holder, _, stop = _start_holder(tmp_path)
    try:
        second = _run_cli(tmp_path, "task", "add", "任务B")
        assert second.returncode != 0
        assert str((tmp_path / LOCK_REL).resolve()) in second.stderr
        assert "另一个进程" in second.stderr
        assert _titles(tmp_path) == ["任务A"]
    finally:
        _stop_holder(holder, stop)


def test_released_lock_can_be_acquired_by_another_process(tmp_path: Path) -> None:
    holder, _, stop = _start_holder(tmp_path)
    _stop_holder(holder, stop)
    assert holder.returncode == 0

    result = _run_cli(tmp_path, "task", "add", "锁释放后")
    assert result.returncode == 0, result.stderr
    assert _titles(tmp_path) == ["锁释放后"]


def test_stale_empty_lock_file_does_not_block(tmp_path: Path) -> None:
    lock_path = tmp_path / LOCK_REL
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("", encoding="utf-8")

    result = _run_cli(tmp_path, "task", "add", "陈旧锁之后")
    assert result.returncode == 0, result.stderr
    assert _titles(tmp_path) == ["陈旧锁之后"]


def test_two_processes_cannot_mark_running(tmp_path: Path) -> None:
    from keeper.tasks import TaskStore

    store = TaskStore.load(tmp_path)
    first = store.add("先跑")
    second = store.add("后跑")

    loaded = tmp_path / "loader-loaded"
    go = tmp_path / "loader-go"
    loader = _spawn(LOADER, tmp_path, second.id, loaded, go)
    try:
        _wait_for(loaded)  # 进程 B 已拿着「还没人 running」的旧快照
        marked = subprocess.run(
            [sys.executable, "-c", MARKER, str(tmp_path), first.id],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert marked.returncode == 0, marked.stderr
        go.write_text("go", encoding="utf-8")
        out, err = loader.communicate(timeout=30)
    finally:
        if loader.poll() is None:
            loader.kill()
            loader.communicate()

    assert loader.returncode == 3, f"stdout={out!r} stderr={err!r}"
    assert "已有 running" in out
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text(encoding="utf-8"))
    running = [task["id"] for task in data["tasks"] if task["status"] == "running"]
    assert running == [first.id]
    queued = [task["id"] for task in data["tasks"] if task["status"] == "queued"]
    assert queued == [second.id]