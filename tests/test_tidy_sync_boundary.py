"""M5.1b 边界：同步整理遇到另一个 CLI 时的现有闸门。

跨进程只保证“整理与整理 / update_memory”互斥；它不阻止另一个仍有活动
会话的 CLI 继续调用 LLM。后台执行方案因缺少可靠、轻量的跨进程准入而
撤销，这里只验证同步路径的现有边界。
"""

import os
import subprocess
import sys
from pathlib import Path

from keeper.memory import FILE_NAMES
from keeper.tidy import (
    TidyLock,
    backup_memory,
    check_due,
    lock_path,
    read_run_record,
    state_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root, name: str, text: str) -> None:
    path = root / ".agent" / "memory" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _seed(root) -> None:
    _write(
        root,
        FILE_NAMES["decisions"],
        "# DECISIONS\n\n"
        "- Alpha\n"
        "  <!--mem id:alpha created:2026-01-01 touched:2026-01-01-->\n"
        "  detail\n",
    )


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["KEEPER_API_KEY"] = "dummy-key"  # 被锁拒绝时不会发起网络请求
    return env


def _run_cli(root, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "keeper", *args, "--root", str(root)],
        cwd=REPO_ROOT,
        env=_child_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


def test_check_in_another_process_sees_busy_lock(tmp_path) -> None:
    _seed(tmp_path)
    lock = TidyLock(tmp_path)
    assert lock.acquire()
    try:
        result = _run_cli(tmp_path, "memory", "tidy", "--check")
    finally:
        lock.release()

    assert result.returncode == 0
    assert "lock: busy" in result.stdout
    assert not state_path(tmp_path).exists()


def test_second_cli_tidy_is_refused_without_running(tmp_path) -> None:
    _seed(tmp_path)
    lock = TidyLock(tmp_path)
    assert lock.acquire()
    try:
        result = _run_cli(tmp_path, "memory", "tidy")
    finally:
        lock.release()

    assert result.returncode == 1
    assert "tidy locked" in result.stdout or "lock busy" in result.stderr
    # 没有备份 / run record / state；锁拒绝发生在任何整理写入之前。
    assert read_run_record(tmp_path) is None
    assert not (tmp_path / ".agent" / "memory" / "backups").exists()
    assert not state_path(tmp_path).exists()


def test_lock_busy_blocks_automatic_gate_without_llm(tmp_path, monkeypatch) -> None:
    import keeper.memory as memory
    from keeper.tidy import run_tidy

    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "soft", 1)
    _seed(tmp_path)

    class LLM:
        calls = 0

        def chat(self, messages, tools=None, temperature=None):
            self.calls += 1
            raise AssertionError("must not call LLM while another CLI holds the lock")

    lock = TidyLock(tmp_path)
    assert lock.acquire()
    try:
        decision = check_due(tmp_path)
        assert not decision.due and decision.lock_busy

        llm = LLM()
        report = run_tidy(tmp_path, llm)
        assert report.outcome == "skipped"
        assert llm.calls == 0
        forced = run_tidy(tmp_path, llm, force=True)
        assert forced.outcome == "locked"
        assert llm.calls == 0
    finally:
        lock.release()

    assert lock_path(tmp_path).exists() is False
    assert backup_memory(tmp_path).path is not None  # 锁释放后整理仍可备份