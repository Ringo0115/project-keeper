"""M5.0 due 闸门、单飞锁、update_memory 互斥与 CLI 入口。"""

import json
import os
import socket
import time
from datetime import datetime, timedelta, timezone

import keeper.cli as cli
import keeper.memory as memory
import keeper.tidy as tidy
from keeper.config import Config
from keeper.llm import LLMResponse
from keeper.memory import FILE_NAMES, memory_path
from keeper.tidy import (
    TidyLock,
    check_due,
    format_check,
    lock_path,
    read_run_record,
    read_state,
    run_tidy,
    state_path,
    write_state,
)
from keeper.tools.memory import UpdateMemoryTool

NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)


def _mem(root):
    return root / ".agent" / "memory"


def _write(root, name: str, text: str) -> None:
    path = _mem(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _entry(title: str, entry_id: str, detail: str) -> str:
    return (
        f"- {title}\n"
        f"  <!--mem id:{entry_id} created:2026-01-01 touched:2026-01-01-->\n"
        f"  {detail}\n"
    )


def _seed_over_soft(root) -> None:
    _write(
        root,
        FILE_NAMES["decisions"],
        "# DECISIONS\n\n"
        + _entry("Alpha", "alpha", "a")
        + "\n"
        + _entry("Beta", "beta", "b"),
    )


def _small_budget(monkeypatch) -> None:
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "soft", 1)
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "hard", 2)


def _snapshot(root) -> dict[str, bytes]:
    directory = _mem(root)
    if not directory.exists():
        return {}
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class FakeLLM:
    def __init__(self, responses: list[dict] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=None):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses")
        return LLMResponse(message=self.responses.pop(0), usage={})


def _final(text: str) -> dict:
    return {"role": "assistant", "content": text}


def test_due_true_without_state_and_over_soft(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)

    decision = check_due(tmp_path, now=NOW)

    assert decision.due
    assert decision.over_soft == ["decisions"]
    assert decision.delta
    assert not decision.lock_busy
    assert decision.cooldown_remaining == 0
    assert "over soft" in decision.reason_text()


def test_no_delta_blocks_after_successful_state(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    assert write_state(
        tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW
    )

    decision = check_due(tmp_path, now=NOW + timedelta(minutes=31))

    assert not decision.due
    assert not decision.delta
    assert "no delta" in decision.reason_text()
    assert decision.cooldown_remaining == 0


def test_cooldown_blocks_then_expires(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW)
    # 内容再变一次，制造 delta；此时只剩冷却闸门。
    path = memory_path(tmp_path, FILE_NAMES["decisions"])
    path.write_bytes(path.read_bytes() + b"\n- Gamma\n")

    early = check_due(tmp_path, now=NOW + timedelta(minutes=5))
    assert not early.due
    assert early.delta
    assert "cooldown" in early.reason_text()

    late = check_due(tmp_path, now=NOW + timedelta(minutes=31))
    assert late.due
    assert late.cooldown_remaining == 0


def test_run_tidy_skips_without_llm_when_not_due(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW)
    llm = FakeLLM()

    report = run_tidy(tmp_path, llm, now=NOW + timedelta(minutes=5))

    assert report.outcome == "skipped"
    assert llm.calls == []
    assert "cooldown" in report.reason or "no delta" in report.reason


def test_force_runs_despite_cooldown_and_writes_state(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW)
    llm = FakeLLM([_final("no change")])

    report = run_tidy(
        tmp_path, llm, force=True, now=NOW + timedelta(minutes=5)
    )

    assert report.outcome in {"no_change", "index_only"}
    assert len(llm.calls) == 1
    state = read_state(tmp_path)
    assert state["last_attempt_at"].startswith("2026-11-23T12:05")
    assert report.backup_dir is not None


def test_lock_busy_blocks_due_and_force(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    lock = TidyLock(tmp_path)
    assert lock.acquire()
    try:
        decision = check_due(tmp_path, now=NOW)
        assert not decision.due
        assert decision.lock_busy
        assert "lock busy" in decision.reason_text()

        llm = FakeLLM()
        report = run_tidy(tmp_path, llm, force=True, now=NOW)
        assert report.outcome == "locked"
        assert llm.calls == []
    finally:
        lock.release()


def _write_lock(root, pid: int, host: str | None = None) -> None:
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pid": pid, "host": host or socket.gethostname()}),
        encoding="utf-8",
    )


def test_stale_lock_is_reclaimed_live_lock_is_not(tmp_path) -> None:
    _seed_over_soft(tmp_path)

    _write_lock(tmp_path, 999_999_999)
    dead = TidyLock(tmp_path)
    assert dead.acquire()
    dead.release()

    _write_lock(tmp_path, os.getpid())
    live = TidyLock(tmp_path)
    assert not live.acquire()
    lock_path(tmp_path).unlink()


def test_invalid_fresh_lock_is_busy_but_old_lock_is_stale(tmp_path) -> None:
    _seed_over_soft(tmp_path)
    path = lock_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")

    assert not TidyLock(tmp_path, stale_seconds=3600).acquire()

    old = time.time() - 10
    os.utime(path, (old, old))
    lock = TidyLock(tmp_path, stale_seconds=1)
    assert lock.acquire()
    lock.release()


def test_update_memory_is_blocked_by_tidy_lock(tmp_path) -> None:
    _seed_over_soft(tmp_path)
    path = memory_path(tmp_path, FILE_NAMES["decisions"])
    before = path.read_bytes()

    lock = TidyLock(tmp_path)
    assert lock.acquire()
    try:
        blocked = UpdateMemoryTool(tmp_path).execute(
            {"file": "decisions", "action": "add", "entry": "Gamma"}
        )
        assert not blocked.ok
        assert "being tidied" in blocked.content
        assert path.read_bytes() == before
    finally:
        lock.release()

    allowed = UpdateMemoryTool(tmp_path).execute(
        {"file": "decisions", "action": "add", "entry": "Gamma"}
    )
    assert allowed.ok, allowed.content
    assert "Gamma" in path.read_text(encoding="utf-8")


def test_check_due_and_format_check_are_read_only(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    before = _snapshot(tmp_path)

    decision = check_due(tmp_path, now=NOW)
    report = read_run_record(tmp_path)
    text = format_check(tmp_path, decision, report)

    assert "memory tidy: due=yes" in text
    assert "DECISIONS.md" in text
    assert "soft" in text and "hard" in text
    assert "lock: free" in text
    assert "no tidy run recorded" in text
    assert _snapshot(tmp_path) == before
    assert not state_path(tmp_path).exists()


def test_format_check_tolerates_old_run_record(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    decision = check_due(tmp_path, now=NOW)
    old_record = {
        "version": 1,
        "outcome": "no_change",
        "trigger": "manual",
        "ended_at": "2026-11-23T12:00:00+00:00",
        "tokens_before": {},
        "tokens_after": {},
        "entries_before": {},
        "entries_after": {},
    }

    text = format_check(tmp_path, decision, old_record)

    assert "outcome=no_change" in text
    assert "sessions: evidence=0 cited=0" in text


def test_cli_memory_check_needs_no_api_key(tmp_path, monkeypatch, capsys) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    before = _snapshot(tmp_path)

    rc = cli.main(["memory", "tidy", "--check", "--root", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "memory tidy: due=" in out
    assert _snapshot(tmp_path) == before


def test_cli_memory_tidy_with_fake_client(tmp_path, monkeypatch, capsys) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)

    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, **kwargs) -> None:
            self.calls: list[list[dict]] = []
            self.closed = False
            FakeClient.instances.append(self)

        def chat(self, messages, tools=None, temperature=None):
            self.calls.append(messages)
            return LLMResponse(
                message={"role": "assistant", "content": "no change"}, usage={}
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(cli, "LLMClient", FakeClient)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda **kwargs: Config(base_url="http://x", api_key="k", model="m"),
    )

    rc = cli.main(["memory", "tidy", "--root", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "tidy" in out
    assert FakeClient.instances
    assert FakeClient.instances[0].closed
    assert len(FakeClient.instances[0].calls) == 1


def test_cli_memory_tidy_requires_api_key(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda **kwargs: Config(base_url="http://x", api_key="", model="m"),
    )

    rc = cli.main(["memory", "tidy", "--root", str(tmp_path)])

    err = capsys.readouterr().err
    assert rc == 2
    assert "API key" in err


def test_tidy_module_exports_lock_and_state_paths(tmp_path) -> None:
    assert tidy.state_path(tmp_path).name == ".tidy-state.json"
    assert tidy.lock_path(tmp_path).name == ".tidy.lock"
    assert tidy.TIDY_COOLDOWN_SECONDS == 30 * 60