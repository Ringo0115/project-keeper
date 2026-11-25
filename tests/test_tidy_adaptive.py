"""M5.1b：自适应冷却（连续 no-change 只延长、不缩短基础闸门）。"""

import json
from datetime import datetime, timedelta, timezone

import keeper.memory as memory
import keeper.tidy as tidy
from keeper.llm import LLMResponse
from keeper.memory import FILE_NAMES, memory_path
from keeper.tidy import (
    TIDY_COOLDOWN_MAX_SECONDS,
    TIDY_COOLDOWN_SECONDS,
    TidyReport,
    check_due,
    effective_cooldown_seconds,
    format_check,
    read_state,
    run_tidy,
    state_path,
    write_state,
    write_state_after,
)

NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)


def _mem(root):
    return root / ".agent" / "memory"


def _write(root, name: str, text: str) -> None:
    path = _mem(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _seed_over_soft(root) -> None:
    _write(
        root,
        FILE_NAMES["decisions"],
        "# DECISIONS\n\n"
        "- Alpha\n"
        "  <!--mem id:alpha created:2026-01-01 touched:2026-01-01-->\n"
        "  detail\n",
    )


def _small_budget(monkeypatch) -> None:
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "soft", 1)
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "hard", 9999)


def _touch_content(root) -> None:
    path = memory_path(root, FILE_NAMES["decisions"])
    path.write_bytes(path.read_bytes() + b"\n- Gamma\n")


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0

    def chat(self, messages, tools=None, temperature=None):
        self.calls += 1
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses")
        return LLMResponse(message=self.responses.pop(0), usage={})


def _final(text: str) -> dict:
    return {"role": "assistant", "content": text}


def test_effective_cooldown_scales_and_caps() -> None:
    assert TIDY_COOLDOWN_SECONDS == 30 * 60
    assert effective_cooldown_seconds({}) == 1800
    assert effective_cooldown_seconds({"no_change_streak": 0}) == 1800
    assert effective_cooldown_seconds({"no_change_streak": 1}) == 3600
    assert effective_cooldown_seconds({"no_change_streak": 2}) == 7200
    assert effective_cooldown_seconds({"no_change_streak": 3}) == 14400
    assert effective_cooldown_seconds({"no_change_streak": 10}) == TIDY_COOLDOWN_MAX_SECONDS
    assert effective_cooldown_seconds({"no_change_streak": "bad"}) == 1800
    assert effective_cooldown_seconds({"no_change_streak": -2}) == 1800


def test_old_state_without_streak_keeps_base_cooldown(
    tmp_path, monkeypatch
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW)
    # 模拟旧版本的 state：没有 no_change_streak。
    path = state_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("no_change_streak", None)
    path.write_text(json.dumps(data), encoding="utf-8")
    _touch_content(tmp_path)

    decision = check_due(tmp_path, now=NOW + timedelta(minutes=31))

    assert decision.due
    assert decision.cooldown_seconds == 1800
    assert decision.no_change_streak == 0


def test_one_no_change_run_doubles_cooldown(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW)
    write_state(
        tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW,
        no_change_streak=1,
    )
    _touch_content(tmp_path)

    early = check_due(tmp_path, now=NOW + timedelta(minutes=31))
    assert not early.due
    assert "cooldown" in early.reason_text()
    assert early.cooldown_seconds == 3600

    late = check_due(tmp_path, now=NOW + timedelta(minutes=61))
    assert late.due


def test_cooldown_never_drops_below_base(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(
        tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW,
        no_change_streak=0,
    )
    _touch_content(tmp_path)

    decision = check_due(tmp_path, now=NOW + timedelta(minutes=5))

    assert not decision.due
    assert decision.cooldown_seconds == TIDY_COOLDOWN_SECONDS


def test_write_state_after_counts_no_change_and_resets_on_edit(
    tmp_path, monkeypatch
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)

    write_state_after(
        tmp_path, TidyReport(outcome="no_change", backup_dir="b"), now=NOW
    )
    assert read_state(tmp_path)["no_change_streak"] == 1

    write_state_after(
        tmp_path, TidyReport(outcome="no_change", backup_dir="b"), now=NOW
    )
    assert read_state(tmp_path)["no_change_streak"] == 2

    write_state_after(
        tmp_path, TidyReport(outcome="edited", backup_dir="b"), now=NOW
    )
    assert read_state(tmp_path)["no_change_streak"] == 0


def test_write_state_after_skipped_keeps_streak_and_no_backup_no_write(
    tmp_path,
) -> None:
    write_state(
        tmp_path,
        last_attempt_at=NOW,
        last_success_at=NOW,
        now=NOW,
        no_change_streak=3,
    )
    write_state_after(
        tmp_path, TidyReport(outcome="skipped", backup_dir=None), now=NOW
    )
    assert read_state(tmp_path)["no_change_streak"] == 3

    # 空项目 / 没备份的 no_change 不写 state（M5.0 行为）。
    path = state_path(tmp_path)
    before = path.read_bytes()
    assert not write_state_after(
        tmp_path, TidyReport(outcome="no_change", backup_dir=None), now=NOW
    )
    assert path.read_bytes() == before


def test_force_tidy_still_runs_under_long_cooldown(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(
        tmp_path,
        last_attempt_at=NOW,
        last_success_at=NOW,
        now=NOW,
        no_change_streak=3,
    )
    llm = FakeLLM([_final("no change")])

    report = run_tidy(tmp_path, llm, force=True, now=NOW + timedelta(minutes=1))

    assert report.outcome in {"no_change", "index_only"}
    assert llm.calls == 1


def test_format_check_reports_effective_cooldown(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    write_state(
        tmp_path,
        last_attempt_at=NOW,
        last_success_at=NOW,
        now=NOW,
        no_change_streak=2,
    )
    _touch_content(tmp_path)

    decision = check_due(tmp_path, now=NOW)
    text = format_check(tmp_path, decision, None)

    assert "cooldown: 120 min (no-change streak 2)" in text