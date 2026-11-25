"""M5.1b：agent 主动 checkpoint（只请求 / 标记，不直接整理）。"""

import json
from datetime import datetime, timedelta, timezone

import keeper.memory as memory
import keeper.tidy as tidy
from keeper.approval import Decision, review
from keeper.checkpoint import (
    CHECKPOINT_NAME,
    clear_checkpoint_request,
    read_checkpoint_request,
    request_checkpoint,
)
from keeper.llm import LLMResponse
from keeper.memory import FILE_NAMES, memory_path
from keeper.tidy import (
    TIDY_SYSTEM_PROMPT,
    check_due,
    format_check,
    lock_path,
    read_run_record,
    run_tidy,
    state_path,
)
from keeper.tools.checkpoint import RequestCheckpointTool

NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)


def _mem(root):
    return root / ".agent" / "memory"


def _write(root, name: str, text: str) -> None:
    path = _mem(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _entry(title: str, entry_id: str) -> str:
    return (
        f"- {title}\n"
        f"  <!--mem id:{entry_id} created:2026-01-01 touched:2026-01-01-->\n"
        "  detail\n"
    )


def _seed_over_soft(root) -> None:
    _write(
        root,
        FILE_NAMES["decisions"],
        "# DECISIONS\n\n" + _entry("Alpha", "alpha"),
    )


def _small_budget(monkeypatch) -> None:
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "soft", 1)
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "hard", 9999)


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
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=None):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses")
        return LLMResponse(message=self.responses.pop(0), usage={})


def _final(text: str) -> dict:
    return {"role": "assistant", "content": text}


def test_checkpoint_tool_only_writes_marker(tmp_path) -> None:
    _seed_over_soft(tmp_path)
    before = _snapshot(tmp_path)

    result = RequestCheckpointTool(tmp_path).execute(
        {"reason": "line one\nline two"}
    )

    assert result.ok
    assert "next CLI boundary" in result.content
    data = json.loads(
        memory_path(tmp_path, CHECKPOINT_NAME).read_text(encoding="utf-8")
    )
    assert data["version"] == 1
    assert data["reason"] == "line one line two"
    assert data["requested_at"]
    # 只多了一个标记文件；记忆、锁、备份、state 均未变化。
    after = _snapshot(tmp_path)
    after.pop(CHECKPOINT_NAME)
    assert after == before
    assert not state_path(tmp_path).exists()
    assert not lock_path(tmp_path).exists()
    assert not (_mem(tmp_path) / "backups").exists()


def test_checkpoint_tool_is_allowed_without_prompt(tmp_path) -> None:
    verdict = review(
        tool_name="request_memory_checkpoint",
        danger="write",
        args={"reason": "x"},
        root=tmp_path,
    )
    assert verdict.decision is Decision.ALLOW


def test_checkpoint_reason_is_truncated_and_single_line(tmp_path) -> None:
    request = request_checkpoint(tmp_path, "  a\n\nb  " + "x" * 400)
    assert "\n" not in request.reason
    assert len(request.reason) <= 200
    assert read_checkpoint_request(tmp_path) == request


def test_checkpoint_marker_tolerates_corruption(tmp_path) -> None:
    _write(tmp_path, CHECKPOINT_NAME, "not json")
    assert read_checkpoint_request(tmp_path) is None

    _write(tmp_path, CHECKPOINT_NAME, json.dumps({"reason": 5}))
    request = read_checkpoint_request(tmp_path)
    assert request is not None and request.reason == ""

    assert clear_checkpoint_request(tmp_path)
    assert read_checkpoint_request(tmp_path) is None


def test_run_tidy_consumes_checkpoint_and_records_reason(
    tmp_path, monkeypatch
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    request_checkpoint(tmp_path, "本次会话新增了决策")
    llm = FakeLLM([_final("no change")])

    report = run_tidy(tmp_path, llm, now=NOW, trigger="chat")

    assert report.checkpoint_requested
    prompt = "\n".join(
        message.get("content", "") for message in llm.calls[0]
    )
    assert "本次会话新增了决策" in prompt
    assert read_checkpoint_request(tmp_path) is None

    record = read_run_record(tmp_path)
    assert record is not None
    assert record["checkpoint_requested"] is True
    assert record["checkpoint_reason"] == "本次会话新增了决策"


def test_checkpoint_is_not_a_gate_bypass(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    tidy.write_state(
        tmp_path, last_attempt_at=NOW, last_success_at=NOW, now=NOW
    )
    request_checkpoint(tmp_path, "please tidy")
    llm = FakeLLM()

    report = run_tidy(tmp_path, llm, now=NOW + timedelta(minutes=5))

    assert report.outcome == "skipped"
    assert llm.calls == []
    # 没真正跑整理，请求保留给下一次边界。
    assert read_checkpoint_request(tmp_path) is not None


def test_failed_pass_keeps_checkpoint_for_retry(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    request_checkpoint(tmp_path, "retry me")
    llm = FakeLLM(error=RuntimeError("provider down"))

    report = run_tidy(tmp_path, llm, now=NOW)

    assert report.outcome == "failed"
    assert read_checkpoint_request(tmp_path) is not None


def test_manual_force_tidy_also_consumes_checkpoint(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    request_checkpoint(tmp_path, "manual")
    llm = FakeLLM([_final("no change")])

    report = run_tidy(tmp_path, llm, force=True, now=NOW)

    assert report.checkpoint_requested
    assert read_checkpoint_request(tmp_path) is None


def test_format_check_shows_pending_checkpoint_read_only(
    tmp_path, monkeypatch
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    request_checkpoint(tmp_path, "pending")
    before = _snapshot(tmp_path)

    decision = check_due(tmp_path, now=NOW)
    text = format_check(tmp_path, decision, read_run_record(tmp_path))

    assert "checkpoint: requested" in text
    assert _snapshot(tmp_path) == before


def test_checkpoint_written_during_pass_is_not_lost(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    request_checkpoint(tmp_path, "first")

    class MidPassLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, temperature=None):
            self.calls += 1
            # 模拟整理期间另一个会话写入新的 checkpoint。
            request_checkpoint(tmp_path, "second")
            return LLMResponse(
                message={"role": "assistant", "content": "no change"}, usage={}
            )

    llm = MidPassLLM()
    report = run_tidy(tmp_path, llm, now=NOW, trigger="chat")

    assert llm.calls == 1
    assert report.checkpoint_requested
    assert report.checkpoint_reason == "first"
    pending = read_checkpoint_request(tmp_path)
    assert pending is not None and pending.reason == "second"
    record = read_run_record(tmp_path)
    assert record["checkpoint_reason"] == "first"


def test_consume_checkpoint_is_conditional(tmp_path) -> None:
    from keeper.checkpoint import consume_checkpoint_request

    request = request_checkpoint(tmp_path, "first")
    assert consume_checkpoint_request(tmp_path, request) is True
    assert read_checkpoint_request(tmp_path) is None

    request = request_checkpoint(tmp_path, "first")
    newer = request_checkpoint(tmp_path, "second")
    assert consume_checkpoint_request(tmp_path, request) is False
    assert read_checkpoint_request(tmp_path) == newer


def test_checkpoint_message_is_not_an_archiving_authorization(tmp_path) -> None:
    from keeper.tidy import build_tidy_message, read_snapshot

    _seed_over_soft(tmp_path)
    message = build_tidy_message(
        read_snapshot(tmp_path),
        trigger="chat",
        today="2026-11-23",
        checkpoint_reason="line one\nline two",
    )

    assert "整理员 checkpoint 请求：line one line two" in message
    assert "不等于归档授权" in message
    # 整理员规则仍然保守：请求本身不解除“证据不足就保留”。
    assert "证据不足就保留" in TIDY_SYSTEM_PROMPT