from __future__ import annotations

from pathlib import Path

from keeper.llm import LLMResponse
from keeper.tidy import (
    RecordMemoryTool,
    TidyLock,
    TidyReport,
    _record_path,
    format_check,
    read_run_record,
    run_record_round,
    write_run_record,
    check_due,
)


def _call(name: str, args: dict, call_id: str = "c1") -> dict:
    import json

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []
        self.schemas = []

    def chat(self, messages, tools=None, temperature=None):
        self.messages.append([dict(item) for item in messages])
        self.schemas.append(tools)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(message=response)


def _seed(root: Path) -> Path:
    memory = root / ".agent" / "memory"
    memory.mkdir(parents=True)
    (memory / "DECISIONS.md").write_bytes(
        b"# Decisions\n\n- Keep the boundary\n  <!--mem id:boundary created:2026-01-01 touched:2026-01-01-->\n"
    )
    return memory


def test_record_round_is_labeled_backed_up_and_conserves_ids(tmp_path):
    memory = _seed(tmp_path)
    llm = FakeLLM([
        _call("record_write", {"path": "DECISIONS.md", "content": "# Decisions\n\n- Keep the boundary updated\n"}),
        {"role": "assistant", "content": "recorded"},
    ])

    report = run_record_round(tmp_path, llm, trigger="chat")

    assert report.kind == "record"
    assert report.outcome == "restored", llm.messages
    assert "mem id:boundary" in (memory / "DECISIONS.md").read_text(encoding="utf-8")
    assert Path(report.backup_dir).is_dir()
    stored = read_run_record(tmp_path)
    assert stored["kind"] == "record"
    assert all(schema["function"]["name"] in {"record_read", "record_write"} for schema in llm.schemas[0])
    output = format_check(tmp_path, check_due(tmp_path), stored)
    assert "last kind: record" in output
    write_run_record(tmp_path, TidyReport(kind="tidy", outcome="no_change"))
    stored = read_run_record(tmp_path)
    assert stored["kind"] == "tidy" and stored["record_round"]["kind"] == "record"
    assert "record round:" in format_check(tmp_path, check_due(tmp_path), stored)


def test_bad_path_is_rejected_and_model_can_continue(tmp_path):
    _seed(tmp_path)
    llm = FakeLLM([
        _call("record_write", {"path": "../DECISIONS.md", "content": "bad"}),
        {"role": "assistant", "content": "no change"},
    ])

    result = run_record_round(tmp_path, llm, trigger="manager")

    assert result.kind == "record"
    assert len(llm.messages) == 2
    assert llm.messages[1][-1]["content"].startswith("ERROR:")


def test_paths_reject_absolute_case_variants_and_symlinks(tmp_path):
    _seed(tmp_path)
    for bad in ("../DECISIONS.md", str(tmp_path / ".agent/memory/DECISIONS.md"),
                "decisions.md", ".agent/MEMORY/DECISIONS.md"):
        try:
            _record_path(tmp_path, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe path: {bad}")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / ".agent" / "memory" / "LESSONS.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        return
    tool = RecordMemoryTool(tmp_path, write=True)
    assert not tool.execute({"path": "LESSONS.md", "content": "overwrite"}).ok
    assert outside.read_text(encoding="utf-8") == "outside"


def test_failed_round_restores_exact_original_bytes(tmp_path):
    memory = _seed(tmp_path)
    path = memory / "DECISIONS.md"
    original = path.read_bytes()
    llm = FakeLLM([
        _call("record_write", {"path": "DECISIONS.md", "content": "# changed\n"}),
        RuntimeError("provider failed"),
    ])

    report = run_record_round(tmp_path, llm, trigger="chat")

    assert report.outcome == "failed"
    assert path.read_bytes() == original
    assert report.backup_dir and Path(report.backup_dir).is_dir()


def test_busy_lock_skips_without_touching_memory(tmp_path):
    memory = _seed(tmp_path)
    original = (memory / "DECISIONS.md").read_bytes()
    lock = TidyLock(tmp_path)
    assert lock.acquire()
    try:
        llm = FakeLLM([])
        report = run_record_round(tmp_path, llm, trigger="task")
    finally:
        lock.release()
    assert report.outcome == "locked"
    assert llm.messages == []
    assert (memory / "DECISIONS.md").read_bytes() == original


def test_unapproved_tool_name_is_rejected_without_crashing(tmp_path):
    _seed(tmp_path)
    llm = FakeLLM([
        _call("shell", {"command": "echo nope"}),
        {"role": "assistant", "content": "done"},
    ])
    report = run_record_round(tmp_path, llm, trigger="chat")
    assert report.outcome == "no_change"
    assert llm.messages[1][-1]["content"].startswith("ERROR:")
