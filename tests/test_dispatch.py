from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from keeper.context import WorkerBrief
from keeper.dispatch import ScopedWriteTool, _worker_registry, dispatch_fanout, dispatch_one
from keeper.projects import Project
from keeper.tools.dispatch import manager_dispatch_tools
from keeper.workers import worker_memory_path


def project(root: Path) -> Project:
    return Project("demo", "Demo", root, ("built-in", "codex", "pi"), "built-in")


class Approver:
    def __init__(self, answer=True, error=None): self.answer, self.error, self.requests = answer, error, []
    def confirm(self, request):
        self.requests.append(request)
        if self.error: raise self.error
        return self.answer


def codex(monkeypatch, code: str):
    monkeypatch.setenv("KEEPER_WORKER_CODEX_ARGV", json.dumps([sys.executable, "-c", code, "{input}"]))


def test_approval_denied_or_exception_never_launches(tmp_path, monkeypatch):
    codex(monkeypatch, "raise RuntimeError('launched')")
    brief = WorkerBrief("read", "do it", ("src",), ("docs",))
    approver = Approver(False)
    record = dispatch_one(project(tmp_path), brief, "codex", approver)
    assert record.result == "blocked" and "拒绝" in record.summary
    assert "src" in approver.requests[0].detail and "docs" in approver.requests[0].detail
    assert not list(tmp_path.rglob("*.log"))
    record = dispatch_one(project(tmp_path), brief, "codex", Approver(error=TimeoutError()))
    assert record.result == "blocked" and "审批失败" in record.summary


def test_cli_done_failed_empty_and_transcript(tmp_path, monkeypatch):
    brief = WorkerBrief("x", "do", ("src",))
    codex(monkeypatch, "from pathlib import Path; Path('src').mkdir(); Path('src/out.txt').write_text('ok'); print('ok')")
    ok = dispatch_one(project(tmp_path), brief, "codex", Approver())
    assert ok.result == "done" and ok.verified and ok.reply == "ok" and Path(ok.transcript).is_file()
    codex(monkeypatch, "import sys; print('bad', file=sys.stderr); sys.exit(7)")
    failed = dispatch_one(project(tmp_path), brief, "codex", Approver())
    assert failed.result == "failed" and "7" in failed.summary and "bad" in failed.summary
    codex(monkeypatch, "pass")
    empty = dispatch_one(project(tmp_path), brief, "codex", Approver())
    assert empty.result == "failed" and "为空" in empty.summary


def test_cli_success_without_allowed_changes_fails(tmp_path, monkeypatch):
    codex(monkeypatch, "print('I am done')")
    record = dispatch_one(project(tmp_path), WorkerBrief("x", "do", ("src",)), "codex", Approver())
    assert record.result == "failed"
    assert "声称完成但可写范围内无任何改动" in record.summary


def test_cli_read_only_completion_is_unverified(tmp_path, monkeypatch):
    codex(monkeypatch, "print('inspected')")
    record = dispatch_one(project(tmp_path), WorkerBrief("x", "inspect"), "codex", Approver())
    assert record.result == "done" and record.verified is False


def test_cli_timeout_interrupted(tmp_path, monkeypatch):
    codex(monkeypatch, "import time; time.sleep(2)")
    record = dispatch_one(project(tmp_path), WorkerBrief("x", "do"), "codex", Approver(), timeout=.05)
    assert record.result == "interrupted" and "超时" in record.summary and Path(record.transcript).is_file()


def test_external_worker_memory_violation_restores_and_fails(tmp_path, monkeypatch):
    target = tmp_path / ".agent" / "memory" / "DECISIONS.md"
    target.parent.mkdir(parents=True)
    target.write_text("before", encoding="utf-8")
    script = "from pathlib import Path; Path('.agent/memory/DECISIONS.md').write_text('after'); print('reply')"
    monkeypatch.setenv("KEEPER_WORKER_CODEX_ARGV", json.dumps([sys.executable, "-c", script, "{input}"]))
    record = dispatch_one(project(tmp_path), WorkerBrief("x", "do", ("src",)), "codex", Approver())
    assert record.result == "failed" and "违规写了项目记忆文件" in record.summary
    assert target.read_text(encoding="utf-8") == "before"
    assert Path(record.transcript).is_file()


def test_fanout_parallel_waits_for_all_and_preserves_order(tmp_path, monkeypatch):
    code = "import time,sys; from pathlib import Path; time.sleep(.25); Path('out').mkdir(exist_ok=True); Path('out/'+sys.argv[-1][:1]+'.txt').write_text('ok'); print('ok')"
    monkeypatch.setenv("KEEPER_WORKER_CODEX_ARGV", json.dumps([sys.executable,"-c",code,"{input}"]))
    monkeypatch.setenv("KEEPER_WORKER_PI_ARGV", json.dumps([sys.executable,"-c",code,"{input}"]))
    briefs = [WorkerBrief("a", "A"), WorkerBrief("b", "B")]
    start = time.monotonic()
    records = dispatch_fanout(project(tmp_path), briefs, ["codex", "pi"], Approver())
    elapsed = time.monotonic() - start
    assert [r.result for r in records] == ["done", "done"]
    assert elapsed < .7


@pytest.mark.parametrize("briefs,workers", [
    ([WorkerBrief("a","1")], ["codex"]),
    ([WorkerBrief(str(i),str(i)) for i in range(6)], ["codex"]*6),
    ([WorkerBrief("same","1"),WorkerBrief("SAME","2")], ["codex","pi"]),
])
def test_fanout_validates_count_and_labels(tmp_path, briefs, workers):
    with pytest.raises(ValueError): dispatch_fanout(project(tmp_path), briefs, workers, Approver())


def test_fanout_failure_does_not_cancel_sibling(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEPER_WORKER_CODEX_ARGV", json.dumps([sys.executable,"-c","import sys;sys.exit(3)","{input}"]))
    monkeypatch.setenv("KEEPER_WORKER_PI_ARGV", json.dumps([sys.executable,"-c","print('survived')","{input}"]))
    records = dispatch_fanout(project(tmp_path), [WorkerBrief("bad","x"),WorkerBrief("ok","x")],
                              ["codex","pi"], Approver())
    assert [r.result for r in records] == ["failed", "done"]


def test_builtin_registry_and_manager_only_dispatch_tools(tmp_path):
    names = _worker_registry(tmp_path, WorkerBrief("x", "do", ("src",)), "built-in").names()
    assert set(names) == {"read_file", "list_dir", "search", "write_file", "submit_result"}
    assert not {"dispatch_task", "dispatch_fanout", "shell", "update_memory",
                "request_memory_checkpoint", "complete_task", "block_task"} & set(names)
    assert {tool.name for tool in manager_dispatch_tools(project(tmp_path), Approver())} == {
        "dispatch_task", "dispatch_fanout"
    }


def test_manager_cli_requires_registered_project(tmp_path, monkeypatch, capsys):
    from keeper.cli import _manager_main
    monkeypatch.chdir(tmp_path)
    assert _manager_main([]) == 2
    assert "已注册项目" in capsys.readouterr().err


def test_scoped_writer_blocks_memory_escape_symlink_and_forbidden(tmp_path):
    brief = WorkerBrief("x", "do", ("src",), ("src/private",))
    tool = ScopedWriteTool(tmp_path, brief, "codex")
    paths = [".agent/memory/DECISIONS.md", "../outside.md",
             str(tmp_path / ".agent" / "memory" / "LESSONS.md"),
             "src/private/secret.txt", "outside/file.txt"]
    for path in paths:
        assert not tool.execute({"path": path, "content": "bad"}).ok, path
    memory = tmp_path / ".agent" / "memory"
    memory.mkdir(parents=True)
    (memory / "DECISIONS.md").write_text("keep", encoding="utf-8")
    assert not tool.execute({"path": ".agent/memory/DECISIONS.md", "content": "bad"}).ok
    private = worker_memory_path(tmp_path, "codex")
    result = tool.execute({"path": str(private), "content": "private note"})
    assert result.ok and private.read_text(encoding="utf-8") == "private note"
    (tmp_path / "src").mkdir()
    try:
        (tmp_path / "src" / "link").symlink_to(memory / "DECISIONS.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    assert not tool.execute({"path": "src/link", "content": "bad"}).ok
