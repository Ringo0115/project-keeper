from __future__ import annotations

import json
import asyncio
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

from keeper.context import WorkerBrief
from keeper.dispatch import ScopedWriteTool, _cli_argv, _worker_registry, dispatch_fanout, dispatch_one
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


def test_cli_timeout_reports_outside_allowed_changes(tmp_path, monkeypatch):
    codex(monkeypatch, "import time; from pathlib import Path; Path('stray.txt').write_text('x'); time.sleep(5)")
    record = dispatch_one(project(tmp_path), WorkerBrief("x", "do", ("src",)), "codex", Approver(), timeout=1.0)
    assert record.result == "interrupted" and "范围外" in record.summary
    assert (tmp_path / "stray.txt").read_text() == "x"


def test_cli_timeout_interrupted(tmp_path, monkeypatch):
    codex(monkeypatch, "import time; time.sleep(2)")
    record = dispatch_one(project(tmp_path), WorkerBrief("x", "do"), "codex", Approver(), timeout=.05)
    assert record.result == "interrupted" and "超时" in record.summary and Path(record.transcript).is_file()


def test_cli_timeout_keeps_already_printed_output(tmp_path, monkeypatch):
    # 先打印再睡死：超时杀树后，文件里已写出的 stdout/stderr 必须留在 transcript。
    code = ("import sys, time; print('TIMEOUT-STDOUT-MARKER', flush=True); "
            "print('TIMEOUT-STDERR-MARKER', file=sys.stderr, flush=True); time.sleep(5)")
    codex(monkeypatch, code)
    record = dispatch_one(project(tmp_path), WorkerBrief("slow", "do"), "codex", Approver(), timeout=1.0)
    assert record.result == "interrupted" and "超时" in record.summary
    assert "TIMEOUT-STDOUT-MARKER" in record.reply
    transcript = Path(record.transcript).read_text(encoding="utf-8")
    stdout_at = transcript.index("TIMEOUT-STDOUT-MARKER")
    stderr_marker_at = transcript.index("TIMEOUT-STDERR-MARKER")
    # stdout 落在 STDOUT 段、stderr 落在 STDERR 段——两个文件不能被合并。
    assert transcript.index("STDOUT:") < stdout_at < transcript.index("STDERR:") < stderr_marker_at


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


def test_cli_argv_resolves_existing_executable(monkeypatch):
    monkeypatch.setenv("KEEPER_WORKER_CODEX_ARGV", json.dumps([sys.executable, "-c", "print(1)", "{input}"]))
    argv = _cli_argv("codex", "hello")
    assert argv[0] == shutil.which(sys.executable) and Path(argv[0]).is_file()
    assert argv[1:] == ["-c", "print(1)", "hello"]


def test_cli_argv_rejects_missing_executable(monkeypatch):
    monkeypatch.setenv("KEEPER_WORKER_CODEX_ARGV", json.dumps(["keeper-no-such-cli-xyz", "{input}"]))
    with pytest.raises(ValueError) as exc:
        _cli_argv("codex", "hello")
    assert "keeper-no-such-cli-xyz" in str(exc.value)
    assert "KEEPER_WORKER_CODEX_ARGV" in str(exc.value)


def test_cli_argv_resolves_env_argv0(monkeypatch):
    monkeypatch.setenv("KEEPER_WORKER_PI_ARGV", json.dumps([sys.executable, "{input}"]))
    argv = _cli_argv("pi", "task text")
    assert argv == [shutil.which(sys.executable), "task text"]


def _install_fake_claude_sdk(monkeypatch, *, pid=12345, slow=False):
    module = types.ModuleType("claude_agent_sdk")
    captured = {}
    events = []

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class PermissionResultAllow:
        pass

    class PermissionResultDeny:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self, *, options):
            captured["client_options"] = options
            self._transport = types.SimpleNamespace(_process=types.SimpleNamespace(pid=pid))

        async def connect(self):
            events.append("connect")

        async def query(self, prompt):
            captured["prompt"] = prompt

        async def receive_response(self):
            if slow:
                # 先吐一块文本再睡死：超时前已收到的内容必须能被带出去。
                yield types.SimpleNamespace(
                    content=[types.SimpleNamespace(text="sdk partial")], is_error=False)
                await asyncio.sleep(1)
                return
            yield types.SimpleNamespace(content=[types.SimpleNamespace(text="sdk reply")], is_error=False)

        async def disconnect(self):
            events.append("disconnect")

    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.ClaudeSDKClient = FakeClient
    module.PermissionResultAllow = PermissionResultAllow
    module.PermissionResultDeny = PermissionResultDeny
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    monkeypatch.setattr("keeper.claude_sdk.native_claude_cli", lambda _path: r"C:\fake\claude.exe")
    return captured, events


def test_claude_sdk_normal_path_uses_default_permissions_and_native_cli(tmp_path, monkeypatch):
    captured, _events = _install_fake_claude_sdk(monkeypatch)
    monkeypatch.setenv("KEEPER_WORKER_CLAUDE_CODE_ARGV", json.dumps([sys.executable, "{input}"]))
    brief = WorkerBrief("sdk", "do", ("src",))
    record = dispatch_one(project(tmp_path), brief, "claude-code", Approver())
    assert record.result == "failed"  # no allowed-scope change was made
    assert captured["permission_mode"] == "default"
    assert captured["cli_path"].endswith(".exe") and not captured["cli_path"].endswith(".cmd")
    assert captured["env"]["CLAUDECODE"] == ""
    assert captured["prompt"]


def test_claude_sdk_missing_pid_falls_back_to_popen(tmp_path, monkeypatch):
    captured, _events = _install_fake_claude_sdk(monkeypatch)
    fake_module = sys.modules["claude_agent_sdk"]
    original_client = fake_module.ClaudeSDKClient

    class NoPidClient(original_client):
        def __init__(self, *, options):
            super().__init__(options=options)
            class BrokenProcess:
                @property
                def pid(self):
                    raise RuntimeError("pid unavailable")
            self._transport._process = BrokenProcess()

    fake_module.ClaudeSDKClient = NoPidClient
    code = "from pathlib import Path; Path('src').mkdir(); Path('src/fallback.txt').write_text('ok'); print('popen reply')"
    monkeypatch.setenv("KEEPER_WORKER_CLAUDE_CODE_ARGV", json.dumps([sys.executable, "-c", code, "{input}"]))
    record = dispatch_one(project(tmp_path), WorkerBrief("fallback", "do", ("src",)), "claude-code", Approver())
    assert record.result == "done" and record.verified and record.reply == "popen reply"
    assert (tmp_path / "src" / "fallback.txt").read_text() == "ok"


def test_claude_sdk_import_failure_falls_back_to_popen(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    code = "from pathlib import Path; Path('src').mkdir(); Path('src/fallback.txt').write_text('ok'); print('popen reply')"
    monkeypatch.setenv("KEEPER_WORKER_CLAUDE_CODE_ARGV", json.dumps([sys.executable, "-c", code, "{input}"]))
    record = dispatch_one(project(tmp_path), WorkerBrief("fallback", "do", ("src",)), "claude-code", Approver())
    assert record.result == "done" and record.verified and record.reply == "popen reply"


def test_claude_sdk_timeout_kills_before_disconnect(tmp_path, monkeypatch):
    _captured, events = _install_fake_claude_sdk(monkeypatch, slow=True)
    monkeypatch.setenv("KEEPER_WORKER_CLAUDE_CODE_ARGV", json.dumps([sys.executable, "{input}"]))
    monkeypatch.setattr("keeper.dispatch._kill_sdk_pid", lambda _pid: events.append("kill"))
    record = dispatch_one(project(tmp_path), WorkerBrief("timeout", "do"), "claude-code", Approver(), timeout=.01)
    assert record.result == "interrupted" and "0.01s" in record.summary
    assert events.index("kill") < events.index("disconnect")


def test_claude_sdk_timeout_keeps_partial_text_in_transcript(tmp_path, monkeypatch):
    _install_fake_claude_sdk(monkeypatch, slow=True)
    monkeypatch.setenv("KEEPER_WORKER_CLAUDE_CODE_ARGV", json.dumps([sys.executable, "{input}"]))
    monkeypatch.setattr("keeper.dispatch._kill_sdk_pid", lambda _pid: None)
    record = dispatch_one(project(tmp_path), WorkerBrief("timeout", "do"), "claude-code",
                          Approver(), timeout=.05)
    assert record.result == "interrupted"
    assert "sdk partial" in record.reply
    assert "sdk partial" in Path(record.transcript).read_text(encoding="utf-8")


def test_codex_default_template_enables_workspace_write(monkeypatch):
    monkeypatch.delenv("KEEPER_WORKER_CODEX_ARGV", raising=False)
    monkeypatch.setattr("keeper.dispatch.shutil.which", lambda _name: sys.executable)
    argv = _cli_argv("codex", "x")
    assert argv[1:] == ["exec", "-", "-s", "workspace-write"]


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
