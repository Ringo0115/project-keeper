"""M5.0 切片 5：自动触发（启动提示、会话 / attempt 结束整理、开关）。"""

import builtins
from pathlib import Path

import keeper.cli as cli
import keeper.memory as memory
import keeper.tidy as tidy
from keeper.config import Config, load_config
from keeper.llm import LLMResponse
from keeper.memory import FILE_NAMES, memory_path
from keeper.projects import register_project
from keeper.tasks import TaskStore
from keeper.tidy import (
    RECORD_SYSTEM_PROMPT,
    TIDY_SYSTEM_PROMPT,
    lock_path,
    read_run_record,
    state_path,
)


def _mem(root: Path) -> Path:
    return root / ".agent" / "memory"


def _write(root: Path, name: str, text: str) -> None:
    path = _mem(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _entry(title: str, entry_id: str) -> str:
    return (
        f"- {title}\n"
        f"  <!--mem id:{entry_id} created:2026-01-01 touched:2026-01-01-->\n"
        "  detail\n"
    )


def _seed_over_soft(root: Path) -> None:
    _write(
        root,
        FILE_NAMES["decisions"],
        "# DECISIONS\n\n" + _entry("Alpha", "alpha"),
    )


def _small_budget(monkeypatch) -> None:
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "soft", 1)
    monkeypatch.setitem(memory.FILE_BUDGETS["decisions"], "hard", 999)


def _config(*, memory_auto: bool = True) -> Config:
    return Config(
        base_url="http://example.invalid",
        api_key="test-key",
        model="fake-model",
        max_steps=5,
        memory_auto=memory_auto,
    )


def _is_tidy(messages: list[dict]) -> bool:
    return bool(messages) and messages[0].get("content") == TIDY_SYSTEM_PROMPT


class FakeClient:
    """按 system prompt 区分会话调用与整理调用。"""

    instances: list["FakeClient"] = []

    def __init__(self, **kwargs) -> None:
        self.chat_calls = 0
        self.tidy_calls = 0
        self.record_calls = 0
        self.tidy_error: Exception | None = None
        self.tidy_messages: list[list[dict]] = []
        self.chat_responses: list[dict] = []
        self.closed = False
        FakeClient.instances.append(self)

    def chat(self, messages, tools=None, temperature=None):
        if messages and messages[0].get("content") == RECORD_SYSTEM_PROMPT:
            self.record_calls += 1
            return LLMResponse(
                message={"role": "assistant", "content": "record complete"}, usage={}
            )
        if _is_tidy(messages):
            self.tidy_calls += 1
            self.tidy_messages.append(messages)
            if self.tidy_error is not None:
                raise self.tidy_error
            return LLMResponse(
                message={"role": "assistant", "content": "no change"}, usage={}
            )
        self.chat_calls += 1
        message = (
            self.chat_responses.pop(0)
            if self.chat_responses
            else {"role": "assistant", "content": "done"}
        )
        return LLMResponse(message=message, usage={})

    def close(self) -> None:
        self.closed = True


def _patch_client(monkeypatch, *, memory_auto: bool = True) -> None:
    FakeClient.instances = []
    monkeypatch.setattr(cli, "LLMClient", FakeClient)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda **kwargs: _config(memory_auto=memory_auto),
    )


def test_first_chat_run_tidies_and_second_without_delta_runs_no_llm(
    tmp_path, monkeypatch, capsys
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    monkeypatch.setattr(tidy, "TIDY_COOLDOWN_SECONDS", 0)
    _patch_client(monkeypatch)

    assert cli.main(["hello", "--root", str(tmp_path)]) == 0
    first = FakeClient.instances[-1]
    assert first.chat_calls == 1
    assert first.record_calls == 1
    assert first.tidy_calls == 1
    assert state_path(tmp_path).is_file()
    record = read_run_record(tmp_path)
    assert record is not None and record["trigger"] == "chat"
    session_files = sorted(
        (tmp_path / ".agent" / "sessions").glob("*.jsonl")
    )
    assert len(session_files) == 1
    session_id = session_files[0].stem
    tidy_prompt = "\n".join(
        message.get("content", "") for message in first.tidy_messages[0]
    )
    assert "RECENT SESSIONS" in tidy_prompt
    assert session_id in tidy_prompt
    assert record["evidence_sessions"] == [session_id]
    err = capsys.readouterr().err
    assert "[memory] tidy due" in err
    assert "[memory] tidy" in err

    # 状态已记录当前指纹；内容没变时第二次只跳过，不调用整理 LLM。
    assert cli.main(["hello", "--root", str(tmp_path)]) == 0
    second = FakeClient.instances[-1]
    assert second.chat_calls == 1
    assert second.record_calls == 1
    assert second.tidy_calls == 0


def test_startup_hint_is_read_only_and_calls_no_llm(
    tmp_path, monkeypatch, capsys
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    _patch_client(monkeypatch)
    before = {
        path.relative_to(_mem(tmp_path)).as_posix(): path.read_bytes()
        for path in _mem(tmp_path).rglob("*")
        if path.is_file()
    }
    # 只验证启动阶段：把 REPL 换成"非 clean 退出"，避免结束整理干扰。
    monkeypatch.setattr(cli, "_repl", lambda loop, session: (0, False))

    assert cli.main(["--root", str(tmp_path)]) == 0

    client = FakeClient.instances[-1]
    assert client.chat_calls == 0
    assert client.record_calls == 0
    assert client.tidy_calls == 0
    err = capsys.readouterr().err
    assert "[memory] tidy due: over soft: DECISIONS.md" in err
    after = {
        path.relative_to(_mem(tmp_path)).as_posix(): path.read_bytes()
        for path in _mem(tmp_path).rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not state_path(tmp_path).exists()
    assert not lock_path(tmp_path).exists()
    assert not (_mem(tmp_path) / "backups").exists()


def test_no_tidy_flag_suppresses_hint_and_pass(
    tmp_path, monkeypatch, capsys
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    _patch_client(monkeypatch)

    assert cli.main(["hello", "--root", str(tmp_path), "--no-tidy"]) == 0

    client = FakeClient.instances[-1]
    assert client.chat_calls == 1
    assert client.record_calls == 0
    assert client.tidy_calls == 0
    assert "[memory]" not in capsys.readouterr().err
    assert not state_path(tmp_path).exists()
    assert not (_mem(tmp_path) / "backups").exists()


def test_memory_auto_disabled_suppresses_hint_and_pass(
    tmp_path, monkeypatch, capsys
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    _patch_client(monkeypatch, memory_auto=False)

    assert cli.main(["hello", "--root", str(tmp_path)]) == 0

    client = FakeClient.instances[-1]
    assert client.chat_calls == 1
    assert client.record_calls == 0
    assert client.tidy_calls == 0
    assert "[memory]" not in capsys.readouterr().err
    assert not state_path(tmp_path).exists()


def test_load_config_reads_keeper_memory_auto(monkeypatch) -> None:
    for value in ("0", "false", "off", "no", "FALSE"):
        monkeypatch.setenv("KEEPER_MEMORY_AUTO", value)
        assert load_config().memory_auto is False

    for value in ("1", "true", "on", "yes"):
        monkeypatch.setenv("KEEPER_MEMORY_AUTO", value)
        assert load_config().memory_auto is True

    monkeypatch.setenv("KEEPER_MEMORY_AUTO", "unexpected")
    assert load_config().memory_auto is True
    monkeypatch.delenv("KEEPER_MEMORY_AUTO")
    assert load_config().memory_auto is True


def test_tidy_failure_keeps_exit_code_and_only_warns(
    tmp_path, monkeypatch, capsys
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    _patch_client(monkeypatch)

    original = FakeClient.__init__

    def with_error(self, **kwargs) -> None:
        original(self, **kwargs)
        self.tidy_error = RuntimeError("provider down")

    monkeypatch.setattr(FakeClient, "__init__", with_error)

    assert cli.main(["hello", "--root", str(tmp_path)]) == 0

    client = FakeClient.instances[-1]
    assert client.chat_calls == 1
    assert client.tidy_calls == 1
    err = capsys.readouterr().err
    assert "[memory] tidy warning:" in err
    record = read_run_record(tmp_path)
    assert record is not None and record["outcome"] == "failed"


def test_repl_reports_clean_exit_only_for_exit_and_eof(monkeypatch) -> None:
    class Loop:
        def run(self, text):
            return iter(())

    class Session:
        path = Path("session.jsonl")

    monkeypatch.setattr(builtins, "input", lambda prompt="": "/exit")
    assert cli._repl(Loop(), Session()) == (0, True)

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert cli._repl(Loop(), Session()) == (0, True)

    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupt)
    assert cli._repl(Loop(), Session()) == (0, False)


def test_repl_clean_exit_triggers_tidy_but_interrupt_does_not(
    tmp_path, monkeypatch
) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    _patch_client(monkeypatch)

    monkeypatch.setattr(cli, "_repl", lambda loop, session: (0, True))
    assert cli.main(["--root", str(tmp_path)]) == 0
    assert FakeClient.instances[-1].tidy_calls == 1
    assert FakeClient.instances[-1].record_calls == 1

    monkeypatch.setattr(cli, "_repl", lambda loop, session: (0, False))
    assert cli.main(["--root", str(tmp_path)]) == 0
    assert FakeClient.instances[-1].tidy_calls == 0
    assert FakeClient.instances[-1].record_calls == 0


def test_task_attempt_end_triggers_tidy(tmp_path, monkeypatch) -> None:
    _small_budget(monkeypatch)
    _seed_over_soft(tmp_path)
    store = TaskStore.load(tmp_path)
    task = store.add("跑一次自动整理")
    _patch_client(monkeypatch)

    original = FakeClient.__init__

    def with_complete(self, **kwargs) -> None:
        original(self, **kwargs)
        self.chat_responses = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "complete_task", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "完成了"},
        ]

    monkeypatch.setattr(FakeClient, "__init__", with_complete)

    assert cli.main(["task", "run", task.id, "--root", str(tmp_path)]) == 0

    reloaded = TaskStore.load(tmp_path).get(task.id)
    assert reloaded.status == "done"
    client = FakeClient.instances[-1]
    assert client.tidy_calls == 1
    assert client.record_calls == 1
    record = read_run_record(tmp_path)
    assert record is not None and record["trigger"] == "task"
    session_files = sorted(
        (tmp_path / ".agent" / "sessions").glob("*.jsonl")
    )
    assert len(session_files) == 1
    session_id = session_files[0].stem
    tidy_prompt = "\n".join(
        message.get("content", "") for message in client.tidy_messages[0]
    )
    assert session_id in tidy_prompt
    assert record["evidence_sessions"] == [session_id]


def test_manager_session_end_runs_record_once(tmp_path, monkeypatch) -> None:
    register_project(tmp_path, project_id="manager-demo")
    _patch_client(monkeypatch)

    assert cli.main(["manager", "review this project", "--root", str(tmp_path)]) == 0

    client = FakeClient.instances[-1]
    assert client.chat_calls == 1
    assert client.record_calls == 1
    assert client.tidy_calls == 0
    record = read_run_record(tmp_path)
    assert record is not None and record["kind"] == "record"
    assert record["trigger"] == "manager"
