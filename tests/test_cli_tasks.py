from pathlib import Path

from keeper.cli import main
from keeper.config import Config
from keeper.llm import LLMResponse
from keeper.session import Session
from keeper.tasks import TASKS_FILE, TaskStore


def _patch_fake_llm(monkeypatch) -> None:
    """让 task run / --resume 不碰网络：先调 complete_task，再给文本。"""
    monkeypatch.setattr(
        "keeper.cli.load_config",
        lambda **kwargs: Config(
            base_url="http://example.invalid",
            api_key="test-key",
            model="fake-model",
            max_steps=5,
        ),
    )

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.responses = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "complete_task",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "完成了"},
            ]

        def chat(self, messages, tools=None) -> LLMResponse:
            return LLMResponse(message=self.responses.pop(0), usage={})

        def close(self) -> None:
            pass

    monkeypatch.setattr("keeper.cli.LLMClient", FakeClient)


def test_cli_add_and_list(tmp_path: Path, capsys) -> None:
    assert main(["task", "add", "修复 login 测试", "--root", str(tmp_path)]) == 0
    added = capsys.readouterr()
    assert "status=queued" in added.err
    assert "修复 login 测试" in added.err
    # id 也走 stdout，方便脚本取。
    assert added.out.strip().startswith("t-")

    store = TaskStore.load(tmp_path)
    assert len(store.tasks) == 1
    task_id = store.tasks[0].id
    assert added.out.strip() == task_id

    assert main(["task", "list", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert task_id in out
    assert "queued" in out
    assert "attempts=0" in out
    assert "last=-" in out
    assert "修复 login 测试" in out


def test_cli_add_rejects_empty_title(tmp_path: Path, capsys) -> None:
    assert main(["task", "add", "   ", "--root", str(tmp_path)]) == 2

    err = capsys.readouterr().err
    assert "不能为空" in err
    assert not (tmp_path / TASKS_FILE).exists()


def test_cli_list_reports_corrupt_file_and_does_not_overwrite(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / TASKS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    before = path.read_bytes()

    assert main(["task", "list", "--root", str(tmp_path)]) == 2

    err = capsys.readouterr().err
    assert "损坏" in err
    assert path.read_bytes() == before


def test_cli_list_recovers_interrupted_task(tmp_path: Path, capsys) -> None:
    store = TaskStore.load(tmp_path)
    front = store.add("前排任务")
    target = store.add("被 kill 的任务")
    store.start_attempt(target.id, "session-killed")

    assert main(["task", "list", "--root", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "[task] recover" in captured.err
    assert "interrupted" in captured.err

    # 队首任务被恢复成 queued，并排到 index 0。
    reloaded = TaskStore.load(tmp_path)
    assert [task.id for task in reloaded.tasks] == [target.id, front.id]
    assert reloaded.get(target.id).status == "queued"
    assert reloaded.get(target.id).attempts[-1].outcome == "interrupted"

    assert target.id in captured.out
    assert "queued" in captured.out
    assert "interrupted" in captured.out


def test_cli_list_shows_block_reason(tmp_path: Path, capsys) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("需要人工处理")
    store.block_before_start(task.id, "需要人工检查 login 服务")

    assert main(["task", "list", "--root", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "blocked" in out
    assert "需要人工检查 login 服务" in out


def test_cli_task_run_recovers_then_runs_one_attempt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("跑一次 CLI 任务")
    crashed = Session.create(tmp_path, "fake-model")
    store.start_attempt(task.id, crashed.session_id)  # 模拟上次进程被 kill
    crashed.close()

    _patch_fake_llm(monkeypatch)

    assert main(["task", "run", task.id, "--root", str(tmp_path)]) == 0

    reloaded = TaskStore.load(tmp_path)
    task = reloaded.get(task.id)
    assert task.status == "done"
    assert [attempt.outcome for attempt in task.attempts] == [
        "interrupted",
        "done",
    ]

    err = capsys.readouterr().err
    assert "[task] recover" in err
    assert "[task] start" in err
    assert "[task] complete" in err


def test_cli_resume_runs_queued_task_before_chat(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    store = TaskStore.load(tmp_path)
    task = store.add("resume 先跑任务")
    # 故意没有任何 session：--resume 不应退回“最近 session”而报错。
    assert not (tmp_path / ".agent" / "sessions").exists()

    _patch_fake_llm(monkeypatch)

    assert main(["--resume", "--root", str(tmp_path)]) == 0

    reloaded = TaskStore.load(tmp_path)
    assert reloaded.get(task.id).status == "done"
    assert len(reloaded.get(task.id).attempts) == 1

    captured = capsys.readouterr()
    assert "session=" in captured.out  # SessionStart 已渲染
    assert "[task] start" in captured.err