import json
import time
from pathlib import Path

from keeper.session import Session


def test_create_append_reload(tmp_path: Path) -> None:
    session = Session.create(tmp_path, "test-model")
    session.append({"role": "user", "content": "hi"})
    session.append({"role": "assistant", "content": "hello"})
    path = session.path
    session.close()

    loaded = Session.load(path, tmp_path)
    try:
        assert loaded.session_id == path.stem
        assert loaded.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["type"] == "meta"
        assert json.loads(lines[0])["model"] == "test-model"
        assert json.loads(lines[1])["type"] == "message"
    finally:
        loaded.close()


def test_repair_dangling_tool_call(tmp_path: Path) -> None:
    session = Session.create(tmp_path, "m")
    try:
        session.append({"role": "user", "content": "read it"})
        session.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )

        assert session.repair_dangling_tool_calls() == 1
        last = session.messages[-1]
        assert last["role"] == "tool"
        assert last["tool_call_id"] == "call_1"
        assert last["content"].startswith("[interrupted]")

        # 已经补过，不会再补一次
        assert session.repair_dangling_tool_calls() == 0
    finally:
        session.close()


def test_latest_returns_most_recent(tmp_path: Path) -> None:
    first = Session.create(tmp_path, "m")
    first.close()
    time.sleep(0.01)
    second = Session.create(tmp_path, "m")
    second.close()

    assert Session.latest(tmp_path) == second.path


def test_latest_without_sessions(tmp_path: Path) -> None:
    assert Session.latest(tmp_path) is None