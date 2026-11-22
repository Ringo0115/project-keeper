import json
from pathlib import Path

import keeper.memory as memory
from keeper.events import (
    ApprovalRequest,
    ApprovalResult,
    FinalAnswer,
    ToolResultEvent,
)
from keeper.llm import LLMResponse
from keeper.loop import AgentLoop
from keeper.memory import (
    ARCHIVE_NAMES,
    FILE_NAMES,
    memory_path,
    render_memory,
)
from keeper.prompts import system_prompt
from keeper.session import Session
from keeper.tools.memory import UpdateMemoryTool
from keeper.tools.registry import Registry


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of responses")
        return LLMResponse(message=self._responses.pop(0), usage={})


class AutoApprover:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list = []

    def confirm(self, request) -> bool:
        self.requests.append(request)
        return self.answer


def _tool_call(name: str, args: dict, call_id: str = "call_mem") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


def _final(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _mem_dir(root: Path) -> Path:
    return root / ".agent" / "memory"


def _write_memory(root: Path, name: str, text: str) -> Path:
    path = _mem_dir(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def _snapshot(root: Path) -> dict[str, bytes]:
    directory = _mem_dir(root)
    if not directory.exists():
        return {}
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _loop(
    tmp_path: Path,
    llm: FakeLLM,
    approver: object | None,
    tools: list,
) -> tuple[AgentLoop, Session]:
    session = Session.create(tmp_path, "test")
    loop = AgentLoop(
        llm=llm,
        registry=Registry(tools),
        session=session,
        root=tmp_path,
        max_steps=5,
        approver=approver,  # type: ignore[arg-type]
    )
    return loop, session


def _run_update(tmp_path: Path, args: dict) -> tuple[list, AutoApprover]:
    approver = AutoApprover(answer=True)
    llm = FakeLLM([_tool_call("update_memory", args), _final("ok")])
    loop, session = _loop(tmp_path, llm, approver, [UpdateMemoryTool(tmp_path)])
    try:
        events = list(loop.run("更新记忆"))
    finally:
        session.close()
    return events, approver


# ---------------------------------------------------------------------------
# 注入：新会话不读代码也能看到 memory
# ---------------------------------------------------------------------------


def test_memory_injected_in_order_with_project_state_last(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n## Testing\n\n- 测试框架用 pytest。\n")
    _write_memory(tmp_path, "LESSONS.md", "# LESSONS\n\n- 工具失败先读 ERROR 再改方案。\n")
    _write_memory(tmp_path, "INDEX.md", "# INDEX\n\n- generated navigation\n")
    _write_memory(tmp_path, "PROJECT_STATE.md", "# PROJECT_STATE\n\n- 下一步：M3 任务队列。\n")

    llm = FakeLLM([_final("用 pytest；下一步是 M3 任务队列。")])
    loop, session = _loop(tmp_path, llm, None, [UpdateMemoryTool(tmp_path)])
    try:
        events = list(loop.run("这个项目测试用什么框架？下一步是什么？只根据记忆回答。"))
    finally:
        session.close()

    assert isinstance(events[-1], FinalAnswer)
    assert "pytest" in events[-1].text

    content = llm.calls[0][0]["content"]
    assert "测试框架用 pytest" in content
    assert "下一步：M3 任务队列" in content

    markers = ["DECISIONS.md", "LESSONS.md", "INDEX.md", "PROJECT_STATE.md"]
    positions = [content.index(f"## 项目记忆 / {name}") for name in markers]
    assert positions == sorted(positions)
    assert content.rindex("## 项目记忆 / PROJECT_STATE.md") > content.rindex(
        "## 项目记忆 / INDEX.md"
    )

    # 静态说明（含 update_memory 用法）在记忆块之前。
    assert content.index("记忆维护：") < content.index("## 项目记忆 / DECISIONS.md")
    # system prompt 最后一段就是 PROJECT_STATE 的内容。
    assert content.rstrip().endswith("下一步：M3 任务队列。")

    # 证明测试里根本没有代码文件可读。
    assert not (tmp_path / "README.md").exists()
    assert list(tmp_path.glob("*.py")) == []


def test_system_prompt_is_a_startup_snapshot(tmp_path: Path) -> None:
    _write_memory(
        tmp_path,
        "PROJECT_STATE.md",
        "# PROJECT_STATE\n\n- 启动时的状态\n",
    )
    llm = FakeLLM([])
    loop, session = _loop(tmp_path, llm, None, [UpdateMemoryTool(tmp_path)])
    try:
        assert "启动时的状态" in loop.system
        result = UpdateMemoryTool(tmp_path).execute(
            {
                "file": "project_state",
                "action": "add",
                "entry": "后来新增的状态",
            }
        )
        assert result.ok, result.content
        # 注入只在启动时做一次；批准写入后不重建 prompt。
        assert "后来新增的状态" not in loop.system
    finally:
        session.close()


def test_render_memory_survives_unreadable_file(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 正常条目\n")
    (memory_path(tmp_path, "DECISIONS.md")).write_bytes(b"\xff\xfe")
    _write_memory(tmp_path, "PROJECT_STATE.md", "# PROJECT_STATE\n\n- 状态仍可读。\n")

    text = render_memory(tmp_path)
    assert "DECISIONS.md 读取失败" in text
    assert "状态仍可读" in text

    prompt = system_prompt(tmp_path)
    assert "DECISIONS.md 读取失败" in prompt
    assert "状态仍可读" in prompt


# ---------------------------------------------------------------------------
# 审批：拒绝后字节不变 / fail-closed
# ---------------------------------------------------------------------------


def _add_args() -> dict:
    return {
        "file": "decisions",
        "action": "add",
        "entry": "新增决策",
        "detail": ["细节一行。"],
    }


def test_denied_memory_write_keeps_bytes_unchanged(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 原有决策\n")
    _write_memory(tmp_path, "INDEX.md", "# INDEX\n\n- old index\n")
    before = _snapshot(tmp_path)

    approver = AutoApprover(answer=False)
    llm = FakeLLM(
        [
            _tool_call("update_memory", _add_args()),
            _final("没写成，我换方案。"),
        ]
    )
    loop, session = _loop(tmp_path, llm, approver, [UpdateMemoryTool(tmp_path)])
    try:
        events = list(loop.run("更新记忆"))
    finally:
        session.close()

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].ok is False
    assert "denied" in results[0].output

    assert len(approver.requests) == 1
    assert approver.requests[0].summary == "update_memory: decisions/add"
    assert "--- a/.agent/memory/DECISIONS.md" in approver.requests[0].detail

    assert _snapshot(tmp_path) == before


def test_denied_write_does_not_create_memory_dir(tmp_path: Path) -> None:
    approver = AutoApprover(answer=False)
    llm = FakeLLM([_tool_call("update_memory", _add_args()), _final("ok")])
    loop, session = _loop(tmp_path, llm, approver, [UpdateMemoryTool(tmp_path)])
    try:
        list(loop.run("更新记忆"))
    finally:
        session.close()

    assert _snapshot(tmp_path) == {}
    assert not _mem_dir(tmp_path).exists()


def test_memory_write_without_approver_is_denied(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 原有决策\n")
    before = _snapshot(tmp_path)

    llm = FakeLLM([_tool_call("update_memory", _add_args()), _final("ok")])
    loop, session = _loop(tmp_path, llm, None, [UpdateMemoryTool(tmp_path)])
    try:
        events = list(loop.run("更新记忆"))
    finally:
        session.close()

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert results and results[0].ok is False
    assert "no approver configured" in results[0].output
    assert _snapshot(tmp_path) == before


def test_invalid_memory_file_is_denied_before_approval(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 现有决策\n")
    before = _snapshot(tmp_path)

    events, approver = _run_update(
        tmp_path,
        {"file": "index", "action": "add", "entry": "x"},
    )

    # 非法参数在 review 阶段就 DENY：不弹审批，也不执行工具。
    assert approver.requests == []
    assert not any(isinstance(event, ApprovalRequest) for event in events)
    approvals = [event for event in events if isinstance(event, ApprovalResult)]
    assert approvals and approvals[0].approved is False
    assert "invalid memory update" in approvals[0].reason

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert results and results[0].ok is False
    assert "denied" in results[0].output
    assert _snapshot(tmp_path) == before


def test_missing_supersede_is_denied_before_approval(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 现有决策\n")
    before = _snapshot(tmp_path)

    events, approver = _run_update(
        tmp_path,
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新决策",
            "supersedes": "不存在的旧条目",
        },
    )

    assert approver.requests == []
    assert not any(isinstance(event, ApprovalRequest) for event in events)
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert results and results[0].ok is False
    assert "not found" in results[0].output
    assert _snapshot(tmp_path) == before


# ---------------------------------------------------------------------------
# add / INDEX
# ---------------------------------------------------------------------------


def test_add_creates_entry_and_auto_generated_index(tmp_path: Path) -> None:
    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "add",
            "entry": "测试框架用 pytest。",
            "detail": ["CI 也跑 pytest -q。"],
            "section": "Testing",
        }
    )
    assert result.ok, result.content

    decisions = memory_path(tmp_path, "DECISIONS.md").read_text(encoding="utf-8")
    lines = decisions.splitlines()
    assert "# DECISIONS" in lines
    assert "## Testing" in lines
    assert "- 测试框架用 pytest。" in lines
    assert "  CI 也跑 pytest -q。" in lines

    index = memory_path(tmp_path, "INDEX.md").read_text(encoding="utf-8")
    assert "# INDEX" in index
    assert "自动生成，勿手改" in index
    for name in (
        FILE_NAMES["decisions"],
        FILE_NAMES["lessons"],
        FILE_NAMES["project_state"],
        ARCHIVE_NAMES["decisions"],
        ARCHIVE_NAMES["lessons"],
        ARCHIVE_NAMES["project_state"],
    ):
        assert name in index
    assert "1 entry" in index
    assert "0 archived decisions" in index


def test_index_is_regenerated_and_cannot_be_edited(tmp_path: Path) -> None:
    _write_memory(tmp_path, "INDEX.md", "# INDEX\n\n- stale manual line\n")
    stale_bytes = memory_path(tmp_path, "INDEX.md").read_bytes()
    tool = UpdateMemoryTool(tmp_path)

    rejected = tool.execute({"file": "index", "action": "add", "entry": "x"})
    assert rejected.ok is False
    assert "INDEX.md is generated automatically" in rejected.content
    assert memory_path(tmp_path, "INDEX.md").read_bytes() == stale_bytes

    unknown = tool.execute({"file": "state", "action": "add", "entry": "x"})
    assert unknown.ok is False
    assert "decisions, lessons, or project_state" in unknown.content
    assert memory_path(tmp_path, "INDEX.md").read_bytes() == stale_bytes

    added = tool.execute(
        {"file": "decisions", "action": "add", "entry": "新决策"}
    )
    assert added.ok, added.content
    index = memory_path(tmp_path, "INDEX.md").read_text(encoding="utf-8")
    assert "stale manual line" not in index
    assert "- .agent/memory/DECISIONS.md" in index
    assert "0 archived decisions" in index


# ---------------------------------------------------------------------------
# supersede / archive
# ---------------------------------------------------------------------------


def test_supersede_moves_old_block_verbatim_to_archive(tmp_path: Path) -> None:
    old_block = (
        "- 测试框架用 unittest\n"
        "  旧理由：标准库自带。\n"
        "  旧细节：不需要额外依赖。"
    )
    _write_memory(
        tmp_path,
        "DECISIONS.md",
        "# DECISIONS\n\n## Testing\n\n"
        f"{old_block}\n\n## Other\n\n- 别的决策。\n",
    )

    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "测试框架改用 pytest",
            "detail": ["更快，插件多。"],
            "supersedes": "测试框架用 unittest",  # 不带 "- " 也要能匹配
        }
    )
    assert result.ok, result.content

    live = memory_path(tmp_path, "DECISIONS.md").read_text(encoding="utf-8")
    archive = memory_path(tmp_path, ARCHIVE_NAMES["decisions"]).read_text(
        encoding="utf-8"
    )

    # 新条目在旧 section 里，旧 section 标题和其他 section 都保留。
    assert "## Testing" in live
    assert "## Other" in live
    assert live.index("## Testing") < live.index("- 测试框架改用 pytest")
    assert live.index("- 测试框架改用 pytest") < live.index("## Other")
    # supersedes 的值取自文件里的原行，而不是模型参数。
    assert "  supersedes: - 测试框架用 unittest" in live
    assert "  更快，插件多。" in live

    # 旧条目整块原样在 archive；live 里不再有旧 entry 块。
    assert old_block in archive
    assert archive.count("- 测试框架用 unittest") == 1
    assert "\n- 测试框架用 unittest\n" not in live
    assert "旧理由：标准库自带。" not in live
    assert "旧细节：不需要额外依赖。" not in live

    index = memory_path(tmp_path, "INDEX.md").read_text(encoding="utf-8")
    assert "1 archived decision" in index


def test_supersede_with_section_disambiguates_duplicates(tmp_path: Path) -> None:
    _write_memory(
        tmp_path,
        "DECISIONS.md",
        "# DECISIONS\n\n## A\n\n- 同名\n\n## B\n\n- 同名\n",
    )
    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新的同名",
            "section": "B",
            "supersedes": "同名",
        }
    )
    assert result.ok, result.content

    live = memory_path(tmp_path, "DECISIONS.md").read_text(encoding="utf-8")
    archive = memory_path(tmp_path, ARCHIVE_NAMES["decisions"]).read_text(
        encoding="utf-8"
    )
    lines = live.splitlines()
    assert lines.count("- 同名") == 1  # A 里那条还在
    assert "- 新的同名" in lines
    assert "  supersedes: - 同名" in live
    assert archive.splitlines().count("- 同名") == 1


def test_supersede_section_only_disambiguates(tmp_path: Path) -> None:
    _write_memory(
        tmp_path,
        "DECISIONS.md",
        "# DECISIONS\n\n## A\n\n- 旧决策\n\n## B\n\n- 别的决策\n",
    )
    before = _snapshot(tmp_path)

    # 旧条目在 A，却指定 B：section 只用于消歧，找不到就失败。
    failed = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新决策",
            "section": "B",
            "supersedes": "旧决策",
        }
    )
    assert failed.ok is False
    assert "not found in section 'B'" in failed.content
    assert "follows the old entry's section" in failed.content
    assert _snapshot(tmp_path) == before

    # 指定 A 能匹配；新条目跟随旧条目的 A，不会落进 B。
    ok = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新决策",
            "section": "A",
            "supersedes": "旧决策",
        }
    )
    assert ok.ok, ok.content
    live = memory_path(tmp_path, "DECISIONS.md").read_text(encoding="utf-8")
    assert live.index("- 新决策") < live.index("## B")


def test_supersede_before_first_heading_stays_in_place(tmp_path: Path) -> None:
    _write_memory(
        tmp_path,
        "DECISIONS.md",
        "# DECISIONS\n\n- 旧决策\n\n## A\n\n- a\n",
    )
    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新决策",
            "supersedes": "旧决策",
        }
    )
    assert result.ok, result.content

    live = memory_path(tmp_path, "DECISIONS.md").read_text(encoding="utf-8")
    lines = live.splitlines()
    assert lines.count("- 旧决策") == 0
    assert "- 新决策" in lines
    # 没有 section 的旧条目替换后仍留在第一个 heading 之前。
    assert live.index("- 新决策") < live.index("## A")


def test_ambiguous_supersede_is_rejected_without_writing(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 重复\n\n- 重复\n")
    before = _snapshot(tmp_path)

    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新的",
            "supersedes": "重复",
        }
    )
    assert result.ok is False
    assert "ambiguous" in result.content.lower()
    assert _snapshot(tmp_path) == before


def test_missing_supersede_target_is_rejected_without_writing(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 只有这条\n")
    before = _snapshot(tmp_path)

    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新的",
            "supersedes": "不存在的旧条目",
        }
    )
    assert result.ok is False
    assert "not found" in result.content
    assert _snapshot(tmp_path) == before


# ---------------------------------------------------------------------------
# 审批 diff
# ---------------------------------------------------------------------------


def test_approval_preview_shows_archive_live_and_index_diff(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 测试框架用 unittest\n")
    approver = AutoApprover(answer=True)
    args = {
        "file": "decisions",
        "action": "supersede",
        "entry": "测试框架改用 pytest",
        "supersedes": "- 测试框架用 unittest",
    }
    llm = FakeLLM([_tool_call("update_memory", args), _final("done")])
    loop, session = _loop(tmp_path, llm, approver, [UpdateMemoryTool(tmp_path)])
    try:
        events = list(loop.run("更新记忆"))
    finally:
        session.close()

    assert len(approver.requests) == 1
    request = approver.requests[0]
    assert request.summary == "update_memory: decisions/supersede"
    detail = request.detail
    assert "--- a/.agent/memory/DECISIONS.md" in detail
    assert "+++ b/.agent/memory/DECISIONS.md" in detail
    assert "DECISIONS_ARCHIVE.md" in detail
    assert "INDEX.md" in detail
    assert "+- 测试框架改用 pytest" in detail
    assert "+  supersedes: - 测试框架用 unittest" in detail

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert results and results[0].ok is True
    assert memory_path(tmp_path, ARCHIVE_NAMES["decisions"]).exists()


# ---------------------------------------------------------------------------
# 换行保真 / 写入顺序 / 部分失败
# ---------------------------------------------------------------------------


def test_crlf_live_file_stays_crlf(tmp_path: Path) -> None:
    raw = (
        "# DECISIONS\r\n\r\n## Testing\r\n\r\n"
        "- 测试框架用 unittest\r\n  旧理由。\r\n"
    )
    _write_memory(tmp_path, "DECISIONS.md", raw)

    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "add",
            "entry": "新增决策",
            "detail": ["细节。"],
        }
    )
    assert result.ok, result.content

    data = memory_path(tmp_path, "DECISIONS.md").read_bytes()
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b"")
    text = data.decode("utf-8")
    assert "- 新增决策" in text
    assert "  细节。" in text

    # 新文件按约定使用 LF。
    index = memory_path(tmp_path, "INDEX.md").read_bytes()
    assert b"\r\n" not in index
    assert b"\n" in index


def test_existing_archive_stays_crlf(tmp_path: Path) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 旧决策\n")
    _write_memory(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\r\n\r\n## Superseded\r\n\r\n- 更早的旧决策\r\n",
    )

    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新决策",
            "supersedes": "旧决策",
        }
    )
    assert result.ok, result.content

    data = memory_path(tmp_path, ARCHIVE_NAMES["decisions"]).read_bytes()
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b"")
    text = data.decode("utf-8")
    assert "- 更早的旧决策" in text
    assert "- 旧决策" in text


def test_write_order_is_archive_then_live_then_index(
    tmp_path: Path, monkeypatch
) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 旧决策\n")
    calls: list[str] = []
    real_write = memory._write_change

    def fake_write(change) -> None:
        calls.append(change.path.name)
        if change.path.name == "DECISIONS.md":
            raise OSError("simulated live write failure")
        real_write(change)

    monkeypatch.setattr(memory, "_write_change", fake_write)

    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "新决策",
            "supersedes": "旧决策",
        }
    )
    assert result.ok is False
    assert calls == ["DECISIONS_ARCHIVE.md", "DECISIONS.md"]
    assert "memory write failed" in result.content
    assert "已写" in result.content and "未写" in result.content

    # archive 先成功，live 失败：旧条目仍在 live，没有丢失。
    live = memory_path(tmp_path, "DECISIONS.md").read_text(encoding="utf-8")
    archive = memory_path(tmp_path, ARCHIVE_NAMES["decisions"]).read_text(
        encoding="utf-8"
    )
    assert "- 旧决策" in live
    assert "- 旧决策" in archive
    assert not memory_path(tmp_path, "INDEX.md").exists()


def test_index_failure_reports_memory_updated(tmp_path: Path, monkeypatch) -> None:
    _write_memory(tmp_path, "DECISIONS.md", "# DECISIONS\n\n- 原有决策\n")
    real_write = memory._write_change

    def fake_write(change) -> None:
        if change.path.name == "INDEX.md":
            raise OSError("simulated index failure")
        real_write(change)

    monkeypatch.setattr(memory, "_write_change", fake_write)

    result = UpdateMemoryTool(tmp_path).execute(
        {"file": "decisions", "action": "add", "entry": "新决策"}
    )
    assert result.ok is False
    assert "memory 已更新，INDEX 刷新失败" in result.content
    live = memory_path(tmp_path, "DECISIONS.md").read_text(encoding="utf-8")
    assert "- 新决策" in live
    assert not memory_path(tmp_path, "INDEX.md").exists()
