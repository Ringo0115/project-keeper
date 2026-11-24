"""M5.0 存储层：entry id、时间戳、预算与注入视图。"""

import hashlib

from keeper.memory import (
    inject_view,
    memory_path,
    memory_tokens,
    new_entry_id,
    parse_entries,
    stamp_entries,
    strip_mem_comments,
)
from keeper.tools.memory import UpdateMemoryTool


def test_new_entry_id_uses_slug_for_ascii() -> None:
    assert new_entry_id("Test Framework: pytest", set()) == "test-framework-pytest"


def test_new_entry_id_uses_sha1_for_non_ascii() -> None:
    title = "测试框架用 pytest"
    expected = hashlib.sha1(title.encode("utf-8")).hexdigest()[:6]
    assert new_entry_id(title, set()) == expected


def test_new_entry_id_avoids_collisions() -> None:
    assert new_entry_id("Test", {"test"}) == "test-2"
    assert new_entry_id("Test", {"test", "test-2"}) == "test-3"


def test_stamp_preserves_id_and_created_and_bumps_touched() -> None:
    before = "# DECISIONS\n\n- 测试框架用 pytest\n  理由：插件多。\n"
    stamped, warnings = stamp_entries(
        before, before, "decisions", today="2026-11-23"
    )
    assert not warnings
    entry = parse_entries(stamped)[0]
    assert entry.mem is not None
    entry_id = entry.mem["id"]
    assert entry.mem["created"] == "2026-11-23"
    assert entry.mem["touched"] == "2026-11-23"

    changed = stamped.replace("理由：插件多。", "理由：插件多，断言清晰。")
    restamped, _ = stamp_entries(
        changed, stamped, "decisions", today="2026-11-24"
    )
    entry = parse_entries(restamped)[0]
    assert entry.mem is not None
    assert entry.mem["id"] == entry_id
    assert entry.mem["created"] == "2026-11-23"
    assert entry.mem["touched"] == "2026-11-24"

    # 内容没变：touched 不动。
    same, _ = stamp_entries(
        restamped, restamped, "decisions", today="2026-11-25"
    )
    entry = parse_entries(same)[0]
    assert entry.mem is not None
    assert entry.mem["touched"] == "2026-11-24"


def test_duplicate_id_gets_fresh_id_and_warning() -> None:
    raw = (
        "# DECISIONS\n\n"
        "- Foo\n"
        "  <!--mem id:foo created:2026-01-01 touched:2026-01-01-->\n"
        "- Bar\n"
        "  <!--mem id:foo created:2026-01-01 touched:2026-01-01-->\n"
    )
    stamped, warnings = stamp_entries(raw, raw, "decisions", today="2026-11-23")
    ids = [entry.mem["id"] for entry in parse_entries(stamped)]
    assert ids[0] == "foo"
    assert ids[1] != "foo"
    assert warnings and "appeared twice" in warnings[0]


def test_stamp_keeps_crlf() -> None:
    raw = (
        "# DECISIONS\r\n\r\n- Test framework\r\n  reason.\r\n"
    )
    stamped, _ = stamp_entries(raw, raw, "decisions", today="2026-11-23")
    data = stamped.encode("utf-8")
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b"")
    assert "<!--mem id:test-framework" in stamped


def test_add_stamps_entry_and_keeps_crlf(tmp_path) -> None:
    path = memory_path(tmp_path, "DECISIONS.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"# DECISIONS\r\n\r\n- Existing decision\r\n")

    result = UpdateMemoryTool(tmp_path).execute(
        {"file": "decisions", "action": "add", "entry": "New decision"}
    )
    assert result.ok, result.content

    data = path.read_bytes()
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b"")
    entries = parse_entries(data.decode("utf-8"))
    assert [entry.mem["id"] for entry in entries if entry.mem] == [
        "existing-decision",
        "new-decision",
    ]


def test_visible_text_ignores_mem_comments() -> None:
    raw = (
        "# DECISIONS\n\n- Foo\n"
        "  <!--mem id:foo created:2026-01-01 touched:2026-01-01-->\n"
        "  detail\n"
    )
    visible = strip_mem_comments(raw)
    assert "<!--mem" not in visible
    assert visible == "# DECISIONS\n\n- Foo\n  detail\n"
    assert memory_tokens(raw) == memory_tokens(visible)


def test_inject_view_strips_comments_and_folds_over_hard() -> None:
    entries = []
    for index in range(10):
        entries.append(
            f"- 第 {index} 条决策，内容足够长以便触发折叠。\n"
            f"  <!--mem id:d{index} created:2026-01-01 "
            f"touched:2026-11-{index + 10:02d}-->\n"
            "  这是一行不会被折叠视图保留的细节。\n"
        )
    raw = "# DECISIONS\n\n## Testing\n\n" + "\n".join(entries)

    full = inject_view(raw, "decisions", 10_000)
    assert "<!--mem" not in full
    assert "细节" in full

    folded = inject_view(raw, "decisions", 100)
    assert "<!--mem" not in folded
    assert len(folded.encode("utf-8")) // 3 <= 100
    # 最新一条按 touched 保留，其余进入计数提示。
    assert "第 9 条决策" in folded
    assert "more entries omitted" in folded
    assert "不会被折叠视图保留" not in folded


def test_inject_view_pure_and_stable() -> None:
    raw = "# DECISIONS\n\n- Foo\n  <!--mem id:foo created:2026-01-01 touched:2026-01-01-->\n  detail\n"
    first = inject_view(raw, "decisions", 10_000)
    second = inject_view(raw, "decisions", 10_000)
    assert first == second