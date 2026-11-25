"""M5.1b：pinned 的代码级保护（改 / 移 / 删 / 转档都从备份回滚）。"""

import pytest

from keeper.memory import (
    ARCHIVE_NAMES,
    FILE_NAMES,
    is_pinned,
    memory_path,
    parse_entries,
    stamp_entries,
    strip_mem_comments,
)
from keeper.tidy import (
    backup_memory,
    postpass,
    read_run_record,
    read_snapshot,
)


def _mem(root):
    return root / ".agent" / "memory"


def _write(root, name: str, text: str) -> None:
    path = _mem(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _entry(title: str, entry_id: str, detail: str, *, pinned: bool = False) -> str:
    field = " pinned:1" if pinned else ""
    return (
        f"- {title}\n"
        f"  <!--mem id:{entry_id} created:2026-01-01 touched:2026-01-01{field}-->\n"
        f"  {detail}\n"
    )


def _decisions(*entries: str) -> str:
    return "# DECISIONS\n\n" + "\n".join(entries)


def _prepare(root):
    """整理前的快照与备份；必须在模拟模型改写之前调用。"""
    before = read_snapshot(root)
    backup = backup_memory(root)
    assert backup.path is not None
    return before, backup


def _postpass(root, before, backup, today: str = "2026-11-23"):
    return postpass(
        root,
        before,
        backup_dir=str(backup.path),
        trigger="test",
        today=today,
    )


def _read(root, name: str) -> str:
    return memory_path(root, name).read_text(encoding="utf-8")


def test_pinned_marker_parses_survives_stamp_and_injection() -> None:
    text = _decisions(_entry("Alpha", "alpha", "a", pinned=True))
    entry = parse_entries(text)[0]
    assert is_pinned(entry)
    assert entry.mem["pinned"] == "1"

    stamped, warnings = stamp_entries(text, text, "decisions")
    assert warnings == []
    assert "pinned:1" in stamped
    assert "<!--mem" not in strip_mem_comments(stamped)

    unpinned = parse_entries(_entry("Beta", "beta", "b"))[0]
    assert not is_pinned(unpinned)
    off = parse_entries(
        _decisions(
            "- Gamma\n"
            "  <!--mem id:gamma created:2026-01-01 touched:2026-01-01 pinned:0-->\n"
            "  c\n"
        )
    )[0]
    assert not is_pinned(off)


def test_pinned_entry_edited_in_decisions_is_rolled_back(tmp_path) -> None:
    original = _entry("Alpha", "alpha", "original detail", pinned=True)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(original, _entry("Beta", "beta", "b")),
    )
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(
            _entry("Alpha", "alpha", "model rewrote this", pinned=True),
            _entry("Beta", "beta", "b"),
        ),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["alpha"]
    assert report.outcome == "restored"
    live = _read(tmp_path, FILE_NAMES["decisions"])
    assert original.strip() in live
    assert "model rewrote this" not in live
    assert "- Beta" in live


def test_pinned_entry_edited_in_project_state_is_rolled_back(tmp_path) -> None:
    original = _entry("Current", "state-a", "original state", pinned=True)
    _write(tmp_path, FILE_NAMES["project_state"], "# PROJECT_STATE\n\n" + original)
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["project_state"],
        "# PROJECT_STATE\n\n"
        + _entry("Current", "state-a", "model rewrite", pinned=True),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["state-a"]
    live = _read(tmp_path, FILE_NAMES["project_state"])
    assert original.strip() in live
    assert "model rewrite" not in live


def test_pinned_entry_deleted_from_project_state_is_restored(tmp_path) -> None:
    original = _entry("Current", "state-a", "keep me", pinned=True)
    _write(tmp_path, FILE_NAMES["project_state"], "# PROJECT_STATE\n\n" + original)
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["project_state"],
        "# PROJECT_STATE\n\n" + _entry("Other", "state-b", "new"),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["state-a"]
    live = _read(tmp_path, FILE_NAMES["project_state"])
    assert original.strip() in live
    assert "- Other" in live


def test_pinned_entry_archived_is_pulled_back_without_duplicate(tmp_path) -> None:
    pinned = _entry("Alpha", "alpha", "a", pinned=True)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(pinned, _entry("Beta", "beta", "b")),
    )
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Beta", "beta", "b")),
    )
    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n" + pinned,
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["alpha"]
    live = _read(tmp_path, FILE_NAMES["decisions"])
    archive = _read(tmp_path, ARCHIVE_NAMES["decisions"])
    assert pinned.strip() in live
    assert live.count("id:alpha") == 1
    assert "id:alpha" not in archive
    assert report.restored == []


def test_pinned_entry_moved_in_live_is_put_back_in_place(tmp_path) -> None:
    alpha = _entry("Alpha", "alpha", "a", pinned=True)
    beta = _entry("Beta", "beta", "b")
    original = _decisions(alpha, beta)
    _write(tmp_path, FILE_NAMES["decisions"], original)
    before, backup = _prepare(tmp_path)
    _write(tmp_path, FILE_NAMES["decisions"], _decisions(beta, alpha))

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["alpha"]
    assert _read(tmp_path, FILE_NAMES["decisions"]) == original


def test_copied_pinned_entry_is_removed_from_archive(tmp_path) -> None:
    pinned = _entry("Alpha", "alpha", "a", pinned=True)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(pinned, _entry("Beta", "beta", "b")),
    )
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n" + pinned,
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["alpha"]
    assert _read(tmp_path, FILE_NAMES["decisions"]).count("id:alpha") == 1
    assert "id:alpha" not in _read(tmp_path, ARCHIVE_NAMES["decisions"])


def test_untouched_pinned_entry_keeps_bytes_and_is_not_counted(tmp_path) -> None:
    text = _decisions(
        _entry("Alpha", "alpha", "a", pinned=True),
        _entry("Beta", "beta", "b"),
    )
    _write(tmp_path, FILE_NAMES["decisions"], text)
    before, backup = _prepare(tmp_path)
    before_bytes = memory_path(tmp_path, FILE_NAMES["decisions"]).read_bytes()

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == []
    assert report.outcome in {"no_change", "index_only"}
    assert memory_path(tmp_path, FILE_NAMES["decisions"]).read_bytes() == before_bytes
    record = read_run_record(tmp_path)
    assert record is not None
    assert record["restored_pinned"] == []


def test_unpinned_entry_can_still_be_archived(tmp_path) -> None:
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Alpha", "alpha", "a"), _entry("Beta", "beta", "b")),
    )
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Alpha", "alpha", "a")),
    )
    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n" + _entry("Beta", "beta", "b"),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == []
    assert report.archived == ["beta"]
    assert "id:beta" in _read(tmp_path, ARCHIVE_NAMES["decisions"])


def test_update_memory_can_still_supersede_pinned_entry(tmp_path) -> None:
    from keeper.tools.memory import UpdateMemoryTool

    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Alpha", "alpha", "a", pinned=True)),
    )

    result = UpdateMemoryTool(tmp_path).execute(
        {
            "file": "decisions",
            "action": "supersede",
            "entry": "Alpha replaced",
            "supersedes": "- Alpha",
        }
    )

    assert result.ok, result.content
    live = _read(tmp_path, FILE_NAMES["decisions"])
    archive = _read(tmp_path, ARCHIVE_NAMES["decisions"])
    assert "- Alpha replaced" in live
    live_ids = {
        entry.mem["id"] for entry in parse_entries(live) if entry.mem
    }
    assert "alpha" not in live_ids
    assert "id:alpha" in archive and "pinned:1" in archive


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            (
                "# DECISIONS\n\n"
                + _entry("Alpha", "alpha", "a", pinned=True)
                + _entry("Beta", "beta", "b")
            ).rstrip("\n"),
            id="no-terminal-newline",
        ),
        pytest.param(
            "# DECISIONS\n\n"
            + _entry("Alpha", "alpha", "a", pinned=True)
            + _entry("Beta", "beta", "b")
            + "\n\n",
            id="trailing-blank-lines",
        ),
        pytest.param(
            (
                "# DECISIONS\r\n\r\n"
                "- Alpha\r\n"
                "  <!--mem id:alpha created:2026-01-01 touched:2026-01-01 pinned:1-->\r\n"
                "  a\r\n\r\n"
                "- Beta\r\n"
                "  <!--mem id:beta created:2026-01-01 touched:2026-01-01-->\r\n"
                "  b\r\n"
            ),
            id="crlf",
        ),
        pytest.param(
            "# DECISIONS\n\n"
            "- Alpha\n"
            "  <!--mem id:alpha created:2026-01-01 touched:2026-01-01 pinned:1-->\n"
            "  a\n"
            "- Beta\n"
            "  <!--mem id:beta created:2026-01-01 touched:2026-01-01-->\n"
            "  b\n",
            id="no-blank-between-entries",
        ),
        pytest.param(
            "# DECISIONS\n\n"
            "- Alpha   \n"
            "  <!--mem id:alpha created:2026-01-01 touched:2026-01-01 pinned:1-->\n"
            "    a  \n"
            "\n"
            "- Beta\n"
            "  <!--mem id:beta created:2026-01-01 touched:2026-01-01-->\n"
            "  b\n",
            id="internal-whitespace",
        ),
    ],
)
def test_unedited_pinned_file_is_byte_identical(tmp_path, payload) -> None:
    _write(tmp_path, FILE_NAMES["decisions"], payload)
    before, backup = _prepare(tmp_path)
    before_bytes = memory_path(tmp_path, FILE_NAMES["decisions"]).read_bytes()

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == []
    assert memory_path(tmp_path, FILE_NAMES["decisions"]).read_bytes() == before_bytes


def test_pinned_without_id_keeps_the_id_added_by_stamp(tmp_path) -> None:
    hand_written = (
        "# DECISIONS\n\n"
        "- Alpha\n"
        "  <!--mem pinned:1-->\n"
        "  a\n"
    )
    _write(tmp_path, FILE_NAMES["decisions"], hand_written)
    before, backup = _prepare(tmp_path)

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == []
    live = _read(tmp_path, FILE_NAMES["decisions"])
    assert "id:" in live and "pinned:1" in live
    assert live.count("- Alpha") == 1


def test_unedited_live_and_archive_files_are_byte_identical(tmp_path) -> None:
    decisions = (
        "# DECISIONS\n\n"
        + _entry("Alpha", "alpha", "a", pinned=True)
        + _entry("Beta", "beta", "b")
    ).rstrip("\n")
    archive = (
        "# DECISIONS_ARCHIVE\r\n\r\n## Superseded\r\n\r\n"
        "- Old\r\n"
        "  <!--mem id:old created:2026-01-01 touched:2026-01-01-->\r\n"
        "  old detail\r\n"
    )
    state = "# PROJECT_STATE\n\n" + _entry(
        "Current", "state-a", "keep", pinned=True
    )
    _write(tmp_path, FILE_NAMES["decisions"], decisions)
    _write(tmp_path, ARCHIVE_NAMES["decisions"], archive)
    _write(tmp_path, FILE_NAMES["project_state"], state)
    before, backup = _prepare(tmp_path)
    before_bytes = {
        name: memory_path(tmp_path, name).read_bytes()
        for name in (
            FILE_NAMES["decisions"],
            ARCHIVE_NAMES["decisions"],
            FILE_NAMES["project_state"],
        )
    }

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == []
    for name, data in before_bytes.items():
        assert memory_path(tmp_path, name).read_bytes() == data, name


def test_edited_pinned_file_without_terminal_newline_keeps_that_style(
    tmp_path,
) -> None:
    original = (
        "# DECISIONS\n\n"
        + _entry("Alpha", "alpha", "a", pinned=True)
        + _entry("Beta", "beta", "b")
    ).rstrip("\n")
    _write(tmp_path, FILE_NAMES["decisions"], original)
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        original.replace("  a", "  changed"),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["alpha"]
    data = memory_path(tmp_path, FILE_NAMES["decisions"]).read_bytes()
    assert "changed" not in data.decode("utf-8")
    assert "  a" in data.decode("utf-8")
    # 虽然重写过，仍保留“无终止换行”的文件风格。
    assert not data.endswith((b"\n", b"\r"))


def test_pinned_moved_to_other_live_file_is_restored(tmp_path) -> None:
    pinned = _entry("Current", "state-a", "keep me", pinned=True)
    _write(tmp_path, FILE_NAMES["project_state"], "# PROJECT_STATE\n\n" + pinned)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Beta", "beta", "b")),
    )
    before, backup = _prepare(tmp_path)
    _write(tmp_path, FILE_NAMES["project_state"], "# PROJECT_STATE\n")
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Beta", "beta", "b"), pinned),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["state-a"]
    assert _read(tmp_path, FILE_NAMES["project_state"]).endswith(
        pinned.strip() + "\n"
    )
    assert "id:state-a" not in _read(tmp_path, FILE_NAMES["decisions"])
    assert "- Beta" in _read(tmp_path, FILE_NAMES["decisions"])


def test_pinned_moved_to_other_archive_is_restored(tmp_path) -> None:
    pinned = _entry("Current", "state-a", "keep me", pinned=True)
    _write(tmp_path, FILE_NAMES["project_state"], "# PROJECT_STATE\n\n" + pinned)
    before, backup = _prepare(tmp_path)
    _write(tmp_path, FILE_NAMES["project_state"], "# PROJECT_STATE\n")
    _write(
        tmp_path,
        ARCHIVE_NAMES["lessons"],
        "# LESSONS_ARCHIVE\n\n## Superseded\n\n" + pinned,
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["state-a"]
    assert "id:state-a" in _read(tmp_path, FILE_NAMES["project_state"])
    assert "id:state-a" not in _read(tmp_path, ARCHIVE_NAMES["lessons"])


def test_duplicate_id_copies_are_collapsed_to_one(tmp_path) -> None:
    pinned = _entry("Alpha", "alpha", "a", pinned=True)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(pinned, _entry("Beta", "beta", "b")),
    )
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(
            pinned,
            _entry("Beta", "beta", "b"),
            pinned,
            pinned.replace("  a", "  changed"),
        ),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["alpha"]
    live = _read(tmp_path, FILE_NAMES["decisions"])
    assert live.count("id:alpha") == 1
    assert pinned.strip() in live
    assert "changed" not in live
    assert "- Beta" in live


def test_independent_same_bullet_entry_in_archive_is_kept(tmp_path) -> None:
    pinned = _entry("Policy", "policy-live", "new", pinned=True)
    independent = _entry("Policy", "policy-old", "older independent")
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(pinned, _entry("Beta", "beta", "b")),
    )
    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n" + independent,
    )
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Beta", "beta", "b")),
    )
    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n"
        + independent
        + "\n"
        + pinned,
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["policy-live"]
    live = _read(tmp_path, FILE_NAMES["decisions"])
    archive = _read(tmp_path, ARCHIVE_NAMES["decisions"])
    assert "id:policy-live" in live
    assert "id:policy-live" not in archive
    assert "id:policy-old" in archive
    assert "older independent" in archive


def test_independent_same_bullet_entry_in_other_file_is_kept(tmp_path) -> None:
    pinned = _entry("Policy", "policy-live", "new", pinned=True)
    independent = _entry("Policy", "lesson-policy", "lesson text")
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(pinned, _entry("Beta", "beta", "b")),
    )
    _write(tmp_path, FILE_NAMES["lessons"], "# LESSONS\n\n" + independent)
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Beta", "beta", "b")),
    )
    _write(
        tmp_path,
        FILE_NAMES["lessons"],
        "# LESSONS\n\n" + independent + "\n" + pinned,
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["policy-live"]
    live = _read(tmp_path, FILE_NAMES["decisions"])
    lessons = _read(tmp_path, FILE_NAMES["lessons"])
    assert "id:policy-live" in live
    assert "id:policy-live" not in lessons
    assert "id:lesson-policy" in lessons
    assert "lesson text" in lessons


def test_pinned_restore_keeps_crlf_bytes(tmp_path) -> None:
    original = (
        "# DECISIONS\r\n\r\n"
        "- Alpha\r\n"
        "  <!--mem id:alpha created:2026-01-01 touched:2026-01-01 pinned:1-->\r\n"
        "  original\r\n\r\n"
        "- Beta\r\n"
        "  <!--mem id:beta created:2026-01-01 touched:2026-01-01-->\r\n"
        "  b\r\n"
    )
    _write(tmp_path, FILE_NAMES["decisions"], original)
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        original.replace("original", "changed"),
    )

    report = _postpass(tmp_path, before, backup)

    assert report.restored_pinned == ["alpha"]
    data = memory_path(tmp_path, FILE_NAMES["decisions"]).read_bytes()
    assert data == original.encode("utf-8")
    assert b"\n" not in data.replace(b"\r\n", b"")


def test_multiple_pinned_entries_restore_in_original_order(tmp_path) -> None:
    a = _entry("Alpha", "alpha", "a", pinned=True)
    b = _entry("Beta", "beta", "b")
    c = _entry("Gamma", "gamma", "c", pinned=True)
    original = _decisions(a, b, c)
    _write(tmp_path, FILE_NAMES["decisions"], original)
    before, backup = _prepare(tmp_path)
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(
            _entry("Beta", "beta", "b"),
            _entry("Gamma", "gamma", "changed", pinned=True),
            _entry("Alpha", "alpha", "changed", pinned=True),
        ),
    )

    report = _postpass(tmp_path, before, backup)

    assert sorted(report.restored_pinned) == ["alpha", "gamma"]
    assert _read(tmp_path, FILE_NAMES["decisions"]) == original