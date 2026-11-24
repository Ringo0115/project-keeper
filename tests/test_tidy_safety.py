"""M5.0 整理安全网：备份、守恒、恢复、INDEX、run record。"""

from datetime import datetime, timedelta

import keeper.tidy as tidy
from keeper.memory import ARCHIVE_NAMES, FILE_NAMES, memory_path, parse_entries
from keeper.tidy import (
    backup_memory,
    conserve_ids,
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


def _entry(title: str, entry_id: str, detail: str) -> str:
    return (
        f"- {title}\n"
        f"  <!--mem id:{entry_id} created:2026-01-01 touched:2026-01-01-->\n"
        f"  {detail}\n"
    )


def _decisions(*entries: str) -> str:
    return "# DECISIONS\n\n" + "\n".join(entries)


def test_backup_copies_files_and_keeps_five(tmp_path) -> None:
    _write(tmp_path, FILE_NAMES["decisions"], _decisions(_entry("Alpha", "alpha", "a")))
    _write(tmp_path, FILE_NAMES["index"], "# INDEX\n")

    base = datetime(2026, 11, 23, 12, 0, 0)
    for index in range(6):
        result = backup_memory(tmp_path, now=base + timedelta(seconds=index))
        assert result.path is not None
        assert not result.failed
        assert result.copied == 2

    backups = sorted((_mem(tmp_path) / "backups").iterdir())
    assert len(backups) == 5
    assert all((path / FILE_NAMES["decisions"]).is_file() for path in backups)
    assert all((path / FILE_NAMES["index"]).is_file() for path in backups)


def test_backup_failure_reports_error_and_cleans_up(tmp_path, monkeypatch) -> None:
    _write(tmp_path, FILE_NAMES["decisions"], _decisions(_entry("Alpha", "alpha", "a")))

    def boom(*args, **kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(tidy.shutil, "copy2", boom)
    result = backup_memory(tmp_path)

    assert result.failed
    assert result.path is None
    assert "simulated copy failure" in result.error
    backups = _mem(tmp_path) / "backups"
    assert not backups.exists() or not list(backups.iterdir())


def test_backup_with_no_memory_is_not_a_failure(tmp_path) -> None:
    result = backup_memory(tmp_path)
    assert result.path is None
    assert not result.failed
    assert result.copied == 0


def test_conserve_restores_missing_entry_before_next(tmp_path) -> None:
    before = _decisions(
        _entry("Alpha", "alpha", "A detail"),
        _entry("Beta", "beta", "B detail"),
    )
    current = _decisions(_entry("Beta", "beta", "B detail"))

    text, restored = conserve_ids(before, current, None, "decisions")

    assert restored == ["alpha"]
    assert "- Alpha" in text
    assert "A detail" in text
    assert text.index("- Alpha") < text.index("- Beta")


def test_conserve_does_not_restore_when_entry_is_archived(tmp_path) -> None:
    before = _decisions(
        _entry("Alpha", "alpha", "A detail"),
        _entry("Beta", "beta", "B detail"),
    )
    current = _decisions(_entry("Beta", "beta", "B detail"))
    archive = (
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n"
        + _entry("Alpha", "alpha", "A detail")
    )

    text, restored = conserve_ids(before, current, archive, "decisions")

    assert restored == []
    assert "- Alpha" not in text


def test_conserve_skips_project_state(tmp_path) -> None:
    before = "# PROJECT_STATE\n\n" + _entry("State A", "state-a", "old")
    current = "# PROJECT_STATE\n\n" + _entry("State B", "state-b", "new")

    text, restored = conserve_ids(before, current, None, "project_state")

    assert restored == []
    assert "State A" not in text


def test_conserve_keeps_crlf(tmp_path) -> None:
    before = (
        "# DECISIONS\r\n\r\n"
        "- Alpha\r\n"
        "  <!--mem id:alpha created:2026-01-01 touched:2026-01-01-->\r\n"
        "  A detail\r\n\r\n"
        "- Beta\r\n"
        "  <!--mem id:beta created:2026-01-01 touched:2026-01-01-->\r\n"
        "  B detail\r\n"
    )
    current = (
        "# DECISIONS\r\n\r\n"
        "- Beta\r\n"
        "  <!--mem id:beta created:2026-01-01 touched:2026-01-01-->\r\n"
        "  B detail\r\n"
    )

    text, restored = conserve_ids(before, current, None, "decisions")

    assert restored == ["alpha"]
    data = text.encode("utf-8")
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b"")


def test_postpass_restores_lost_entry_rebuilds_index_and_records_run(
    tmp_path,
) -> None:
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(
            _entry("Alpha", "alpha", "A detail"),
            _entry("Beta", "beta", "B detail"),
        ),
    )
    before = read_snapshot(tmp_path)
    backup = backup_memory(tmp_path)
    assert backup.path is not None

    # 模拟模型把 Alpha 删掉，却没有写进 archive。
    _write(tmp_path, FILE_NAMES["decisions"], _decisions(_entry("Beta", "beta", "B detail")))

    report = postpass(
        tmp_path,
        before,
        backup_dir=str(backup.path),
        trigger="test",
        calls=2,
        today="2026-11-23",
    )

    assert report.outcome == "restored"
    assert report.restored == ["alpha"]
    assert report.entries_before["decisions"] == 2
    assert report.entries_after["decisions"] == 2
    text = memory_path(tmp_path, FILE_NAMES["decisions"]).read_text(encoding="utf-8")
    assert "- Alpha" in text and "A detail" in text
    assert "- Beta" in text
    index = memory_path(tmp_path, FILE_NAMES["index"]).read_text(encoding="utf-8")
    assert "2 entries" in index

    record = read_run_record(tmp_path)
    assert record is not None
    assert record["outcome"] == "restored"
    assert record["restored"] == ["alpha"]
    assert record["calls"] == 2
    assert record["backup_dir"] == str(backup.path)


def test_postpass_does_not_apply_strict_conservation_to_project_state(
    tmp_path,
) -> None:
    _write(
        tmp_path,
        FILE_NAMES["project_state"],
        "# PROJECT_STATE\n\n" + _entry("State A", "state-a", "old"),
    )
    before = read_snapshot(tmp_path)
    backup = backup_memory(tmp_path)
    assert backup.path is not None

    _write(
        tmp_path,
        FILE_NAMES["project_state"],
        "# PROJECT_STATE\n\n" + _entry("State B", "state-b", "new"),
    )
    report = postpass(
        tmp_path, before, backup_dir=str(backup.path), trigger="test"
    )

    assert report.restored == []
    text = memory_path(tmp_path, FILE_NAMES["project_state"]).read_text(
        encoding="utf-8"
    )
    assert "State A" not in text
    assert "State B" in text


def test_postpass_counts_archived_and_merged(tmp_path) -> None:
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(
            _entry("Alpha", "alpha", "line one\n  line two"),
            _entry("Beta", "beta", "beta detail"),
        ),
    )
    before = read_snapshot(tmp_path)
    backup = backup_memory(tmp_path)
    assert backup.path is not None

    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Alpha", "alpha", "line one")),
    )
    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n"
        + _entry("Beta", "beta", "beta detail"),
    )

    report = postpass(
        tmp_path, before, backup_dir=str(backup.path), trigger="test"
    )

    assert report.archived == ["beta"]
    assert report.merged == ["alpha"]
    assert report.restored_total == 0
    assert report.outcome == "edited"
    index = memory_path(tmp_path, FILE_NAMES["index"]).read_text(encoding="utf-8")
    assert "1 entry" in index
    assert "1 archived decision" in index


def test_postpass_restores_deleted_archive_entry(tmp_path) -> None:
    _write(
        tmp_path,
        FILE_NAMES["decisions"],
        _decisions(_entry("Alpha", "alpha", "a")),
    )
    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n"
        + _entry("Old", "old", "old detail"),
    )
    before = read_snapshot(tmp_path)
    backup = backup_memory(tmp_path)
    assert backup.path is not None

    _write(
        tmp_path,
        ARCHIVE_NAMES["decisions"],
        "# DECISIONS_ARCHIVE\n\n## Superseded\n\n",
    )
    report = postpass(
        tmp_path, before, backup_dir=str(backup.path), trigger="test"
    )

    assert report.restored_archive == ["old"]
    assert report.outcome == "restored"
    archive = memory_path(tmp_path, ARCHIVE_NAMES["decisions"]).read_text(
        encoding="utf-8"
    )
    assert "- Old" in archive
    assert "old detail" in archive


def test_postpass_writes_valid_run_record(tmp_path) -> None:
    _write(tmp_path, FILE_NAMES["decisions"], _decisions(_entry("Alpha", "alpha", "a")))
    before = read_snapshot(tmp_path)
    backup = backup_memory(tmp_path)
    assert backup.path is not None

    report = postpass(tmp_path, before, backup_dir=str(backup.path), trigger="manual")
    record = read_run_record(tmp_path)

    assert record is not None
    assert record["version"] == 1
    assert record["trigger"] == "manual"
    assert record["outcome"] in {"no_change", "index_only"}
    assert record["tokens_before"]["decisions"] > 0
    assert record["tokens_after"]["decisions"] > 0
    assert record["ended_at"]
    assert report.to_dict()["entries_before"] == record["entries_before"]