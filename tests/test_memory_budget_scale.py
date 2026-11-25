"""M5.1b：预算按 worker 上下文窗口缩放（只影响预算，不改口径）。"""

import keeper.memory as memory
from keeper.memory import (
    FILE_BUDGETS,
    FILE_NAMES,
    budget_scale,
    effective_budgets,
    memory_tokens,
    over_soft_budget,
    render_memory,
    strip_mem_comments,
)
from keeper.prompts import system_prompt
from keeper.tidy import build_tidy_message, check_due, format_check, read_snapshot


def _write(root, name: str, text: str) -> None:
    path = root / ".agent" / "memory" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _entry(title: str, entry_id: str, detail: str) -> str:
    return (
        f"- {title}\n"
        f"  <!--mem id:{entry_id} created:2026-01-01 touched:2026-01-01-->\n"
        f"  {detail}\n"
    )


def _bulk_decisions(lines: int = 50) -> str:
    body = "\n".join(
        _entry(f"Entry {index}", f"e{index}", "x" * 150) for index in range(lines)
    )
    return "# DECISIONS\n\n" + body


def _mid_sized_decisions(lines: int = 100) -> str:
    body = "\n".join(
        _entry(f"E{index}", f"e{index}", "x" * 60) for index in range(lines)
    )
    return "# DECISIONS\n\n" + body


def test_budget_scale_uses_default_window_and_clamps() -> None:
    assert budget_scale(None) == 1.0
    assert budget_scale(0) == 1.0
    assert budget_scale(65536) == 1.0
    assert budget_scale(32768) == 0.5
    assert budget_scale(131072) == 2.0
    assert budget_scale(1) == 0.5
    assert budget_scale(10**9) == 2.0
    assert budget_scale("bad") == 1.0


def test_effective_budgets_scale_soft_and_hard() -> None:
    assert effective_budgets(None) == FILE_BUDGETS
    small = effective_budgets(32768)
    assert small["decisions"]["soft"] == 1500
    assert small["decisions"]["hard"] == 2250
    large = effective_budgets(131072)
    assert large["decisions"]["soft"] == 6000
    assert large["decisions"]["hard"] == 9000
    for budgets in (small, large):
        for key, value in budgets.items():
            assert value["hard"] > value["soft"] > 0


def test_over_soft_budget_follows_scaled_budget() -> None:
    # ~2200 tok：低于默认 soft(3000)，高于 32768 窗口的 soft(1500)。
    text = _mid_sized_decisions()
    tokens = memory_tokens(text)
    assert 1500 < tokens < 3000

    assert not over_soft_budget(text, "decisions")
    assert over_soft_budget(text, "decisions", 32768)


def test_inject_view_uses_scaled_hard_budget() -> None:
    from keeper.memory import inject_view

    text = _bulk_decisions()
    tokens = memory_tokens(text)
    assert 2250 < tokens < 4500

    budgets = effective_budgets(None)
    default_view = inject_view(text, "decisions", budgets["decisions"]["hard"])
    small_budgets = effective_budgets(32768)
    small_view = inject_view(
        text, "decisions", small_budgets["decisions"]["hard"]
    )

    assert "x" * 150 in default_view
    # 小窗口超 hard 后先丢 detail，条目仍在；预算口径不变。
    assert "x" * 150 not in small_view
    assert "- Entry 49" in small_view


def test_system_prompt_and_render_memory_scale_with_window(tmp_path) -> None:
    text = _bulk_decisions()
    path = tmp_path / ".agent" / "memory" / FILE_NAMES["decisions"]
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")

    full = render_memory(tmp_path)
    small = render_memory(tmp_path, context_window=32768)
    assert "x" * 150 in full
    assert "x" * 150 not in small

    prompt_full = system_prompt(tmp_path)
    prompt_small = system_prompt(tmp_path, context_window=32768)
    assert "x" * 150 in prompt_full
    assert "x" * 150 not in prompt_small
    assert "<!--mem" not in strip_mem_comments(small)


def test_check_due_and_format_check_use_scaled_soft(tmp_path) -> None:
    text = _mid_sized_decisions()
    _write(tmp_path, FILE_NAMES["decisions"], text)

    base = check_due(tmp_path)
    small = check_due(tmp_path, context_window=32768)
    assert not base.due  # 默认 soft 3000 下没超
    assert small.due
    assert small.over_soft == ["decisions"]

    text_out = format_check(tmp_path, small, None, 32768)
    assert "/1500 soft" in text_out
    assert "2250 hard" in text_out


def test_build_tidy_message_reports_scaled_budgets(tmp_path) -> None:
    text = _mid_sized_decisions()
    _write(tmp_path, FILE_NAMES["decisions"], text)

    message = build_tidy_message(
        read_snapshot(tmp_path),
        trigger="test",
        today="2026-11-23",
        context_window=32768,
    )

    assert "(soft 1500, hard 2250)" in message
    assert "OVER SOFT" in message

    default_message = build_tidy_message(
        read_snapshot(tmp_path), trigger="test", today="2026-11-23"
    )
    assert "(soft 3000, hard 4500)" in default_message


def test_budget_helpers_do_not_mutate_module_budgets() -> None:
    before = {key: dict(value) for key, value in FILE_BUDGETS.items()}
    effective_budgets(32768)
    effective_budgets(131072)
    assert FILE_BUDGETS == before