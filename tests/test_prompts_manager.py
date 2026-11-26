"""M6 R2：管理者 system prompt。"""

from __future__ import annotations

from pathlib import Path

from keeper.prompts import manager_system_prompt
from keeper.projects import Project
from keeper.workers import default_workers


def _project(root: Path, instructions: str = "") -> Project:
    return Project(
        id="demo",
        name="演示项目",
        root=root,
        workers=("built-in", "claude-code"),
        default_worker="built-in",
        instructions=instructions,
    )


def test_prompt_has_root_roster_and_manager_duties(tmp_path: Path) -> None:
    prompt = manager_system_prompt(tmp_path, _project(tmp_path), default_workers())
    assert str(tmp_path.resolve()) in prompt
    assert "演示项目" in prompt
    for worker in default_workers():
        assert worker.slug in prompt
        assert worker.routing_hint in prompt
    assert "自己处理小事" in prompt
    assert "编码和长任务优先派给 worker" in prompt
    assert "不可逆操作仍需用户审批" in prompt
    assert "worker 不可再派 worker" in prompt
    assert "不按固定打分规则选择" in prompt


def test_instructions_precede_roster_and_declare_precedence(
    tmp_path: Path,
) -> None:
    prompt = manager_system_prompt(
        tmp_path, _project(tmp_path, instructions="先跑 pytest，再改代码"), default_workers()
    )
    assert "先跑 pytest，再改代码" in prompt
    assert "部署指令优先于名册" in prompt
    assert prompt.index("先跑 pytest，再改代码") < prompt.index("worker 名册：")


def test_no_instructions_section_when_empty(tmp_path: Path) -> None:
    prompt = manager_system_prompt(tmp_path, _project(tmp_path), default_workers())
    assert "项目部署指令" not in prompt


def test_memory_is_reread_on_every_call(tmp_path: Path) -> None:
    memory_dir = tmp_path / ".agent" / "memory"
    memory_dir.mkdir(parents=True)
    decisions = memory_dir / "DECISIONS.md"
    decisions.write_text("- 第一条决策\n", encoding="utf-8")

    first = manager_system_prompt(tmp_path, _project(tmp_path), default_workers())
    assert "第一条决策" in first

    decisions.write_text("- 第二条决策\n", encoding="utf-8")
    second = manager_system_prompt(tmp_path, _project(tmp_path), default_workers())
    assert "第二条决策" in second
    assert "第一条决策" not in second