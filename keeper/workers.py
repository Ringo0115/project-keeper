"""M6 R3：worker 名册与 R11 私有记忆路径。

名册是静态数据，渲染进管理者 prompt，由模型选择；不做打分路由，也不做
CLI 可用性探测。R11 这里只给出路径，写入限制随 R4 落地。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from keeper.projects import DEFAULT_WORKER, Project

WORKER_KINDS = frozenset({"built-in", "cli"})
WORKER_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


class WorkerError(RuntimeError):
    """名册或 worker 路径不合法。调用方必须报错，不能静默跳过。"""


@dataclass(frozen=True)
class WorkerSpec:
    slug: str
    name: str
    kind: str
    description: str
    routing_hint: str
    permissions: str
    autonomy: str
    available: bool = True

    def __post_init__(self) -> None:
        if WORKER_SLUG_RE.fullmatch(self.slug) is None:
            raise WorkerError(f"worker slug 不合法: {self.slug!r}")
        for attr_name in ("name", "description", "routing_hint", "permissions", "autonomy"):
            if not isinstance(getattr(self, attr_name), str) or not getattr(self, attr_name):
                raise WorkerError(f"worker {self.slug} 的 {attr_name} 不能为空")
        if not isinstance(self.available, bool):
            raise WorkerError(f"worker {self.slug} 的 available 必须是布尔值")
        if self.kind not in WORKER_KINDS:
            raise WorkerError(
                f"worker {self.slug} 的 kind={self.kind!r} 不合法；"
                f"允许: {', '.join(sorted(WORKER_KINDS))}"
            )


DEFAULT_WORKER_SPECS: tuple[WorkerSpec, ...] = (
    WorkerSpec(
        slug="built-in",
        name="内置 loop",
        kind="built-in",
        description="内置 loop，使用 keeper 当前模型与工具，在本进程内运行。",
        routing_hint="小改动、读文件、短问答；没有外部 CLI 需求时优先。",
        permissions="keeper 工具；写文件 / shell 需审批。",
        autonomy="写入需人工确认。",
    ),
    WorkerSpec(
        slug="claude-code",
        name="Claude Code",
        kind="cli",
        description="Claude Code CLI，适合编码、重构与长任务。",
        routing_hint="需要项目内编码或较长任务时优先。",
        permissions="由 CLI 自身配置；派发时限定可写范围。",
        autonomy="可在 brief 范围内自主执行。",
    ),
    WorkerSpec(
        slug="codex",
        name="Codex",
        kind="cli",
        description="Codex CLI，适合代码实现与测试修复。",
        routing_hint="需要独立实现与验证代码时使用。",
        permissions="由 CLI 自身配置；派发时限定可写范围。",
        autonomy="可在 brief 范围内自主执行。",
    ),
    WorkerSpec(
        slug="pi",
        name="pi",
        kind="cli",
        description="外部 pi agent CLI，适合已有 pi 工作流。",
        routing_hint="用户指定或项目已有 pi 工作流时使用。",
        permissions="由 CLI 自身配置；派发时限定可写范围。",
        autonomy="由自身配置决定。",
    ),
)


def default_workers() -> tuple[WorkerSpec, ...]:
    return DEFAULT_WORKER_SPECS


def roster_for(project: Project) -> list[WorkerSpec]:
    """按项目启用的 slug 过滤名册；空列表按 R1 默认值处理。"""
    slugs = tuple(project.workers) or (DEFAULT_WORKER,)
    by_slug = {worker.slug for worker in DEFAULT_WORKER_SPECS}
    unknown = [slug for slug in slugs if slug not in by_slug]
    if unknown:
        raise WorkerError(
            f"项目 {project.id} 启用了未知 worker: {', '.join(unknown)}；"
            f"可用: {', '.join(sorted(by_slug))}"
        )
    return [worker for worker in DEFAULT_WORKER_SPECS if worker.slug in slugs]


def render_roster(workers: Iterable[WorkerSpec]) -> str:
    """渲染成管理者 prompt 用的纯文本；不排序，保留传入顺序。"""
    specs = list(workers)
    if not specs:
        return "worker 名册：\n(没有启用的 worker)"
    lines = ["worker 名册："]
    for worker in specs:
        suffix = "" if worker.available else " · 不可用（静态）"
        lines.append(f"- {worker.slug}（{worker.name}，{worker.kind}）{suffix}")
        lines.append(f"  擅长：{worker.description}")
        lines.append(f"  路由：{worker.routing_hint}")
        lines.append(f"  权限：{worker.permissions}")
        lines.append(f"  建议 autonomy：{worker.autonomy}")
    return "\n".join(lines)


def worker_memory_path(root: Path, slug: str) -> Path:
    """返回 ``.agent/sub_agents/<slug>/MEMORY.md``；只算路径，不建目录。"""
    if not isinstance(slug, str) or WORKER_SLUG_RE.fullmatch(slug) is None:
        raise WorkerError(
            f"worker slug 不合法: {slug!r}（只允许小写字母、数字、- 和 _）"
        )
    return (
        Path(root).expanduser().resolve()
        / ".agent"
        / "sub_agents"
        / slug
        / "MEMORY.md"
    )