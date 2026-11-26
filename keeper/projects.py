"""M6 R1：全局项目注册表与项目选择。

注册表是 ``KEEPER_HOME/projects.json``（默认 ``~/.keeper``），只记录
``project_id -> 工作区`` 等元数据；项目数据仍落在各自工作区的 ``.agent/``。
读路径不创建任何全局状态；文件损坏时拒绝写入，交给人工修复。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_VERSION = 1
REGISTRY_FILE = "projects.json"
DEFAULT_WORKER = "built-in"
PROJECT_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


class RegistryError(RuntimeError):
    """注册表读写失败或内容损坏。调用方必须报错，不能覆盖原文件。"""


def keeper_home() -> Path:
    raw = os.getenv("KEEPER_HOME")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / ".keeper"


def registry_path() -> Path:
    return keeper_home() / REGISTRY_FILE


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    root: Path
    workers: tuple[str, ...] = (DEFAULT_WORKER,)
    default_worker: str = DEFAULT_WORKER
    instructions: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "root": str(self.root),
            "workers": list(self.workers),
            "default_worker": self.default_worker,
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: Any, *, key: str) -> "Project":
        where = f"projects.json 损坏：项目 {key}"
        if not isinstance(data, dict):
            raise RegistryError(f"{where} 不是对象")
        for name in ("id", "name", "root", "default_worker"):
            if not isinstance(data.get(name), str) or not data[name]:
                raise RegistryError(f"{where} 的 {name} 缺失或类型错误")
        instructions = data.get("instructions", "")
        if not isinstance(instructions, str):
            raise RegistryError(f"{where} 的 instructions 类型错误")
        raw_workers = data.get("workers")
        if not isinstance(raw_workers, list) or not raw_workers:
            raise RegistryError(f"{where} 的 workers 不是非空数组")
        workers = tuple(raw_workers)
        if any(not isinstance(worker, str) or not worker for worker in workers):
            raise RegistryError(f"{where} 的 workers 含非法项")
        project_id = data["id"]
        if project_id != key:
            raise RegistryError(f"{where} 的 id={project_id!r} 与键不一致")
        if not PROJECT_ID_RE.fullmatch(project_id):
            raise RegistryError(f"{where} 的 id 不合法: {project_id!r}")
        default_worker = data["default_worker"]
        if default_worker not in workers:
            raise RegistryError(
                f"{where} 的 default_worker={default_worker!r} 不在 workers 里"
            )
        return cls(
            id=project_id,
            name=data["name"],
            root=Path(data["root"]),
            workers=workers,
            default_worker=default_worker,
            instructions=instructions,
        )


@dataclass
class Registry:
    projects: dict[str, Project] = field(default_factory=dict)
    current: str | None = None

    def current_project(self) -> Project | None:
        if self.current is None:
            return None
        return self.projects.get(self.current)


def load_registry(path: Path | None = None) -> Registry:
    """缺失文件视为空注册表；损坏一律抛错，绝不静默重置。"""
    path = registry_path() if path is None else path
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Registry()
    except (OSError, UnicodeDecodeError) as exc:
        raise RegistryError(f"无法读取 {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(
            f"projects.json 损坏（{path}）：{exc}。请手工修复；不会覆盖原文件。"
        ) from exc
    if not isinstance(data, dict):
        raise RegistryError(f"projects.json 损坏（{path}）：顶层不是对象")
    if data.get("version") != REGISTRY_VERSION:
        raise RegistryError(
            f"projects.json 版本不支持（{path}）：{data.get('version')!r}，"
            f"期望 {REGISTRY_VERSION}"
        )
    raw_projects = data.get("projects")
    if not isinstance(raw_projects, dict):
        raise RegistryError(f"projects.json 损坏（{path}）：projects 不是对象")
    projects: dict[str, Project] = {}
    for key, item in raw_projects.items():
        if not isinstance(key, str) or not PROJECT_ID_RE.fullmatch(key):
            raise RegistryError(f"projects.json 损坏（{path}）：项目键不合法 {key!r}")
        projects[key] = Project.from_dict(item, key=key)
    current = data.get("current")
    if current is not None:
        if not isinstance(current, str) or current not in projects:
            raise RegistryError(
                f"projects.json 损坏（{path}）：current={current!r} 指向未注册项目"
            )
    return Registry(projects=projects, current=current)


def save_registry(registry: Registry, path: Path | None = None) -> None:
    """原子写；只有调用方已成功加载后才允许走到这里。"""
    path = registry_path() if path is None else path
    data = {
        "version": REGISTRY_VERSION,
        "current": registry.current,
        "projects": {
            project_id: project.to_dict()
            for project_id, project in registry.projects.items()
        },
    }
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
            fh.flush()
        os.replace(tmp, path)
    except Exception as exc:  # 原文件保持不动
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RegistryError(
            f"无法写入 {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "project"


def _validate_project_id(project_id: str) -> None:
    if not PROJECT_ID_RE.fullmatch(project_id):
        raise RegistryError(
            f"项目 id 不合法: {project_id!r}（只允许小写字母、数字、- 和 _）"
        )


def register_project(
    root: Path,
    *,
    project_id: str | None = None,
    name: str | None = None,
    workers: tuple[str, ...] | None = None,
    default_worker: str | None = None,
    instructions: str = "",
) -> Project:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RegistryError(f"项目根目录不存在或不是目录: {root}")

    registry = load_registry()  # 损坏时在这里抛错，不进入写路径
    pid = (project_id or _slugify(root.name)).strip()
    _validate_project_id(pid)
    if pid in registry.projects:
        raise RegistryError(
            f"项目 id 已存在: {pid} ({registry.projects[pid].root})"
        )
    for other in registry.projects.values():
        if other.root == root:
            raise RegistryError(f"该目录已注册为项目 {other.id}: {root}")

    normalized = tuple(worker.strip() for worker in (workers or ()))
    if not normalized:
        normalized = ((default_worker or DEFAULT_WORKER),)
    if any(not worker for worker in normalized):
        raise RegistryError("workers 不能包含空项")
    default = default_worker or normalized[0]
    if default not in normalized:
        raise RegistryError(
            f"默认 worker {default!r} 不在启用的 worker 里: {', '.join(normalized)}"
        )
    display_name = (name or pid).strip() or pid
    project = Project(
        id=pid,
        name=display_name,
        root=root,
        workers=normalized,
        default_worker=default,
        instructions=instructions or "",
    )
    registry.projects[pid] = project
    save_registry(registry)
    return project


def use_project(project_id: str) -> Project:
    registry = load_registry()
    project = registry.projects.get(project_id)
    if project is None:
        known = ", ".join(sorted(registry.projects)) or "(none)"
        raise RegistryError(f"未注册的项目: {project_id}；已注册: {known}")
    registry.current = project_id
    save_registry(registry)
    return project


def current_project() -> Project | None:
    return load_registry().current_project()