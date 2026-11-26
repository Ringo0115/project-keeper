"""CLI 入口：解析参数、渲染事件、交互模式、M3 任务子命令。"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path
from typing import Iterable

from keeper import __version__
from keeper.approval import StdioApprover
from keeper.config import Config, load_config
from keeper.events import (
    ApprovalRequest,
    ApprovalResult,
    AssistantText,
    CompactionDone,
    CompactionFailed,
    CompactionStart,
    ErrorEvent,
    Event,
    FinalAnswer,
    SessionStart,
    StepStart,
    Stop,
    TaskEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from keeper.llm import LLMClient
from keeper.loop import AgentLoop
from keeper.projects import (
    RegistryError,
    current_project,
    load_registry,
    register_project,
    use_project,
)
from keeper.prompts import manager_system_prompt
from keeper.workers import WorkerError, default_workers, roster_for
from keeper.session import Session
from keeper.task_runner import (
    TaskRunError,
    prepare_task_run,
    run_task_once,
    run_task_worker,
)
from keeper.tasks import Task, TaskStore, TaskStoreError
from keeper.task_lock import TaskLock
from keeper.tidy import (
    check_due,
    format_check,
    read_run_record,
    run_record_round,
    run_tidy,
)
from keeper.tools.checkpoint import RequestCheckpointTool
from keeper.tools.dispatch import manager_dispatch_tools
from keeper.tools.fs import ListDirTool, ReadFileTool, SearchTool, WriteFileTool
from keeper.tools.memory import UpdateMemoryTool
from keeper.tools.registry import Registry
from keeper.tools.shell import ShellTool


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _short(text: str, limit: int) -> str:
    text = text.replace("\r", "").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(events: Iterable[Event]) -> None:
    for event in events:
        if isinstance(event, SessionStart):
            tag = " (resumed)" if event.resumed else ""
            print(f"keeper {__version__} · model={event.model} · session={event.session_id}{tag}")
            print(f"session file: {event.path}\n")
        elif isinstance(event, StepStart):
            print(f"[step {event.step}]", file=sys.stderr)
        elif isinstance(event, ToolCallEvent):
            args = json.dumps(event.args, ensure_ascii=False)
            print(f"  -> {event.name} {_short(args, 160)}", file=sys.stderr)
        elif isinstance(event, ToolResultEvent):
            mark = "ok" if event.ok else "ERR"
            first = event.output.splitlines()[0] if event.output else ""
            print(f"  <- {mark} {_short(first, 110)} ({event.duration_ms}ms)", file=sys.stderr)
        elif isinstance(event, ApprovalRequest):
            print(f"\n[approval] {event.summary}", file=sys.stderr)
            for line in event.detail.splitlines():
                print(f"    {line}", file=sys.stderr)
        elif isinstance(event, ApprovalResult):
            state = "approved" if event.approved else "denied"
            reason = f" ({event.reason})" if event.reason else ""
            print(f"[approval] {state}{reason}", file=sys.stderr)
        elif isinstance(event, AssistantText):
            if event.text.strip():
                print(event.text)
                print()
        elif isinstance(event, FinalAnswer):
            print(event.text if event.text.strip() else "(模型没有输出文本)")
        elif isinstance(event, ErrorEvent):
            print(f"error: {event.message}", file=sys.stderr)
        elif isinstance(event, Stop):
            print(f"[stopped: {event.reason}]", file=sys.stderr)
        elif isinstance(event, CompactionStart):
            print(
                f"[compaction] start · {event.tokens_before} tokens · "
                f"cut={event.cut_index}",
                file=sys.stderr,
            )
        elif isinstance(event, CompactionDone):
            suffix = " · degraded" if event.degraded else ""
            print(
                f"[compaction] done · {event.tokens_before} -> "
                f"{event.tokens_after} tokens{suffix}",
                file=sys.stderr,
            )
        elif isinstance(event, CompactionFailed):
            detail = event.error or event.reason
            print(f"[compaction] failed: {_short(detail, 160)}", file=sys.stderr)
        elif isinstance(event, TaskEvent):
            line = f"[task] {event.action} {event.task_id} status={event.status}"
            if event.detail:
                line += f" · {event.detail}"
            print(line, file=sys.stderr)


def _resolve_root(explicit: Path | None) -> Path:
    """--root 优先；否则已选项目根目录；再否则当前目录。

    显式 --root 与已选项目不同时只提示，不静默改写：数据仍只写入实际
    使用的根目录。注册表损坏时回退当前目录并提示，不覆盖原文件。
    """
    if explicit is not None:
        root = explicit.expanduser().resolve()
        try:
            selected = current_project()
        except RegistryError:
            selected = None
        if selected is not None and selected.root != root:
            print(
                f"note: --root {root} 覆盖已选项目 {selected.id}"
                f"（{selected.root}）；数据只写入 --root",
                file=sys.stderr,
            )
        return root

    try:
        selected = current_project()
    except RegistryError as exc:
        cwd = Path.cwd().resolve()
        print(
            f"warning: {exc}；回退到当前目录 {cwd}",
            file=sys.stderr,
        )
        return cwd
    if selected is not None:
        return selected.root
    return Path.cwd().resolve()


def _build_registry(root: Path) -> Registry:
    return Registry(
        [
            ReadFileTool(root),
            ListDirTool(root),
            WriteFileTool(root),
            SearchTool(root),
            UpdateMemoryTool(root),
            RequestCheckpointTool(root),
            ShellTool(root),
        ]
    )


def _manager_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="keeper manager", description="项目管理 agent（M6）")
    parser.add_argument("message", nargs="*", help="给管理者的任务；省略则进入交互模式")
    parser.add_argument("--root", type=Path, default=None, help="项目根目录；默认已选项目，否则当前目录")
    parser.add_argument("--model", help="覆盖模型名")
    parser.add_argument("--max-steps", type=int, help="单次运行最大步数")
    parser.add_argument("--no-tidy", action="store_true", help="本次运行不自动记录/整理记忆")
    args = parser.parse_args(argv)
    root = _resolve_root(args.root)
    try:
        registry_data = load_registry()
    except RegistryError as exc:
        print(f"error: 无法读取项目注册表：{exc}", file=sys.stderr)
        return 2
    project = next((p for p in registry_data.projects.values() if p.root.resolve() == root), None)
    if project is None:
        print(f"error: 管理者需要已注册项目；请先运行 keeper project register {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2
    config = load_config(model=args.model, max_steps=args.max_steps)
    if not config.api_key:
        print(_api_key_error(), file=sys.stderr)
        return 2
    session = Session.create(root, config.model)
    llm = LLMClient(base_url=config.base_url, api_key=config.api_key, model=config.model,
                    temperature=config.temperature)
    approver = StdioApprover(timeout=config.approval_timeout)
    tools = [ReadFileTool(root), ListDirTool(root), WriteFileTool(root), SearchTool(root),
             UpdateMemoryTool(root), RequestCheckpointTool(root), ShellTool(root),
             *manager_dispatch_tools(project, approver, llm=llm, max_steps=config.max_steps)]
    loop = AgentLoop(llm=llm, registry=Registry(tools), session=session, root=root,
                     max_steps=config.max_steps, approver=approver,
                     context_window=config.context_window, reserve_tokens=config.reserve_tokens,
                     keep_tokens=config.keep_tokens,
                     system_override=manager_system_prompt(root, project, roster_for(project), config.context_window))
    try:
        if args.message:
            render(loop.run(" ".join(args.message)))
            _maybe_auto_tidy(root, config, llm, "manager", no_tidy=args.no_tidy,
                             session_id=session.session_id)
            return 0
        render([SessionStart(session_id=session.session_id, path=str(session.path), model=config.model)])
        rc, clean_exit = _repl(loop, session)
        if clean_exit:
            _maybe_auto_tidy(root, config, llm, "manager", no_tidy=args.no_tidy,
                             session_id=session.session_id)
        return rc
    except KeyboardInterrupt:
        print("\ninterrupted; session saved.", file=sys.stderr)
        return 130
    finally:
        llm.close()
        session.close()


def _open_session(
    args: argparse.Namespace, root: Path, model: str
) -> tuple[Session, bool]:
    if args.session:
        path = args.session.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(path)
        return Session.load(path, root), True

    if args.resume:
        latest = Session.latest(root)
        if latest is None:
            raise FileNotFoundError(root / ".agent" / "sessions")
        return Session.load(latest, root), True

    return Session.create(root, model), False


def _repl(loop: AgentLoop, session: Session) -> tuple[int, bool]:
    """交互模式；返回 ``(exit_code, clean_exit)``。

    只有 ``/exit``、``/quit`` 或 EOF 算 clean_exit；输入提示处 Ctrl+C
    不算，调用方据此决定是否在会话结束后自动整理记忆。
    """
    print("输入任务，/exit 退出，/session 查看会话文件。\n")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            print()
            return 0, True
        except KeyboardInterrupt:
            print()
            return 0, False

        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0, True
        if text == "/session":
            print(session.path)
            continue
        if text.startswith("/"):
            print("未知命令。可用：/exit, /session")
            continue

        try:
            render(loop.run(text))
        except KeyboardInterrupt:
            print("\ninterrupted; 可以继续输入下一个任务。", file=sys.stderr)


# ---------------------------------------------------------------------------
# M3 任务队列
# ---------------------------------------------------------------------------


def _load_store(root: Path) -> TaskStore | None:
    try:
        return TaskStore.load(root)
    except TaskStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _load_recovered_store(root: Path) -> TaskStore | None:
    """add / list / run / --resume 共用：加载后先恢复一次，只提示真的变化。"""
    store = _load_store(root)
    if store is None:
        return None
    render(_task_events(store.recover()))
    return store


def _task_events(changes: Iterable) -> list[TaskEvent]:
    return [
        TaskEvent(
            action=change.action,
            task_id=change.task_id,
            status=change.status,
            detail=change.detail,
        )
        for change in changes
    ]


def _api_key_error() -> str:
    return (
        "error: 没有找到 API key。\n"
        "请在项目根目录创建 .env（参考 .env.example），"
        "或设置环境变量 KEEPER_API_KEY / DEEPSEEK_API_KEY。"
    )


def _print_task_list(store: TaskStore) -> None:
    if not store.tasks:
        print("(no tasks)")
        return
    for task in store.tasks:
        last = task.latest_outcome or "-"
        line = (
            f"{task.id}  {task.status:<7}  attempts={len(task.attempts)}  "
            f"last={last}  {task.title}"
        )
        if task.block_reason:
            line += f"  · blocked: {_short(task.block_reason, 160)}"
        if task.worker:
            line += f"  · worker={task.worker}"
        print(line)


def _execute_task(
    root: Path,
    store: TaskStore,
    task: Task,
    config: Config,
    *,
    no_tidy: bool = False,
) -> int:
    try:
        prepared = prepare_task_run(store, task.id, root, config.model)
    except (TaskRunError, TaskStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    llm: LLMClient | None = None
    try:
        llm = LLMClient(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
        )
        if prepared.repaired:
            print(
                f"note: 修复了 {prepared.repaired} 个中断的工具调用（已写入会话）",
                file=sys.stderr,
            )
        start = SessionStart(
            session_id=prepared.session.session_id,
            path=str(prepared.session.path),
            resumed=prepared.resumed,
            model=config.model,
        )
        events = chain(
            [start],
            run_task_once(
                prepared,
                llm=llm,
                registry=_build_registry(root),
                approver=StdioApprover(timeout=config.approval_timeout),
                max_steps=config.max_steps,
                context_window=config.context_window,
                reserve_tokens=config.reserve_tokens,
                keep_tokens=config.keep_tokens,
            ),
        )
        try:
            render(events)
        except KeyboardInterrupt:
            print(
                "\ninterrupted; attempt 保持打开，下次启动会标记 interrupted。",
                file=sys.stderr,
            )
            return 130
        _maybe_auto_tidy(
            root,
            config,
            llm,
            "task",
            no_tidy=no_tidy,
            session_id=prepared.session.session_id,
        )
        return 0
    finally:
        if llm is not None:
            llm.close()
        prepared.session.close()


def _registered_project(root: Path):
    """按根目录找已注册项目；注册表损坏时提示并返回 None。"""
    try:
        registry_data = load_registry()
    except RegistryError as exc:
        print(f"warning: 无法读取项目注册表：{exc}", file=sys.stderr)
        return None
    resolved = Path(root).resolve()
    return next(
        (p for p in registry_data.projects.values() if p.root.resolve() == resolved),
        None,
    )


def _worker_specs(root: Path):
    """项目启用的 worker；未注册项目退到默认名册。"""
    project = _registered_project(root)
    if project is not None:
        return roster_for(project)
    return list(default_workers())


def _resolve_worker_slug(root: Path, slug: str) -> str:
    """校验 worker 名册；非法值抛 WorkerError，调用方明确报错。"""
    available = {spec.slug for spec in _worker_specs(root)}
    if slug not in available:
        raise WorkerError(
            f"未知或未启用的 worker: {slug!r}；可用: {', '.join(sorted(available))}"
        )
    return slug


def _execute_task_worker(
    root: Path,
    store: TaskStore,
    task: Task,
    config: Config,
    worker_slug: str,
    *,
    retry: bool,
    no_tidy: bool = False,
) -> int:
    """把队列项交给 worker：指派执行或阻塞后重试。"""
    project = _registered_project(root)
    if project is None:
        print(
            "error: worker 派发需要已注册项目；"
            f"请先运行 keeper project register {root}",
            file=sys.stderr,
        )
        return 2
    try:
        specs = {spec.slug: spec for spec in roster_for(project)}
    except WorkerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    spec = specs.get(worker_slug)
    if spec is None:
        print(
            f"error: worker {worker_slug!r} 不在项目 {project.id} 启用的名册里；"
            f"可用: {', '.join(sorted(specs))}",
            file=sys.stderr,
        )
        return 2

    llm: LLMClient | None = None
    try:
        if spec.kind == "built-in":
            if not config.api_key:
                print(_api_key_error(), file=sys.stderr)
                return 2
            llm = LLMClient(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
            )
        record = run_task_worker(
            project,
            store,
            task.id,
            worker_slug,
            retry=retry,
            approver=StdioApprover(timeout=config.approval_timeout),
            llm=llm,
            max_steps=config.max_steps,
        )
        print(
            f"task {task.id} · worker={record.worker} · result={record.result}"
        )
        print(f"summary: {_short(record.summary, 300)}")
        if record.transcript:
            print(f"transcript: {record.transcript}")
        if _tidy_enabled(config, no_tidy):
            record_llm = llm
            own_llm = False
            if record_llm is None and config.api_key:
                record_llm = LLMClient(base_url=config.base_url, api_key=config.api_key,
                                       model=config.model, temperature=config.temperature)
                own_llm = True
            if record_llm is not None:
                _maybe_auto_tidy(root, config, record_llm, "task-worker",
                                 no_tidy=no_tidy, session_id=None)
                if own_llm:
                    record_llm.close()
            else:
                print("[memory] record warning: no API key for memory record round", file=sys.stderr)
        return 0
    except (TaskRunError, TaskStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if llm is not None:
            llm.close()


def _task_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="keeper task", description="持久任务队列（M3）")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        default=None,
        help="项目根目录；默认已选项目，否则当前目录",
    )
    common.add_argument("--model", help="覆盖模型名")
    common.add_argument("--max-steps", type=int, help="单个任务的最大步数")

    sub = parser.add_subparsers(dest="command", required=True)
    add_parser = sub.add_parser("add", parents=[common], help="入队一个任务")
    add_parser.add_argument("title", nargs="+", help="任务标题")
    add_parser.add_argument("--worker", help="指派 worker（须在项目名册里）")
    sub.add_parser("list", parents=[common], help="列出任务")
    run_parser = sub.add_parser("run", parents=[common], help="运行队首或指定任务")
    run_parser.add_argument("id", nargs="?", help="任务 id；省略则跑队首 queued 任务")
    run_parser.add_argument(
        "--no-tidy",
        action="store_true",
        help="本次 attempt 结束后不自动整理项目记忆",
    )
    retry_parser = sub.add_parser(
        "retry", parents=[common], help="重跑阻塞或失败的任务，可换 worker"
    )
    retry_parser.add_argument("id", help="任务 id")
    retry_parser.add_argument("--worker", help="换一个 worker；省略则沿用原指派")
    retry_parser.add_argument("--no-tidy", action="store_true",
                              help="本次 attempt 结束后不自动记录/整理项目记忆")

    args = parser.parse_args(argv)

    root = _resolve_root(args.root)
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2

    store = _load_recovered_store(root)
    if store is None:
        return 2

    if args.command == "add":
        worker = None
        if args.worker:
            try:
                worker = _resolve_worker_slug(root, args.worker.strip())
            except WorkerError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
        try:
            task = store.add(" ".join(args.title), worker=worker)
        except TaskStoreError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        render(
            [
                TaskEvent(
                    action="add",
                    task_id=task.id,
                    status=task.status,
                    detail=task.title,
                )
            ]
        )
        print(task.id)
        return 0

    if args.command == "list":
        _print_task_list(store)
        return 0

    if args.command == "retry":
        task = store.get(args.id)
        if task is None:
            print(f"error: 任务不存在: {args.id}", file=sys.stderr)
            return 2
        if task.status != "blocked":
            print(
                f"error: 任务 {task.id} 当前状态为 {task.status}；"
                "只有 blocked（阻塞或失败）可以重试",
                file=sys.stderr,
            )
            return 2
        worker_slug = args.worker.strip() if args.worker else task.worker
        if worker_slug is None:
            project = _registered_project(root)
            if project is None:
                print(
                    "error: 需要 --worker，或先注册项目以使用 default_worker",
                    file=sys.stderr,
                )
                return 2
            worker_slug = project.default_worker
        try:
            worker_slug = _resolve_worker_slug(root, worker_slug)
        except WorkerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        config = load_config(model=args.model, max_steps=args.max_steps)
        return _execute_task_worker(
            root, store, task, config, worker_slug, retry=True,
            no_tidy=args.no_tidy,
        )

    # run
    config = load_config(model=args.model, max_steps=args.max_steps)
    no_tidy = args.no_tidy

    if args.id:
        task = store.get(args.id)
        if task is None:
            print(f"error: 任务不存在: {args.id}", file=sys.stderr)
            return 2
    else:
        task = store.next_queued()
        if task is None:
            print("error: 没有 queued 任务", file=sys.stderr)
            return 1

    if task.status != "queued":
        print(
            f"error: 任务 {task.id} 当前状态为 {task.status}，只有 queued 可以运行",
            file=sys.stderr,
        )
        return 2

    if task.worker:
        return _execute_task_worker(
            root, store, task, config, task.worker, retry=False,
            no_tidy=args.no_tidy,
        )

    if not config.api_key:
        print(_api_key_error(), file=sys.stderr)
        return 2

    _startup_tidy_hint(
        root, _tidy_enabled(config, no_tidy), config.context_window
    )
    return _execute_task(root, store, task, config, no_tidy=no_tidy)


# ---------------------------------------------------------------------------
# M6 R1 项目注册表
# ---------------------------------------------------------------------------


def _project_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="keeper project", description="项目注册表与项目选择（M6 R1）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add_parser = sub.add_parser("register", help="注册一个项目工作区")
    add_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="项目根目录，默认当前目录",
    )
    add_parser.add_argument("--id", dest="project_id", help="项目 id，默认由目录名生成")
    add_parser.add_argument("--name", help="显示名，默认用 id")
    add_parser.add_argument(
        "--worker", action="append", dest="workers", help="启用的 worker，可重复"
    )
    add_parser.add_argument("--default-worker", help="默认 worker")
    add_parser.add_argument("--instructions", default="", help="项目级部署指令")

    sub.add_parser("list", help="列出已注册项目")
    show_parser = sub.add_parser("show", help="查看一个项目")
    show_parser.add_argument("id")
    use_parser = sub.add_parser("use", help="选择当前项目")
    use_parser.add_argument("id")

    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            project = register_project(
                args.path,
                project_id=args.project_id,
                name=args.name,
                workers=tuple(args.workers) if args.workers else None,
                default_worker=args.default_worker,
                instructions=args.instructions,
            )
            print(f"registered {project.id}  {project.name}  {project.root}")
            return 0

        if args.command == "list":
            registry = load_registry()
            if not registry.projects:
                print("(no projects)")
                return 0
            for pid in sorted(registry.projects):
                project = registry.projects[pid]
                mark = "*" if registry.current == pid else " "
                print(
                    f"{mark} {project.id}  {project.name}  {project.root}"
                    f"  workers={','.join(project.workers)}"
                    f"  default={project.default_worker}"
                )
            return 0

        if args.command == "show":
            project = load_registry().projects.get(args.id)
            if project is None:
                raise RegistryError(f"未注册的项目: {args.id}")
            print(f"id: {project.id}")
            print(f"name: {project.name}")
            print(f"root: {project.root}")
            print(f"workers: {', '.join(project.workers)}")
            print(f"default_worker: {project.default_worker}")
            print(f"instructions: {project.instructions}")
            return 0

        project = use_project(args.id)
        print(f"current project: {project.id}  {project.root}")
        return 0
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# M5 记忆治理
# ---------------------------------------------------------------------------


def _memory_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="keeper memory", description="项目记忆治理（M5）"
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        default=None,
        help="项目根目录；默认已选项目，否则当前目录",
    )
    common.add_argument("--model", help="覆盖模型名")

    sub = parser.add_subparsers(dest="command", required=True)
    tidy_parser = sub.add_parser(
        "tidy", parents=[common], help="整理记忆（force，跳 due 闸门）"
    )
    tidy_parser.add_argument(
        "--check",
        action="store_true",
        help="只读查看 due / 预算 / 上次报告，不整理",
    )
    args = parser.parse_args(argv)
    root = _resolve_root(args.root)
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2

    config = load_config(model=args.model)
    context_window = config.context_window

    if args.check:
        try:
            decision = check_due(root, context_window=context_window)
            report = read_run_record(root)
        except Exception as exc:
            print(
                f"error: cannot check memory: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
        print(format_check(root, decision, report, context_window))
        return 0

    if not config.api_key:
        print(_api_key_error(), file=sys.stderr)
        return 2

    llm = LLMClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
    )
    try:
        report = run_tidy(
            root, llm, force=True, trigger="manual", context_window=context_window
        )
    except Exception as exc:
        print(
            f"error: tidy failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    finally:
        llm.close()

    print(report.text())
    if report.error:
        print(f"error: {report.error}", file=sys.stderr)
    if report.outcome == "failed":
        return 2
    if report.outcome == "locked":
        print("[memory] tidy lock busy; retry later", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# 自动整理（M5.0 切片 5）
# ---------------------------------------------------------------------------


def _tidy_enabled(config: Config, no_tidy: bool) -> bool:
    return config.memory_auto and not no_tidy


def _startup_tidy_hint(
    root: Path, enabled: bool, context_window: int | None = None
) -> None:
    """启动时的廉价 due 检查：只读、只提示，不同步整理、不调用 LLM。"""
    if not enabled:
        return
    try:
        decision = check_due(root, context_window=context_window)
    except Exception:
        return
    if decision.due:
        print(f"[memory] tidy due: {decision.reason_text()}", file=sys.stderr)


def _maybe_auto_tidy(
    root: Path,
    config: Config,
    llm: LLMClient,
    trigger: str,
    *,
    no_tidy: bool = False,
    session_id: str | None = None,
) -> None:
    """会话 / attempt 正常结束后先同步记录，再按 due 闸门整理。

    跨 CLI 的“整理不抢占其它会话”仍没有可靠的低成本准入机制，所以这里
    保持同步：只在本进程的 CLI 边界跑，不拉起后台进程。
    """
    if not _tidy_enabled(config, no_tidy):
        return
    try:
        record = run_record_round(
            root, llm, trigger=trigger, session_id=session_id
        )
        if record.outcome == "locked":
            print("[memory] record skipped: tidy lock busy", file=sys.stderr)
        elif record.outcome == "failed":
            print(f"[memory] record warning: {record.error or 'failed'}", file=sys.stderr)
        else:
            print(f"[memory] record {record.outcome}", file=sys.stderr)
        report = run_tidy(
            root,
            llm,
            force=False,
            trigger=trigger,
            session_id=session_id,
            context_window=config.context_window,
        )
    except Exception as exc:
        print(
            f"[memory] tidy warning: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return

    if report.outcome == "skipped":
        return
    if report.outcome in {"failed", "locked"}:
        detail = report.error or report.reason or "unknown error"
        print(f"[memory] tidy warning: {detail}", file=sys.stderr)
        return
    print(f"[memory] {report.text()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 聊天入口（M0–M2 行为保持不变）
# ---------------------------------------------------------------------------


def _chat_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="keeper", description="项目级持久化 Agent（M2）")
    parser.add_argument("task", nargs="*", help="要执行的任务；省略则进入交互模式")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="项目根目录；默认已选项目，否则当前目录",
    )
    parser.add_argument("--resume", action="store_true", help="续跑这个项目最近一次会话")
    parser.add_argument("--session", type=Path, help="加载指定会话文件")
    parser.add_argument("--model", help="覆盖模型名")
    parser.add_argument("--max-steps", type=int, help="单个任务的最大步数")
    parser.add_argument(
        "--no-tidy",
        action="store_true",
        help="本次会话结束后不自动整理项目记忆",
    )
    parser.add_argument("--version", action="version", version=f"keeper {__version__}")
    args = parser.parse_args(argv)

    root = _resolve_root(args.root)
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2

    config = load_config(model=args.model, max_steps=args.max_steps)
    if not config.api_key:
        print(_api_key_error(), file=sys.stderr)
        return 2

    no_tidy = args.no_tidy
    _startup_tidy_hint(
        root, _tidy_enabled(config, no_tidy), config.context_window
    )

    # --resume 先读 task 再读 session：恢复后若有 queued 任务，跑队首任务。
    if args.resume:
        store = _load_recovered_store(root)
        if store is None:
            return 2
        task = store.next_queued()
        if task is not None:
            return _execute_task(root, store, task, config, no_tidy=no_tidy)

    try:
        session, resumed = _open_session(args, root, config.model)
    except FileNotFoundError as exc:
        print(f"error: 找不到会话：{exc}", file=sys.stderr)
        return 2

    repaired = session.repair_dangling_tool_calls()
    if repaired:
        print(
            f"note: 修复了 {repaired} 个中断的工具调用（已写入会话）",
            file=sys.stderr,
        )

    llm = LLMClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
    )
    registry = _build_registry(root)
    approver = StdioApprover(timeout=config.approval_timeout)
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        session=session,
        root=root,
        max_steps=config.max_steps,
        approver=approver,
        context_window=config.context_window,
        reserve_tokens=config.reserve_tokens,
        keep_tokens=config.keep_tokens,
    )
    start = SessionStart(
        session_id=session.session_id,
        path=str(session.path),
        resumed=resumed,
        model=config.model,
    )

    try:
        if args.task:
            render(chain([start], loop.run(" ".join(args.task))))
            _maybe_auto_tidy(
                root,
                config,
                llm,
                "chat",
                no_tidy=no_tidy,
                session_id=session.session_id,
            )
            return 0
        render([start])
        rc, clean_exit = _repl(loop, session)
        if clean_exit:
            _maybe_auto_tidy(
                root,
                config,
                llm,
                "repl",
                no_tidy=no_tidy,
                session_id=session.session_id,
            )
        return rc
    except KeyboardInterrupt:
        print("\ninterrupted; session saved.", file=sys.stderr)
        return 130
    finally:
        llm.close()
        session.close()


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    raw = list(sys.argv[1:] if argv is None else argv)

    if raw and raw[0] == "project":
        try:
            return _project_main(raw[1:])
        except KeyboardInterrupt:
            print("\ninterrupted.", file=sys.stderr)
            return 130

    if raw and raw[0] == "manager":
        try:
            return _manager_main(raw[1:])
        except KeyboardInterrupt:
            print("\ninterrupted; manager session saved.", file=sys.stderr)
            return 130

    if raw and raw[0] == "memory":
        try:
            return _memory_main(raw[1:])
        except KeyboardInterrupt:
            print(
                "\ninterrupted; memory files may be partially written; "
                "a backup exists under .agent/memory/backups/.",
                file=sys.stderr,
            )
            return 130

    if raw and raw[0] == "task":
        lock = None
        task_args = raw[1:]
        if task_args and task_args[0] in {"add", "run", "retry"}:
            root_arg = None
            for index, value in enumerate(task_args):
                if value == "--root" and index + 1 < len(task_args):
                    root_arg = Path(task_args[index + 1])
                    break
                if value.startswith("--root="):
                    root_arg = Path(value.split("=", 1)[1])
                    break
            lock = TaskLock(_resolve_root(root_arg))
            try:
                acquired = lock.acquire()
            except OSError as exc:
                print(f"error: 无法创建任务锁 {lock.path}: {exc}", file=sys.stderr)
                return 2
            if not acquired:
                owner = lock._read() or {}
                pid = owner.get("pid", "未知")
                print(
                    f"error: 该项目任务队列正在由 pid {pid} 使用；锁文件: {lock.path}",
                    file=sys.stderr,
                )
                return 2
        try:
            return _task_main(raw[1:])
        except KeyboardInterrupt:
            print(
                "\ninterrupted; 任务未收尾，下次启动会标记 interrupted。",
                file=sys.stderr,
            )
            return 130
        except TaskStoreError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            if lock is not None:
                lock.release()

    try:
        return _chat_main(raw)
    except TaskStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
