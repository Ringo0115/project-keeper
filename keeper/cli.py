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
from keeper.session import Session
from keeper.task_runner import TaskRunError, prepare_task_run, run_task_once
from keeper.tasks import Task, TaskStore, TaskStoreError
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
            print(
                f"[compaction] done · {event.tokens_before} -> "
                f"{event.tokens_after} tokens",
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


def _build_registry(root: Path) -> Registry:
    return Registry(
        [
            ReadFileTool(root),
            ListDirTool(root),
            WriteFileTool(root),
            SearchTool(root),
            UpdateMemoryTool(root),
            ShellTool(root),
        ]
    )


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


def _repl(loop: AgentLoop, session: Session) -> int:
    print("输入任务，/exit 退出，/session 查看会话文件。\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
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
        print(line)


def _execute_task(
    root: Path, store: TaskStore, task: Task, config: Config
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
        return 0
    finally:
        if llm is not None:
            llm.close()
        prepared.session.close()


def _task_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="keeper task", description="持久任务队列（M3）")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root", type=Path, default=Path.cwd(), help="项目根目录，默认当前目录"
    )
    common.add_argument("--model", help="覆盖模型名")
    common.add_argument("--max-steps", type=int, help="单个任务的最大步数")

    sub = parser.add_subparsers(dest="command", required=True)
    add_parser = sub.add_parser("add", parents=[common], help="入队一个任务")
    add_parser.add_argument("title", nargs="+", help="任务标题")
    sub.add_parser("list", parents=[common], help="列出任务")
    run_parser = sub.add_parser("run", parents=[common], help="运行队首或指定任务")
    run_parser.add_argument("id", nargs="?", help="任务 id；省略则跑队首 queued 任务")

    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2

    store = _load_recovered_store(root)
    if store is None:
        return 2

    if args.command == "add":
        try:
            task = store.add(" ".join(args.title))
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

    # run
    config = load_config(model=args.model, max_steps=args.max_steps)
    if not config.api_key:
        print(_api_key_error(), file=sys.stderr)
        return 2

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
    return _execute_task(root, store, task, config)


# ---------------------------------------------------------------------------
# 聊天入口（M0–M2 行为保持不变）
# ---------------------------------------------------------------------------


def _chat_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="keeper", description="项目级持久化 Agent（M2）")
    parser.add_argument("task", nargs="*", help="要执行的任务；省略则进入交互模式")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="项目根目录，默认当前目录")
    parser.add_argument("--resume", action="store_true", help="续跑这个项目最近一次会话")
    parser.add_argument("--session", type=Path, help="加载指定会话文件")
    parser.add_argument("--model", help="覆盖模型名")
    parser.add_argument("--max-steps", type=int, help="单个任务的最大步数")
    parser.add_argument("--version", action="version", version=f"keeper {__version__}")
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2

    config = load_config(model=args.model, max_steps=args.max_steps)
    if not config.api_key:
        print(_api_key_error(), file=sys.stderr)
        return 2

    # --resume 先读 task 再读 session：恢复后若有 queued 任务，跑队首任务。
    if args.resume:
        store = _load_recovered_store(root)
        if store is None:
            return 2
        task = store.next_queued()
        if task is not None:
            return _execute_task(root, store, task, config)

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
            return 0
        render([start])
        return _repl(loop, session)
    except KeyboardInterrupt:
        print("\ninterrupted; session saved.", file=sys.stderr)
        return 130
    finally:
        llm.close()
        session.close()


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    raw = list(sys.argv[1:] if argv is None else argv)

    if raw and raw[0] == "task":
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

    try:
        return _chat_main(raw)
    except TaskStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
