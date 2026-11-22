"""CLI 入口：解析参数、渲染事件、交互模式。"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path
from typing import Iterable

from keeper import __version__
from keeper.config import load_config
from keeper.events import (
    AssistantText,
    ErrorEvent,
    Event,
    FinalAnswer,
    SessionStart,
    StepStart,
    Stop,
    ToolCallEvent,
    ToolResultEvent,
)
from keeper.llm import LLMClient
from keeper.loop import AgentLoop
from keeper.session import Session
from keeper.tools.fs import ReadFileTool
from keeper.tools.registry import Registry


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


def main(argv: list[str] | None = None) -> int:
    _setup_console()

    parser = argparse.ArgumentParser(prog="keeper", description="项目级持久化 Agent（M0）")
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
        print(
            "error: 没有找到 API key。\n"
            "请在项目根目录创建 .env（参考 .env.example），"
            "或设置环境变量 KEEPER_API_KEY / DEEPSEEK_API_KEY。",
            file=sys.stderr,
        )
        return 2

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
    registry = Registry([ReadFileTool(root)])
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        session=session,
        root=root,
        max_steps=config.max_steps,
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


if __name__ == "__main__":
    raise SystemExit(main())