"""M3 编排：选 session → 修 dangling tool_call → 落盘 running → 跑一次 attempt。

这里不写死 CLI 细节：prepare_task_run / run_task_once 都可以注入 FakeLLM、
Registry、Approver，便于测试。session 的生命周期由调用方关闭。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from keeper.approval import Approver
from keeper.events import Event, FinalAnswer, TaskEvent, ToolResultEvent
from keeper.loop import AgentLoop, ChatLLM
from keeper.prompts import TaskContext
from keeper.session import SESSIONS_DIR, Session
from keeper.tasks import TaskStore
from keeper.tools.registry import Registry
from keeper.tools.tasks import BlockTaskTool, CompleteTaskTool


class TaskRunError(RuntimeError):
    """任务无法运行或启动前失败；CLI 直接展示给用户。"""


@dataclass
class PreparedTaskRun:
    store: TaskStore
    task_id: str
    root: Path
    session: Session
    attempt_number: int
    message: str
    resumed: bool
    repaired: int = 0


def prepare_task_run(
    store: TaskStore,
    task_id: str,
    root: Path,
    model: str,
) -> PreparedTaskRun:
    """按顺序：解析 session → 修 dangling tool_call → start_attempt → 返回。

    - 无历史 session_id：新建 session；排队期间被 kill 等于还没开始，仍是 queued。
    - 有历史但文件丢失：queued → blocked（启动前失败），不产生假 attempt。
    - 有历史且存在：精确加载该 session，不用“最近 session”。
    """
    root = Path(root).resolve()
    task = store.get(task_id)
    if task is None:
        raise TaskRunError(f"任务不存在: {task_id}")
    if task.status != "queued":
        raise TaskRunError(
            f"任务 {task.id} 当前状态为 {task.status}，只有 queued 可以运行"
        )
    running = store.running()
    if running is not None:
        raise TaskRunError(
            f"已有 running 任务 {running.id}；同一项目同时只能有一个"
        )

    session_id = task.last_session_id
    resumed = session_id is not None
    if session_id is None:
        session = Session.create(root, model)
    else:
        path = root / SESSIONS_DIR / f"{session_id}.jsonl"
        if not path.exists():
            reason = f"session 文件丢失: {path}"
            store.block_before_start(task.id, reason)
            raise TaskRunError(reason)
        session = Session.load(path, root)

    try:
        repaired = session.repair_dangling_tool_calls()
        store.start_attempt(task.id, session.session_id)
    except Exception:
        session.close()
        raise

    message = f"[task {task.id}] {task.title}"
    if resumed:
        message += (
            "\n上一个 attempt 没有显式完成，继续该任务；"
            "完成时必须调用 complete_task。"
        )

    return PreparedTaskRun(
        store=store,
        task_id=task.id,
        root=root,
        session=session,
        attempt_number=len(task.attempts),
        message=message,
        resumed=resumed,
        repaired=repaired,
    )


def run_task_once(
    prepared: PreparedTaskRun,
    *,
    llm: ChatLLM,
    registry: Registry,
    approver: Approver | None,
    max_steps: int = 30,
) -> Iterator[Event]:
    """跑一次 attempt，并在 loop 正常结束后按结束方式收尾。

    - 显式 complete / block：工具执行时已落盘，这里只补 TaskEvent。
    - 纯文本 final answer：attempt=incomplete，任务回 queued（不挪位置）。
    - LLM 错误 / max_steps / 异常 / Ctrl+C：不收尾，保持 running 等下次恢复。
    """
    store = prepared.store
    task = store.get(prepared.task_id)
    if task is None:
        raise TaskRunError(f"任务不存在: {prepared.task_id}")

    tools = [registry.get(name) for name in registry.names()]
    tools.extend(
        [
            CompleteTaskTool(store, task.id),
            BlockTaskTool(store, task.id),
        ]
    )
    run_registry = Registry(tools)  # type: ignore[arg-type]
    loop = AgentLoop(
        llm=llm,
        registry=run_registry,
        session=prepared.session,
        root=prepared.root,
        max_steps=max_steps,
        approver=approver,
        task=TaskContext(
            id=task.id,
            title=task.title,
            attempt_number=prepared.attempt_number,
            session_id=prepared.session.session_id,
        ),
    )

    yield TaskEvent(
        action="start",
        task_id=task.id,
        status="running",
        detail=(
            f"attempt #{prepared.attempt_number} · "
            f"session={prepared.session.session_id}"
        ),
    )

    final_answer = False
    for event in loop.run(prepared.message):
        yield event
        if isinstance(event, FinalAnswer):
            final_answer = True
        elif isinstance(event, ToolResultEvent) and event.ok:
            if event.name == "complete_task":
                yield TaskEvent(
                    action="complete",
                    task_id=task.id,
                    status="done",
                    detail=f"attempt #{prepared.attempt_number} 显式完成",
                )
            elif event.name == "block_task":
                current = store.get(task.id)
                reason = current.block_reason if current else ""
                yield TaskEvent(
                    action="block",
                    task_id=task.id,
                    status="blocked",
                    detail=reason or "",
                )

    if final_answer:
        current = store.get(task.id)
        if current is not None and current.status == "running":
            store.mark_incomplete(task.id)
            yield TaskEvent(
                action="incomplete",
                task_id=task.id,
                status="queued",
                detail=(
                    f"attempt #{prepared.attempt_number} 未显式完成；"
                    "任务回到 queued"
                ),
            )
    else:
        current = store.get(task.id)
        if current is not None and current.status == "running":
            yield TaskEvent(
                action="left-open",
                task_id=task.id,
                status="running",
                detail=(
                    f"attempt #{prepared.attempt_number} 保持打开；"
                    "下次启动会标记 interrupted"
                ),
            )
