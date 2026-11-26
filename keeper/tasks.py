"""M3 持久任务队列：tasks.json 的模型、原子写、状态机与启动恢复。

设计要点：
- session 是执行日志，task 是持久单位；tasks.json 按项目保存。
- 同一项目同时最多一个 running；状态只有 queued / running / blocked / done。
- 每次执行追加一条 attempt；打开中的 attempt 的 ended_at / outcome 为 None。
- 进程被 kill 时最后一次完整落盘保证是 running + open attempt，启动恢复统一收尾。
- 队列变更（add / start_attempt / retry_attempt）在「锁内重载 → 检查 → 落盘」
  的短临界区里完成；worker 执行期间不持锁。锁是系统级文件锁（见
  ``keeper.file_lock``），进程一死自动释放。
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from keeper.atomic_write import atomic_write_text
from keeper.file_lock import FileLock

TASKS_FILE = Path(".agent") / "tasks.json"
LOCK_FILE = Path(".agent") / "tasks.lock"
VERSION = 1

STATUSES = {"queued", "running", "blocked", "done"}
OUTCOMES = {"done", "blocked", "interrupted", "incomplete", "failed"}
# R4 CompletionRecord 的结果取值；attempt.result 与 outcome 分开记录，
# outcome 驱动 M3 状态机，result 记录 worker 的完成判定。
ATTEMPT_RESULTS = {"done", "blocked", "failed", "interrupted"}
TERMINAL = {"blocked", "done"}


class TaskStoreError(RuntimeError):
    """tasks.json 读写或状态转换失败。调用方必须明确报错，不能静默覆盖。"""


class TaskLockError(TaskStoreError):
    """任务队列正被另一个进程写。调用方立即失败，不等待、不重试、不排队。"""


def now_iso() -> str:
    """本地时间，ISO 8601 带时区偏移（与 session 的 isoformat 风格一致）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _new_task_id(now: str) -> str:
    stamp = now[:19].replace("-", "").replace(":", "").replace("T", "-")
    return f"t-{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class Attempt:
    session_id: str | None
    started_at: str
    ended_at: str | None = None
    outcome: str | None = None
    block_reason: str | None = None
    worker: str | None = None
    result: str | None = None
    summary: str | None = None
    transcript: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "outcome": self.outcome,
            "block_reason": self.block_reason,
            "worker": self.worker,
            "result": self.result,
            "summary": self.summary,
            "transcript": self.transcript,
        }

    @classmethod
    def from_dict(cls, data: Any, *, index: int) -> "Attempt":
        if not isinstance(data, dict):
            raise TaskStoreError(f"tasks.json 损坏：attempt[{index}] 不是对象")
        session_id = data.get("session_id")
        started_at = data.get("started_at")
        ended_at = data.get("ended_at")
        outcome = data.get("outcome")
        block_reason = data.get("block_reason")
        # M6 R7 之前写入的 attempt 没有这四个字段，缺失一律按 None 读取。
        worker = data.get("worker")
        result = data.get("result")
        summary = data.get("summary")
        transcript = data.get("transcript")
        if session_id is not None and not isinstance(session_id, str):
            raise TaskStoreError(f"tasks.json 损坏：attempt[{index}].session_id 类型错误")
        if not isinstance(started_at, str) or not started_at:
            raise TaskStoreError(f"tasks.json 损坏：attempt[{index}].started_at 缺失")
        if ended_at is not None and not isinstance(ended_at, str):
            raise TaskStoreError(f"tasks.json 损坏：attempt[{index}].ended_at 类型错误")
        if outcome is not None and outcome not in OUTCOMES:
            raise TaskStoreError(
                f"tasks.json 损坏：attempt[{index}].outcome={outcome!r} 不合法"
            )
        if block_reason is not None and not isinstance(block_reason, str):
            raise TaskStoreError(f"tasks.json 损坏：attempt[{index}].block_reason 类型错误")
        if worker is not None and not isinstance(worker, str):
            raise TaskStoreError(f"tasks.json 损坏：attempt[{index}].worker 类型错误")
        if result is not None and result not in ATTEMPT_RESULTS:
            raise TaskStoreError(
                f"tasks.json 损坏：attempt[{index}].result={result!r} 不合法"
            )
        for key, value in (("summary", summary), ("transcript", transcript)):
            if value is not None and not isinstance(value, str):
                raise TaskStoreError(f"tasks.json 损坏：attempt[{index}].{key} 类型错误")
        return cls(
            session_id=session_id,
            started_at=started_at,
            ended_at=ended_at,
            outcome=outcome,
            block_reason=block_reason,
            worker=worker,
            result=result,
            summary=summary,
            transcript=transcript,
        )


@dataclass
class Task:
    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    block_reason: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    worker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "block_reason": self.block_reason,
            "worker": self.worker,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: Any, *, index: int) -> "Task":
        if not isinstance(data, dict):
            raise TaskStoreError(f"tasks.json 损坏：tasks[{index}] 不是对象")
        required = ("id", "title", "status", "created_at", "updated_at", "attempts")
        for key in required:
            if key not in data:
                raise TaskStoreError(f"tasks.json 损坏：tasks[{index}] 缺少 {key}")
        task_id = data["id"]
        title = data["title"]
        status = data["status"]
        created_at = data["created_at"]
        updated_at = data["updated_at"]
        block_reason = data.get("block_reason")
        worker = data.get("worker")
        if not isinstance(task_id, str) or not task_id:
            raise TaskStoreError(f"tasks.json 损坏：tasks[{index}].id 不合法")
        if not isinstance(title, str):
            raise TaskStoreError(f"tasks.json 损坏：tasks[{index}].title 类型错误")
        if status not in STATUSES:
            raise TaskStoreError(
                f"tasks.json 损坏：tasks[{index}].status={status!r} 不合法"
            )
        if not isinstance(created_at, str) or not isinstance(updated_at, str):
            raise TaskStoreError(f"tasks.json 损坏：tasks[{index}] 时间字段类型错误")
        if block_reason is not None and not isinstance(block_reason, str):
            raise TaskStoreError(f"tasks.json 损坏：tasks[{index}].block_reason 类型错误")
        if worker is not None and not isinstance(worker, str):
            raise TaskStoreError(f"tasks.json 损坏：tasks[{index}].worker 类型错误")
        raw_attempts = data["attempts"]
        if not isinstance(raw_attempts, list):
            raise TaskStoreError(f"tasks.json 损坏：tasks[{index}].attempts 不是数组")
        attempts = [
            Attempt.from_dict(item, index=attempt_index)
            for attempt_index, item in enumerate(raw_attempts)
        ]
        return cls(
            id=task_id,
            title=title,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            block_reason=block_reason,
            attempts=attempts,
            worker=worker,
        )

    # ---------- 查询 ----------

    @property
    def open_attempts(self) -> list[Attempt]:
        return [attempt for attempt in self.attempts if attempt.ended_at is None]

    @property
    def last_session_id(self) -> str | None:
        for attempt in reversed(self.attempts):
            if attempt.session_id:
                return attempt.session_id
        return None

    @property
    def latest_outcome(self) -> str | None:
        if not self.attempts:
            return None
        last = self.attempts[-1]
        return last.outcome or "open"


def _validate_invariants(tasks: list[Task], path: Path) -> None:
    """加载时拒绝明显自相矛盾的状态；不做修复，要求人工处理。"""
    for task in tasks:
        if task.status in {"queued", "blocked", "done"} and task.open_attempts:
            raise TaskStoreError(
                f"tasks.json 损坏（{path}）：任务 {task.id} 状态为 {task.status} "
                "但仍有未结束 attempt"
            )
        if task.status == "blocked" and not (task.block_reason or "").strip():
            raise TaskStoreError(
                f"tasks.json 损坏（{path}）：任务 {task.id} 为 blocked 但 block_reason 为空"
            )
        if task.status == "done" and task.block_reason:
            raise TaskStoreError(
                f"tasks.json 损坏（{path}）：任务 {task.id} 为 done 但仍带 block_reason"
            )


@dataclass(frozen=True)
class TaskChange:
    task_id: str
    status: str
    detail: str = ""
    action: str = "recover"


class TaskStore:
    """tasks.json 的内存视图 + 原子写。所有修改方法只在成功时落盘。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.path = self.root / TASKS_FILE
        self.lock_path = self.root / LOCK_FILE
        self.tasks: list[Task] = []

    # ---------- 加载 / 写入 ----------

    @classmethod
    def load(cls, root: Path) -> "TaskStore":
        store = cls(root)
        if not store.path.exists():
            return store
        try:
            raw = store.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise TaskStoreError(f"无法读取 {store.path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TaskStoreError(
                f"tasks.json 损坏（{store.path}）：{exc}。请手工修复；不会覆盖原文件。"
            ) from exc
        if not isinstance(data, dict):
            raise TaskStoreError(f"tasks.json 损坏（{store.path}）：顶层不是对象")
        if data.get("version") != VERSION:
            raise TaskStoreError(
                f"tasks.json 版本不支持（{store.path}）：{data.get('version')!r}，"
                f"期望 {VERSION}"
            )
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            raise TaskStoreError(f"tasks.json 损坏（{store.path}）：tasks 不是数组")
        tasks = [
            Task.from_dict(item, index=index)
            for index, item in enumerate(raw_tasks)
        ]
        seen: set[str] = set()
        for task in tasks:
            if task.id in seen:
                raise TaskStoreError(
                    f"tasks.json 损坏（{store.path}）：任务 id 重复 {task.id}"
                )
            seen.add(task.id)
        _validate_invariants(tasks, store.path)
        store.tasks = tasks
        return store

    def save(self) -> None:
        data = {"version": VERSION, "tasks": [task.to_dict() for task in self.tasks]}
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        try:
            atomic_write_text(self.path, payload)
        except Exception as exc:  # 原子写失败：原文件保持不动
            raise TaskStoreError(
                f"无法写入 {self.path}: {type(exc).__name__}: {exc}"
            ) from exc

    # ---------- 跨进程临界区 ----------

    @contextmanager
    def locked(self) -> Iterator["TaskStore"]:
        """持锁重载 → 交给 with 体修改并落盘 → 释放。

        必须重载：调用方手里的 ``self.tasks`` 可能是锁外的旧快照，
        直接拿它检查 / 写回会覆盖别的进程刚写进去的变更。
        """
        lock = FileLock(self.lock_path)
        if not lock.acquire():
            raise TaskLockError(
                f"该项目任务队列正被另一个进程写入；锁文件: {self.lock_path}；"
                "请等它结束或确认没有残留进程"
            )
        try:
            self.tasks = type(self).load(self.root).tasks
            yield self
        finally:
            lock.release()

    # ---------- 查询 ----------

    def get(self, task_id: str) -> Task | None:
        return next((task for task in self.tasks if task.id == task_id), None)

    def next_queued(self) -> Task | None:
        return next((task for task in self.tasks if task.status == "queued"), None)

    def running(self) -> Task | None:
        return next((task for task in self.tasks if task.status == "running"), None)

    def _require(self, task_id: str) -> Task:
        task = self.get(task_id)
        if task is None:
            raise TaskStoreError(f"任务不存在: {task_id}")
        return task

    def _move_to_head(self, task: Task) -> None:
        index = next(
            (i for i, item in enumerate(self.tasks) if item is task),
            None,
        )
        if index is None:
            raise TaskStoreError(f"任务不在队列中: {task.id}")
        if index != 0:
            self.tasks.insert(0, self.tasks.pop(index))

    @staticmethod
    def _open_attempt(task: Task) -> Attempt:
        open_attempts = task.open_attempts
        if not open_attempts:
            raise TaskStoreError(f"任务 {task.id} 没有打开中的 attempt")
        if len(open_attempts) > 1:
            raise TaskStoreError(
                f"任务 {task.id} 有 {len(open_attempts)} 个打开中的 attempt，状态损坏"
            )
        return open_attempts[0]

    # ---------- 状态转换 ----------

    def add(self, title: str, *, worker: str | None = None) -> Task:
        cleaned = title.strip()
        if not cleaned:
            raise TaskStoreError("任务标题不能为空")
        if worker is not None and (not isinstance(worker, str) or not worker.strip()):
            raise TaskStoreError("worker 指派不能为空字符串")
        with self.locked():
            now = now_iso()
            task = Task(
                id=_new_task_id(now),
                title=cleaned,
                status="queued",
                created_at=now,
                updated_at=now,
                worker=worker.strip() if worker else None,
            )
            self.tasks.append(task)
            self.save()
        return task

    def start_attempt(
        self, task_id: str, session_id: str | None, *, worker: str | None = None
    ) -> Attempt:
        """queued → running，追加一条打开中的 attempt，先落盘再跑 loop。"""
        with self.locked():
            task = self._require(task_id)
            if task.status != "queued":
                raise TaskStoreError(
                    f"任务 {task.id} 当前状态为 {task.status}，只有 queued 可以开始"
                )
            other = self.running()
            if other is not None:
                raise TaskStoreError(
                    f"已有 running 任务 {other.id}；同一项目同时只能有一个"
                )
            now = now_iso()
            attempt = Attempt(session_id=session_id, started_at=now, worker=worker)
            task.attempts.append(attempt)
            task.status = "running"
            task.block_reason = None
            if worker is not None:
                task.worker = worker
            task.updated_at = now
            self.save()
        return attempt

    def retry_attempt(
        self, task_id: str, session_id: str | None = None, *, worker: str | None = None
    ) -> Attempt:
        """blocked → running：阻塞或失败的项在新会话里重跑，可换 worker。"""
        with self.locked():
            task = self._require(task_id)
            if task.status != "blocked":
                raise TaskStoreError(
                    f"任务 {task.id} 当前状态为 {task.status}，"
                    "只有 blocked（阻塞或失败）可以重试"
                )
            other = self.running()
            if other is not None:
                raise TaskStoreError(
                    f"已有 running 任务 {other.id}；同一项目同时只能有一个"
                )
            now = now_iso()
            attempt = Attempt(session_id=session_id, started_at=now, worker=worker)
            task.attempts.append(attempt)
            task.status = "running"
            task.block_reason = None
            if worker is not None:
                task.worker = worker
            task.updated_at = now
            self.save()
        return attempt

    def complete(
        self, task_id: str, *, summary: str | None = None, transcript: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """running → done，立即收尾，防止完成之后崩溃丢失 done。"""
        with self.locked():
            task = self._require(task_id)
            if task.status != "running":
                raise TaskStoreError(
                    f"任务 {task.id} 当前状态为 {task.status}，不能标记完成"
                )
            attempt = self._open_attempt(task)
            now = now_iso()
            attempt.ended_at = now
            attempt.outcome = "done"
            attempt.block_reason = None
            attempt.result = "done"
            attempt.summary = summary
            attempt.transcript = transcript
            if session_id is not None:
                attempt.session_id = session_id
            task.status = "done"
            task.block_reason = None
            task.updated_at = now
            self.save()

    def block(
        self,
        task_id: str,
        reason: str,
        *,
        summary: str | None = None,
        transcript: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """running → blocked；reason 去空白后为空则拒绝，且不落盘。"""
        cleaned = reason.strip() if isinstance(reason, str) else ""
        if not cleaned:
            raise TaskStoreError("block 原因不能为空")
        with self.locked():
            task = self._require(task_id)
            if task.status != "running":
                raise TaskStoreError(
                    f"任务 {task.id} 当前状态为 {task.status}，不能标记阻塞"
                )
            attempt = self._open_attempt(task)
            now = now_iso()
            attempt.ended_at = now
            attempt.outcome = "blocked"
            attempt.block_reason = cleaned
            attempt.result = "blocked"
            attempt.summary = summary if summary is not None else cleaned
            attempt.transcript = transcript
            if session_id is not None:
                attempt.session_id = session_id
            task.status = "blocked"
            task.block_reason = cleaned
            task.updated_at = now
            self.save()

    def fail(
        self,
        task_id: str,
        reason: str,
        *,
        result: str = "failed",
        transcript: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """running → blocked，记录 failed / interrupted；任务可 retry。"""
        cleaned = reason.strip() if isinstance(reason, str) else ""
        if not cleaned:
            raise TaskStoreError("fail 原因不能为空")
        if result not in {"failed", "interrupted"}:
            raise TaskStoreError(f"fail result={result!r} 不合法")
        with self.locked():
            task = self._require(task_id)
            if task.status != "running":
                raise TaskStoreError(
                    f"任务 {task.id} 当前状态为 {task.status}，不能标记失败"
                )
            attempt = self._open_attempt(task)
            now = now_iso()
            attempt.ended_at = now
            attempt.outcome = result
            attempt.block_reason = cleaned
            attempt.result = result
            attempt.summary = cleaned
            attempt.transcript = transcript
            if session_id is not None:
                attempt.session_id = session_id
            task.status = "blocked"
            task.block_reason = cleaned
            task.updated_at = now
            self.save()

    def block_before_start(self, task_id: str, reason: str) -> None:
        """queued → blocked：启动前失败（例如 session 文件丢失），不产生 attempt。"""
        cleaned = reason.strip() if isinstance(reason, str) else ""
        if not cleaned:
            raise TaskStoreError("block 原因不能为空")
        with self.locked():
            task = self._require(task_id)
            if task.status != "queued":
                raise TaskStoreError(
                    f"任务 {task.id} 当前状态为 {task.status}，不能标记启动前失败"
                )
            task.status = "blocked"
            task.block_reason = cleaned
            task.updated_at = now_iso()
            self.save()

    def mark_incomplete(self, task_id: str) -> None:
        """running → queued：正常结束但没有显式 complete；任务留在原队列位置。"""
        with self.locked():
            task = self._require(task_id)
            if task.status != "running":
                raise TaskStoreError(
                    f"任务 {task.id} 当前状态为 {task.status}，不能标记 incomplete"
                )
            attempt = self._open_attempt(task)
            now = now_iso()
            attempt.ended_at = now
            attempt.outcome = "incomplete"
            attempt.block_reason = None
            task.status = "queued"
            task.block_reason = None
            task.updated_at = now
            self.save()

    # ---------- 启动恢复 ----------

    def recover(self) -> list[TaskChange]:
        """把上次进程留下的 running 收尾：第一次回队首，连续两次中断 → blocked。

        防御性处理所有 running（正常路径最多一个）；按数组顺序处理，最终不留 running。
        """
        changes: list[TaskChange] = []
        running_tasks = [task for task in self.tasks if task.status == "running"]
        for task in running_tasks:
            now = now_iso()
            open_attempts = task.open_attempts
            if not open_attempts:
                task.status = "blocked"
                task.block_reason = "状态损坏：running 但没有未完成 attempt"
                task.updated_at = now
                changes.append(
                    TaskChange(
                        task_id=task.id,
                        status=task.status,
                        detail=task.block_reason,
                    )
                )
                continue

            for attempt in open_attempts:
                attempt.ended_at = now
                attempt.outcome = "interrupted"
                attempt.block_reason = None

            streak = 0
            for attempt in reversed(task.attempts):
                if attempt.outcome == "interrupted":
                    streak += 1
                else:
                    break

            last_session = task.last_session_id or "-"
            if streak >= 2:
                task.status = "blocked"
                task.block_reason = (
                    f"连续 {streak} 次中断；最近 session={last_session}，需人工检查"
                )
                detail = f"{len(open_attempts)} 个 attempt 标记 interrupted；连续 {streak} 次中断 → blocked"
            else:
                task.status = "queued"
                task.block_reason = None
                self._move_to_head(task)
                detail = "attempt 标记 interrupted → queued（移到队首）"
            task.updated_at = now
            changes.append(
                TaskChange(task_id=task.id, status=task.status, detail=detail)
            )

        if changes:
            self.save()
        return changes
