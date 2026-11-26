"""fanout 批次记录：一次并行派发的落盘状态 + 只收尾一次。

对齐 Orbital 的 ``agent_os/daemon_v2/fanout.py``：一组并行任务由
``FanoutGroup`` 持有，每个任务的状态只有 ``running`` 与非终态之分，
终止事件经 ``absorb_terminal`` 单一入口吸收，最后一个任务落地时整组
``resolve`` 一次。keeper 没有常驻 daemon，所以组状态落盘到
``.agent/fanout/<id>.json``，进程被打断后仍能查到自己派过什么、跑到哪。
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from keeper.atomic_write import atomic_write_text

FANOUT_DIR = Path(".agent") / "fanout"
VERSION = 1
TERMINAL = frozenset({"done", "blocked", "failed", "interrupted"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FanoutTaskState:
    label: str
    worker: str
    status: str = "running"
    summary: str = ""
    transcript: str = ""
    started_at: str = field(default_factory=_now)
    ended_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "worker": self.worker,
            "status": self.status,
            "summary": self.summary,
            "transcript": self.transcript,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FanoutTaskState":
        return cls(
            label=data["label"],
            worker=data["worker"],
            status=data.get("status", "running"),
            summary=data.get("summary", ""),
            transcript=data.get("transcript", ""),
            started_at=data.get("started_at") or _now(),
            ended_at=data.get("ended_at"),
        )


class FanoutBatch:
    """一次 fanout 的落盘记录。创建即落盘，每次状态变化再落盘一次。"""

    def __init__(self, root: Path, batch_id: str, tasks: list[FanoutTaskState],
                 *, status: str = "running", resolve_reason: str | None = None,
                 created_at: str | None = None, resolved_at: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.id = batch_id
        self.tasks = tasks
        self.status = status
        self.resolve_reason = resolve_reason
        self.created_at = created_at or _now()
        self.resolved_at = resolved_at
        # ThreadPoolExecutor 的多个线程会并发 absorb；读-改-写必须串行。
        # RLock：absorb 内部可重入调用 resolve / _write。
        self._lock = threading.RLock()

    # ---------- 构造与读取 ----------

    @classmethod
    def path_for(cls, root: Path, batch_id: str) -> Path:
        return Path(root).resolve() / FANOUT_DIR / f"{batch_id}.json"

    @classmethod
    def create(cls, root: Path, assignments: list[tuple[str, str]]) -> "FanoutBatch":
        """``assignments`` 为 (worker slug, label) 列表；创建即落盘。"""
        batch = cls(Path(root), uuid.uuid4().hex[:12],
                    [FanoutTaskState(label=label, worker=worker) for worker, label in assignments])
        batch._write()
        return batch

    @classmethod
    def load(cls, root: Path, batch_id: str) -> "FanoutBatch":
        data = json.loads(cls.path_for(root, batch_id).read_text(encoding="utf-8"))
        return cls(
            Path(root), data["id"], [FanoutTaskState.from_dict(item) for item in data.get("tasks", [])],
            status=data.get("status", "running"), resolve_reason=data.get("resolve_reason"),
            created_at=data.get("created_at"), resolved_at=data.get("resolved_at"),
        )

    @property
    def path(self) -> Path:
        return self.path_for(self.root, self.id)

    # ---------- 状态 ----------

    def get(self, label: str) -> FanoutTaskState | None:
        return next((task for task in self.tasks if task.label == label), None)

    @property
    def all_terminal(self) -> bool:
        return all(task.status in TERMINAL for task in self.tasks)

    def absorb(self, label: str, *, status: str, summary: str = "",
               transcript: str | None = None) -> bool:
        """吸收一个任务的终止事件；已经终态的任务重复上报不会翻状态。

        与 Orbital 的 ``absorb_terminal`` 一致：幂等，且最后一个任务落地
        时整组收尾一次（``resolved`` 之后再来的事件一律忽略）。
        """
        with self._lock:
            # 锁内复查：等锁期间可能已有别的线程收尾。
            if self.status != "running":
                return False
            task = self.get(label)
            if task is None or task.status in TERMINAL:
                return False
            task.status = status
            task.summary = summary
            task.transcript = transcript or task.transcript
            task.ended_at = _now()
            if self.all_terminal:
                self.resolve("all_terminal")
            else:
                self._write()
            return True

    def resolve(self, reason: str) -> bool:
        """收尾整组：还没落地的一律记为 interrupted。返回是否本次收尾。"""
        with self._lock:
            if self.status != "running":
                return False
            for task in self.tasks:
                if task.status not in TERMINAL:
                    task.status = "interrupted"
                    task.summary = task.summary or f"批次收尾（{reason}），该子任务未回报"
                    task.ended_at = _now()
            self.status = "resolved"
            self.resolve_reason = reason
            self.resolved_at = _now()
            self._write()
            return True

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "id": self.id,
            "status": self.status,
            "resolve_reason": self.resolve_reason,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def _write(self) -> None:
        with self._lock:
            payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
            atomic_write_text(self.path, payload)
