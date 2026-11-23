"""内核产出的事件。loop 只 yield 事件，渲染交给调用方。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class Event:
    type: ClassVar[str] = "event"
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("ts", None)
        return {"type": self.type, "ts": self.ts, **data}


@dataclass
class SessionStart(Event):
    type: ClassVar[str] = "session_start"
    session_id: str = ""
    path: str = ""
    resumed: bool = False
    model: str = ""


@dataclass
class UserMessage(Event):
    type: ClassVar[str] = "user_message"
    text: str = ""


@dataclass
class StepStart(Event):
    type: ClassVar[str] = "step_start"
    step: int = 0


@dataclass
class AssistantText(Event):
    """中间轮次的 assistant 文本（后面还跟着工具调用）。"""

    type: ClassVar[str] = "assistant_text"
    text: str = ""


@dataclass
class ToolCallEvent(Event):
    type: ClassVar[str] = "tool_call"
    call_id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    valid_json: bool = True


@dataclass
class ToolResultEvent(Event):
    type: ClassVar[str] = "tool_result"
    call_id: str = ""
    name: str = ""
    ok: bool = False
    output: str = ""
    duration_ms: int = 0


@dataclass
class ApprovalRequest(Event):
    """执行写入/命令前，等待用户确认。"""

    type: ClassVar[str] = "approval_request"
    call_id: str = ""
    tool: str = ""
    summary: str = ""
    detail: str = ""


@dataclass
class ApprovalResult(Event):
    type: ClassVar[str] = "approval_result"
    call_id: str = ""
    tool: str = ""
    approved: bool = False
    reason: str = ""


@dataclass
class FinalAnswer(Event):
    type: ClassVar[str] = "final_answer"
    text: str = ""


@dataclass
class Stop(Event):
    type: ClassVar[str] = "stop"
    reason: str = ""


@dataclass
class ErrorEvent(Event):
    type: ClassVar[str] = "error"
    message: str = ""


@dataclass
class CompactionStart(Event):
    """开始压缩：把切点之前的前缀送去摘要。"""

    type: ClassVar[str] = "compaction_start"
    tokens_before: int = 0
    cut_index: int = 0
    reason: str = "threshold"


@dataclass
class CompactionDone(Event):
    type: ClassVar[str] = "compaction_done"
    tokens_before: int = 0
    tokens_after: int = 0
    cut_index: int = 0
    messages_before: int = 0
    messages_after: int = 0


@dataclass
class CompactionFailed(Event):
    """摘要失败：不写 compaction、不中断任务。"""

    type: ClassVar[str] = "compaction_failed"
    tokens_before: int = 0
    tokens_after: int = 0
    cut_index: int = 0
    reason: str = ""
    error: str = ""


@dataclass
class TaskEvent(Event):
    """任务队列状态变化：入队、开始 attempt、完成、阻塞、恢复。"""

    type: ClassVar[str] = "task"
    action: str = ""
    task_id: str = ""
    status: str = ""
    detail: str = ""
