"""M6 R4 的最小完成记录。

R3/R5 只需要这个数据形状；派发、审批与进程适配留到 R4 实现。
"""

from __future__ import annotations

from dataclasses import dataclass

COMPLETION_RESULTS = frozenset({"done", "blocked", "failed", "interrupted"})


@dataclass(frozen=True)
class CompletionRecord:
    worker: str
    result: str
    summary: str
    reply: str
    transcript: str | None
    started_at: str
    ended_at: str

    def __post_init__(self) -> None:
        if not self.worker:
            raise ValueError("completion record 缺少 worker")
        if self.result not in COMPLETION_RESULTS:
            raise ValueError(
                f"completion record 的 result={self.result!r} 不合法；"
                f"允许: {', '.join(sorted(COMPLETION_RESULTS))}"
            )