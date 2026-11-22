"""会话持久化：JSONL 只追加，支持加载续跑。"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path(".agent") / "sessions"


class Session:
    """一个项目下的一段对话。文件只追加，不重写。"""

    def __init__(self, path: Path, root: Path, session_id: str, messages: list[dict[str, Any]]):
        self.path = path
        self.root = root
        self.session_id = session_id
        self.messages = messages
        self._fh = path.open("a", encoding="utf-8")

    # ---------- 创建 / 加载 ----------

    @classmethod
    def create(cls, root: Path, model: str) -> "Session":
        root = root.resolve()
        sessions_dir = root / SESSIONS_DIR
        sessions_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        session_id = f"{now:%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex[:6]}"
        path = sessions_dir / f"{session_id}.jsonl"
        session = cls(path, root, session_id, [])
        session._write(
            {
                "type": "meta",
                "version": 1,
                "session_id": session_id,
                "created_at": now.isoformat(timespec="seconds"),
                "cwd": str(root),
                "model": model,
            }
        )
        return session

    @classmethod
    def load(cls, path: Path, root: Path) -> "Session":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        meta: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # 进程被杀时可能留下半行，忽略即可
                    continue
                if record.get("type") == "meta":
                    meta = record
                elif record.get("type") == "message" and isinstance(record.get("message"), dict):
                    messages.append(record["message"])
        return cls(path, Path(root).resolve(), meta.get("session_id") or path.stem, messages)

    @staticmethod
    def latest(root: Path) -> Path | None:
        sessions_dir = Path(root).resolve() / SESSIONS_DIR
        files = sorted(sessions_dir.glob("*.jsonl"))
        return files[-1] if files else None

    # ---------- 写入 ----------

    def append(self, message: dict[str, Any]) -> None:
        self._write({"type": "message", "ts": time.time(), "message": message})
        self.messages.append(message)

    def note(self, text: str) -> None:
        self._write({"type": "note", "ts": time.time(), "text": text})

    def _write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ---------- 续跑修复 ----------

    def repair_dangling_tool_calls(self) -> int:
        """中断可能留下「有 tool_call 没有 tool 结果」的尾巴。

        resume 时补一条 tool 消息，否则模型 API 会拒收这段历史。
        返回补了几条。
        """
        last_assistant: int | None = None
        for index, message in enumerate(self.messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                last_assistant = index
        if last_assistant is None:
            return 0

        call_ids = [
            call.get("id", "")
            for call in self.messages[last_assistant].get("tool_calls", [])
        ]
        answered = {
            message.get("tool_call_id")
            for message in self.messages[last_assistant + 1 :]
            if message.get("role") == "tool"
        }
        missing = [call_id for call_id in call_ids if call_id not in answered]
        for call_id in missing:
            self.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "[interrupted] 工具未执行完成，需要的话请重新调用。",
                }
            )
        return len(missing)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()