"""会话持久化：JSONL 只追加，支持加载续跑。"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path(".agent") / "sessions"

# 压缩后放进有效消息列表的第一条；和 compaction 记录回放共用。
SUMMARY_PREFIX = "[历史摘要]\n"

# M5.1a 会话证据的边界。session id 只允许这些字符，避免把任意路径交给
# 读取函数；真实 id 形如 20261122-191353-094559-6f0ff6。
SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
SESSION_ID_MAX_LEN = 128
SESSION_EVIDENCE_LIMIT = 3
SESSION_EVIDENCE_CHARS = 1000
SESSION_EVIDENCE_TOTAL_CHARS = 3000
SESSION_READ_MAX_BYTES = 1_000_000
# meta.created_at 是外部 JSONL 里的不可信字段：只接受 ISO 风格字符，
# 并限制长度；换行 / 伪造段落一律丢弃。
TIMESTAMP_RE = re.compile(r"[0-9T:+\-.Z ]+")
TIMESTAMP_MAX_LEN = 64


def _summary_message(summary: str) -> dict[str, Any]:
    return {"role": "user", "content": SUMMARY_PREFIX + summary}


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
                elif record.get("type") == "compaction":
                    messages = cls._apply_compaction(record, messages)
        return cls(path, Path(root).resolve(), meta.get("session_id") or path.stem, messages)

    @staticmethod
    def _apply_compaction(
        record: dict[str, Any], messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """按 JSONL 顺序回放一条 compaction 记录。

        cut_index 是「记录写入时有效消息列表」的下标，因此连续压缩时
        第二条记录仍然指向回放后的列表，不需要额外换算。
        """
        summary = record.get("summary")
        cut_index = record.get("cut_index")
        if (
            not isinstance(summary, str)
            or type(cut_index) is not int
            or not 0 <= cut_index <= len(messages)
        ):
            # 损坏的记录不能删（只追加），但也不该破坏加载；跳过它。
            return messages
        return [_summary_message(summary), *messages[cut_index:]]

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

    def compact(
        self,
        summary: str,
        cut_index: int,
        *,
        tokens_before: int = 0,
        tokens_after: int | None = None,
        degraded: bool = False,
    ) -> None:
        """追加一条 compaction 记录，并把内存消息换成「摘要 + 切点之后」。

        旧 message 记录原样留在 JSONL 里；加载时按顺序回放 compaction。
        ``degraded`` 记录这次摘要是否由确定性兜底生成。
        """
        if type(cut_index) is not int or not 0 <= cut_index <= len(self.messages):
            raise ValueError(f"invalid cut_index: {cut_index!r}")

        messages_before = len(self.messages)
        record = {
            "type": "compaction",
            "ts": time.time(),
            "summary": summary,
            "cut_index": cut_index,
            "messages_before": messages_before,
            "tokens_before": tokens_before,
        }
        if tokens_after is not None:
            record["tokens_after"] = tokens_after
        if degraded:
            record["degraded"] = True
        self._write(record)
        self.messages = [_summary_message(summary), *self.messages[cut_index:]]

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


# ---------------------------------------------------------------------------
# M5.1a 只读会话列表 / 读取 / 证据抽取
# ---------------------------------------------------------------------------
#
# 这里刻意不复用 ``Session.load``：后者会打开 append 句柄，并把 compaction
# 记录回放成有效消息列表；证据只需要原始记录里的事实，不能产生写句柄。


@dataclass(frozen=True)
class SessionInfo:
    """``.agent/sessions/`` 下的一个平铺 JSONL 文件。"""

    session_id: str
    path: Path
    mtime: float
    size: int


@dataclass(frozen=True)
class SessionRead:
    """一份会话的只读抽取结果；文本都已截断，适合进入证据。"""

    session_id: str
    path: Path
    created_at: str | None
    last_active: str | None
    user_texts: tuple[str, ...]
    assistant_texts: tuple[str, ...]
    summaries: tuple[str, ...]
    truncated: bool = False


@dataclass(frozen=True)
class SessionEvidence:
    """给整理员的一条会话证据；字段是摘录，不是完整会话。"""

    session_id: str
    created_at: str | None
    last_active: str | None
    summary: str = ""
    last_user: str = ""
    conclusions: tuple[str, ...] = ()

    @property
    def has_text(self) -> bool:
        return bool(self.summary or self.last_user or self.conclusions)


def _sessions_dir(root: Path) -> Path:
    return Path(root).resolve() / SESSIONS_DIR


def _normalise_session_id(raw: Any) -> str | None:
    """把用户/调用方给的 id 收敛成白名单内的 id；非法返回 None。"""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.lower().endswith(".jsonl"):
        text = text[: -len(".jsonl")]
    if not text or text in {".", ".."}:
        return None
    if len(text) > SESSION_ID_MAX_LEN:
        return None
    if not SESSION_ID_RE.fullmatch(text):
        return None
    return text


def _clean_created_at(value: Any) -> str | None:
    """只保留短、单行、ISO 风格的 created_at；否则丢弃。"""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > TIMESTAMP_MAX_LEN:
        return None
    if not TIMESTAMP_RE.fullmatch(text):
        return None
    return text


def _safe_session_path(root: Path, session_id: str) -> Path:
    """由已验证 id 构造路径，并确认它没有越出 ``.agent/sessions/``。"""
    sessions_dir = _sessions_dir(root)
    path = sessions_dir / f"{session_id}.jsonl"
    resolved = path.resolve()
    if not resolved.is_relative_to(sessions_dir):
        raise ValueError("session path must stay inside .agent/sessions/")
    if resolved.parent != sessions_dir:
        raise ValueError("session files are flat; no subdirectories")
    return resolved


def list_sessions(root: Path) -> list[SessionInfo]:
    """只列 ``.agent/sessions/`` 下平铺的 ``<id>.jsonl``，按最近活动倒序。"""
    sessions_dir = _sessions_dir(root)
    try:
        candidates = list(sessions_dir.iterdir())
    except OSError:
        return []

    found: list[SessionInfo] = []
    for path in candidates:
        session_id = _normalise_session_id(path.name)
        if session_id is None or path.name != f"{session_id}.jsonl":
            continue
        try:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved.parent != sessions_dir:
                continue
            stat = resolved.stat()
        except (OSError, ValueError):
            continue
        found.append(
            SessionInfo(
                session_id=session_id,
                path=resolved,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        )
    # mtime 倒序；同一时刻按 id 倒序，保证结果确定。
    return sorted(found, key=lambda info: (info.mtime, info.session_id), reverse=True)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _read_session_bytes(path: Path) -> tuple[bytes, bool]:
    """读取整个文件；超过上限时读尾部并丢掉第一条半行。"""
    stat = path.stat()
    with path.open("rb") as fh:
        if stat.st_size > SESSION_READ_MAX_BYTES:
            fh.seek(stat.st_size - SESSION_READ_MAX_BYTES)
            data = fh.read()
            newline = data.find(b"\n")
            data = data[newline + 1 :] if newline != -1 else b""
            return data, True
        return fh.read(), False


def read_session(root: Path, name: Any) -> SessionRead:
    """只读解析一份会话；坏行、半行、非法 UTF-8 都跳过。

    只接受 session id 或 ``<id>.jsonl``，不接受任意路径。文件不存在时
    抛 ``FileNotFoundError``，名字非法时抛 ``ValueError``。
    """
    session_id = _normalise_session_id(name)
    if session_id is None:
        raise ValueError("invalid session id")
    path = _safe_session_path(root, session_id)
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        data, truncated = _read_session_bytes(path)
        mtime = path.stat().st_mtime
    except OSError:
        raise

    meta_id: str | None = None
    created_at: str | None = None
    user_texts: list[str] = []
    assistant_texts: list[str] = []
    summaries: list[str] = []

    for raw in data.split(b"\n"):
        raw = raw.rstrip(b"\r")
        if not raw.strip():
            continue
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue

        kind = record.get("type")
        if kind == "meta":
            raw_id = record.get("session_id")
            normalised = (
                _normalise_session_id(raw_id)
                if isinstance(raw_id, str)
                else None
            )
            if normalised:
                meta_id = normalised
            cleaned_created = _clean_created_at(record.get("created_at"))
            if cleaned_created:
                created_at = cleaned_created
        elif kind == "compaction":
            summary = record.get("summary")
            if isinstance(summary, str) and summary.strip():
                summaries.append(_clip(summary, 4000))
                summaries = summaries[-3:]
        elif kind == "message":
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            role = message.get("role")
            if role == "user":
                user_texts.append(_clip(content, 4000))
                user_texts = user_texts[-3:]
            elif role == "assistant":
                assistant_texts.append(_clip(content, 4000))
                assistant_texts = assistant_texts[-5:]

    try:
        last_active = datetime.fromtimestamp(mtime).astimezone().isoformat(
            timespec="seconds"
        )
    except (OSError, OverflowError, ValueError):
        # 极端 mtime（例如伪造的 1970 年前时间）在部分平台会失败；
        # 证据只需要一个可读时间，退回本地 naive 值，再不行就留空。
        try:
            last_active = datetime.fromtimestamp(mtime).isoformat(
                timespec="seconds"
            )
        except (OSError, OverflowError, ValueError):
            last_active = None
    return SessionRead(
        session_id=meta_id or session_id,
        path=path,
        created_at=created_at,
        last_active=last_active,
        user_texts=tuple(user_texts),
        assistant_texts=tuple(assistant_texts),
        summaries=tuple(summaries),
        truncated=truncated,
    )


def _evidence_from_read(read: SessionRead, *, max_chars: int) -> SessionEvidence:
    # 四个字段各占四分之一，保证一条证据不超过 max_chars。
    share = max(1, max_chars // 4)
    summary = _clip(read.summaries[-1], share) if read.summaries else ""
    last_user = _clip(read.user_texts[-1], share) if read.user_texts else ""
    conclusions = tuple(
        _clip(text, share) for text in read.assistant_texts[-2:]
    )
    return SessionEvidence(
        session_id=read.session_id,
        created_at=read.created_at,
        last_active=read.last_active,
        summary=summary,
        last_user=last_user,
        conclusions=conclusions,
    )


def _evidence_chars(evidence: SessionEvidence) -> int:
    return (
        len(evidence.summary)
        + len(evidence.last_user)
        + sum(len(text) for text in evidence.conclusions)
    )


def collect_session_evidence(
    root: Path,
    *,
    prefer_session_id: str | None = None,
    limit: int = SESSION_EVIDENCE_LIMIT,
    max_chars: int = SESSION_EVIDENCE_CHARS,
    total_chars: int = SESSION_EVIDENCE_TOTAL_CHARS,
) -> list[SessionEvidence]:
    """取最近会话的证据；只读，任何文件级失败都退化为“没有证据”。

    ``prefer_session_id`` 是刚结束的会话（由 CLI 传入），始终优先；其余按
    mtime 倒序补足 ``limit`` 条。没有实际文本的会话不占名额。
    """
    try:
        infos = list_sessions(root)
    except (OSError, ValueError):
        return []

    preferred = _normalise_session_id(prefer_session_id) if prefer_session_id else None
    ordered: list[SessionInfo] = []
    if preferred:
        ordered.extend(info for info in infos if info.session_id == preferred)
    ordered.extend(info for info in infos if info.session_id != preferred)

    result: list[SessionEvidence] = []
    seen: set[str] = set()
    seen_evidence: set[str] = set()
    remaining = max(0, total_chars)
    for info in ordered:
        if len(result) >= limit or remaining <= 0:
            break
        if info.session_id in seen:
            continue
        seen.add(info.session_id)
        try:
            read = read_session(root, info.session_id)
        except (OSError, ValueError, FileNotFoundError):
            continue
        evidence = _evidence_from_read(
            read, max_chars=min(max_chars, remaining)
        )
        if not evidence.has_text or evidence.session_id in seen_evidence:
            continue
        seen_evidence.add(evidence.session_id)
        result.append(evidence)
        remaining -= _evidence_chars(evidence)
    return result