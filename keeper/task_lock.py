"""Per-project process lock for persistent task queue operations."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from keeper.tidy import _pid_alive

TASK_LOCK_STALE_SECONDS = 2 * 60 * 60
TASK_LOCK_NAME = "tasks.lock"


class TaskLock:
    """Non-blocking O_EXCL lock; dead owners are reclaimed, live owners never are."""

    def __init__(self, root: Path, *, stale_seconds: float = TASK_LOCK_STALE_SECONDS):
        self.root = Path(root).resolve()
        self.path = self.root / ".agent" / TASK_LOCK_NAME
        self.stale_seconds = stale_seconds
        self._owned = False

    def _read(self) -> dict | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _stale(self) -> bool:
        data = self._read()
        age = None
        try:
            age = max(0.0, time.time() - self.path.stat().st_mtime)
        except OSError:
            return True
        pid = data.get("pid") if data else None
        host = data.get("host") if data else None
        if host and host != socket.gethostname():
            return age > self.stale_seconds and not isinstance(pid, int)
        if isinstance(pid, int) and pid > 0:
            return not _pid_alive(pid)
        return age > self.stale_seconds

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._stale():
                    return False
                try:
                    self.path.unlink()
                except OSError:
                    return False
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_at": time.time()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self._owned = True
            return True
        return False

    def release(self) -> None:
        if not self._owned:
            return
        data = self._read()
        if data and data.get("pid") == os.getpid():
            try:
                self.path.unlink()
            except OSError:
                pass
        self._owned = False
