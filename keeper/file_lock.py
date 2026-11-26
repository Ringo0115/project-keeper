"""跨进程文件锁：Windows 用 ``msvcrt.locking``，POSIX 用 ``fcntl.flock``。

照 Orbital 的 ``agent_os/utils/file_lock.py``：锁由操作系统持有，进程一死
就自动释放，因此不需要记 pid、也不需要判断锁龄是不是过期。keeper 用
「旁边放一个同名 ``.lock`` 空文件」这一路（Orbital 在 Windows 上也是这么做
的），锁文件本身不装内容，释放后留在原地，下次照样能锁。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _acquire_nb(fd: int) -> bool:
    """非阻塞抢锁；抢不到返回 False。"""
    if sys.platform == "win32":
        import msvcrt

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _release(fd: int) -> None:
    """放锁；失败不抛。"""
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


class FileLock:
    """一个 ``.lock`` 文件上的非阻塞互斥锁。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """抢锁；已被别的进程（或本进程另一个描述符）持有时返回 False。"""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return False
        if not _acquire_nb(fd):
            try:
                os.close(fd)
            except OSError:
                pass
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        _release(fd)
        try:
            os.close(fd)
        except OSError:
            pass

    def is_locked(self) -> bool:
        """锁是否正被某个进程持有。锁文件不存在时不会顺手创建它。"""
        if not self.path.exists():
            return False
        if not self.acquire():
            return True
        self.release()
        return False

    def __enter__(self) -> "FileLock":
        if not self.acquire():
            raise RuntimeError(f"lock is busy: {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass