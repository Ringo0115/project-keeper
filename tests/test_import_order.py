"""回归：keeper.tidy 必须能作为第一个导入的 keeper 模块。

修复前 ``keeper.tidy`` -> ``keeper.tools.base`` 会触发包初始化
``keeper.tools.__init__`` -> ``keeper.tools.memory`` ->
``from keeper.tidy import TidyLock``，而彼时 ``TidyLock`` 尚未定义。
用子进程模拟全新解释器，避免进程内 sys.modules 污染。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


def test_tidy_can_be_imported_first() -> None:
    result = _run("from keeper.tidy import TidyLock, run_tidy")
    assert result.returncode == 0, result.stderr


def test_tools_then_tidy_import_still_works() -> None:
    result = _run("import keeper.tools.memory; from keeper.tidy import TidyLock")
    assert result.returncode == 0, result.stderr