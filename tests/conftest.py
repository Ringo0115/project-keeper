"""测试隔离：每个用例使用独立的 KEEPER_HOME，绝不读写真实 ~/.keeper。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_keeper_home(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("keeper-home")
    monkeypatch.setenv("KEEPER_HOME", str(home))