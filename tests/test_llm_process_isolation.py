"""跨进程 LLM 请求不在 keeper 内本地排队（真实 HTTP + 两个子进程）。

这是 M5.1b 后台整理可行性评估的常驻回归：后台整理请求在飞时，前台会话
的请求必须在 keeper 侧立刻发出，而不是等待后台请求自然完成。它只证明
“keeper 没有本地串行化”；provider 若串行 / 限流，前台仍可能等待，见
docs/BACKGROUND_TIDY_FEASIBILITY.md。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHILD = """
import sys
from keeper.llm import LLMClient

base_url, label = sys.argv[1], sys.argv[2]
client = LLMClient(base_url=base_url, api_key="dummy", model="fake", timeout=20)
try:
    prompt = "TIDY_PROBE" if label == "tidy" else "USER_PROBE"
    response = client.chat([{"role": "user", "content": prompt}])
    print(response.message.get("content"))
finally:
    client.close()
"""


class _Coordinator:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[str] = []
        self.tidy_arrived = threading.Event()
        self.user_arrived = threading.Event()
        self.tidy_release = threading.Event()

    def record(self, kind: str, label: str) -> None:
        with self.lock:
            self.events.append(f"{kind}:{label}")

    def snapshot(self) -> list[str]:
        with self.lock:
            return list(self.events)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        coord: _Coordinator = self.server.coordinator  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            body = {}
        text = "\n".join(
            message.get("content") or ""
            for message in body.get("messages") or []
        )
        label = "tidy" if "TIDY_PROBE" in text else "user"
        coord.record("arrive", label)
        if label == "tidy":
            coord.tidy_arrived.set()
            coord.tidy_release.wait(timeout=15)
        else:
            coord.user_arrived.set()
        coord.record("done", label)

        payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        ).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except OSError:
            pass


def _start() -> tuple[ThreadingHTTPServer, _Coordinator]:
    coord = _Coordinator()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.coordinator = coord  # type: ignore[attr-defined]

    def quiet_error(request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, OSError)):
            return
        raise exc

    server.handle_error = quiet_error  # type: ignore[method-assign]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, coord


def _spawn(base_url: str, label: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.Popen(
        [sys.executable, "-c", CHILD, base_url, label],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_user_request_is_not_locally_queued_behind_background_tidy() -> None:
    server, coord = _start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    tidy: subprocess.Popen[str] | None = None
    user: subprocess.Popen[str] | None = None
    try:
        tidy = _spawn(base_url, "tidy")
        assert coord.tidy_arrived.wait(timeout=15)

        user = _spawn(base_url, "user")
        assert coord.user_arrived.wait(timeout=15)
        user_out, user_err = user.communicate(timeout=15)
        assert user.returncode == 0, user_err
        assert "ok" in user_out

        # 后台请求仍被服务端 hold 住，前台已经完成。
        assert tidy.poll() is None

        coord.tidy_release.set()
        tidy_out, tidy_err = tidy.communicate(timeout=15)
        assert tidy.returncode == 0, tidy_err
        assert "ok" in tidy_out

        events = coord.snapshot()
        assert events.index("done:user") < events.index("done:tidy")
    finally:
        coord.tidy_release.set()
        for proc in (tidy, user):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.communicate()
        server.shutdown()
        server.server_close()