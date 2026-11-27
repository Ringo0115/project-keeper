"""M6 worker dispatch, approval, and completion records."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from keeper.events import ApprovalRequest
from keeper.fanout import FanoutBatch
from keeper.memory import FILE_NAMES, INJECTION_ORDER, memory_path
from keeper.projects import Project
from keeper.tools.base import Tool, ToolResult, error
from keeper.tools.fs import ListDirTool, ReadFileTool, SearchTool
from keeper.tools.registry import Registry
from keeper.workers import WorkerSpec, default_workers, worker_memory_path

if TYPE_CHECKING:
    from keeper.context import WorkerBrief

COMPLETION_RESULTS = frozenset({"done", "blocked", "failed", "interrupted"})
try:
    DEFAULT_TIMEOUT = max(0.1, float(os.getenv("KEEPER_WORKER_TIMEOUT", "1800")))
except ValueError:
    DEFAULT_TIMEOUT = 1800.0
_CLI_DEFAULTS = {
    # Defaults only; KEEPER_WORKER_<SLUG>_ARGV JSON array overrides each template.
    "claude-code": ["claude", "-p"],
    "codex": ["codex", "exec", "-"],
    "pi": ["pi", "--print"],
}


@dataclass(frozen=True)
class CompletionRecord:
    worker: str
    result: str
    summary: str
    reply: str
    transcript: str | None
    started_at: str
    ended_at: str
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.worker:
            raise ValueError("completion record 缺少 worker")
        if self.result not in COMPLETION_RESULTS:
            raise ValueError(f"completion record 的 result={self.result!r} 不合法")
        if self.result == "blocked" and not self.summary.strip():
            raise ValueError("blocked completion 必须包含原因")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(worker: str, result: str, summary: str, reply: str = "", transcript: str | None = None,
            started: str | None = None, verified: bool = False) -> CompletionRecord:
    return CompletionRecord(worker, result, summary, reply, transcript, started or _now(), _now(), verified)


def _approval(project: Project, assignments: list[tuple[WorkerSpec, WorkerBrief]], approver: Any) -> str | None:
    rows = []
    for worker, brief in assignments:
        rows.append(
            f"worker: {worker.slug} ({worker.name})\nlabel: {brief.label}\n"
            f"task: {brief.task}\nallowed: {', '.join(brief.allowed) or '(未限定)'}\n"
            f"forbidden: {', '.join(brief.forbidden) or '(无)'}"
        )
    req = ApprovalRequest(tool="dispatch", summary=f"派发 {len(assignments)} 个任务到项目 {project.name} ({project.root})",
                          detail="\n\n---\n\n".join(rows))
    try:
        if approver is None or not bool(approver.confirm(req)):
            return "用户拒绝或审批失败"
    except BaseException as exc:
        return f"审批失败（{type(exc).__name__}: {exc}）"
    return None


def _snapshot(root: Path) -> dict[Path, bytes | None]:
    return {memory_path(root, FILE_NAMES[key]): memory_path(root, FILE_NAMES[key]).read_bytes()
            if memory_path(root, FILE_NAMES[key]).is_file() else None for key in INJECTION_ORDER}


def _restore_changed(snapshot: dict[Path, bytes | None]) -> list[Path]:
    changed = []
    for path, prior in snapshot.items():
        current = path.read_bytes() if path.is_file() else None
        if current == prior:
            continue
        changed.append(path)
        if prior is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(prior)
    return changed


def _project_fingerprint(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for name in files:
            path = Path(base) / name
            try:
                rel = path.relative_to(root).as_posix().casefold()
                found[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
    return found


def _within_allowed(rel: str, root: Path, brief: WorkerBrief) -> bool:
    for allowed in brief.allowed:
        try:
            base = _safe_target(root, allowed).relative_to(root).as_posix().casefold()
            if rel == base or rel.startswith(base.rstrip("/") + "/"):
                return True
        except (ValueError, OSError):
            continue
    return False


def _scope_note(before_files: dict[str, str], root: Path, brief: WorkerBrief, slug: str,
                transcript_path: Path) -> str:
    """中断 / 超时 / 启动失败路径上的越界提示；只提示，不还原。"""
    try:
        after = _project_fingerprint(root)
        outside = _outside_allowed(before_files, after, root, brief, slug, transcript_path)
    except (OSError, ValueError):
        return ""
    return f"；提示：检测到 allowed 范围外改动：{', '.join(outside[:20])}" if outside else ""


def _outside_allowed(before: dict[str, str], after: dict[str, str], root: Path,
                     brief: WorkerBrief, slug: str, exclude: Path | None = None) -> list[str]:
    changed = [path for path in set(before) | set(after) if before.get(path) != after.get(path)]
    outside = []
    excluded = exclude.relative_to(root).as_posix().casefold() if exclude else None
    for rel in changed:
        if rel == excluded:
            continue
        try:
            target = (root / Path(rel)).resolve()
            if not _allowed_target(target, root, brief, slug):
                outside.append(rel)
        except (OSError, ValueError):
            outside.append(rel)
    return sorted(outside)


def _transcript_path(root: Path, slug: str) -> Path:
    path = worker_memory_path(root, slug).parent / "runs" / f"{uuid.uuid4().hex}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cli_argv(slug: str, task: str) -> list[str]:
    key = f"KEEPER_WORKER_{slug.upper().replace('-', '_')}_ARGV"
    raw = os.getenv(key)
    if raw:
        value = json.loads(raw)
        if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
            raise ValueError(f"{key} must be a non-empty JSON argv array")
        argv = [x.replace("{input}", task) for x in value]
    else:
        template = _CLI_DEFAULTS.get(slug)
        if template is None:
            raise ValueError(f"没有 {slug} 的 CLI argv 模板")
        argv = [part.replace("{input}", task) for part in template]
    # Windows 上 npm 装的是 .CMD 垫片，CreateProcess 只补 .exe 不补 .CMD；
    # 用 which() 解析成真实可执行文件路径，shell=False 才能直接启动。
    name = argv[0]
    resolved = shutil.which(name)
    if resolved is None:
        looks_like_path = Path(name).is_absolute() or os.sep in name or bool(os.altsep and os.altsep in name)
        hint = "" if looks_like_path else f"；可用环境变量 {key} 覆盖"
        raise ValueError(f"{name} 不存在或不是可执行文件{hint}")
    argv[0] = resolved
    return argv


def _safe_target(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("path escapes project root")
    return target


def _inside_memory(target: Path, root: Path) -> bool:
    memory_root = (root / ".agent" / "memory").resolve()
    try:
        return os.path.normcase(str(target)).startswith(os.path.normcase(str(memory_root)) + os.sep)
    except OSError:
        return False


def _protected_memory_target(target: Path, root: Path) -> bool:
    if not _inside_memory(target, root):
        return False
    name = target.name.casefold()
    # Block canonical four files, archives/backups, and tidy markers in memory.
    canonical = {FILE_NAMES[k].casefold() for k in INJECTION_ORDER}
    return (name in canonical or name.endswith("_archive.md") or ".tidy" in name
            or "backup" in {p.casefold() for p in target.parts})


def _allowed_target(target: Path, root: Path, brief: WorkerBrief, slug: str) -> bool:
    private = worker_memory_path(root, slug).resolve()
    try:
        if os.path.normcase(str(target)) == os.path.normcase(str(private)):
            return True
    except OSError:
        pass
    rel = target.relative_to(root).as_posix().casefold()
    for denied in brief.forbidden:
        try:
            deny = _safe_target(root, denied).relative_to(root).as_posix().casefold()
            if rel == deny or rel.startswith(deny.rstrip("/") + "/"):
                return False
        except (ValueError, OSError):
            return False
    if not brief.allowed:
        return False
    for allowed in brief.allowed:
        try:
            base = _safe_target(root, allowed).relative_to(root).as_posix().casefold()
            if rel == base or rel.startswith(base.rstrip("/") + "/"):
                return True
        except (ValueError, OSError):
            continue
    return False


class ScopedWriteTool(Tool):
    """Worker-only writer enforcing allowed/forbidden at the tool boundary."""
    name = "write_file"
    description = "写入本次派发 allowed 范围或自己的私有 MEMORY.md。"
    parameters = {"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}
    danger = "write"

    def __init__(self, root: Path, brief: WorkerBrief, slug: str) -> None:
        self.root = Path(root).resolve()
        self.brief = brief
        self.slug = slug

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            target = _safe_target(self.root, args.get("path", ""))
            if _protected_memory_target(target, self.root):
                return error("拒绝写入项目共享记忆、备份/归档或 tidy 标记")
            if not _allowed_target(target, self.root, self.brief, self.slug):
                return error("path 不在本次 allowed 范围内或命中 forbidden")
            content = args.get("content")
            if not isinstance(content, str):
                return error("content must be a string")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
            return ToolResult(True, f"wrote {target}")
        except (ValueError, OSError) as exc:
            return error(str(exc))


class SubmitResultTool(Tool):
    name = "submit_result"
    description = "Submit a non-empty summary of the completed task."
    parameters = {"type": "object", "properties": {"summary": {"type": "string"}},
                  "required": ["summary"]}

    def __init__(self, submitted: list[str]) -> None:
        self.submitted = submitted

    def execute(self, args: dict[str, Any]) -> ToolResult:
        summary = args.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return error("summary must be a non-empty string")
        self.submitted.append(summary.strip())
        return ToolResult(True, "result submitted")


class _DispatchAuthorizedApprover:
    """The one dispatch approval authorizes scoped worker writes hands-off."""
    def confirm(self, request: ApprovalRequest) -> bool:
        return request.tool == "write_file"


def _worker_registry(root: Path, brief: WorkerBrief, slug: str) -> Registry:
    return Registry([ReadFileTool(root), ListDirTool(root), SearchTool(root),
                     ScopedWriteTool(root, brief, slug), SubmitResultTool([])])


# ---------------------------------------------------------------------------
# 活着的 worker 子进程：中断时整棵进程树一起杀，不留后台孤儿
# ---------------------------------------------------------------------------

_LIVE_LOCK = threading.Lock()
_LIVE_PROCS: dict[int, subprocess.Popen] = {}


def _register_live(proc: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE_PROCS[proc.pid] = proc


def _unregister_live(proc: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE_PROCS.pop(proc.pid, None)


def kill_live_workers() -> list[int]:
    """杀掉所有仍在运行的 worker 子进程树，返回被杀的 pid 列表。"""
    with _LIVE_LOCK:
        procs = list(_LIVE_PROCS.values())
    killed: list[int] = []
    for proc in procs:
        if proc.poll() is None:
            _kill_tree(proc)
            killed.append(proc.pid)
    return killed


def _kill_tree(proc: subprocess.Popen) -> None:
    """Windows 用 taskkill /T 连子进程一起杀；POSIX 先 terminate 再 kill。"""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _builtin_dispatch(project: Project, worker: WorkerSpec, brief: WorkerBrief, context: str,
                      *, llm: Any, max_steps: int, timeout: float) -> tuple[str, str, str | None, str]:
    """Run an isolated AgentLoop. Shell and all mutating project tools are absent."""
    from keeper.loop import AgentLoop
    from keeper.session import Session
    root = Path(project.root).resolve()
    session = Session.create(root, "worker")
    transcript = str(session.path)
    try:
        submitted: list[str] = []
        registry = Registry([ReadFileTool(root), ListDirTool(root), SearchTool(root),
                             ScopedWriteTool(root, brief, worker.slug), SubmitResultTool(submitted)])
        loop = AgentLoop(llm=llm, registry=registry, session=session, root=root,
                         max_steps=max_steps, approver=_DispatchAuthorizedApprover(), system_override=(
                             "Restricted worker: MUST call submit_result with a non-empty summary before "
                             "finishing; a final answer alone is not delivery.\n" + context))
        result = "failed"; reply = ""
        for event in loop.run(context):
            if hasattr(event, "text") and event.__class__.__name__ == "FinalAnswer":
                reply = event.text
                result = "done" if submitted else "failed"
            if event.__class__.__name__ == "Stop":
                result = "failed"
        summary = submitted[-1] if submitted else "未交付结果：worker 结束时没有调用 submit_result"
        if not reply.strip(): reply = summary if not submitted else "内置 worker 未产生非空最终回复"
        return result, reply, transcript, summary
    finally:
        session.close()


def _run_one_inner(project: Project, worker: WorkerSpec, brief: WorkerBrief, *, llm: Any,
             max_steps: int, timeout: float, completed: Iterable[CompletionRecord] = (),
             since: str | int | float | None = None,
             session_path: Path | None = None) -> CompletionRecord:
    started = _now(); root = Path(project.root).resolve()
    try:
        from keeper.context import worker_context
        context = worker_context(root, project, worker, brief, since=since, completed=completed,
                                 session_path=session_path)
        if worker.kind == "built-in":
            result, reply, transcript, summary = _builtin_dispatch(project, worker, brief, context,
                                                          llm=llm, max_steps=max_steps, timeout=timeout)
            return _record(worker.slug, result, summary, reply, transcript, started, verified=result == "done")
        snapshot = _snapshot(root)
        before_files = _project_fingerprint(root)
        transcript_path = _transcript_path(root, worker.slug)
        argv = _cli_argv(worker.slug, context)
        try:
            proc = subprocess.Popen(argv, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    errors="replace", shell=False)
            _register_live(proc)
            try:
                out, err = proc.communicate(input=context, timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                raise
            except KeyboardInterrupt:
                _kill_tree(proc)
                raise
            finally:
                _unregister_live(proc)
            stdout = out or ""
            stderr_text = err or ""
            reply = stdout.strip()
            stderr = stderr_text.strip()[:2000]
            with transcript_path.open("w", encoding="utf-8") as fh:
                fh.write("ARGV: " + json.dumps(argv, ensure_ascii=False) + "\nSTDIN:\n" + context +
                         "\nSTDOUT:\n" + stdout + "\nSTDERR:\n" + stderr_text)
            changed = _restore_changed(snapshot)
            if changed:
                return _record(worker.slug, "failed", "worker 违规写了项目记忆文件；已从快照还原", reply, str(transcript_path), started)
            after_files = _project_fingerprint(root)
            outside = _outside_allowed(before_files, after_files, root, brief, worker.slug, transcript_path)
            allowed_changed = any(before_files.get(rel) != after_files.get(rel)
                                  for rel in set(before_files) | set(after_files)
                                  if _within_allowed(rel, root, brief))
            scope_note = f"；提示：检测到 allowed 范围外改动：{', '.join(outside[:20])}" if outside else ""
            if proc.returncode != 0:
                return _record(worker.slug, "failed", f"退出码 {proc.returncode}; stderr: {stderr or '(空)'}{scope_note}", reply, str(transcript_path), started)
            if not reply:
                return _record(worker.slug, "failed", f"退出码为 0 但回复为空，不能判定成功{scope_note}", "", str(transcript_path), started)
            if brief.allowed and not allowed_changed:
                return _record(worker.slug, "failed", "声称完成但可写范围内无任何改动" + scope_note, reply, str(transcript_path), started)
            return _record(worker.slug, "done", f"外部 worker 返回候选结果（需管理者核验）{scope_note}", reply, str(transcript_path), started, verified=bool(brief.allowed and allowed_changed))
        except subprocess.TimeoutExpired as exc:
            _restore_changed(snapshot)
            note = _scope_note(before_files, root, brief, worker.slug, transcript_path)
            output = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            transcript_path.write_text(f"STDIN:\n{context}\nSTDOUT:\n{output}\nSTDERR:\n{err}", encoding="utf-8")
            return _record(worker.slug, "interrupted", f"worker 超时（{timeout:g}s）{note}", output, str(transcript_path), started)
        except KeyboardInterrupt:
            _restore_changed(snapshot)
            note = _scope_note(before_files, root, brief, worker.slug, transcript_path)
            transcript_path.write_text(f"STDIN:\n{context}\n\n[interrupted]", encoding="utf-8")
            return _record(worker.slug, "interrupted", f"worker 被中断{note}", "", str(transcript_path), started)
        except OSError as exc:
            _restore_changed(snapshot)
            note = _scope_note(before_files, root, brief, worker.slug, transcript_path)
            transcript_path.write_text(f"STDIN:\n{context}\n\n[launch failed: {exc}]", encoding="utf-8")
            return _record(worker.slug, "failed", f"无法启动 worker：{exc}{note}", "", str(transcript_path), started)
        finally:
            pass
    except Exception as exc:
        return _record(worker.slug, "failed", f"派发失败：{type(exc).__name__}: {exc}", "", None, started)


def _run_one(project: Project, worker: WorkerSpec, brief: WorkerBrief, *, batch: FanoutBatch | None = None,
             **rest: Any) -> CompletionRecord:
    """跑一个 worker；有批次记录时把结果吸收进批次（只吸收一次）。"""
    record = _run_one_inner(project, worker, brief, **rest)
    if batch is not None:
        try:
            batch.absorb(brief.label, status=record.result, summary=record.summary,
                         transcript=record.transcript)
        except OSError:
            pass
    return record


def dispatch_one(project: Project, brief: WorkerBrief, worker: WorkerSpec | str,
                 approver: Any, *, llm: Any = None, max_steps: int = 30,
                 timeout: float = DEFAULT_TIMEOUT,
                 completed: Iterable[CompletionRecord] = (),
                 since: str | int | float | None = None,
                 session_path: Path | None = None) -> CompletionRecord:
    spec = next((w for w in default_workers() if w.slug == worker), None) if isinstance(worker, str) else worker
    if spec is None: return _record(str(worker), "failed", "未知 worker")
    reason = _approval(project, [(spec, brief)], approver)
    if reason: return _record(spec.slug, "blocked", reason)
    if spec.kind == "built-in" and llm is None:
        return _record(spec.slug, "failed", "built-in worker 缺少 LLM client")
    return _run_one(project, spec, brief, llm=llm, max_steps=max_steps, timeout=timeout,
                    completed=completed, since=since, session_path=session_path)


def dispatch_fanout(project: Project, briefs: Iterable[WorkerBrief], workers: Iterable[WorkerSpec | str],
                    approver: Any, *, llm: Any = None, max_steps: int = 30,
                    timeout: float = DEFAULT_TIMEOUT) -> list[CompletionRecord]:
    jobs = list(briefs); selected = list(workers)
    if not 2 <= len(jobs) <= 5: raise ValueError("fanout 子任务数量必须为 2–5")
    if len(jobs) != len(selected): raise ValueError("fanout briefs 与 workers 数量必须一致")
    labels = [job.label.casefold() for job in jobs]
    if len(set(labels)) != len(labels): raise ValueError("fanout label 必须互不重复")
    specs = [next((w for w in default_workers() if w.slug == item), None) if isinstance(item, str) else item for item in selected]
    if any(spec is None for spec in specs): raise ValueError("fanout 含未知 worker")
    typed = [(spec, job) for spec, job in zip(specs, jobs) if spec is not None]
    reason = _approval(project, typed, approver)
    if reason: return [_record(spec.slug, "blocked", reason) for spec, _ in typed]
    if any(spec.kind == "built-in" for spec, _ in typed) and llm is None:
        return [_record(spec.slug, "failed", "built-in worker 缺少 LLM client") for spec, _ in typed]
    root = Path(project.root).resolve()
    try:
        batch: FanoutBatch | None = FanoutBatch.create(
            root, [(spec.slug, job.label) for spec, job in typed])
    except OSError:
        batch = None  # 批次记录落盘失败不阻断派发，只是事后查不到这一批
    results: list[CompletionRecord | None] = [None] * len(typed)
    pool = ThreadPoolExecutor(max_workers=len(typed), thread_name_prefix="keeper-worker")
    futures = {pool.submit(_run_one, project, spec, job, batch=batch, llm=llm,
                           max_steps=max_steps, timeout=timeout): i
               for i, (spec, job) in enumerate(typed)}
    try:
        for future in as_completed(futures):
            i = futures[future]
            try: results[i] = future.result()
            except BaseException as exc: results[i] = _record(typed[i][0].slug, "failed", f"worker crashed: {exc}")
    except BaseException:
        # 中断：先杀掉还在跑的子进程树，再把整批收尾落盘，最后照旧往外抛
        kill_live_workers()
        if batch is not None:
            try: batch.resolve("interrupted")
            except OSError: pass
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=False)
    if batch is not None:
        try: batch.resolve("finished")
        except OSError: pass
    return [record for record in results if record is not None]
