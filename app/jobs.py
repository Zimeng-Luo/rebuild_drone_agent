"""Persistent bounded FIFO subprocess scheduler. This module never imports torch."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Callable
from urllib.parse import quote
import uuid

ALLOWED_OUTPUTS = {".glb", ".ply", ".log", ".json"}
ACTIVE = {"queued", "running"}
STAGES = (
    ("开始第一遍流式扫描", "preprocessing"),
    ("开始第二遍候选筛选", "keyframe_selection"),
    ("开始分块:", "chunking"),
    ("开始加载去除遮罩模型", "masking"),
    ("开始处理遮罩", "masking"),
    ("开始拼接分块", "aligning"),
    ("开始导出最终 GLB", "exporting_glb"),
    ("开始导出最终 PLY", "exporting_ply"),
    ("开始导出最终下采样点云", "downsampling"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def tail(path: Path, max_lines: int = 200, max_bytes: int = 262144) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            data = stream.read(max_bytes)
        if size > max_bytes:
            data = data.partition(b"\n")[2]
        return list(deque(data.decode("utf-8", errors="replace").splitlines(), maxlen=max_lines))
    except OSError:
        return []


def process_start(pid: int) -> str | None:
    try:
        # comm can contain spaces; fields after its final ')' begin at field 3.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


class BusyError(Exception):
    pass


def gpu_tokens(value: str | list[str]) -> list[str]:
    tokens = value.split(",") if isinstance(value, str) else list(value)
    if not tokens:
        raise ValueError("gpu_devices must contain at least one GPU")
    tokens = [token.strip() if isinstance(token, str) else "" for token in tokens]
    index = re.compile(r"(?:0|[1-9][0-9]*)\Z")
    gpu_uuid = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
    if any(not (index.fullmatch(token) or gpu_uuid.fullmatch(token)) for token in tokens):
        raise ValueError("GPU devices must be physical indices or full GPU UUIDs")
    if len({token.casefold() for token in tokens}) != len(tokens):
        raise ValueError("Duplicate GPU devices are not allowed")
    if len({bool(index.fullmatch(token)) for token in tokens}) > 1:
        raise ValueError("Do not mix GPU indices and UUIDs: they may identify the same device")
    return tokens


def group_alive(group_id: int) -> bool:
    # Zombies no longer own GPU resources; orphan children are reaped by init.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def reap_group(group_id: int) -> None:
    kill_group(group_id, signal.SIGKILL)
    while group_alive(group_id):
        time.sleep(0.01)


@dataclass
class Worker:
    slot: int
    process: subprocess.Popen
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class JobManager:
    def __init__(self, root: Path, worker_command_factory: Callable[[Path], list[str]] | None = None,
                 *, max_concurrent: int = 1, max_queue: int = 8, gpu_devices: str | list[str] | None = None):
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be a non-negative integer")
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.gpu_devices = gpu_tokens(gpu_devices) if gpu_devices is not None else []
        if self.gpu_devices and max_concurrent > len(self.gpu_devices):
            raise ValueError("max_concurrent exceeds the configured GPU pool; each job requires one GPU")
        if max_concurrent > 1 and gpu_devices is None and worker_command_factory is None:
            raise ValueError("max_concurrent > 1 requires an explicit gpu_devices pool")
        self.configuration_error = None
        if not self.gpu_devices and worker_command_factory is None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            try:
                inherited = gpu_tokens(visible)
                if len(inherited) != 1:
                    raise ValueError("CUDA_VISIBLE_DEVICES must contain exactly one GPU")
                self.gpu_devices = inherited
            except ValueError:
                self.configuration_error = ("Set --gpu-devices or CUDA_VISIBLE_DEVICES to one GPU before submitting jobs; "
                                            "inspect nvidia-smi before starting the server")
        self.root = root.resolve()
        self.jobs_dir = self.root / "data/workbench/jobs"
        self.results_dir = self.root / "data/result/workbench"
        self.command_factory = worker_command_factory or (
            lambda job_dir: [sys.executable, "-u", "-m", "app.worker", "--job-dir", str(job_dir)]
        )
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.pending: deque[str] = deque()
        self.running: dict[str, Worker] = {}
        self.closed = False
        self.lock_file = None

    @property
    def active_job_id(self) -> str | None:
        with self.lock:
            return next(iter(self.running), None)

    def scheduler_state(self) -> dict:
        with self.lock:
            active_ids = list(self.running)
            return dict(active_job_id=next(iter(active_ids), None), active_job_ids=active_ids,
                        scheduler=dict(max_concurrent=self.max_concurrent, max_queue=self.max_queue,
                                       running_count=len(active_ids), queued_count=len(self.pending),
                                       available_queue_slots=self.max_queue - len(self.pending),
                                       gpu_devices=list(self.gpu_devices), ready=not self.closed and not self.configuration_error,
                                       configuration_error=self.configuration_error))

    def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.jobs_dir / ".server.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError("Another workbench server already owns this jobs directory")
        for folder in self.jobs_dir.iterdir():
            if not folder.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", folder.name):
                continue
            job = read_json(folder / "job.json")
            if not job or job.get("id") != folder.name:
                continue
            if job.get("status") in ACTIVE:
                process_info = read_json(folder / "process.json") or {}
                pid = process_info.get("pid")
                if isinstance(pid, int) and process_info.get("start") and process_start(pid) == process_info["start"]:
                    try:
                        if os.getpgid(pid) == pid:
                            reap_group(pid)
                    except ProcessLookupError:
                        pass
                job.update(status="interrupted", stage="interrupted", finished_at=now(),
                           error="Server restarted while the job was active; submit a new job to retry.")
                atomic_json(folder / "job.json", job)
            self.jobs[folder.name] = job

    def _save(self, job: dict) -> None:
        atomic_json(self.jobs_dir / job["id"] / "job.json", job)

    def _snapshot(self, job: dict) -> dict:
        result = copy.deepcopy(job)
        result["queue_position"] = self.pending.index(job["id"]) + 1 if job["id"] in self.pending else None
        result.setdefault("gpu_device", None)
        result.setdefault("retry_of", None)
        return result

    def submit(self, input_path: str, read_type: str, name: str, options: dict,
               *, retry_of: str | None = None) -> dict:
        with self.lock:
            if self.closed:
                raise BusyError("Workbench is shutting down")
            if self.configuration_error:
                raise BusyError(self.configuration_error)
            if len(self.running) >= self.max_concurrent and len(self.pending) >= self.max_queue:
                raise BusyError("Job queue is full; wait for a queue slot or cancel a queued job")
            if retry_of is not None:
                original = self.jobs[retry_of]
                if original["status"] not in {"failed", "cancelled", "interrupted"}:
                    raise BusyError("Only failed, cancelled or interrupted jobs can be retried")
            job_id = uuid.uuid4().hex
            folder = self.jobs_dir / job_id
            folder.mkdir()
            (self.results_dir / job_id).mkdir()
            job = dict(id=job_id, name=name or Path(input_path).name, status="queued", stage="queued",
                       created_at=now(), started_at=None, finished_at=None, error=None,
                       options=copy.deepcopy(options), input_path=input_path, read_type=read_type,
                       outputs=[], summary=None, gpu_device=None, retry_of=retry_of)
            atomic_json(folder / "request.json", dict(input_path=input_path, read_type=read_type,
                        options=options, result_dir=str(self.results_dir / job_id), parent_pid=os.getpid()))
            self._save(job)
            self.jobs[job_id] = job
            self.pending.append(job_id)
            self._dispatch()
            return self._snapshot(job)

    def _dispatch(self) -> None:
        # Caller holds the scheduler lock; a slot remains occupied through group cleanup.
        while not self.closed and self.pending and len(self.running) < self.max_concurrent:
            job_id = self.pending.popleft()
            job = self.jobs[job_id]
            slot = next(index for index in range(self.max_concurrent)
                        if index not in {worker.slot for worker in self.running.values()})
            device = self.gpu_devices[slot] if self.gpu_devices else None
            job["gpu_device"] = device
            folder = self.jobs_dir / job_id
            process = None
            try:
                environment = os.environ.copy()
                if device is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = device
                with (folder / "stdout.log").open("ab", buffering=0) as output:
                    process = subprocess.Popen(self.command_factory(folder), cwd=self.root,
                                               stdout=output, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True,
                                               env=environment)
                worker = Worker(slot, process)
                self.running[job_id] = worker
                atomic_json(folder / "process.json", {"pid": process.pid, "start": process_start(process.pid)})
                job.update(status="running", stage="starting", started_at=now())
                self._save(job)
            except Exception as exc:
                if process is None:
                    job.update(status="failed", stage="failed", error=f"Could not start worker: {exc}", finished_at=now())
                    self._save(job)
                    continue
                # Even a metadata write failure must retain the slot until cleanup.
                job.update(status="failed", stage="failed", error=f"Could not start worker: {exc}", finished_at=now())
                worker.stop.set()
            worker.thread = threading.Thread(target=self._watch, args=(job_id, worker), daemon=True)
            worker.thread.start()

    def _refresh_stage(self, job_id: str) -> None:
        lines = tail(self.jobs_dir / job_id / "stdout.log")
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] != "running":
                return
            stage = job["stage"]
            for line in lines:
                marker = re.search(r"\[workbench-stage\] ([a-z_]+)", line)
                if marker:
                    stage = marker.group(1)
                for fragment, candidate in STAGES:
                    if fragment in line:
                        stage = candidate
            if stage != job["stage"]:
                job["stage"] = stage
                self._save(job)

    def _watch(self, job_id: str, worker: Worker) -> None:
        process = worker.process
        result = None
        outputs = []
        monitor_error = None
        code = None
        context = ""
        try:
            while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                if worker.stop.is_set():
                    kill_group(process.pid, signal.SIGTERM)
                    deadline = time.monotonic() + 2
                    while (os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
                           and time.monotonic() < deadline):
                        time.sleep(0.02)
                    break
                worker.stop.wait(0.1)
                self._refresh_stage(job_id)
        except Exception as exc:
            monitor_error = f"Worker monitor failed: {exc}"
        finally:
            # Keep the unreaped leader PID reserved while killing its group. waitid
            # observes exit without reaping, so signals can never hit a reused PID.
            reap_group(process.pid)
            code = process.wait()
        try:
            result = read_json(self.jobs_dir / job_id / "result.json")
            outputs = self._outputs(job_id)
            context = "\n".join(tail(self.jobs_dir / job_id / "stdout.log", 8))
        except Exception as exc:
            monitor_error = f"Worker monitor failed: {exc}"
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] == "running":
                if monitor_error is None and code == 0 and result and result.get("status") == "completed":
                    job.update(status="completed", stage="completed", error=None,
                               summary=result.get("summary") if isinstance(result.get("summary"), dict) else None)
                else:
                    reason = monitor_error or (result.get("error") if result else None)
                    if not reason:
                        reason = f"Worker exited with code {code} without a valid result"
                        if context:
                            reason += f"\n{context}"
                    job.update(status="failed", stage="failed", error=str(reason)[-8000:])
                job["finished_at"] = now()
            job["outputs"] = outputs
            try:
                self._save(job)
            finally:
                self.running.pop(job_id)
                self._dispatch()

    def _outputs(self, job_id: str) -> list[dict]:
        base = (self.results_dir / job_id).resolve()
        outputs = []
        # Pipeline outputs live directly in one timestamp directory. Do not walk raw frames.
        for folder in base.iterdir():
            if not folder.is_dir() or folder.is_symlink():
                continue
            for path in folder.iterdir():
                if path.suffix.lower() not in ALLOWED_OUTPUTS or not path.is_file() or path.is_symlink():
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(base):
                    continue
                relative = path.relative_to(base).as_posix()
                outputs.append(dict(name=relative, url=f"/api/jobs/{job_id}/artifacts/{quote(relative)}",
                                    size_bytes=path.stat().st_size))
        return sorted(outputs, key=lambda item: item["name"])

    def get(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self._snapshot(self.jobs[job_id])

    def list(self) -> list[dict]:
        with self.lock:
            return [self._snapshot(job) for job in sorted(self.jobs.values(), key=lambda job: job["created_at"], reverse=True)]

    def cancel(self, job_id: str, interrupted: bool = False) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            worker = self.running.get(job_id)
            if job["status"] in ACTIVE:
                job.update(status="interrupted" if interrupted else "cancelled",
                           stage="interrupted" if interrupted else "cancelled", finished_at=now(),
                           error="Server shutdown interrupted the job" if interrupted else None)
                if job_id in self.pending:
                    self.pending.remove(job_id)
                self._save(job)
            if worker is not None:
                worker.stop.set()
        # Only the owning monitor signals this job's group, preventing stale-pid races.
        if worker is not None and worker.thread is not threading.current_thread():
            worker.thread.join()
        return self.get(job_id)

    def artifact(self, job_id: str, relative: str) -> Path:
        job = self.get(job_id)
        # Access is limited to enumerated products; no arbitrary file reader.
        if relative not in {item["name"] for item in job["outputs"]}:
            raise FileNotFoundError(relative)
        base = (self.results_dir / job_id).resolve()
        candidate = base / relative
        path = candidate.resolve()
        if candidate.is_symlink() or not path.is_relative_to(base) or path.suffix.lower() not in ALLOWED_OUTPUTS or not path.is_file():
            raise FileNotFoundError(relative)
        return path

    def close(self) -> None:
        with self.lock:
            self.closed = True
            workers = list(self.running.values())
            self.pending.clear()
            for job in self.jobs.values():
                if job["status"] in ACTIVE:
                    job.update(status="interrupted", stage="interrupted", finished_at=now(),
                               error="Server shutdown interrupted the job")
                    self._save(job)
            for worker in workers:
                worker.stop.set()
        for worker in workers:
            if worker.thread is not threading.current_thread():
                worker.thread.join()
        with self.lock:
            if self.running:
                raise RuntimeError("Worker cleanup failed; refusing to release the server lock")
            if self.lock_file is not None:
                fcntl.flock(self.lock_file, fcntl.LOCK_UN)
                self.lock_file.close()
                self.lock_file = None
