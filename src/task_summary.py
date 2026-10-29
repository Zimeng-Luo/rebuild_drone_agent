from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import torch


SUMMARY_LOG_NAME = "task_summary.log"
_SAMPLE_INTERVAL_SECONDS = 0.5
_INTERMEDIATE_SUFFIXES = {".npz", ".pkl"}


@dataclass
class StepSummary:
    name: str
    started_at: str
    finished_at: str
    duration_seconds: float
    cpu_percent_mean: float | None
    cpu_percent_peak: float | None
    memory_rss_mb_mean: float | None
    memory_rss_mb_peak: float | None
    gpu_memory_allocated_mb_mean: float | None
    gpu_memory_allocated_mb_peak: float | None
    gpu_memory_reserved_mb_mean: float | None
    gpu_memory_reserved_mb_peak: float | None


class ResourceStepMonitor:
    """Sample process CPU, RSS, and torch CUDA memory during one pipeline step."""

    def __init__(self, name: str, interval_seconds: float = _SAMPLE_INTERVAL_SECONDS):
        self.name = name
        self.interval_seconds = interval_seconds
        self.started_at = ""
        self.finished_at = ""
        self._start_wall = 0.0
        self._start_process_cpu = 0.0
        self._last_sample_wall = 0.0
        self._last_sample_process_cpu = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_percent_samples: list[float] = []
        self._rss_samples_mb: list[float] = []
        self._gpu_allocated_samples_mb: list[float] = []
        self._gpu_reserved_samples_mb: list[float] = []

    def __enter__(self) -> ResourceStepMonitor:
        self.started_at = _now_text()
        self._start_wall = time.perf_counter()
        self._start_process_cpu = time.process_time()
        self._last_sample_wall = self._start_wall
        self._last_sample_process_cpu = self._start_process_cpu
        self._sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)
        self._sample()
        self.finished_at = _now_text()

    def summary(self) -> StepSummary:
        duration_seconds = time.perf_counter() - self._start_wall
        cpu_seconds = time.process_time() - self._start_process_cpu
        cpu_percent_mean = _cpu_percent(cpu_seconds, duration_seconds)
        return StepSummary(
            name=self.name,
            started_at=self.started_at,
            finished_at=self.finished_at or _now_text(),
            duration_seconds=round(duration_seconds, 3),
            cpu_percent_mean=cpu_percent_mean,
            cpu_percent_peak=_max(self._cpu_percent_samples),
            memory_rss_mb_mean=_mean(self._rss_samples_mb),
            memory_rss_mb_peak=_max(self._rss_samples_mb),
            gpu_memory_allocated_mb_mean=_mean(self._gpu_allocated_samples_mb),
            gpu_memory_allocated_mb_peak=_max(self._gpu_allocated_samples_mb),
            gpu_memory_reserved_mb_mean=_mean(self._gpu_reserved_samples_mb),
            gpu_memory_reserved_mb_peak=_max(self._gpu_reserved_samples_mb),
        )

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        now_wall = time.perf_counter()
        now_process_cpu = time.process_time()
        cpu_percent = _cpu_percent(
            now_process_cpu - self._last_sample_process_cpu,
            now_wall - self._last_sample_wall,
        )
        if cpu_percent is not None:
            self._cpu_percent_samples.append(cpu_percent)
        self._last_sample_wall = now_wall
        self._last_sample_process_cpu = now_process_cpu

        rss_mb = _process_rss_mb()
        if rss_mb is not None:
            self._rss_samples_mb.append(rss_mb)

        if torch.cuda.is_available():
            self._gpu_allocated_samples_mb.append(_bytes_to_mb(torch.cuda.memory_allocated()))
            self._gpu_reserved_samples_mb.append(_bytes_to_mb(torch.cuda.memory_reserved()))


def collect_output_sizes(task_dir: str | Path) -> dict[str, Any]:
    """Collect per-output and aggregate file sizes under the task directory."""
    task_dir = Path(task_dir)
    files: list[dict[str, Any]] = []
    total_bytes = 0
    by_suffix: dict[str, int] = {}

    if not task_dir.exists():
        return {"total_bytes": 0, "total_mb": 0.0, "by_suffix": {}, "files": []}

    for path in sorted(item for item in task_dir.rglob("*") if item.is_file()):
        size_bytes = path.stat().st_size
        suffix = path.suffix.lower() or "<no_suffix>"
        total_bytes += size_bytes
        by_suffix[suffix] = by_suffix.get(suffix, 0) + size_bytes
        files.append(
            {
                "path": str(path.relative_to(task_dir)),
                "bytes": size_bytes,
                "mb": round(_bytes_to_mb(size_bytes), 3),
            }
        )

    return {
        "total_bytes": total_bytes,
        "total_mb": round(_bytes_to_mb(total_bytes), 3),
        "by_suffix": {
            suffix: {"bytes": size, "mb": round(_bytes_to_mb(size), 3)}
            for suffix, size in sorted(by_suffix.items())
        },
        "files": files,
    }


def delete_numbered_intermediate_files(task_dir: str | Path) -> list[str]:
    """Delete numbered chunk intermediates and preprocessed frame images."""
    task_dir = Path(task_dir)
    if not task_dir.exists():
        return []

    deleted_paths: list[str] = []
    for path in sorted(task_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _INTERMEDIATE_SUFFIXES:
            continue
        if not path.stem.isdigit():
            continue

        path.unlink()
        deleted_paths.append(str(path.relative_to(task_dir)))

    preprocessed_frame_dir = task_dir / "preprocessed_frames"
    if preprocessed_frame_dir.is_dir():
        for path in sorted(preprocessed_frame_dir.rglob("*")):
            if path.is_file():
                deleted_paths.append(str(path.relative_to(task_dir)))
        shutil.rmtree(preprocessed_frame_dir)
        deleted_paths.append(str(preprocessed_frame_dir.relative_to(task_dir)))

    return deleted_paths


def write_task_summary_log(
    task_dir: str | Path,
    *,
    created_at: str,
    parameters: dict[str, Any],
    steps: list[StepSummary],
    total: StepSummary,
    outputs: dict[str, Any],
    deleted_intermediate_files: list[str] | None = None,
) -> Path:
    """Write one independent JSON summary log under the task directory."""
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_dir / SUMMARY_LOG_NAME
    payload = {
        "basic": {
            "task_created_at": created_at,
            "summary_created_at": _now_text(),
        },
        "time": {
            "total": _step_to_dict(total),
            "steps": [_step_to_dict(step) for step in steps],
        },
        "space": outputs,
        "parameters": parameters,
        "cleanup": {
            "deleted_intermediate_files": deleted_intermediate_files or [],
        },
        "performance": {
            "total": _performance_to_dict(total),
            "steps": {
                step.name: _performance_to_dict(step)
                for step in steps
            },
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary_path


def _step_to_dict(step: StepSummary) -> dict[str, Any]:
    return {
        "name": step.name,
        "started_at": step.started_at,
        "finished_at": step.finished_at,
        "duration_seconds": step.duration_seconds,
    }


def _performance_to_dict(step: StepSummary) -> dict[str, Any]:
    return {
        "cpu_percent": {
            "mean": step.cpu_percent_mean,
            "peak": step.cpu_percent_peak,
        },
        "memory_rss_mb": {
            "mean": step.memory_rss_mb_mean,
            "peak": step.memory_rss_mb_peak,
        },
        "gpu_memory_allocated_mb": {
            "mean": step.gpu_memory_allocated_mb_mean,
            "peak": step.gpu_memory_allocated_mb_peak,
        },
        "gpu_memory_reserved_mb": {
            "mean": step.gpu_memory_reserved_mb_mean,
            "peak": step.gpu_memory_reserved_mb_peak,
        },
    }


def _process_rss_mb() -> float | None:
    statm_path = Path("/proc/self/statm")
    if not statm_path.exists():
        return None

    parts = statm_path.read_text(encoding="utf-8").split()
    if len(parts) < 2:
        return None

    resident_pages = int(parts[1])
    return round(_bytes_to_mb(resident_pages * os.sysconf("SC_PAGE_SIZE")), 3)


def _cpu_percent(cpu_seconds: float, wall_seconds: float) -> float | None:
    if wall_seconds <= 0:
        return None
    return round((cpu_seconds / wall_seconds) * 100.0, 3)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 3)


def _bytes_to_mb(value: int | float) -> float:
    return float(value) / 1024.0 / 1024.0


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
