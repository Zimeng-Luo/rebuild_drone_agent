"""Isolated adapter for the existing reconstruction pipeline."""
from __future__ import annotations

import argparse
import ctypes
import functools
import os
from pathlib import Path
import signal
import sys
import traceback

from app.jobs import atomic_json, read_json


def protect_parent(parent_pid: int) -> None:
    """Linux parent-death signal covers abrupt server death as well as shutdown."""
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "Cannot configure worker parent-death signal")
    def terminate(_signum, _frame):
        # The worker owns its session, including pipeline descendants.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    if os.getppid() != parent_pid:
        terminate(signal.SIGTERM, None)


def run(job_dir: Path) -> int:
    request = read_json(job_dir / "request.json")
    try:
        if request is None:
            raise ValueError("Missing or invalid request.json")
        protect_parent(request["parent_pid"])
        print("[workbench-stage] loading_pipeline", flush=True)
        # Deliberately delayed: importing the API must not load torch or models.
        import main as pipeline_main
        for attribute, stage in (
            ("run_data_preprocess_with_task_dir", "preprocessing"),
            ("run_rebuild_chunks", "rebuilding"),
            ("run_align_and_merge", "aligning"),
        ):
            original = getattr(pipeline_main, attribute)
            def with_stage(*args, _function=original, _stage=stage, **kwargs):
                print(f"[workbench-stage] {_stage}", flush=True)
                return _function(*args, **kwargs)
            setattr(pipeline_main, attribute, functools.wraps(original)(with_stage))
        options = request["options"]
        output = pipeline_main.run_pipeline(
            input_path=request["input_path"], result_dir=request["result_dir"], read_type=request["read_type"],
            image_sample_interval=options["image_sample_interval"],
            remove_mask_enable=options["remove_mask_enable"],
            align_glb_conf_thres=options["confidence_percentile"],
            align_ply_conf_thres=options["confidence_percentile"],
            pointcloud_downsample_enable=options["downsample_enable"],
            export_glb_enable=True, export_ply_enable=True,
        )
        output = Path(output).resolve()
        base = Path(request["result_dir"]).resolve()
        if not output.is_relative_to(base) or not output.is_file():
            raise RuntimeError("Pipeline returned no valid output file in the job result directory")
        summary = read_json(output.parent / "task_summary.log")
        atomic_json(job_dir / "result.json", dict(status="completed", output_path=str(output), summary=summary))
        return 0
    except Exception as exc:
        traceback.print_exc()
        atomic_json(job_dir / "result.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated workbench reconstruction job")
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.job_dir.resolve()))


if __name__ == "__main__":
    main()
