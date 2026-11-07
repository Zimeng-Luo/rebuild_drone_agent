"""Lightweight run provenance and scope, without model quality claims or input hashing."""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from src.datatype import Chunk


PREPROCESS_EVIDENCE_NAME = "preprocess_evidence.json"
MIN_SELECTED_FRAME_COUNT = 2


def build_preprocess_evidence(
    *,
    input_path: str | Path,
    config: dict[str, Any],
    scanned_frame_count: int,
    selected_frame_ids: Iterable[int],
    chunks: list[Chunk] | None = None,
) -> dict[str, Any]:
    """Count sampled scan entries and unique chunk coverage separately from overlap."""
    selected_ids = [int(frame_id) for frame_id in selected_frame_ids]
    selected_count = len(selected_ids)
    unique_selected_count = len(set(selected_ids))
    chunk_list = chunks if chunks is not None else []
    chunk_frame_ids = [[int(frame.id) for frame in chunk.frames] for chunk in chunk_list]
    chunk_instances = sum(len(frame_ids) for frame_ids in chunk_frame_ids)
    unique_chunk_count = len({frame_id for frame_ids in chunk_frame_ids for frame_id in frame_ids})
    blocked = unique_selected_count < MIN_SELECTED_FRAME_COUNT
    return {
        "schema_version": 1,
        "status": "blocked" if blocked else ("chunked" if chunks is not None else "selected"),
        "blocked_reason": "insufficient_selected_frames" if blocked else None,
        "minimum_selected_frame_count": MIN_SELECTED_FRAME_COUNT,
        "input_path": str(input_path),
        "config": config,
        "sampled_frame_count": int(scanned_frame_count),
        "scanned_frame_count": int(scanned_frame_count),
        "frame_count_basis": "frames emitted by the input sampler and scanned; not original video total frames",
        "selected_frame_count": selected_count,
        "unique_selected_frame_count": unique_selected_count,
        "selected_frame_ids": selected_ids,
        "selection_retention_ratio": selected_count / scanned_frame_count if scanned_frame_count else 0.0,
        "chunk_count": len(chunk_list),
        "chunk_ids": [int(chunk.id) for chunk in chunk_list],
        "per_chunk_frame_counts": [len(frame_ids) for frame_ids in chunk_frame_ids],
        "chunk_frame_instance_count": chunk_instances,
        "unique_chunk_frame_count": unique_chunk_count,
        "overlap_duplicate_frame_instance_count": chunk_instances - unique_chunk_count,
    }


def write_preprocess_evidence(task_dir: str | Path, evidence: dict[str, Any]) -> Path:
    """Keep selection diagnostics even if downstream processing is blocked or fails."""
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / PREPROCESS_EVIDENCE_NAME
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=task_dir,
            prefix=f".{PREPROCESS_EVIDENCE_NAME}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(evidence, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def build_completed_run_evidence(
    task_dir: str | Path, *, chunk_count: int, loop_requested: bool
) -> dict[str, Any]:
    """Call only after rebuild and alignment return; execution does not prove quality."""
    evidence_path = Path(task_dir) / PREPROCESS_EVIDENCE_NAME
    preprocess = json.loads(evidence_path.read_text(encoding="utf-8"))
    alignment_path = Path(task_dir) / "alignment_evidence.json"
    return {
        "alignment_evidence_file": str(alignment_path) if alignment_path.is_file() else None,
        "alignment": json.loads(alignment_path.read_text(encoding="utf-8")) if alignment_path.is_file() else None,
        "preprocess_evidence_path": str(evidence_path),
        "preprocess": preprocess,
        "processing_scope": "single_chunk" if chunk_count == 1 else "cross_chunk",
        "single_chunk": chunk_count == 1,
        "cross_chunk_processing_executed": chunk_count > 1,
        "execution_basis": "rebuild_chunks and align_and_merge returned successfully for the recorded chunk count",
        "loop_closure": {
            "requested": bool(loop_requested),
            "accepted_constraint_count": None,
            "effectiveness_verified": False,
        },
        "quality_assessment": "not_performed",
    }


@contextmanager
def record_pipeline_failure(task_dir: str | Path, stage: str):
    """Add downstream failure diagnostics while preserving the original exception."""
    try:
        yield
    except Exception as error:
        try:
            path = Path(task_dir) / PREPROCESS_EVIDENCE_NAME
            evidence = json.loads(path.read_text(encoding="utf-8"))
            evidence["pipeline_status"] = "failed"
            evidence["pipeline_failure"] = {
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error)[:2000],
            }
            write_preprocess_evidence(task_dir, evidence)
        except Exception:
            logging.getLogger(__name__).exception("Could not persist pipeline failure evidence")
        raise


def validate_multiview_chunks(task_dir: str | Path, chunks: list[Chunk]) -> None:
    """Guard the rebuild boundary even when preprocessing is replaced or bypassed."""
    unique_count = len({int(frame.id) for chunk in chunks for frame in chunk.frames})
    if unique_count >= MIN_SELECTED_FRAME_COUNT:
        return
    path = Path(task_dir) / PREPROCESS_EVIDENCE_NAME
    evidence = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema_version": 1}
    evidence.update({
        "status": "blocked",
        "blocked_reason": "insufficient_unique_chunk_frames",
        "minimum_selected_frame_count": MIN_SELECTED_FRAME_COUNT,
        "unique_chunk_frame_count": unique_count,
        "pipeline_status": "blocked",
    })
    write_preprocess_evidence(task_dir, evidence)
    raise ValueError(
        "multi-view reconstruction requires at least 2 distinct frame IDs across chunks; "
        f"got {unique_count}. Diagnostics: {path}"
    )
