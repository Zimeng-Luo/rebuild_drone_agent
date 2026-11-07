"""CPU-only provenance and multi-view guard regression tests; never infer models."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import main
import pipeline.data_preprocess as preprocessing
from src.datatype import BaseFrame, OriginalFrames
from src.run_evidence import PREPROCESS_EVIDENCE_NAME, build_preprocess_evidence


@pytest.fixture
def scanned_input(monkeypatch):
    def configure(count=4, selected_ids=(0, 1)):
        frames = [BaseFrame(id=i, image=np.full((8, 8, 3), i, dtype=np.uint8), gps_location=None) for i in range(count)]
        monkeypatch.setattr(preprocessing, "_iter_original_frame_segments", lambda *a, **kw: iter([OriginalFrames(frames=frames)]))
        monkeypatch.setattr(preprocessing, "preprocess_original_frame", lambda frame: frame)
        monkeypatch.setattr(preprocessing, "build_frame_metadata", lambda frame, image_path, **kw: SimpleNamespace(
            frame_id=frame.id, image_path=image_path, gps_location=None, previous_keyframe_difference=10.0,
        ))
        monkeypatch.setattr(preprocessing, "select_key_frame_metadata", lambda items: [items[i] for i in selected_ids])
        return frames
    return configure


def read_evidence(result_dir):
    paths = list(result_dir.glob(f"*/{PREPROCESS_EVIDENCE_NAME}"))
    assert len(paths) == 1
    return paths[0], json.loads(paths[0].read_text())


@pytest.mark.parametrize("selected_ids", [(), (0,), (0, 0)])
def test_less_than_two_distinct_frames_blocks_before_reload_and_models(tmp_path, monkeypatch, scanned_input, selected_ids):
    scanned_input(selected_ids=selected_ids)
    def forbidden(*args, **kwargs):
        pytest.fail("blocked selection reached reload or model processing")
    monkeypatch.setattr(preprocessing, "_load_selected_preprocessed_frames", forbidden)
    monkeypatch.setattr(preprocessing, "load_remove_mask_model", forbidden)
    monkeypatch.setattr(main, "run_rebuild_chunks", forbidden)
    monkeypatch.setattr(main, "run_align_and_merge", forbidden)
    with pytest.raises(ValueError, match="at least 2 distinct selected frames"):
        main.run_pipeline("sample.mp4", result_dir=tmp_path, remove_mask_enable=True)
    path, evidence = read_evidence(tmp_path)
    assert evidence["status"] == "blocked"
    assert evidence["blocked_reason"] == "insufficient_selected_frames"
    assert evidence["sampled_frame_count"] == evidence["scanned_frame_count"] == 4
    assert evidence["selected_frame_ids"] == list(selected_ids)
    assert evidence["selection_retention_ratio"] == len(selected_ids) / 4
    assert "not original video total frames" in evidence["frame_count_basis"]
    assert not list(path.parent.glob("*.pkl"))
    assert list((path.parent / "preprocessed_frames").glob("*.png"))


def test_empty_scan_evidence_has_defined_zero_retention(tmp_path, scanned_input):
    scanned_input(count=0, selected_ids=())
    with pytest.raises(ValueError, match="got 0 from 0 sampled/scanned frames"):
        preprocessing.run_data_preprocess_with_task_dir("sample.mp4", result_dir=tmp_path)
    _, evidence = read_evidence(tmp_path)
    assert evidence["selection_retention_ratio"] == 0.0


def test_two_selected_frames_reload_and_save_normally(tmp_path, scanned_input):
    scanned_input()
    chunks, task_dir = preprocessing.run_data_preprocess_with_task_dir(
        "sample.mp4", result_dir=tmp_path, remove_mask_enable=False,
    )
    evidence = json.loads((task_dir / PREPROCESS_EVIDENCE_NAME).read_text())
    assert evidence["status"] == "chunked"
    assert evidence["selected_frame_count"] == 2
    assert chunks[0].frame_ids == [0, 1]
    assert evidence["chunk_ids"] == [0]
    assert evidence["per_chunk_frame_counts"] == [2]
    assert (task_dir / "0.pkl").is_file()


def test_chunk_overlap_counts_instances_without_inflating_unique_frames(tmp_path, scanned_input):
    scanned_input(count=36, selected_ids=range(36))
    _, task_dir = preprocessing.run_data_preprocess_with_task_dir("sample.mp4", result_dir=tmp_path)
    evidence = json.loads((task_dir / PREPROCESS_EVIDENCE_NAME).read_text())
    assert evidence["chunk_count"] == 2
    assert evidence["per_chunk_frame_counts"] == [30, 21]
    assert evidence["chunk_frame_instance_count"] == 51
    assert evidence["unique_chunk_frame_count"] == evidence["selected_frame_count"] == 36
    assert evidence["overlap_duplicate_frame_instance_count"] == 15
    assert "per_chunk_frame_ids" not in evidence


@pytest.mark.parametrize("count", [2, 36])
def test_success_summary_retains_evidence_after_cleanup(tmp_path, monkeypatch, scanned_input, count):
    scanned_input(count=count, selected_ids=range(count))
    calls = []
    monkeypatch.setattr(main, "run_rebuild_chunks", lambda paths, **kw: calls.append(("rebuild", len(paths))))
    def align(storage_dir, chunk_count, **kwargs):
        calls.append(("align", chunk_count))
        path = storage_dir / "aligned.glb"
        path.write_bytes(b"CPU fixture, not a reconstruction")
        (storage_dir / "alignment_evidence.json").write_text(json.dumps({"test_fixture": True, "chunk_count": chunk_count}))
        return path
    monkeypatch.setattr(main, "run_align_and_merge", align)
    output = main.run_pipeline("sample.mp4", result_dir=tmp_path, remove_mask_enable=False)
    summary = json.loads((output.parent / "task_summary.log").read_text())
    evidence = summary["parameters"]["run_evidence"]
    multi = count > 30
    assert calls == [("rebuild", 2 if multi else 1), ("align", 2 if multi else 1)]
    assert evidence["single_chunk"] is not multi
    assert evidence["cross_chunk_processing_executed"] is multi
    assert evidence["processing_scope"] == ("cross_chunk" if multi else "single_chunk")
    assert evidence["quality_assessment"] == "not_performed"
    assert evidence["loop_closure"]["accepted_constraint_count"] is None
    assert evidence["loop_closure"]["effectiveness_verified"] is False
    assert Path(evidence["preprocess_evidence_path"]).is_file()
    assert evidence["preprocess"]["selected_frame_count"] == count
    assert Path(evidence["alignment_evidence_file"]).is_file()
    assert evidence["alignment"]["test_fixture"] is True
    assert not (output.parent / "preprocessed_frames").exists()
    assert not list(output.parent.glob("*.pkl"))


@pytest.mark.parametrize("stage", ["rebuild_chunks", "align_and_merge"])
def test_downstream_failure_keeps_diagnostics_and_original_exception(tmp_path, monkeypatch, scanned_input, stage):
    scanned_input()
    monkeypatch.setattr(main, "run_rebuild_chunks", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_align_and_merge", lambda *a, **kw: None)
    error = RuntimeError("CPU fixture downstream failure")
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(main, "run_" + stage, fail)
    with pytest.raises(RuntimeError) as caught:
        main.run_pipeline("sample.mp4", result_dir=tmp_path, remove_mask_enable=False)
    assert caught.value is error
    path, evidence = read_evidence(tmp_path)
    assert evidence["status"] == "chunked"
    assert evidence["pipeline_status"] == "failed"
    assert evidence["pipeline_failure"] == {"stage": stage, "error_type": "RuntimeError", "message": str(error)}
    assert (path.parent / "0.pkl").is_file()


def test_evidence_accepts_numpy_frame_ids_without_hashing_input(tmp_path):
    evidence = build_preprocess_evidence(input_path=tmp_path / "absent-large-video.mp4", config={},
        scanned_frame_count=4, selected_frame_ids=[np.int64(2), np.int64(5)])
    assert json.loads(json.dumps(evidence))["selected_frame_ids"] == [2, 5]
    assert "hash" not in evidence


@pytest.mark.parametrize("chunk_frame_ids", [[], [[7], [7]]])
def test_main_validates_global_unique_chunk_ids_even_if_preprocessing_is_bypassed(tmp_path, monkeypatch, chunk_frame_ids):
    from src.datatype import Chunk
    chunks = [Chunk(id=i, frames=[BaseFrame(id=frame_id, image=np.zeros((8, 8, 3), dtype=np.uint8), gps_location=None)
        for frame_id in ids]) for i, ids in enumerate(chunk_frame_ids)]
    task_dir = tmp_path / "bypassed"
    monkeypatch.setattr(main, "run_data_preprocess_with_task_dir", lambda *a, **kw: (chunks, task_dir))
    def forbidden(*a, **kw):
        pytest.fail("invalid chunk input reached model reconstruction")
    monkeypatch.setattr(main, "run_rebuild_chunks", forbidden)
    with pytest.raises(ValueError, match="at least 2 distinct frame IDs across chunks"):
        main.run_pipeline("sample.mp4", result_dir=tmp_path)
    evidence = json.loads((task_dir / PREPROCESS_EVIDENCE_NAME).read_text())
    assert evidence["status"] == "blocked"
    assert evidence["blocked_reason"] == "insufficient_unique_chunk_frames"
    assert evidence["unique_chunk_frame_count"] == (1 if chunk_frame_ids else 0)
