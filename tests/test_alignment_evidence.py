"""CPU-only alignment evidence and optimizer fallback tests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import importlib

pipeline_align = importlib.import_module("pipeline.align")
alignment = importlib.import_module("src.align")


def sim3(tx=0.0):
    return (1.0, np.eye(3), np.array([tx, 0.0, 0.0]))


def test_single_chunk_does_not_load_or_probe_loops(monkeypatch, tmp_path):
    path = tmp_path / "0.npz"
    monkeypatch.setattr(alignment, "npz_load", lambda _: pytest.fail("unexpected load"))
    diagnostics = {}
    transforms = alignment.estimate_npz_file_transforms([path], diagnostics=diagnostics)
    assert len(transforms) == 1
    assert diagnostics["mode"] == "single_chunk"
    assert diagnostics["optimizer_attempted"] is False


def test_two_chunks_and_no_constraints_stay_sequential(monkeypatch, tmp_path):
    paths = [tmp_path / "0.npz", tmp_path / "1.npz"]
    monkeypatch.setattr(alignment, "npz_load", lambda path: SimpleNamespace(chunk_id=int(path.stem)))
    monkeypatch.setattr(alignment, "estimate_chunk_pair_sim3", lambda *_: sim3(1.0))
    monkeypatch.setattr(alignment, "estimate_loop_constraints_from_npz_files", lambda *a, **k: [])
    diagnostics = {}
    transforms = alignment.estimate_npz_file_transforms(paths, diagnostics=diagnostics)
    assert len(transforms) == 2
    assert diagnostics["loop_candidate_count"] is None
    assert diagnostics["loop_accepted_count"] == 0
    assert diagnostics["mode"] == "sequential_chunks"


def test_successful_optimizer_result_is_used_without_diagnostics(monkeypatch, tmp_path):
    paths = [tmp_path / f"{i}.npz" for i in range(3)]
    monkeypatch.setattr(alignment, "npz_load", lambda path: SimpleNamespace(chunk_id=int(path.stem)))
    monkeypatch.setattr(alignment, "estimate_chunk_pair_sim3", lambda *_: sim3(1.0))
    monkeypatch.setattr(alignment, "estimate_loop_constraints_from_npz_files", lambda *a, **k: [(2, 0, sim3())])
    optimized = [sim3(3.0), sim3(4.0)]
    monkeypatch.setattr(alignment, "optimize_sim3_pose_graph", lambda adjacent, constraints, diagnostics=None: optimized)
    transforms = alignment.estimate_npz_file_transforms(paths)
    assert len(transforms) == 3
    np.testing.assert_array_equal(transforms[1][2], [3.0, 0.0, 0.0])
    np.testing.assert_array_equal(transforms[2][2], [7.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "result,reason",
    [
        (SimpleNamespace(success=False, x=np.zeros(12), status=0), "optimizer_unsuccessful"),
        (SimpleNamespace(success=True, x=np.full(12, np.nan), status=1), "non_finite_optimizer_parameters"),
    ],
)
def test_optimizer_failure_or_nan_falls_back(monkeypatch, result, reason):
    original = [sim3(1.0), sim3(2.0)]
    snapshot = [(s, r.copy(), t.copy()) for s, r, t in original]
    monkeypatch.setattr(alignment, "least_squares", lambda *a, **k: result)
    diagnostics = {}
    returned = alignment.optimize_sim3_pose_graph(original, [(2, 0, sim3())], diagnostics)
    assert returned is original
    assert diagnostics["optimizer_applied"] is False
    assert diagnostics["fallback_reason"] == reason
    for actual, expected in zip(original, snapshot):
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])


def test_optimizer_exception_falls_back(monkeypatch):
    original = [sim3(1.0), sim3(2.0)]
    monkeypatch.setattr(alignment, "least_squares", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("solver error")))
    diagnostics = {}
    returned = alignment.optimize_sim3_pose_graph(original, [(2, 0, sim3())], diagnostics)
    assert returned is original
    assert diagnostics["fallback_reason"] == "optimizer_exception"


def test_pipeline_writes_task_local_alignment_evidence(monkeypatch, tmp_path):
    (tmp_path / "0.npz").write_bytes(b"fixture")

    def save(paths, output, diagnostics=None, **kwargs):
        output.write_bytes(b"merged")
        diagnostics.update({"mode": "single_chunk", "chunk_count": 1})

    monkeypatch.setattr(pipeline_align, "save_aligned_npz_files_streaming", save)
    result = pipeline_align.run_align_and_merge(
        tmp_path, 1, export_glb_enable=False, export_ply_enable=False,
        pointcloud_downsample_enable=False,
    )
    assert result == tmp_path / "aligned.npz"
    evidence = json.loads((tmp_path / "alignment_evidence.json").read_text(encoding="utf-8"))
    assert evidence["mode"] == "single_chunk"
    assert evidence["chunk_count"] == 1


@pytest.mark.parametrize("chunk_count", [1, 2])
def test_fewer_than_three_chunks_never_load_loop_model(monkeypatch, chunk_count):
    monkeypatch.setattr(alignment, "load_vggt_model", lambda **k: pytest.fail("loop model loaded"))
    diagnostics = {}
    assert alignment.estimate_loop_constraints_from_npz_files(
        [f"{i}.npz" for i in range(chunk_count)], diagnostics=diagnostics
    ) == []
    assert diagnostics["loop_candidate_count"] == 0
    assert diagnostics["fallback_reason"] == "fewer_than_three_chunks"


def test_optimizer_success_applies_finite_decoded_transforms(monkeypatch):
    original = [sim3(1.0), sim3(2.0)]

    def solve(residual, initial, **kwargs):
        assert np.all(np.isfinite(residual(initial)))
        return SimpleNamespace(success=True, x=initial, status=1, nfev=1)

    monkeypatch.setattr(alignment, "least_squares", solve)
    diagnostics = {}
    result = alignment.optimize_sim3_pose_graph(original, [(2, 0, sim3(3.0))], diagnostics)
    assert result is not original
    assert diagnostics["optimizer_success"] is True
    assert diagnostics["optimizer_applied"] is True
    for actual, expected in zip(result, original):
        np.testing.assert_allclose(actual[2], expected[2])


def test_nonfinite_decoded_transform_falls_back(monkeypatch):
    original = [sim3(1.0), sim3(2.0)]
    monkeypatch.setattr(alignment, "least_squares", lambda *a, **k: SimpleNamespace(success=True, x=np.zeros(14)))
    monkeypatch.setattr(alignment, "_params_to_absolute_transforms", lambda _: [(np.inf, np.eye(3), np.zeros(3))] * 2)
    diagnostics = {}
    assert alignment.optimize_sim3_pose_graph(original, [(2, 0, sim3())], diagnostics) is original
    assert diagnostics["fallback_reason"] == "invalid_decoded_transform"
    assert diagnostics["optimizer_applied"] is False


def test_candidates_are_nonadjacent(monkeypatch):
    monkeypatch.setattr(alignment, "_chunk_visual_score", lambda *a: 1.0)
    monkeypatch.setattr(alignment, "ALIGN_LOOP_MIN_CHUNK_GAP", 1)
    pairs = alignment._find_loop_candidate_pairs([object()] * 4)
    assert pairs
    assert all(source - target >= 2 for source, target in pairs)


def test_same_frame_is_not_visual_loop_evidence(monkeypatch):
    frame = SimpleNamespace(id=7, image=np.zeros((2, 2, 3), dtype=np.uint8))
    chunk = SimpleNamespace(frames=[frame])
    monkeypatch.setattr(alignment, "_match_frame_features", lambda *a: pytest.fail("same frame matched"))
    assert alignment._chunk_visual_score(chunk, chunk) == 0.0
    with pytest.raises(ValueError, match="no reliable visual loop frame pair"):
        alignment.estimate_loop_pair_sim3(chunk, chunk, model=object())
