"""Regression checks for configuration and frame/export correctness (no model inference)."""
from pathlib import Path

import pytest

import config
import main
import pipeline.align as alignment
from pipeline.data_preprocess import _create_task_id
from src.key_frame_select import FrameMetadata, _remove_near_similar_overlapped_frames


def metadata(frame_id: int, quality: float = 0.5) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id, image_path=Path(f"{frame_id}.png"), gps_location=None,
        sharpness_score=1, exposure_score=1, feature_count=0, pose=None,
        previous_difference=0, previous_keyframe_difference=0,
        previous_overlap_score=1, previous_keyframe_overlap_score=1,
        descriptor=None, quality_score=quality,
    )


def test_unrelated_predecessor_overlap_does_not_delete_distinct_frames():
    # All frames overlap their own predecessor, but are distant candidate pairs.
    frames = [metadata(0), metadata(10), metadata(20)]
    assert [f.frame_id for f in _remove_near_similar_overlapped_frames(frames)] == [0, 10, 20]


def test_near_duplicate_keeps_the_sharper_candidate():
    frames = [metadata(0, 0.1), metadata(1, 0.9), metadata(10, 0.5)]
    assert [f.frame_id for f in _remove_near_similar_overlapped_frames(frames)] == [1, 10]


def test_cli_defaults_match_python_configuration():
    args = main.parse_args(["example.mp4"])
    assert args.remove_mask_enable == config.REMOVE_MASK_ENABLE
    assert args.pointcloud_downsample_enable == config.POINTCLOUD_DOWNSAMPLE_ENABLE


@pytest.mark.parametrize("flags,mask,downsample", [
    (["--remove-mask", "--pointcloud-downsample"], True, True),
    (["--no-remove-mask", "--no-pointcloud-downsample"], False, False),
])
def test_cli_boolean_overrides(flags, mask, downsample):
    args = main.parse_args(["example.mp4", *flags])
    assert args.remove_mask_enable is mask
    assert args.pointcloud_downsample_enable is downsample


@pytest.mark.parametrize("percentile", [0.0, 20.0, 80.0])
def test_glb_and_ply_respect_requested_percentile(tmp_path, monkeypatch, percentile):
    (tmp_path / "0.npz").write_bytes(b"fixture")
    monkeypatch.setattr(alignment, "save_aligned_npz_files_streaming", lambda paths, out, **kw: out.write_bytes(b"merged"))
    used = {}
    def export(kind):
        def save(**kwargs):
            used[kind] = kwargs["conf_thres"]
            kwargs["output_path"].write_bytes(b"fixture")
        return save
    monkeypatch.setattr(alignment, "rebuild_npz_to_glb_streaming", export("glb"))
    monkeypatch.setattr(alignment, "rebuild_npz_to_ply_streaming", export("ply"))
    alignment.run_align_and_merge(tmp_path, 1, conf_thres=percentile, ply_conf_thres=percentile, enable_loop=False)
    assert used == {"glb": percentile, "ply": percentile}


def test_task_ids_do_not_collide_within_one_second():
    assert len({_create_task_id() for _ in range(100)}) == 100
