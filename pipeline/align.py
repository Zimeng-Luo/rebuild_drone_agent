from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path

import torch

from config import (
    EXPORT_GLB_ENABLE,
    EXPORT_PLY_ENABLE,
    ALIGN_GLB_CONF_THRES,
    ALIGN_LOOP_MODEL_NAME_OR_PATH,
    ALIGN_PLY_CONF_THRES,
    POINTCLOUD_DOWNSAMPLE_ENABLE,
    POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO,
    POINTCLOUD_DOWNSAMPLE_SUFFIX,
    POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE,
)
from src.align import save_aligned_npz_files_streaming
from src.logging_utils import configure_logging, configure_task_file_logging
from src.rebuild import rebuild_npz_to_glb_streaming, rebuild_npz_to_ply_streaming
from src.task_summary import ResourceStepMonitor, StepSummary
from utils.pointcloud_downsample import downsample_point_cloud_file


logger = logging.getLogger(__name__)


def run_align_and_merge(
    storage_dir: str | Path,
    chunk_count: int,
    output_npz_name: str = "aligned.npz",
    output_glb_name: str = "aligned.glb",
    output_ply_name: str = "aligned.ply",
    export_glb_enable: bool = EXPORT_GLB_ENABLE,
    export_ply_enable: bool = EXPORT_PLY_ENABLE,
    conf_thres: float = ALIGN_GLB_CONF_THRES,
    ply_conf_thres: float = ALIGN_PLY_CONF_THRES,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = False,
    mask_sky: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
    pointcloud_downsample_enable: bool = POINTCLOUD_DOWNSAMPLE_ENABLE,
    pointcloud_downsample_voxel_size: float = POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE,
    pointcloud_downsample_min_keep_ratio: float = POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO,
    pointcloud_downsample_suffix: str = POINTCLOUD_DOWNSAMPLE_SUFFIX,
    loop_pairs: list[tuple[int, int]] | None = None,
    enable_loop: bool = True,
    loop_model: torch.nn.Module | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | torch.device | None = None,
    summary_steps: list[StepSummary] | None = None,
) -> Path:
    """Align numbered chunk npz files in one folder and save a merged GLB there."""
    configure_logging()
    if chunk_count <= 0:
        raise ValueError(f"chunk_count must be positive, got {chunk_count}")

    storage_dir = Path(storage_dir)
    configure_task_file_logging(storage_dir)
    chunk_paths = [storage_dir / f"{index}.npz" for index in range(chunk_count)]
    logger.info(
        "开始拼接分块: storage_dir=%s, chunk_count=%d, chunk_paths=%s",
        storage_dir,
        chunk_count,
        chunk_paths,
    )
    missing_paths = [path for path in chunk_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing chunk npz files: {missing_paths}")

    merged_npz_path = storage_dir / output_npz_name
    merged_glb_path = storage_dir / output_glb_name
    merged_ply_path = storage_dir / output_ply_name

    logger.info("开始计算并保存拼接结果: output_path=%s", merged_npz_path)
    alignment_diagnostics = {}
    with _summary_step(summary_steps, "align_compute_and_save"):
        save_aligned_npz_files_streaming(
            chunk_paths,
            merged_npz_path,
            loop_pairs=loop_pairs,
            enable_loop=enable_loop,
            loop_model=loop_model,
            loop_model_name_or_path=loop_model_name_or_path,
            device=str(device) if device is not None else None,
            diagnostics=alignment_diagnostics,
        )
    evidence_path = storage_dir / "alignment_evidence.json"
    evidence_tmp_path = evidence_path.with_name(f".{evidence_path.name}.tmp")
    evidence_tmp_path.write_text(
        json.dumps(alignment_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_tmp_path.replace(evidence_path)
    logger.info("对齐证据已保存: %s", evidence_path)
    logger.info("拼接结果保存完成: output_path=%s", merged_npz_path)
    if export_glb_enable:
        logger.info("开始导出最终 GLB: npz_path=%s, output_path=%s", merged_npz_path, merged_glb_path)
        with _summary_step(summary_steps, "align_export_glb"):
            rebuild_npz_to_glb_streaming(
                npz_path=merged_npz_path,
                output_path=merged_glb_path,
                conf_thres=conf_thres,
                filter_by_frames=filter_by_frames,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                show_cam=show_cam,
                mask_sky=mask_sky,
                prediction_mode=prediction_mode,
            )
    else:
        logger.info("跳过最终 GLB 导出: export_glb_enable=False")

    if export_ply_enable:
        logger.info("开始导出最终 PLY: npz_path=%s, output_path=%s", merged_npz_path, merged_ply_path)
        with _summary_step(summary_steps, "align_export_ply"):
            rebuild_npz_to_ply_streaming(
                npz_path=merged_npz_path,
                output_path=merged_ply_path,
                conf_thres=ply_conf_thres,
                filter_by_frames=filter_by_frames,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                mask_sky=mask_sky,
                prediction_mode=prediction_mode,
            )
    else:
        logger.info("跳过最终 PLY 导出: export_ply_enable=False")

    if pointcloud_downsample_enable:
        logger.info("开始导出最终下采样点云")
        if export_glb_enable:
            with _summary_step(summary_steps, "align_downsample_glb"):
                downsample_point_cloud_file(
                    merged_glb_path,
                    _downsample_output_path(merged_glb_path, pointcloud_downsample_suffix),
                    enabled=True,
                    voxel_size=pointcloud_downsample_voxel_size,
                    min_keep_ratio=pointcloud_downsample_min_keep_ratio,
                )
        if export_ply_enable:
            with _summary_step(summary_steps, "align_downsample_ply"):
                downsample_point_cloud_file(
                    merged_ply_path,
                    _downsample_output_path(merged_ply_path, pointcloud_downsample_suffix),
                    enabled=True,
                    voxel_size=pointcloud_downsample_voxel_size,
                    min_keep_ratio=pointcloud_downsample_min_keep_ratio,
                )

    output_path = _select_final_output_path(
        merged_npz_path,
        merged_glb_path,
        merged_ply_path,
        export_glb_enable,
        export_ply_enable,
    )
    logger.info("最终输出模型: %s", output_path)
    return output_path


def _downsample_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _select_final_output_path(
    npz_path: Path,
    glb_path: Path,
    ply_path: Path,
    export_glb_enable: bool,
    export_ply_enable: bool,
) -> Path:
    if export_glb_enable:
        return glb_path
    if export_ply_enable:
        return ply_path
    return npz_path


@contextmanager
def _summary_step(summary_steps: list[StepSummary] | None, name: str):
    if summary_steps is None:
        yield
        return

    with ResourceStepMonitor(name) as monitor:
        yield
    summary_steps.append(monitor.summary())
