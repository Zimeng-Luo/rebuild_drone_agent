from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    ALIGN_CONF_THRESHOLD,
    ALIGN_GLB_CONF_THRES,
    ALIGN_IRLS_DELTA,
    ALIGN_IRLS_MAX_ITERS,
    ALIGN_LOOP_ENABLE,
    ALIGN_LOOP_MAX_PAIRS,
    ALIGN_LOOP_MIN_CHUNK_GAP,
    ALIGN_LOOP_MIN_MATCHES,
    ALIGN_LOOP_MIN_SCORE,
    ALIGN_LOOP_MODEL_NAME_OR_PATH,
    ALIGN_LOOP_OPT_MAX_ITERS,
    ALIGN_LOOP_TOP_K,
    ALIGN_LOOP_WINDOW_RADIUS,
    ALIGN_MIN_POINTS,
    ALIGN_PLY_CONF_THRES,
    CHUNK_OVERLAP_SIZE,
    CHUNK_SIZE,
    EXPORT_GLB_ENABLE,
    EXPORT_PLY_ENABLE,
    IMAGE_SAMPLE_INTERVAL,
    IMAGE_SEGMENT_SIZE,
    KEY_FRAME_DELETE_RATIO,
    READ_TYPE,
    READ_TYPE_CHOICES,
    READ_TYPE_IMAGE,
    READ_TYPE_VIDEO,
    RESULT_DIR,
    REMOVE_MASK_CLASS_IDS,
    REMOVE_MASK_DEVICE,
    REMOVE_MASK_ENABLE,
    REMOVE_MASK_MEAN,
    REMOVE_MASK_MODEL_PATH,
    REMOVE_MASK_STD,
    POINTCLOUD_DOWNSAMPLE_ENABLE,
    POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO,
    POINTCLOUD_DOWNSAMPLE_SUFFIX,
    POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE,
    VIDEO_EXTRACT_FPS,
    VIDEO_SEGMENT_SECONDS,
)
from pipeline import (
    run_align_and_merge,
    run_data_preprocess_with_task_dir,
    run_rebuild_chunks,
)
from src.logging_utils import configure_logging
from src.task_summary import (
    ResourceStepMonitor,
    collect_output_sizes,
    delete_numbered_intermediate_files,
    write_task_summary_log,
)


def run_pipeline(
    input_path: str | Path,
    result_dir: str | Path = RESULT_DIR,
    read_type: str = READ_TYPE,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
    remove_mask_enable: bool = REMOVE_MASK_ENABLE,
    remove_mask_model_path: str | Path = REMOVE_MASK_MODEL_PATH,
    remove_mask_device: str | None = REMOVE_MASK_DEVICE,
    remove_mask_class_ids: tuple[int, ...] = REMOVE_MASK_CLASS_IDS,
    export_glb_enable: bool = EXPORT_GLB_ENABLE,
    export_ply_enable: bool = EXPORT_PLY_ENABLE,
    align_glb_conf_thres: float = ALIGN_GLB_CONF_THRES,
    align_ply_conf_thres: float = ALIGN_PLY_CONF_THRES,
    pointcloud_downsample_enable: bool = POINTCLOUD_DOWNSAMPLE_ENABLE,
    pointcloud_downsample_voxel_size: float = POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE,
    pointcloud_downsample_min_keep_ratio: float = POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO,
) -> Path:
    """Run preprocess, rebuild, and align workflows for one input."""
    input_path = Path(input_path)
    result_dir = Path(result_dir)

    task_created_at = ""
    step_summaries = []
    with ResourceStepMonitor("total") as total_monitor:
        with ResourceStepMonitor("data_preprocess") as step_monitor:
            chunks, task_dir = run_data_preprocess_with_task_dir(
                input_path,
                result_dir=result_dir,
                read_type=read_type,
                image_segment_size=image_segment_size,
                image_sample_interval=image_sample_interval,
                remove_mask_enable=remove_mask_enable,
                remove_mask_model_path=remove_mask_model_path,
                remove_mask_device=remove_mask_device,
                remove_mask_class_ids=remove_mask_class_ids,
                summary_steps=step_summaries,
            )
        task_created_at = step_monitor.started_at
        step_summaries.append(step_monitor.summary())

        if not chunks:
            raise ValueError(f"no chunks generated from input: {input_path}")

        with ResourceStepMonitor("rebuild_chunks") as step_monitor:
            run_rebuild_chunks(
                [task_dir / f"{chunk.id}.pkl" for chunk in chunks],
                summary_steps=step_summaries,
            )
        step_summaries.append(step_monitor.summary())

        with ResourceStepMonitor("align_and_merge") as step_monitor:
            glb_path = run_align_and_merge(
                storage_dir=task_dir,
                chunk_count=len(chunks),
                export_glb_enable=export_glb_enable,
                export_ply_enable=export_ply_enable,
                conf_thres=align_glb_conf_thres,
                ply_conf_thres=align_ply_conf_thres,
                pointcloud_downsample_enable=pointcloud_downsample_enable,
                pointcloud_downsample_voxel_size=pointcloud_downsample_voxel_size,
                pointcloud_downsample_min_keep_ratio=pointcloud_downsample_min_keep_ratio,
                summary_steps=step_summaries,
            )
        step_summaries.append(step_monitor.summary())

    deleted_intermediate_files = delete_numbered_intermediate_files(task_dir)
    write_task_summary_log(
        task_dir,
        created_at=task_created_at,
        parameters=_build_summary_parameters(
            input_path=input_path,
            result_dir=result_dir,
            read_type=read_type,
            image_segment_size=image_segment_size,
            image_sample_interval=image_sample_interval,
            chunk_count=len(chunks),
            remove_mask_enable=remove_mask_enable,
            remove_mask_model_path=remove_mask_model_path,
            remove_mask_device=remove_mask_device,
            remove_mask_class_ids=remove_mask_class_ids,
            export_glb_enable=export_glb_enable,
            export_ply_enable=export_ply_enable,
            align_glb_conf_thres=align_glb_conf_thres,
            align_ply_conf_thres=align_ply_conf_thres,
            pointcloud_downsample_enable=pointcloud_downsample_enable,
            pointcloud_downsample_voxel_size=pointcloud_downsample_voxel_size,
            pointcloud_downsample_min_keep_ratio=pointcloud_downsample_min_keep_ratio,
        ),
        steps=step_summaries,
        total=total_monitor.summary(),
        outputs=collect_output_sizes(task_dir),
        deleted_intermediate_files=deleted_intermediate_files,
    )
    return glb_path


def _build_summary_parameters(
    *,
    input_path: Path,
    result_dir: Path,
    read_type: str,
    image_segment_size: int,
    image_sample_interval: int,
    chunk_count: int,
    remove_mask_enable: bool,
    remove_mask_model_path: str | Path,
    remove_mask_device: str | None,
    remove_mask_class_ids: tuple[int, ...],
    export_glb_enable: bool,
    export_ply_enable: bool,
    align_glb_conf_thres: float,
    align_ply_conf_thres: float,
    pointcloud_downsample_enable: bool,
    pointcloud_downsample_voxel_size: float,
    pointcloud_downsample_min_keep_ratio: float,
) -> dict[str, object]:
    return {
        "input_path": str(input_path),
        "read_type": read_type,
        "read_type_video": READ_TYPE_VIDEO,
        "read_type_image": READ_TYPE_IMAGE,
        "read_type_choices": READ_TYPE_CHOICES,
        "image_segment_size": image_segment_size,
        "image_sample_interval": image_sample_interval,
        "result_dir": str(result_dir),
        "video_extract_fps": VIDEO_EXTRACT_FPS,
        "video_segment_seconds": VIDEO_SEGMENT_SECONDS,
        "key_frame_delete_ratio": KEY_FRAME_DELETE_RATIO,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap_size": CHUNK_OVERLAP_SIZE,
        "chunk_count": chunk_count,
        "remove_mask_enable": remove_mask_enable,
        "remove_mask_model_path": str(remove_mask_model_path),
        "remove_mask_device": remove_mask_device,
        "remove_mask_class_ids": remove_mask_class_ids,
        "remove_mask_mean": REMOVE_MASK_MEAN,
        "remove_mask_std": REMOVE_MASK_STD,
        "rebuild_model_name_or_path": "facebook/VGGT-1B",
        "rebuild_device": None,
        "align_output_npz_name": "aligned.npz",
        "align_output_glb_name": "aligned.glb",
        "align_output_ply_name": "aligned.ply",
        "export_glb_enable": export_glb_enable,
        "export_ply_enable": export_ply_enable,
        "align_glb_conf_thres": align_glb_conf_thres,
        "align_ply_conf_thres": align_ply_conf_thres,
        "pointcloud_downsample_enable": pointcloud_downsample_enable,
        "pointcloud_downsample_voxel_size": pointcloud_downsample_voxel_size,
        "pointcloud_downsample_min_keep_ratio": pointcloud_downsample_min_keep_ratio,
        "pointcloud_downsample_suffix": POINTCLOUD_DOWNSAMPLE_SUFFIX,
        "align_filter_by_frames": "all",
        "align_mask_black_bg": False,
        "align_mask_white_bg": False,
        "align_show_cam": False,
        "align_mask_sky": False,
        "align_prediction_mode": "Depthmap and Camera Branch",
        "align_conf_threshold": ALIGN_CONF_THRESHOLD,
        "align_irls_delta": ALIGN_IRLS_DELTA,
        "align_irls_max_iters": ALIGN_IRLS_MAX_ITERS,
        "align_min_points": ALIGN_MIN_POINTS,
        "align_enable_loop": ALIGN_LOOP_ENABLE,
        "align_loop_min_chunk_gap": ALIGN_LOOP_MIN_CHUNK_GAP,
        "align_loop_max_pairs": ALIGN_LOOP_MAX_PAIRS,
        "align_loop_min_matches": ALIGN_LOOP_MIN_MATCHES,
        "align_loop_min_score": ALIGN_LOOP_MIN_SCORE,
        "align_loop_top_k": ALIGN_LOOP_TOP_K,
        "align_loop_opt_max_iters": ALIGN_LOOP_OPT_MAX_ITERS,
        "align_loop_window_radius": ALIGN_LOOP_WINDOW_RADIUS,
        "align_loop_model_name_or_path": ALIGN_LOOP_MODEL_NAME_OR_PATH,
        "align_device": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run input-to-GLB reconstruction pipeline.",
    )
    parser.add_argument("input_path", help="Input video file or image directory path.")
    parser.add_argument(
        "--read-type",
        choices=READ_TYPE_CHOICES,
        default=READ_TYPE,
        help=f"Input read type. Default: {READ_TYPE}",
    )
    parser.add_argument(
        "--result-dir",
        default=RESULT_DIR,
        help=f"Directory for intermediate and output files. Default: {RESULT_DIR}",
    )
    parser.add_argument(
        "--image-segment-size",
        type=int,
        default=IMAGE_SEGMENT_SIZE,
        help=f"Image frames read per segment. Default: {IMAGE_SEGMENT_SIZE}",
    )
    parser.add_argument(
        "--image-sample-interval",
        type=int,
        default=IMAGE_SAMPLE_INTERVAL,
        help=f"Read one image every N sorted images. Default: {IMAGE_SAMPLE_INTERVAL}",
    )
    parser.add_argument(
        "--no-remove-mask",
        action="store_true",
        help="Disable SegFormer semantic mask generation after chunk saving.",
    )
    parser.add_argument(
        "--remove-mask-model-path",
        default=REMOVE_MASK_MODEL_PATH,
        help=f"SegFormer model name or local path. Default: {REMOVE_MASK_MODEL_PATH}",
    )
    parser.add_argument(
        "--remove-mask-device",
        default=REMOVE_MASK_DEVICE,
        help="Device for SegFormer mask inference, such as cuda or cpu.",
    )
    parser.add_argument(
        "--remove-mask-class-ids",
        type=int,
        nargs="+",
        default=REMOVE_MASK_CLASS_IDS,
        help=f"Sky class ids in SegFormer output. Default: {REMOVE_MASK_CLASS_IDS}",
    )
    parser.add_argument(
        "--align-glb-conf-thres",
        type=float,
        default=ALIGN_GLB_CONF_THRES,
        help=(
            "Final GLB confidence percentile threshold. "
            f"Default: {ALIGN_GLB_CONF_THRES}"
        ),
    )
    parser.add_argument(
        "--no-export-glb",
        action="store_true",
        default=not EXPORT_GLB_ENABLE,
        help=f"Disable final GLB export. Default enabled: {EXPORT_GLB_ENABLE}",
    )
    parser.add_argument(
        "--align-ply-conf-thres",
        type=float,
        default=ALIGN_PLY_CONF_THRES,
        help=(
            "Final PLY confidence percentile threshold. "
            f"Default: {ALIGN_PLY_CONF_THRES}"
        ),
    )
    parser.add_argument(
        "--no-export-ply",
        action="store_true",
        default=not EXPORT_PLY_ENABLE,
        help=f"Disable final PLY export. Default enabled: {EXPORT_PLY_ENABLE}",
    )
    parser.add_argument(
        "--no-pointcloud-downsample",
        action="store_true",
        help="Disable final downsampled GLB/PLY export.",
    )
    parser.add_argument(
        "--pointcloud-downsample-voxel-size",
        type=float,
        default=POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE,
        help=f"Voxel grid size. Default: {POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE}",
    )
    parser.add_argument(
        "--pointcloud-downsample-min-keep-ratio",
        type=float,
        default=POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO,
        help=(
            "Minimum point ratio to keep after voxel downsampling. "
            f"Default: {POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    glb_path = run_pipeline(
        args.input_path,
        result_dir=args.result_dir,
        read_type=args.read_type,
        image_segment_size=args.image_segment_size,
        image_sample_interval=args.image_sample_interval,
        remove_mask_enable=not args.no_remove_mask,
        remove_mask_model_path=args.remove_mask_model_path,
        remove_mask_device=args.remove_mask_device,
        remove_mask_class_ids=tuple(args.remove_mask_class_ids),
        export_glb_enable=not args.no_export_glb,
        export_ply_enable=not args.no_export_ply,
        align_glb_conf_thres=args.align_glb_conf_thres,
        align_ply_conf_thres=args.align_ply_conf_thres,
        pointcloud_downsample_enable=not args.no_pointcloud_downsample,
        pointcloud_downsample_voxel_size=args.pointcloud_downsample_voxel_size,
        pointcloud_downsample_min_keep_ratio=args.pointcloud_downsample_min_keep_ratio,
    )
    print(glb_path)


if __name__ == "__main__":
    main()
