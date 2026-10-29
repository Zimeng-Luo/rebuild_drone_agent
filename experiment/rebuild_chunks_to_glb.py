from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ALIGN_GLB_CONF_THRES,
    CHUNK_OVERLAP_SIZE,
    CHUNK_SIZE,
    IMAGE_SAMPLE_INTERVAL,
    IMAGE_SEGMENT_SIZE,
    KEY_FRAME_DELETE_RATIO,
    READ_TYPE,
    READ_TYPE_CHOICES,
    READ_TYPE_IMAGE,
    READ_TYPE_VIDEO,
    REMOVE_MASK_CLASS_IDS,
    REMOVE_MASK_DEVICE,
    REMOVE_MASK_ENABLE,
    REMOVE_MASK_MEAN,
    REMOVE_MASK_MODEL_PATH,
    REMOVE_MASK_STD,
    RESULT_DIR,
    VIDEO_EXTRACT_FPS,
    VIDEO_SEGMENT_SECONDS,
)
from pipeline import (
    run_data_preprocess_with_task_dir,
    run_rebuild_chunks,
    run_rebuild_to_glb,
)
from src.logging_utils import configure_logging
from src.task_summary import (
    ResourceStepMonitor,
    collect_output_sizes,
    delete_numbered_intermediate_files,
    write_task_summary_log,
)


def run_chunks_to_glb_experiment(
    input_path: str | Path,
    result_dir: str | Path = RESULT_DIR,
    read_type: str = READ_TYPE,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
    remove_mask_enable: bool = REMOVE_MASK_ENABLE,
    remove_mask_model_path: str | Path = REMOVE_MASK_MODEL_PATH,
    remove_mask_device: str | None = REMOVE_MASK_DEVICE,
    remove_mask_class_ids: tuple[int, ...] = REMOVE_MASK_CLASS_IDS,
    chunk_glb_conf_thres: float = ALIGN_GLB_CONF_THRES,
) -> list[Path]:
    """Run preprocess and chunk rebuild, then export each chunk directly to GLB."""
    input_path = Path(input_path)
    result_dir = Path(result_dir)

    task_created_at = ""
    step_summaries = []
    chunk_glb_paths: list[Path] = []
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

        chunk_pkl_paths = [task_dir / f"{chunk.id}.pkl" for chunk in chunks]
        with ResourceStepMonitor("rebuild_chunks") as step_monitor:
            chunk_npz_paths = run_rebuild_chunks(
                chunk_pkl_paths,
                summary_steps=step_summaries,
            )
        step_summaries.append(step_monitor.summary())

        with ResourceStepMonitor("export_chunk_glbs") as step_monitor:
            for npz_path in chunk_npz_paths:
                chunk_glb_paths.append(
                    run_rebuild_to_glb(
                        npz_path=npz_path,
                        output_path=npz_path.with_suffix(".glb"),
                        conf_thres=chunk_glb_conf_thres,
                        filter_by_frames="all",
                        mask_black_bg=False,
                        mask_white_bg=False,
                        show_cam=False,
                        mask_sky=False,
                        prediction_mode="Depthmap and Camera Branch",
                    )
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
            chunk_glb_conf_thres=chunk_glb_conf_thres,
        ),
        steps=step_summaries,
        total=total_monitor.summary(),
        outputs=collect_output_sizes(task_dir),
        deleted_intermediate_files=deleted_intermediate_files,
    )
    return chunk_glb_paths


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
    chunk_glb_conf_thres: float,
) -> dict[str, object]:
    return {
        "experiment": "rebuild_chunks_to_glb",
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
        "chunk_output_glb_pattern": "{chunk_id}.glb",
        "chunk_glb_conf_thres": chunk_glb_conf_thres,
        "filter_by_frames": "all",
        "mask_black_bg": False,
        "mask_white_bg": False,
        "show_cam": False,
        "mask_sky": False,
        "prediction_mode": "Depthmap and Camera Branch",
        "align_and_merge": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chunk reconstruction experiment and export each chunk as its own GLB "
            "without inter-chunk alignment or merging."
        ),
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
        "--chunk-glb-conf-thres",
        type=float,
        default=ALIGN_GLB_CONF_THRES,
        help=f"Per-chunk GLB confidence percentile threshold. Default: {ALIGN_GLB_CONF_THRES}",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    glb_paths = run_chunks_to_glb_experiment(
        args.input_path,
        result_dir=args.result_dir,
        read_type=args.read_type,
        image_segment_size=args.image_segment_size,
        image_sample_interval=args.image_sample_interval,
        remove_mask_enable=not args.no_remove_mask,
        remove_mask_model_path=args.remove_mask_model_path,
        remove_mask_device=args.remove_mask_device,
        remove_mask_class_ids=tuple(args.remove_mask_class_ids),
        chunk_glb_conf_thres=args.chunk_glb_conf_thres,
    )
    for glb_path in glb_paths:
        print(glb_path)


if __name__ == "__main__":
    main()
