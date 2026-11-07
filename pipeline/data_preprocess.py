from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import pickle
from contextlib import contextmanager

import cv2

from config import (
    CHUNK_OVERLAP_SIZE,
    CHUNK_SIZE,
    KEY_FRAME_DELETE_RATIO,
    VIDEO_EXTRACT_FPS,
    VIDEO_SEGMENT_SECONDS,
    IMAGE_SAMPLE_INTERVAL,
    IMAGE_SEGMENT_SIZE,
    READ_TYPE,
    READ_TYPE_CHOICES,
    READ_TYPE_VIDEO,
    RESULT_DIR,
    REMOVE_MASK_CLASS_IDS,
    REMOVE_MASK_ENABLE,
    REMOVE_MASK_MODEL_PATH,
)
from src.datatype import BaseFrame, Chunk, OriginalFrames
from src.frame_chunk import split_original_frames_into_chunks
from src.image_load import iter_original_frame_segments_from_image
from src.image_preprocess import preprocess_original_frame
from src.key_frame_select import (
    FrameMetadata,
    build_frame_metadata,
    select_key_frame_metadata,
)
from src.logging_utils import configure_logging, configure_task_file_logging
from src.remove_mask import add_remove_masks_to_chunk_files, load_remove_mask_model
from src.run_evidence import build_preprocess_evidence, write_preprocess_evidence
from src.task_summary import ResourceStepMonitor, StepSummary
from src.video_load import iter_original_frame_segments_from_video


logger = logging.getLogger(__name__)

PREPROCESSED_FRAME_DIR_NAME = "preprocessed_frames"


def run_data_preprocess(
    input_path: str | Path,
    result_dir: str | Path = RESULT_DIR,
    read_type: str = READ_TYPE,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
    remove_mask_enable: bool = REMOVE_MASK_ENABLE,
    remove_mask_model_path: str | Path | None = None,
    remove_mask_device: str | None = None,
    remove_mask_class_ids: tuple[int, ...] | None = None,
    summary_steps: list[StepSummary] | None = None,
) -> list[Chunk]:
    """Run video loading, preprocessing, key-frame selection, chunking, and saving."""
    chunks, _ = run_data_preprocess_with_task_dir(
        input_path,
        result_dir,
        read_type=read_type,
        image_segment_size=image_segment_size,
        image_sample_interval=image_sample_interval,
        remove_mask_enable=remove_mask_enable,
        remove_mask_model_path=remove_mask_model_path,
        remove_mask_device=remove_mask_device,
        remove_mask_class_ids=remove_mask_class_ids,
        summary_steps=summary_steps,
    )
    return chunks


def run_data_preprocess_with_task_dir(
    input_path: str | Path,
    result_dir: str | Path = RESULT_DIR,
    read_type: str = READ_TYPE,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
    remove_mask_enable: bool = REMOVE_MASK_ENABLE,
    remove_mask_model_path: str | Path | None = None,
    remove_mask_device: str | None = None,
    remove_mask_class_ids: tuple[int, ...] | None = None,
    summary_steps: list[StepSummary] | None = None,
) -> tuple[list[Chunk], Path]:
    """Run preprocessing and return the generated chunks with their task directory."""
    configure_logging()
    task_id = _create_task_id()
    task_dir = Path(result_dir) / task_id
    log_path = configure_task_file_logging(task_dir)
    logger.info("任务日志文件: %s", log_path)
    preprocessed_frame_dir = task_dir / PREPROCESSED_FRAME_DIR_NAME
    preprocessed_frame_dir.mkdir(parents=True, exist_ok=True)
    frame_segments = iter(_iter_original_frame_segments(
        input_path,
        read_type,
        image_segment_size=image_segment_size,
        image_sample_interval=image_sample_interval,
    ))

    chunks: list[Chunk] = []
    remove_mask_model = None
    effective_remove_mask_model_path = (
        remove_mask_model_path if remove_mask_model_path is not None else REMOVE_MASK_MODEL_PATH
    )
    effective_remove_mask_class_ids = (
        remove_mask_class_ids if remove_mask_class_ids is not None else REMOVE_MASK_CLASS_IDS
    )
    metadata_list: list[FrameMetadata] = []
    previous_image = None
    previous_keyframe_image = None
    segment_index = 0

    logger.info("开始第一遍流式扫描: read_type=%s", read_type)
    while True:
        read_step_name = (
            "read_video_segment"
            if read_type == READ_TYPE_VIDEO
            else "read_image_segment"
        )
        logger.info(
            "开始读取输入批次: segment_index=%d, read_type=%s",
            segment_index,
            read_type,
        )
        with _summary_step(summary_steps, f"{read_step_name}_{segment_index}"):
            try:
                original_frames = next(frame_segments)
            except StopIteration:
                logger.info("输入批次读取结束: next_segment_index=%d", segment_index)
                break

        logger.info(
            "输入批次读取完成: segment_index=%d, frame_count=%d, frame_id_range=%s",
            segment_index,
            len(original_frames.frames),
            _frame_id_range(original_frames.frames),
        )
        with _summary_step(summary_steps, f"stream_scan_segment_{segment_index}"):
            for frame in original_frames.frames:
                preprocessed_frame = preprocess_original_frame(frame)
                image_path = _save_preprocessed_frame(
                    preprocessed_frame,
                    preprocessed_frame_dir,
                )
                metadata = build_frame_metadata(
                    preprocessed_frame,
                    image_path,
                    previous_image=previous_image,
                    previous_keyframe_image=previous_keyframe_image,
                )
                metadata_list.append(metadata)
                previous_image = preprocessed_frame.image
                if previous_keyframe_image is None or metadata.previous_keyframe_difference > 8.0:
                    previous_keyframe_image = preprocessed_frame.image
                del preprocessed_frame

        logger.info(
            "流式扫描批次完成: segment_index=%d, batch_frames=%d, total_metadata=%d",
            segment_index,
            len(original_frames.frames),
            len(metadata_list),
        )
        del original_frames
        segment_index += 1

    logger.info(
        "第一遍流式扫描完成: total_frames=%d, preprocessed_dir=%s",
        len(metadata_list),
        preprocessed_frame_dir,
    )

    logger.info("开始第二遍候选筛选和第三遍全局选择: total_frames=%d", len(metadata_list))
    with _summary_step(summary_steps, "key_frame_select_global"):
        selected_metadata = select_key_frame_metadata(metadata_list)
    logger.info(
        "全局关键帧选择完成: total_frames=%d, selected_frames=%d, frame_id_range=%s",
        len(metadata_list),
        len(selected_metadata),
        _metadata_frame_id_range(selected_metadata),
    )

    evidence = build_preprocess_evidence(
        input_path=input_path,
        config={
            "read_type": read_type,
            "image_segment_size": image_segment_size,
            "image_sample_interval": image_sample_interval,
            "video_extract_fps": VIDEO_EXTRACT_FPS,
            "video_segment_seconds": VIDEO_SEGMENT_SECONDS,
            "key_frame_delete_ratio": KEY_FRAME_DELETE_RATIO,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap_size": CHUNK_OVERLAP_SIZE,
            "remove_mask_enable": remove_mask_enable,
        },
        scanned_frame_count=len(metadata_list),
        selected_frame_ids=(metadata.frame_id for metadata in selected_metadata),
    )
    evidence_path = write_preprocess_evidence(task_dir, evidence)
    if evidence["status"] == "blocked":
        raise ValueError(
            "multi-view reconstruction requires at least 2 distinct selected frames; "
            f"got {evidence['unique_selected_frame_count']} from "
            f"{len(metadata_list)} sampled/scanned frames. "
            f"Diagnostics: {evidence_path}"
        )

    logger.info("开始分块前重新加载预处理图像: selected_frames=%d", len(selected_metadata))
    with _summary_step(summary_steps, "reload_selected_preprocessed_frames"):
        selected_frames = _load_selected_preprocessed_frames(selected_metadata)
    logger.info(
        "预处理图像重新加载完成: selected_frames=%d, frame_id_range=%s",
        len(selected_frames.frames),
        _frame_id_range(selected_frames.frames),
    )

    logger.info("开始分块: selected_frames=%d", len(selected_frames.frames))
    with _summary_step(summary_steps, "chunk_split_global"):
        segment_chunks = split_original_frames_into_chunks(selected_frames)
    logger.info(
        "分块完成: chunk_count=%d, frame_counts=%s",
        len(segment_chunks),
        [len(chunk.frames) for chunk in segment_chunks],
    )

    chunks_to_save: list[Chunk] = []
    for chunk in segment_chunks:
        chunks_to_save.append(
            Chunk(id=len(chunks_to_save), frames=chunk.frames)
        )
    logger.info("开始保存分块: chunk_ids=%s", [chunk.id for chunk in chunks_to_save])
    with _summary_step(summary_steps, "chunk_save_global"):
        task_dir = save_chunks(chunks_to_save, task_id, result_dir)
    logger.info(
        "保存分块完成: task_dir=%s, chunk_ids=%s",
        task_dir,
        [chunk.id for chunk in chunks_to_save],
    )

    evidence = build_preprocess_evidence(
        input_path=input_path,
        config=evidence["config"],
        scanned_frame_count=len(metadata_list),
        selected_frame_ids=(metadata.frame_id for metadata in selected_metadata),
        chunks=chunks_to_save,
    )
    write_preprocess_evidence(task_dir, evidence)

    if remove_mask_enable:
        logger.info(
            "开始加载去除遮罩模型: model_path=%s, device=%s",
            effective_remove_mask_model_path,
            remove_mask_device,
        )
        with _summary_step(summary_steps, "remove_mask_model_load"):
            remove_mask_model = load_remove_mask_model(
                effective_remove_mask_model_path,
                remove_mask_device,
            )
        logger.info("去除遮罩模型加载完成")
        chunk_paths = [task_dir / f"{chunk.id}.pkl" for chunk in chunks_to_save]
        logger.info("开始处理遮罩: chunk_paths=%s", chunk_paths)
        with _summary_step(summary_steps, "remove_mask_global"):
            masked_chunks = add_remove_masks_to_chunk_files(
                chunk_paths,
                model=remove_mask_model,
                device=remove_mask_device,
                remove_class_ids=effective_remove_mask_class_ids,
            )
        chunks_to_save = masked_chunks
        logger.info("处理遮罩完成: chunk_ids=%s", [chunk.id for chunk in chunks_to_save])
    else:
        logger.info("跳过去除遮罩")

    chunks.extend(chunks_to_save)
    logger.info(
        "全部输入片段分块完成: task_id=%s, task_dir=%s, chunk_count=%d",
        task_id,
        task_dir,
        len(chunks),
    )
    return chunks, task_dir


def _iter_original_frame_segments(
    input_path: str | Path,
    read_type: str,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
):
    if read_type not in READ_TYPE_CHOICES:
        raise ValueError(
            f"read_type must be one of {READ_TYPE_CHOICES}, got {read_type!r}"
        )

    if read_type == READ_TYPE_VIDEO:
        return iter_original_frame_segments_from_video(input_path)

    return iter_original_frame_segments_from_image(
        input_path,
        segment_size=image_segment_size,
        sample_interval=image_sample_interval,
    )


@contextmanager
def _summary_step(summary_steps: list[StepSummary] | None, name: str):
    if summary_steps is None:
        yield
        return

    with ResourceStepMonitor(name) as monitor:
        yield
    summary_steps.append(monitor.summary())


def _frame_id_range(frames) -> str:
    if not frames:
        return "empty"

    return f"{frames[0].id}-{frames[-1].id}"


def _metadata_frame_id_range(metadata_list: list[FrameMetadata]) -> str:
    if not metadata_list:
        return "empty"

    return f"{metadata_list[0].frame_id}-{metadata_list[-1].frame_id}"


def _save_preprocessed_frame(frame: BaseFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{frame.id:08d}.png"
    image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), image):
        raise ValueError(f"Could not write preprocessed frame: {image_path}")
    return image_path


def _load_selected_preprocessed_frames(
    selected_metadata: list[FrameMetadata],
) -> OriginalFrames:
    frames = OriginalFrames()
    for metadata in selected_metadata:
        image = cv2.imread(str(metadata.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read preprocessed frame: {metadata.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.add_frame(
            BaseFrame(
                id=metadata.frame_id,
                image=image,
                gps_location=metadata.gps_location,
            )
        )
    return frames


def save_chunks(
    chunks: list[Chunk],
    task_id: str,
    result_dir: str | Path = RESULT_DIR,
) -> Path:
    """Save Chunk objects under result/<task_id>/<chunk_id>.pkl."""
    task_dir = Path(result_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    if not chunks:
        return task_dir

    for chunk in chunks:
        chunk_path = task_dir / f"{chunk.id}.pkl"
        with chunk_path.open("wb") as file:
            pickle.dump(chunk, file)

    logger.info(
        "分块完成并保存: task_id=%s, task_dir=%s, chunk_count=%d, chunk_ids=%s",
        task_id,
        task_dir,
        len(chunks),
        [chunk.id for chunk in chunks],
    )
    return task_dir


def _create_task_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
