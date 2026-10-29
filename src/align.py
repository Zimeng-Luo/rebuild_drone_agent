from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from config import (
    ALIGN_CONF_THRESHOLD,
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
    CHUNK_OVERLAP_SIZE,
)
from src.datatype import BaseFrame, Chunk, ChunkToProcess, FrameInChunk
from src.rebuild import NPZ_LOAD_KEYS, load_vggt_model, npz_load, rebuild_chunk


logger = logging.getLogger(__name__)


Sim3 = tuple[float, NDArray[np.floating], NDArray[np.floating]]
LoopPair = tuple[int, int]
LoopConstraint = tuple[int, int, Sim3]
SKY_CLASS_ID = 2


def detect_loop_constraints(
    chunk_to_processes: list[ChunkToProcess],
    candidate_pairs: list[LoopPair] | None = None,
    model: Any | None = None,
    model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> list[LoopConstraint]:
    """Estimate non-adjacent chunk Sim3 constraints through joint loop-window rebuilds."""
    if len(chunk_to_processes) < 3:
        return []

    if candidate_pairs is None:
        candidate_pairs = _find_loop_candidate_pairs(chunk_to_processes)
    if not candidate_pairs:
        return []

    if model is None:
        model = load_vggt_model(model_name_or_path=model_name_or_path, device=device)

    constraints: list[LoopConstraint] = []
    for source_index, target_index in candidate_pairs:
        if source_index == target_index:
            continue
        if source_index < target_index:
            source_index, target_index = target_index, source_index
        if source_index >= len(chunk_to_processes) or target_index < 0:
            continue
        if source_index - target_index < ALIGN_LOOP_MIN_CHUNK_GAP:
            continue

        try:
            transform = estimate_loop_pair_sim3(
                chunk_to_processes[source_index],
                chunk_to_processes[target_index],
                model=model,
                device=device,
            )
        except ValueError as exc:
            logger.debug(
                "跳过回环约束: source_chunk_id=%s, target_chunk_id=%s, reason=%s",
                chunk_to_processes[source_index].chunk_id,
                chunk_to_processes[target_index].chunk_id,
                exc,
            )
            continue

        constraints.append((source_index, target_index, transform))
        logger.info(
            "回环约束估计完成: source_chunk_id=%s, target_chunk_id=%s, transform_matrix=%s",
            chunk_to_processes[source_index].chunk_id,
            chunk_to_processes[target_index].chunk_id,
            np.array2string(_sim3_to_matrix(transform), precision=6, suppress_small=False),
        )
        if len(constraints) >= ALIGN_LOOP_MAX_PAIRS:
            break

    return constraints


def estimate_loop_pair_sim3(
    source_chunk: ChunkToProcess,
    target_chunk: ChunkToProcess,
    model: Any,
    device: str | None = None,
) -> Sim3:
    """Estimate source_chunk -> target_chunk Sim3 from a jointly rebuilt loop window."""
    best: tuple[float, int, int] | None = None

    for source_index, source_frame in _loop_probe_frame_items(source_chunk):
        for target_index, target_frame in _loop_probe_frame_items(target_chunk):
            score, matches, _, _ = _match_frame_features(source_frame.image, target_frame.image)
            if len(matches) < ALIGN_LOOP_MIN_MATCHES or score < ALIGN_LOOP_MIN_SCORE:
                continue
            if best is None or score > best[0]:
                best = (score, source_index, target_index)

    if best is None:
        raise ValueError("no reliable visual loop frame pair")

    _, source_center_index, target_center_index = best
    source_window = _loop_window(source_chunk, source_center_index)
    target_window = _loop_window(target_chunk, target_center_index)
    joint_chunk = _joint_loop_chunk(source_window, target_window)
    predictions = rebuild_chunk(joint_chunk, model=model, device=device)
    loop_points = np.asarray(predictions["world_points_from_depth"])
    loop_conf = np.asarray(predictions["depth_conf"])

    source_loop = _prediction_window_to_chunk(
        predictions=predictions,
        start=0,
        end=len(source_window),
        chunk_id=source_chunk.chunk_id,
    )
    target_loop = _prediction_window_to_chunk(
        predictions=predictions,
        start=len(source_window),
        end=len(source_window) + len(target_window),
        chunk_id=target_chunk.chunk_id,
    )

    source_to_loop = estimate_chunk_sequence_pair_sim3(
        source_loop,
        _chunk_window_to_process(source_window, source_chunk.chunk_id),
    )
    target_to_loop = estimate_chunk_sequence_pair_sim3(
        target_loop,
        _chunk_window_to_process(target_window, target_chunk.chunk_id),
    )
    target_to_source = compose_sim3(invert_sim3(source_to_loop), target_to_loop)
    source_to_target = invert_sim3(target_to_source)

    logger.info(
        "回环窗口共同推理完成: source_chunk_id=%s, target_chunk_id=%s, "
        "source_window=%s, target_window=%s, loop_points_shape=%s, loop_conf_shape=%s",
        source_chunk.chunk_id,
        target_chunk.chunk_id,
        [frame.id for frame in source_window],
        [frame.id for frame in target_window],
        loop_points.shape,
        loop_conf.shape,
    )
    return source_to_target


def extract_overlap_points(
    chunk_a: ChunkToProcess,
    chunk_b: ChunkToProcess,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Extract VGGT-Long style tail/head overlap points from adjacent chunks."""
    overlap_count = min(CHUNK_OVERLAP_SIZE, len(chunk_a.frames), len(chunk_b.frames))
    if overlap_count <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return _extract_frame_pair_points(
        chunk_a.frames[-overlap_count:],
        chunk_b.frames[:overlap_count],
        target_chunk_id=chunk_a.chunk_id,
        source_chunk_id=chunk_b.chunk_id,
        log_context="VGGT-Long重叠窗口取点",
    )


def extract_sequence_pair_points(
    chunk_a: ChunkToProcess,
    chunk_b: ChunkToProcess,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Extract point pairs from same-length frame sequences."""
    pair_count = min(len(chunk_a.frames), len(chunk_b.frames))
    if pair_count <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return _extract_frame_pair_points(
        chunk_a.frames[:pair_count],
        chunk_b.frames[:pair_count],
        target_chunk_id=chunk_a.chunk_id,
        source_chunk_id=chunk_b.chunk_id,
        log_context="VGGT-Long逐帧窗口取点",
    )


def _extract_frame_pair_points(
    frames_a: list[FrameInChunk],
    frames_b: list[FrameInChunk],
    target_chunk_id: int,
    source_chunk_id: int,
    log_context: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    pair_count = min(len(frames_a), len(frames_b))
    frames_a = frames_a[:pair_count]
    frames_b = frames_b[:pair_count]
    if pair_count <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    points_a = np.stack([frame.world_points for frame in frames_a])
    conf_a = np.stack([frame.world_points_conf for frame in frames_a])
    points_b = np.stack([frame.world_points for frame in frames_b])
    conf_b = np.stack([frame.world_points_conf for frame in frames_b])

    conf_a = _normalize_conf_shape(conf_a, points_a)
    conf_b = _normalize_conf_shape(conf_b, points_b)
    conf_threshold = _vggt_long_conf_threshold(conf_a, conf_b)
    sky_mask_a = np.stack(
        [
            _frame_sky_mask_for_map(frame, points_a.shape[1:3])
            for frame in frames_a
        ]
    )
    sky_mask_b = np.stack(
        [
            _frame_sky_mask_for_map(frame, points_b.shape[1:3])
            for frame in frames_b
        ]
    )

    valid = (
        np.isfinite(points_a).all(axis=-1)
        & np.isfinite(points_b).all(axis=-1)
        & (conf_a > conf_threshold)
        & (conf_b > conf_threshold)
        & ~sky_mask_a
        & ~sky_mask_b
    )

    target = points_a[valid]
    source = points_b[valid]
    weights = np.sqrt(conf_a[valid] * conf_b[valid])
    logger.info(
        "%s: target_chunk_id=%s, source_chunk_id=%s, "
        "frame_pair_count=%d, conf_threshold=%.6g, matched_points=%d",
        log_context,
        target_chunk_id,
        source_chunk_id,
        pair_count,
        conf_threshold,
        len(weights),
    )
    return target, source, weights


def estimate_sim3(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> Sim3:
    """Estimate weighted Sim3 that maps source points to target points."""
    _validate_point_pairs(source, target, weights)

    normalized_weights = weights.astype(np.float64, copy=False)
    normalized_weights = normalized_weights / (normalized_weights.sum() + 1e-12)
    source = source.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)

    mu_src = np.sum(normalized_weights[:, None] * source, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target, axis=0)

    src_centered = source - mu_src
    tgt_centered = target - mu_tgt

    scale_src = np.sqrt(
        np.sum(normalized_weights * np.sum(src_centered**2, axis=1))
    )
    scale_tgt = np.sqrt(
        np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1))
    )
    if scale_src <= 1e-12:
        raise ValueError("source points are degenerate; cannot estimate Sim3")

    scale = float(scale_tgt / scale_src)
    weighted_src = scale * src_centered * np.sqrt(normalized_weights[:, None])
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights[:, None])

    h_matrix = weighted_src.T @ weighted_tgt
    u_matrix, _, vt_matrix = np.linalg.svd(h_matrix)
    rotation = vt_matrix.T @ u_matrix.T

    if np.linalg.det(rotation) < 0:
        vt_matrix[2, :] *= -1
        rotation = vt_matrix.T @ u_matrix.T

    translation = mu_tgt - scale * rotation @ mu_src
    return scale, rotation, translation


def _vggt_long_conf_threshold(
    conf_a: NDArray[np.floating],
    conf_b: NDArray[np.floating],
) -> float:
    """Match VGGT-Long's dynamic confidence threshold for point-map alignment."""
    finite_a = conf_a[np.isfinite(conf_a)]
    finite_b = conf_b[np.isfinite(conf_b)]
    if finite_a.size == 0 or finite_b.size == 0:
        return ALIGN_CONF_THRESHOLD
    return float(min(np.median(finite_a), np.median(finite_b)) * 0.1)


def robust_estimate_sim3(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> Sim3:
    """Estimate Sim3 with Huber IRLS for outlier robustness."""
    scale, rotation, translation = estimate_sim3(source, target, weights)

    for _ in range(ALIGN_IRLS_MAX_ITERS):
        transformed = apply_sim3_to_points(source, scale, rotation, translation)
        residuals = np.linalg.norm(target - transformed, axis=1)

        huber = np.ones_like(residuals)
        large = residuals > ALIGN_IRLS_DELTA
        huber[large] = ALIGN_IRLS_DELTA / residuals[large]

        scale, rotation, translation = estimate_sim3(
            source,
            target,
            weights * huber,
        )

    return scale, rotation, translation


def apply_sim3_to_points(
    points: NDArray[np.floating],
    scale: float,
    rotation: NDArray[np.floating],
    translation: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply a Sim3 transform to point arrays with last dimension 3."""
    return scale * np.einsum("ij,...j->...i", rotation, points) + translation


def compose_sim3(first: Sim3, second: Sim3) -> Sim3:
    """Compose two transforms as first(second(points))."""
    scale_first, rotation_first, translation_first = first
    scale_second, rotation_second, translation_second = second
    scale = scale_first * scale_second
    rotation = rotation_first @ rotation_second
    translation = scale_first * (rotation_first @ translation_second) + translation_first
    return scale, rotation, translation


def invert_sim3(transform: Sim3) -> Sim3:
    """Invert a Sim3 transform."""
    scale, rotation, translation = transform
    inv_scale = 1.0 / scale
    inv_rotation = rotation.T
    inv_translation = -inv_scale * (inv_rotation @ translation)
    return inv_scale, inv_rotation, inv_translation


def relative_sim3(absolute_source: Sim3, absolute_target: Sim3) -> Sim3:
    """Return transform source -> target from two chunk-to-world transforms."""
    return compose_sim3(invert_sim3(absolute_target), absolute_source)


def estimate_chunk_pair_sim3(
    chunk_a: ChunkToProcess,
    chunk_b: ChunkToProcess,
) -> Sim3:
    """Estimate chunk_b -> chunk_a Sim3 from overlapping frames."""
    target, source, weights = extract_overlap_points(chunk_a, chunk_b)
    if len(weights) < ALIGN_MIN_POINTS:
        raise ValueError(
            f"not enough valid overlap points: got {len(weights)}, "
            f"need at least {ALIGN_MIN_POINTS}"
        )

    return robust_estimate_sim3(source, target, weights)


def estimate_chunk_sequence_pair_sim3(
    chunk_a: ChunkToProcess,
    chunk_b: ChunkToProcess,
) -> Sim3:
    """Estimate chunk_b -> chunk_a Sim3 from same-length frame windows."""
    target, source, weights = extract_sequence_pair_points(chunk_a, chunk_b)
    if len(weights) < ALIGN_MIN_POINTS:
        raise ValueError(
            f"not enough valid paired-window points: got {len(weights)}, "
            f"need at least {ALIGN_MIN_POINTS}"
        )

    return robust_estimate_sim3(source, target, weights)


def accumulate_sim3_transforms(transforms: list[Sim3]) -> list[Sim3]:
    """Accumulate adjacent transforms so every transform maps a chunk to chunk0."""
    cumulative_transforms: list[Sim3] = []
    for scale_next, rotation_next, translation_next in transforms:
        if not cumulative_transforms:
            cumulative_transforms.append(
                (scale_next, rotation_next, translation_next)
            )
            continue

        scale_prev, rotation_prev, translation_prev = cumulative_transforms[-1]
        scale_new = scale_prev * scale_next
        rotation_new = rotation_prev @ rotation_next
        translation_new = (
            scale_prev * (rotation_prev @ translation_next) + translation_prev
        )
        cumulative_transforms.append((scale_new, rotation_new, translation_new))

    return cumulative_transforms


def align_chunk_to_processes(
    chunk_to_processes: list[ChunkToProcess],
    loop_pairs: list[LoopPair] | None = None,
    enable_loop: bool = ALIGN_LOOP_ENABLE,
    loop_model: Any | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> list[ChunkToProcess]:
    """Align all ChunkToProcess objects into the first chunk coordinate system."""
    if not chunk_to_processes:
        return []
    if len(chunk_to_processes) == 1:
        return [chunk_to_processes[0]]

    adjacent_transforms = []
    for index in range(len(chunk_to_processes) - 1):
        chunk_a = chunk_to_processes[index]
        chunk_b = chunk_to_processes[index + 1]
        transform = estimate_chunk_pair_sim3(chunk_a, chunk_b)
        adjacent_transforms.append(transform)
        logger.info(
            "每组分块拼接完成: source_chunk_id=%s, target_chunk_id=%s, transform_matrix=%s",
            chunk_b.chunk_id,
            chunk_a.chunk_id,
            np.array2string(_sim3_to_matrix(transform), precision=6, suppress_small=False),
        )
    if enable_loop:
        loop_constraints = detect_loop_constraints(
            chunk_to_processes,
            loop_pairs,
            model=loop_model,
            model_name_or_path=loop_model_name_or_path,
            device=device,
        )
        if loop_constraints:
            adjacent_transforms = optimize_sim3_pose_graph(
                adjacent_transforms,
                loop_constraints,
            )
            logger.info("回环全局优化完成: loop_constraint_count=%d", len(loop_constraints))
        else:
            logger.info("未找到可靠回环约束，使用顺序拼接结果")

    cumulative_transforms = accumulate_sim3_transforms(adjacent_transforms)

    aligned_chunks = [chunk_to_processes[0]]
    for chunk, (scale, rotation, translation) in zip(
        chunk_to_processes[1:],
        cumulative_transforms,
    ):
        aligned_chunk = ChunkToProcess(chunk_id=chunk.chunk_id)
        for frame in chunk.frames:
            aligned_world_points = apply_sim3_to_points(
                frame.world_points,
                scale,
                rotation,
                translation,
            ).astype(frame.world_points.dtype, copy=False)
            aligned_frame = FrameInChunk(
                id=frame.id,
                image=frame.image,
                gps_location=frame.gps_location,
                mask=frame.mask,
                chunk_id=frame.chunk_id,
                world_points=aligned_world_points,
                world_points_conf=frame.world_points_conf,
            )
            aligned_chunk.add_frame(aligned_frame)
        aligned_chunks.append(aligned_chunk)

    return aligned_chunks


def save_aligned_chunks_npz(
    chunk_to_processes: list[ChunkToProcess],
    output_path: str | Path,
    loop_pairs: list[LoopPair] | None = None,
    enable_loop: bool = ALIGN_LOOP_ENABLE,
    loop_model: Any | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> Path:
    """Align, concatenate, and save ChunkToProcess objects into one npz file."""
    aligned_chunks = align_chunk_to_processes(
        chunk_to_processes,
        loop_pairs=loop_pairs,
        enable_loop=enable_loop,
        loop_model=loop_model,
        loop_model_name_or_path=loop_model_name_or_path,
        device=device,
    )
    aligned_chunks = filter_aligned_chunks_by_confidence(aligned_chunks)
    predictions = _chunks_to_npz_predictions(aligned_chunks)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **predictions)
    return output_path


def filter_aligned_chunks_by_confidence(
    chunks: list[ChunkToProcess],
    keep_ratio: float = 0.5,
) -> list[ChunkToProcess]:
    """Keep the highest-confidence points after alignment while preserving npz shapes."""
    if not chunks:
        return []
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    finite_entries: list[tuple[int, int, NDArray[np.bool_], NDArray[np.floating]]] = []
    confidence_values = []
    for chunk_index, chunk in enumerate(chunks):
        for frame_index, frame in enumerate(chunk.frames):
            flat_confidence = np.asarray(frame.world_points_conf).reshape(-1)
            finite_mask = np.isfinite(flat_confidence)
            if not finite_mask.any():
                continue
            finite_entries.append((chunk_index, frame_index, finite_mask, flat_confidence))
            confidence_values.append(flat_confidence[finite_mask])

    if not confidence_values:
        return chunks

    finite_confidence = np.concatenate(confidence_values)
    keep_count = max(1, int(np.ceil(finite_confidence.size * keep_ratio)))
    keep_indexes = np.argpartition(finite_confidence, -keep_count)[-keep_count:]
    keep_flags = np.zeros(finite_confidence.size, dtype=bool)
    keep_flags[keep_indexes] = True

    filtered_chunks: list[ChunkToProcess] = []
    kept_points = 0
    total_points = 0
    entry_masks: dict[tuple[int, int], NDArray[np.bool_]] = {}
    offset = 0
    for chunk_index, frame_index, finite_mask, flat_confidence in finite_entries:
        frame_keep_mask = np.zeros(flat_confidence.shape, dtype=bool)
        finite_count = int(np.count_nonzero(finite_mask))
        frame_keep_mask[finite_mask] = keep_flags[offset : offset + finite_count]
        entry_masks[(chunk_index, frame_index)] = frame_keep_mask
        offset += finite_count

    for chunk in chunks:
        filtered_chunk = ChunkToProcess(chunk_id=chunk.chunk_id)
        for frame_index, frame in enumerate(chunk.frames):
            confidence = np.asarray(frame.world_points_conf)
            keep_mask = entry_masks.get(
                (len(filtered_chunks), frame_index),
                np.zeros(confidence.size, dtype=bool),
            ).reshape(confidence.shape)
            filtered_confidence = np.where(keep_mask, confidence, 0).astype(
                confidence.dtype,
                copy=False,
            )
            kept_points += int(np.count_nonzero(keep_mask))
            total_points += int(confidence.size)
            filtered_chunk.add_frame(
                FrameInChunk(
                    id=frame.id,
                    image=frame.image,
                    gps_location=frame.gps_location,
                    mask=frame.mask,
                    chunk_id=frame.chunk_id,
                    world_points=frame.world_points,
                    world_points_conf=filtered_confidence,
                )
            )
        filtered_chunks.append(filtered_chunk)

    logger.info(
        "拼接后按置信度过滤点云: keep_ratio=%.3f, kept_points=%d, total_points=%d",
        keep_ratio,
        kept_points,
        total_points,
    )
    return filtered_chunks


def align_npz_files(
    npz_paths: list[str | Path],
    output_path: str | Path,
    loop_pairs: list[LoopPair] | None = None,
    enable_loop: bool = ALIGN_LOOP_ENABLE,
    loop_model: Any | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> Path:
    """Load chunk npz files, align them, and save one concatenated npz file."""
    paths = [Path(npz_path) for npz_path in npz_paths]
    return save_aligned_npz_files_streaming(
        paths,
        output_path,
        loop_pairs=loop_pairs,
        enable_loop=enable_loop,
        loop_model=loop_model,
        loop_model_name_or_path=loop_model_name_or_path,
        device=device,
    )


def align(
    chunk_to_processes: list[ChunkToProcess],
    output_path: str | Path,
    loop_pairs: list[LoopPair] | None = None,
    enable_loop: bool = ALIGN_LOOP_ENABLE,
    loop_model: Any | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> Path:
    """Align a ChunkToProcess array and save one concatenated npz file."""
    return save_aligned_chunks_npz(
        chunk_to_processes,
        output_path,
        loop_pairs=loop_pairs,
        enable_loop=enable_loop,
        loop_model=loop_model,
        loop_model_name_or_path=loop_model_name_or_path,
        device=device,
    )


def save_aligned_npz_files_streaming(
    npz_paths: list[str | Path],
    output_path: str | Path,
    loop_pairs: list[LoopPair] | None = None,
    enable_loop: bool = ALIGN_LOOP_ENABLE,
    loop_model: Any | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> Path:
    """Align chunk npz files and stream the merged arrays through npy memmaps."""
    paths = [Path(npz_path) for npz_path in npz_paths]
    if not paths:
        raise ValueError("no npz files to align")

    transforms = estimate_npz_file_transforms(
        paths,
        loop_pairs=loop_pairs,
        enable_loop=enable_loop,
        loop_model=loop_model,
        loop_model_name_or_path=loop_model_name_or_path,
        device=device,
    )
    return write_aligned_npz_files_streaming(paths, transforms, output_path)


def estimate_npz_file_transforms(
    npz_paths: list[str | Path],
    loop_pairs: list[LoopPair] | None = None,
    enable_loop: bool = ALIGN_LOOP_ENABLE,
    loop_model: Any | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> list[Sim3]:
    """Estimate absolute chunk transforms while loading only needed chunks."""
    paths = [Path(npz_path) for npz_path in npz_paths]
    if not paths:
        return []
    if len(paths) == 1:
        return [_identity_sim3()]

    adjacent_transforms: list[Sim3] = []
    previous_chunk = npz_load(paths[0])
    for index, path in enumerate(paths[1:]):
        current_chunk = npz_load(path)
        transform = estimate_chunk_pair_sim3(previous_chunk, current_chunk)
        adjacent_transforms.append(transform)
        logger.info(
            "每组分块拼接完成: source_chunk_id=%s, target_chunk_id=%s, transform_matrix=%s",
            current_chunk.chunk_id,
            previous_chunk.chunk_id,
            np.array2string(_sim3_to_matrix(transform), precision=6, suppress_small=False),
        )
        previous_chunk = current_chunk

    if enable_loop:
        loop_constraints = estimate_loop_constraints_from_npz_files(
            paths,
            loop_pairs,
            model=loop_model,
            model_name_or_path=loop_model_name_or_path,
            device=device,
        )
        if loop_constraints:
            adjacent_transforms = optimize_sim3_pose_graph(
                adjacent_transforms,
                loop_constraints,
            )
            logger.info("回环全局优化完成: loop_constraint_count=%d", len(loop_constraints))
        else:
            logger.info("未找到可靠回环约束，使用顺序拼接结果")

    return [_identity_sim3(), *accumulate_sim3_transforms(adjacent_transforms)]


def estimate_loop_constraints_from_npz_files(
    npz_paths: list[str | Path],
    candidate_pairs: list[LoopPair] | None = None,
    model: Any | None = None,
    model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> list[LoopConstraint]:
    """Estimate loop constraints by loading only the two chunks needed per pair."""
    paths = [Path(npz_path) for npz_path in npz_paths]
    if len(paths) < 3:
        return []

    if candidate_pairs is None:
        probe_chunks = [_load_npz_probe_chunk(path) for path in paths]
        candidate_pairs = _find_loop_candidate_pairs(probe_chunks)
    if not candidate_pairs:
        return []

    if model is None:
        model = load_vggt_model(model_name_or_path=model_name_or_path, device=device)

    constraints: list[LoopConstraint] = []
    for source_index, target_index in candidate_pairs:
        if source_index == target_index:
            continue
        if source_index < target_index:
            source_index, target_index = target_index, source_index
        if source_index >= len(paths) or target_index < 0:
            continue
        if source_index - target_index < ALIGN_LOOP_MIN_CHUNK_GAP:
            continue

        source_chunk = npz_load(paths[source_index])
        target_chunk = npz_load(paths[target_index])
        try:
            transform = estimate_loop_pair_sim3(
                source_chunk,
                target_chunk,
                model=model,
                device=device,
            )
        except ValueError as exc:
            logger.debug(
                "跳过回环约束: source_chunk_id=%s, target_chunk_id=%s, reason=%s",
                source_chunk.chunk_id,
                target_chunk.chunk_id,
                exc,
            )
            continue

        constraints.append((source_index, target_index, transform))
        logger.info(
            "回环约束估计完成: source_chunk_id=%s, target_chunk_id=%s, transform_matrix=%s",
            source_chunk.chunk_id,
            target_chunk.chunk_id,
            np.array2string(_sim3_to_matrix(transform), precision=6, suppress_small=False),
        )
        if len(constraints) >= ALIGN_LOOP_MAX_PAIRS:
            break

    return constraints


def write_aligned_npz_files_streaming(
    npz_paths: list[str | Path],
    transforms: list[Sim3],
    output_path: str | Path,
) -> Path:
    """Stream transformed chunks into a VGGT-style aligned npz file."""
    paths = [Path(npz_path) for npz_path in npz_paths]
    if len(paths) != len(transforms):
        raise ValueError(
            f"npz_paths length {len(paths)} does not match transforms length {len(transforms)}"
        )
    if not paths:
        raise ValueError("no npz files to write")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _collect_stream_npz_metadata(paths)
    frame_count = len(metadata["frame_ids"])
    if frame_count == 0:
        raise ValueError("no frames to save")

    with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}_memmap_", dir=output_path.parent) as tmp_dir:
        tmp_path = Path(tmp_dir)
        arrays: dict[str, Any] = {
            "world_points_from_depth": open_memmap(
                tmp_path / "world_points_from_depth.npy",
                mode="w+",
                dtype=metadata["world_points_dtype"],
                shape=(frame_count, *metadata["world_points_shape"]),
            ),
            "depth_conf": open_memmap(
                tmp_path / "depth_conf.npy",
                mode="w+",
                dtype=metadata["confidence_dtype"],
                shape=(frame_count, *metadata["confidence_shape"]),
            ),
            "images": open_memmap(
                tmp_path / "images.npy",
                mode="w+",
                dtype=metadata["image_dtype"],
                shape=(frame_count, *metadata["image_shape"]),
            ),
            "masks": open_memmap(
                tmp_path / "masks.npy",
                mode="w+",
                dtype=metadata["mask_dtype"],
                shape=(frame_count, *metadata["mask_shape"]),
            ),
            "point_labels": open_memmap(
                tmp_path / "point_labels.npy",
                mode="w+",
                dtype=metadata["mask_dtype"],
                shape=(frame_count, *metadata["mask_shape"]),
            ),
        }
        frame_offset = 0
        seen_frame_ids: set[int] = set()
        for path, transform in zip(paths, transforms):
            chunk = npz_load(path)
            scale, rotation, translation = transform
            for frame in chunk.frames:
                if frame.id in seen_frame_ids:
                    continue
                seen_frame_ids.add(frame.id)
                aligned_points = apply_sim3_to_points(
                    frame.world_points,
                    scale,
                    rotation,
                    translation,
                ).astype(metadata["world_points_dtype"], copy=False)
                confidence = np.asarray(frame.world_points_conf)
                sky_mask = _frame_sky_mask_for_map(frame, aligned_points.shape[:2])
                filtered_confidence = np.where(sky_mask, 0, confidence).astype(
                    metadata["confidence_dtype"],
                    copy=False,
                )
                arrays["world_points_from_depth"][frame_offset] = aligned_points
                arrays["depth_conf"][frame_offset] = filtered_confidence
                arrays["images"][frame_offset] = frame.image
                arrays["masks"][frame_offset] = frame.mask
                arrays["point_labels"][frame_offset] = frame.mask
                frame_offset += 1

        if frame_offset != frame_count:
            raise ValueError(f"expected {frame_count} unique frames, wrote {frame_offset}")

        for value in arrays.values():
            value.flush()
        arrays.update(
            {
                "pose_enc": np.empty((frame_count, 0), dtype=np.float32),
                "depth": np.zeros((frame_count, *metadata["confidence_shape"]), dtype=np.float32),
                "extrinsic": _identity_extrinsics(frame_count),
                "intrinsic": _identity_intrinsics(frame_count),
                "chunk_id": np.array(metadata["chunk_id"]),
                "frame_ids": np.array(metadata["frame_ids"], dtype=np.int64),
                "gps_locations": np.array(metadata["gps_locations"], dtype=np.float64),
            }
        )
        np.savez(output_path, **arrays)

    return output_path


def align_chunks(
    chunks: list[ChunkToProcess],
    loop_pairs: list[LoopPair] | None = None,
    enable_loop: bool = ALIGN_LOOP_ENABLE,
    loop_model: Any | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | None = None,
) -> list[ChunkToProcess]:
    """Backward-compatible alias for align_chunk_to_processes."""
    return align_chunk_to_processes(
        chunks,
        loop_pairs=loop_pairs,
        enable_loop=enable_loop,
        loop_model=loop_model,
        loop_model_name_or_path=loop_model_name_or_path,
        device=device,
    )


def optimize_sim3_pose_graph(
    adjacent_transforms: list[Sim3],
    loop_constraints: list[LoopConstraint],
) -> list[Sim3]:
    """Optimize chunk poses using adjacent and loop Sim3 constraints."""
    if not adjacent_transforms or not loop_constraints:
        return adjacent_transforms

    initial_absolute = [_identity_sim3()]
    initial_absolute.extend(accumulate_sim3_transforms(adjacent_transforms))
    initial_params = _absolute_transforms_to_params(initial_absolute[1:])

    edges: list[LoopConstraint] = []
    for index, transform in enumerate(adjacent_transforms):
        edges.append((index + 1, index, transform))
    edges.extend(loop_constraints)

    def residual(params: NDArray[np.float64]) -> NDArray[np.float64]:
        absolute = [_identity_sim3()]
        absolute.extend(_params_to_absolute_transforms(params))
        values = []
        for source_index, target_index, observed in edges:
            predicted = relative_sim3(
                absolute[source_index],
                absolute[target_index],
            )
            error = compose_sim3(invert_sim3(observed), predicted)
            values.extend(_sim3_error_vector(error))
        return np.array(values, dtype=np.float64)

    result = least_squares(
        residual,
        initial_params,
        max_nfev=ALIGN_LOOP_OPT_MAX_ITERS,
        loss="soft_l1",
        f_scale=1.0,
    )
    optimized_absolute = [_identity_sim3()]
    optimized_absolute.extend(_params_to_absolute_transforms(result.x))
    optimized_adjacent = []
    for index in range(len(adjacent_transforms)):
        optimized_adjacent.append(
            relative_sim3(optimized_absolute[index + 1], optimized_absolute[index])
        )

    logger.info(
        "Sim3图优化状态: success=%s, initial_cost=%.6g, final_cost=%.6g, iterations=%s",
        result.success,
        float(np.mean(residual(initial_params) ** 2)),
        float(np.mean(residual(result.x) ** 2)),
        result.nfev,
    )
    return optimized_adjacent


def _frame_id_to_index(chunk: ChunkToProcess) -> dict[int, int]:
    if len(chunk.frame_ids) != len(chunk.frames):
        raise ValueError(
            f"chunk {chunk.chunk_id} frame_ids length does not match frames length"
        )

    frame_index: dict[int, int] = {}
    for index, frame_id in enumerate(chunk.frame_ids):
        if frame_id in frame_index:
            raise ValueError(f"chunk {chunk.chunk_id} has duplicate frame_id {frame_id}")
        frame_index[frame_id] = index

    return frame_index


def _normalize_conf_shape(
    conf: NDArray[np.floating],
    points: NDArray[np.floating],
) -> NDArray[np.floating]:
    if conf.shape == points.shape[:-1]:
        return conf
    if conf.shape == points.shape[:-1] + (1,):
        return np.squeeze(conf, axis=-1)

    raise ValueError(
        f"world_points_conf shape {conf.shape} does not match world_points shape "
        f"{points.shape}"
    )


def _find_loop_candidate_pairs(chunks: list[ChunkToProcess]) -> list[LoopPair]:
    scored_pairs: list[tuple[float, int, int]] = []
    for source_index, source_chunk in enumerate(chunks):
        per_source: list[tuple[float, int, int]] = []
        for target_index in range(0, source_index - ALIGN_LOOP_MIN_CHUNK_GAP + 1):
            target_chunk = chunks[target_index]
            score = _chunk_visual_score(source_chunk, target_chunk)
            if score >= ALIGN_LOOP_MIN_SCORE:
                per_source.append((score, source_index, target_index))

        per_source.sort(reverse=True, key=lambda item: item[0])
        scored_pairs.extend(per_source[:ALIGN_LOOP_TOP_K])

    scored_pairs.sort(reverse=True, key=lambda item: item[0])
    return [
        (source_index, target_index)
        for _, source_index, target_index in scored_pairs[:ALIGN_LOOP_MAX_PAIRS]
    ]


def _load_npz_probe_chunk(npz_path: str | Path) -> ChunkToProcess:
    """Load only first/middle/last frames from a chunk npz for loop scoring."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as loaded:
        missing_keys = [key for key in ("images", "chunk_id", "frame_ids", "gps_locations") if key not in loaded.files]
        if missing_keys:
            raise KeyError(f"{npz_path} is missing loop probe keys: {missing_keys}")

        frame_ids = np.asarray(loaded["frame_ids"])
        if frame_ids.size == 0:
            return ChunkToProcess(chunk_id=int(np.asarray(loaded["chunk_id"]).item()))
        indexes = sorted({0, int(frame_ids.size // 2), int(frame_ids.size - 1)})
        images = np.asarray(loaded["images"][indexes])
        gps_locations = np.asarray(loaded["gps_locations"][indexes])
        masks = np.asarray(loaded["masks"][indexes]) if "masks" in loaded.files else None
        chunk_id = int(np.asarray(loaded["chunk_id"]).item())

    chunk = ChunkToProcess(chunk_id=chunk_id, frames=[], frame_ids=[])
    empty_points = np.empty((0, 0, 3), dtype=np.float32)
    empty_conf = np.empty((0, 0), dtype=np.float32)
    for output_index, frame_index in enumerate(indexes):
        chunk.add_frame(
            FrameInChunk(
                id=int(frame_ids[frame_index]),
                image=images[output_index],
                gps_location=tuple(gps_locations[output_index]),
                mask=None if masks is None else masks[output_index],
                chunk_id=chunk_id,
                world_points=empty_points,
                world_points_conf=empty_conf,
            )
        )
    return chunk


def _chunk_visual_score(chunk_a: ChunkToProcess, chunk_b: ChunkToProcess) -> float:
    best = 0.0
    for frame_a in _loop_probe_frames(chunk_a):
        for frame_b in _loop_probe_frames(chunk_b):
            score, _, _, _ = _match_frame_features(frame_a.image, frame_b.image)
            best = max(best, score)
    return best


def _loop_probe_frames(chunk: ChunkToProcess) -> list[FrameInChunk]:
    if not chunk.frames:
        return []
    indexes = sorted({0, len(chunk.frames) // 2, len(chunk.frames) - 1})
    return [chunk.frames[index] for index in indexes]


def _loop_probe_frame_items(chunk: ChunkToProcess) -> list[tuple[int, FrameInChunk]]:
    if not chunk.frames:
        return []
    indexes = sorted({0, len(chunk.frames) // 2, len(chunk.frames) - 1})
    return [(index, chunk.frames[index]) for index in indexes]


def _loop_window(chunk: ChunkToProcess, center_index: int) -> list[FrameInChunk]:
    start = max(0, center_index - ALIGN_LOOP_WINDOW_RADIUS)
    end = min(len(chunk.frames), center_index + ALIGN_LOOP_WINDOW_RADIUS + 1)
    return chunk.frames[start:end]


def _joint_loop_chunk(
    source_window: list[FrameInChunk],
    target_window: list[FrameInChunk],
) -> Chunk:
    frames = [
        BaseFrame(
            id=frame.id,
            image=_frame_image_to_uint8_hwc(frame.image),
            gps_location=frame.gps_location,
            mask=_frame_image_to_uint8_hwc(frame.mask),
        )
        for frame in [*source_window, *target_window]
    ]
    return Chunk(id=-1, frames=frames)


def _chunk_window_to_process(
    window: list[FrameInChunk],
    chunk_id: int,
) -> ChunkToProcess:
    chunk = ChunkToProcess(chunk_id=chunk_id, frames=[], frame_ids=[])
    for frame in window:
        chunk.add_frame(frame)
    return chunk


def _prediction_window_to_chunk(
    predictions: dict[str, Any],
    start: int,
    end: int,
    chunk_id: int,
) -> ChunkToProcess:
    points = np.asarray(predictions["world_points_from_depth"])[start:end]
    confidence = np.asarray(predictions["depth_conf"])[start:end]
    images = np.asarray(predictions["images"])[start:end]
    frame_ids = np.asarray(predictions["frame_ids"])[start:end]
    gps_locations = np.asarray(predictions["gps_locations"])[start:end]

    chunk = ChunkToProcess(chunk_id=chunk_id, frames=[], frame_ids=[])
    for index, frame_id in enumerate(frame_ids):
        chunk.add_frame(
            FrameInChunk(
                id=int(frame_id),
                image=images[index],
                gps_location=tuple(gps_locations[index]),
                mask=None,
                chunk_id=chunk_id,
                world_points=points[index],
                world_points_conf=confidence[index],
            )
        )
    return chunk


def _frame_image_to_uint8_hwc(image: NDArray[Any]) -> NDArray[np.uint8]:
    image_array = np.asarray(image)
    if image_array.ndim == 3 and image_array.shape[0] == 3 and image_array.shape[-1] != 3:
        image_array = np.transpose(image_array, (1, 2, 0))
    if image_array.dtype != np.uint8:
        image_array = (np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image_array.ndim == 2:
        return cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    if image_array.ndim != 3:
        raise ValueError(f"frame image must be 2D or 3D, got shape {image_array.shape}")
    if image_array.shape[2] == 4:
        return cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    if image_array.shape[2] != 3:
        raise ValueError(f"frame image must have 3 or 4 channels, got {image_array.shape}")
    return np.ascontiguousarray(image_array)


def _match_frame_features(
    source_image: NDArray[np.uint8],
    target_image: NDArray[np.uint8],
) -> tuple[float, list[cv2.DMatch], list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    source_gray = _to_gray_uint8(source_image)
    target_gray = _to_gray_uint8(target_image)
    detector = cv2.ORB_create(nfeatures=3000)
    source_keypoints, source_descriptors = detector.detectAndCompute(source_gray, None)
    target_keypoints, target_descriptors = detector.detectAndCompute(target_gray, None)
    if (
        source_descriptors is None
        or target_descriptors is None
        or not source_keypoints
        or not target_keypoints
    ):
        return 0.0, [], [], []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    good_matches = []
    for match_group in raw_matches:
        if len(match_group) < 2:
            continue
        best, second_best = match_group
        if best.distance < 0.75 * second_best.distance:
            good_matches.append(best)

    denominator = max(1, min(len(source_keypoints), len(target_keypoints)))
    score = len(good_matches) / denominator
    return float(score), good_matches, source_keypoints, target_keypoints


def _matched_points_from_frames(
    source_frame: FrameInChunk,
    target_frame: FrameInChunk,
    matches: list[cv2.DMatch],
    keypoints_source: list[cv2.KeyPoint],
    keypoints_target: list[cv2.KeyPoint],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    source_conf = _normalize_conf_shape(
        np.asarray(source_frame.world_points_conf),
        np.asarray(source_frame.world_points),
    )
    target_conf = _normalize_conf_shape(
        np.asarray(target_frame.world_points_conf),
        np.asarray(target_frame.world_points),
    )
    source_points = []
    target_points = []
    weights = []
    for match in matches:
        sx, sy = keypoints_source[match.queryIdx].pt
        tx, ty = keypoints_target[match.trainIdx].pt
        source_xy = _image_point_to_map_index(
            sx,
            sy,
            _image_hw(source_frame.image),
            source_frame.world_points.shape[:2],
        )
        target_xy = _image_point_to_map_index(
            tx,
            ty,
            _image_hw(target_frame.image),
            target_frame.world_points.shape[:2],
        )
        if source_xy is None or target_xy is None:
            continue

        source_row, source_col = source_xy
        target_row, target_col = target_xy
        if _is_sky_point(source_frame, source_frame.world_points.shape[:2], source_row, source_col):
            continue
        if _is_sky_point(target_frame, target_frame.world_points.shape[:2], target_row, target_col):
            continue

        source_confidence = float(source_conf[source_row, source_col])
        target_confidence = float(target_conf[target_row, target_col])
        if (
            source_confidence <= ALIGN_CONF_THRESHOLD
            or target_confidence <= ALIGN_CONF_THRESHOLD
        ):
            continue

        source_point = source_frame.world_points[source_row, source_col]
        target_point = target_frame.world_points[target_row, target_col]
        if not np.isfinite(source_point).all() or not np.isfinite(target_point).all():
            continue

        source_points.append(source_point)
        target_points.append(target_point)
        weights.append(np.sqrt(source_confidence * target_confidence))

    return (
        np.asarray(target_points, dtype=np.float64),
        np.asarray(source_points, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _image_point_to_map_index(
    x: float,
    y: float,
    image_shape: tuple[int, int],
    map_shape: tuple[int, int],
) -> tuple[int, int] | None:
    image_h, image_w = image_shape
    map_h, map_w = map_shape
    if image_h <= 0 or image_w <= 0 or map_h <= 0 or map_w <= 0:
        return None
    col = int(round(x * (map_w - 1) / max(image_w - 1, 1)))
    row = int(round(y * (map_h - 1) / max(image_h - 1, 1)))
    if 0 <= row < map_h and 0 <= col < map_w:
        return row, col
    return None


def _to_gray_uint8(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image_array = np.asarray(image)
    if image_array.dtype != np.uint8:
        image_array = (np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image_array.ndim == 2:
        return image_array
    if image_array.ndim != 3:
        raise ValueError(f"image must be 2D or 3D, got shape {image_array.shape}")
    if image_array.shape[0] == 3 and image_array.shape[-1] != 3:
        image_array = np.transpose(image_array, (1, 2, 0))
    if image_array.shape[2] == 3:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    if image_array.shape[2] == 4:
        return cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"image must have 1, 3, or 4 channels, got {image_array.shape}")


def _image_hw(image: NDArray[np.uint8]) -> tuple[int, int]:
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array.shape
    if image_array.ndim != 3:
        raise ValueError(f"image must be 2D or 3D, got shape {image_array.shape}")
    if image_array.shape[0] in (3, 4) and image_array.shape[-1] not in (3, 4):
        return image_array.shape[1], image_array.shape[2]
    return image_array.shape[0], image_array.shape[1]


def _is_sky_point(
    frame: FrameInChunk,
    map_shape: tuple[int, int],
    row: int,
    col: int,
) -> bool:
    sky_mask = _frame_sky_mask_for_map(frame, map_shape)
    return bool(sky_mask[row, col])


def _frame_sky_mask_for_map(
    frame: FrameInChunk,
    map_shape: tuple[int, int],
) -> NDArray[np.bool_]:
    return _mask_to_map_sky_mask(frame.mask, _image_hw(frame.image), map_shape)


def _mask_to_map_sky_mask(
    mask: NDArray[Any],
    image_shape: tuple[int, int],
    map_shape: tuple[int, int],
) -> NDArray[np.bool_]:
    mask_array = _mask_to_hwc(mask)
    if not np.any(mask_array == SKY_CLASS_ID):
        return np.zeros(map_shape, dtype=bool)

    sky_mask = np.any(mask_array == SKY_CLASS_ID, axis=2)
    image_h, image_w = image_shape
    if sky_mask.shape != (image_h, image_w):
        sky_mask = cv2.resize(
            sky_mask.astype(np.uint8),
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    map_h, map_w = map_shape
    if sky_mask.shape != (map_h, map_w):
        sky_mask = cv2.resize(
            sky_mask.astype(np.uint8),
            (map_w, map_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return sky_mask


def _mask_to_hwc(mask: NDArray[Any]) -> NDArray[Any]:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 2:
        return mask_array[:, :, None]
    if mask_array.ndim != 3:
        raise ValueError(f"frame mask must be 2D or 3D, got shape {mask_array.shape}")
    if mask_array.shape[0] in (1, 3, 4) and mask_array.shape[-1] not in (1, 3, 4):
        return np.transpose(mask_array, (1, 2, 0))
    return mask_array


def _validate_point_pairs(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    if source.shape != target.shape:
        raise ValueError(
            f"source shape {source.shape} does not match target shape {target.shape}"
        )
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must have shape (N, 3), got {source.shape}"
        )
    if weights.shape != (source.shape[0],):
        raise ValueError(
            f"weights must have shape ({source.shape[0]},), got {weights.shape}"
        )
    if source.shape[0] < ALIGN_MIN_POINTS:
        raise ValueError(
            f"not enough point pairs: got {source.shape[0]}, "
            f"need at least {ALIGN_MIN_POINTS}"
        )
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target must contain only finite points")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if weights.sum() <= 1e-12:
        raise ValueError("weights sum is zero; cannot estimate Sim3")


def _chunks_to_npz_predictions(chunks: list[ChunkToProcess]) -> dict[str, Any]:
    frames = []
    seen_frame_ids: set[int] = set()
    for chunk in chunks:
        for frame in chunk.frames:
            if frame.id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame.id)
            frames.append(frame)

    if not frames:
        raise ValueError("no frames to save")

    world_points = np.stack([frame.world_points for frame in frames])
    depth_conf = np.stack([frame.world_points_conf for frame in frames])
    images = np.stack([frame.image for frame in frames])
    masks = np.stack([frame.mask for frame in frames])
    sky_masks = np.stack(
        [
            _frame_sky_mask_for_map(frame, frame.world_points.shape[:2])
            for frame in frames
        ]
    )
    depth_conf = np.where(sky_masks, 0, depth_conf).astype(depth_conf.dtype, copy=False)
    frame_ids = np.array([frame.id for frame in frames], dtype=np.int64)
    frame_count = len(frames)
    gps_locations = np.array(
        [
            frame.gps_location
            if frame.gps_location is not None
            else (np.nan, np.nan, np.nan)
            for frame in frames
        ],
        dtype=np.float64,
    )

    return {
        "world_points_from_depth": world_points,
        "depth_conf": depth_conf,
        "images": images,
        "masks": masks,
        "point_labels": masks,
        "pose_enc": np.empty((frame_count, 0), dtype=np.float32),
        "depth": np.zeros(depth_conf.shape, dtype=np.float32),
        "extrinsic": _identity_extrinsics(frame_count),
        "intrinsic": _identity_intrinsics(frame_count),
        "chunk_id": np.array(chunks[0].chunk_id if chunks else -1),
        "frame_ids": frame_ids,
        "gps_locations": gps_locations,
    }


def _collect_stream_npz_metadata(paths: list[Path]) -> dict[str, Any]:
    frame_ids: list[int] = []
    gps_locations: list[tuple[float, float, float]] = []
    seen_frame_ids: set[int] = set()
    chunk_id: int | None = None
    world_points_shape: tuple[int, ...] | None = None
    confidence_shape: tuple[int, ...] | None = None
    image_shape: tuple[int, ...] | None = None
    mask_shape: tuple[int, ...] | None = None
    world_points_dtype: np.dtype[Any] | None = None
    confidence_dtype: np.dtype[Any] | None = None
    image_dtype: np.dtype[Any] | None = None
    mask_dtype: np.dtype[Any] | None = None

    for path in paths:
        with np.load(path) as loaded:
            missing_keys = [key for key in NPZ_LOAD_KEYS if key not in loaded.files]
            if missing_keys:
                raise KeyError(f"{path} is missing ChunkToProcess keys: {missing_keys}")
            if "masks" not in loaded.files:
                raise KeyError(f"{path} is missing masks")

            if chunk_id is None:
                chunk_id = int(np.asarray(loaded["chunk_id"]).item())
                world_points_shape = tuple(loaded["world_points_from_depth"].shape[1:])
                confidence_shape = tuple(loaded["depth_conf"].shape[1:])
                image_shape = tuple(loaded["images"].shape[1:])
                mask_shape = tuple(loaded["masks"].shape[1:])
                world_points_dtype = loaded["world_points_from_depth"].dtype
                confidence_dtype = loaded["depth_conf"].dtype
                image_dtype = loaded["images"].dtype
                mask_dtype = loaded["masks"].dtype

            _assert_stream_array_compatible(path, "world_points_from_depth", loaded["world_points_from_depth"], world_points_shape, world_points_dtype)
            _assert_stream_array_compatible(path, "depth_conf", loaded["depth_conf"], confidence_shape, confidence_dtype)
            _assert_stream_array_compatible(path, "images", loaded["images"], image_shape, image_dtype)
            _assert_stream_array_compatible(path, "masks", loaded["masks"], mask_shape, mask_dtype)

            loaded_frame_ids = np.asarray(loaded["frame_ids"])
            loaded_gps = np.asarray(loaded["gps_locations"])
            for index, frame_id in enumerate(loaded_frame_ids):
                frame_id_int = int(frame_id)
                if frame_id_int in seen_frame_ids:
                    continue
                seen_frame_ids.add(frame_id_int)
                frame_ids.append(frame_id_int)
                gps = loaded_gps[index]
                gps_locations.append(
                    (
                        float(gps[0]) if gps.size > 0 else float("nan"),
                        float(gps[1]) if gps.size > 1 else float("nan"),
                        float(gps[2]) if gps.size > 2 else float("nan"),
                    )
                )

    return {
        "chunk_id": -1 if chunk_id is None else chunk_id,
        "frame_ids": frame_ids,
        "gps_locations": gps_locations,
        "world_points_shape": world_points_shape,
        "confidence_shape": confidence_shape,
        "image_shape": image_shape,
        "mask_shape": mask_shape,
        "world_points_dtype": world_points_dtype,
        "confidence_dtype": confidence_dtype,
        "image_dtype": image_dtype,
        "mask_dtype": mask_dtype,
    }


def _assert_stream_array_compatible(
    path: Path,
    key: str,
    value: NDArray[Any],
    expected_shape: tuple[int, ...] | None,
    expected_dtype: np.dtype[Any] | None,
) -> None:
    if expected_shape is not None and tuple(value.shape[1:]) != expected_shape:
        raise ValueError(
            f"{path} {key} trailing shape {value.shape[1:]} does not match {expected_shape}"
        )
    if expected_dtype is not None and value.dtype != expected_dtype:
        raise ValueError(f"{path} {key} dtype {value.dtype} does not match {expected_dtype}")


def _identity_extrinsics(frame_count: int) -> NDArray[np.float32]:
    extrinsic = np.zeros((frame_count, 3, 4), dtype=np.float32)
    extrinsic[:, :3, :3] = np.eye(3, dtype=np.float32)
    return extrinsic


def _identity_intrinsics(frame_count: int) -> NDArray[np.float32]:
    intrinsic = np.zeros((frame_count, 3, 3), dtype=np.float32)
    intrinsic[:, :, :] = np.eye(3, dtype=np.float32)
    return intrinsic


def _identity_sim3() -> Sim3:
    return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)


def _absolute_transforms_to_params(transforms: list[Sim3]) -> NDArray[np.float64]:
    params = []
    for scale, rotation, translation in transforms:
        params.extend(translation.astype(np.float64, copy=False))
        params.extend(Rotation.from_matrix(rotation).as_rotvec())
        params.append(np.log(max(float(scale), 1e-12)))
    return np.asarray(params, dtype=np.float64)


def _params_to_absolute_transforms(params: NDArray[np.float64]) -> list[Sim3]:
    if params.size % 7 != 0:
        raise ValueError(f"Sim3 parameter vector length must be multiple of 7, got {params.size}")
    transforms = []
    for offset in range(0, params.size, 7):
        translation = params[offset : offset + 3]
        rotation = Rotation.from_rotvec(params[offset + 3 : offset + 6]).as_matrix()
        scale = float(np.exp(params[offset + 6]))
        transforms.append((scale, rotation, translation))
    return transforms


def _sim3_error_vector(transform: Sim3) -> list[float]:
    scale, rotation, translation = transform
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(rotvec[0]),
        float(rotvec[1]),
        float(rotvec[2]),
        float(np.log(max(scale, 1e-12))),
    ]


def _sim3_to_matrix(transform: Sim3) -> NDArray[np.float64]:
    scale, rotation, translation = transform
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix
