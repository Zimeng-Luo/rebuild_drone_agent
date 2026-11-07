from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from config import KEY_FRAME_DELETE_RATIO
from src.datatype import BaseFrame, OriginalFrames
from utils.calcula import (
    calculate_optical_flow_score,
    calculate_overlap_score,
    calculate_sharpness_score,
)


logger = logging.getLogger(__name__)

ImageArray = NDArray[np.uint8]


@dataclass
class FrameMetadata:
    frame_id: int
    image_path: Path
    gps_location: tuple[float, float, float] | None
    sharpness_score: float
    exposure_score: float
    feature_count: int
    pose: tuple[float, float, float] | None
    previous_difference: float
    previous_keyframe_difference: float
    previous_overlap_score: float
    previous_keyframe_overlap_score: float
    descriptor: NDArray[np.uint8] | None
    quality_score: float = 0.0
    coverage_score: float = 0.0
    total_score: float = 0.0


@dataclass
class FrameScore:
    frame: BaseFrame
    optical_flow_score: float = 0.0
    overlap_score: float = 0.0
    sharpness_score: float = 0.0
    total_score: float = 0.0


def build_frame_metadata(
    frame: BaseFrame,
    image_path: Path,
    previous_image: ImageArray | None = None,
    previous_keyframe_image: ImageArray | None = None,
) -> FrameMetadata:
    """Build metadata and descriptor for one preprocessed frame."""
    gray = _to_gray(frame.image)
    detector = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptor = detector.detectAndCompute(gray, None)
    feature_count = len(keypoints) if keypoints else 0

    previous_difference = _image_difference(previous_image, frame.image)
    previous_keyframe_difference = _image_difference(previous_keyframe_image, frame.image)
    previous_overlap_score = _overlap_with_optional_image(previous_image, frame.image)
    previous_keyframe_overlap_score = _overlap_with_optional_image(
        previous_keyframe_image,
        frame.image,
    )

    return FrameMetadata(
        frame_id=frame.id,
        image_path=image_path,
        gps_location=frame.gps_location,
        sharpness_score=calculate_sharpness_score(frame.image),
        exposure_score=_calculate_exposure_score(gray),
        feature_count=feature_count,
        pose=frame.gps_location,
        previous_difference=previous_difference,
        previous_keyframe_difference=previous_keyframe_difference,
        previous_overlap_score=previous_overlap_score,
        previous_keyframe_overlap_score=previous_keyframe_overlap_score,
        descriptor=descriptor,
    )


def select_key_frame_metadata(
    metadata_list: list[FrameMetadata],
    delete_ratio: float = KEY_FRAME_DELETE_RATIO,
) -> list[FrameMetadata]:
    """Run candidate filtering and global key-frame selection on metadata."""
    if not 0.0 <= delete_ratio < 1.0:
        raise ValueError(f"delete_ratio must be in [0.0, 1.0), got {delete_ratio}")

    if not metadata_list:
        return []

    candidate_count = max(1, int(round(len(metadata_list) * (1.0 - delete_ratio))))
    candidates = _select_quality_candidates(metadata_list, candidate_count)
    candidates = _remove_near_similar_overlapped_frames(candidates)
    selected = _select_global_coverage(candidates, candidate_count)

    logger.info(
        "关键帧三遍选择完成: total=%d, candidates=%d, selected=%d, delete_ratio=%.3f",
        len(metadata_list),
        len(candidates),
        len(selected),
        delete_ratio,
    )
    return selected


def select_key_frames(
    frames: OriginalFrames,
    delete_ratio: float = KEY_FRAME_DELETE_RATIO,
) -> OriginalFrames:
    """Backward-compatible in-memory key-frame selection."""
    if not 0.0 <= delete_ratio < 1.0:
        raise ValueError(f"delete_ratio must be in [0.0, 1.0), got {delete_ratio}")

    if not frames.frames:
        return OriginalFrames()

    frame_scores = _calculate_frame_scores(frames.frames)
    _normalize_and_sum_scores(frame_scores)

    delete_count = int(len(frame_scores) * delete_ratio)
    deleted_frame_ids = {
        frame_score.frame.id
        for frame_score in sorted(frame_scores, key=lambda item: item.total_score)[
            :delete_count
        ]
    }

    selected_frames = OriginalFrames()
    for frame in frames.frames:
        if frame.id not in deleted_frame_ids:
            selected_frames.add_frame(frame)

    logger.info(
        "关键帧计算完成: total=%d, selected=%d, deleted=%d, delete_ratio=%.3f",
        len(frame_scores),
        len(selected_frames.frames),
        delete_count,
        delete_ratio,
    )
    return selected_frames


def _select_quality_candidates(
    metadata_list: list[FrameMetadata],
    target_count: int,
) -> list[FrameMetadata]:
    sharpness_scores = _normalize_scores(
        [metadata.sharpness_score for metadata in metadata_list]
    )
    exposure_scores = _normalize_scores(
        [metadata.exposure_score for metadata in metadata_list]
    )
    feature_scores = _normalize_scores(
        [float(metadata.feature_count) for metadata in metadata_list]
    )
    difference_scores = _normalize_scores(
        [metadata.previous_keyframe_difference for metadata in metadata_list]
    )

    for index, metadata in enumerate(metadata_list):
        metadata.quality_score = (
            0.35 * sharpness_scores[index]
            + 0.25 * exposure_scores[index]
            + 0.25 * feature_scores[index]
            + 0.15 * difference_scores[index]
        )

    reserve_count = min(len(metadata_list), max(target_count * 3, target_count))
    candidates = sorted(
        metadata_list,
        key=lambda metadata: metadata.quality_score,
        reverse=True,
    )[:reserve_count]
    return sorted(candidates, key=lambda metadata: metadata.frame_id)


def _remove_near_similar_overlapped_frames(
    candidates: list[FrameMetadata],
    min_frame_gap: int = 2,
    similarity_threshold: float = 0.88,
) -> list[FrameMetadata]:
    selected: list[FrameMetadata] = []
    for metadata in candidates:
        duplicate_index = _find_duplicate_index(
            selected,
            metadata,
            min_frame_gap,
            similarity_threshold,
        )
        if duplicate_index is None:
            selected.append(metadata)
            continue

        if metadata.quality_score > selected[duplicate_index].quality_score:
            selected[duplicate_index] = metadata

    return sorted(selected, key=lambda metadata: metadata.frame_id)


def _select_global_coverage(
    candidates: list[FrameMetadata],
    target_count: int,
) -> list[FrameMetadata]:
    if len(candidates) <= target_count:
        return sorted(candidates, key=lambda metadata: metadata.frame_id)

    selected: list[FrameMetadata] = []
    remaining = list(candidates)
    while remaining and len(selected) < target_count:
        best_metadata = max(
            remaining,
            key=lambda metadata: _global_selection_score(metadata, selected),
        )
        selected.append(best_metadata)
        remaining.remove(best_metadata)

    return sorted(selected, key=lambda metadata: metadata.frame_id)


def _global_selection_score(
    metadata: FrameMetadata,
    selected: list[FrameMetadata],
) -> float:
    if not selected:
        coverage_score = 1.0
    else:
        frame_distances = [
            abs(metadata.frame_id - selected_metadata.frame_id)
            for selected_metadata in selected
        ]
        max_distance = max(1, metadata.frame_id, selected[-1].frame_id)
        time_coverage = min(frame_distances) / max_distance
        pose_coverage = _minimum_pose_distance(metadata, selected)
        descriptor_novelty = 1.0 - max(
            _descriptor_similarity(metadata.descriptor, selected_metadata.descriptor)
            for selected_metadata in selected
        )
        coverage_score = 0.4 * time_coverage + 0.3 * pose_coverage + 0.3 * descriptor_novelty

    metadata.coverage_score = float(coverage_score)
    metadata.total_score = 0.55 * metadata.quality_score + 0.45 * metadata.coverage_score
    return metadata.total_score


def _find_duplicate_index(
    selected: list[FrameMetadata],
    metadata: FrameMetadata,
    min_frame_gap: int,
    similarity_threshold: float,
) -> int | None:
    for index, selected_metadata in enumerate(selected):
        frame_gap = abs(metadata.frame_id - selected_metadata.frame_id)
        similarity = _descriptor_similarity(metadata.descriptor, selected_metadata.descriptor)
        # Cached overlap scores describe each frame's *previous* neighbours,
        # not this candidate pair. They cannot establish a duplicate here.
        if frame_gap <= min_frame_gap or similarity >= similarity_threshold:
            return index
    return None


def _minimum_pose_distance(
    metadata: FrameMetadata,
    selected: list[FrameMetadata],
) -> float:
    if metadata.pose is None:
        return 0.0

    distances = []
    for selected_metadata in selected:
        if selected_metadata.pose is None:
            continue
        distances.append(
            float(np.linalg.norm(np.asarray(metadata.pose) - np.asarray(selected_metadata.pose)))
        )
    if not distances:
        return 0.0

    return float(np.clip(min(distances), 0.0, 1.0))


def _descriptor_similarity(
    first_descriptor: NDArray[np.uint8] | None,
    second_descriptor: NDArray[np.uint8] | None,
    lowe_ratio: float = 0.75,
) -> float:
    if first_descriptor is None or second_descriptor is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(first_descriptor, second_descriptor, k=2)
    good_matches = []
    for match_group in raw_matches:
        if len(match_group) < 2:
            continue
        best_match, second_best_match = match_group
        if best_match.distance < lowe_ratio * second_best_match.distance:
            good_matches.append(best_match)

    denominator = min(len(first_descriptor), len(second_descriptor))
    if denominator == 0:
        return 0.0
    return float(np.clip(len(good_matches) / denominator, 0.0, 1.0))


def _calculate_exposure_score(gray: ImageArray) -> float:
    mean_value = float(np.mean(gray))
    centered_score = 1.0 - abs(mean_value - 127.5) / 127.5
    clipped_ratio = float(np.mean((gray <= 5) | (gray >= 250)))
    return float(np.clip(centered_score * (1.0 - clipped_ratio), 0.0, 1.0))


def _image_difference(
    previous_image: ImageArray | None,
    current_image: ImageArray,
) -> float:
    if previous_image is None:
        return 0.0
    return calculate_optical_flow_score(previous_image, current_image)


def _overlap_with_optional_image(
    previous_image: ImageArray | None,
    current_image: ImageArray,
) -> float:
    if previous_image is None:
        return 0.0
    return calculate_overlap_score(previous_image, current_image)


def _calculate_frame_scores(frames: list[BaseFrame]) -> list[FrameScore]:
    frame_scores = [FrameScore(frame=frame) for frame in frames]

    for index, frame_score in enumerate(frame_scores):
        frame_score.sharpness_score = calculate_sharpness_score(frame_score.frame.image)

        if index == 0:
            continue

        previous_image = frame_scores[index - 1].frame.image
        current_image = frame_score.frame.image
        frame_score.optical_flow_score = calculate_optical_flow_score(
            previous_image,
            current_image,
        )
        frame_score.overlap_score = calculate_overlap_score(previous_image, current_image)

    return frame_scores


def _normalize_and_sum_scores(frame_scores: list[FrameScore]) -> None:
    optical_flow_scores = _normalize_scores(
        [frame_score.optical_flow_score for frame_score in frame_scores]
    )
    overlap_scores = _normalize_scores(
        [frame_score.overlap_score for frame_score in frame_scores]
    )
    sharpness_scores = _normalize_scores(
        [frame_score.sharpness_score for frame_score in frame_scores]
    )

    for index, frame_score in enumerate(frame_scores):
        frame_score.optical_flow_score = optical_flow_scores[index]
        frame_score.overlap_score = overlap_scores[index]
        frame_score.sharpness_score = sharpness_scores[index]
        frame_score.total_score = (
            frame_score.optical_flow_score
            + frame_score.overlap_score
            + frame_score.sharpness_score
        )


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    values = np.asarray(scores, dtype=np.float64)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return [0.0 for _ in scores]

    normalized = (values - min_value) / (max_value - min_value)
    return [float(score) for score in normalized]


def _to_gray(image: ImageArray) -> ImageArray:
    if image.ndim == 2:
        return image

    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    raise ValueError(f"Expected image with 1, 3, or 4 channels, got {image.shape[2]}")
