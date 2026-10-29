"""
测试对可靠性的预测是否准确
data:2026/7/9
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULT_DIR
from src.datatype import BaseFrame, Chunk
from src.image_preprocess import preprocess_original_frame
from src.key_frame_select import build_frame_metadata, select_key_frame_metadata
from src.logging_utils import configure_logging, configure_task_file_logging
from src.rebuild import chunk_to_vggt_images, load_vggt_model
from src.task_summary import ResourceStepMonitor
from src.video_load import iter_original_frame_segments_from_video


logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "reliability_satellite_full"
DEFAULT_RESULT_DIR = Path(RESULT_DIR) / "experiment_reliability"
MODEL_NAME_OR_PATH = "facebook/VGGT-1B"
DEVICE = None
SATELLITE_FRAME_ID_BASE = -10_000


def run_reliability_experiment(
    satellite_image_path: str | Path,
    plane_image_path: str | Path,
    plane_image_count: int,
) -> dict[str, Path]:
    """Run VGGT on one satellite image plus selected ground frames."""
    configure_logging()
    satellite_image_path = Path(satellite_image_path)
    plane_image_path = Path(plane_image_path)
    task_dir = _create_task_dir(DEFAULT_RESULT_DIR, EXPERIMENT_NAME)
    configure_task_file_logging(task_dir)

    with ResourceStepMonitor("prepare_frames"):
        satellite_frame = load_satellite_frame(satellite_image_path, frame_id=SATELLITE_FRAME_ID_BASE)
        ground_frames = load_preprocessed_keyframes_from_video(
            plane_image_path,
            task_dir / "ground_preprocessed_frames",
            plane_image_count=plane_image_count,
        )
        if not ground_frames:
            raise ValueError(f"no ground key frames selected from video: {plane_image_path}")
        chunk = Chunk(id=0, frames=[satellite_frame, *ground_frames])
        chunk_path = _save_chunk(chunk, task_dir / "0.pkl")

    logger.info(
        "实验输入准备完成: satellite=%s, video=%s, ground_frames=%d, task_dir=%s",
        satellite_image_path,
        plane_image_path,
        len(ground_frames),
        task_dir,
    )

    with ResourceStepMonitor("vggt_feature_similarity"):
        model = load_vggt_model(model_name_or_path=MODEL_NAME_OR_PATH, device=DEVICE)
        similarity_path = save_feature_similarity_predictions(
            chunk=chunk,
            output_path=task_dir / "feature_similarity.npz",
            model=model,
            satellite_frame_count=1,
        )

    with ResourceStepMonitor("export_satellite_similarity_heatmap"):
        satellite_shape = satellite_frame.image.shape[:2]
        heatmap_path = export_similarity_heatmap_from_npz(
            similarity_path,
            frame_indexes=[0],
            output_path=task_dir / "satellite_similarity_heatmap.png",
            source_shapes=[satellite_shape],
        )
        top_similarity_path = export_top_similarity_points_from_npz(
            similarity_path,
            frame_indexes=[0],
            output_path=task_dir / "satellite_similarity_top10.png",
            base_image=satellite_frame.image,
            source_shapes=[satellite_shape],
        )

    return {
        "task_dir": task_dir,
        "chunk": chunk_path,
        "npz": similarity_path,
        "heatmap": heatmap_path,
        "top_similarity": top_similarity_path,
    }


def load_satellite_frame(image_path: str | Path, frame_id: int) -> BaseFrame:
    """Read a satellite image without illumination/key-frame preprocessing."""
    image = read_rgb_image(image_path)
    return BaseFrame(id=frame_id, image=image, gps_location=None)


def read_rgb_image(image_path: str | Path) -> NDArray[np.uint8]:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"satellite image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    raise ValueError(f"unsupported image channel count: {image.shape[2]}")


def load_preprocessed_keyframes_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    plane_image_count: int,
) -> list[BaseFrame]:
    """Load video frames, run size/light preprocessing, then select key frames."""
    if plane_image_count <= 0:
        raise ValueError(f"plane_image_count must be greater than 0, got {plane_image_count}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_list = []
    previous_image = None
    previous_keyframe_image = None

    for original_frames in iter_original_frame_segments_from_video(video_path):
        for frame in original_frames.frames:
            preprocessed_frame = preprocess_original_frame(frame)
            image_path = _save_rgb_image(
                preprocessed_frame.image,
                output_dir / f"{preprocessed_frame.id:08d}.png",
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

    if not metadata_list:
        raise ValueError(f"no frames read from video: {video_path}")

    selected_metadata = select_key_frame_metadata(
        metadata_list,
        delete_ratio=max(0.0, 1.0 - plane_image_count / len(metadata_list)),
    )
    selected_metadata = _select_exact_frame_count(
        selected_metadata,
        metadata_list,
        plane_image_count,
        video_path,
    )

    frames = []
    for metadata in selected_metadata:
        frames.append(
            BaseFrame(
                id=metadata.frame_id,
                image=read_rgb_image(metadata.image_path),
                gps_location=metadata.gps_location,
            )
        )
    return frames


def _select_exact_frame_count(
    metadata_list,
    fallback_metadata_list,
    plane_image_count: int,
    video_path: str | Path,
):
    if len(fallback_metadata_list) < plane_image_count:
        raise ValueError(
            f"video frames from {video_path} are fewer than plane_image_count: "
            f"{len(fallback_metadata_list)} < {plane_image_count}"
        )
    if len(metadata_list) < plane_image_count:
        metadata_list = fallback_metadata_list
    if len(metadata_list) < plane_image_count:
        raise ValueError(
            f"selected key frames from {video_path} are fewer than plane_image_count: "
            f"{len(metadata_list)} < {plane_image_count}"
        )
    if len(metadata_list) == plane_image_count:
        return metadata_list

    indexes = np.linspace(0, len(metadata_list) - 1, plane_image_count, dtype=int)
    return [metadata_list[int(index)] for index in indexes]


def save_feature_similarity_predictions(
    chunk: Chunk,
    output_path: str | Path,
    model: torch.nn.Module,
    satellite_frame_count: int,
    temperature: float = 0.07,
) -> Path:
    """Run VGGT aggregator features and save satellite-to-drone softmax maps."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_maps, cosine_similarity = compute_feature_similarity_maps(
        chunk,
        model,
        satellite_frame_count=satellite_frame_count,
        temperature=temperature,
    )
    np.savez(
        output_path,
        similarity_maps=similarity_maps,
        cosine_similarity=cosine_similarity,
        satellite_frame_count=np.array(satellite_frame_count, dtype=np.int64),
        frame_ids=np.array(chunk.frame_ids, dtype=np.int64),
        temperature=np.array(temperature, dtype=np.float32),
    )
    return output_path


def compute_feature_similarity_maps(
    chunk: Chunk,
    model: torch.nn.Module,
    satellite_frame_count: int,
    temperature: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if satellite_frame_count <= 0:
        raise ValueError(f"satellite_frame_count must be greater than 0, got {satellite_frame_count}")
    if satellite_frame_count >= len(chunk.frames):
        raise ValueError("chunk must contain at least one drone frame after satellite frames")

    device = next(model.parameters()).device
    model.eval()
    images = chunk_to_vggt_images(chunk).unsqueeze(0).to(device)
    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = model.aggregator(images)

    tokens = next(tokens for tokens in reversed(aggregated_tokens_list) if tokens is not None)
    patch_tokens = tokens[:, :, patch_start_idx:]
    batch_size, frame_count, patch_count, channel_count = patch_tokens.shape
    if batch_size != 1:
        raise ValueError(f"expected batch size 1, got {batch_size}")

    patch_size = model.aggregator.patch_size
    patch_height = images.shape[-2] // patch_size
    patch_width = images.shape[-1] // patch_size
    expected_patch_count = patch_height * patch_width
    if patch_count != expected_patch_count:
        raise ValueError(
            f"patch token count {patch_count} does not match feature grid "
            f"{patch_height}x{patch_width}"
        )

    features = F.normalize(patch_tokens.squeeze(0), dim=-1)
    satellite_features = features[:satellite_frame_count]
    drone_features = features[satellite_frame_count:].reshape(-1, channel_count)

    similarity_maps = []
    cosine_matrices = []
    for satellite_feature in satellite_features:
        cosine_matrix = satellite_feature @ drone_features.T
        softmax_matrix = F.softmax(cosine_matrix / temperature, dim=0)
        similarity_map = softmax_matrix.mean(dim=1).reshape(patch_height, patch_width)
        similarity_maps.append(similarity_map.detach().cpu().numpy().astype(np.float32))
        cosine_matrices.append(cosine_matrix.detach().cpu().numpy().astype(np.float32))

    return np.stack(similarity_maps), np.stack(cosine_matrices)


def export_similarity_heatmap_from_npz(
    npz_path: str | Path,
    frame_indexes: list[int],
    output_path: str | Path,
    source_shapes: list[tuple[int, int]] | None = None,
) -> Path:
    """Export selected feature softmax maps as one heatmap image."""
    similarity = load_similarity_map_from_npz(
        npz_path,
        frame_indexes=frame_indexes,
        source_shapes=source_shapes,
    )
    heatmap = score_to_heatmap(similarity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), heatmap):
        raise ValueError(f"could not write heatmap: {output_path}")
    return output_path


def export_top_similarity_points_from_npz(
    npz_path: str | Path,
    frame_indexes: list[int],
    output_path: str | Path,
    base_image: NDArray[np.uint8],
    source_shapes: list[tuple[int, int]] | None = None,
) -> Path:
    """Export an image marking the highest 10 percent similarity points."""
    similarity = load_similarity_map_from_npz(
        npz_path,
        frame_indexes=frame_indexes,
        source_shapes=source_shapes,
    )
    top_image = top_score_points_to_image(similarity, base_image=base_image)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), top_image):
        raise ValueError(f"could not write top similarity image: {output_path}")
    return output_path


def load_similarity_map_from_npz(
    npz_path: str | Path,
    frame_indexes: list[int],
    source_shapes: list[tuple[int, int]] | None = None,
) -> NDArray[np.float32]:
    npz_path = Path(npz_path)
    with np.load(npz_path) as loaded:
        if "similarity_maps" not in loaded.files:
            raise KeyError(f"{npz_path} is missing similarity_maps")
        similarity_maps = np.asarray(loaded["similarity_maps"])

    maps = [np.asarray(similarity_maps[index]).squeeze().astype(np.float32) for index in frame_indexes]
    if source_shapes is not None:
        maps = [
            cv2.resize(conf_map, (width, height), interpolation=cv2.INTER_CUBIC)
            for conf_map, (height, width) in zip(maps, source_shapes)
        ]
    return maps[0] if len(maps) == 1 else np.concatenate(maps, axis=1)


def score_to_heatmap(score: NDArray[np.floating]) -> NDArray[np.uint8]:
    score = np.asarray(score, dtype=np.float32)
    finite_mask = np.isfinite(score)
    if not np.any(finite_mask):
        normalized = np.zeros(score.shape, dtype=np.uint8)
    else:
        values = score[finite_mask]
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value == min_value:
            normalized = np.full(score.shape, 255, dtype=np.uint8)
        else:
            scaled = (score - min_value) / (max_value - min_value)
            normalized = (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
        normalized[~finite_mask] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def top_score_points_to_image(
    score: NDArray[np.floating],
    base_image: NDArray[np.uint8],
    top_ratio: float = 0.10,
) -> NDArray[np.uint8]:
    score = np.asarray(score, dtype=np.float32)
    base_image = _prepare_base_image_for_output(base_image, score.shape)
    finite_mask = np.isfinite(score)
    top_mask = np.zeros(score.shape, dtype=np.uint8)
    finite_values = score[finite_mask]
    if finite_values.size:
        keep_count = max(1, int(np.ceil(finite_values.size * top_ratio)))
        keep_indexes = np.argpartition(finite_values, -keep_count)[-keep_count:]
        flat_finite_mask = np.flatnonzero(finite_mask.reshape(-1))
        top_mask.reshape(-1)[flat_finite_mask[keep_indexes]] = 255

    image = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
    top_mask = cv2.dilate(top_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    image[top_mask > 0] = (0, 0, 255)
    return image


def _prepare_base_image_for_output(
    base_image: NDArray[np.uint8],
    output_shape: tuple[int, int],
) -> NDArray[np.uint8]:
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    elif base_image.shape[2] == 4:
        base_image = base_image[:, :, :3]
    if base_image.shape[:2] != output_shape:
        base_image = cv2.resize(
            base_image,
            (output_shape[1], output_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(base_image)


def _save_rgb_image(image: NDArray[np.uint8], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise ValueError(f"could not write image: {output_path}")
    return output_path


def _save_chunk(chunk: Chunk, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(chunk, file)
    return output_path


def _create_task_dir(result_dir: str | Path, experiment_name: str) -> Path:
    task_dir = Path(result_dir) / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{experiment_name}"
    task_dir.mkdir(parents=True, exist_ok=False)
    return task_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test VGGT backbone feature similarity on a satellite image."
    )
    parser.add_argument("satellite_image_path", help="Satellite image path.")
    parser.add_argument("plane_image_path", help="Ground-view video path.")
    parser.add_argument(
        "plane_image_count",
        type=int,
        help="Number of ground-view frames selected from the whole video and sent to VGGT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_reliability_experiment(
        args.satellite_image_path,
        args.plane_image_path,
        args.plane_image_count,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
