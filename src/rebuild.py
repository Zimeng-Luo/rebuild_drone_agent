from __future__ import annotations

import logging
from pathlib import Path
import pickle
import struct
import sys
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import trimesh

from config import configure_huggingface_endpoint
from src.datatype import Chunk, ChunkToProcess, FrameInChunk


VGGT_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "utils" / "vggt"
VGGT_TARGET_SIZE = 518
SKY_CLASS_ID = 2
VGGT_GLB_KEYS = [
    "pose_enc",
    "depth",
    "depth_conf",
    "images",
    "extrinsic",
    "intrinsic",
    "world_points_from_depth",
]
NPZ_LOAD_KEYS = [
    "world_points_from_depth",
    "depth_conf",
    "images",
    "chunk_id",
    "frame_ids",
    "gps_locations",
]


logger = logging.getLogger(__name__)


def load_vggt_model(
    model_name_or_path: str = "facebook/VGGT-1B",
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Load a local VGGT model with only the heads used by this pipeline."""
    configure_huggingface_endpoint()
    _ensure_vggt_reference_importable()

    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(
        model_name_or_path,
        enable_camera=True,
        enable_depth=True,
    )
    model = model.to(_resolve_device(device))
    model.eval()
    return model


def rebuild_chunk_to_npz(
    chunk: Chunk,
    output_path: str | Path,
    model: torch.nn.Module | None = None,
    model_name_or_path: str = "facebook/VGGT-1B",
    device: str | torch.device | None = None,
) -> Path:
    """Run VGGT reconstruction for a Chunk and save predictions to an npz file."""
    model = model if model is not None else load_vggt_model(model_name_or_path, device)
    predictions = rebuild_chunk(chunk, model, device=device)
    return save_rebuild_predictions(predictions, output_path)


def load_chunk(chunk_path: str | Path) -> Chunk:
    """Load a Chunk object from a pickle file."""
    chunk_path = Path(chunk_path)
    with chunk_path.open("rb") as file:
        chunk = pickle.load(file)

    if not isinstance(chunk, Chunk):
        raise TypeError(f"{chunk_path} does not contain a Chunk object")

    logger.info(
        "分块重新读取完成: path=%s, chunk_id=%s, frame_count=%d",
        chunk_path,
        chunk.id,
        len(chunk.frames),
    )
    return chunk


def rebuild_chunk(
    chunk: Chunk,
    model: torch.nn.Module,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run VGGT reconstruction for a Chunk and return a VGGT-style prediction dict."""
    if not chunk.frames:
        raise ValueError(f"chunk {chunk.id} has no frames")

    _ensure_vggt_reference_importable()
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    images = chunk_to_vggt_images(chunk).to(resolved_device)
    dtype = _select_autocast_dtype(resolved_device)

    with torch.no_grad():
        with _autocast_context(resolved_device, dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        images.shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    np_predictions = _tensor_predictions_to_numpy(predictions)
    np_predictions["pose_enc_list"] = None
    np_predictions["images"] = images.detach().cpu().numpy()

    depth_map = np_predictions["depth"]
    np_predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map,
        np_predictions["extrinsic"],
        np_predictions["intrinsic"],
    ).astype(np.float32)
    np_predictions["chunk_id"] = np.array(chunk.id)
    np_predictions["frame_ids"] = np.array(chunk.frame_ids, dtype=np.int64)
    np_predictions["gps_locations"] = _chunk_gps_locations(chunk)
    np_predictions["masks"] = chunk_to_vggt_masks(chunk).numpy().astype(np.uint8, copy=False)
    np_predictions["point_labels"] = np_predictions["masks"]

    return np_predictions


def save_rebuild_predictions(
    predictions: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save VGGT-style reconstruction predictions to npz."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **predictions)
    logger.info(
        "分块重建完成: chunk_id=%s, output_path=%s",
        np.array(predictions["chunk_id"]).item(),
        output_path,
    )
    return output_path


def npz_load(npz_path: str | Path) -> ChunkToProcess:
    """Load reconstruction npz data into a ChunkToProcess object."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as loaded:
        missing_keys = [key for key in NPZ_LOAD_KEYS if key not in loaded.files]
        if missing_keys:
            raise KeyError(f"{npz_path} is missing ChunkToProcess keys: {missing_keys}")

        world_points_from_depth = np.array(loaded["world_points_from_depth"])
        depth_conf = np.array(loaded["depth_conf"])
        images = np.array(loaded["images"])
        masks = np.array(loaded["masks"]) if "masks" in loaded.files else None
        chunk_id = int(np.array(loaded["chunk_id"]).item())
        frame_ids = np.array(loaded["frame_ids"])
        gps_locations = np.array(loaded["gps_locations"])

    frame_count = len(frame_ids)
    _validate_npz_load_frame_count(
        npz_path,
        frame_count,
        world_points_from_depth=world_points_from_depth,
        depth_conf=depth_conf,
        images=images,
        gps_locations=gps_locations,
    )
    if masks is not None:
        _validate_npz_load_frame_count(npz_path, frame_count, masks=masks)

    chunk_to_process = ChunkToProcess(chunk_id=chunk_id, frames=[], frame_ids=[])
    for frame_index, frame_id in enumerate(frame_ids):
        frame = FrameInChunk(
            id=int(frame_id),
            image=images[frame_index],
            gps_location=tuple(gps_locations[frame_index]),
            mask=None if masks is None else masks[frame_index],
            chunk_id=chunk_id,
            world_points=world_points_from_depth[frame_index],
            world_points_conf=depth_conf[frame_index],
        )
        chunk_to_process.add_frame(frame)

    return chunk_to_process


def rebuild_npz_to_glb(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
) -> Path:
    """Convert a VGGT-style npz prediction file to GLB."""
    npz_path = Path(npz_path)
    if output_path is None:
        output_path = npz_path.with_suffix(".glb")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = load_rebuild_predictions_for_glb(npz_path)
    glb_scene = rebuild_predictions_to_glb_scene(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        prediction_mode=prediction_mode,
    )
    glb_scene.export(file_obj=str(output_path))
    return output_path


def rebuild_npz_to_glb_streaming(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = False,
    mask_sky: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
) -> Path:
    """Convert npz point data to GLB with streaming reads and writes."""
    if mask_sky:
        raise NotImplementedError("mask_sky requires the reference VGGT sky segmentation assets")
    if show_cam:
        logger.warning("流式GLB导出不支持相机可视化，将只导出点云")

    npz_path = Path(npz_path)
    if output_path is None:
        output_path = npz_path.with_suffix(".glb")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _scan_npz_points_for_export(
        npz_path,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        prediction_mode=prediction_mode,
    )
    _write_streaming_glb(
        npz_path,
        output_path,
        stats,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        prediction_mode=prediction_mode,
    )
    logger.info(
        "GLB流式导出完成: npz_path=%s, output_path=%s, output_points=%d, conf_percentile=%.3f, conf_threshold=%.6g",
        npz_path,
        output_path,
        stats["point_count"],
        conf_thres,
        stats["threshold"],
    )
    return output_path


def rebuild_npz_to_ply_streaming(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    conf_thres: float = 25.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    mask_sky: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
) -> Path:
    """Convert npz point data to binary PLY with streaming reads and writes."""
    if mask_sky:
        raise NotImplementedError("mask_sky requires the reference VGGT sky segmentation assets")

    npz_path = Path(npz_path)
    if output_path is None:
        output_path = npz_path.with_suffix(".ply")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _scan_npz_points_for_export(
        npz_path,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        prediction_mode=prediction_mode,
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {stats['point_count']}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex_dtype = _export_vertex_dtype()
    with output_path.open("wb") as file:
        file.write(header)
        for vertices, colors in _iter_npz_export_points(
            npz_path,
            threshold=stats["threshold"],
            filter_by_frames=filter_by_frames,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            prediction_mode=prediction_mode,
        ):
            records = np.empty(vertices.shape[0], dtype=vertex_dtype)
            records["x"] = vertices[:, 0].astype("<f4", copy=False)
            records["y"] = vertices[:, 1].astype("<f4", copy=False)
            records["z"] = vertices[:, 2].astype("<f4", copy=False)
            records["red"] = colors[:, 0]
            records["green"] = colors[:, 1]
            records["blue"] = colors[:, 2]
            records.tofile(file)

    logger.info(
        "PLY流式导出完成: npz_path=%s, output_path=%s, output_points=%d, conf_percentile=%.3f, conf_threshold=%.6g",
        npz_path,
        output_path,
        stats["point_count"],
        conf_thres,
        stats["threshold"],
    )
    return output_path


def load_rebuild_predictions_for_glb(npz_path: str | Path) -> dict[str, NDArray[Any]]:
    """Load the subset of VGGT predictions required for GLB conversion."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as loaded:
        missing_keys = [key for key in VGGT_GLB_KEYS if key not in loaded.files]
        if missing_keys:
            raise KeyError(f"{npz_path} is missing GLB prediction keys: {missing_keys}")
        predictions = {key: np.array(loaded[key]) for key in VGGT_GLB_KEYS}
        if "masks" in loaded.files:
            predictions["masks"] = np.array(loaded["masks"])
        if "point_labels" in loaded.files:
            predictions["point_labels"] = np.array(loaded["point_labels"])
        return predictions


def _validate_npz_load_frame_count(
    npz_path: Path,
    frame_count: int,
    **frame_arrays: NDArray[Any],
) -> None:
    for key, value in frame_arrays.items():
        if value.shape[0] != frame_count:
            raise ValueError(
                f"{npz_path} has {frame_count} frame_ids but {key} has first dimension {value.shape[0]}"
            )


def rebuild_predictions_to_glb_scene(
    predictions: dict[str, NDArray[Any]],
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
) -> trimesh.Scene:
    """Build a previewable GLB scene from VGGT-style predictions."""
    if mask_sky:
        raise NotImplementedError("mask_sky requires the reference VGGT sky segmentation assets")

    world_points, confidence = _select_glb_points_and_confidence(
        predictions,
        prediction_mode,
    )
    images = _images_to_nhwc(predictions["images"])
    remove_masks = _optional_masks_to_nhwc(predictions.get("masks"))

    world_points, confidence, images = _select_glb_frame(
        world_points,
        confidence,
        images,
        filter_by_frames,
    )
    if remove_masks is not None:
        remove_masks = _select_glb_optional_frame(remove_masks, filter_by_frames)

    vertices = world_points.reshape(-1, 3)
    colors = (np.clip(images.reshape(-1, 3), 0.0, 1.0) * 255.0).astype(np.uint8)
    confidence = confidence.reshape(-1)

    candidate_mask = _build_glb_candidate_mask(
        confidence,
        colors,
        mask_black_bg,
        mask_white_bg,
    )
    removed_by_input_mask = 0
    if remove_masks is not None:
        remove_mask = _resize_remove_masks_for_points(
            remove_masks,
            world_points.shape[:3],
        ).reshape(-1)
        removed_by_input_mask = int(np.count_nonzero(candidate_mask & remove_mask))
        candidate_mask = candidate_mask & ~remove_mask

    mask, confidence_threshold = _build_glb_point_mask(
        confidence,
        conf_thres,
        candidate_mask,
    )
    vertices = vertices[mask]
    colors = colors[mask]

    logger.info(
        "GLB点云过滤完成: total_points=%d, candidate_points=%d, removed_by_input_mask=%d, "
        "conf_percentile=%.3f, conf_threshold=%.6g, output_points=%d",
        confidence.size,
        int(np.count_nonzero(candidate_mask)),
        removed_by_input_mask,
        conf_thres,
        confidence_threshold,
        vertices.shape[0],
    )

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    if show_cam:
        camera_meshes = _build_camera_markers(predictions["extrinsic"], vertices)
        for camera_mesh in camera_meshes:
            scene.add_geometry(camera_mesh)

    return scene


def _select_glb_points_and_confidence(
    predictions: dict[str, NDArray[Any]],
    prediction_mode: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if "Pointmap" in prediction_mode:
        raise ValueError(
            "Predicted Pointmap is disabled in the local VGGT model; use "
            "'Depthmap and Camera Branch'."
        )
    points = predictions["world_points_from_depth"]
    confidence = predictions["depth_conf"]
    return points, confidence


def _images_to_nhwc(images: NDArray[np.floating]) -> NDArray[np.floating]:
    if images.ndim != 4:
        raise ValueError(f"images must have shape [S, C, H, W] or [S, H, W, C], got {images.shape}")
    if images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    if images.shape[-1] == 3:
        return images
    raise ValueError(f"images must have 3 color channels, got {images.shape}")


def _optional_masks_to_nhwc(masks: Any) -> NDArray[Any] | None:
    if masks is None:
        return None

    mask_array = np.asarray(masks)
    if mask_array.ndim == 3:
        return mask_array[:, :, :, None]
    if mask_array.ndim != 4:
        raise ValueError(f"masks must have shape [S,H,W], [S,C,H,W], or [S,H,W,C], got {mask_array.shape}")
    if mask_array.shape[1] in (1, 3, 4) and mask_array.shape[-1] not in (1, 3, 4):
        return np.transpose(mask_array, (0, 2, 3, 1))
    return mask_array


def _select_glb_frame(
    points: NDArray[np.floating],
    confidence: NDArray[np.floating],
    images: NDArray[np.floating],
    filter_by_frames: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    if filter_by_frames in ("all", "All"):
        return points, confidence, images

    frame_index = int(filter_by_frames.split(":")[0])
    return points[frame_index][None], confidence[frame_index][None], images[frame_index][None]


def _select_glb_optional_frame(
    values: NDArray[Any],
    filter_by_frames: str,
) -> NDArray[Any]:
    if filter_by_frames in ("all", "All"):
        return values

    frame_index = int(filter_by_frames.split(":")[0])
    return values[frame_index][None]


def _resize_remove_masks_for_points(
    masks: NDArray[Any],
    point_shape: tuple[int, int, int],
) -> NDArray[np.bool_]:
    frame_count, point_height, point_width = point_shape
    if masks.shape[0] != frame_count:
        raise ValueError(
            f"masks frame count {masks.shape[0]} does not match points frame count {frame_count}"
        )

    remove_masks = []
    for mask in masks:
        mask_2d = _class_mask_to_sky_mask(mask)
        if mask_2d.shape != (point_height, point_width):
            mask_2d = cv2.resize(
                mask_2d.astype(np.uint8),
                (point_width, point_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        remove_masks.append(mask_2d.astype(bool))

    return np.stack(remove_masks)


def _build_glb_candidate_mask(
    confidence: NDArray[np.floating],
    colors: NDArray[np.uint8],
    mask_black_bg: bool,
    mask_white_bg: bool,
) -> NDArray[np.bool_]:
    mask = np.isfinite(confidence) & (confidence > 1e-5)

    if mask_black_bg:
        mask = mask & (colors.sum(axis=1) >= 16)

    if mask_white_bg:
        white_mask = (colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240)
        mask = mask & ~white_mask

    return mask


def _build_glb_point_mask(
    confidence: NDArray[np.floating],
    conf_thres: float,
    candidate_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], float]:
    if not 0.0 <= conf_thres <= 100.0:
        raise ValueError(f"conf_thres must be in [0, 100], got {conf_thres}")

    candidate_confidence = confidence[candidate_mask]
    if candidate_confidence.size == 0:
        return np.zeros(confidence.shape, dtype=bool), float("nan")

    if conf_thres == 0.0:
        threshold = 0.0
    else:
        threshold = float(np.percentile(candidate_confidence, conf_thres))

    return candidate_mask & (confidence >= threshold), threshold


def _build_camera_markers(
    extrinsic: NDArray[np.floating],
    points: NDArray[np.floating],
) -> list[trimesh.Trimesh]:
    if points.size == 0:
        scene_scale = 1.0
    else:
        lower = np.percentile(points, 5, axis=0)
        upper = np.percentile(points, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper - lower))
        if scene_scale <= 0:
            scene_scale = 1.0

    marker_radius = scene_scale * 0.01
    markers = []
    for camera_extrinsic in extrinsic:
        camera_to_world = np.linalg.inv(_to_homogeneous_extrinsic(camera_extrinsic))
        marker = trimesh.creation.icosphere(subdivisions=1, radius=marker_radius)
        marker.apply_translation(camera_to_world[:3, 3])
        marker.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        markers.append(marker)
    return markers


def _scan_npz_points_for_export(
    npz_path: Path,
    conf_thres: float,
    filter_by_frames: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    prediction_mode: str,
) -> dict[str, Any]:
    if not 0.0 <= conf_thres <= 100.0:
        raise ValueError(f"conf_thres must be in [0, 100], got {conf_thres}")

    confidence_chunks = []
    candidate_points = 0
    total_points = 0
    for confidence, _, candidate_mask in _iter_npz_export_candidates(
        npz_path,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        prediction_mode=prediction_mode,
    ):
        total_points += int(confidence.size)
        candidate_points += int(np.count_nonzero(candidate_mask))
        if np.any(candidate_mask):
            confidence_chunks.append(confidence[candidate_mask])

    if confidence_chunks:
        candidate_confidence = np.concatenate(confidence_chunks)
        threshold = 0.0 if conf_thres == 0.0 else float(np.percentile(candidate_confidence, conf_thres))
    else:
        threshold = float("nan")

    point_count = 0
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    for vertices, _ in _iter_npz_export_points(
        npz_path,
        threshold=threshold,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        prediction_mode=prediction_mode,
    ):
        point_count += int(vertices.shape[0])
        bounds_min = np.minimum(bounds_min, np.min(vertices, axis=0))
        bounds_max = np.maximum(bounds_max, np.max(vertices, axis=0))

    if point_count == 0:
        bounds_min = np.zeros(3, dtype=np.float64)
        bounds_max = np.zeros(3, dtype=np.float64)

    logger.info(
        "点云导出过滤完成: total_points=%d, candidate_points=%d, conf_percentile=%.3f, "
        "conf_threshold=%.6g, output_points=%d",
        total_points,
        candidate_points,
        conf_thres,
        threshold,
        point_count,
    )
    return {
        "threshold": threshold,
        "point_count": point_count,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }


def _iter_npz_export_candidates(
    npz_path: Path,
    filter_by_frames: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    prediction_mode: str,
):
    with np.load(npz_path, mmap_mode="r") as loaded:
        points_key, confidence_key = _export_points_and_confidence_keys(prediction_mode)
        missing_keys = [key for key in (points_key, confidence_key, "images") if key not in loaded.files]
        if missing_keys:
            raise KeyError(f"{npz_path} is missing export keys: {missing_keys}")
        frame_indexes = _export_frame_indexes(loaded[points_key].shape[0], filter_by_frames)
        for frame_index in frame_indexes:
            points = np.asarray(loaded[points_key][frame_index])
            confidence = np.asarray(loaded[confidence_key][frame_index]).reshape(-1)
            colors = _frame_export_colors(np.asarray(loaded["images"][frame_index]), points.shape[:2])
            candidate_mask = _build_glb_candidate_mask(
                confidence,
                colors,
                mask_black_bg,
                mask_white_bg,
            )
            if "masks" in loaded.files:
                remove_mask = _frame_remove_mask_for_points(
                    np.asarray(loaded["masks"][frame_index]),
                    points.shape[:2],
                ).reshape(-1)
                candidate_mask = candidate_mask & ~remove_mask
            yield confidence, colors, candidate_mask


def _iter_npz_export_points(
    npz_path: Path,
    threshold: float,
    filter_by_frames: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    prediction_mode: str,
):
    with np.load(npz_path, mmap_mode="r") as loaded:
        points_key, confidence_key = _export_points_and_confidence_keys(prediction_mode)
        frame_indexes = _export_frame_indexes(loaded[points_key].shape[0], filter_by_frames)
        for frame_index in frame_indexes:
            points = np.asarray(loaded[points_key][frame_index])
            vertices = points.reshape(-1, 3)
            confidence = np.asarray(loaded[confidence_key][frame_index]).reshape(-1)
            colors = _frame_export_colors(np.asarray(loaded["images"][frame_index]), points.shape[:2])
            candidate_mask = _build_glb_candidate_mask(
                confidence,
                colors,
                mask_black_bg,
                mask_white_bg,
            )
            if "masks" in loaded.files:
                remove_mask = _frame_remove_mask_for_points(
                    np.asarray(loaded["masks"][frame_index]),
                    points.shape[:2],
                ).reshape(-1)
                candidate_mask = candidate_mask & ~remove_mask
            if np.isnan(threshold):
                mask = np.zeros(confidence.shape, dtype=bool)
            else:
                mask = candidate_mask & (confidence >= threshold)
            if np.any(mask):
                yield vertices[mask].astype(np.float32, copy=False), colors[mask]


def _write_streaming_glb(
    npz_path: Path,
    output_path: Path,
    stats: dict[str, Any],
    filter_by_frames: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    prediction_mode: str,
) -> None:
    vertex_count = int(stats["point_count"])
    vertex_stride = 16
    bin_length = vertex_count * vertex_stride
    json_chunk = _build_glb_json_chunk(
        vertex_count,
        vertex_stride,
        bin_length,
        stats["bounds_min"],
        stats["bounds_max"],
    )
    json_padding = _align4(len(json_chunk)) - len(json_chunk)
    total_length = 12 + 8 + len(json_chunk) + json_padding + 8 + bin_length

    with output_path.open("wb") as file:
        file.write(struct.pack("<4sII", b"glTF", 2, total_length))
        file.write(struct.pack("<I4s", len(json_chunk) + json_padding, b"JSON"))
        file.write(json_chunk)
        file.write(b" " * json_padding)
        file.write(struct.pack("<I4s", bin_length, b"BIN\x00"))

        written_positions = 0
        vertex_dtype = _glb_vertex_dtype()
        for vertices, colors in _iter_npz_export_points(
            npz_path,
            threshold=stats["threshold"],
            filter_by_frames=filter_by_frames,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            prediction_mode=prediction_mode,
        ):
            records = np.zeros(vertices.shape[0], dtype=vertex_dtype)
            records["x"] = vertices[:, 0].astype("<f4", copy=False)
            records["y"] = vertices[:, 1].astype("<f4", copy=False)
            records["z"] = vertices[:, 2].astype("<f4", copy=False)
            records["red"] = colors[:, 0]
            records["green"] = colors[:, 1]
            records["blue"] = colors[:, 2]
            records.tofile(file)
            written_positions += vertices.shape[0]
        if written_positions != vertex_count:
            raise ValueError(f"expected {vertex_count} GLB vertices, wrote {written_positions}")


def _build_glb_json_chunk(
    vertex_count: int,
    vertex_stride: int,
    bin_length: int,
    bounds_min: NDArray[np.float64],
    bounds_max: NDArray[np.float64],
) -> bytes:
    import json

    payload = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "mode": 0,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": bin_length}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": bin_length, "byteStride": vertex_stride, "target": 34962},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
                "min": bounds_min.astype(float).tolist(),
                "max": bounds_max.astype(float).tolist(),
            },
            {
                "bufferView": 0,
                "byteOffset": 12,
                "componentType": 5121,
                "count": vertex_count,
                "type": "VEC3",
                "normalized": True,
            },
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _export_points_and_confidence_keys(prediction_mode: str) -> tuple[str, str]:
    if "Pointmap" in prediction_mode:
        raise ValueError(
            "Predicted Pointmap is disabled in the local VGGT model; use "
            "'Depthmap and Camera Branch'."
        )
    return "world_points_from_depth", "depth_conf"


def _export_frame_indexes(frame_count: int, filter_by_frames: str) -> list[int]:
    if filter_by_frames in ("all", "All"):
        return list(range(frame_count))
    frame_index = int(filter_by_frames.split(":")[0])
    if frame_index < 0 or frame_index >= frame_count:
        raise IndexError(f"frame index {frame_index} is out of range for {frame_count} frames")
    return [frame_index]


def _frame_export_colors(image: NDArray[Any], point_shape: tuple[int, int]) -> NDArray[np.uint8]:
    image_nhwc = _single_image_to_hwc(image)
    if image_nhwc.shape[:2] != point_shape:
        image_nhwc = cv2.resize(
            image_nhwc,
            (point_shape[1], point_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return (np.clip(image_nhwc.reshape(-1, 3), 0.0, 1.0) * 255.0).astype(np.uint8)


def _single_image_to_hwc(image: NDArray[Any]) -> NDArray[Any]:
    if image.ndim != 3:
        raise ValueError(f"image must have shape [C,H,W] or [H,W,C], got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        return np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 3:
        return image
    raise ValueError(f"image must have 3 color channels, got {image.shape}")


def _frame_remove_mask_for_points(mask: NDArray[Any], point_shape: tuple[int, int]) -> NDArray[np.bool_]:
    mask_hwc = mask
    if mask_hwc.ndim == 3 and mask_hwc.shape[0] in (1, 3, 4) and mask_hwc.shape[-1] not in (1, 3, 4):
        mask_hwc = np.transpose(mask_hwc, (1, 2, 0))
    mask_2d = _class_mask_to_sky_mask(mask_hwc)
    if mask_2d.shape != point_shape:
        mask_2d = cv2.resize(
            mask_2d.astype(np.uint8),
            (point_shape[1], point_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask_2d


def _class_mask_to_sky_mask(mask: NDArray[Any]) -> NDArray[np.bool_]:
    if mask.ndim == 2:
        return mask == SKY_CLASS_ID
    if mask.ndim == 3:
        return np.any(mask == SKY_CLASS_ID, axis=-1)
    raise ValueError(f"mask must have shape [H,W], [C,H,W], or [H,W,C], got {mask.shape}")


def _export_vertex_dtype() -> np.dtype[Any]:
    return np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )


def _glb_vertex_dtype() -> np.dtype[Any]:
    return np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("pad", "u1"),
        ]
    )


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _to_homogeneous_extrinsic(extrinsic: NDArray[np.floating]) -> NDArray[np.float64]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = extrinsic
    return matrix


def chunk_to_vggt_images(chunk: Chunk, target_size: int = VGGT_TARGET_SIZE) -> torch.Tensor:
    """Convert Chunk frames to a VGGT input tensor with shape [S, 3, H, W]."""
    if not chunk.frames:
        raise ValueError(f"chunk {chunk.id} has no frames")

    tensors = []
    for frame in chunk.frames:
        image = _prepare_frame_image(frame.image, target_size)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)

    return torch.stack(tensors)


def chunk_to_vggt_masks(chunk: Chunk, target_size: int = VGGT_TARGET_SIZE) -> torch.Tensor:
    """Convert Chunk frame class-label masks to VGGT output resolution."""
    if not chunk.frames:
        raise ValueError(f"chunk {chunk.id} has no frames")

    tensors = []
    for frame in chunk.frames:
        mask = _prepare_frame_mask(frame.mask, target_size)
        tensor = torch.from_numpy(mask)
        tensors.append(tensor)

    return torch.stack(tensors)


def _ensure_vggt_reference_importable() -> None:
    if VGGT_REFERENCE_DIR.exists():
        vggt_reference_path = str(VGGT_REFERENCE_DIR)
        if vggt_reference_path not in sys.path:
            sys.path.insert(0, vggt_reference_path)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=False)


def _tensor_predictions_to_numpy(predictions: dict[str, Any]) -> dict[str, Any]:
    np_predictions: dict[str, Any] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            np_predictions[key] = value.detach().cpu().numpy().squeeze(0)
        else:
            np_predictions[key] = value
    return np_predictions


def _prepare_frame_image(image: NDArray[np.uint8], target_size: int) -> NDArray[np.uint8]:
    image = _ensure_rgb_image(image)
    image = _resize_for_vggt(image, target_size)
    return np.ascontiguousarray(image)


def _prepare_frame_mask(mask: NDArray[np.uint8], target_size: int) -> NDArray[np.uint8]:
    mask = _ensure_label_mask(mask)
    mask = _resize_for_vggt_mask(mask, target_size)
    return np.ascontiguousarray(mask)


def _ensure_rgb_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

    if image.shape[2] != 3:
        raise ValueError(f"frame image must have 1, 3, or 4 channels, got {image.shape[2]}")

    return image


def _ensure_label_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if mask.ndim == 2:
        return mask

    if mask.ndim != 3:
        raise ValueError(f"frame mask must be 2D or 3D, got shape {mask.shape}")
    if mask.shape[2] == 4:
        mask = mask[:, :, :3]

    if mask.shape[2] == 1:
        return mask[:, :, 0]
    if mask.shape[2] != 3:
        raise ValueError(f"frame mask must have 1, 3, or 4 channels, got {mask.shape[2]}")

    if np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(mask[:, :, 0], mask[:, :, 2]):
        return mask[:, :, 0]
    return np.where(np.any(mask == SKY_CLASS_ID, axis=2), SKY_CLASS_ID, 0).astype(mask.dtype)


def _resize_for_vggt(image: NDArray[np.uint8], target_size: int) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"frame image has invalid shape {image.shape}")

    scale = target_size / min(height, width)
    new_width = max(target_size, round(width * scale))
    new_height = max(target_size, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    return _center_crop(image, target_size)


def _resize_for_vggt_mask(mask: NDArray[np.uint8], target_size: int) -> NDArray[np.uint8]:
    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"frame mask has invalid shape {mask.shape}")

    scale = target_size / min(height, width)
    new_width = max(target_size, round(width * scale))
    new_height = max(target_size, round(height * scale))

    mask = cv2.resize(
        mask,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return _center_crop(mask, target_size)


def _center_crop(image: NDArray[np.uint8], target_size: int) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    start_y = (height - target_size) // 2
    start_x = (width - target_size) // 2
    if image.ndim == 2:
        return image[start_y : start_y + target_size, start_x : start_x + target_size]
    return image[start_y : start_y + target_size, start_x : start_x + target_size, :]


def _chunk_gps_locations(chunk: Chunk) -> NDArray[np.float32]:
    locations = []
    for frame in chunk.frames:
        if frame.gps_location is None:
            locations.append((np.nan, np.nan, np.nan))
        else:
            locations.append(frame.gps_location)
    return np.array(locations, dtype=np.float32)
