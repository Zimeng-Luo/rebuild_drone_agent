from __future__ import annotations

import argparse
import logging
from pathlib import Path
import struct
import sys
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALIGN_GLB_CONF_THRES
from src.logging_utils import configure_logging, configure_task_file_logging
from src.rebuild import (
    _build_glb_candidate_mask,
    _build_glb_json_chunk,
    _export_frame_indexes,
    _export_points_and_confidence_keys,
    _frame_export_colors,
    _glb_vertex_dtype,
)


logger = logging.getLogger(__name__)
LABEL_COLOR = np.array([255, 0, 0], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one reconstruction npz file to a GLB file and color points "
            "whose frame mask label equals the given label value red."
        ),
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Input npz file path.",
    )
    parser.add_argument(
        "label",
        type=int,
        help="Mask label value whose corresponding points should be colored red.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output GLB path. Default: input name plus _label_<label>.glb.",
    )
    parser.add_argument(
        "--filter-by-frames",
        default="all",
        help='Frame filter, for example "all" or "0".',
    )
    parser.add_argument(
        "--mask-key",
        default="masks",
        help='NPZ key storing per-frame label masks. Default: "masks".',
    )
    parser.add_argument(
        "--mask-black-bg",
        action="store_true",
        help="Filter black background pixels.",
    )
    parser.add_argument(
        "--mask-white-bg",
        action="store_true",
        help="Filter white background pixels.",
    )
    parser.add_argument(
        "--prediction-mode",
        default="Depthmap and Camera Branch",
        choices=["Depthmap and Camera Branch"],
        help="Prediction branch used by export.",
    )
    args = parser.parse_args()

    if args.npz_path.suffix != ".npz":
        raise ValueError(f"Expected an npz file, got {args.npz_path}")

    output_path = args.output_path
    if output_path is None:
        output_path = args.npz_path.with_name(f"{args.npz_path.stem}_label_{args.label}.glb")

    configure_logging()
    configure_task_file_logging(args.npz_path.parent)
    glb_path = npz_to_labeled_glb(
        npz_path=args.npz_path,
        output_path=output_path,
        label=args.label,
        conf_thres=ALIGN_GLB_CONF_THRES,
        filter_by_frames=args.filter_by_frames,
        mask_key=args.mask_key,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        prediction_mode=args.prediction_mode,
    )
    print(glb_path)


def npz_to_labeled_glb(
    npz_path: str | Path,
    output_path: str | Path,
    label: int,
    conf_thres: float = ALIGN_GLB_CONF_THRES,
    filter_by_frames: str = "all",
    mask_key: str = "masks",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
) -> Path:
    """Convert npz point data to GLB and color selected semantic-label points red."""
    npz_path = Path(npz_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _scan_labeled_glb_points(
        npz_path=npz_path,
        label=label,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_key=mask_key,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        prediction_mode=prediction_mode,
    )
    _write_labeled_glb(
        npz_path=npz_path,
        output_path=output_path,
        stats=stats,
        label=label,
        filter_by_frames=filter_by_frames,
        mask_key=mask_key,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        prediction_mode=prediction_mode,
    )
    logger.info(
        "带标签GLB导出完成: npz_path=%s, output_path=%s, output_points=%d, labeled_points=%d, "
        "conf_percentile=%.3f, conf_threshold=%.6g",
        npz_path,
        output_path,
        stats["point_count"],
        stats["labeled_point_count"],
        conf_thres,
        stats["threshold"],
    )
    return output_path


def _scan_labeled_glb_points(
    npz_path: Path,
    label: int,
    conf_thres: float,
    filter_by_frames: str,
    mask_key: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    prediction_mode: str,
) -> dict[str, Any]:
    if not 0.0 <= conf_thres <= 100.0:
        raise ValueError(f"conf_thres must be in [0, 100], got {conf_thres}")

    confidence_chunks = []
    total_points = 0
    candidate_points = 0
    with np.load(npz_path, mmap_mode="r") as loaded:
        _validate_npz_keys(npz_path, loaded.files, mask_key, prediction_mode)
        points_key, confidence_key = _export_points_and_confidence_keys(prediction_mode)
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
        labeled_point_count = 0
        bounds_min = np.full(3, np.inf, dtype=np.float64)
        bounds_max = np.full(3, -np.inf, dtype=np.float64)
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
            if np.isnan(threshold):
                keep_mask = np.zeros(confidence.shape, dtype=bool)
            else:
                keep_mask = candidate_mask & (confidence >= threshold)
            if not np.any(keep_mask):
                continue

            label_mask = _frame_label_mask_for_points(
                np.asarray(loaded[mask_key][frame_index]),
                points.shape[:2],
                label,
            ).reshape(-1)
            kept_vertices = vertices[keep_mask]
            point_count += int(kept_vertices.shape[0])
            labeled_point_count += int(np.count_nonzero(label_mask[keep_mask]))
            bounds_min = np.minimum(bounds_min, np.min(kept_vertices, axis=0))
            bounds_max = np.maximum(bounds_max, np.max(kept_vertices, axis=0))

    if point_count == 0:
        bounds_min = np.zeros(3, dtype=np.float64)
        bounds_max = np.zeros(3, dtype=np.float64)

    logger.info(
        "带标签GLB点云过滤完成: total_points=%d, candidate_points=%d, output_points=%d, "
        "labeled_points=%d, conf_percentile=%.3f, conf_threshold=%.6g",
        total_points,
        candidate_points,
        point_count,
        labeled_point_count,
        conf_thres,
        threshold,
    )
    return {
        "threshold": threshold,
        "point_count": point_count,
        "labeled_point_count": labeled_point_count,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }


def _write_labeled_glb(
    npz_path: Path,
    output_path: Path,
    stats: dict[str, Any],
    label: int,
    filter_by_frames: str,
    mask_key: str,
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
        for vertices, colors in _iter_labeled_glb_points(
            npz_path=npz_path,
            threshold=stats["threshold"],
            label=label,
            filter_by_frames=filter_by_frames,
            mask_key=mask_key,
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


def _iter_labeled_glb_points(
    npz_path: Path,
    threshold: float,
    label: int,
    filter_by_frames: str,
    mask_key: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    prediction_mode: str,
):
    with np.load(npz_path, mmap_mode="r") as loaded:
        _validate_npz_keys(npz_path, loaded.files, mask_key, prediction_mode)
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
            if np.isnan(threshold):
                keep_mask = np.zeros(confidence.shape, dtype=bool)
            else:
                keep_mask = candidate_mask & (confidence >= threshold)
            if not np.any(keep_mask):
                continue

            label_mask = _frame_label_mask_for_points(
                np.asarray(loaded[mask_key][frame_index]),
                points.shape[:2],
                label,
            ).reshape(-1)
            colors = colors.copy()
            colors[label_mask] = LABEL_COLOR
            yield vertices[keep_mask].astype(np.float32, copy=False), colors[keep_mask]


def _frame_label_mask_for_points(
    mask: NDArray[Any],
    point_shape: tuple[int, int],
    label: int,
) -> NDArray[np.bool_]:
    mask_hwc = _mask_to_hwc(mask)
    if mask_hwc.shape[-1] == 1:
        label_mask = mask_hwc[:, :, 0] == label
    else:
        label_mask = np.any(mask_hwc == label, axis=-1)

    if label_mask.shape != point_shape:
        label_mask = cv2.resize(
            label_mask.astype(np.uint8),
            (point_shape[1], point_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return label_mask


def _mask_to_hwc(mask: NDArray[Any]) -> NDArray[Any]:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 2:
        return mask_array[:, :, None]
    if mask_array.ndim != 3:
        raise ValueError(f"frame mask must be 2D or 3D, got shape {mask_array.shape}")
    if mask_array.shape[0] in (1, 3, 4) and mask_array.shape[-1] not in (1, 3, 4):
        return np.transpose(mask_array, (1, 2, 0))
    return mask_array


def _validate_npz_keys(
    npz_path: Path,
    keys: list[str],
    mask_key: str,
    prediction_mode: str,
) -> None:
    points_key, confidence_key = _export_points_and_confidence_keys(prediction_mode)
    missing_keys = [key for key in (points_key, confidence_key, "images", mask_key) if key not in keys]
    if missing_keys:
        raise KeyError(f"{npz_path} is missing labeled GLB export keys: {missing_keys}")


def _align4(value: int) -> int:
    return (value + 3) & ~3


if __name__ == "__main__":
    main()
