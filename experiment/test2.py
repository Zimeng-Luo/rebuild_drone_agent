"""
测试对可靠性的预测是否准确(卫星图分割)
data:2026/7/9
"""

from __future__ import annotations

import argparse             #接收用户在的命令行输入
from pathlib import Path
import sys

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))        #将根目录加入到python包的搜索路径中


from experiment.test1 import (
    SATELLITE_FRAME_ID_BASE,
    _create_task_dir,
    _save_chunk,
    score_to_heatmap,
    save_feature_similarity_predictions,
    load_preprocessed_keyframes_from_video,
    read_rgb_image,
    top_score_points_to_image,
)
from src.datatype import BaseFrame, Chunk
from src.logging_utils import configure_logging, configure_task_file_logging
from src.rebuild import load_vggt_model
from src.task_summary import ResourceStepMonitor

RESULT_DIR = "../data/result"
EXPERIMENT_NAME = "reliability_satellite_tiles"
DEFAULT_RESULT_DIR = Path(RESULT_DIR) / "experiment_reliability"
MODEL_NAME_OR_PATH = "facebook/VGGT-1B"
DEVICE = None


def run_tiled_satellite_reliability_experiment(
    satellite_image_path: str | Path,
    plane_image_path: str | Path,
    plane_image_count: int,
) -> dict[str, Path]:
    """Run VGGT on four satellite tiles plus selected ground frames."""
    configure_logging()
    plane_image_path = Path(plane_image_path)
    task_dir = _create_task_dir(DEFAULT_RESULT_DIR, EXPERIMENT_NAME)
    configure_task_file_logging(task_dir)

    with ResourceStepMonitor("prepare_frames"):
        satellite_image = read_rgb_image(satellite_image_path)

        satellite_tiles = split_satellite_image_2x2(satellite_image)
        tile_frames = [
            BaseFrame(
                id=SATELLITE_FRAME_ID_BASE + index,
                image=tile.image,
                gps_location=None,
            )
            for index, tile in enumerate(satellite_tiles)
        ]
        ground_frames = load_preprocessed_keyframes_from_video(
            plane_image_path,
            task_dir / "ground_preprocessed_frames",
            plane_image_count=plane_image_count,
        )
        if not ground_frames:
            raise ValueError(f"no ground key frames selected from video: {plane_image_path}")
        chunk = Chunk(id=0, frames=[*tile_frames, *ground_frames])
        chunk_path = _save_chunk(chunk, task_dir / "0.pkl")

    with ResourceStepMonitor("vggt_feature_similarity"):
        model = load_vggt_model(model_name_or_path=MODEL_NAME_OR_PATH, device=DEVICE)
        similarity_path = save_feature_similarity_predictions(
            chunk=chunk,
            output_path=task_dir / "feature_similarity.npz",
            model=model,
            satellite_frame_count=len(satellite_tiles),
        )

    with ResourceStepMonitor("export_tiled_satellite_similarity_heatmap"):
        heatmap_path, top_similarity_path = export_tiled_similarity_outputs(
            similarity_path,
            satellite_tiles,
            heatmap_output_path=task_dir / "satellite_tiled_similarity_heatmap.png",
            top_output_path=task_dir / "satellite_tiled_similarity_top10.png",
        )

    return {
        "task_dir": task_dir,
        "chunk": chunk_path,
        "npz": similarity_path,
        "heatmap": heatmap_path,
        "top_similarity": top_similarity_path,
    }


class SatelliteTile:
    def __init__(
        self,
        image: NDArray[np.uint8],
        row: int,
        col: int,
        height: int,
        width: int,
    ) -> None:
        self.image = image
        self.row = row
        self.col = col
        self.height = height
        self.width = width


def split_satellite_image_2x2(image: NDArray[np.uint8]) -> list[SatelliteTile]:
    height, width = image.shape[:2]
    mid_y = height // 2
    mid_x = width // 2
    boxes = [
        (0, 0, mid_y, mid_x),
        (0, mid_x, mid_y, width),
        (mid_y, 0, height, mid_x),
        (mid_y, mid_x, height, width),
    ]
    tiles = []
    for row_start, col_start, row_end, col_end in boxes:
        tile_image = image[row_start:row_end, col_start:col_end].copy()
        tiles.append(
            SatelliteTile(
                image=tile_image,
                row=row_start,
                col=col_start,
                height=row_end - row_start,
                width=col_end - col_start,
            )
        )
    return tiles


def export_tiled_similarity_outputs(
    npz_path: str | Path,
    satellite_tiles: list[SatelliteTile],
    heatmap_output_path: str | Path,
    top_output_path: str | Path,
) -> tuple[Path, Path]:
    heatmap_output_path = Path(heatmap_output_path)
    top_output_path = Path(top_output_path)
    merged_similarity = load_tiled_similarity_map(npz_path, satellite_tiles)

    heatmap = score_to_heatmap(merged_similarity)
    heatmap_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(heatmap_output_path), heatmap):
        raise ValueError(f"could not write heatmap: {heatmap_output_path}")

    base_image = merge_satellite_tiles(satellite_tiles)
    top_image = top_score_points_to_image(merged_similarity, base_image=base_image)
    top_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(top_output_path), top_image):
        raise ValueError(f"could not write top similarity image: {top_output_path}")

    return heatmap_output_path, top_output_path


def load_tiled_similarity_map(
    npz_path: str | Path,
    satellite_tiles: list[SatelliteTile],
) -> NDArray[np.float32]:
    with np.load(npz_path) as loaded:
        if "similarity_maps" not in loaded.files:
            raise KeyError(f"{npz_path} is missing similarity_maps")
        similarity_maps = np.asarray(loaded["similarity_maps"])

    full_height = max(tile.row + tile.height for tile in satellite_tiles)
    full_width = max(tile.col + tile.width for tile in satellite_tiles)
    merged_similarity = np.zeros((full_height, full_width), dtype=np.float32)
    for index, tile in enumerate(satellite_tiles):
        tile_similarity = np.asarray(similarity_maps[index]).squeeze().astype(np.float32)
        tile_similarity = cv2.resize(
            tile_similarity,
            (tile.width, tile.height),
            interpolation=cv2.INTER_CUBIC,
        )
        merged_similarity[
            tile.row : tile.row + tile.height,
            tile.col : tile.col + tile.width,
        ] = tile_similarity
    return merged_similarity


def merge_satellite_tiles(satellite_tiles: list[SatelliteTile]) -> NDArray[np.uint8]:
    full_height = max(tile.row + tile.height for tile in satellite_tiles)
    full_width = max(tile.col + tile.width for tile in satellite_tiles)
    merged_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    for tile in satellite_tiles:
        merged_image[tile.row : tile.row + tile.height, tile.col : tile.col + tile.width] = tile.image[:, :, :3]
    return merged_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test VGGT backbone feature similarity on four satellite image tiles."
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
    outputs = run_tiled_satellite_reliability_experiment(
        args.satellite_image_path,
        args.plane_image_path,
        args.plane_image_count,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
