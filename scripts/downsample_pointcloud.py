from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    POINTCLOUD_DOWNSAMPLE_ENABLE,
    POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO,
    POINTCLOUD_DOWNSAMPLE_SUFFIX,
    POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE,
)
from utils.pointcloud_downsample import downsample_point_cloud_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample GLB/PLY point clouds with voxel grid. "
            "If an NPZ path is given, same-stem GLB and PLY files beside it are processed."
        ),
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input GLB/PLY point cloud path, or an NPZ path whose same-stem GLB/PLY files should be processed.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path. Default: input name plus configured downsample suffix.",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Disable processing and copy input to output.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE,
        help=f"Voxel grid size. Default: {POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE}",
    )
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO,
        help=(
            "Minimum point ratio to keep after voxel downsampling. "
            f"Default: {POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO}"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result_paths = _downsample_inputs(args)
    for result_path in result_paths:
        print(result_path)


def _downsample_inputs(args: argparse.Namespace) -> list[Path]:
    suffix = args.input_path.suffix.lower()
    if suffix in (".glb", ".ply"):
        return [_downsample_one(args.input_path, args.output_path, args)]
    if suffix == ".npz":
        if args.output_path is not None:
            raise ValueError("--output-path is only supported when input_path is a GLB or PLY file")

        pointcloud_paths = [
            args.input_path.with_suffix(".glb"),
            args.input_path.with_suffix(".ply"),
        ]
        existing_paths = [path for path in pointcloud_paths if path.exists()]
        if not existing_paths:
            raise FileNotFoundError(
                f"no same-stem GLB/PLY files found beside {args.input_path}: {pointcloud_paths}"
            )
        return [_downsample_one(path, None, args) for path in existing_paths]
    raise ValueError(f"Expected a GLB, PLY, or NPZ file, got {args.input_path}")


def _downsample_one(
    input_path: Path,
    output_path: Path | None,
    args: argparse.Namespace,
) -> Path:
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}{POINTCLOUD_DOWNSAMPLE_SUFFIX}{input_path.suffix}")
    return downsample_point_cloud_file(
        input_path,
        output_path,
        enabled=POINTCLOUD_DOWNSAMPLE_ENABLE and not args.no_enable,
        voxel_size=args.voxel_size,
        min_keep_ratio=args.min_keep_ratio,
    )


if __name__ == "__main__":
    main()
