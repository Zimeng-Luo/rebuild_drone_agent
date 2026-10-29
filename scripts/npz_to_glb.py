from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rebuild import run_rebuild_to_glb


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one VGGT-style reconstruction npz file to a GLB file. "
            "If the npz contains masks, masked points are filtered from output."
        ),
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Input npz file path. The GLB file is saved beside it with the same name.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=50.0,
        help="Confidence percentile threshold used by VGGT visualization.",
    )
    parser.add_argument(
        "--filter-by-frames",
        default="all",
        help='Frame filter passed to VGGT visualization, for example "all" or "0".',
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
        "--hide-cam",
        action="store_true",
        help="Do not include camera visualization in the GLB.",
    )
    parser.add_argument(
        "--prediction-mode",
        default="Depthmap and Camera Branch",
        choices=["Depthmap and Camera Branch"],
        help="Prediction branch used by VGGT visualization.",
    )
    args = parser.parse_args()

    if args.npz_path.suffix != ".npz":
        raise ValueError(f"Expected an npz file, got {args.npz_path}")

    glb_path = run_rebuild_to_glb(
        npz_path=args.npz_path,
        output_path=args.npz_path.with_suffix(".glb"),
        conf_thres=args.conf_thres,
        filter_by_frames=args.filter_by_frames,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=not args.hide_cam,
        mask_sky=False,
        prediction_mode=args.prediction_mode,
    )
    print(glb_path)


if __name__ == "__main__":
    main()
