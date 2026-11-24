"""用 COLMAP（pycolmap）在同一批 81 帧上跑传统 SfM 基线，记录耗时与产物规模。

对照条件与本文流水线一致：同一段视频、同一批 81 个关键帧、同一台机器、
同一张 RTX 4090 的 CPU 侧（COLMAP 的 SIFT 与增量重建走 CPU）。
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import pycolmap

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("baseline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--max-image-size", type=int, default=3200)
    args = ap.parse_args()

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    (work / "sparse").mkdir(parents=True)
    db = work / "database.db"

    marks: dict[str, float] = {}
    t_all = time.time()

    LOGGER.info("=== 1/4 特征提取 ===")
    t = time.time()
    pycolmap.extract_features(
        db,
        args.images,
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=pycolmap.FeatureExtractionOptions(
            max_image_size=args.max_image_size
        ),
    )
    marks["extract_features"] = time.time() - t
    LOGGER.info("特征提取 %.1f s", marks["extract_features"])

    LOGGER.info("=== 2/4 顺序匹配（重叠 %d）===", args.overlap)
    t = time.time()
    pycolmap.match_sequential(
        db,
        pairing_options=pycolmap.SequentialPairingOptions(overlap=args.overlap),
    )
    marks["match_sequential"] = time.time() - t
    LOGGER.info("匹配 %.1f s", marks["match_sequential"])

    LOGGER.info("=== 3/4 增量重建 ===")
    t = time.time()
    recs = pycolmap.incremental_mapping(db, args.images, work / "sparse")
    marks["incremental_mapping"] = time.time() - t
    LOGGER.info("重建 %.1f s，得到 %d 个模型", marks["incremental_mapping"], len(recs))

    best = max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None
    marks["total"] = time.time() - t_all

    summary = {
        "images_input": len(list(Path(args.images).glob("*.jpg"))),
        "max_image_size": args.max_image_size,
        "overlap": args.overlap,
        "models": len(recs),
        "registered_images": best.num_reg_images() if best else 0,
        "points3d": best.num_points3D() if best else 0,
        "mean_track_length": float(best.compute_mean_track_length()) if best else 0.0,
        "mean_reproj_error": float(best.compute_mean_reprojection_error())
        if best
        else 0.0,
        "seconds": {k: round(v, 2) for k, v in marks.items()},
    }
    if best:
        out = work / "sparse" / "best"
        out.mkdir(parents=True, exist_ok=True)
        best.write(str(out))
        ply = Path(args.work) / "points.ply"
        best.export_PLY(str(ply))
        summary["ply_bytes"] = ply.stat().st_size if ply.exists() else 0

    (work / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("结果 %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
