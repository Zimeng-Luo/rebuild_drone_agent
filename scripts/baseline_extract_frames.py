"""导出主样例实际参与重建的 81 帧，供传统方法基线使用。

aligned.npz 里的 frame_ids 是「按每秒 3 帧抽取后的帧序号」，
源视频帧号需要乘以抽样间隔（fps / extract_fps，本例为 10）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extract-fps", type=float, default=3.0)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)

    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(round(fps / args.extract_fps)))
    source_ids = [int(i) * interval for i in frame_ids]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in out.glob("*.jpg"):
        name.unlink()

    wanted = dict(zip(source_ids, range(len(source_ids))))
    last = max(source_ids)
    index = 0
    written = 0
    while index <= last:
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted:
            order = wanted[index]
            cv2.imwrite(
                str(out / f"{order:03d}_src{index:05d}.jpg"),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            written += 1
        index += 1
    capture.release()

    meta = {
        "video": str(args.video),
        "fps": fps,
        "extract_fps": args.extract_fps,
        "frame_interval": interval,
        "selected_frames": int(frame_ids.size),
        "written": written,
        "source_frame_ids": source_ids,
    }
    (out.parent / "frames_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "source_frame_ids"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
