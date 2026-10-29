from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import configure_huggingface_endpoint
from src.datatype import BaseFrame
from src.image_preprocess import preprocess_original_frame

configure_huggingface_endpoint()

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RED_RGB = np.array([255, 0, 0], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SegFormer MiT-B5 to segment the first frame of each video second."
    )
    parser.add_argument("--video", type=str, help="输入视频路径")
    parser.add_argument(
        "--target-values",
        type=str,
        help="目标标签 value，多个值用逗号分隔，例如: 2 或 2,4,6",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="置信度阈值，范围 [0,1]；不传则运行后输入",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备，例如 cuda、cuda:0 或 cpu；默认自动选择",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=str(RESULT_DIR),
        help="实验结果输出目录，默认 experiment/result",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video or input("请输入视频路径: ").strip()).expanduser()
    target_values_text = args.target_values or input(
        "请输入目标标签 value，多个值用逗号分隔: "
    ).strip()
    target_values = parse_target_values(target_values_text)
    confidence_threshold = args.confidence_threshold
    if confidence_threshold is None:
        confidence_threshold = parse_confidence_threshold(
            input("请输入置信度阈值，范围 [0,1]，例如 0.6: ").strip()
        )
    else:
        confidence_threshold = parse_confidence_threshold(str(confidence_threshold))

    result_dir = Path(args.result_dir).expanduser()
    overlay_dir = result_dir / "overlay"
    mask_dir = result_dir / "mask"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    device = resolve_device(args.device)
    print(f"加载模型: {MODEL_NAME}, device={device}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    processed_count = 0
    for second_index, source_frame_index, rgb_image in iter_first_frame_each_second(
        video_path
    ):
        frame = BaseFrame(id=second_index, image=rgb_image, gps_location=None)
        preprocessed = preprocess_original_frame(frame)
        class_map, confidence_map = predict_class_map(
            preprocessed.image,
            processor,
            model,
            device,
        )
        target_mask = np.isin(class_map, target_values) & (
            confidence_map >= confidence_threshold
        )
        overlay = paint_mask_red(preprocessed.image, target_mask)

        stem = f"second_{second_index:06d}_source_{source_frame_index:08d}"
        cv2.imwrite(str(overlay_dir / f"{stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), target_mask.astype(np.uint8) * 255)
        processed_count += 1
        print(
            f"已处理 second={second_index}, source_frame={source_frame_index}, "
            f"target_pixels={int(target_mask.sum())}, "
            f"confidence_threshold={confidence_threshold:.3f}"
        )

    print(f"完成: {processed_count} 帧，结果目录: {result_dir}")


def parse_target_values(values_text: str) -> tuple[int, ...]:
    values = []
    for item in values_text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))

    if not values:
        raise ValueError("至少需要输入一个目标标签 value")
    return tuple(values)


def parse_confidence_threshold(value_text: str) -> float:
    threshold = float(value_text)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"置信度阈值必须在 [0,1] 范围内，当前值: {threshold}")
    return threshold


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_first_frame_each_second(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        capture.release()
        raise ValueError("无法读取有效视频 FPS，不能按每秒第一帧采样")

    second_count = int(math.ceil(frame_count / fps))
    last_frame_index = -1
    try:
        for second_index in range(second_count):
            source_frame_index = int(round(second_index * fps))
            if source_frame_index >= frame_count or source_frame_index == last_frame_index:
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
            success, bgr_image = capture.read()
            if not success:
                continue

            last_frame_index = source_frame_index
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            yield second_index, source_frame_index, rgb_image
    finally:
        capture.release()


def predict_class_map(
    image: np.ndarray,
    processor: AutoImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence, class_map = probabilities.max(dim=1)
    return (
        class_map[0].detach().cpu().numpy(),
        confidence[0].detach().cpu().numpy(),
    )


def paint_mask_red(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    result[mask] = RED_RGB
    return result


if __name__ == "__main__":
    main()
