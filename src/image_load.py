import logging
import re
from collections.abc import Iterator
from pathlib import Path

import cv2

from config import IMAGE_SAMPLE_INTERVAL, IMAGE_SEGMENT_SIZE
from src.datatype import BaseFrame, OriginalFrames


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def load_original_frames_from_image(image_dir: str | Path) -> OriginalFrames:
    segments = list(
        iter_original_frame_segments_from_image(image_dir, segment_size=None)
    )
    return segments[0] if segments else OriginalFrames()


def iter_original_frame_segments_from_image(
    image_dir: str | Path,
    segment_size: int | None = IMAGE_SEGMENT_SIZE,
    sample_interval: int = IMAGE_SAMPLE_INTERVAL,
) -> Iterator[OriginalFrames]:
    all_image_paths = _resolve_image_paths(image_dir)
    image_paths = _select_image_paths(all_image_paths, sample_interval=sample_interval)
    segment_size = _validate_segment_size(segment_size)
    logger.info(
        "读取图片目录成功: path=%s, total_images=%d, selected_images=%d, "
        "segment_size=%s, sample_interval=%d, first_image=%s, last_image=%s",
        image_dir,
        len(all_image_paths),
        len(image_paths),
        segment_size,
        sample_interval,
        image_paths[0].name if image_paths else None,
        image_paths[-1].name if image_paths else None,
    )

    frame_id = 0
    segment_index = 0
    segment_frames = OriginalFrames()
    segment_first_path: Path | None = None
    segment_last_path: Path | None = None
    segment_start_frame_id = 0

    for path in image_paths:
        if not segment_frames.frames:
            segment_first_path = path
            segment_start_frame_id = frame_id
            logger.info(
                "开始读取图片片段: path=%s, segment_index=%d, start_frame_id=%d, "
                "first_image=%s, segment_size=%s",
                image_dir,
                segment_index,
                segment_start_frame_id,
                path.name,
                segment_size,
            )
        segment_frames.add_frame(
            BaseFrame(
                id=frame_id,
                image=_read_image(path),
                gps_location=None,
            )
        )
        segment_last_path = path
        frame_id += 1

        if segment_size is not None and len(segment_frames.frames) >= segment_size:
            logger.info(
                "图片片段读取完成: path=%s, segment_index=%d, extracted_frames=%d, "
                "frame_id_range=%s-%s, first_image=%s, last_image=%s",
                image_dir,
                segment_index,
                len(segment_frames.frames),
                segment_start_frame_id,
                frame_id - 1,
                segment_first_path.name if segment_first_path else None,
                segment_last_path.name if segment_last_path else None,
            )
            yield segment_frames
            segment_index += 1
            segment_frames = OriginalFrames()
            segment_first_path = None
            segment_last_path = None

    if segment_frames.frames:
        logger.info(
            "图片片段读取完成: path=%s, segment_index=%d, extracted_frames=%d, "
            "frame_id_range=%s-%s, first_image=%s, last_image=%s",
            image_dir,
            segment_index,
            len(segment_frames.frames),
            segment_start_frame_id,
            frame_id - 1,
            segment_first_path.name if segment_first_path else None,
            segment_last_path.name if segment_last_path else None,
        )
        yield segment_frames

    logger.info(
        "图片读取完成: path=%s, extracted_frames=%d",
        image_dir,
        frame_id,
    )


def _resolve_image_paths(image_dir: str | Path) -> list[Path]:
    path = Path(image_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"Image input path must be a directory: {path}")

    image_paths = sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )
    image_paths = sorted(
        image_paths,
        key=_numeric_image_sort_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found in directory: {path}")
    return image_paths


def _numeric_image_sort_key(
    path: Path,
) -> tuple[int, tuple[tuple[int, int | str], ...], str]:
    parts = re.split(r"(\d+)", path.stem)
    numeric_parts: list[tuple[int, int | str]] = []
    has_number = False
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            numeric_parts.append((0, int(part)))
            has_number = True
        else:
            numeric_parts.append((1, part))

    return (0 if has_number else 1, tuple(numeric_parts), path.name)


def _select_image_paths(image_paths: list[Path], sample_interval: int) -> list[Path]:
    if sample_interval <= 0:
        raise ValueError(
            f"sample_interval must be greater than 0, got {sample_interval}"
        )

    selected_paths: list[Path] = []
    for index, image_path in enumerate(image_paths):
        if index % sample_interval == 0:
            selected_paths.append(image_path)
    return selected_paths


def _validate_segment_size(segment_size: int | None) -> int | None:
    if segment_size is None:
        return None

    if segment_size <= 0:
        raise ValueError(f"segment_size must be greater than 0, got {segment_size}")

    return segment_size


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not open image file: {path}")
    return _to_rgb(image)


def _to_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    raise ValueError(f"Expected image with 1, 3, or 4 channels, got {image.shape[2]}")
