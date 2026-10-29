import logging

import cv2

from src.datatype import BaseFrame, OriginalFrames


logger = logging.getLogger(__name__)


def preprocess_original_frames(frames: OriginalFrames) -> OriginalFrames:
    preprocessed_frames = OriginalFrames()
    for frame in frames.frames:
        preprocessed_frames.add_frame(preprocess_original_frame(frame))

    logger.info("所有视频帧处理完成: count=%d", len(preprocessed_frames.frames))
    return preprocessed_frames


def preprocess_original_frame(frame: BaseFrame) -> BaseFrame:
    image = _ensure_rgb_image(frame.image)
    image = _equalize_illumination(image)
    image = _resize_for_vggt(image)
    mask = _prepare_mask(frame.mask)
    return BaseFrame(
        id=frame.id,
        image=image,
        gps_location=frame.gps_location,
        mask=mask,
    )


def _prepare_mask(mask):
    mask = _ensure_label_mask(mask)
    return _resize_for_vggt(mask, interpolation=cv2.INTER_NEAREST)


def _ensure_label_mask(mask):
    if mask.ndim == 2:
        return mask

    if mask.shape[2] == 4:
        mask = mask[:, :, :3]

    if mask.shape[2] == 1:
        return mask[:, :, 0]
    if mask.shape[2] == 3:
        if (mask[:, :, 0] == mask[:, :, 1]).all() and (mask[:, :, 0] == mask[:, :, 2]).all():
            return mask[:, :, 0]
        return ((mask == 2).any(axis=2).astype(mask.dtype)) * 2

    raise ValueError(f"frame mask must have 1, 3, or 4 channels, got {mask.shape[2]}")


def _ensure_rgb_image(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype("float32") / 255.0
        rgb = image[:, :, :3].astype("float32")
        white = 255.0
        return (rgb * alpha + white * (1.0 - alpha)).astype("uint8")

    return image


def _equalize_illumination(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    lab_image = cv2.merge((lightness, channel_a, channel_b))
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def _resize_for_vggt(image, target_size: int = 518, interpolation: int | None = None):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"frame image has invalid shape {image.shape}")

    scale = target_size / min(height, width)
    new_width = max(target_size, round(width * scale))
    new_height = max(target_size, round(height * scale))
    resize_interpolation = interpolation
    if resize_interpolation is None:
        resize_interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=resize_interpolation,
    )

    start_y = (new_height - target_size) // 2
    start_x = (new_width - target_size) // 2
    if image.ndim == 2:
        return image[start_y : start_y + target_size, start_x : start_x + target_size]
    return image[start_y : start_y + target_size, start_x : start_x + target_size, :]
