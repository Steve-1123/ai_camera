from __future__ import annotations

import numpy as np
from PIL import Image


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("Expected an RGB or RGBA image array.")

    rgb_image = image[:, :, :3]
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb_image)


def resize_longest_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return np.ascontiguousarray(image)

    scale = max_side / longest_side
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(resized_size, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)

