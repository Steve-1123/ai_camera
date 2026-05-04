from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.schemas import ImageInfo

logger = logging.getLogger(__name__)


class ImageLoadError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


@dataclass(frozen=True)
class LoadedImage:
    image: np.ndarray
    info: ImageInfo


async def load_upload_image(upload_file: UploadFile) -> LoadedImage:
    raw = await upload_file.read()
    if not raw:
        raise ImageLoadError("Uploaded file is empty or cannot be read as an image.")

    try:
        with Image.open(BytesIO(raw)) as pil_image:
            rgb_image = pil_image.convert("RGB")
            image = np.array(rgb_image, dtype=np.uint8)
            width, height = rgb_image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError("Uploaded file is not a valid image.") from exc

    logger.info(
        "image_loader_debug filename=%s image_shape=%s image_dtype=%s min_pixel=%s max_pixel=%s",
        upload_file.filename,
        image.shape,
        image.dtype,
        int(image.min()),
        int(image.max()),
    )

    return LoadedImage(
        image=image,
        info=ImageInfo(
            filename=upload_file.filename or "uploaded_image",
            width=width,
            height=height,
        ),
    )
