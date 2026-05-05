from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.image import AppImage


def create_image(
    session: Session,
    image_url: str,
    source: str = "upload",
    compressed_image_url: str | None = None,
    width: int | None = None,
    height: int | None = None,
    mime_type: str | None = None,
    file_size_bytes: int | None = None,
) -> AppImage:
    image = AppImage(
        source=source,
        image_url=image_url,
        compressed_image_url=compressed_image_url,
        width=width,
        height=height,
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
        analysis_status="pending",
    )
    session.add(image)
    session.flush()
    return image


def update_image_status(
    session: Session,
    image: AppImage,
    status: str,
    analysis_error: str | None = None,
) -> AppImage:
    if status not in {"pending", "analyzed", "failed"}:
        raise ValueError(f"Invalid image analysis status: {status}")
    image.analysis_status = status
    image.analysis_error = analysis_error
    session.add(image)
    session.flush()
    return image


def get_image(session: Session, image_id: int) -> AppImage | None:
    return session.get(AppImage, image_id)
