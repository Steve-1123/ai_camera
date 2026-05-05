from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

IdType = BigInteger().with_variant(Integer, "sqlite")


class AppImage(Base):
    __tablename__ = "app_images"

    id: Mapped[int] = mapped_column(IdType, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(50), default="upload", nullable=False)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    compressed_image_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    analysis_status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    analysis_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    poses = relationship("Pose", back_populates="image", cascade="all, delete-orphan")
