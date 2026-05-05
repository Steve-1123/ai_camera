from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

IdType = BigInteger().with_variant(Integer, "sqlite")


class Pose(Base):
    __tablename__ = "poses"

    id: Mapped[int] = mapped_column(IdType, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        IdType,
        ForeignKey("app_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    detector: Mapped[str] = mapped_column(String(50), default="local_pose_estimator", nullable=False)
    pose_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    pose_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    raw_result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    image = relationship("AppImage", back_populates="poses")
    keypoints = relationship("PoseKeypoint", back_populates="pose", cascade="all, delete-orphan")
