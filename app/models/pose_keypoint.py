from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

IdType = BigInteger().with_variant(Integer, "sqlite")


class PoseKeypoint(Base):
    __tablename__ = "pose_keypoints"

    id: Mapped[int] = mapped_column(IdType, primary_key=True, autoincrement=True)
    pose_id: Mapped[int] = mapped_column(
        IdType,
        ForeignKey("poses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    keypoint_name: Mapped[str] = mapped_column(String(100), nullable=False)
    x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    pose = relationship("Pose", back_populates="keypoints")
