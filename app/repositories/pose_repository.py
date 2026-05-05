from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.pose import Pose
from app.models.pose_keypoint import PoseKeypoint


def create_pose(
    session: Session,
    image_id: int,
    detector: str = "local_pose_estimator",
    pose_name: str | None = None,
    pose_description: str | None = None,
    confidence: float | None = None,
    bbox: dict[str, Any] | None = None,
    raw_result: dict[str, Any] | None = None,
) -> Pose:
    pose = Pose(
        image_id=image_id,
        detector=detector,
        pose_name=pose_name,
        pose_description=pose_description,
        confidence=confidence,
        bbox=bbox,
        raw_result=raw_result,
    )
    session.add(pose)
    session.flush()
    return pose


def create_keypoints(
    session: Session,
    pose_id: int,
    keypoints: list[dict[str, Any]],
) -> list[PoseKeypoint]:
    rows = [
        PoseKeypoint(
            pose_id=pose_id,
            keypoint_name=str(keypoint["keypoint_name"]),
            x=keypoint.get("x"),
            y=keypoint.get("y"),
            confidence=keypoint.get("confidence"),
        )
        for keypoint in keypoints
    ]
    session.add_all(rows)
    session.flush()
    return rows


def get_poses_by_image_id(session: Session, image_id: int) -> list[Pose]:
    return list(
        session.scalars(
            select(Pose).where(Pose.image_id == image_id).options(selectinload(Pose.keypoints))
        )
    )


def get_pose_with_keypoints(session: Session, pose_id: int) -> Pose | None:
    return session.scalar(
        select(Pose).where(Pose.id == pose_id).options(selectinload(Pose.keypoints))
    )
