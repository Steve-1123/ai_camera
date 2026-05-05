from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.core.database import session_scope
from app.models.image import AppImage
from app.models.pose import Pose
from app.models.pose_keypoint import PoseKeypoint
from app.repositories.image_repository import create_image, update_image_status
from app.repositories.pose_repository import create_keypoints, create_pose
from app.schemas import (
    AnalyzeImagePathResponse,
    ImageInfo,
    ImageStorageResult,
    PoseEstimationResult,
)
from app.vision.pose_estimator import PoseEstimator
from app.vision.scene_classifier import SceneClassifier


class ImageAnalysisError(RuntimeError):
    """Raised when an image cannot be analyzed."""


def analyze_image_path(
    image_url: str,
    should_store: bool = False,
    source: str = "upload",
    detector: str = "local_pose_estimator",
    session: Session | None = None,
    pose_estimator: Any | None = None,
    scene_classifier: Any | None = None,
) -> AnalyzeImagePathResponse:
    if not image_url or not image_url.strip():
        raise ValueError("image_url cannot be empty.")

    image_array, image_metadata = _load_image(image_url)
    estimator = pose_estimator or PoseEstimator()
    classifier = scene_classifier or SceneClassifier()

    pose_result = estimator.estimate(image_array)
    scene_result = classifier.classify(image_array)
    storage = ImageStorageResult(requested=should_store)

    if should_store:
        storage = _store_analysis_result(
            image_url=image_url,
            source=source,
            detector=detector,
            image_metadata=image_metadata,
            pose_result=pose_result,
            session=session,
        )

    return AnalyzeImagePathResponse(
        image_info=ImageInfo(
            filename=Path(image_url).name,
            width=image_metadata["width"],
            height=image_metadata["height"],
        ),
        pose=pose_result,
        scene=scene_result,
        storage=storage,
    )


def _store_analysis_result(
    image_url: str,
    source: str,
    detector: str,
    image_metadata: dict[str, Any],
    pose_result: PoseEstimationResult,
    session: Session | None,
) -> ImageStorageResult:
    if session is None:
        try:
            with session_scope() as managed_session:
                return _store_with_session(
                    managed_session,
                    image_url=image_url,
                    source=source,
                    detector=detector,
                    image_metadata=image_metadata,
                    pose_result=pose_result,
                )
        except Exception as exc:
            return ImageStorageResult(requested=True, analysis_status="failed", error=str(exc))

    return _store_with_session(
        session,
        image_url=image_url,
        source=source,
        detector=detector,
        image_metadata=image_metadata,
        pose_result=pose_result,
    )


def _store_with_session(
    session: Session,
    image_url: str,
    source: str,
    detector: str,
    image_metadata: dict[str, Any],
    pose_result: PoseEstimationResult,
) -> ImageStorageResult:
    image_row: AppImage | None = None
    try:
        image_row = create_image(
            session,
            image_url=image_url,
            source=source,
            width=image_metadata["width"],
            height=image_metadata["height"],
            mime_type=image_metadata["mime_type"],
            file_size_bytes=image_metadata["file_size_bytes"],
        )
        standardized = standardize_pose_result(pose_result, detector=detector)
        created_poses: list[tuple[Pose, list[PoseKeypoint]]] = []
        for pose_payload in standardized["poses"]:
            pose = create_pose(
                session,
                image_id=image_row.id,
                detector=detector,
                pose_name=pose_payload.get("pose_name"),
                pose_description=pose_payload.get("pose_description"),
                confidence=pose_payload.get("confidence"),
                bbox=pose_payload.get("bbox"),
                raw_result=pose_payload.get("raw_result"),
            )
            keypoints = create_keypoints(session, pose.id, pose_payload.get("keypoints", []))
            created_poses.append((pose, keypoints))

        update_image_status(session, image_row, "analyzed")
        session.commit()
        return ImageStorageResult(
            requested=True,
            image_id=image_row.id,
            analysis_status=image_row.analysis_status,
            pose_count=len(created_poses),
            keypoint_count=sum(len(keypoints) for _, keypoints in created_poses),
        )
    except Exception as exc:
        session.rollback()
        if image_row is not None:
            try:
                image_row = session.get(AppImage, image_row.id)
                if image_row is not None:
                    update_image_status(session, image_row, "failed", str(exc))
                    session.commit()
                    return ImageStorageResult(
                        requested=True,
                        image_id=image_row.id,
                        analysis_status="failed",
                        error=str(exc),
                    )
            except Exception:
                session.rollback()
        return ImageStorageResult(requested=True, analysis_status="failed", error=str(exc))


def standardize_pose_result(
    pose_result: PoseEstimationResult,
    detector: str = "local_pose_estimator",
) -> dict[str, Any]:
    if not pose_result.has_person:
        return {"poses": []}

    landmarks = [landmark.model_dump() for landmark in pose_result.landmarks]
    keypoints = [
        {
            "keypoint_name": landmark["name"],
            "x": landmark.get("x"),
            "y": landmark.get("y"),
            "confidence": landmark.get("visibility"),
        }
        for landmark in landmarks
    ]
    return {
        "poses": [
            {
                "pose_name": "detected_person",
                "pose_description": "Person pose detected by local MediaPipe estimator.",
                "confidence": _mean_visibility(landmarks),
                "bbox": _landmark_bbox(landmarks),
                "keypoints": keypoints,
                "raw_result": {
                    "detector": detector,
                    "has_person": pose_result.has_person,
                    "landmark_count": pose_result.landmark_count,
                    "landmarks": landmarks,
                },
            }
        ]
    }


def _load_image(image_url: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(image_url)
    try:
        with Image.open(path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            image = np.asarray(rgb_image, dtype=np.uint8)
            width, height = rgb_image.size
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ImageAnalysisError(f"Image could not be loaded: {image_url}") from exc

    return image, {
        "width": width,
        "height": height,
        "mime_type": mimetypes.guess_type(path.name)[0],
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }


def _landmark_bbox(landmarks: list[dict[str, Any]]) -> dict[str, float] | None:
    coordinates = [
        (float(landmark["x"]), float(landmark["y"]))
        for landmark in landmarks
        if isinstance(landmark.get("x"), (int, float)) and isinstance(landmark.get("y"), (int, float))
    ]
    if not coordinates:
        return None
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        "x": round(min_x, 4),
        "y": round(min_y, 4),
        "w": round(max_x - min_x, 4),
        "h": round(max_y - min_y, 4),
    }


def _mean_visibility(landmarks: list[dict[str, Any]]) -> float | None:
    values = [
        float(landmark["visibility"])
        for landmark in landmarks
        if isinstance(landmark.get("visibility"), (int, float))
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)
