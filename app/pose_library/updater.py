from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.features.skeleton_features import compute_skeleton_features
from app.pose_library.loader import load_pose_library_document, save_pose_library_document
from app.vision.pose_estimator import PoseEstimator
from app.vision.scene_classifier import SceneClassifier


class PoseIngestError(ValueError):
    """Raised when a stored pose cannot be updated from an image."""


def ingest_pose_image(
    poses_json_path: str,
    pose_id: str,
    image_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    document = load_pose_library_document(poses_json_path)
    pose = _find_pose(document, pose_id)
    if pose is None:
        raise PoseIngestError(f"Pose id not found: {pose_id}")

    if pose.get("landmark_template") and not overwrite:
        raise PoseIngestError(
            f"Pose {pose_id} already has landmark_template. Use overwrite=True to replace it."
        )

    image, width, height = _load_image_file(image_path)
    pose_result = PoseEstimator().estimate(image)
    if not pose_result.has_person:
        raise PoseIngestError(f"No person detected in image: {image_path}")

    landmarks = [landmark.model_dump() for landmark in pose_result.landmarks]
    scene_result = SceneClassifier().classify(image)
    skeleton_features = compute_skeleton_features(landmarks)
    landmark_template = build_landmark_template(
        image_path=image_path,
        image_width=width,
        image_height=height,
        landmarks=landmarks,
        quality=_build_quality(landmarks),
        derived_features=skeleton_features,
    )

    pose["landmark_template"] = landmark_template
    pose["scene_analysis"] = scene_result.model_dump()
    pose["embedding_document"] = build_embedding_document(pose)
    save_pose_library_document(document, poses_json_path)
    return pose


def build_landmark_template(
    image_path: str,
    image_width: int,
    image_height: int,
    landmarks: list[dict[str, Any]],
    quality: dict[str, Any],
    derived_features: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": {
            "type": "image",
            "filename": Path(image_path).name,
            "image_width": image_width,
            "image_height": image_height,
            "created_by": "mediapipe_pose",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "format": "mediapipe_pose_33",
        "coordinate_space": "normalized_image",
        "normalization": {
            "is_normalized": True,
            "origin": "image_top_left",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "z_scale": "mediapipe_relative",
            "visibility_range": [0.0, 1.0],
        },
        "quality": quality,
        "landmarks": landmarks,
        "derived_features": derived_features,
    }


def build_embedding_document(pose: dict[str, Any]) -> str:
    scene_analysis = pose.get("scene_analysis") if isinstance(pose.get("scene_analysis"), dict) else {}
    landmark_template = pose.get("landmark_template") if isinstance(pose.get("landmark_template"), dict) else {}
    features = landmark_template.get("derived_features") if isinstance(landmark_template, dict) else {}
    if not isinstance(features, dict):
        features = {}

    feature_text = [
        f"body_height_ratio={features.get('body_height_ratio')}",
        f"shoulder_width_ratio={features.get('shoulder_width_ratio')}",
        f"left_arm_raised={features.get('left_arm_raised')}",
        f"right_arm_raised={features.get('right_arm_raised')}",
        f"hands_near_face={features.get('hands_near_face')}",
        f"legs_crossed={features.get('legs_crossed')}",
        f"is_sitting={features.get('is_sitting')}",
        f"is_full_body={features.get('is_full_body')}",
    ]
    parts = [
        _text("display_name", pose.get("display_name")),
        _text("category", pose.get("category")),
        _text("scene_tags", pose.get("scene_tags")),
        _text("object_tags", pose.get("object_tags")),
        _text("style_tags", pose.get("style_tags")),
        _text("scene_primary_category", scene_analysis.get("primary_category")),
        _text("skeleton_features", feature_text),
        _text("instructions", pose.get("instructions")),
    ]
    return "\n".join(part for part in parts if part)


def _find_pose(document: dict[str, Any], pose_id: str) -> dict[str, Any] | None:
    poses = document.get("poses")
    if not isinstance(poses, list):
        raise PoseIngestError("Pose library document must contain a poses array.")
    return next((pose for pose in poses if isinstance(pose, dict) and pose.get("pose_id") == pose_id), None)


def _load_image_file(image_path: str) -> tuple[np.ndarray, int, int]:
    try:
        with Image.open(image_path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            image = np.asarray(rgb_image, dtype=np.uint8)
            width, height = rgb_image.size
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise PoseIngestError(f"Image could not be loaded: {image_path}") from exc
    return image, width, height


def _build_quality(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    visibilities = [
        float(landmark["visibility"])
        for landmark in landmarks
        if isinstance(landmark.get("visibility"), (int, float))
    ]
    visible_landmark_count = sum(visibility >= 0.5 for visibility in visibilities)
    warnings = []
    if len(landmarks) != 33:
        warnings.append(f"expected_33_landmarks_got_{len(landmarks)}")
    if visible_landmark_count < 12:
        warnings.append("low_visible_landmark_count")

    return {
        "has_person": True,
        "landmark_count": len(landmarks),
        "visible_landmark_count": visible_landmark_count,
        "min_visibility": 0.5,
        "mean_visibility": round(sum(visibilities) / len(visibilities), 4) if visibilities else 0.0,
        "is_usable": len(landmarks) == 33 and visible_landmark_count >= 12,
        "warnings": warnings,
    }


def _text(label: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        cleaned = [str(item) for item in value if item is not None and str(item)]
        if not cleaned:
            return ""
        return f"{label}: {', '.join(cleaned)}"
    if isinstance(value, str) and not value.strip():
        return ""
    return f"{label}: {value}"
