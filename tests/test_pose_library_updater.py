from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.pose_library import updater
from app.pose_library.updater import PoseIngestError, build_embedding_document, ingest_pose_image
from app.schemas import Landmark, PoseEstimationResult, SceneCandidate, SceneClassificationResult


def test_updater_pose_id_not_found_raises(tmp_path: Path) -> None:
    poses_json = _write_library(tmp_path, [{"pose_id": "known", "display_name": "Known"}])
    image_path = _write_image(tmp_path)

    try:
        ingest_pose_image(str(poses_json), "missing", str(image_path))
    except PoseIngestError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Expected missing pose_id to raise.")


def test_updater_existing_template_without_overwrite_raises(tmp_path: Path) -> None:
    poses_json = _write_library(
        tmp_path,
        [{"pose_id": "known", "display_name": "Known", "landmark_template": {"landmarks": []}}],
    )
    image_path = _write_image(tmp_path)

    try:
        ingest_pose_image(str(poses_json), "known", str(image_path), overwrite=False)
    except PoseIngestError as exc:
        assert "already has landmark_template" in str(exc)
    else:
        raise AssertionError("Expected overwrite guard to raise.")


def test_updater_ingests_pose_image_with_test_doubles(tmp_path: Path, monkeypatch) -> None:
    poses_json = _write_library(
        tmp_path,
        [
            {
                "pose_id": "known",
                "display_name": "Known Pose",
                "category": "standing",
                "scene_tags": ["street"],
                "object_tags": ["wall"],
                "style_tags": ["soft"],
                "instructions": ["Turn sideways"],
            }
        ],
    )
    image_path = _write_image(tmp_path)
    monkeypatch.setattr(updater, "PoseEstimator", DummyPoseEstimator)
    monkeypatch.setattr(updater, "SceneClassifier", DummySceneClassifier)

    pose = ingest_pose_image(str(poses_json), "known", str(image_path))

    assert pose["landmark_template"]["quality"]["has_person"] is True
    assert pose["landmark_template"]["derived_features"]["is_full_body"] is True
    assert pose["landmark_template"]["derived_features"]["legs_crossed"] is False
    assert pose["scene_analysis"]["primary_category"] == "street"
    assert pose["embedding_document"]

    saved = json.loads(poses_json.read_text(encoding="utf-8"))
    assert saved["poses"][0]["landmark_template"]["format"] == "mediapipe_pose_33"


def test_build_embedding_document_returns_non_empty_string() -> None:
    document = build_embedding_document(
        {
            "display_name": "Side Pose",
            "category": "standing",
            "scene_tags": ["street"],
            "object_tags": ["wall"],
            "style_tags": ["soft"],
            "instructions": ["Turn sideways"],
            "scene_analysis": {"primary_category": "street"},
            "landmark_template": {"derived_features": {"left_arm_raised": True, "is_full_body": True}},
        }
    )

    assert "Side Pose" in document
    assert "left_arm_raised=True" in document


class DummyPoseEstimator:
    def estimate(self, image: np.ndarray) -> PoseEstimationResult:
        return PoseEstimationResult(
            has_person=True,
            landmark_count=33,
            landmarks=[
                Landmark(name=name, x=x, y=y, z=0.0, visibility=visibility)
                for name, x, y, visibility in _landmarks()
            ],
        )


class DummySceneClassifier:
    def classify(self, image: np.ndarray) -> SceneClassificationResult:
        return SceneClassificationResult(
            primary_category="street",
            candidates=[SceneCandidate(label="street", score=0.9)],
            model_name="test_scene_classifier",
        )


def _write_library(tmp_path: Path, poses: list[dict]) -> Path:
    poses_json = tmp_path / "poses.json"
    poses_json.write_text(json.dumps({"schema_version": "1.0.0", "poses": poses}), encoding="utf-8")
    return poses_json


def _write_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "pose.jpg"
    Image.new("RGB", (20, 30), color=(128, 160, 190)).save(image_path)
    return image_path


def _landmarks() -> list[tuple[str, float, float, float]]:
    base = [
        ("nose", 0.5, 0.1, 0.9),
        ("left_shoulder", 0.4, 0.3, 0.9),
        ("right_shoulder", 0.6, 0.3, 0.9),
        ("left_elbow", 0.35, 0.45, 0.9),
        ("right_elbow", 0.65, 0.45, 0.9),
        ("left_wrist", 0.35, 0.6, 0.9),
        ("right_wrist", 0.65, 0.6, 0.9),
        ("left_hip", 0.43, 0.55, 0.9),
        ("right_hip", 0.57, 0.55, 0.9),
        ("left_knee", 0.43, 0.75, 0.9),
        ("right_knee", 0.57, 0.75, 0.9),
        ("left_ankle", 0.43, 0.95, 0.9),
        ("right_ankle", 0.57, 0.95, 0.9),
    ]
    filler = [(f"unused_{index}", 0.5, 0.5, 0.9) for index in range(20)]
    return base + filler
