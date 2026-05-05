from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.models import AppImage, Pose, PoseKeypoint  # noqa: F401
from app.schemas import Landmark, PoseEstimationResult, SceneCandidate, SceneClassificationResult
from app.services.image_analysis_service import analyze_image_path
from app.services.pose_library_service import get_all_library_poses, get_library_pose_by_id


def test_pose_library_service_reads_poses_json() -> None:
    poses = get_all_library_poses()

    assert poses
    assert get_library_pose_by_id(poses[0]["pose_id"])["pose_id"] == poses[0]["pose_id"]


def test_analyze_image_path_empty_url_raises(db_session: Session) -> None:
    try:
        analyze_image_path("", session=db_session, pose_estimator=DummyPoseEstimator())
    except ValueError as exc:
        assert "image_url cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected empty image_url to raise.")


def test_analyze_image_path_returns_analysis_without_storage(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "person.jpg"
    Image.new("RGB", (32, 24), color=(100, 120, 140)).save(image_path)

    result = analyze_image_path(
        str(image_path),
        should_store=False,
        session=db_session,
        pose_estimator=DummyPoseEstimator(),
        scene_classifier=DummySceneClassifier(),
    )

    assert result.pose.has_person is True
    assert result.scene.primary_category == "street"
    assert result.storage.requested is False
    assert db_session.query(AppImage).count() == 0


def test_analyze_image_path_persists_image_pose_and_keypoints(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "person.jpg"
    Image.new("RGB", (32, 24), color=(100, 120, 140)).save(image_path)

    result = analyze_image_path(
        str(image_path),
        should_store=True,
        session=db_session,
        pose_estimator=DummyPoseEstimator(),
        scene_classifier=DummySceneClassifier(),
    )

    assert result.pose.has_person is True
    assert result.scene.primary_category == "street"
    assert result.storage.image_id
    assert result.storage.analysis_status == "analyzed"
    assert result.storage.pose_count == 1
    assert result.storage.keypoint_count == 3

    assert db_session.query(AppImage).count() == 1
    assert db_session.query(Pose).count() == 1
    assert db_session.query(PoseKeypoint).count() == 3


def test_analyze_image_path_returns_analysis_when_storage_config_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    image_path = tmp_path / "person.jpg"
    Image.new("RGB", (32, 24), color=(100, 120, 140)).save(image_path)

    result = analyze_image_path(
        str(image_path),
        should_store=True,
        pose_estimator=DummyPoseEstimator(),
        scene_classifier=DummySceneClassifier(),
    )

    assert result.pose.has_person is True
    assert result.scene.primary_category == "street"
    assert result.storage.requested is True
    assert result.storage.analysis_status == "failed"
    assert "DATABASE_URL is not configured" in result.storage.error


class DummyPoseEstimator:
    def estimate(self, image: np.ndarray) -> PoseEstimationResult:
        return PoseEstimationResult(
            has_person=True,
            landmark_count=3,
            landmarks=[
                Landmark(name="nose", x=0.5, y=0.1, z=0.0, visibility=0.9),
                Landmark(name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.8),
                Landmark(name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.85),
            ],
        )


class DummySceneClassifier:
    def classify(self, image: np.ndarray) -> SceneClassificationResult:
        return SceneClassificationResult(
            primary_category="street",
            candidates=[SceneCandidate(label="street", score=0.9)],
            model_name="dummy_scene",
        )


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    return SessionLocal()


@pytest.fixture
def db_session() -> Session:
    session = _make_session()
    try:
        yield session
    finally:
        session.close()
