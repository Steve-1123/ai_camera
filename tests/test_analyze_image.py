from __future__ import annotations

import asyncio
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.main import app
from app.schemas import (
    AnalyzeImagePathResponse,
    ImageInfo,
    ImageStorageResult,
    PoseEstimationResult,
    SceneCandidate,
    SceneClassificationResult,
)
from app.vision.image_loader import ImageLoadError, load_upload_image
from app.vision.pose_estimator import PoseEstimator


client = TestClient(app)


class DummyUploadFile:
    def __init__(self, content: bytes, filename: str = "test.png") -> None:
        self._content = content
        self.filename = filename
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._content):
            return b""
        if size < 0:
            size = len(self._content) - self._offset
        start = self._offset
        self._offset = min(len(self._content), self._offset + size)
        return self._content[start:self._offset]


def make_image_bytes(color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_image_path_endpoint_returns_analysis_when_storage_disabled(monkeypatch) -> None:
    def fake_analyze_image_path(**kwargs):
        return AnalyzeImagePathResponse(
            image_info=ImageInfo(filename="sample.jpg", width=16, height=12),
            pose=PoseEstimationResult(has_person=False, landmark_count=0, landmarks=[]),
            scene=SceneClassificationResult(
                primary_category="street",
                candidates=[SceneCandidate(label="street", score=0.9)],
                model_name="test_scene",
            ),
            storage=ImageStorageResult(requested=False),
        )

    monkeypatch.setattr(main_module, "analyze_image_path", fake_analyze_image_path)

    response = client.post(
        "/analyze_image_path",
        json={"image_url": "images/sample.jpg", "should_store": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["pose"]["has_person"] is False
    assert body["scene"]["primary_category"] == "street"
    assert body["storage"]["requested"] is False


def test_analyze_image_path_endpoint_rejects_empty_image_url() -> None:
    response = client.post(
        "/analyze_image_path",
        json={"image_url": "", "should_store": False},
    )

    assert response.status_code == 400
    assert "image_url cannot be empty" in response.json()["detail"]


def test_image_loader_reads_image_metadata_and_rgb_array() -> None:
    upload = DummyUploadFile(make_image_bytes((12, 34, 56)), filename="sample.png")

    loaded = asyncio.run(load_upload_image(upload))  # type: ignore[arg-type]

    assert loaded.info.filename == "sample.png"
    assert loaded.info.width == 16
    assert loaded.info.height == 12
    assert loaded.image.shape == (12, 16, 3)
    assert loaded.image.dtype == np.uint8


def test_image_loader_rejects_oversized_upload() -> None:
    upload = DummyUploadFile(b"x" * (21 * 1024 * 1024), filename="large.jpg")

    try:
        asyncio.run(load_upload_image(upload))  # type: ignore[arg-type]
    except ImageLoadError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("Expected oversized upload to be rejected.")


def test_pose_estimator_blank_image_does_not_crash() -> None:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    result = PoseEstimator().estimate(image)

    assert result.landmark_count == len(result.landmarks)
    if not result.has_person:
        assert result.landmarks == []


def test_analyze_image_path_endpoint_rejects_invalid_image_path() -> None:
    response = client.post(
        "/analyze_image_path",
        json={"image_url": "missing-image.jpg", "should_store": False},
    )

    assert response.status_code == 400
    assert "Image could not be loaded" in response.json()["detail"]
