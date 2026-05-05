from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pose_library.loader import load_pose_library
from app.visualization.pose_card import PoseCardRenderer
from app.visualization.pose_renderer import PoseRenderer

client = TestClient(app)

SAMPLE_LANDMARKS = [
    {"name": "nose", "x": 0.50, "y": 0.10, "visibility": 1.0},
    {"name": "left_shoulder", "x": 0.35, "y": 0.30, "visibility": 1.0},
    {"name": "right_shoulder", "x": 0.65, "y": 0.30, "visibility": 1.0},
    {"name": "left_elbow", "x": 0.25, "y": 0.45, "visibility": 1.0},
    {"name": "right_elbow", "x": 0.75, "y": 0.45, "visibility": 1.0},
    {"name": "left_wrist", "x": 0.20, "y": 0.60, "visibility": 1.0},
    {"name": "right_wrist", "x": 0.80, "y": 0.60, "visibility": 1.0},
    {"name": "left_hip", "x": 0.42, "y": 0.56, "visibility": 1.0},
    {"name": "right_hip", "x": 0.58, "y": 0.56, "visibility": 1.0},
    {"name": "left_knee", "x": 0.40, "y": 0.78, "visibility": 1.0},
    {"name": "right_knee", "x": 0.60, "y": 0.78, "visibility": 1.0},
    {"name": "left_ankle", "x": 0.38, "y": 0.95, "visibility": 1.0},
    {"name": "right_ankle", "x": 0.62, "y": 0.95, "visibility": 1.0},
]


def test_pose_library_loader_preserves_landmarks(tmp_path: Path) -> None:
    library_path = tmp_path / "poses.json"
    library_path.write_text(
        json.dumps(
            {
                "poses": [
                    {
                        "pose_id": "sample_pose",
                        "display_name": "Sample Pose",
                        "scene_tags": ["park"],
                        "style_tags": ["natural"],
                        "instructions": ["Stand tall"],
                        "landmarks": SAMPLE_LANDMARKS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    poses = load_pose_library(library_path)

    assert poses[0]["pose_id"] == "sample_pose"
    assert poses[0]["landmarks"] == SAMPLE_LANDMARKS


def test_pose_renderer_returns_rgb_image() -> None:
    image = PoseRenderer().render_skeleton(SAMPLE_LANDMARKS)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (520, 680)


def test_pose_card_renderer_returns_png_style_card() -> None:
    image = PoseCardRenderer().render_pose_card(
        {
            "pose_id": "sample_pose",
            "display_name": "Sample Pose",
            "scene_tags": ["park"],
            "style_tags": ["natural"],
            "instructions": ["Stand tall", "Relax shoulders", "Look at camera"],
            "landmarks": SAMPLE_LANDMARKS,
        }
    )

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (900, 1200)


def test_poses_endpoint_returns_basic_pose_info() -> None:
    response = client.get("/poses")

    assert response.status_code == 200
    poses = response.json()
    assert poses
    assert {"pose_id", "display_name", "scene_tags", "style_tags", "has_landmarks"} <= set(poses[0])


def test_pose_detail_endpoint_returns_raw_pose_data() -> None:
    first_pose = client.get("/poses").json()[0]

    response = client.get(f"/poses/{first_pose['pose_id']}")

    assert response.status_code == 200
    pose = response.json()
    assert pose["pose_id"] == first_pose["pose_id"]
    assert "instructions" in pose


def test_pose_card_endpoint_returns_png() -> None:
    first_pose = client.get("/poses").json()[0]

    response = client.get(f"/poses/{first_pose['pose_id']}/card")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
