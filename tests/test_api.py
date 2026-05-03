from fastapi.testclient import TestClient

from app.main import app


def test_recommend_pose_endpoint_returns_top_three_poses() -> None:
    client = TestClient(app)

    response = client.post(
        "/recommend_pose",
        json={
            "scene": {
                "scene_tags": ["urban", "street"],
                "style_tags": ["editorial", "confident"],
            },
            "person": {
                "age_group": "adult",
                "body_type": "average",
                "comfort_level": "medium",
                "mobility_notes": [],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [pose["id"] for pose in payload["poses"]] == [
        "street-stride",
        "studio-power-stance",
        "stairs-sculptural-sit",
    ]
    assert all("score" in pose for pose in payload["poses"])
