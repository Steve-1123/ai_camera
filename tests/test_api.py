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
        "street_hands_pocket_004",
        "street_side_hair_touch_002",
        "street_cross_leg_003",
    ]
    assert all("score" in pose for pose in payload["poses"])


def test_recommend_pose_embedding_endpoint_returns_embedding_results() -> None:
    client = TestClient(app)

    response = client.post(
        "/recommend_pose_embedding",
        json={
            "scene": {
                "scene_tags": ["cafe"],
                "object_tags": ["cup"],
                "style_tags": ["cozy"],
            },
            "user_intent": "holding a warm drink at a table",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["poses"][0]["pose_id"] == "cafe_cup_010"
    assert set(payload["poses"][0]) == {
        "pose_id",
        "display_name",
        "score",
        "instructions",
        "explanation",
    }
