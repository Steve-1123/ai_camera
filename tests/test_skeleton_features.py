from __future__ import annotations

from app.features.skeleton_features import compute_skeleton_features


def test_skeleton_features_missing_points_do_not_crash() -> None:
    features = compute_skeleton_features(
        [
            {"name": "nose", "x": 0.5, "y": 0.1, "visibility": 0.9},
            {"name": "left_shoulder", "x": 0.4, "y": 0.3, "visibility": 0.9},
            {"name": "right_shoulder", "x": 0.6, "y": 0.3, "visibility": 0.1},
        ]
    )

    assert features["bbox"] is not None
    assert features["person_center"] is not None
    assert features["right_arm_raised"] is False
    assert features["is_full_body"] is False
