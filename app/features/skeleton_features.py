from __future__ import annotations

from math import hypot
from typing import Any

MIN_VISIBILITY = 0.2


def compute_skeleton_features(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    points = {
        str(landmark.get("name")): landmark
        for landmark in landmarks
        if isinstance(landmark, dict) and _is_visible(landmark)
    }
    coordinates = [(float(point["x"]), float(point["y"])) for point in points.values() if _has_xy(point)]

    bbox = _bbox(coordinates)
    body_height_ratio = bbox["height"] if bbox else 0.0
    shoulder_width_ratio = _distance(points.get("left_shoulder"), points.get("right_shoulder"))
    hip_width_ratio = _distance(points.get("left_hip"), points.get("right_hip"))

    left_wrist = points.get("left_wrist")
    right_wrist = points.get("right_wrist")
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    nose = points.get("nose")

    features = {
        "bbox": bbox,
        "person_center": _center(coordinates),
        "body_height_ratio": round(body_height_ratio, 4),
        "shoulder_width_ratio": round(shoulder_width_ratio, 4),
        "hip_width_ratio": round(hip_width_ratio, 4),
        "left_arm_raised": _is_above(left_wrist, left_shoulder),
        "right_arm_raised": _is_above(right_wrist, right_shoulder),
        "hands_near_face": _near(left_wrist, nose, 0.18) or _near(right_wrist, nose, 0.18),
        "legs_crossed": _legs_crossed(points),
        "is_sitting": _is_sitting(points),
        "is_full_body": bool(points.get("nose") and points.get("left_ankle") and points.get("right_ankle")),
    }
    return features


def _is_visible(landmark: dict[str, Any]) -> bool:
    visibility = landmark.get("visibility")
    if visibility is None:
        return True
    try:
        return float(visibility) >= MIN_VISIBILITY
    except (TypeError, ValueError):
        return False


def _has_xy(landmark: dict[str, Any] | None) -> bool:
    if landmark is None:
        return False
    return isinstance(landmark.get("x"), (int, float)) and isinstance(landmark.get("y"), (int, float))


def _bbox(coordinates: list[tuple[float, float]]) -> dict[str, float] | None:
    if not coordinates:
        return None
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))
    return {
        "min_x": round(min_x, 4),
        "min_y": round(min_y, 4),
        "max_x": round(max_x, 4),
        "max_y": round(max_y, 4),
        "width": round(max_x - min_x, 4),
        "height": round(max_y - min_y, 4),
    }


def _center(coordinates: list[tuple[float, float]]) -> dict[str, float] | None:
    if not coordinates:
        return None
    return {
        "x": round(sum(point[0] for point in coordinates) / len(coordinates), 4),
        "y": round(sum(point[1] for point in coordinates) / len(coordinates), 4),
    }


def _distance(first: dict[str, Any] | None, second: dict[str, Any] | None) -> float:
    if not _has_xy(first) or not _has_xy(second):
        return 0.0
    return hypot(float(first["x"]) - float(second["x"]), float(first["y"]) - float(second["y"]))


def _is_above(point: dict[str, Any] | None, reference: dict[str, Any] | None) -> bool:
    if not _has_xy(point) or not _has_xy(reference):
        return False
    return float(point["y"]) < float(reference["y"])


def _near(first: dict[str, Any] | None, second: dict[str, Any] | None, threshold: float) -> bool:
    if not _has_xy(first) or not _has_xy(second):
        return False
    return _distance(first, second) <= threshold


def _legs_crossed(points: dict[str, dict[str, Any]]) -> bool:
    left_ankle = points.get("left_ankle")
    right_ankle = points.get("right_ankle")
    left_knee = points.get("left_knee")
    right_knee = points.get("right_knee")
    ankles_crossed = (
        _has_xy(left_ankle)
        and _has_xy(right_ankle)
        and float(left_ankle["x"]) > float(right_ankle["x"])
    )
    knees_crossed = (
        _has_xy(left_knee)
        and _has_xy(right_knee)
        and float(left_knee["x"]) > float(right_knee["x"])
    )
    return bool(ankles_crossed or knees_crossed)


def _is_sitting(points: dict[str, dict[str, Any]]) -> bool:
    left_hip = points.get("left_hip")
    right_hip = points.get("right_hip")
    left_knee = points.get("left_knee")
    right_knee = points.get("right_knee")
    left_ankle = points.get("left_ankle")
    right_ankle = points.get("right_ankle")
    if not all(_has_xy(point) for point in (left_hip, right_hip, left_knee, right_knee)):
        return False

    hip_y = (float(left_hip["y"]) + float(right_hip["y"])) / 2
    knee_y = (float(left_knee["y"]) + float(right_knee["y"])) / 2
    ankle_y = None
    if _has_xy(left_ankle) and _has_xy(right_ankle):
        ankle_y = (float(left_ankle["y"]) + float(right_ankle["y"])) / 2

    compact_legs = ankle_y is not None and abs(ankle_y - knee_y) < 0.16
    bent_legs = knee_y > hip_y and (knee_y - hip_y) < 0.28
    return bool(bent_legs or compact_legs)
