from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

CANVAS_SIZE = (520, 680)
PADDING = 72

LANDMARK_NAMES = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

SKELETON_CONNECTIONS = [
    ("left_ear", "left_eye"),
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

FALLBACK_LANDMARKS = [
    {"name": "nose", "x": 0.5, "y": 0.12, "visibility": 1.0},
    {"name": "left_eye", "x": 0.47, "y": 0.10, "visibility": 1.0},
    {"name": "right_eye", "x": 0.53, "y": 0.10, "visibility": 1.0},
    {"name": "left_ear", "x": 0.42, "y": 0.12, "visibility": 1.0},
    {"name": "right_ear", "x": 0.58, "y": 0.12, "visibility": 1.0},
    {"name": "left_shoulder", "x": 0.36, "y": 0.28, "visibility": 1.0},
    {"name": "right_shoulder", "x": 0.64, "y": 0.28, "visibility": 1.0},
    {"name": "left_elbow", "x": 0.28, "y": 0.45, "visibility": 1.0},
    {"name": "right_elbow", "x": 0.72, "y": 0.45, "visibility": 1.0},
    {"name": "left_wrist", "x": 0.32, "y": 0.62, "visibility": 1.0},
    {"name": "right_wrist", "x": 0.68, "y": 0.62, "visibility": 1.0},
    {"name": "left_hip", "x": 0.42, "y": 0.56, "visibility": 1.0},
    {"name": "right_hip", "x": 0.58, "y": 0.56, "visibility": 1.0},
    {"name": "left_knee", "x": 0.40, "y": 0.78, "visibility": 1.0},
    {"name": "right_knee", "x": 0.60, "y": 0.78, "visibility": 1.0},
    {"name": "left_ankle", "x": 0.38, "y": 0.96, "visibility": 1.0},
    {"name": "right_ankle", "x": 0.62, "y": 0.96, "visibility": 1.0},
]


class PoseRenderer:
    def render_skeleton(self, landmarks: Any) -> Image.Image:
        points = _extract_points(landmarks)
        if not points:
            points = _extract_points(FALLBACK_LANDMARKS)
        normalized_points = _normalize_points(points)

        image = Image.new("RGB", CANVAS_SIZE, "white")
        draw = ImageDraw.Draw(image)
        for start_name, end_name in SKELETON_CONNECTIONS:
            start = normalized_points.get(start_name)
            end = normalized_points.get(end_name)
            if start and end:
                draw.line([start, end], fill=(37, 99, 235), width=9, joint="curve")

        nose = normalized_points.get("nose")
        if nose:
            radius = 38
            draw.ellipse(
                [nose[0] - radius, nose[1] - radius, nose[0] + radius, nose[1] + radius],
                outline=(15, 23, 42),
                width=7,
            )

        for point in normalized_points.values():
            radius = 10
            draw.ellipse(
                [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                fill=(15, 23, 42),
            )

        return image


def _extract_points(landmarks: Any) -> dict[str, tuple[float, float]]:
    if landmarks is None:
        return {}
    if hasattr(landmarks, "landmark"):
        landmarks = landmarks.landmark
    if not isinstance(landmarks, list):
        try:
            landmarks = list(landmarks)
        except TypeError:
            return {}

    points: dict[str, tuple[float, float]] = {}
    for index, landmark in enumerate(landmarks):
        name = _read_landmark_value(landmark, "name")
        if not name:
            name = LANDMARK_NAMES.get(index)
        if not name:
            continue
        x = _read_landmark_value(landmark, "x")
        y = _read_landmark_value(landmark, "y")
        visibility = _read_landmark_value(landmark, "visibility", 1.0)
        if x is None or y is None:
            continue
        if isinstance(visibility, (int, float)) and visibility < 0.2:
            continue
        points[str(name)] = (float(x), float(y))
    return points


def _read_landmark_value(landmark: Any, key: str, default: Any = None) -> Any:
    if isinstance(landmark, dict):
        return landmark.get(key, default)
    return getattr(landmark, key, default)


def _normalize_points(points: dict[str, tuple[float, float]]) -> dict[str, tuple[int, int]]:
    xs = [point[0] for point in points.values()]
    ys = [point[1] for point in points.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 0.01)
    height = max(max_y - min_y, 0.01)

    available_width = CANVAS_SIZE[0] - PADDING * 2
    available_height = CANVAS_SIZE[1] - PADDING * 2
    scale = min(available_width / width, available_height / height)
    scaled_width = width * scale
    scaled_height = height * scale
    offset_x = (CANVAS_SIZE[0] - scaled_width) / 2
    offset_y = (CANVAS_SIZE[1] - scaled_height) / 2

    return {
        name: (
            int(offset_x + (point[0] - min_x) * scale),
            int(offset_y + (point[1] - min_y) * scale),
        )
        for name, point in points.items()
    }
