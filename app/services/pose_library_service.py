from __future__ import annotations

from pathlib import Path
from typing import Any

from app.pose_library.loader import DEFAULT_POSE_LIBRARY_PATH, get_pose_by_id, load_pose_library


def get_all_library_poses(path: Path | str = DEFAULT_POSE_LIBRARY_PATH) -> list[dict[str, Any]]:
    return load_pose_library(path)


def get_library_pose_by_id(
    pose_id: str,
    path: Path | str = DEFAULT_POSE_LIBRARY_PATH,
) -> dict[str, Any] | None:
    if not pose_id:
        raise ValueError("pose_id cannot be empty.")
    return get_pose_by_id(pose_id, path)


def search_library_poses(
    query: str = "",
    scene_tag: str | None = None,
    style_tag: str | None = None,
    path: Path | str = DEFAULT_POSE_LIBRARY_PATH,
) -> list[dict[str, Any]]:
    poses = load_pose_library(path)
    normalized_query = query.strip().lower()
    results: list[dict[str, Any]] = []
    for pose in poses:
        if scene_tag and scene_tag not in pose.get("scene_tags", []):
            continue
        if style_tag and style_tag not in pose.get("style_tags", []):
            continue
        if normalized_query and normalized_query not in _search_text(pose):
            continue
        results.append(pose)
    return results


def _search_text(pose: dict[str, Any]) -> str:
    values = [
        pose.get("pose_id"),
        pose.get("display_name"),
        pose.get("category"),
        pose.get("scene_tags"),
        pose.get("object_tags"),
        pose.get("style_tags"),
        pose.get("instructions"),
    ]
    return " ".join(_flatten(values)).lower()


def _flatten(values: list[Any]) -> list[str]:
    flattened: list[str] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(str(item) for item in value if item is not None)
        elif value is not None:
            flattened.append(str(value))
    return flattened
