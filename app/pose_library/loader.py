from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_POSE_LIBRARY_PATH = Path(__file__).with_name("poses.json")


def load_pose_library(path: Path | str = DEFAULT_POSE_LIBRARY_PATH) -> list[dict[str, Any]]:
    library_path = Path(path)
    with library_path.open("r", encoding="utf-8") as pose_file:
        raw_library = json.load(pose_file)

    raw_poses = raw_library.get("poses") if isinstance(raw_library, dict) else raw_library
    if not isinstance(raw_poses, list):
        raise ValueError("Pose library must contain a JSON array of poses or a 'poses' array.")

    poses: list[dict[str, Any]] = []
    for pose in raw_poses:
        if not isinstance(pose, dict):
            raise ValueError("Each pose must be a JSON object.")
        pose_id = pose.get("pose_id") or pose.get("id")
        display_name = pose.get("display_name")
        if not isinstance(pose_id, str) or not pose_id:
            raise ValueError("Each pose must include a non-empty pose_id.")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError(f"Pose {pose_id} must include a non-empty display_name.")
        normalized_pose = dict(pose)
        normalized_pose["pose_id"] = pose_id
        poses.append(normalized_pose)

    return poses


def get_pose_by_id(pose_id: str, path: Path | str = DEFAULT_POSE_LIBRARY_PATH) -> dict[str, Any] | None:
    return next((pose for pose in load_pose_library(path) if pose["pose_id"] == pose_id), None)


def load_pose_library_document(path: Path | str = DEFAULT_POSE_LIBRARY_PATH) -> dict[str, Any]:
    library_path = Path(path)
    with library_path.open("r", encoding="utf-8") as pose_file:
        raw_library = json.load(pose_file)
    if isinstance(raw_library, list):
        return {"schema_version": "1.0.0", "poses": raw_library}
    if not isinstance(raw_library, dict) or not isinstance(raw_library.get("poses"), list):
        raise ValueError("Pose library must contain a JSON array of poses or a 'poses' array.")
    return raw_library


def save_pose_library_document(document: dict[str, Any], path: Path | str = DEFAULT_POSE_LIBRARY_PATH) -> None:
    library_path = Path(path)
    library_path.parent.mkdir(parents=True, exist_ok=True)
    with library_path.open("w", encoding="utf-8") as pose_file:
        json.dump(document, pose_file, ensure_ascii=False, indent=2)
        pose_file.write("\n")
