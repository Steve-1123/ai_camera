from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from app.schemas import Pose

DEFAULT_POSE_LIBRARY_PATH = Path(__file__).with_name("poses.json")


def load_pose_library(path: Union[Path, str] = DEFAULT_POSE_LIBRARY_PATH) -> list[Pose]:
    library_path = Path(path)
    with library_path.open("r", encoding="utf-8") as pose_file:
        raw_library = json.load(pose_file)

    raw_poses = raw_library.get("poses") if isinstance(raw_library, dict) else raw_library

    if not isinstance(raw_poses, list):
        raise ValueError("Pose library must contain a JSON array of poses or a 'poses' array.")

    return [Pose.model_validate(pose) for pose in raw_poses]
