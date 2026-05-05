from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pose_library.updater import PoseIngestError, ingest_pose_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a reference image into a stored pose.")
    parser.add_argument("--pose-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--poses-json", default="app/pose_library/poses.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        pose = ingest_pose_image(
            poses_json_path=args.poses_json,
            pose_id=args.pose_id,
            image_path=args.image,
            overwrite=args.overwrite,
        )
    except PoseIngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    template = pose["landmark_template"]
    source = template["source"]
    quality = template["quality"]
    scene = pose["scene_analysis"]

    print(f"pose_id: {pose['pose_id']}")
    print(f"image size: {source['image_width']}x{source['image_height']}")
    print(f"has_person: {quality['has_person']}")
    print(f"landmark_count: {quality['landmark_count']}")
    print(f"mean_visibility: {quality['mean_visibility']}")
    print(f"scene primary_category: {scene['primary_category']}")
    print("top scene candidates:")
    for candidate in scene.get("candidates", [])[:5]:
        print(f"- {candidate['label']}: {candidate['score']}")
    print(f"saved path: {args.poses_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
