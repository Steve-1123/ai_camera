from __future__ import annotations

import argparse

from fastapi import FastAPI, HTTPException

from app.schemas import (
    AnalyzeImagePathRequest,
    AnalyzeImagePathResponse,
    PoseEstimationResult,
)
from app.services.image_analysis_service import ImageAnalysisError, analyze_image_path


app = FastAPI(
    title="AI Camera",
    description="MVP backend for image path analysis, pose landmarks, scene classification, and optional storage.",
    version="0.2.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze_image_path", response_model=AnalyzeImagePathResponse)
def analyze_image_path_endpoint(request: AnalyzeImagePathRequest) -> AnalyzeImagePathResponse:
    try:
        return analyze_image_path(
            image_url=request.image_url,
            should_store=request.should_store,
            source=request.source,
            detector=request.detector,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImageAnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Camera CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Create database tables from SQLAlchemy models.")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze an image and store results in MySQL.")
    analyze_parser.add_argument("--image-url", required=True)
    analyze_parser.add_argument("--source", default="upload")
    analyze_parser.add_argument("--detector", default="local_pose_estimator")

    subparsers.add_parser("list-library-poses", help="List poses from app/pose_library/poses.json.")
    args = parser.parse_args()

    try:
        if args.command == "init-db":
            from app.core.database import init_db

            init_db()
            print("database initialized")
            return 0

        if args.command == "analyze":
            result = analyze_image_path(
                image_url=args.image_url,
                should_store=True,
                source=args.source,
                detector=args.detector,
            )
            _print_analysis_result(result)
            return 0

        if args.command == "list-library-poses":
            from app.services.pose_library_service import get_all_library_poses

            poses = get_all_library_poses()
            for pose in poses:
                print(f"{pose['pose_id']}\t{pose['display_name']}")
            print(f"total: {len(poses)}")
            return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    parser.print_help()
    return 1


def _print_analysis_result(result: AnalyzeImagePathResponse) -> None:
    print(f"image_id: {result.storage.image_id}")
    print(f"analysis_status: {result.storage.analysis_status}")
    print(f"pose count: {result.storage.pose_count}")
    print("pose_name: detected_person" if result.pose.has_person else "pose_name: none")
    print(f"confidence: {_mean_visibility(result.pose)}")
    print(f"keypoints count: {result.storage.keypoint_count}")
    if result.storage.error:
        print(f"storage_error: {result.storage.error}")


def _mean_visibility(pose: PoseEstimationResult) -> float | None:
    if not pose.landmarks:
        return None
    return round(sum(landmark.visibility for landmark in pose.landmarks) / len(pose.landmarks), 4)


if __name__ == "__main__":
    raise SystemExit(main())
