from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.pose_library.loader import load_pose_library
from app.schemas import AnalyzeImageResponse, PoseEstimationResult, SceneClassificationResult
from app.visualization.pose_card import PoseCardRenderer
from app.vision.image_loader import ImageLoadError, load_upload_image
from app.vision.pose_estimator import PoseEstimator
from app.vision.scene_classifier import SceneClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pose_estimator = None
    app.state.scene_classifier = None
    app.state.pose_lock = threading.Lock()
    app.state.scene_lock = threading.Lock()
    try:
        yield
    finally:
        for service_name in ("pose_estimator", "scene_classifier"):
            service = getattr(app.state, service_name, None)
            if service is not None and hasattr(service, "close"):
                service.close()


app = FastAPI(
    title="AI Camera",
    description="MVP backend for image upload, pose landmarks, and scene classification.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/poses")
def list_poses() -> list[dict[str, object]]:
    return [
        {
            "pose_id": pose["pose_id"],
            "display_name": pose["display_name"],
            "scene_tags": pose.get("scene_tags", []),
            "style_tags": pose.get("style_tags", []),
            "has_landmarks": bool(pose.get("landmarks")),
        }
        for pose in _load_pose_library_or_500()
    ]


@app.get("/poses/{pose_id}")
def read_pose(pose_id: str) -> dict[str, object]:
    return _get_pose_or_404(pose_id)


@app.get("/poses/{pose_id}/card")
def read_pose_card(pose_id: str) -> StreamingResponse:
    pose = _get_pose_or_404(pose_id)

    image = PoseCardRenderer().render_pose_card(pose)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/analyze_image", response_model=AnalyzeImageResponse)
async def analyze_image(
    request: Request,
    image: Annotated[UploadFile, File(description="Image file to analyze.")],
) -> AnalyzeImageResponse:
    try:
        loaded_image = await load_upload_image(image)
    except ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pose, scene = await asyncio.gather(
        asyncio.to_thread(_estimate_pose, request.app, loaded_image.image),
        asyncio.to_thread(_classify_scene, request.app, loaded_image.image),
    )

    return AnalyzeImageResponse(
        image_info=loaded_image.info,
        pose=pose,
        scene=scene,
    )


def _estimate_pose(app: FastAPI, image: np.ndarray) -> PoseEstimationResult:
    with app.state.pose_lock:
        if app.state.pose_estimator is None:
            app.state.pose_estimator = PoseEstimator()
        estimator = app.state.pose_estimator
    return estimator.estimate(image)


def _classify_scene(app: FastAPI, image: np.ndarray) -> SceneClassificationResult:
    with app.state.scene_lock:
        if app.state.scene_classifier is None:
            app.state.scene_classifier = SceneClassifier()
        classifier = app.state.scene_classifier
    return classifier.classify(image)


def _load_pose_library_or_500() -> list[dict[str, object]]:
    try:
        return load_pose_library()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Pose library could not be loaded.") from exc


def _get_pose_or_404(pose_id: str) -> dict[str, object]:
    for pose in _load_pose_library_or_500():
        if pose["pose_id"] == pose_id:
            return pose
    raise HTTPException(status_code=404, detail="Pose not found.")
