from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from app.schemas import AnalyzeImageResponse, PoseEstimationResult, SceneClassificationResult
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
