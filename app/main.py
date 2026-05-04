from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import AnalyzeImageResponse
from app.vision.image_loader import ImageLoadError, load_upload_image
from app.vision.pose_estimator import PoseEstimator
from app.vision.scene_classifier import SceneClassifier

app = FastAPI(
    title="AI Camera",
    description="MVP backend for image upload, pose landmarks, and scene classification.",
    version="0.2.0",
)

pose_estimator = PoseEstimator()
scene_classifier = SceneClassifier()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze_image", response_model=AnalyzeImageResponse)
async def analyze_image(
    image: Annotated[UploadFile, File(description="Image file to analyze.")],
) -> AnalyzeImageResponse:
    try:
        loaded_image = await load_upload_image(image)
    except ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pose = pose_estimator.estimate(loaded_image.image)
    scene = scene_classifier.classify(loaded_image.image)

    return AnalyzeImageResponse(
        image_info=loaded_image.info,
        pose=pose,
        scene=scene,
    )

