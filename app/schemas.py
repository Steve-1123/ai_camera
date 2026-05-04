from __future__ import annotations

from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    filename: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class Landmark(BaseModel):
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimationResult(BaseModel):
    has_person: bool
    landmark_count: int = Field(ge=0)
    landmarks: list[Landmark] = Field(default_factory=list)


class SceneCandidate(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class SceneClassificationResult(BaseModel):
    primary_category: str
    candidates: list[SceneCandidate] = Field(default_factory=list)
    model_name: str


class AnalyzeImageResponse(BaseModel):
    image_info: ImageInfo
    pose: PoseEstimationResult
    scene: SceneClassificationResult

