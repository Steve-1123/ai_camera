from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    filename: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class Landmark(BaseModel):
    name: str
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    z: float
    visibility: float = Field(ge=0.0, le=1.0)


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


class AnalyzeImagePathRequest(BaseModel):
    image_url: str
    should_store: bool = False
    source: str = "upload"
    detector: str = "local_pose_estimator"


class ImageStorageResult(BaseModel):
    requested: bool
    image_id: Optional[int] = None
    analysis_status: Optional[str] = None
    pose_count: int = 0
    keypoint_count: int = 0
    error: Optional[str] = None


class AnalyzeImagePathResponse(BaseModel):
    image_info: ImageInfo
    pose: PoseEstimationResult
    scene: SceneClassificationResult
    storage: ImageStorageResult
