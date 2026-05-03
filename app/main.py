from __future__ import annotations

from fastapi import FastAPI

from app.pose_library.loader import load_pose_library
from app.recommender.rule_based import RuleBasedPoseRecommender
from app.schemas import RecommendationRequest, RecommendationResponse

app = FastAPI(
    title="AI Photo Pose",
    description="MVP backend for rule-based photography pose recommendations.",
    version="0.1.0",
)

pose_library = load_pose_library()
recommender = RuleBasedPoseRecommender(pose_library)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend_pose", response_model=RecommendationResponse)
def recommend_pose(request: RecommendationRequest) -> RecommendationResponse:
    poses = recommender.recommend(request, limit=3)
    return RecommendationResponse(poses=poses)
