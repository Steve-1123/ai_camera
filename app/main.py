from __future__ import annotations

from fastapi import FastAPI

from app.pose_library.loader import load_pose_library
from app.recommender.embedding_based import EmbeddingBasedPoseRecommender
from app.recommender.rule_based import RuleBasedPoseRecommender
from app.schemas import (
    EmbeddingRecommendationResponse,
    RecommendationRequest,
    RecommendationResponse,
)

app = FastAPI(
    title="AI Photo Pose",
    description="MVP backend for rule-based photography pose recommendations.",
    version="0.1.0",
)

pose_library = load_pose_library()
recommender = RuleBasedPoseRecommender(pose_library)
embedding_recommender = EmbeddingBasedPoseRecommender(pose_library)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/recommend_pose", response_model=RecommendationResponse)
def recommend_pose(request: RecommendationRequest) -> RecommendationResponse:
    poses = recommender.recommend(request, top_k=request.top_k)
    return RecommendationResponse(poses=poses)

@app.post("/recommend_pose_embedding", response_model=EmbeddingRecommendationResponse)
def recommend_pose_embedding(request: RecommendationRequest) -> EmbeddingRecommendationResponse:
    poses = embedding_recommender.recommend(request, top_k=request.top_k)
    return EmbeddingRecommendationResponse(poses=poses)
