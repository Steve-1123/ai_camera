from __future__ import annotations

from typing import Optional

from app.legacy.recommender.embedding_index import PoseEmbeddingIndex
from app.legacy.recommender.text_encoder import PoseTextEncoder
from app.legacy.schemas import EmbeddingPoseRecommendation, Pose, RecommendationRequest


class EmbeddingBasedPoseRecommender:
    def __init__(
        self,
        poses: list[Pose],
        novelty_weight: float = 0.15,
        popularity_penalty: float = 0.05,
        text_encoder: Optional[PoseTextEncoder] = None,
    ) -> None:
        self._poses = poses
        self._novelty_weight = novelty_weight
        self._popularity_penalty = popularity_penalty
        self._text_encoder = text_encoder or PoseTextEncoder()
        documents = [self._text_encoder.pose_to_document(pose) for pose in poses]
        self._index = PoseEmbeddingIndex(poses, documents)

    def recommend(
        self,
        request: RecommendationRequest,
        top_k: int = 3,
    ) -> list[EmbeddingPoseRecommendation]:
        query = self._text_encoder.request_to_query(request)
        scored = [
            self._build_recommendation(pose, similarity)
            for pose, similarity in self._index.search(query)
        ]
        scored.sort(key=lambda recommendation: recommendation.score, reverse=True)
        return scored[:top_k]

    def _build_recommendation(
        self,
        pose: Pose,
        embedding_similarity: float,
    ) -> EmbeddingPoseRecommendation:
        novelty_bonus = self._novelty_weight * pose.novelty
        popularity_cost = self._popularity_penalty * pose.popularity
        score = round(0.75 * embedding_similarity + novelty_bonus - popularity_cost, 3)

        return EmbeddingPoseRecommendation(
            pose_id=pose.pose_id,
            display_name=pose.display_name,
            score=score,
            instructions=pose.instructions,
            explanation=(
                f"score={score}: embedding_similarity={embedding_similarity:.3f} weighted by 0.75, "
                f"novelty_bonus={novelty_bonus:.3f}, popularity_penalty={popularity_cost:.3f}."
            ),
        )
