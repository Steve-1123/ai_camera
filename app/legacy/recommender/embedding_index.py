from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.legacy.schemas import Pose


class PoseEmbeddingIndex:
    def __init__(self, poses: list[Pose], documents: list[str]) -> None:
        if len(poses) != len(documents):
            raise ValueError("Pose and document counts must match.")
        if not poses:
            raise ValueError("At least one pose is required to build an embedding index.")

        self._poses = poses
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(documents)

    def search(self, query: str) -> list[tuple[Pose, float]]:
        if not query.strip():
            return [(pose, 0.0) for pose in self._poses]

        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._matrix).flatten()
        results = [
            (pose, float(similarity))
            for pose, similarity in zip(self._poses, similarities)
        ]
        return sorted(results, key=lambda item: item[1], reverse=True)
