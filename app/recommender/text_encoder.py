from __future__ import annotations

from app.schemas import Pose, RecommendationRequest


class PoseTextEncoder:
    def pose_to_document(self, pose: Pose) -> str:
        return self._join_parts(
            [
                pose.display_name,
                pose.category,
                pose.scene_tags,
                pose.object_tags,
                pose.style_tags,
                pose.best_for,
                pose.instructions,
            ]
        )

    def request_to_query(self, request: RecommendationRequest) -> str:
        return self._join_parts(
            [
                request.scene.scene_tags,
                request.scene.object_tags,
                request.scene.style_tags,
                request.user_intent,
            ]
        )

    @classmethod
    def _join_parts(cls, parts: list[object]) -> str:
        tokens: list[str] = []
        for part in parts:
            if part is None:
                continue
            if isinstance(part, list):
                tokens.extend(str(item) for item in part if item)
                continue
            if part:
                tokens.append(str(part))
        return " ".join(tokens)
