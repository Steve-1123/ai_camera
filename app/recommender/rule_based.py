from app.schemas import Pose, RecommendationRequest


class RuleBasedPoseRecommender:
    def __init__(self, poses: list[Pose]) -> None:
        self._poses = poses

    def recommend(self, request: RecommendationRequest, top_k: int = 3) -> list[Pose]:
        scored_poses = [
            pose.model_copy(
                update={
                    "score": self.score_pose(pose, request),
                    "explanation": self.explain_pose(pose, request),
                }
            )
            for pose in self._poses
        ]
        scored_poses.sort(key=lambda pose: pose.score or 0.0, reverse=True)
        return scored_poses[:top_k]

    def score_pose(self, pose: Pose, request: RecommendationRequest) -> float:
        scene_match_score = self._tag_match_score(request.scene.scene_tags, pose.scene_tags)
        style_match_score = self._tag_match_score(request.scene.style_tags, pose.style_tags)
        object_match_score = self._tag_match_score(request.scene.object_tags, pose.object_tags)

        return round(
            3.0 * scene_match_score
            + 2.0 * style_match_score
            + 1.0 * object_match_score
            + 1.0 * pose.novelty
            - 0.8 * pose.popularity
            - 0.3 * pose.difficulty,
            3,
        )

    def explain_pose(self, pose: Pose, request: RecommendationRequest) -> str:
        scene_match_score = self._tag_match_score(request.scene.scene_tags, pose.scene_tags)
        style_match_score = self._tag_match_score(request.scene.style_tags, pose.style_tags)
        object_match_score = self._tag_match_score(request.scene.object_tags, pose.object_tags)
        score = self.score_pose(pose, request)

        return (
            f"score={score}: scene_match={scene_match_score:.2f} weighted by 3.0, "
            f"style_match={style_match_score:.2f} weighted by 2.0, "
            f"object_match={object_match_score:.2f} weighted by 1.0, "
            f"novelty={pose.novelty:.2f}, popularity_penalty={0.8 * pose.popularity:.2f}, "
            f"difficulty_penalty={0.3 * pose.difficulty:.2f}."
        )

    @staticmethod
    def _tag_match_score(requested_tags: list[str], pose_tags: list[str]) -> float:
        requested = {tag.lower() for tag in requested_tags}
        if not requested:
            return 0.0

        available = {tag.lower() for tag in pose_tags}
        return len(requested & available) / len(requested)
