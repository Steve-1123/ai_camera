from app.schemas import Pose, RecommendationRequest


DIFFICULTY_PENALTIES = {
    "easy": 0.0,
    "medium": 1.0,
    "hard": 2.0,
}

COMFORT_DIFFICULTY_LIMITS = {
    "low": {"easy"},
    "medium": {"easy", "medium"},
    "high": {"easy", "medium", "hard"},
}


class RuleBasedPoseRecommender:
    def __init__(self, poses: list[Pose]) -> None:
        self._poses = poses

    def recommend(self, request: RecommendationRequest, limit: int = 3) -> list[Pose]:
        scored_poses = [
            pose.model_copy(update={"score": self.score_pose(pose, request)})
            for pose in self._poses
        ]
        scored_poses.sort(key=lambda pose: (pose.score or 0.0, -pose.popularity), reverse=True)
        return scored_poses[:limit]

    def score_pose(self, pose: Pose, request: RecommendationRequest) -> float:
        scene_score = self._tag_overlap_score(request.scene.scene_tags, pose.scene_tags, weight=4.0)
        style_score = self._tag_overlap_score(request.scene.style_tags, pose.style_tags, weight=3.0)
        difficulty_penalty = DIFFICULTY_PENALTIES[pose.difficulty]
        popularity_penalty = pose.popularity / 100.0

        person_fit_bonus = self._person_fit_bonus(pose, request)
        mobility_penalty = self._mobility_penalty(pose, request)
        comfort_penalty = self._comfort_penalty(pose, request)

        return round(
            scene_score
            + style_score
            + person_fit_bonus
            - difficulty_penalty
            - popularity_penalty
            - mobility_penalty
            - comfort_penalty,
            3,
        )

    @staticmethod
    def _tag_overlap_score(requested_tags: list[str], pose_tags: list[str], weight: float) -> float:
        requested = {tag.lower() for tag in requested_tags}
        available = {tag.lower() for tag in pose_tags}
        return len(requested & available) * weight

    @staticmethod
    def _person_fit_bonus(pose: Pose, request: RecommendationRequest) -> float:
        person = request.person
        bonus = 0.0

        if person.body_type and ("all" in pose.suitable_body_types or person.body_type in pose.suitable_body_types):
            bonus += 0.5

        if person.age_group and ("all" in pose.suitable_age_groups or person.age_group in pose.suitable_age_groups):
            bonus += 0.5

        return bonus

    @staticmethod
    def _mobility_penalty(pose: Pose, request: RecommendationRequest) -> float:
        mobility_notes = {note.lower() for note in request.person.mobility_notes}
        requirements = {requirement.lower() for requirement in pose.mobility_requirements}

        if not mobility_notes:
            return 0.0

        blocked_requirements = mobility_notes & requirements
        return len(blocked_requirements) * 3.0

    @staticmethod
    def _comfort_penalty(pose: Pose, request: RecommendationRequest) -> float:
        allowed_difficulties = COMFORT_DIFFICULTY_LIMITS[request.person.comfort_level]
        if pose.difficulty in allowed_difficulties:
            return 0.0
        return 2.5
