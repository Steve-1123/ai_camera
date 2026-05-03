from app.pose_library.loader import load_pose_library
from app.recommender.rule_based import RuleBasedPoseRecommender
from app.schemas import PersonContext, RecommendationRequest, SceneContext


def test_recommendation_ranking_prefers_scene_and_style_matches() -> None:
    recommender = RuleBasedPoseRecommender(load_pose_library())
    request = RecommendationRequest(
        scene=SceneContext(
            scene_tags=["urban", "street"],
            style_tags=["editorial", "confident"],
        ),
        person=PersonContext(age_group="adult", body_type="average", comfort_level="medium"),
    )

    recommendations = recommender.recommend(request)

    assert [pose.id for pose in recommendations] == [
        "street-stride",
        "studio-power-stance",
        "stairs-sculptural-sit",
    ]
    assert len(recommendations) == 3
    assert recommendations[0].score is not None
    assert recommendations[0].score > recommendations[1].score


def test_low_comfort_penalizes_harder_poses() -> None:
    recommender = RuleBasedPoseRecommender(load_pose_library())
    request = RecommendationRequest(
        scene=SceneContext(
            scene_tags=["urban", "stairs", "architecture"],
            style_tags=["editorial", "dramatic", "fashion"],
        ),
        person=PersonContext(age_group="adult", body_type="average", comfort_level="low"),
    )

    recommendations = recommender.recommend(request)

    assert recommendations[0].id == "stairs-sculptural-sit"
    assert recommendations[0].score is not None
    assert recommendations[0].score < 18
