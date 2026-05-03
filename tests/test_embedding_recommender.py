from app.pose_library.loader import load_pose_library
from app.recommender.embedding_based import EmbeddingBasedPoseRecommender
from app.recommender.text_encoder import PoseTextEncoder
from app.schemas import RecommendationRequest, SceneContext


def test_text_encoder_builds_pose_document_from_pose_fields() -> None:
    pose = load_pose_library()[0]
    document = PoseTextEncoder().pose_to_document(pose)

    assert pose.display_name in document
    assert pose.category in document
    assert pose.scene_tags[0] in document
    assert pose.instructions[0] in document


def test_embedding_recommender_prefers_matching_pose_context() -> None:
    recommender = EmbeddingBasedPoseRecommender(load_pose_library())
    request = RecommendationRequest(
        scene=SceneContext(
            scene_tags=["cafe"],
            object_tags=["cup"],
            style_tags=["cozy"],
        ),
        user_intent="holding a warm drink at a table",
    )

    recommendations = recommender.recommend(request)

    assert recommendations[0].pose_id == "cafe_cup_010"
    assert recommendations[0].display_name
    assert recommendations[0].instructions
    assert "embedding_similarity" in recommendations[0].explanation
    assert recommendations[0].score > recommendations[1].score
