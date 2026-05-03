from __future__ import annotations

from app.pose_library.loader import load_pose_library
from app.recommender.embedding_based import EmbeddingBasedPoseRecommender
from app.schemas import RecommendationRequest, SceneContext


REQUESTS = [
    {
        "label": "cafe + window + soft + ins",
        "scene_tags": ["cafe"],
        "object_tags": ["window"],
        "style_tags": ["soft", "ins"],
        "user_intent": "soft cafe portrait by the window with an instagram style",
    },
    {
        "label": "street + city + casual + dynamic",
        "scene_tags": ["street", "city"],
        "object_tags": [],
        "style_tags": ["casual", "dynamic"],
        "user_intent": "casual dynamic city street pose",
    },
    {
        "label": "wall + cool + minimal",
        "scene_tags": ["wall"],
        "object_tags": ["wall"],
        "style_tags": ["cool", "minimal"],
        "user_intent": "minimal cool wall pose",
    },
    {
        "label": "beach + travel + romantic",
        "scene_tags": ["beach"],
        "object_tags": [],
        "style_tags": ["travel", "romantic"],
        "user_intent": "romantic travel photo at the beach",
    },
    {
        "label": "nature + flower + soft",
        "scene_tags": ["nature"],
        "object_tags": ["flower"],
        "style_tags": ["soft"],
        "user_intent": "soft nature portrait with flowers",
    },
    {
        "label": "stairs + casual + fashion",
        "scene_tags": ["stairs"],
        "object_tags": ["stairs"],
        "style_tags": ["casual", "fashion"],
        "user_intent": "casual fashion pose on stairs",
    },
]


def build_request(raw_request: dict[str, object]) -> RecommendationRequest:
    return RecommendationRequest(
        scene=SceneContext(
            scene_tags=list(raw_request["scene_tags"]),
            object_tags=list(raw_request["object_tags"]),
            style_tags=list(raw_request["style_tags"]),
        ),
        user_intent=str(raw_request["user_intent"]),
        top_k=5,
    )


def main() -> None:
    poses = load_pose_library()
    recommender = EmbeddingBasedPoseRecommender(poses)

    for raw_request in REQUESTS:
        request = build_request(raw_request)
        recommendations = recommender.recommend(request, top_k=5)

        print(f"Input query: {raw_request['label']}")
        print(f"Query text: {raw_request['user_intent']}")
        for index, pose in enumerate(recommendations, start=1):
            print(f"{index}. pose_id: {pose.pose_id}")
            print(f"   display_name: {pose.display_name}")
            print(f"   score: {pose.score}")
            print(f"   explanation: {pose.explanation}")
        print()


if __name__ == "__main__":
    main()
