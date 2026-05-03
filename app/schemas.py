from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Pose(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pose_id: str = Field(validation_alias=AliasChoices("pose_id", "id"), serialization_alias="id")
    display_name: str
    category: Optional[str] = None
    scene_tags: list[str] = Field(default_factory=list)
    object_tags: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    difficulty: int = Field(ge=1, le=5)
    popularity: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    pose_features: dict[str, object] = Field(default_factory=dict)
    instructions: list[str] = Field(default_factory=list)
    delta_rules: dict[str, str] = Field(default_factory=dict)
    negative_constraints: list[str] = Field(default_factory=list)
    best_for: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    score: Optional[float] = None
    explanation: Optional[str] = None

    @property
    def id(self) -> str:
        return self.pose_id


class SceneContext(BaseModel):
    scene_tags: list[str] = Field(default_factory=list)
    object_tags: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)


class PersonContext(BaseModel):
    age_group: Optional[str] = None
    body_type: Optional[str] = None
    comfort_level: str = "medium"
    mobility_notes: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    scene: SceneContext
    person: PersonContext = Field(default_factory=PersonContext)
    top_k: int = Field(default=3, ge=1, le=20)
    user_intent: Optional[str] = None


class RecommendationResponse(BaseModel):
    poses: list[Pose]


class EmbeddingPoseRecommendation(BaseModel):
    pose_id: str
    display_name: str
    score: float
    instructions: list[str] = Field(default_factory=list)
    explanation: str


class EmbeddingRecommendationResponse(BaseModel):
    poses: list[EmbeddingPoseRecommendation]
