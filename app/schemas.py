from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]
ComfortLevel = Literal["low", "medium", "high"]


class Pose(BaseModel):
    id: str
    name: str
    description: str
    scene_tags: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    popularity: int = Field(ge=0, le=100)
    suitable_body_types: list[str] = Field(default_factory=list)
    suitable_age_groups: list[str] = Field(default_factory=list)
    mobility_requirements: list[str] = Field(default_factory=list)
    score: Optional[float] = None


class SceneContext(BaseModel):
    scene_tags: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)


class PersonContext(BaseModel):
    age_group: Optional[str] = None
    body_type: Optional[str] = None
    comfort_level: ComfortLevel = "medium"
    mobility_notes: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    scene: SceneContext
    person: PersonContext = Field(default_factory=PersonContext)


class RecommendationResponse(BaseModel):
    poses: list[Pose]
