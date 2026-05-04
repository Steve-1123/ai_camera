from __future__ import annotations

import numpy as np

from app.schemas import SceneCandidate, SceneClassificationResult

SCENE_LABELS = [
    "cafe",
    "street",
    "beach",
    "nature",
    "stairs",
    "wall",
    "indoor",
    "landmark",
    "park",
    "restaurant",
    "hotel",
    "museum",
    "unknown",
]


class SceneClassifier:
    model_name = "heuristic-color-v1"

    def classify(self, image: np.ndarray) -> SceneClassificationResult:
        normalized = image.astype(np.float32) / 255.0
        red = float(normalized[:, :, 0].mean())
        green = float(normalized[:, :, 1].mean())
        blue = float(normalized[:, :, 2].mean())
        brightness = float(normalized.mean())
        contrast = float(normalized.std())

        scores = dict.fromkeys(SCENE_LABELS, 0.03)
        scores["unknown"] = 0.2

        if green > red + 0.08 and green > blue + 0.05:
            scores["nature"] = 0.72
            scores["park"] = 0.62
        elif blue > red + 0.08 and brightness > 0.45:
            scores["beach"] = 0.68
            scores["nature"] = 0.35
        elif contrast < 0.08 and brightness > 0.45:
            scores["wall"] = 0.7
            scores["indoor"] = 0.42
        elif brightness < 0.35:
            scores["indoor"] = 0.58
            scores["restaurant"] = 0.36
            scores["hotel"] = 0.32
        elif contrast > 0.22:
            scores["street"] = 0.54
            scores["landmark"] = 0.34
        else:
            scores["unknown"] = 0.55
            scores["indoor"] = 0.3

        candidates = [
            SceneCandidate(label=label, score=round(score, 3))
            for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ][:5]

        return SceneClassificationResult(
            primary_category=candidates[0].label if candidates else "unknown",
            candidates=candidates,
            model_name=self.model_name,
        )

