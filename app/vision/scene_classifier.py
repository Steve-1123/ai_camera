from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.schemas import SceneCandidate, SceneClassificationResult

logger = logging.getLogger(__name__)

# CLIP 使用描述性 prompt 比单词 label 更稳定。
SCENE_PROMPTS = {
    "cafe": "a portrait photo taken in a cafe",
    "street": "a portrait photo taken on a city street",
    "beach": "a portrait photo taken at the beach",
    "park": "a portrait photo taken in a park with trees and grass",
    "stairs": "a portrait photo taken on stairs",
    "wall": "a portrait photo taken near a plain wall",
    "indoor": "a portrait photo taken indoors",
    "landmark": "a portrait photo taken near a famous landmark",
    "restaurant": "a portrait photo taken in a restaurant",
    "hotel": "a portrait photo taken in a hotel",
    "museum": "a portrait photo taken in a museum or gallery",
}

SCENE_LABELS = [*SCENE_PROMPTS, "unknown"]


class HeuristicSceneClassifier:
    model_name = "heuristic_rgb_fallback"

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
            scores["park"] = 0.72
        elif blue > red + 0.08 and brightness > 0.45:
            scores["beach"] = 0.68
            scores["park"] = 0.35
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


class ClipSceneClassifier:
    model_name = "clip:openai/clip-vit-base-patch32"
    model_id = "openai/clip-vit-base-patch32"

    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._processor = self._from_pretrained_with_cache_first(CLIPProcessor)
        self._model = self._from_pretrained_with_cache_first(CLIPModel)
        self._model.eval()

    def classify(self, image: np.ndarray) -> SceneClassificationResult:
        pil_image = Image.fromarray(self._prepare_rgb_image(image))
        labels = list(SCENE_PROMPTS)
        prompts = [SCENE_PROMPTS[label] for label in labels]

        inputs = self._processor(
            text=prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits_per_image.softmax(dim=1)[0]

        ranked = sorted(
            zip(labels, scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        candidates = [
            SceneCandidate(label=label, score=round(float(score), 3))
            for label, score in ranked[:5]
        ]

        return SceneClassificationResult(
            primary_category=candidates[0].label if candidates else "unknown",
            candidates=candidates,
            model_name=self.model_name,
        )

    @staticmethod
    def _prepare_rgb_image(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise ValueError("ClipSceneClassifier expects an RGB or RGBA image array.")

        rgb_image = image[:, :, :3]
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(rgb_image)

    @classmethod
    def _from_pretrained_with_cache_first(cls, loader: object) -> object:
        try:
            return loader.from_pretrained(cls.model_id, local_files_only=True)
        except Exception:
            return loader.from_pretrained(cls.model_id)


class SceneClassifier:
    def __init__(self) -> None:
        try:
            self.impl = ClipSceneClassifier()
        except Exception as exc:
            logger.warning("clip_scene_classifier_unavailable fallback=heuristic error=%s", exc)
            self.impl = HeuristicSceneClassifier()

    def classify(self, image: np.ndarray) -> SceneClassificationResult:
        return self.impl.classify(image)
