from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from huggingface_hub import scan_cache_dir

from app.schemas import SceneCandidate, SceneClassificationResult

# CLIP 使用描述性 prompt 比单词 label 更稳定，也更适合开放场景的背景识别。
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


class ClipSceneClassifier:
    model_name = "clip:openai/clip-vit-base-patch32"
    model_id = "openai/clip-vit-base-patch32"

    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        local_only = self._model_is_cached()
        self._processor = CLIPProcessor.from_pretrained(
            self.model_id,
            local_files_only=local_only,
        )
        self._model = CLIPModel.from_pretrained(
            self.model_id,
            local_files_only=local_only,
        )
        self._model.eval()

    def close(self) -> None:
        self._model = None
        self._processor = None

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
        return ensure_rgb_uint8(image)

    @classmethod
    def _model_is_cached(cls) -> bool:
        try:
            cache_info = scan_cache_dir()
        except Exception:
            return False

        return any(
            repo.repo_id == cls.model_id and Path(repo.repo_path).exists()
            for repo in cache_info.repos
        )


class SceneClassifier(ClipSceneClassifier):
    """Primary background classifier. Kept as a stable app-level name for callers."""


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("Expected an RGB or RGBA image array.")

    rgb_image = image[:, :, :3]
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb_image)
