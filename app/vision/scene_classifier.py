from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from huggingface_hub import scan_cache_dir

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

# These fallback thresholds are intentionally coarse. They are hand-tuned guardrails
# for when CLIP is unavailable, not a calibrated scene-recognition model.
GREEN_DOMINANCE_MARGIN = 0.08
GREEN_BLUE_MARGIN = 0.05
BLUE_DOMINANCE_MARGIN = 0.08
BRIGHT_SCENE_THRESHOLD = 0.45
LOW_CONTRAST_THRESHOLD = 0.08
DARK_SCENE_THRESHOLD = 0.35
HIGH_CONTRAST_THRESHOLD = 0.22
CLIP_RETRY_SECONDS = 300


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

        if green > red + GREEN_DOMINANCE_MARGIN and green > blue + GREEN_BLUE_MARGIN:
            scores["park"] = 0.72
        elif blue > red + BLUE_DOMINANCE_MARGIN and brightness > BRIGHT_SCENE_THRESHOLD:
            scores["beach"] = 0.68
            scores["park"] = 0.35
        elif contrast < LOW_CONTRAST_THRESHOLD and brightness > BRIGHT_SCENE_THRESHOLD:
            scores["wall"] = 0.7
            scores["indoor"] = 0.42
        elif brightness < DARK_SCENE_THRESHOLD:
            scores["indoor"] = 0.58
            scores["restaurant"] = 0.36
            scores["hotel"] = 0.32
        elif contrast > HIGH_CONTRAST_THRESHOLD:
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


class SceneClassifier:
    def __init__(self) -> None:
        self._fallback = HeuristicSceneClassifier()
        self.impl: ClipSceneClassifier | HeuristicSceneClassifier | None = None
        self._next_clip_retry_at = 0.0
        self._load_clip()

    def close(self) -> None:
        if hasattr(self.impl, "close"):
            self.impl.close()

    def classify(self, image: np.ndarray) -> SceneClassificationResult:
        if not isinstance(self.impl, ClipSceneClassifier) and time.monotonic() >= self._next_clip_retry_at:
            self._load_clip()

        impl = self.impl or self._fallback
        return impl.classify(image)

    def _load_clip(self) -> None:
        try:
            self.impl = ClipSceneClassifier()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "clip_scene_classifier_unavailable fallback=heuristic retry_seconds=%s error=%s",
                CLIP_RETRY_SECONDS,
                exc,
            )
            self.impl = self._fallback
            self._next_clip_retry_at = time.monotonic() + CLIP_RETRY_SECONDS


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("Expected an RGB or RGBA image array.")

    rgb_image = image[:, :, :3]
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb_image)
