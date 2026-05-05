from __future__ import annotations

import logging
import threading
from importlib import import_module

import numpy as np
from PIL import Image

from app.schemas import Landmark, PoseEstimationResult

logger = logging.getLogger(__name__)


class PoseEstimator:
    max_detection_side = 1280

    def __init__(self) -> None:
        self._mp_pose = None
        self._pose = None
        self._lock = threading.Lock()
        self._mediapipe_import_error = None
        try:
            self._mp_pose = import_module("mediapipe.solutions.pose")
        except ImportError as exc:
            self._mediapipe_import_error = exc
            try:
                self._mp_pose = import_module("mediapipe.python.solutions.pose")
            except ImportError as fallback_exc:
                self._mediapipe_import_error = fallback_exc
                self._mediapipe_available = False
                logger.warning(
                    "mediapipe_pose_unavailable import_error=%s. Install a MediaPipe "
                    "version that provides mediapipe.solutions.pose.",
                    fallback_exc,
                )
                return

        if self._mp_pose is None:
            self._mediapipe_available = False
            return

        self._mediapipe_available = True

    def close(self) -> None:
        with self._lock:
            if self._pose is not None:
                self._pose.close()
                self._pose = None

    def estimate(self, image: np.ndarray) -> PoseEstimationResult:
        if not self._mediapipe_available or self._mp_pose is None:
            self._log_debug(image=image, detected_landmarks=False)
            return PoseEstimationResult(
                has_person=False,
                landmark_count=0,
                landmarks=[],
            )

        rgb_image = self._prepare_rgb_image(image)
        try:
            with self._lock:
                pose = self._ensure_pose()
                results = pose.process(rgb_image)
        except RuntimeError as exc:
            logger.warning("mediapipe_pose_runtime_error error=%s", exc)
            self._log_debug(image=rgb_image, detected_landmarks=False)
            return PoseEstimationResult(
                has_person=False,
                landmark_count=0,
                landmarks=[],
            )

        pose_landmarks = results.pose_landmarks
        self._log_debug(
            image=rgb_image,
            detected_landmarks=pose_landmarks is not None,
        )
        if pose_landmarks is None:
            return PoseEstimationResult(
                has_person=False,
                landmark_count=0,
                landmarks=[],
            )

        landmarks = [
            Landmark(
                name=self._mp_pose.PoseLandmark(index).name.lower(),
                x=self._clamp_unit(landmark.x),
                y=self._clamp_unit(landmark.y),
                z=float(landmark.z),
                visibility=self._clamp_unit(landmark.visibility),
            )
            for index, landmark in enumerate(pose_landmarks.landmark)
        ]

        return PoseEstimationResult(
            has_person=True,
            landmark_count=len(landmarks),
            landmarks=landmarks,
        )

    def _prepare_rgb_image(self, image: np.ndarray) -> np.ndarray:
        return resize_longest_side(ensure_rgb_uint8(image), self.max_detection_side)

    def _ensure_pose(self) -> object:
        if self._pose is None:
            self._pose = self._mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.3,
            )
        return self._pose

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    @staticmethod
    def _log_debug(image: np.ndarray, detected_landmarks: bool) -> None:
        if image.size:
            min_pixel = int(image.min())
            max_pixel = int(image.max())
        else:
            min_pixel = None
            max_pixel = None

        logger.info(
            "mediapipe_pose_debug image_shape=%s image_dtype=%s min_pixel=%s "
            "max_pixel=%s mediapipe_detected_landmarks=%s",
            image.shape,
            image.dtype,
            min_pixel,
            max_pixel,
            detected_landmarks,
        )


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("Expected an RGB or RGBA image array.")

    rgb_image = image[:, :, :3]
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb_image)


def resize_longest_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return np.ascontiguousarray(image)

    scale = max_side / longest_side
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(resized_size, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)
