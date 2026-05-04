from __future__ import annotations

import numpy as np

from app.schemas import Landmark, PoseEstimationResult


class PoseEstimator:
    def __init__(self) -> None:
        self._mp_pose = None
        try:
            from mediapipe.python.solutions import pose as mp_pose
        except ImportError:
            self._mediapipe_available = False
            return

        self._mediapipe_available = True
        self._mp_pose = mp_pose

    def estimate(self, image: np.ndarray) -> PoseEstimationResult:
        if not self._mediapipe_available or self._mp_pose is None:
            return PoseEstimationResult(
                has_person=False,
                landmark_count=0,
                landmarks=[],
            )

        image = np.ascontiguousarray(image)
        with self._mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(image)

        pose_landmarks = results.pose_landmarks
        if pose_landmarks is None:
            return PoseEstimationResult(
                has_person=False,
                landmark_count=0,
                landmarks=[],
            )

        landmarks = [
            Landmark(
                name=self._mp_pose.PoseLandmark(index).name.lower(),
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(landmark.visibility),
            )
            for index, landmark in enumerate(pose_landmarks.landmark)
        ]

        return PoseEstimationResult(
            has_person=True,
            landmark_count=len(landmarks),
            landmarks=landmarks,
        )
