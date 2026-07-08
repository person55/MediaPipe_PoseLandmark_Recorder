"""MediaPipe Tasks Pose Landmarker wrapper."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "dance_pose_recorder_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dance_pose_recorder.landmark_schema import landmark_name


class PoseExtractor:
    def __init__(self, model_path: str | Path, num_poses: int = 1, delegate: str = "cpu") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {self.model_path}")

        self.delegate = delegate
        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path),
            delegate=_delegate(delegate),
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_bgr: object, timestamp_ms: int) -> dict:
        if self.delegate == "gpu":
            image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=image_rgba)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_landmarks = []
        pose_world_landmarks = []
        if result.pose_landmarks:
            pose_landmarks = _landmarks_to_dicts(result.pose_landmarks[0])
        if result.pose_world_landmarks:
            pose_world_landmarks = _landmarks_to_dicts(result.pose_world_landmarks[0])

        return {
            "pose_landmarks": pose_landmarks,
            "pose_world_landmarks": pose_world_landmarks,
        }

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _landmarks_to_dicts(landmarks: object) -> list[dict]:
    rows = []
    for landmark_id, landmark in enumerate(landmarks):
        rows.append(
            {
                "id": landmark_id,
                "name": landmark_name(landmark_id),
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibility": _optional_float(getattr(landmark, "visibility", None)),
                "presence": _optional_float(getattr(landmark, "presence", None)),
            }
        )
    return rows


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _delegate(delegate: str) -> python.BaseOptions.Delegate:
    if delegate == "cpu":
        return python.BaseOptions.Delegate.CPU
    if delegate == "gpu":
        return python.BaseOptions.Delegate.GPU
    raise ValueError(f"Unsupported delegate: {delegate}")
