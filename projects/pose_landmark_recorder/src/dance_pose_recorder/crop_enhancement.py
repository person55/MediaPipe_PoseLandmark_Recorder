"""Low-light crop enhancement for re-detection rescue passes.

CLAHE on the LAB lightness channel raises local contrast in dark stage
footage without shifting colors. Enhancement only changes what the detector
SEES; original video frames and all stored coordinates are untouched, so
this stays a measurement-side improvement.
"""

from __future__ import annotations

import cv2
import numpy as np


def enhance_crop(crop_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    merged = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def mean_visibility(landmarks: list[dict]) -> float:
    values = [
        float(landmark["visibility"])
        for landmark in landmarks
        if landmark.get("visibility") is not None and np.isfinite(float(landmark["visibility"]))
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def detection_is_weak(landmarks: list[dict], visibility_threshold: float) -> bool:
    if not landmarks:
        return True
    return mean_visibility(landmarks) < visibility_threshold
