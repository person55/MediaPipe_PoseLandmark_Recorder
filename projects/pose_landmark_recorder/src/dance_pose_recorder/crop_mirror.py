"""Mirror/reverse detection passes and confusion diagnostics.

Mirroring the crop horizontally probes the detector's left-right asymmetry:
detecting on the mirrored pixels and un-mirroring (x -> 1-x plus swapping
left/right landmark identities) yields a real measurement of the same image.
Disagreement between the forward and mirror passes is recorded as a pure
diagnostic signal for possible left-right confusion / target switching; it
never fabricates coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

# MediaPipe pose left/right landmark id pairs (center landmarks map to themselves).
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

SWAP_INDEX = list(range(33))
for _left, _right in _LR_PAIRS:
    SWAP_INDEX[_left] = _right
    SWAP_INDEX[_right] = _left


def mirror_crop(crop_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(crop_bgr, 1)


def unmirror_pose_landmarks(landmarks: list[dict]) -> list[dict]:
    """Un-mirror x and restore left/right identities for a 33-landmark list."""

    if len(landmarks) != len(SWAP_INDEX):
        return []
    restored = []
    for landmark_id in range(len(SWAP_INDEX)):
        item = dict(landmarks[SWAP_INDEX[landmark_id]])
        x = item.get("x")
        if x is not None and pd.notna(x):
            item["x"] = 1.0 - float(x)
        restored.append(item)
    return restored


def unmirror_world_landmarks(landmarks: list[dict]) -> list[dict]:
    if len(landmarks) != len(SWAP_INDEX):
        return []
    restored = []
    for landmark_id in range(len(SWAP_INDEX)):
        item = dict(landmarks[SWAP_INDEX[landmark_id]])
        x = item.get("x")
        if x is not None and pd.notna(x):
            item["x"] = -float(x)
        restored.append(item)
    return restored


def pass_disagreement(reference: list[dict], other: list[dict]) -> float | None:
    """Mean 2D distance between two landmark sets in normalized crop units."""

    if not reference or not other or len(reference) != len(other):
        return None
    distances = []
    for ref, alt in zip(reference, other):
        rx, ry, ax, ay = ref.get("x"), ref.get("y"), alt.get("x"), alt.get("y")
        if any(value is None or pd.isna(value) for value in (rx, ry, ax, ay)):
            continue
        distances.append(float(np.hypot(float(rx) - float(ax), float(ry) - float(ay))))
    if not distances:
        return None
    return float(np.mean(distances))


CONFUSION_COLUMNS = [
    "frame",
    "crop_segment_id",
    "forward_mean_visibility",
    "mirror_disagreement",
    "rotated_disagreement",
    "possible_confusion",
]


def confusion_row(
    frame: int,
    segment_id: int,
    forward_visibility: float,
    mirror_dis: float | None,
    rotated_dis: float | None,
    threshold: float,
) -> dict:
    flagged = any(value is not None and value > threshold for value in (mirror_dis, rotated_dis))
    return {
        "frame": int(frame),
        "crop_segment_id": int(segment_id),
        "forward_mean_visibility": float(forward_visibility),
        "mirror_disagreement": mirror_dis,
        "rotated_disagreement": rotated_dis,
        "possible_confusion": bool(flagged),
    }
