"""Rotation-augmented crop re-detection for inverted poses.

MediaPipe pose is trained on upright bodies, so inverted poses (cartwheels,
floor rolls) fall outside its distribution and produce left-right confusion.
Rotating the crop so the body is upright puts the detector back inside its
training distribution; detected landmarks are rotated back afterwards, so the
candidate remains a real detection on the real pixels.

The body-axis snap is conservative: rotation only engages when the cleaned
hip-to-shoulder axis deviates far from upright, and only in 90-degree steps.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

from dance_pose_recorder.crop_refiner import CropSegment
from dance_pose_recorder.landmark_schema import POSE_LANDMARK_NAMES

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

# Broad problem flags: inverted-pose confusion usually carries confident
# "measured" flags, so acceptance eligibility must include them. The scorer
# and guards still decide acceptance row by row.
INVERTED_SEGMENT_FLAGS = {
    "measured",
    "crop_refined_measured",
    "refined_measured",
    "interpolated_short_gap",
    "interpolated_outlier_removed",
    "estimated_occluded_arm",
    "low_visibility_leg_kept",
    "unreliable",
}

_CV2_ROTATIONS = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,  # head pointing right -> rotate CCW to upright
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,  # head pointing left -> rotate CW to upright
}


def body_axis_angle_deg(frame_rows: pd.DataFrame, frame_width: int, frame_height: int) -> float | None:
    """Angle of the hip->shoulder axis from upright, in degrees (-180, 180]."""

    points: dict[int, tuple[float, float]] = {}
    for row in frame_rows.itertuples(index=False):
        landmark_id = int(row.landmark_id)
        if landmark_id in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
            x, y = getattr(row, "x", None), getattr(row, "y", None)
            if x is None or y is None or pd.isna(x) or pd.isna(y):
                continue
            points[landmark_id] = (float(x) * frame_width, float(y) * frame_height)
    if not all(key in points for key in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)):
        return None
    shoulder = np.mean([points[LEFT_SHOULDER], points[RIGHT_SHOULDER]], axis=0)
    hip = np.mean([points[LEFT_HIP], points[RIGHT_HIP]], axis=0)
    dx, dy = float(shoulder[0] - hip[0]), float(shoulder[1] - hip[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    # Upright = shoulders above hips = dy < 0. Positive angle = head tilted
    # toward +x (screen right).
    return math.degrees(math.atan2(dx, -dy))


def snap_rotation(angle_deg: float | None, min_angle_deg: float) -> int:
    """Snap a body-axis angle to the 90-degree rotation bucket (0/90/180/270)."""

    if angle_deg is None or abs(angle_deg) < min_angle_deg:
        return 0
    snapped = int(round(angle_deg / 90.0)) % 4 * 90
    if snapped == 0:
        return 0
    return snapped


def rotate_crop(crop: np.ndarray, snap: int) -> np.ndarray:
    if snap == 0:
        return crop
    return cv2.rotate(crop, _CV2_ROTATIONS[snap])


def unrotate_norm(x_rotated: float, y_rotated: float, snap: int) -> tuple[float, float]:
    """Map normalized coords detected in the rotated crop back to crop coords."""

    if snap == 180:
        return 1.0 - x_rotated, 1.0 - y_rotated
    if snap == 90:  # applied CCW: (x, y) -> (y, 1 - x)
        return 1.0 - y_rotated, x_rotated
    if snap == 270:  # applied CW: (x, y) -> (1 - y, x)
        return y_rotated, 1.0 - x_rotated
    return x_rotated, y_rotated


def unrotate_direction(dx_rotated: float, dy_rotated: float, snap: int) -> tuple[float, float]:
    """Map an image-plane vector (e.g. world x/y) back to original orientation."""

    if snap == 180:
        return -dx_rotated, -dy_rotated
    if snap == 90:  # inverse of CCW direction map (dx, dy) -> (dy, -dx)
        return -dy_rotated, dx_rotated
    if snap == 270:  # inverse of CW direction map (dx, dy) -> (-dy, dx)
        return dy_rotated, -dx_rotated
    return dx_rotated, dy_rotated


def unrotate_pose_landmarks(landmarks: list[dict], snap: int) -> list[dict]:
    if snap == 0:
        return landmarks
    restored = []
    for landmark in landmarks:
        item = dict(landmark)
        item["x"], item["y"] = unrotate_norm(float(landmark["x"]), float(landmark["y"]), snap)
        restored.append(item)
    return restored


def unrotate_world_landmarks(landmarks: list[dict], snap: int) -> list[dict]:
    if snap == 0:
        return landmarks
    restored = []
    for landmark in landmarks:
        item = dict(landmark)
        x, y = landmark.get("x"), landmark.get("y")
        if x is not None and y is not None and pd.notna(x) and pd.notna(y):
            item["x"], item["y"] = unrotate_direction(float(x), float(y), snap)
        restored.append(item)
    return restored


def detect_width_for_z(bbox_w: float, bbox_h: float, snap: int) -> float:
    """Width of the image the detector actually saw (z is normalized by it)."""

    return float(bbox_h) if snap in (90, 270) else float(bbox_w)


def frame_rotations(
    pose_by_frame: dict[int, pd.DataFrame],
    frames: list[int],
    frame_width: int,
    frame_height: int,
    min_angle_deg: float,
) -> dict[int, int]:
    rotations: dict[int, int] = {}
    for frame in frames:
        rows = pose_by_frame.get(frame)
        if rows is None or rows.empty:
            continue
        snap = snap_rotation(body_axis_angle_deg(rows, frame_width, frame_height), min_angle_deg)
        if snap:
            rotations[frame] = snap
    return rotations


def build_inverted_segments(
    pose_by_frame: dict[int, pd.DataFrame],
    total_frames: int,
    frame_width: int,
    frame_height: int,
    min_angle_deg: float,
    existing_segments: list[CropSegment],
    max_segment_length: int,
    gap_tolerance: int = 2,
    min_length: int = 2,
) -> list[CropSegment]:
    """Add crop segments for inverted-pose runs not covered by existing targets."""

    covered = set()
    for segment in existing_segments:
        if segment.crop_attempted:
            covered.update(range(segment.start_frame, segment.end_frame + 1))

    rotations = frame_rotations(
        pose_by_frame, sorted(pose_by_frame), frame_width, frame_height, min_angle_deg
    )
    frames = sorted(frame for frame in rotations if frame not in covered and frame < total_frames)
    if not frames:
        return []

    runs: list[tuple[int, int]] = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame - prev > gap_tolerance:
            runs.append((start, prev))
            start = frame
        prev = frame
    runs.append((start, prev))

    next_id = max((segment.crop_segment_id for segment in existing_segments), default=0) + 1
    segments: list[CropSegment] = []
    for start_frame, end_frame in runs:
        length = end_frame - start_frame + 1
        if length < min_length:
            continue
        review_only = length > max_segment_length
        segments.append(
            CropSegment(
                crop_segment_id=next_id,
                start_frame=start_frame,
                end_frame=end_frame,
                target_landmarks=set(POSE_LANDMARK_NAMES),
                problem_flags=set(INVERTED_SEGMENT_FLAGS),
                segment_type="inverted_pose_segment",
                review_only=review_only,
                crop_attempted=not review_only,
                selected_for_crop=not review_only,
                selection_reason="inverted_body_axis",
            )
        )
        next_id += 1
    return segments
