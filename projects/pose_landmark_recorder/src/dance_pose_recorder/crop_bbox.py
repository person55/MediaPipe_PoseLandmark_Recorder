"""Torso-centered crop box helpers for crop-based pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
HAND_PROXY_IDS = {17, 18, 19, 20, 21, 22}
UNSTABLE_CENTER_IDS = HAND_PROXY_IDS | {15, 16, 27, 28, 29, 30, 31, 32}
FALLBACK_CENTER_IDS = {11, 12, 13, 14, 23, 24, 25, 26}
SIZE_EXPANSION_IDS = set(range(11, 33)) - HAND_PROXY_IDS


@dataclass(frozen=True)
class CropBBox:
    """Pixel-space crop box in the original frame."""

    frame: int
    x0: float
    y0: float
    w: float
    h: float
    frame_width: int
    frame_height: int
    margin_ratio: float
    source: str = "torso"

    @property
    def x1(self) -> float:
        return self.x0 + self.w

    @property
    def y1(self) -> float:
        return self.y0 + self.h

    @property
    def center_x(self) -> float:
        return self.x0 + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y0 + self.h / 2.0

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        x0 = int(round(self.x0))
        y0 = int(round(self.y0))
        x1 = int(round(self.x1))
        y1 = int(round(self.y1))
        return x0, y0, x1, y1

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "crop_x0": self.x0,
            "crop_y0": self.y0,
            "crop_w": self.w,
            "crop_h": self.h,
            "crop_margin_ratio": self.margin_ratio,
            "crop_source": self.source,
        }


def compute_torso_center(frame_rows: pd.DataFrame, frame_width: int, frame_height: int) -> tuple[float, float] | None:
    """Return a stable torso-based center in pixel coordinates."""

    points = _points_by_id(frame_rows, frame_width, frame_height)
    left_shoulder = points.get(LEFT_SHOULDER)
    right_shoulder = points.get(RIGHT_SHOULDER)
    left_hip = points.get(LEFT_HIP)
    right_hip = points.get(RIGHT_HIP)

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    pelvis = _midpoint(left_hip, right_hip)
    torso_center = _midpoint(shoulder_mid, pelvis)
    if torso_center is not None:
        return torso_center
    if pelvis is not None:
        return pelvis
    if shoulder_mid is not None:
        return shoulder_mid

    fallback = [point for landmark_id, point in points.items() if landmark_id in FALLBACK_CENTER_IDS]
    if fallback:
        return _mean_point(fallback)
    return None


def compute_crop_bbox(
    frame_rows: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    frame: int | None = None,
    previous_bbox: CropBBox | None = None,
    crop_margin_ratio: float = 1.8,
    full_body_margin_ratio: float | None = None,
    crop_square: bool = True,
    crop_min_size: int = 512,
    crop_source: str = "torso",
) -> CropBBox | None:
    """Compute a torso-centered person crop from pose rows in original-frame coordinates."""

    if frame_rows.empty:
        return previous_bbox

    frame_index = int(frame if frame is not None else frame_rows["frame"].iloc[0])
    points = _points_by_id(frame_rows, frame_width, frame_height)
    center = compute_torso_center(frame_rows, frame_width, frame_height)
    if center is None and previous_bbox is not None:
        center = (previous_bbox.center_x, previous_bbox.center_y)
    if center is None:
        return None

    torso_width = _distance(points.get(LEFT_SHOULDER), points.get(RIGHT_SHOULDER))
    shoulder_mid = _midpoint(points.get(LEFT_SHOULDER), points.get(RIGHT_SHOULDER))
    pelvis = _midpoint(points.get(LEFT_HIP), points.get(RIGHT_HIP))
    torso_height = _distance(shoulder_mid, pelvis)
    body_bbox_size = _full_body_bbox_size(points)
    full_body_margin = float(crop_margin_ratio if full_body_margin_ratio is None else full_body_margin_ratio)

    crop_size = max(
        float(crop_min_size),
        torso_width * 4.0 * float(crop_margin_ratio) if torso_width else 0.0,
        torso_height * 4.5 * float(crop_margin_ratio) if torso_height else 0.0,
        body_bbox_size * full_body_margin if body_bbox_size else 0.0,
    )
    if crop_square:
        crop_w = crop_h = crop_size
    else:
        crop_w = crop_h = crop_size

    return _clamped_bbox(
        frame=frame_index,
        center_x=center[0],
        center_y=center[1],
        width=crop_w,
        height=crop_h,
        frame_width=frame_width,
        frame_height=frame_height,
        margin_ratio=crop_margin_ratio,
        source=crop_source,
    )


def smooth_bboxes(
    bboxes: dict[int, CropBBox],
    window: int = 5,
    shrink_limit: float = 0.85,
) -> dict[int, CropBBox]:
    """Smooth crop centers with a rolling median and sizes with a rolling max."""

    if not bboxes:
        return {}
    frames = sorted(bboxes)
    smoothed: dict[int, CropBBox] = {}
    previous_size: float | None = None
    for frame in frames:
        recent_frames = [item for item in frames if item <= frame][-max(1, int(window)) :]
        recent = [bboxes[item] for item in recent_frames]
        current = bboxes[frame]
        center_x = float(np.median([bbox.center_x for bbox in recent]))
        center_y = float(np.median([bbox.center_y for bbox in recent]))
        size = float(max(max(bbox.w, bbox.h) for bbox in recent))
        if previous_size is not None and size < previous_size * shrink_limit:
            size = previous_size * shrink_limit
        previous_size = size
        smoothed[frame] = _clamped_bbox(
            frame=frame,
            center_x=center_x,
            center_y=center_y,
            width=size,
            height=size,
            frame_width=current.frame_width,
            frame_height=current.frame_height,
            margin_ratio=current.margin_ratio,
            source=current.source,
        )
    return smoothed


def crop_to_original_norm(x_crop_norm: float, y_crop_norm: float, bbox: CropBBox) -> tuple[float, float]:
    """Convert crop-normalized coordinates back to original-frame normalized coordinates."""

    x = (bbox.x0 + float(x_crop_norm) * bbox.w) / float(bbox.frame_width)
    y = (bbox.y0 + float(y_crop_norm) * bbox.h) / float(bbox.frame_height)
    return x, y


def is_near_crop_edge(x_crop_norm: float, y_crop_norm: float, edge_margin_norm: float = 0.03) -> bool:
    if not (_finite(x_crop_norm) and _finite(y_crop_norm)):
        return True
    return (
        x_crop_norm < edge_margin_norm
        or y_crop_norm < edge_margin_norm
        or x_crop_norm > 1.0 - edge_margin_norm
        or y_crop_norm > 1.0 - edge_margin_norm
    )


def _clamped_bbox(
    frame: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
    source: str,
) -> CropBBox:
    width = min(max(1.0, float(width)), float(frame_width))
    height = min(max(1.0, float(height)), float(frame_height))
    x0 = min(max(0.0, float(center_x) - width / 2.0), max(0.0, float(frame_width) - width))
    y0 = min(max(0.0, float(center_y) - height / 2.0), max(0.0, float(frame_height) - height))
    return CropBBox(
        frame=int(frame),
        x0=x0,
        y0=y0,
        w=width,
        h=height,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        margin_ratio=float(margin_ratio),
        source=source,
    )


def _points_by_id(frame_rows: pd.DataFrame, frame_width: int, frame_height: int) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    for row in frame_rows.itertuples(index=False):
        landmark_id = int(row.landmark_id)
        x = getattr(row, "x", None)
        y = getattr(row, "y", None)
        if not (_finite(x) and _finite(y)):
            continue
        points[landmark_id] = (float(x) * frame_width, float(y) * frame_height)
    return points


def _full_body_bbox_size(points: dict[int, tuple[float, float]]) -> float:
    usable = [point for landmark_id, point in points.items() if landmark_id in SIZE_EXPANSION_IDS]
    if not usable:
        return 0.0
    xs = [point[0] for point in usable]
    ys = [point[1] for point in usable]
    return float(max(max(xs) - min(xs), max(ys) - min(ys)))


def _midpoint(
    start: tuple[float, float] | None,
    end: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if start is None or end is None:
        return None
    return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (float(np.mean([point[0] for point in points])), float(np.mean([point[1] for point in points])))


def _distance(start: tuple[float, float] | None, end: tuple[float, float] | None) -> float:
    if start is None or end is None:
        return 0.0
    return float(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5)


def _finite(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return isfinite(float(value))
