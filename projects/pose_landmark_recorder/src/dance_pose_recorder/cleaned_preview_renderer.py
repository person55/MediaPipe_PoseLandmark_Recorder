"""Corrected preview rendering for cleaned pose data."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from dance_pose_recorder.landmark_schema import POSE_CONNECTIONS
from dance_pose_recorder.video_input import VideoFileReader

COLOR_MEASURED = (0, 255, 0)
COLOR_REFINED = (80, 255, 80)
COLOR_INTERPOLATED = (0, 255, 255)
COLOR_OUTLIER_INTERPOLATED = (255, 180, 0)
COLOR_OCCLUDED_ARM = (255, 0, 255)
COLOR_TORSO_CORRECTED = (0, 180, 255)
COLOR_INVALID = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
TORSO_SIDE_CONNECTIONS = {(11, 23), (12, 24)}


def render_corrected_preview(
    input_video: str | Path,
    output_path: str | Path,
    cleaned: pd.DataFrame,
    frame_status: pd.DataFrame,
    metadata: dict,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    status_by_frame = frame_status.set_index("frame").to_dict(orient="index")
    pose_rows = cleaned[cleaned["source"] == "pose"].copy()
    pose_by_frame = {int(frame): group for frame, group in pose_rows.groupby("frame")}

    with VideoFileReader(input_video) as reader:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, reader.info.fps, (reader.info.width, reader.info.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create corrected preview video: {output}")
        try:
            total = min(reader.info.frame_count, int(metadata.get("frame_count_written") or reader.info.frame_count))
            for frame in tqdm(reader.frames(max_frames=total), total=total, unit="frame"):
                image = frame.image_bgr.copy()
                _draw_frame(image, pose_by_frame.get(frame.frame_index), status_by_frame.get(frame.frame_index, {}))
                writer.write(image)
        finally:
            writer.release()
    return output


def _draw_frame(image: object, frame_df: pd.DataFrame | None, status: dict) -> None:
    height, width = image.shape[:2]
    if status.get("is_inside_long_missing_range"):
        cv2.putText(image, "LONG MISSING GAP", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)
        return
    if frame_df is None or frame_df.empty:
        cv2.putText(image, "NO POSE", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_INVALID, 2)
        return

    points = {}
    qualities = {}
    for row in frame_df.itertuples(index=False):
        if not bool(row.is_valid) and not bool(row.is_interpolated):
            continue
        if pd.isna(row.x) or pd.isna(row.y):
            continue
        x = int(round(float(row.x) * width))
        y = int(round(float(row.y) * height))
        points[int(row.landmark_id)] = (x, y)
        qualities[int(row.landmark_id)] = row.quality_flag

    hide_torso_side = _should_hide_torso_side(points)
    for start, end in POSE_CONNECTIONS:
        if hide_torso_side and (start, end) in TORSO_SIDE_CONNECTIONS:
            continue
        if start in points and end in points:
            cv2.line(image, points[start], points[end], _connection_color(qualities.get(start), qualities.get(end)), 2)
    for landmark_id, point in points.items():
        cv2.circle(image, point, 3, _point_color(qualities.get(landmark_id)), -1)
    if "refined_measured" in set(qualities.values()):
        cv2.putText(image, "REFINED", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_REFINED, 2)
    if not points:
        cv2.putText(image, "NO VALID POSE", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_INVALID, 2)


def _should_hide_torso_side(points: dict[int, tuple[int, int]]) -> bool:
    required = [11, 12, 23, 24]
    if not all(landmark_id in points for landmark_id in required):
        return False
    side_cost = _point_distance(points[11], points[23]) + _point_distance(points[12], points[24])
    cross_cost = _point_distance(points[11], points[24]) + _point_distance(points[12], points[23])
    return side_cost > cross_cost


def _point_distance(start: tuple[int, int], end: tuple[int, int]) -> float:
    return ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5


def _point_color(quality_flag: str | None) -> tuple[int, int, int]:
    if quality_flag == "refined_measured":
        return COLOR_REFINED
    if quality_flag == "interpolated_short_gap":
        return COLOR_INTERPOLATED
    if quality_flag == "interpolated_outlier_removed":
        return COLOR_OUTLIER_INTERPOLATED
    if quality_flag == "estimated_occluded_arm":
        return COLOR_OCCLUDED_ARM
    if quality_flag in {"shoulder_swap_corrected", "pelvis_swap_corrected", "torso_swap_corrected"}:
        return COLOR_TORSO_CORRECTED
    if quality_flag == "unreliable":
        return COLOR_INVALID
    return COLOR_MEASURED


def _connection_color(start_quality: str | None, end_quality: str | None) -> tuple[int, int, int]:
    if "refined_measured" in {start_quality, end_quality}:
        return COLOR_REFINED
    if "estimated_occluded_arm" in {start_quality, end_quality}:
        return COLOR_OCCLUDED_ARM
    if "interpolated_outlier_removed" in {start_quality, end_quality}:
        return COLOR_OUTLIER_INTERPOLATED
    if {"shoulder_swap_corrected", "pelvis_swap_corrected", "torso_swap_corrected"} & {start_quality, end_quality}:
        return COLOR_TORSO_CORRECTED
    if "interpolated_short_gap" in {start_quality, end_quality}:
        return COLOR_INTERPOLATED
    return COLOR_MEASURED
