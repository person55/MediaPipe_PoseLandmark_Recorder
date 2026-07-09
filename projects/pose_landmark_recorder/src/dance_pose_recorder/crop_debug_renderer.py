"""Debug image rendering for crop refinement."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from dance_pose_recorder.crop_bbox import CropBBox
from dance_pose_recorder.landmark_schema import POSE_CONNECTIONS
from dance_pose_recorder.video_input import VideoFileReader


COLOR_BBOX = (0, 180, 255)
COLOR_CLEANED = (255, 255, 0)
COLOR_CANDIDATE = (0, 255, 0)
COLOR_ACCEPTED = (80, 255, 80)
COLOR_REJECTED = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)


def render_crop_debug_images(
    input_video: str | Path,
    output_dir: str | Path,
    segments: list,
    bboxes: dict[int, CropBBox],
    cleaned: pd.DataFrame,
    candidates: pd.DataFrame,
    score_rows: pd.DataFrame,
    max_images: int = 160,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_frames = _selected_frames(segments, score_rows, max_images)
    if not selected_frames:
        return []

    cleaned_pose = cleaned[cleaned["source"] == "pose"]
    candidate_pose = candidates[candidates["source"] == "pose"] if not candidates.empty else pd.DataFrame()
    cleaned_by_frame = {int(frame): group.copy() for frame, group in cleaned_pose.groupby("frame", sort=False)}
    candidate_by_frame = {int(frame): group.copy() for frame, group in candidate_pose.groupby("frame", sort=False)}
    score_by_frame = {int(frame): group.copy() for frame, group in score_rows.groupby("frame", sort=False)} if not score_rows.empty else {}

    written: list[Path] = []
    with VideoFileReader(input_video) as reader:
        max_frame = max(selected_frames)
        for frame in reader.frames(max_frames=min(reader.info.frame_count, max_frame + 1)):
            if frame.frame_index not in selected_frames:
                continue
            image = frame.image_bgr.copy()
            bbox = bboxes.get(frame.frame_index)
            if bbox is not None:
                x0, y0, x1, y1 = bbox.to_int_tuple()
                cv2.rectangle(image, (x0, y0), (x1, y1), COLOR_BBOX, 2)
            _draw_pose(image, cleaned_by_frame.get(frame.frame_index), COLOR_CLEANED)
            _draw_pose(image, candidate_by_frame.get(frame.frame_index), COLOR_CANDIDATE)
            _draw_text(image, frame.frame_index, score_by_frame.get(frame.frame_index))
            output_path = output / f"crop_debug_frame_{frame.frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), image)
            written.append(output_path)
            if len(written) >= max_images:
                break
    return written


def _selected_frames(segments: list, score_rows: pd.DataFrame, max_images: int) -> set[int]:
    frames: set[int] = set()
    for segment in segments:
        if not getattr(segment, "crop_attempted", True):
            continue
        start = int(segment.start_frame)
        end = int(segment.end_frame)
        frames.update({start, (start + end) // 2, end})
    if not score_rows.empty:
        accepted = score_rows[score_rows["crop_refine_status"] == "crop_accepted"]
        frames.update(int(frame) for frame in accepted["frame"].head(max_images).tolist())
        rejected = score_rows[score_rows["crop_refine_status"] == "crop_rejected"].copy()
        if "score_delta" in rejected:
            rejected = rejected.sort_values("score_delta", ascending=False)
        frames.update(int(frame) for frame in rejected["frame"].head(max(0, max_images - len(frames))).tolist())
    return set(sorted(frames)[:max_images])


def _draw_pose(image, frame_df: pd.DataFrame | None, color: tuple[int, int, int]) -> None:
    if frame_df is None or frame_df.empty:
        return
    height, width = image.shape[:2]
    points: dict[int, tuple[int, int]] = {}
    for row in frame_df.itertuples(index=False):
        if pd.isna(row.x) or pd.isna(row.y):
            continue
        points[int(row.landmark_id)] = (int(round(float(row.x) * width)), int(round(float(row.y) * height)))
    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], color, 1)
    for point in points.values():
        cv2.circle(image, point, 2, color, -1)


def _draw_text(image, frame_index: int, score_rows: pd.DataFrame | None) -> None:
    accepted = 0
    rejected = 0
    unavailable = 0
    max_delta = None
    if score_rows is not None and not score_rows.empty:
        counts = score_rows["crop_refine_status"].value_counts()
        accepted = int(counts.get("crop_accepted", 0))
        rejected = int(counts.get("crop_rejected", 0))
        unavailable = int(counts.get("crop_unavailable", 0))
        if "score_delta" in score_rows:
            max_delta = float(score_rows["score_delta"].max())
    text = f"frame {frame_index}  accepted={accepted} rejected={rejected} unavailable={unavailable}"
    cv2.putText(image, text, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    if max_delta is not None:
        cv2.putText(image, f"max score delta={max_delta:.3f}", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
