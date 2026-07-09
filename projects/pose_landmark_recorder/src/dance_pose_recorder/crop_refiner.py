"""Segment detection helpers for torso-centered crop refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from dance_pose_recorder.interpolation import contiguous_ranges


CROP_TARGET_GROUPS = {
    "arms": {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    },
    "feet": {
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    },
    "hands_proxy": {
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
    },
    "legs": {
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    },
}

DEFAULT_CROP_FLAGS = {
    "unreliable",
    "interpolated_outlier_removed",
    "estimated_occluded_arm",
}

DEFAULT_TARGET_SEGMENT_TYPES = {"mixed_problem_segment"}


@dataclass
class CropSegment:
    crop_segment_id: int
    start_frame: int
    end_frame: int
    source: str = "pose"
    target_landmarks: set[str] = field(default_factory=set)
    problem_flags: set[str] = field(default_factory=set)
    segment_type: str = "short_crop_candidate"
    review_only: bool = False
    crop_attempted: bool = True
    selected_for_crop: bool = True
    selection_reason: str = "selected_mixed_problem_segment"

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self, fps: float | None = None) -> dict:
        data = {
            "crop_segment_id": self.crop_segment_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "length": self.length,
            "source": self.source,
            "target_landmarks": ",".join(sorted(self.target_landmarks)),
            "problem_flags": ",".join(sorted(self.problem_flags)),
            "segment_type": self.segment_type,
            "review_only": self.review_only,
            "crop_attempted": self.crop_attempted,
            "selected_for_crop": self.selected_for_crop,
            "selection_reason": self.selection_reason,
        }
        if fps:
            data["duration_sec"] = self.length / float(fps)
        return data


@dataclass(frozen=True)
class _RawRange:
    start_frame: int
    end_frame: int
    landmark_name: str
    quality_flag: str
    source: str = "pose"


def parse_crop_target_landmarks(spec: str) -> set[str]:
    if not spec or spec == "all":
        return set().union(*CROP_TARGET_GROUPS.values())
    names: set[str] = set()
    for item in (part.strip() for part in spec.split(",")):
        if not item:
            continue
        if item in CROP_TARGET_GROUPS:
            names.update(CROP_TARGET_GROUPS[item])
        else:
            names.add(item)
    return names


def parse_flags(spec: str | Iterable[str]) -> set[str]:
    if isinstance(spec, str):
        return {item.strip() for item in spec.split(",") if item.strip()}
    return {str(item) for item in spec}


def detect_crop_segments(
    cleaned: pd.DataFrame,
    total_frames: int | None = None,
    target_landmarks: str | Iterable[str] = "arms,feet,hands_proxy",
    target_flags: str | Iterable[str] = "unreliable,interpolated_outlier_removed,estimated_occluded_arm",
    min_cluster_length: int = 2,
    max_segment_length: int = 100,
    segment_margin: int = 12,
    source: str = "pose",
    allow_long_segments: bool = False,
    target_segment_types: str | Iterable[str] = "mixed_problem_segment",
    include_short_invalid_cluster: bool = False,
    exclude_missing_long_gap: bool = True,
    exclude_review_only: bool = True,
) -> list[CropSegment]:
    if isinstance(target_landmarks, str):
        target_names = parse_crop_target_landmarks(target_landmarks)
    else:
        target_names = set(target_landmarks)
    flags = parse_flags(target_flags) or DEFAULT_CROP_FLAGS
    allowed_segment_types = parse_flags(target_segment_types) or DEFAULT_TARGET_SEGMENT_TYPES
    if include_short_invalid_cluster:
        allowed_segment_types.add("short_invalid_cluster")
    if total_frames is None:
        total_frames = int(cleaned["frame"].max()) + 1

    raw_ranges = _problem_ranges(cleaned, target_names, flags, min_cluster_length, source)
    merged = _merge_ranges(raw_ranges, total_frames, segment_margin)
    segments: list[CropSegment] = []
    for index, item in enumerate(merged, start=1):
        length = item["end_frame"] - item["start_frame"] + 1
        review_only = length > max_segment_length
        segment_type = _segment_type(length, item["problem_flags"], max_segment_length)
        selected, reason = _selection_policy(
            segment_type=segment_type,
            review_only=review_only,
            length=length,
            max_segment_length=max_segment_length,
            problem_flags=item["problem_flags"],
            allowed_segment_types=allowed_segment_types,
            allow_long_segments=allow_long_segments,
            exclude_review_only=exclude_review_only,
            exclude_missing_long_gap=exclude_missing_long_gap,
        )
        segments.append(
            CropSegment(
                crop_segment_id=index,
                start_frame=item["start_frame"],
                end_frame=item["end_frame"],
                source=source,
                target_landmarks=item["target_landmarks"],
                problem_flags=item["problem_flags"],
                segment_type=segment_type,
                review_only=review_only,
                crop_attempted=selected,
                selected_for_crop=selected,
                selection_reason=reason,
            )
        )
    return segments


def crop_segments_to_dataframe(segments: list[CropSegment], fps: float | None = None) -> pd.DataFrame:
    columns = [
        "crop_segment_id",
        "start_frame",
        "end_frame",
        "length",
        "duration_sec",
        "source",
        "target_landmarks",
        "problem_flags",
        "segment_type",
        "review_only",
        "crop_attempted",
        "selected_for_crop",
        "selection_reason",
    ]
    return pd.DataFrame([segment.to_dict(fps) for segment in segments], columns=columns)


def _problem_ranges(
    cleaned: pd.DataFrame,
    target_names: set[str],
    problem_flags: set[str],
    min_cluster_length: int,
    source: str,
) -> list[_RawRange]:
    pose = cleaned[
        (cleaned["source"] == source)
        & (cleaned["landmark_name"].isin(target_names))
        & (cleaned["quality_flag"].isin(problem_flags))
    ]
    ranges: list[_RawRange] = []
    for (landmark_name, quality_flag), group in pose.groupby(["landmark_name", "quality_flag"], sort=True):
        for segment in contiguous_ranges(group["frame"].tolist()):
            if segment.length < min_cluster_length:
                continue
            ranges.append(_RawRange(segment.start_frame, segment.end_frame, str(landmark_name), str(quality_flag), source))
    return sorted(ranges, key=lambda item: (item.start_frame, item.end_frame, item.landmark_name))


def _merge_ranges(raw_ranges: list[_RawRange], total_frames: int, segment_margin: int) -> list[dict]:
    merged: list[dict] = []
    last_frame = max(0, int(total_frames) - 1)
    for item in raw_ranges:
        start = max(0, item.start_frame - segment_margin)
        end = min(last_frame, item.end_frame + segment_margin)
        if not merged or start > merged[-1]["end_frame"] + segment_margin:
            merged.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "target_landmarks": {item.landmark_name},
                    "problem_flags": {item.quality_flag},
                }
            )
            continue
        current = merged[-1]
        current["end_frame"] = max(current["end_frame"], end)
        current["target_landmarks"].add(item.landmark_name)
        current["problem_flags"].add(item.quality_flag)
    return merged


def _segment_type(length: int, flags: set[str], max_segment_length: int) -> str:
    if length > max_segment_length:
        return "long_unreliable_run"
    if len(flags) > 1:
        return "mixed_problem_segment"
    if length <= 30:
        return "short_invalid_cluster"
    return "medium_invalid_cluster"


def _selection_policy(
    segment_type: str,
    review_only: bool,
    length: int,
    max_segment_length: int,
    problem_flags: set[str],
    allowed_segment_types: set[str],
    allow_long_segments: bool,
    exclude_review_only: bool,
    exclude_missing_long_gap: bool,
) -> tuple[bool, str]:
    if exclude_missing_long_gap and "missing_long_gap" in problem_flags:
        return False, "excluded_missing_long_gap"
    if length > max_segment_length and not allow_long_segments:
        return False, "excluded_too_long"
    if review_only and exclude_review_only and not allow_long_segments:
        return False, "excluded_review_only"
    if segment_type == "long_unreliable_run" and not allow_long_segments:
        return False, "excluded_long_unreliable_run"
    if segment_type not in allowed_segment_types:
        return False, f"excluded_segment_type_{segment_type}"
    if segment_type == "mixed_problem_segment":
        return True, "selected_mixed_problem_segment"
    if segment_type == "short_invalid_cluster":
        return True, "selected_short_invalid_cluster"
    return True, f"selected_{segment_type}"
