"""Candidate segment detection for second-pass pose refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from dance_pose_recorder.interpolation import contiguous_ranges


TARGET_LANDMARK_GROUPS = {
    "arms": {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    },
    "hands": {
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
    "feet": {
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    },
    "torso": {
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    },
}

DEFAULT_PROBLEM_FLAGS = {
    "unreliable",
    "interpolated_outlier_removed",
    "estimated_occluded_arm",
    "low_visibility_leg_kept",
    "missing_long_gap",
}


@dataclass
class CandidateSegment:
    segment_id: int
    start_frame: int
    end_frame: int
    source: str = "pose"
    target_landmarks: set[str] = field(default_factory=set)
    problem_flags: set[str] = field(default_factory=set)
    segment_type: str = "short_invalid_cluster"
    review_only: bool = False

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "length": self.length,
            "source": self.source,
            "target_landmarks": ",".join(sorted(self.target_landmarks)),
            "problem_flags": ",".join(sorted(self.problem_flags)),
            "segment_type": self.segment_type,
            "review_only": self.review_only,
        }


@dataclass
class _RawRange:
    start_frame: int
    end_frame: int
    landmark_name: str
    quality_flag: str
    source: str = "pose"


def parse_target_landmarks(spec: str) -> set[str]:
    """Expand a comma-separated target group spec to landmark names."""

    if not spec or spec == "all":
        return set().union(*TARGET_LANDMARK_GROUPS.values())

    names: set[str] = set()
    for item in (part.strip() for part in spec.split(",")):
        if not item:
            continue
        if item in TARGET_LANDMARK_GROUPS:
            names.update(TARGET_LANDMARK_GROUPS[item])
        else:
            names.add(item)
    return names


def detect_candidate_segments(
    cleaned: pd.DataFrame,
    total_frames: int | None = None,
    target_landmarks: str | Iterable[str] = "arms,hands,feet",
    problem_flags: set[str] | None = None,
    min_cluster_length: int = 2,
    max_cluster_length: int = 90,
    segment_margin: int = 12,
    source: str = "pose",
) -> list[CandidateSegment]:
    """Find problematic frame ranges that are worth second-pass re-detection."""

    if isinstance(target_landmarks, str):
        target_names = parse_target_landmarks(target_landmarks)
    else:
        target_names = set(target_landmarks)
    flags = problem_flags or DEFAULT_PROBLEM_FLAGS
    if total_frames is None:
        total_frames = int(cleaned["frame"].max()) + 1

    raw_ranges = _problem_ranges(cleaned, target_names, flags, min_cluster_length, source)
    if not raw_ranges:
        return []

    merged = _merge_ranges(raw_ranges, total_frames, segment_margin)
    segments: list[CandidateSegment] = []
    for index, item in enumerate(merged, start=1):
        length = item["end_frame"] - item["start_frame"] + 1
        review_only = length > max_cluster_length or "missing_long_gap" in item["problem_flags"]
        segments.append(
            CandidateSegment(
                segment_id=index,
                start_frame=item["start_frame"],
                end_frame=item["end_frame"],
                source=source,
                target_landmarks=item["target_landmarks"],
                problem_flags=item["problem_flags"],
                segment_type=_segment_type(length, item["problem_flags"], max_cluster_length),
                review_only=review_only,
            )
        )
    return segments


def segments_to_dataframe(segments: list[CandidateSegment]) -> pd.DataFrame:
    columns = [
        "segment_id",
        "start_frame",
        "end_frame",
        "length",
        "source",
        "target_landmarks",
        "problem_flags",
        "segment_type",
        "review_only",
    ]
    return pd.DataFrame([segment.to_dict() for segment in segments], columns=columns)


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
            ranges.append(
                _RawRange(
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    landmark_name=str(landmark_name),
                    quality_flag=str(quality_flag),
                    source=source,
                )
            )
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


def _segment_type(length: int, flags: set[str], max_cluster_length: int) -> str:
    if "missing_long_gap" in flags or length > max_cluster_length:
        return "long_unreliable_run"
    if len(flags) > 1:
        return "mixed_problem_segment"
    if length <= 30:
        return "short_invalid_cluster"
    return "medium_invalid_cluster"
