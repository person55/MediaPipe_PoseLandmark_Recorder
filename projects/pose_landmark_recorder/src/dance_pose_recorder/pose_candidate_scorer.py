"""Scoring and acceptance rules for re-detected pose candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from dance_pose_recorder.quality_flags import STABLE_MEASUREMENT_FLAGS


COORD_FIELDS = {
    "pose": ["x", "y"],
    "pose_world": ["tx", "ty", "tz"],
}

BONE_PAIRS = [
    (11, 13, "left_upper_arm"),
    (13, 15, "left_lower_arm"),
    (12, 14, "right_upper_arm"),
    (14, 16, "right_lower_arm"),
    (23, 25, "left_upper_leg"),
    (25, 27, "left_lower_leg"),
    (24, 26, "right_upper_leg"),
    (26, 28, "right_lower_leg"),
]

STABLE_FLAGS = STABLE_MEASUREMENT_FLAGS
EPSILON = 1e-6


@dataclass(frozen=True)
class ScoreBreakdown:
    confidence_score: float
    temporal_score: float
    bone_score: float
    continuity_score: float = 0.5

    @property
    def total(self) -> float:
        return float(
            self.confidence_score * 0.35
            + self.temporal_score * 0.30
            + self.bone_score * 0.25
            + self.continuity_score * 0.10
        )


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    reason: str
    score_before: float
    score_after: float
    score_delta: float


def confidence_score(row: pd.Series | dict) -> float:
    values = [_finite_float(_get(row, "visibility")), _finite_float(_get(row, "presence"))]
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return _clip01(float(sum(values) / len(values)))


def temporal_score(
    row: pd.Series | dict,
    stable_series: pd.DataFrame,
    median_motion: float | None = None,
    source: str = "pose",
) -> float:
    frame = int(_get(row, "frame"))
    coords = _coords(row, source)
    if coords is None or stable_series.empty:
        return 0.5

    previous = stable_series[stable_series["frame"] < frame].tail(1)
    next_row = stable_series[stable_series["frame"] > frame].head(1)
    # Distances are divided by the frame gap to the anchor so a candidate far
    # from its anchors is judged on per-frame motion, not raw displacement.
    # Without this, interpolated values geometrically always win.
    rates = []
    for neighbor in (previous, next_row):
        if neighbor.empty:
            continue
        neighbor_coords = _coords(neighbor.iloc[0], source)
        if neighbor_coords is not None:
            gap = max(1, abs(frame - int(neighbor.iloc[0]["frame"])))
            rates.append(_distance(coords, neighbor_coords) / gap)
    if not rates:
        return 0.5

    baseline = float(median_motion if median_motion and median_motion > 0 else _median_motion(stable_series, source))
    candidate_rate = float(sum(rates) / len(rates))
    jump_ratio = candidate_rate / (baseline + EPSILON)
    return _clip01(1.0 / (1.0 + jump_ratio))


def bone_score_for_length(candidate_length: float | None, median_length: float | None) -> float:
    if candidate_length is None or median_length is None or median_length <= EPSILON:
        return 0.5
    ratio = abs(float(candidate_length) - float(median_length)) / (float(median_length) + EPSILON)
    return _clip01(max(0.0, 1.0 - ratio))


def bone_score(
    row: pd.Series | dict,
    frame_rows: pd.DataFrame,
    median_bone_lengths: dict[str, float],
    source: str = "pose",
) -> float:
    landmark_id = int(_get(row, "landmark_id"))
    scores = []
    for start_id, end_id, bone_name in BONE_PAIRS:
        if landmark_id not in {start_id, end_id}:
            continue
        candidate_length = _bone_length(frame_rows, start_id, end_id, source)
        scores.append(bone_score_for_length(candidate_length, median_bone_lengths.get(bone_name)))
    if not scores:
        return 0.5
    return _clip01(float(sum(scores) / len(scores)))


def score_row(
    row: pd.Series | dict,
    stable_series: pd.DataFrame,
    frame_rows: pd.DataFrame,
    median_bone_lengths: dict[str, float],
    source: str = "pose",
    continuity_score_value: float = 0.5,
    median_motion: float | None = None,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        confidence_score=confidence_score(row),
        temporal_score=temporal_score(row, stable_series, median_motion=median_motion, source=source),
        bone_score=bone_score(row, frame_rows, median_bone_lengths, source=source),
        continuity_score=continuity_score_value,
    )


def decide_candidate(
    cleaned_score: float,
    candidate_score: float,
    accept_score_margin: float,
    candidate_row: pd.Series | dict | None = None,
    candidate_jump: float | None = None,
    cleaned_jump: float | None = None,
    candidate_bone_ratio: float | None = None,
    review_only: bool = False,
) -> CandidateDecision:
    if candidate_row is None:
        return _decision(False, "rejected_missing_candidate", cleaned_score, cleaned_score)
    if review_only:
        return _decision(False, "rejected_review_only", cleaned_score, candidate_score)
    if _both_confidence_values_missing(candidate_row):
        return _decision(False, "rejected_missing_candidate", cleaned_score, candidate_score)
    if candidate_jump is not None and cleaned_jump is not None and candidate_jump > max(cleaned_jump * 2.0, EPSILON):
        return _decision(False, "rejected_temporal_jump", cleaned_score, candidate_score)
    if candidate_bone_ratio is not None and (candidate_bone_ratio < 0.4 or candidate_bone_ratio > 1.8):
        return _decision(False, "rejected_bone_outlier", cleaned_score, candidate_score)
    if candidate_score >= cleaned_score + accept_score_margin:
        return _decision(True, "accepted_higher_confidence", cleaned_score, candidate_score)
    return _decision(False, "rejected_lower_score", cleaned_score, candidate_score)


def stable_landmark_series(cleaned: pd.DataFrame, source: str, landmark_id: int) -> pd.DataFrame:
    group = cleaned[(cleaned["source"] == source) & (cleaned["landmark_id"] == landmark_id)].copy()
    coord_fields = COORD_FIELDS.get(source, ["x", "y"])
    stable = group[group["quality_flag"].isin(STABLE_FLAGS)]
    return stable.dropna(subset=coord_fields).sort_values("frame")


def median_bone_lengths(cleaned: pd.DataFrame, source: str = "pose") -> dict[str, float]:
    result: dict[str, float] = {}
    frame_groups = cleaned[(cleaned["source"] == source) & (cleaned["quality_flag"].isin(STABLE_FLAGS))].groupby("frame")
    lengths_by_name: dict[str, list[float]] = {name: [] for _, _, name in BONE_PAIRS}
    for _, frame_rows in frame_groups:
        for start_id, end_id, bone_name in BONE_PAIRS:
            length = _bone_length(frame_rows, start_id, end_id, source)
            if length is not None:
                lengths_by_name[bone_name].append(length)
    for bone_name, lengths in lengths_by_name.items():
        if lengths:
            result[bone_name] = float(np.median(lengths))
    return result


def _decision(accepted: bool, reason: str, before: float, after: float) -> CandidateDecision:
    return CandidateDecision(
        accepted=accepted,
        reason=reason,
        score_before=float(before),
        score_after=float(after),
        score_delta=float(after - before),
    )


def _bone_length(frame_rows: pd.DataFrame, start_id: int, end_id: int, source: str) -> float | None:
    start = frame_rows[frame_rows["landmark_id"] == start_id]
    end = frame_rows[frame_rows["landmark_id"] == end_id]
    if start.empty or end.empty:
        return None
    start_coords = _coords(start.iloc[0], source)
    end_coords = _coords(end.iloc[0], source)
    if start_coords is None or end_coords is None:
        return None
    return _distance(start_coords, end_coords)


def _coords(row: pd.Series | dict, source: str) -> tuple[float, ...] | None:
    fields = COORD_FIELDS.get(source, ["x", "y"])
    values = []
    for field in fields:
        value = _finite_float(_get(row, field))
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _median_motion(stable_series: pd.DataFrame, source: str) -> float:
    """Median per-frame motion between successive stable rows."""

    rates = []
    previous = None
    previous_frame = None
    for row in stable_series.sort_values("frame").itertuples(index=False):
        data = row._asdict()
        current = _coords(data, source)
        frame = int(data["frame"])
        if current is not None and previous is not None:
            gap = max(1, frame - previous_frame)
            rates.append(_distance(previous, current) / gap)
        if current is not None:
            previous = current
            previous_frame = frame
    if not rates:
        return 1.0
    return float(np.median(rates))


def _distance(start: tuple[float, ...], end: tuple[float, ...]) -> float:
    return float(sum((b - a) ** 2 for a, b in zip(start, end)) ** 0.5)


def _both_confidence_values_missing(row: pd.Series | dict) -> bool:
    return _finite_float(_get(row, "visibility")) is None and _finite_float(_get(row, "presence")) is None


def _get(row: pd.Series | dict, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, row.get(key) if hasattr(row, "get") else None)


def _finite_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return number


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
