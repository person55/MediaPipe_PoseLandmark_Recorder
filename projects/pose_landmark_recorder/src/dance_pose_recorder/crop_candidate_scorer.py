"""Scoring and acceptance rules for crop-based pose candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from dance_pose_recorder.pose_candidate_scorer import (
    bone_score,
    confidence_score,
    temporal_score,
)


EPSILON = 1e-6


@dataclass(frozen=True)
class CropScoreBreakdown:
    confidence_score: float
    temporal_score: float
    crop_valid_score: float
    bone_score: float
    visibility_gain: float

    @property
    def total(self) -> float:
        return float(
            self.confidence_score * 0.30
            + self.temporal_score * 0.30
            + self.crop_valid_score * 0.15
            + self.bone_score * 0.15
            + self.visibility_gain * 0.10
        )


@dataclass(frozen=True)
class CropCandidateDecision:
    accepted: bool
    reason: str
    score_before: float
    score_after: float
    score_delta: float


def crop_valid_score(row: pd.Series | dict | None) -> float:
    if row is None:
        return 0.0
    if bool(_get(row, "crop_edge_risk") or False):
        return 0.0
    x_crop = _finite_float(_get(row, "crop_x_norm"))
    y_crop = _finite_float(_get(row, "crop_y_norm"))
    if x_crop is None or y_crop is None:
        return 1.0
    if x_crop < 0.03 or y_crop < 0.03 or x_crop > 0.97 or y_crop > 0.97:
        return 0.5
    return 1.0


def visibility_gain_score(cleaned_row: pd.Series | dict, candidate_row: pd.Series | dict | None) -> float:
    if candidate_row is None:
        return 0.0
    cleaned_conf = confidence_score(cleaned_row)
    candidate_conf = confidence_score(candidate_row)
    return _clip01((candidate_conf - cleaned_conf + 1.0) / 2.0)


def score_cleaned_row(
    row: pd.Series | dict,
    stable_series: pd.DataFrame,
    frame_rows: pd.DataFrame,
    median_bone_lengths: dict[str, float],
    source: str = "pose",
) -> CropScoreBreakdown:
    return CropScoreBreakdown(
        confidence_score=confidence_score(row),
        temporal_score=temporal_score(row, stable_series, source=source),
        crop_valid_score=1.0,
        bone_score=_safe_bone_score(row, frame_rows, median_bone_lengths, source=source),
        visibility_gain=0.5,
    )


def score_crop_candidate(
    cleaned_row: pd.Series | dict,
    candidate_row: pd.Series | dict,
    stable_series: pd.DataFrame,
    candidate_frame_rows: pd.DataFrame,
    median_bone_lengths: dict[str, float],
    source: str = "pose",
) -> CropScoreBreakdown:
    return CropScoreBreakdown(
        confidence_score=confidence_score(candidate_row),
        temporal_score=temporal_score(candidate_row, stable_series, source=source),
        crop_valid_score=crop_valid_score(candidate_row),
        bone_score=_safe_bone_score(candidate_row, candidate_frame_rows, median_bone_lengths, source=source),
        visibility_gain=visibility_gain_score(cleaned_row, candidate_row),
    )


def decide_crop_candidate(
    cleaned_score: float,
    crop_score: float,
    accept_score_margin: float,
    candidate_row: pd.Series | dict | None = None,
    review_only: bool = False,
) -> CropCandidateDecision:
    if candidate_row is None:
        return _decision(False, "rejected_missing_candidate", cleaned_score, cleaned_score)
    if review_only:
        return _decision(False, "rejected_review_only", cleaned_score, crop_score)
    if _both_confidence_low_or_missing(candidate_row):
        return _decision(False, "rejected_missing_candidate", cleaned_score, crop_score)
    if crop_valid_score(candidate_row) <= 0.0:
        return _decision(False, "rejected_crop_edge", cleaned_score, crop_score)
    if crop_score >= cleaned_score + accept_score_margin:
        return _decision(True, "accepted_higher_score", cleaned_score, crop_score)
    return _decision(False, "rejected_lower_score", cleaned_score, crop_score)


def _decision(accepted: bool, reason: str, before: float, after: float) -> CropCandidateDecision:
    return CropCandidateDecision(
        accepted=accepted,
        reason=reason,
        score_before=float(before),
        score_after=float(after),
        score_delta=float(after - before),
    )


def _safe_bone_score(
    row: pd.Series | dict,
    frame_rows: pd.DataFrame,
    median_bone_lengths: dict[str, float],
    source: str,
) -> float:
    if frame_rows.empty or "landmark_id" not in frame_rows.columns:
        return 0.5
    return bone_score(row, frame_rows, median_bone_lengths, source=source)


def _both_confidence_low_or_missing(row: pd.Series | dict) -> bool:
    visibility = _finite_float(_get(row, "visibility"))
    presence = _finite_float(_get(row, "presence"))
    if visibility is None and presence is None:
        return True
    visibility = visibility if visibility is not None else 0.0
    presence = presence if presence is not None else 0.0
    return visibility < 0.05 and presence < 0.05


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
