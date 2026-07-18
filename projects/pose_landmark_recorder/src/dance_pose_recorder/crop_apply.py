"""Acceptance and merge logic for crop-based pose candidates.

Moved out of scripts/crop_refine_pose.py so the merge rules that rewrite
motion data live in a tested module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dance_pose_recorder.crop_candidate_scorer import (
    crop_valid_score,
    decide_crop_candidate,
    visibility_gain_score,
)
from dance_pose_recorder.crop_refiner import CropSegment
from dance_pose_recorder.landmark_schema import POSE_LANDMARK_NAMES
from dance_pose_recorder.pose_candidate_scorer import confidence_score
from dance_pose_recorder.quality_flags import STABLE_MEASUREMENT_FLAGS
from dance_pose_recorder.stage_schema import COORD_FIELDS


def apply_crop_candidates(
    cleaned: pd.DataFrame,
    candidates: pd.DataFrame,
    segments: list[CropSegment],
    accept_score_margin: float,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    refined = cleaned.copy()
    initialize_crop_columns(refined)

    landmark_name_to_id = {name: index for index, name in enumerate(POSE_LANDMARK_NAMES)}
    candidate_lookup = _candidate_lookup(candidates)
    temporal_refs = _temporal_references(cleaned)
    score_rows: list[dict] = []
    segment_summaries: list[dict] = []

    for segment in segments:
        target_ids = {landmark_name_to_id[name] for name in segment.target_landmarks if name in landmark_name_to_id}
        if not segment.selected_for_crop:
            summary = segment.to_dict()
            summary.update(
                {
                    "accepted_rows": 0,
                    "rejected_rows": 0,
                    "unavailable_rows": 0,
                }
            )
            segment_summaries.append(summary)
            continue
        segment_mask = (
            refined["frame"].between(segment.start_frame, segment.end_frame)
            & refined["landmark_id"].isin(target_ids)
            & refined["source"].isin(["pose", "pose_world"])
            & refined["quality_flag"].isin(segment.problem_flags)
        )
        segment_indices = refined[segment_mask].index.tolist()
        counts = {"accepted": 0, "rejected": 0, "unavailable": 0}

        for index in segment_indices:
            row = refined.loc[index]
            source = str(row["source"])
            frame = int(row["frame"])
            landmark_id = int(row["landmark_id"])
            candidate_row = candidate_lookup.get((frame, source, landmark_id))

            before_score = _fast_cleaned_score(
                row,
                temporal_refs.get((source, landmark_id)),
                source=source,
            )

            if candidate_row is None:
                _mark_crop_row(refined, index, "crop_unavailable", "none", before_score, before_score, segment, "rejected_missing_candidate")
                counts["unavailable"] += 1
                score_rows.append(_score_row(row, segment, "crop_unavailable", "rejected_missing_candidate", before_score, before_score))
                continue

            after_score = _fast_crop_score(
                row,
                candidate_row,
                temporal_refs.get((source, landmark_id)),
                source=source,
            )
            decision = decide_crop_candidate(
                before_score,
                after_score,
                accept_score_margin=accept_score_margin,
                candidate_row=candidate_row,
                review_only=segment.review_only,
            )
            if decision.accepted:
                _accept_crop_candidate(refined, index, candidate_row, decision, segment)
                counts["accepted"] += 1
                status = "crop_accepted"
            else:
                _mark_crop_row(
                    refined,
                    index,
                    "crop_rejected",
                    "cleaned",
                    decision.score_before,
                    decision.score_after,
                    segment,
                    decision.reason,
                )
                counts["rejected"] += 1
                status = "crop_rejected"
            score_rows.append(_score_row(row, segment, status, decision.reason, decision.score_before, decision.score_after))

        summary = segment.to_dict()
        summary.update(
            {
                "accepted_rows": counts["accepted"],
                "rejected_rows": counts["rejected"],
                "unavailable_rows": counts["unavailable"],
            }
        )
        segment_summaries.append(summary)

    return refined, score_rows, segment_summaries


def initialize_crop_columns(refined: pd.DataFrame) -> None:
    refined["crop_refine_status"] = "unchanged"
    refined["crop_refine_source"] = "cleaned"
    refined["crop_score_before"] = np.nan
    refined["crop_score_after"] = np.nan
    refined["crop_score_delta"] = np.nan
    refined["crop_segment_id"] = np.nan
    refined["crop_reason"] = "unchanged_not_target"
    refined["crop_x0"] = np.nan
    refined["crop_y0"] = np.nan
    refined["crop_w"] = np.nan
    refined["crop_h"] = np.nan
    refined["crop_margin_ratio"] = np.nan
    refined["crop_running_mode"] = ""


def _accept_crop_candidate(
    refined: pd.DataFrame,
    index: int,
    candidate_row,
    decision,
    segment: CropSegment,
) -> None:
    for field in COORD_FIELDS:
        if hasattr(candidate_row, field):
            refined.at[index, field] = getattr(candidate_row, field)
    refined.at[index, "is_valid"] = True
    refined.at[index, "is_interpolated"] = False
    refined.at[index, "quality_flag"] = "crop_refined_measured"
    refined.at[index, "invalid_reason"] = ""
    refined.at[index, "interpolation_method"] = ""
    refined.at[index, "gap_length"] = np.nan
    refined.at[index, "source_frame_prev"] = np.nan
    refined.at[index, "source_frame_next"] = np.nan
    _mark_crop_row(
        refined,
        index,
        "crop_accepted",
        "crop_video",
        decision.score_before,
        decision.score_after,
        segment,
        decision.reason,
        candidate_row,
    )


def _mark_crop_row(
    refined: pd.DataFrame,
    index: int,
    status: str,
    source: str,
    before: float,
    after: float,
    segment: CropSegment,
    reason: str,
    candidate_row=None,
) -> None:
    refined.at[index, "crop_refine_status"] = status
    refined.at[index, "crop_refine_source"] = source
    refined.at[index, "crop_score_before"] = before
    refined.at[index, "crop_score_after"] = after
    refined.at[index, "crop_score_delta"] = after - before
    refined.at[index, "crop_segment_id"] = segment.crop_segment_id
    refined.at[index, "crop_reason"] = reason
    if candidate_row is not None:
        for target, attr in (
            ("crop_x0", "crop_x0"),
            ("crop_y0", "crop_y0"),
            ("crop_w", "crop_w"),
            ("crop_h", "crop_h"),
            ("crop_margin_ratio", "crop_margin_ratio"),
            ("crop_running_mode", "crop_running_mode"),
        ):
            if hasattr(candidate_row, attr):
                refined.at[index, target] = getattr(candidate_row, attr)


def _score_row(row: pd.Series, segment: CropSegment, status: str, reason: str, before: float, after: float) -> dict:
    return {
        "crop_segment_id": segment.crop_segment_id,
        "frame": int(row["frame"]),
        "source": row["source"],
        "landmark_id": int(row["landmark_id"]),
        "landmark_name": row["landmark_name"],
        "quality_flag_before": row["quality_flag"],
        "crop_refine_status": status,
        "crop_reason": reason,
        "score_before": before,
        "score_after": after,
        "score_delta": after - before,
    }


def _candidate_lookup(candidates: pd.DataFrame) -> dict[tuple[int, str, int], object]:
    if candidates.empty:
        return {}
    return {
        (int(row.frame), str(row.source), int(row.landmark_id)): row
        for row in candidates.itertuples(index=False)
    }


def _temporal_references(cleaned: pd.DataFrame) -> dict[tuple[str, int], dict]:
    refs: dict[tuple[str, int], dict] = {}
    for (source, landmark_id), group in cleaned[cleaned["quality_flag"].isin(STABLE_MEASUREMENT_FLAGS)].groupby(
        ["source", "landmark_id"], sort=False
    ):
        fields = _coord_fields(str(source))
        usable = group.dropna(subset=fields).sort_values("frame")
        if usable.empty:
            continue
        frames = usable["frame"].to_numpy(dtype=np.int64)
        coords = usable[fields].to_numpy(dtype=float)
        if len(coords) > 1:
            distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            gaps = np.maximum(1.0, np.diff(frames).astype(float))
            median_motion = float(np.median(distances / gaps))
        else:
            median_motion = 1.0
        refs[(str(source), int(landmark_id))] = {
            "frames": frames,
            "coords": coords,
            "median_motion": median_motion if median_motion > 0 else 1.0,
        }
    return refs


def _fast_cleaned_score(row, temporal_ref: dict | None, source: str) -> float:
    return float(
        confidence_score(row) * 0.30
        + _fast_temporal_score(row, temporal_ref, source) * 0.30
        + 1.0 * 0.15
        + 0.5 * 0.15
        + 0.5 * 0.10
    )


def _fast_crop_score(cleaned_row, candidate_row, temporal_ref: dict | None, source: str) -> float:
    return float(
        confidence_score(candidate_row) * 0.30
        + _fast_temporal_score(candidate_row, temporal_ref, source) * 0.30
        + crop_valid_score(candidate_row) * 0.15
        + 0.5 * 0.15
        + visibility_gain_score(cleaned_row, candidate_row) * 0.10
    )


def _fast_temporal_score(row, temporal_ref: dict | None, source: str) -> float:
    coords = _row_coords(row, source)
    if temporal_ref is None or coords is None:
        return 0.5
    frames = temporal_ref["frames"]
    stable_coords = temporal_ref["coords"]
    if len(frames) == 0:
        return 0.5
    frame = int(_row_get(row, "frame"))
    pos = int(np.searchsorted(frames, frame))
    # Per-frame rates (distance / frame gap to the anchor), matching
    # pose_candidate_scorer.temporal_score so interpolated cleaned values do
    # not geometrically dominate genuine re-detections far from anchors.
    rates = []
    if pos > 0:
        gap = max(1, frame - int(frames[pos - 1]))
        rates.append(float(np.linalg.norm(coords - stable_coords[pos - 1])) / gap)
    if pos < len(frames):
        gap = max(1, int(frames[pos]) - frame)
        rates.append(float(np.linalg.norm(coords - stable_coords[pos])) / gap)
    if not rates:
        return 0.5
    rate = float(sum(rates) / len(rates))
    ratio = rate / (float(temporal_ref["median_motion"]) + 1e-6)
    return float(min(1.0, max(0.0, 1.0 / (1.0 + ratio))))


def _row_coords(row, source: str) -> np.ndarray | None:
    values = []
    for field in _coord_fields(source):
        value = _row_get(row, field)
        if value is None or pd.isna(value):
            return None
        values.append(float(value))
    return np.array(values, dtype=float)


def _coord_fields(source: str) -> list[str]:
    if source == "pose_world":
        return ["tx", "ty", "tz"]
    return ["x", "y"]


def _row_get(row, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, row.get(key) if hasattr(row, "get") else None)
