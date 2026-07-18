"""Acceptance and merge logic for full-frame re-detected pose candidates.

Moved out of scripts/refine_pose_segments.py so the merge rules that rewrite
motion data live in a tested module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dance_pose_recorder.landmark_schema import POSE_LANDMARK_NAMES
from dance_pose_recorder.pose_candidate_scorer import (
    decide_candidate,
    median_bone_lengths,
    score_row,
    stable_landmark_series,
)
from dance_pose_recorder.segment_refiner import CandidateSegment
from dance_pose_recorder.stage_schema import COORD_FIELDS


def apply_refine_candidates(
    cleaned: pd.DataFrame,
    candidates: pd.DataFrame,
    segments: list[CandidateSegment],
    accept_score_margin: float,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    refined = cleaned.copy()
    refined["refine_status"] = "unchanged"
    refined["refine_source"] = "cleaned"
    refined["refine_score_before"] = np.nan
    refined["refine_score_after"] = np.nan
    refined["refine_score_delta"] = np.nan
    refined["refine_segment_id"] = np.nan
    refined["refine_reason"] = "unchanged_not_target"

    landmark_name_to_id = {
        name: index for index, name in enumerate(POSE_LANDMARK_NAMES)
    }
    candidate_lookup = _candidate_lookup(candidates)
    candidate_frame_cache = _frame_source_cache(candidates)
    cleaned_frame_cache = _frame_source_cache(cleaned)
    median_bones = {
        "pose": median_bone_lengths(cleaned, "pose"),
        "pose_world": median_bone_lengths(cleaned, "pose_world"),
    }
    stable_cache: dict[tuple[str, int], pd.DataFrame] = {}
    median_motion_cache: dict[tuple[str, int], float] = {}
    score_rows: list[dict] = []
    segment_summaries: list[dict] = []

    for segment in segments:
        target_ids = {landmark_name_to_id[name] for name in segment.target_landmarks if name in landmark_name_to_id}
        segment_mask = (
            refined["frame"].between(segment.start_frame, segment.end_frame)
            & refined["landmark_id"].isin(target_ids)
            & refined["source"].isin(["pose", "pose_world"])
            & refined["quality_flag"].isin(segment.problem_flags)
        )
        segment_indices = refined[segment_mask].index.tolist()
        counts = {"accepted": 0, "rejected": 0, "unavailable": 0}
        redetected = False

        for index in segment_indices:
            row = refined.loc[index]
            source = str(row["source"])
            frame = int(row["frame"])
            landmark_id = int(row["landmark_id"])
            key = (frame, source, landmark_id)
            candidate_row = candidate_lookup.get(key)
            if candidate_row is not None:
                redetected = True

            stable_key = (source, landmark_id)
            if stable_key not in stable_cache:
                stable_cache[stable_key] = stable_landmark_series(cleaned, source, landmark_id)
                median_motion_cache[stable_key] = _median_motion_for_series(stable_cache[stable_key], source)

            before_frame_rows = cleaned_frame_cache.get((frame, source), pd.DataFrame())
            before_score = score_row(
                row,
                stable_cache[stable_key],
                before_frame_rows,
                median_bones.get(source, {}),
                source=source,
                median_motion=median_motion_cache[stable_key],
            ).total

            if candidate_row is None:
                _mark_refine_row(refined, index, "refined_unavailable", "none", before_score, before_score, segment, "rejected_missing_candidate")
                counts["unavailable"] += 1
                score_rows.append(_score_row(row, segment, "refined_unavailable", "rejected_missing_candidate", before_score, before_score))
                continue

            candidate_frame_rows = candidate_frame_cache.get((frame, source), pd.DataFrame())
            after_score = score_row(
                candidate_row,
                stable_cache[stable_key],
                candidate_frame_rows,
                median_bones.get(source, {}),
                source=source,
                median_motion=median_motion_cache[stable_key],
            ).total
            decision = decide_candidate(
                cleaned_score=before_score,
                candidate_score=after_score,
                accept_score_margin=accept_score_margin,
                candidate_row=candidate_row,
                review_only=segment.review_only,
            )
            if decision.accepted:
                _accept_candidate(refined, index, candidate_row, decision, segment)
                counts["accepted"] += 1
                status = "refined_accepted"
            else:
                _mark_refine_row(
                    refined,
                    index,
                    "refined_rejected",
                    "redetect_full_frame",
                    decision.score_before,
                    decision.score_after,
                    segment,
                    decision.reason,
                )
                counts["rejected"] += 1
                status = "refined_rejected"
            score_rows.append(
                _score_row(
                    row,
                    segment,
                    status,
                    decision.reason,
                    decision.score_before,
                    decision.score_after,
                )
            )

        summary = segment.to_dict()
        summary.update(
            {
                "accepted_rows": counts["accepted"],
                "rejected_rows": counts["rejected"],
                "unavailable_rows": counts["unavailable"],
                "redetected": redetected,
            }
        )
        segment_summaries.append(summary)

    return refined, score_rows, segment_summaries


def _candidate_lookup(candidates: pd.DataFrame) -> dict[tuple[int, str, int], pd.Series]:
    if candidates.empty:
        return {}
    return {
        (int(row.frame), str(row.source), int(row.landmark_id)): row
        for row in candidates.itertuples(index=False)
    }


def _frame_source_cache(frame_rows: pd.DataFrame) -> dict[tuple[int, str], pd.DataFrame]:
    if frame_rows.empty:
        return {}
    return {
        (int(frame), str(source)): group.copy()
        for (frame, source), group in frame_rows.groupby(["frame", "source"], sort=False)
    }


def _median_motion_for_series(stable_series: pd.DataFrame, source: str) -> float:
    if stable_series.empty:
        return 1.0
    fields = ["tx", "ty", "tz"] if source == "pose_world" else ["x", "y"]
    usable = stable_series.dropna(subset=fields).sort_values("frame")
    if len(usable) < 2:
        return 1.0
    coords = usable[fields].to_numpy(dtype=float)
    frames = usable["frame"].to_numpy(dtype=float)
    distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    gaps = np.maximum(1.0, np.diff(frames))
    if len(distances) == 0:
        return 1.0
    median = float(np.median(distances / gaps))
    return median if median > 0 else 1.0


def _accept_candidate(
    refined: pd.DataFrame,
    index: int,
    candidate_row: pd.Series,
    decision,
    segment: CandidateSegment,
) -> None:
    for field in COORD_FIELDS:
        if hasattr(candidate_row, field):
            refined.at[index, field] = getattr(candidate_row, field)
    refined.at[index, "is_valid"] = True
    refined.at[index, "is_interpolated"] = False
    refined.at[index, "quality_flag"] = "refined_measured"
    refined.at[index, "invalid_reason"] = ""
    refined.at[index, "interpolation_method"] = ""
    refined.at[index, "gap_length"] = np.nan
    refined.at[index, "source_frame_prev"] = np.nan
    refined.at[index, "source_frame_next"] = np.nan
    _mark_refine_row(
        refined,
        index,
        "refined_accepted",
        "redetect_full_frame",
        decision.score_before,
        decision.score_after,
        segment,
        decision.reason,
    )


def _mark_refine_row(
    refined: pd.DataFrame,
    index: int,
    status: str,
    source: str,
    before: float,
    after: float,
    segment: CandidateSegment,
    reason: str,
) -> None:
    refined.at[index, "refine_status"] = status
    refined.at[index, "refine_source"] = source
    refined.at[index, "refine_score_before"] = before
    refined.at[index, "refine_score_after"] = after
    refined.at[index, "refine_score_delta"] = after - before
    refined.at[index, "refine_segment_id"] = segment.segment_id
    refined.at[index, "refine_reason"] = reason


def _score_row(row: pd.Series, segment: CandidateSegment, status: str, reason: str, before: float, after: float) -> dict:
    return {
        "segment_id": segment.segment_id,
        "frame": int(row["frame"]),
        "source": row["source"],
        "landmark_id": int(row["landmark_id"]),
        "landmark_name": row["landmark_name"],
        "quality_flag_before": row["quality_flag"],
        "refine_status": status,
        "refine_reason": reason,
        "score_before": before,
        "score_after": after,
        "score_delta": after - before,
    }
