"""Contract tests for the stage merge logic and stage output schemas."""

import numpy as np
import pandas as pd

from dance_pose_recorder.crop_apply import apply_crop_candidates
from dance_pose_recorder.crop_refiner import CropSegment
from dance_pose_recorder.refine_apply import apply_refine_candidates
from dance_pose_recorder.segment_refiner import CandidateSegment
from dance_pose_recorder.stage_schema import (
    CROP_SCORE_COLUMNS,
    CROP_STAGE_COLUMNS,
    REFINE_SCORE_COLUMNS,
    REFINE_STAGE_COLUMNS,
)


def _cleaned_row(frame, source, position, quality_flag="measured", visibility=0.9):
    return {
        "session_id": "test",
        "frame": frame,
        "time_sec": frame / 30.0,
        "landmark_id": 15,
        "landmark_name": "left_wrist",
        "source": source,
        "x": position,
        "y": 0.5,
        "z": 0.0,
        "visibility": visibility,
        "presence": visibility,
        "tx": position,
        "ty": 0.0,
        "tz": 0.0,
        "quality_flag": quality_flag,
        "is_valid": quality_flag == "measured",
        "is_interpolated": False,
        "invalid_reason": "",
        "interpolation_method": "",
        "gap_length": np.nan,
        "source_frame_prev": np.nan,
        "source_frame_next": np.nan,
    }


def _synthetic_cleaned():
    rows = []
    for frame in range(10):
        for source in ("pose", "pose_world"):
            if frame == 5:
                # off-trajectory unreliable row the candidate should replace
                rows.append(_cleaned_row(frame, source, 0.5, quality_flag="unreliable", visibility=0.2))
            elif frame == 4:
                rows.append(_cleaned_row(frame, source, 0.5, quality_flag="unreliable", visibility=0.2))
            else:
                rows.append(_cleaned_row(frame, source, 0.05 * frame))
    return pd.DataFrame(rows)


def _candidate_row(frame, source, position, visibility=0.95):
    row = _cleaned_row(frame, source, position, quality_flag="measured", visibility=visibility)
    row.update(
        {
            "crop_x_norm": 0.5,
            "crop_y_norm": 0.5,
            "crop_edge_risk": False,
            "crop_segment_id": 1,
            "crop_x0": 100.0,
            "crop_y0": 100.0,
            "crop_w": 480.0,
            "crop_h": 480.0,
            "crop_margin_ratio": 1.65,
            "crop_running_mode": "video",
        }
    )
    return row


def test_apply_crop_candidates_contract():
    cleaned = _synthetic_cleaned()
    # candidate only for frame 5: frame 4 must be reported crop_unavailable
    candidates = pd.DataFrame(
        [
            _candidate_row(5, "pose", 0.25),
            _candidate_row(5, "pose_world", 0.25, visibility=0.01),
        ]
    )
    segment = CropSegment(
        crop_segment_id=1,
        start_frame=4,
        end_frame=6,
        target_landmarks={"left_wrist"},
        problem_flags={"unreliable"},
    )

    refined, score_rows, summaries = apply_crop_candidates(cleaned, candidates, [segment], accept_score_margin=0.04)

    assert set(CROP_STAGE_COLUMNS) <= set(refined.columns)
    assert set(score_rows[0].keys()) == set(CROP_SCORE_COLUMNS)

    accepted = refined[(refined["frame"] == 5) & (refined["source"] == "pose")].iloc[0]
    assert accepted["crop_refine_status"] == "crop_accepted"
    assert accepted["quality_flag"] == "crop_refined_measured"
    assert abs(float(accepted["x"]) - 0.25) < 1e-9

    rejected = refined[(refined["frame"] == 5) & (refined["source"] == "pose_world")].iloc[0]
    assert rejected["crop_refine_status"] == "crop_rejected"
    assert abs(float(rejected["tx"]) - 0.5) < 1e-9

    unavailable = refined[(refined["frame"] == 4) & (refined["source"] == "pose")].iloc[0]
    assert unavailable["crop_refine_status"] == "crop_unavailable"

    untouched = refined[(refined["frame"] == 3) & (refined["source"] == "pose")].iloc[0]
    assert untouched["crop_refine_status"] == "unchanged"

    assert summaries[0]["accepted_rows"] == 1
    assert summaries[0]["unavailable_rows"] == 2


def test_apply_refine_candidates_contract():
    cleaned = _synthetic_cleaned()
    candidates = pd.DataFrame(
        [
            _candidate_row(5, "pose", 0.25),
            _candidate_row(5, "pose_world", 0.25, visibility=0.01),
        ]
    )
    segment = CandidateSegment(
        segment_id=1,
        start_frame=4,
        end_frame=6,
        target_landmarks={"left_wrist"},
        problem_flags={"unreliable"},
    )

    refined, score_rows, summaries = apply_refine_candidates(cleaned, candidates, [segment], accept_score_margin=0.05)

    assert set(REFINE_STAGE_COLUMNS) <= set(refined.columns)
    assert set(score_rows[0].keys()) == set(REFINE_SCORE_COLUMNS)

    accepted = refined[(refined["frame"] == 5) & (refined["source"] == "pose")].iloc[0]
    assert accepted["refine_status"] == "refined_accepted"
    assert accepted["quality_flag"] == "refined_measured"
    assert abs(float(accepted["x"]) - 0.25) < 1e-9

    unavailable = refined[(refined["frame"] == 4) & (refined["source"] == "pose")].iloc[0]
    assert unavailable["refine_status"] == "refined_unavailable"

    untouched = refined[(refined["frame"] == 3) & (refined["source"] == "pose")].iloc[0]
    assert untouched["refine_status"] == "unchanged"

    assert summaries[0]["accepted_rows"] == 1
    assert summaries[0]["redetected"] is True
