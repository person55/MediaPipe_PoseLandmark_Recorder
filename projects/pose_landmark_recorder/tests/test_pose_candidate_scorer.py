import math

import pandas as pd

from dance_pose_recorder.pose_candidate_scorer import (
    bone_score_for_length,
    confidence_score,
    decide_candidate,
    temporal_score,
)


def test_high_visibility_presence_gets_higher_confidence():
    low = confidence_score({"visibility": 0.2, "presence": 0.2})
    high = confidence_score({"visibility": 0.9, "presence": 0.8})

    assert high > low


def test_large_jump_gets_lower_temporal_score():
    stable = pd.DataFrame(
        [
            {"frame": 0, "x": 0.0, "y": 0.0},
            {"frame": 2, "x": 2.0, "y": 0.0},
        ]
    )
    near = temporal_score({"frame": 1, "x": 1.0, "y": 0.0}, stable, median_motion=1.0)
    far = temporal_score({"frame": 1, "x": 20.0, "y": 0.0}, stable, median_motion=1.0)

    assert near > far


def test_temporal_score_is_gap_invariant_for_plausible_motion():
    near_anchors = pd.DataFrame(
        [
            {"frame": 4, "x": 4.0, "y": 0.0},
            {"frame": 6, "x": 6.0, "y": 0.0},
        ]
    )
    far_anchors = pd.DataFrame(
        [
            {"frame": 0, "x": 0.0, "y": 0.0},
            {"frame": 10, "x": 10.0, "y": 0.0},
        ]
    )
    row = {"frame": 5, "x": 5.0, "y": 0.0}

    near = temporal_score(row, near_anchors, median_motion=1.0)
    far = temporal_score(row, far_anchors, median_motion=1.0)

    assert abs(near - far) < 1e-6


def test_temporal_score_judges_per_frame_rate_not_raw_distance():
    stable = pd.DataFrame(
        [
            {"frame": 0, "x": 0.0, "y": 0.0},
            {"frame": 10, "x": 10.0, "y": 0.0},
        ]
    )

    score = temporal_score({"frame": 5, "x": 5.0, "y": 0.0}, stable, median_motion=1.0)

    assert abs(score - 0.5) < 1e-6


def test_bone_score_prefers_median_length():
    close = bone_score_for_length(1.05, 1.0)
    far = bone_score_for_length(1.8, 1.0)

    assert close > far


def test_accept_score_margin_controls_acceptance():
    accepted = decide_candidate(0.5, 0.6, accept_score_margin=0.08, candidate_row={"visibility": 0.9})
    rejected = decide_candidate(0.5, 0.55, accept_score_margin=0.08, candidate_row={"visibility": 0.9})

    assert accepted.accepted is True
    assert rejected.accepted is False
    assert rejected.reason == "rejected_lower_score"


def test_nan_candidate_is_rejected():
    decision = decide_candidate(
        0.5,
        0.9,
        accept_score_margin=0.08,
        candidate_row={"visibility": math.nan, "presence": math.nan},
    )

    assert decision.accepted is False
    assert decision.reason == "rejected_missing_candidate"
