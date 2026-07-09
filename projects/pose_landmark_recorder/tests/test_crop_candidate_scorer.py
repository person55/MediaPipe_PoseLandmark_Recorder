import pandas as pd

from dance_pose_recorder.crop_candidate_scorer import (
    crop_valid_score,
    decide_crop_candidate,
    score_cleaned_row,
    score_crop_candidate,
    visibility_gain_score,
)


def test_crop_valid_score_penalizes_edge_candidates():
    center = {"crop_x_norm": 0.5, "crop_y_norm": 0.5, "crop_edge_risk": False}
    edge = {"crop_x_norm": 0.01, "crop_y_norm": 0.5, "crop_edge_risk": True}

    assert crop_valid_score(center) > crop_valid_score(edge)


def test_visibility_gain_prefers_higher_confidence_candidate():
    cleaned = {"visibility": 0.2, "presence": 0.2}
    crop = {"visibility": 0.9, "presence": 0.8}

    assert visibility_gain_score(cleaned, crop) > 0.5


def test_crop_candidate_with_better_confidence_scores_higher():
    stable = pd.DataFrame(
        [
            {"frame": 0, "x": 0.0, "y": 0.0},
            {"frame": 2, "x": 2.0, "y": 0.0},
        ]
    )
    frame_rows = pd.DataFrame()
    cleaned = {"frame": 1, "landmark_id": 15, "x": 1.0, "y": 0.0, "visibility": 0.2, "presence": 0.2}
    crop = {
        "frame": 1,
        "landmark_id": 15,
        "x": 1.0,
        "y": 0.0,
        "visibility": 0.9,
        "presence": 0.8,
        "crop_x_norm": 0.5,
        "crop_y_norm": 0.5,
        "crop_edge_risk": False,
    }

    before = score_cleaned_row(cleaned, stable, frame_rows, {}, source="pose")
    after = score_crop_candidate(cleaned, crop, stable, frame_rows, {}, source="pose")

    assert after.total > before.total


def test_accept_score_margin_controls_crop_acceptance():
    accepted = decide_crop_candidate(0.5, 0.6, 0.06, {"visibility": 0.9, "presence": 0.9})
    rejected = decide_crop_candidate(0.5, 0.54, 0.06, {"visibility": 0.9, "presence": 0.9})

    assert accepted.accepted is True
    assert rejected.accepted is False
    assert rejected.reason == "rejected_lower_score"


def test_missing_or_edge_candidate_is_rejected():
    missing = decide_crop_candidate(0.5, 0.8, 0.06, None)
    edge = decide_crop_candidate(0.5, 0.9, 0.06, {"visibility": 0.9, "presence": 0.9, "crop_edge_risk": True})

    assert missing.accepted is False
    assert missing.reason == "rejected_missing_candidate"
    assert edge.accepted is False
    assert edge.reason == "rejected_crop_edge"
