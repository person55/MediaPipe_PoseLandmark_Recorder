import math

import pandas as pd

from dance_pose_recorder.bone_constraints import (
    compute_bone_length_statistics,
    is_bone_length_violation,
    reachability_status,
)


def _row(frame, landmark_id, landmark_name, tx, ty, tz, quality_flag="measured"):
    return {
        "session_id": "test",
        "frame": frame,
        "time_sec": frame / 30,
        "landmark_id": landmark_id,
        "landmark_name": landmark_name,
        "source": "pose_world",
        "x": tx,
        "y": ty,
        "z": tz,
        "visibility": 0.9,
        "presence": 0.9,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "is_valid": True,
        "quality_flag": quality_flag,
    }


def test_bone_length_median_uses_reliable_frames():
    df = pd.DataFrame(
        [
            _row(0, 11, "left_shoulder", 0, 0, 0),
            _row(0, 13, "left_elbow", 1, 0, 0),
            _row(1, 11, "left_shoulder", 0, 0, 0),
            _row(1, 13, "left_elbow", 3, 0, 0, quality_flag="unreliable"),
            _row(2, 11, "left_shoulder", 0, 0, 0),
            _row(2, 13, "left_elbow", 1.2, 0, 0),
        ]
    )
    stats, _ = compute_bone_length_statistics(
        df,
        {"left_upper_arm": {"points": ["left_shoulder", "left_elbow"], "min_ratio": 0.45, "max_ratio": 1.75}},
        {"measured"},
        ["tx", "ty", "tz"],
    )

    assert math.isclose(stats["left_upper_arm"].median_length, 1.1, abs_tol=1e-6)


def test_min_ratio_violation_is_detected():
    assert is_bone_length_violation(0.3, 0.45, 1.75) is True


def test_max_ratio_violation_is_detected():
    assert is_bone_length_violation(2.0, 0.45, 1.75) is True


def test_reachability_max_violation_is_detected():
    violation, side, ratio, penalty = reachability_status(
        (0, 0, 0),
        (1, 0, 0),
        (3, 0, 0),
        root_mid_length=1.0,
        mid_end_length=1.0,
        margin_ratio=0.1,
    )

    assert violation is True
    assert side == "max"
    assert ratio > 1.0
    assert penalty > 0


def test_reachability_min_violation_is_detected():
    violation, side, _, penalty = reachability_status(
        (0, 0, 0),
        (5, 0, 0),
        (0.05, 0, 0),
        root_mid_length=2.0,
        mid_end_length=0.5,
        margin_ratio=0.1,
    )

    assert violation is True
    assert side == "min"
    assert penalty > 0
