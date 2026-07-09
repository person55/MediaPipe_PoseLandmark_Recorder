import numpy as np
import pandas as pd

from dance_pose_recorder.temporal_features import compute_temporal_features


def _row(frame, tx, quality_flag="measured"):
    return {
        "frame": frame,
        "source": "pose_world",
        "landmark_name": "left_wrist",
        "tx": tx,
        "ty": 0.0,
        "tz": 0.0,
        "quality_flag": quality_flag,
    }


def test_constant_motion_has_constant_velocity():
    df = pd.DataFrame([_row(frame, float(frame)) for frame in range(5)])

    features = compute_temporal_features(df)

    assert features["velocity"].dropna().tolist() == [1.0, 1.0, 1.0, 1.0]


def test_jump_creates_velocity_spike_candidate():
    df = pd.DataFrame([_row(0, 0.0), _row(1, 1.0), _row(2, 20.0)])

    features = compute_temporal_features(df)

    assert features.loc[2, "velocity"] > features.loc[1, "velocity"]


def test_acceleration_and_jerk_are_calculated():
    df = pd.DataFrame([_row(0, 0.0), _row(1, 1.0), _row(2, 4.0), _row(3, 8.0)])

    features = compute_temporal_features(df)

    assert np.isfinite(features.loc[2, "acceleration"])
    assert np.isfinite(features.loc[3, "jerk"])


def test_frame_gap_breaks_direct_connection():
    df = pd.DataFrame([_row(0, 0.0), _row(2, 2.0)])

    features = compute_temporal_features(df)

    assert np.isnan(features.loc[1, "velocity"])


def test_nan_coordinate_creates_nan_feature():
    df = pd.DataFrame([_row(0, 0.0), _row(1, np.nan)])

    features = compute_temporal_features(df)

    assert np.isnan(features.loc[1, "velocity"])
