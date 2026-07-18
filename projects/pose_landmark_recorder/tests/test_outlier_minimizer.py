import json

import pandas as pd

from dance_pose_recorder.outlier_minimizer import OutlierMinimizerOptions, minimize_pose_outliers


def _pose_row(frame, landmark_name="left_wrist", tx=0.0, quality_flag="measured"):
    return {
        "session_id": "test_session",
        "frame": frame,
        "time_sec": frame / 30.0,
        "landmark_id": 15,
        "landmark_name": landmark_name,
        "source": "pose_world",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "visibility": 0.9,
        "presence": 0.9,
        "tx": tx,
        "ty": 0.0,
        "tz": 0.0,
        "quality_flag": quality_flag,
    }


def _write_input(tmp_path, rows):
    input_csv = tmp_path / "input_pose.csv"
    metadata = tmp_path / "metadata.json"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    metadata.write_text(json.dumps({"session_id": "test_session", "fps": 30.0, "frame_count_written": len(rows)}))
    return input_csv, metadata


def test_short_spike_is_corrected(tmp_path):
    values = [0.0, 1.0, 2.0, 50.0, 4.0, 5.0, 6.0, 7.0]
    input_csv, metadata = _write_input(tmp_path, [_pose_row(frame, tx=value) for frame, value in enumerate(values)])

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(max_correction_gap_sec=0.2, min_stable_neighbors=1),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])

    assert "outlier_status" in df.columns
    assert "trajectory_connect" in df.columns
    assert (df["outlier_status"] == "outlier_corrected").any()
    assert df.loc[df["frame"] == 3, "tx"].iloc[0] != 50.0
    assert set(df["quality_flag"]) == {"measured"}


def test_long_spike_becomes_trajectory_break(tmp_path):
    values = [float(frame) for frame in range(10)] + [50.0, -50.0, 50.0, -50.0] + [float(frame) for frame in range(14, 20)]
    input_csv, metadata = _write_input(tmp_path, [_pose_row(frame, tx=value) for frame, value in enumerate(values)])

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
            OutlierMinimizerOptions(
                max_correction_gap_sec=0.03,
                velocity_threshold_multiplier=1.1,
                acceleration_threshold_multiplier=1.1,
                jerk_threshold_multiplier=1.1,
                min_stable_neighbors=1,
            ),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])

    assert (df["outlier_status"] == "trajectory_break").any()
    assert (df["trajectory_connect"] == False).any()  # noqa: E712


def test_missing_long_gap_is_not_corrected(tmp_path):
    rows = [_pose_row(frame, tx=float(frame)) for frame in range(5)]
    rows[2]["quality_flag"] = "missing_long_gap"
    input_csv, metadata = _write_input(tmp_path, rows)

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(min_stable_neighbors=1),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])
    row = df[df["frame"] == 2].iloc[0]

    assert row["outlier_status"] == "hidden_unreliable"
    assert bool(row["trajectory_visible"]) is False
    assert row["quality_flag"] == "missing_long_gap"


def test_review_only_is_not_corrected(tmp_path):
    rows = [_pose_row(frame, tx=float(frame)) for frame in range(5)]
    rows[2]["quality_flag"] = "review_only"
    input_csv, metadata = _write_input(tmp_path, rows)

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(min_stable_neighbors=1),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])
    row = df[df["frame"] == 2].iloc[0]

    assert row["outlier_status"] == "review_only"
    assert bool(row["trajectory_connect"]) is False


def test_static_landmark_noise_floor_prevents_false_spike(tmp_path):
    values = [0.0, 0.001, 0.0, 0.001, 0.0, 0.001, 0.0, 0.05, 0.10, 0.15, 0.151, 0.15, 0.151, 0.15]
    input_csv, metadata = _write_input(tmp_path, [_pose_row(frame, tx=value) for frame, value in enumerate(values)])

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(min_stable_neighbors=1),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])

    assert not (df["outlier_status"] == "trajectory_break").any()
    assert not (df["outlier_status"] == "outlier_corrected").any()


def _echo_glitch_values():
    values = [0.1 * frame + (0.001 if frame % 2 else 0.0) for frame in range(12)]
    values[5] += 10.0
    return values


def test_echo_frames_are_trimmed_from_spike_segment(tmp_path):
    input_csv, metadata = _write_input(
        tmp_path, [_pose_row(frame, tx=value) for frame, value in enumerate(_echo_glitch_values())]
    )

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(
            max_correction_gap_sec=0.08,
            velocity_floor=0.01,
            acceleration_floor=0.01,
            jerk_floor=0.01,
            min_stable_neighbors=1,
        ),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])

    assert (df["outlier_status"] == "outlier_corrected").any()
    assert not (df["outlier_status"] == "trajectory_break").any()
    assert (df.loc[df["frame"].isin([7, 8]), "outlier_status"] == "unchanged").all()


def test_echo_trimming_can_be_disabled(tmp_path):
    input_csv, metadata = _write_input(
        tmp_path, [_pose_row(frame, tx=value) for frame, value in enumerate(_echo_glitch_values())]
    )

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(
            max_correction_gap_sec=0.08,
            velocity_floor=0.01,
            acceleration_floor=0.01,
            jerk_floor=0.01,
            trim_feature_echo=False,
            min_stable_neighbors=1,
        ),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])

    assert (df["outlier_status"] == "trajectory_break").any()


def test_short_hip_spike_is_corrected(tmp_path):
    values = [0.0, 1.0, 2.0, 50.0, 4.0, 5.0, 6.0, 7.0]
    input_csv, metadata = _write_input(
        tmp_path,
        [_pose_row(frame, landmark_name="left_hip", tx=value) for frame, value in enumerate(values)],
    )

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(max_correction_gap_sec=0.2, min_stable_neighbors=1),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])

    assert (df["outlier_status"] == "outlier_corrected").any()
    assert df.loc[df["frame"] == 3, "tx"].iloc[0] != 50.0


def test_interpolated_rows_are_excluded_from_baseline(tmp_path):
    rows = []
    value = 0.0
    for frame in range(20):
        if frame <= 11:
            value = 0.01 * frame
            rows.append(_pose_row(frame, tx=value, quality_flag="interpolated_short_gap"))
        else:
            step = 2.0 if frame == 16 else 0.5
            value += step
            rows.append(_pose_row(frame, tx=value))
    input_csv, metadata = _write_input(tmp_path, rows)

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(min_stable_neighbors=1),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])

    assert not (df["outlier_status"] == "trajectory_break").any()
    assert not (df["outlier_status"] == "outlier_corrected").any()


def _dual_source_rows(values_tx, values_x):
    rows = []
    for frame, (tx, x) in enumerate(zip(values_tx, values_x)):
        world = _pose_row(frame, tx=tx)
        screen = _pose_row(frame)
        screen["source"] = "pose"
        screen["x"] = x
        screen["tx"] = float("nan")
        rows.append(world)
        rows.append(screen)
    return rows


def test_sync_sources_corrects_pose_rows(tmp_path):
    values_tx = [0.0, 1.0, 2.0, 50.0, 4.0, 5.0, 6.0, 7.0]
    values_x = [0.0, 0.01, 0.02, 0.9, 0.04, 0.05, 0.06, 0.07]
    input_csv, metadata = _write_input(tmp_path, _dual_source_rows(values_tx, values_x))

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(max_correction_gap_sec=0.2, min_stable_neighbors=1),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])
    pose_row = df[(df["frame"] == 3) & (df["source"] == "pose")].iloc[0]

    assert pose_row["outlier_status"] == "outlier_corrected"
    assert abs(float(pose_row["x"]) - 0.03) < 1e-6


def test_sync_sources_breaks_pose_rows(tmp_path):
    values_tx = [float(frame) for frame in range(10)] + [50.0, -50.0, 50.0, -50.0] + [float(frame) for frame in range(14, 20)]
    values_x = [frame / 100.0 for frame in range(10)] + [0.5, -0.5, 0.5, -0.5] + [frame / 100.0 for frame in range(14, 20)]
    input_csv, metadata = _write_input(tmp_path, _dual_source_rows(values_tx, values_x))

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(
            max_correction_gap_sec=0.03,
            velocity_threshold_multiplier=1.1,
            acceleration_threshold_multiplier=1.1,
            jerk_threshold_multiplier=1.1,
            min_stable_neighbors=1,
        ),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])
    pose_rows = df[df["source"] == "pose"]

    assert (pose_rows["outlier_status"] == "trajectory_break").any()
    assert (pose_rows["trajectory_connect"] == False).any()  # noqa: E712


def test_sync_sources_can_be_disabled(tmp_path):
    values_tx = [0.0, 1.0, 2.0, 50.0, 4.0, 5.0, 6.0, 7.0]
    values_x = [0.0, 0.01, 0.02, 0.9, 0.04, 0.05, 0.06, 0.07]
    input_csv, metadata = _write_input(tmp_path, _dual_source_rows(values_tx, values_x))

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(max_correction_gap_sec=0.2, min_stable_neighbors=1, sync_sources=False),
    )
    df = pd.read_csv(result["outlier_minimized_csv"])
    pose_row = df[(df["frame"] == 3) & (df["source"] == "pose")].iloc[0]

    assert pose_row["outlier_status"] != "outlier_corrected"
    assert abs(float(pose_row["x"]) - 0.9) < 1e-6


def test_reports_are_written(tmp_path):
    values = [0.0, 1.0, 2.0, 50.0, 4.0, 5.0, 6.0, 7.0]
    input_csv, metadata = _write_input(tmp_path, [_pose_row(frame, tx=value) for frame, value in enumerate(values)])

    result = minimize_pose_outliers(
        input_csv,
        metadata,
        tmp_path / "out",
        OutlierMinimizerOptions(max_correction_gap_sec=0.2, min_stable_neighbors=1),
    )

    assert result["outlier_report"].exists()
    assert result["temporal_spike_report"].exists()
    assert result["trajectory_breaks"].exists()
