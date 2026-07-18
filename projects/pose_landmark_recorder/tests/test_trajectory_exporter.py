import json

import pandas as pd

from dance_pose_recorder.trajectory_exporter import TrajectoryExportOptions, export_trajectory


def _row(
    frame,
    landmark_id,
    landmark_name,
    x=0.5,
    y=0.5,
    z=0.1,
    visible=True,
    connect=True,
):
    return {
        "session_id": "test_session",
        "frame": frame,
        "time_sec": frame / 30.0,
        "landmark_id": landmark_id,
        "landmark_name": landmark_name,
        "source": "pose",
        "x": x,
        "y": y,
        "z": z,
        "visibility": 0.9,
        "presence": 0.9,
        "tx": 0.0,
        "ty": 0.0,
        "tz": 0.0,
        "quality_flag": "measured",
        "trajectory_visible": visible,
        "trajectory_connect": connect,
        "trajectory_alpha": 1.0,
        "trajectory_width": 1.0,
        "trajectory_reason": "stable",
        "outlier_status": "unchanged",
        "outlier_action": "none",
        "outlier_reason": "none",
    }


def _write_fixture(tmp_path, rows):
    input_csv = tmp_path / "outlier_minimized_pose.csv"
    metadata = tmp_path / "metadata.json"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    metadata.write_text(json.dumps({"session_id": "test_session", "fps": 30.0, "frame_count_written": 10}))
    return input_csv, metadata


def test_exporter_writes_points_segments_and_report(tmp_path):
    rows = [
        _row(0, 0, "nose"),
        _row(1, 0, "nose"),
        _row(0, 31, "left_foot_index"),
        _row(1, 31, "left_foot_index"),
    ]
    input_csv, metadata = _write_fixture(tmp_path, rows)

    result = export_trajectory(input_csv, metadata, tmp_path / "export", TrajectoryExportOptions())

    assert result["points_csv"].exists()
    assert result["segments_csv"].exists()
    assert result["report_json"].exists()
    points = pd.read_csv(result["points_csv"])
    segments = pd.read_csv(result["segments_csv"])
    report = json.loads(result["report_json"].read_text())
    assert points.shape[0] == 4
    assert segments.shape[0] == 2
    assert report["coordinate_mode"] == "screen_bottom_origin"


def test_excluded_landmarks_are_not_exported(tmp_path):
    rows = [
        _row(0, 7, "left_ear"),
        _row(0, 19, "left_index"),
        _row(0, 21, "left_thumb"),
        _row(0, 0, "nose"),
    ]
    input_csv, metadata = _write_fixture(tmp_path, rows)

    result = export_trajectory(input_csv, metadata, tmp_path / "export", TrajectoryExportOptions())
    points = pd.read_csv(result["points_csv"])

    assert set(points["landmark_name"]) == {"nose"}


def test_foot_index_is_exported_by_default(tmp_path):
    rows = [_row(0, 31, "left_foot_index"), _row(0, 32, "right_foot_index")]
    input_csv, metadata = _write_fixture(tmp_path, rows)

    result = export_trajectory(input_csv, metadata, tmp_path / "export", TrajectoryExportOptions())
    points = pd.read_csv(result["points_csv"])

    assert {"left_foot_index", "right_foot_index"} <= set(points["landmark_name"])


def test_trajectory_visible_false_is_skipped_by_default(tmp_path):
    rows = [_row(0, 0, "nose", visible=False), _row(1, 0, "nose")]
    input_csv, metadata = _write_fixture(tmp_path, rows)

    result = export_trajectory(input_csv, metadata, tmp_path / "export", TrajectoryExportOptions())
    points = pd.read_csv(result["points_csv"])

    assert points["frame"].tolist() == [1]


def test_trajectory_connect_false_does_not_make_segment(tmp_path):
    rows = [_row(0, 0, "nose", connect=False), _row(1, 0, "nose")]
    input_csv, metadata = _write_fixture(tmp_path, rows)

    result = export_trajectory(input_csv, metadata, tmp_path / "export", TrajectoryExportOptions())
    segments = pd.read_csv(result["segments_csv"])

    assert segments.empty


def test_frame_gap_does_not_make_segment(tmp_path):
    rows = [_row(0, 0, "nose"), _row(2, 0, "nose")]
    input_csv, metadata = _write_fixture(tmp_path, rows)

    result = export_trajectory(input_csv, metadata, tmp_path / "export", TrajectoryExportOptions())
    segments = pd.read_csv(result["segments_csv"])

    assert segments.empty


def test_aspect_ratio_scales_width(tmp_path):
    rows = [_row(0, 0, "nose", x=1.0, y=0.5)]
    input_csv, metadata = _write_fixture(tmp_path, rows)
    metadata.write_text(
        json.dumps(
            {"session_id": "test_session", "fps": 30.0, "frame_count_written": 10, "width": 1920, "height": 1080}
        )
    )

    result = export_trajectory(input_csv, metadata, tmp_path / "export", TrajectoryExportOptions())
    points = pd.read_csv(result["points_csv"])
    report = json.loads(result["report_json"].read_text())
    aspect = 1920 / 1080

    assert abs(report["settings"]["aspect_ratio"] - aspect) < 1e-9
    assert abs(report["settings"]["screen_width_scale"] - 6.0 * aspect) < 1e-9
    assert report["settings"]["screen_width_scale_requested"] == 6.0
    assert abs(points["blender_x"].iloc[0] - 0.5 * 6.0 * aspect) < 1e-6


def test_aspect_ratio_can_be_disabled(tmp_path):
    rows = [_row(0, 0, "nose", x=1.0, y=0.5)]
    input_csv, metadata = _write_fixture(tmp_path, rows)
    metadata.write_text(
        json.dumps(
            {"session_id": "test_session", "fps": 30.0, "frame_count_written": 10, "width": 1920, "height": 1080}
        )
    )

    result = export_trajectory(
        input_csv, metadata, tmp_path / "export", TrajectoryExportOptions(apply_aspect_ratio=False)
    )
    points = pd.read_csv(result["points_csv"])
    report = json.loads(result["report_json"].read_text())

    assert report["settings"]["aspect_ratio"] is None
    assert report["settings"]["screen_width_scale"] == 6.0
    assert abs(points["blender_x"].iloc[0] - 0.5 * 6.0) < 1e-6
