"""Regression contracts for trajectory export CSV integrity.

Automates the manual checks from the Codex v2 report: unique point keys with
no null cells, segment endpoints exactly matching point rows (raw and smooth),
raw columns unchanged by smoothing, and chain starts keeping smooth == raw.
"""

import json

import pandas as pd

from dance_pose_recorder.trajectory_exporter import TrajectoryExportOptions, export_trajectory

POINT_KEY = ["frame", "track_id", "landmark_name"]
RAW_COORDS = ["blender_x", "blender_y", "blender_z"]


def _row(frame, landmark_id, landmark_name, x=0.5, y=0.5, z=0.1, connect=True):
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
        "trajectory_visible": True,
        "trajectory_connect": connect,
        "trajectory_alpha": 1.0,
        "trajectory_width": 1.0,
        "trajectory_reason": "stable",
        "outlier_status": "unchanged",
        "outlier_action": "none",
        "outlier_reason": "none",
    }


def _fixture_rows():
    rows = []
    for frame in range(24):
        x = 0.4 + 0.01 * frame + (0.004 if frame % 2 else -0.004)
        connect = frame != 10  # one break splits the nose track into two chains
        rows.append(_row(frame, 0, "nose", x=x, connect=connect))
        rows.append(_row(frame, 31, "left_foot_index", x=0.6, y=0.8 + 0.002 * frame))
    return rows


def _export(tmp_path, name, **options):
    input_csv = tmp_path / f"{name}_pose.csv"
    metadata = tmp_path / f"{name}_meta.json"
    pd.DataFrame(_fixture_rows()).to_csv(input_csv, index=False)
    metadata.write_text(json.dumps({"session_id": "test_session", "fps": 30.0, "frame_count_written": 24}))
    return export_trajectory(input_csv, metadata, tmp_path / name, TrajectoryExportOptions(**options))


def test_point_keys_are_unique_and_cells_non_null(tmp_path):
    result = _export(tmp_path, "base")
    points = pd.read_csv(result["points_csv"])

    assert not points.duplicated(POINT_KEY).any()
    assert points.notna().all().all()


def test_segment_endpoints_match_point_rows_raw_and_smooth(tmp_path):
    result = _export(tmp_path, "base")
    points = pd.read_csv(result["points_csv"]).set_index(POINT_KEY)
    segments = pd.read_csv(result["segments_csv"])

    assert not segments.duplicated(["track_id", "frame_start", "frame_end"]).any()
    assert segments.notna().all().all()
    for row in segments.itertuples(index=False):
        start = points.loc[(row.frame_start, row.track_id, row.landmark_name)]
        end = points.loc[(row.frame_end, row.track_id, row.landmark_name)]
        assert (row.x1, row.y1, row.z1) == tuple(start[RAW_COORDS])
        assert (row.x2, row.y2, row.z2) == tuple(end[RAW_COORDS])
        assert (row.x1_smooth, row.y1_smooth, row.z1_smooth) == tuple(start[f"{c}_smooth"] for c in RAW_COORDS)
        assert (row.x2_smooth, row.y2_smooth, row.z2_smooth) == tuple(end[f"{c}_smooth"] for c in RAW_COORDS)


def test_smoothing_never_touches_raw_columns(tmp_path):
    smoothed = pd.read_csv(_export(tmp_path, "on")["points_csv"])
    plain = pd.read_csv(_export(tmp_path, "off", smooth_trajectory=False)["points_csv"])

    for column in RAW_COORDS + ["trajectory_alpha", "trajectory_width", "trajectory_visible", "trajectory_connect"]:
        assert (smoothed[column] == plain[column]).all()


def test_every_chain_start_keeps_smooth_equal_to_raw(tmp_path):
    result = _export(tmp_path, "base")
    points = pd.read_csv(result["points_csv"])

    for _track, group in points.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        prev_frame = None
        prev_connect = False
        for row in group.itertuples(index=False):
            chained = prev_frame is not None and row.frame - prev_frame == 1 and prev_connect and row.trajectory_connect
            if not chained:
                assert row.blender_x_smooth == row.blender_x
                assert row.blender_y_smooth == row.blender_y
                assert row.blender_z_smooth == row.blender_z
            prev_frame = row.frame
            prev_connect = row.trajectory_connect
