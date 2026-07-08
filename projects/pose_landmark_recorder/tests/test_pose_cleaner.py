import json

import pandas as pd

from dance_pose_recorder.pose_cleaner import CLEANED_COLUMNS, CleaningOptions, clean_pose_session


def _write_fixture(tmp_path, rows, frame_count=5):
    raw_csv = tmp_path / "raw_pose.csv"
    metadata = tmp_path / "metadata.json"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)
    metadata.write_text(
        json.dumps(
            {
                "session_id": "fixture",
                "fps": 10.0,
                "frame_count_written": frame_count,
                "source_frame_count": frame_count,
            }
        ),
        encoding="utf-8",
    )
    return raw_csv, metadata


def _row(frame, landmark_id=0, source="pose_world", tx=0.0, visibility=0.9, presence=0.9):
    return {
        "session_id": "fixture",
        "frame": frame,
        "time_sec": frame / 10.0,
        "landmark_id": landmark_id,
        "landmark_name": str(landmark_id),
        "source": source,
        "x": tx,
        "y": 0.0,
        "z": 0.0,
        "visibility": visibility,
        "presence": presence,
        "tx": tx,
        "ty": 0.0,
        "tz": 0.0,
    }


def _pose_row(frame, landmark_id, x, y=0.0, visibility=0.9, presence=0.9):
    return {
        "session_id": "fixture",
        "frame": frame,
        "time_sec": frame / 10.0,
        "landmark_id": landmark_id,
        "landmark_name": str(landmark_id),
        "source": "pose",
        "x": x,
        "y": y,
        "z": 0.0,
        "visibility": visibility,
        "presence": presence,
        "tx": x,
        "ty": y,
        "tz": 0.0,
    }


def test_visibility_threshold_marks_invalid(tmp_path):
    raw_csv, metadata = _write_fixture(tmp_path, [_row(0, visibility=0.1)], frame_count=1)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            visibility_threshold=0.5,
            presence_threshold=0.5,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    row = cleaned[(cleaned["source"] == "pose_world") & (cleaned["landmark_id"] == 0)].iloc[0]

    assert row["is_valid"] == False
    assert "low_visibility" in row["invalid_reason"]


def test_presence_threshold_marks_invalid(tmp_path):
    raw_csv, metadata = _write_fixture(tmp_path, [_row(0, presence=0.1)], frame_count=1)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            visibility_threshold=0.5,
            presence_threshold=0.5,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    row = cleaned[(cleaned["source"] == "pose_world") & (cleaned["landmark_id"] == 0)].iloc[0]

    assert row["is_valid"] == False
    assert "low_presence" in row["invalid_reason"]


def test_large_jump_marks_invalid(tmp_path):
    rows = [
        _row(0, tx=0.0),
        _row(1, tx=0.1),
        _row(2, tx=0.2),
        _row(3, tx=10.0),
        _row(4, tx=0.4),
        _row(5, tx=0.5),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=6)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            jump_threshold_multiplier=2.0,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    row = cleaned[
        (cleaned["source"] == "pose_world")
        & (cleaned["landmark_id"] == 0)
        & (cleaned["frame"] == 3)
    ].iloc[0]

    assert "jump_outlier" in row["invalid_reason"]
    assert row["is_interpolated"] == True
    assert row["quality_flag"] == "interpolated_outlier_removed"
    assert abs(row["tx"] - 0.3) < 1e-6


def test_recoverable_jump_can_be_left_uninterpolated(tmp_path):
    rows = [
        _row(0, tx=0.0),
        _row(1, tx=0.1),
        _row(2, tx=0.2),
        _row(3, tx=10.0),
        _row(4, tx=0.4),
        _row(5, tx=0.5),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=6)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            jump_threshold_multiplier=2.0,
            no_smoothing=True,
            enable_bone_check=False,
            interpolate_recoverable_outliers=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    row = cleaned[
        (cleaned["source"] == "pose_world")
        & (cleaned["landmark_id"] == 0)
        & (cleaned["frame"] == 3)
    ].iloc[0]

    assert "jump_outlier" in row["invalid_reason"]
    assert row["is_interpolated"] == False
    assert row["quality_flag"] == "unreliable"


def test_outlier_interpolation_respects_outlier_max_gap(tmp_path):
    rows = [
        _row(0, tx=0.0),
        _row(1, tx=10.0, visibility=0.1),
        _row(2, tx=11.0, visibility=0.1),
        _row(3, tx=12.0, visibility=0.1),
        _row(4, tx=13.0, visibility=0.1),
        _row(5, tx=5.0),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=6)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            visibility_threshold=0.5,
            interpolate_outliers=True,
            outlier_max_gap=3,
            max_interpolate_gap=15,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    row = cleaned[
        (cleaned["source"] == "pose_world")
        & (cleaned["landmark_id"] == 0)
        & (cleaned["frame"] == 2)
    ].iloc[0]

    assert row["is_interpolated"] == False
    assert row["quality_flag"] == "unreliable"


def test_missing_interpolation_still_uses_max_interpolate_gap(tmp_path):
    rows = [
        _row(0, tx=0.0),
        _row(5, tx=5.0),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=6)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            outlier_max_gap=3,
            max_interpolate_gap=15,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    row = cleaned[
        (cleaned["source"] == "pose_world")
        & (cleaned["landmark_id"] == 0)
        & (cleaned["frame"] == 2)
    ].iloc[0]

    assert row["is_interpolated"] == True
    assert row["quality_flag"] == "interpolated_short_gap"
    assert abs(row["tx"] - 2.0) < 1e-6


def test_large_step_change_is_not_interpolated_as_spike(tmp_path):
    rows = [
        _row(0, tx=0.0),
        _row(1, tx=0.1),
        _row(2, tx=0.2),
        _row(3, tx=10.0),
        _row(4, tx=10.1),
        _row(5, tx=10.2),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=6)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            jump_threshold_multiplier=2.0,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    row = cleaned[
        (cleaned["source"] == "pose_world")
        & (cleaned["landmark_id"] == 0)
        & (cleaned["frame"] == 3)
    ].iloc[0]

    assert row["is_interpolated"] == False
    assert "jump_outlier" not in str(row["invalid_reason"])
    assert row["quality_flag"] == "measured"


def test_torso_side_lock_corrects_swapped_shoulders(tmp_path):
    rows = [
        _pose_row(0, 11, 0.40),
        _pose_row(0, 12, 0.60),
        _pose_row(1, 11, 0.61),
        _pose_row(1, 12, 0.39),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=2)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    left = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 11) & (cleaned["frame"] == 1)].iloc[0]
    right = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 12) & (cleaned["frame"] == 1)].iloc[0]

    assert abs(left["x"] - 0.39) < 1e-6
    assert abs(right["x"] - 0.61) < 1e-6
    assert left["quality_flag"] == "shoulder_swap_corrected"


def test_pelvis_side_lock_is_disabled_by_default(tmp_path):
    rows = [
        _pose_row(0, 23, 0.40),
        _pose_row(0, 24, 0.60),
        _pose_row(0, 25, 0.42),
        _pose_row(0, 26, 0.58),
        _pose_row(1, 23, 0.61),
        _pose_row(1, 24, 0.39),
        _pose_row(1, 25, 0.42),
        _pose_row(1, 26, 0.58),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=2)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    left_hip = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 23) & (cleaned["frame"] == 1)].iloc[0]
    right_hip = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 24) & (cleaned["frame"] == 1)].iloc[0]

    assert abs(left_hip["x"] - 0.61) < 1e-6
    assert abs(right_hip["x"] - 0.39) < 1e-6
    assert left_hip["quality_flag"] == "measured"


def test_shoulder_side_lock_rejects_swap_that_crosses_torso(tmp_path):
    rows = [
        _pose_row(0, 11, 0.60),
        _pose_row(0, 12, 0.40),
        _pose_row(0, 23, 0.40),
        _pose_row(0, 24, 0.60),
        _pose_row(1, 11, 0.40),
        _pose_row(1, 12, 0.60),
        _pose_row(1, 23, 0.40),
        _pose_row(1, 24, 0.60),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=2)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    left = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 11) & (cleaned["frame"] == 1)].iloc[0]
    right = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 12) & (cleaned["frame"] == 1)].iloc[0]

    assert abs(left["x"] - 0.40) < 1e-6
    assert abs(right["x"] - 0.60) < 1e-6
    assert left["quality_flag"] == "measured"


def test_low_visibility_elbow_is_estimated_from_shoulder_local_offset(tmp_path):
    rows = [
        _pose_row(0, 11, 0.0),
        _pose_row(0, 13, 1.0),
        _pose_row(1, 11, 0.0),
        _pose_row(1, 13, 99.0, visibility=0.1),
        _pose_row(2, 11, 0.0),
        _pose_row(2, 13, 1.2),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=3)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            visibility_threshold=0.5,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    elbow = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 13) & (cleaned["frame"] == 1)].iloc[0]

    assert elbow["quality_flag"] == "estimated_occluded_arm"
    assert elbow["is_interpolated"] == True
    assert abs(elbow["x"] - 1.1) < 1e-6


def test_stable_low_visibility_leg_measurement_is_kept(tmp_path):
    rows = [
        _pose_row(0, 23, 0.0),
        _pose_row(0, 25, 1.0),
        _pose_row(0, 27, 2.0),
        _pose_row(1, 23, 0.0),
        _pose_row(1, 25, 1.1, visibility=0.2),
        _pose_row(1, 27, 2.1),
        _pose_row(2, 23, 0.0),
        _pose_row(2, 25, 1.2),
        _pose_row(2, 27, 2.2),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=3)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            visibility_threshold=0.5,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    knee = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 25) & (cleaned["frame"] == 1)].iloc[0]

    assert knee["is_valid"] == True
    assert knee["is_interpolated"] == False
    assert knee["quality_flag"] == "low_visibility_leg_kept"
    assert "low_visibility" in knee["invalid_reason"]


def test_very_low_visibility_leg_measurement_is_not_kept(tmp_path):
    rows = [
        _pose_row(0, 23, 0.0),
        _pose_row(0, 25, 1.0),
        _pose_row(0, 27, 2.0),
        _pose_row(1, 23, 0.0),
        _pose_row(1, 25, 1.1, visibility=0.05),
        _pose_row(1, 27, 2.1),
        _pose_row(2, 23, 0.0),
        _pose_row(2, 25, 1.2),
        _pose_row(2, 27, 2.2),
    ]
    raw_csv, metadata = _write_fixture(tmp_path, rows, frame_count=3)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            visibility_threshold=0.5,
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)
    knee = cleaned[(cleaned["source"] == "pose") & (cleaned["landmark_id"] == 25) & (cleaned["frame"] == 1)].iloc[0]

    assert knee["is_valid"] == False
    assert knee["is_interpolated"] == False
    assert knee["quality_flag"] == "unreliable"


def test_cleaned_csv_has_required_columns(tmp_path):
    raw_csv, metadata = _write_fixture(tmp_path, [_row(0)], frame_count=1)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    cleaned = pd.read_csv(result.cleaned_csv)

    for column in CLEANED_COLUMNS:
        assert column in cleaned.columns


def test_frame_status_includes_full_frame_range(tmp_path):
    raw_csv, metadata = _write_fixture(tmp_path, [_row(1)], frame_count=3)
    result = clean_pose_session(
        CleaningOptions(
            input_csv=raw_csv,
            metadata=metadata,
            output=tmp_path / "cleaned",
            no_smoothing=True,
            enable_bone_check=False,
        )
    )
    frame_status = pd.read_csv(result.frame_status_csv)

    assert frame_status["frame"].tolist() == [0, 1, 2]
