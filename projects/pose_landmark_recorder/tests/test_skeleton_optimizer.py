import json

import pandas as pd

from dance_pose_recorder.skeleton_optimizer import (
    OPTIMIZER_COLUMNS,
    SkeletonOptimizationOptions,
    optimize_pose_skeleton,
)


BASE_COLUMNS = [
    "session_id",
    "frame",
    "time_sec",
    "landmark_id",
    "landmark_name",
    "source",
    "x",
    "y",
    "z",
    "visibility",
    "presence",
    "tx",
    "ty",
    "tz",
    "is_valid",
    "is_interpolated",
    "is_smoothed",
    "quality_flag",
    "invalid_reason",
    "interpolation_method",
    "gap_length",
    "source_frame_prev",
    "source_frame_next",
    "refine_status",
    "refine_source",
    "refine_score_before",
    "refine_score_after",
    "refine_score_delta",
    "refine_segment_id",
    "refine_reason",
]


def _landmark_row(frame, landmark_id, name, source, x, quality_flag="measured", refine_status="unchanged"):
    return {
        "session_id": "session_test",
        "frame": frame,
        "time_sec": frame / 30.0,
        "landmark_id": landmark_id,
        "landmark_name": name,
        "source": source,
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "visibility": 0.9,
        "presence": 0.9,
        "tx": x if source == "pose_world" else pd.NA,
        "ty": 0.0 if source == "pose_world" else pd.NA,
        "tz": 0.0 if source == "pose_world" else pd.NA,
        "is_valid": quality_flag != "missing_long_gap",
        "is_interpolated": False,
        "is_smoothed": False,
        "quality_flag": quality_flag,
        "invalid_reason": "",
        "interpolation_method": "",
        "gap_length": pd.NA,
        "source_frame_prev": pd.NA,
        "source_frame_next": pd.NA,
        "refine_status": refine_status,
        "refine_source": "cleaned",
        "refine_score_before": pd.NA,
        "refine_score_after": pd.NA,
        "refine_score_delta": pd.NA,
        "refine_segment_id": pd.NA,
        "refine_reason": "unchanged_not_target",
    }


def _write_metadata(tmp_path, frames_total):
    metadata = {
        "session_id": "session_test",
        "fps": 30.0,
        "frame_count_written": frames_total,
        "source_path": "examples/input/test.mp4",
        "origin_policy": "raw",
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _write_refine_report(tmp_path, start_frame, end_frame):
    path = tmp_path / "refine_report.json"
    path.write_text(
        json.dumps(
            {
                "session_id": "session_test",
                "segments": [
                    {
                        "segment_id": 1,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "review_only": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_refined_csv(tmp_path, rows):
    path = tmp_path / "refined_pose.csv"
    pd.DataFrame(rows, columns=BASE_COLUMNS).to_csv(path, index=False)
    return path


def _run_optimizer(tmp_path, rows, frames_total, refine_report=None):
    refined_csv = _write_refined_csv(tmp_path, rows)
    metadata = _write_metadata(tmp_path, frames_total)
    output = tmp_path / "optimized"
    result = optimize_pose_skeleton(
        input_refined_csv=refined_csv,
        metadata_path=metadata,
        output_dir=output,
        constraints_path=None,
        options=SkeletonOptimizationOptions(save_jsonl=True, save_reports=True),
        refine_report_path=refine_report,
    )
    return pd.read_csv(result.optimized_csv), result


def test_short_temporal_jump_is_optimized_constrained(tmp_path):
    rows = []
    for frame in range(11):
        wrist_x = 100.0 if frame == 5 else float(frame) + 2.0
        for source in ["pose", "pose_world"]:
            rows.extend(
                [
                    _landmark_row(frame, 11, "left_shoulder", source, float(frame)),
                    _landmark_row(frame, 13, "left_elbow", source, float(frame) + 1.0),
                    _landmark_row(frame, 15, "left_wrist", source, wrist_x),
                ]
            )

    optimized, result = _run_optimizer(tmp_path, rows, frames_total=11)
    wrist = optimized[
        (optimized["source"] == "pose_world")
        & (optimized["landmark_name"] == "left_wrist")
        & (optimized["frame"].isin([5, 6]))
    ].sort_values("frame")

    assert set(wrist["optimizer_status"]) == {"optimized_constrained"}
    assert set(wrist["quality_flag"]) == {"optimized_constrained"}
    assert wrist["tx"].round(6).tolist() == [7.0, 8.0]
    assert result.optimization_report.exists()
    assert result.optimized_jsonl.exists()


def test_long_review_only_run_is_not_corrected(tmp_path):
    rows = []
    for frame in range(10):
        quality = "unreliable" if 4 <= frame <= 8 else "measured"
        refine_status = "refined_rejected" if 4 <= frame <= 8 else "unchanged"
        for source in ["pose", "pose_world"]:
            rows.extend(
                [
                    _landmark_row(frame, 11, "left_shoulder", source, float(frame)),
                    _landmark_row(frame, 13, "left_elbow", source, float(frame) + 1.0),
                    _landmark_row(frame, 15, "left_wrist", source, float(frame) + 2.0, quality, refine_status),
                ]
            )
    refine_report = _write_refine_report(tmp_path, 4, 8)

    optimized, _ = _run_optimizer(tmp_path, rows, frames_total=10, refine_report=refine_report)
    wrist_review = optimized[
        (optimized["source"] == "pose_world")
        & (optimized["landmark_name"] == "left_wrist")
        & (optimized["frame"].between(4, 8))
    ]

    assert set(wrist_review["optimizer_status"]) == {"review_only"}
    assert "optimized_constrained" not in set(wrist_review["quality_flag"])


def test_missing_long_gap_is_not_auto_corrected(tmp_path):
    rows = []
    for frame in range(4):
        quality = "missing_long_gap" if frame == 2 else "measured"
        for source in ["pose", "pose_world"]:
            rows.append(_landmark_row(frame, 15, "left_wrist", source, float(frame), quality))

    optimized, _ = _run_optimizer(tmp_path, rows, frames_total=4)
    missing = optimized[
        (optimized["source"] == "pose_world")
        & (optimized["landmark_name"] == "left_wrist")
        & (optimized["frame"] == 2)
    ].iloc[0]

    assert missing["optimizer_status"] == "optimization_unreliable"
    assert missing["quality_flag"] == "optimization_unreliable"


def test_existing_columns_are_preserved_and_optimizer_columns_are_added(tmp_path):
    rows = []
    for frame in range(3):
        rows.append(_landmark_row(frame, 15, "left_wrist", "pose_world", float(frame)))

    optimized, _ = _run_optimizer(tmp_path, rows, frames_total=3)

    for column in BASE_COLUMNS:
        assert column in optimized.columns
    for column in OPTIMIZER_COLUMNS:
        assert column in optimized.columns
