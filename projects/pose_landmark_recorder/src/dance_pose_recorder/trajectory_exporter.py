"""Export outlier-minimized pose rows as Blender-ready trajectory CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dance_pose_recorder.blender_coordinate import (
    to_pose_2d_flat_position,
    to_pose_world_direct_position,
    to_screen_bottom_origin_position,
)
from dance_pose_recorder.landmark_sets import (
    DEFAULT_EXCLUDED_FOR_BLENDER,
    get_landmark_names,
    landmark_group_for_export,
)
from dance_pose_recorder.output_layout import (
    TRAJECTORY_EXPORT_DIR,
    TRAJECTORY_EXPORT_POINTS_CSV,
    TRAJECTORY_EXPORT_REPORT_JSON,
    TRAJECTORY_EXPORT_SEGMENTS_CSV,
    normalize_stage_output_dir,
)


POINT_COLUMNS = [
    "session_id",
    "frame",
    "time_sec",
    "landmark_id",
    "landmark_name",
    "landmark_group",
    "track_id",
    "blender_x",
    "blender_y",
    "blender_z",
    "screen_x",
    "screen_y",
    "screen_z",
    "screen_origin_x",
    "screen_origin_y",
    "trajectory_visible",
    "trajectory_connect",
    "trajectory_alpha",
    "trajectory_width",
    "trajectory_reason",
    "quality_flag",
    "outlier_status",
    "outlier_action",
    "outlier_reason",
    "source",
    "coordinate_mode",
    "depth_mode",
]

SEGMENT_COLUMNS = [
    "session_id",
    "segment_id",
    "track_id",
    "landmark_id",
    "landmark_name",
    "landmark_group",
    "frame_start",
    "frame_end",
    "time_start",
    "time_end",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "trajectory_alpha",
    "trajectory_width",
    "trajectory_reason",
    "quality_flag_start",
    "quality_flag_end",
    "outlier_status_start",
    "outlier_status_end",
    "coordinate_mode",
    "depth_mode",
]

REQUIRED_COLUMNS = {
    "frame",
    "landmark_id",
    "landmark_name",
    "source",
    "x",
    "y",
    "z",
    "quality_flag",
    "trajectory_visible",
    "trajectory_connect",
    "trajectory_alpha",
    "trajectory_width",
    "trajectory_reason",
}


@dataclass(frozen=True)
class TrajectoryExportOptions:
    coordinate_mode: str = "screen_bottom_origin"
    source: str = "pose"
    depth_mode: str = "pose_z"
    landmark_preset: str = "blender_default"
    include_landmarks: list[str] | None = None
    exclude_landmarks: list[str] | None = None
    screen_origin_x: float = 0.5
    screen_origin_y: float = 1.0
    screen_width_scale: float = 6.0
    screen_height_scale: float = 6.0
    depth_scale: float = 1.0
    include_hidden: bool = False
    include_disconnected_points: bool = True
    save_points: bool = True
    save_segments: bool = True
    save_report: bool = True


def export_trajectory(
    input_pose_csv: Path,
    metadata_path: Path,
    output_dir: Path,
    options: TrajectoryExportOptions,
) -> dict:
    """Export Blender trajectory points and frame-to-frame line segments."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    pose = pd.read_csv(input_pose_csv, low_memory=False)
    _validate_required_columns(pose, options)
    output_dir = normalize_stage_output_dir(output_dir, TRAJECTORY_EXPORT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(metadata.get("fps") or 30.0)
    frames_total = int(metadata.get("frame_count_written") or metadata.get("frame_count") or pose["frame"].max() + 1)
    session_id = str(metadata.get("session_id") or _first_non_empty(pose.get("session_id")) or output_dir.name)
    if "time_sec" not in pose.columns:
        pose["time_sec"] = pose["frame"].astype(float) / fps
    for column in ("outlier_status", "outlier_action", "outlier_reason"):
        if column not in pose.columns:
            pose[column] = ""

    landmark_names = get_landmark_names(
        options.landmark_preset,
        include_landmarks=options.include_landmarks,
        exclude_landmarks=options.exclude_landmarks,
    )
    landmark_set = set(landmark_names)
    source_rows = pose[pose["source"].astype(str) == options.source].copy()
    excluded_landmark_rows = int((~source_rows["landmark_name"].isin(landmark_set)).sum())
    selected = source_rows[source_rows["landmark_name"].isin(landmark_set)].copy()

    point_rows: list[dict] = []
    missing_required_column_rows = 0
    hidden_skipped_rows = 0
    disconnected_skipped_rows = 0
    for row in selected.sort_values(["landmark_name", "frame"]).itertuples(index=False):
        row_series = pd.Series(row._asdict())
        visible = _as_bool(row_series.get("trajectory_visible"))
        connect = _as_bool(row_series.get("trajectory_connect"))
        if not visible and not options.include_hidden:
            hidden_skipped_rows += 1
            continue
        if not connect and not options.include_disconnected_points:
            disconnected_skipped_rows += 1
            continue
        try:
            coords = _convert_position(row_series, options)
        except ValueError:
            missing_required_column_rows += 1
            continue
        point_rows.append(_point_row(row_series, coords, session_id, options, visible, connect))

    points = pd.DataFrame(point_rows, columns=POINT_COLUMNS)
    segments = _build_segments(points, options)

    points_path = output_dir / TRAJECTORY_EXPORT_POINTS_CSV
    if options.save_points:
        points.to_csv(points_path, index=False)
    else:
        points_path = None

    segments_path = output_dir / TRAJECTORY_EXPORT_SEGMENTS_CSV
    if options.save_segments:
        segments.to_csv(segments_path, index=False)
    else:
        segments_path = None

    report = _build_report(
        metadata=metadata,
        input_pose_csv=input_pose_csv,
        frames_total=frames_total,
        fps=fps,
        session_id=session_id,
        options=options,
        landmark_names=landmark_names,
        input_rows=len(pose),
        source_rows=len(source_rows),
        exported_point_rows=len(points),
        exported_segment_rows=len(segments),
        hidden_skipped_rows=hidden_skipped_rows,
        excluded_landmark_rows=excluded_landmark_rows,
        missing_required_column_rows=missing_required_column_rows,
        disconnected_skipped_rows=disconnected_skipped_rows,
        points=points,
    )
    report_path = output_dir / TRAJECTORY_EXPORT_REPORT_JSON
    if options.save_report:
        report_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        report_path = None

    return {
        "points_csv": points_path,
        "segments_csv": segments_path,
        "report_json": report_path,
        "report": report,
    }


def _validate_required_columns(df: pd.DataFrame, options: TrajectoryExportOptions) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if options.coordinate_mode == "pose_world_direct":
        for column in ("tx", "ty", "tz"):
            if column not in df.columns:
                missing.append(column)
    if missing:
        raise ValueError(f"Input pose CSV is missing required columns: {', '.join(sorted(set(missing)))}")
    if options.depth_mode == "pose_world_y":
        raise ValueError("depth_mode=pose_world_y is not implemented in trajectory export v1; use pose_z or none")
    if options.coordinate_mode not in {"screen_bottom_origin", "pose_world_direct", "pose_2d_flat"}:
        raise ValueError(f"Unsupported coordinate_mode: {options.coordinate_mode}")
    if options.source not in {"pose", "pose_world"}:
        raise ValueError(f"Unsupported source: {options.source}")


def _convert_position(row: pd.Series, options: TrajectoryExportOptions) -> dict:
    if options.coordinate_mode == "screen_bottom_origin":
        return to_screen_bottom_origin_position(
            row,
            screen_origin_x=options.screen_origin_x,
            screen_origin_y=options.screen_origin_y,
            screen_width_scale=options.screen_width_scale,
            screen_height_scale=options.screen_height_scale,
            depth_scale=options.depth_scale,
            depth_mode=options.depth_mode,
        )
    if options.coordinate_mode == "pose_world_direct":
        return to_pose_world_direct_position(
            row,
            screen_width_scale=options.screen_width_scale,
            screen_height_scale=options.screen_height_scale,
            depth_scale=options.depth_scale,
        )
    return to_pose_2d_flat_position(
        row,
        screen_origin_x=options.screen_origin_x,
        screen_origin_y=options.screen_origin_y,
        screen_width_scale=options.screen_width_scale,
        screen_height_scale=options.screen_height_scale,
    )


def _point_row(
    row: pd.Series,
    coords: dict,
    session_id: str,
    options: TrajectoryExportOptions,
    visible: bool,
    connect: bool,
) -> dict:
    landmark_name = str(row["landmark_name"])
    return {
        "session_id": str(row.get("session_id") or session_id),
        "frame": int(row["frame"]),
        "time_sec": float(row["time_sec"]),
        "landmark_id": int(row["landmark_id"]),
        "landmark_name": landmark_name,
        "landmark_group": landmark_group_for_export(landmark_name),
        "track_id": landmark_name,
        "blender_x": coords["blender_x"],
        "blender_y": coords["blender_y"],
        "blender_z": coords["blender_z"],
        "screen_x": coords["screen_x"],
        "screen_y": coords["screen_y"],
        "screen_z": coords["screen_z"],
        "screen_origin_x": float(options.screen_origin_x),
        "screen_origin_y": float(options.screen_origin_y),
        "trajectory_visible": bool(visible),
        "trajectory_connect": bool(connect),
        "trajectory_alpha": _float_or_default(row.get("trajectory_alpha"), 1.0),
        "trajectory_width": _float_or_default(row.get("trajectory_width"), 1.0),
        "trajectory_reason": str(row.get("trajectory_reason") or ""),
        "quality_flag": str(row.get("quality_flag") or ""),
        "outlier_status": str(row.get("outlier_status") or ""),
        "outlier_action": str(row.get("outlier_action") or ""),
        "outlier_reason": str(row.get("outlier_reason") or ""),
        "source": str(row.get("source") or options.source),
        "coordinate_mode": options.coordinate_mode,
        "depth_mode": options.depth_mode,
    }


def _build_segments(points: pd.DataFrame, options: TrajectoryExportOptions) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    rows: list[dict] = []
    segment_id = 0
    for (_source, track_id), group in points.groupby(["source", "track_id"], sort=False):
        ordered = group.sort_values("frame")
        previous = None
        for row in ordered.itertuples(index=False):
            current = row._asdict()
            if previous is not None and _can_connect(previous, current):
                rows.append(_segment_row(segment_id, previous, current, options))
                segment_id += 1
            previous = current
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def _can_connect(previous: dict, current: dict) -> bool:
    if int(current["frame"]) - int(previous["frame"]) != 1:
        return False
    if not bool(previous["trajectory_visible"]) or not bool(current["trajectory_visible"]):
        return False
    if not bool(previous["trajectory_connect"]) or not bool(current["trajectory_connect"]):
        return False
    for key in ("blender_x", "blender_y", "blender_z"):
        if pd.isna(previous[key]) or pd.isna(current[key]):
            return False
    return True


def _segment_row(segment_id: int, previous: dict, current: dict, options: TrajectoryExportOptions) -> dict:
    return {
        "session_id": current["session_id"],
        "segment_id": int(segment_id),
        "track_id": current["track_id"],
        "landmark_id": int(current["landmark_id"]),
        "landmark_name": current["landmark_name"],
        "landmark_group": current["landmark_group"],
        "frame_start": int(previous["frame"]),
        "frame_end": int(current["frame"]),
        "time_start": float(previous["time_sec"]),
        "time_end": float(current["time_sec"]),
        "x1": float(previous["blender_x"]),
        "y1": float(previous["blender_y"]),
        "z1": float(previous["blender_z"]),
        "x2": float(current["blender_x"]),
        "y2": float(current["blender_y"]),
        "z2": float(current["blender_z"]),
        "trajectory_alpha": float(current["trajectory_alpha"]),
        "trajectory_width": float(current["trajectory_width"]),
        "trajectory_reason": current["trajectory_reason"],
        "quality_flag_start": previous["quality_flag"],
        "quality_flag_end": current["quality_flag"],
        "outlier_status_start": previous["outlier_status"],
        "outlier_status_end": current["outlier_status"],
        "coordinate_mode": options.coordinate_mode,
        "depth_mode": options.depth_mode,
    }


def _build_report(
    metadata: dict,
    input_pose_csv: Path,
    frames_total: int,
    fps: float,
    session_id: str,
    options: TrajectoryExportOptions,
    landmark_names: list[str],
    input_rows: int,
    source_rows: int,
    exported_point_rows: int,
    exported_segment_rows: int,
    hidden_skipped_rows: int,
    excluded_landmark_rows: int,
    missing_required_column_rows: int,
    disconnected_skipped_rows: int,
    points: pd.DataFrame,
) -> dict:
    return {
        "session_id": session_id,
        "input_pose_csv": str(input_pose_csv),
        "frames_total": int(frames_total),
        "fps": float(fps),
        "coordinate_mode": options.coordinate_mode,
        "source": options.source,
        "depth_mode": options.depth_mode,
        "landmark_preset": options.landmark_preset,
        "head_proxy": "nose",
        "excluded_landmarks": sorted(set(DEFAULT_EXCLUDED_FOR_BLENDER) | set(options.exclude_landmarks or [])),
        "included_landmarks": landmark_names,
        "settings": {
            "screen_origin_x": options.screen_origin_x,
            "screen_origin_y": options.screen_origin_y,
            "screen_width_scale": options.screen_width_scale,
            "screen_height_scale": options.screen_height_scale,
            "depth_scale": options.depth_scale,
            "include_hidden": options.include_hidden,
            "include_disconnected_points": options.include_disconnected_points,
        },
        "counts": {
            "input_rows": int(input_rows),
            "source_rows": int(source_rows),
            "exported_point_rows": int(exported_point_rows),
            "exported_segment_rows": int(exported_segment_rows),
            "hidden_skipped_rows": int(hidden_skipped_rows),
            "excluded_landmark_rows": int(excluded_landmark_rows),
            "missing_required_column_rows": int(missing_required_column_rows),
            "disconnected_skipped_rows": int(disconnected_skipped_rows),
        },
        "landmark_summary": _landmark_summary(points),
        "notes": (
            "Trajectory export does not correct pose data. It converts outlier-minimized pose rows into "
            "Blender-ready points and line segments using the screen bottom center as the default origin. "
            "screen_bottom_origin is a simple visualization coordinate system, not camera calibration or "
            "real-world 3D reconstruction."
        ),
    }


def _landmark_summary(points: pd.DataFrame) -> dict:
    if points.empty:
        return {}
    summary = {}
    for landmark_name, group in points.groupby("landmark_name", sort=True):
        summary[str(landmark_name)] = {
            "point_rows": int(len(group)),
            "connectable_rows": int(group["trajectory_connect"].fillna(False).astype(bool).sum()),
            "alpha_mean": float(group["trajectory_alpha"].astype(float).mean()),
        }
    return summary


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _first_non_empty(series: pd.Series | None) -> str | None:
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    value = str(values.iloc[0])
    return value if value else None


def _float_or_default(value: object, default: float) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
