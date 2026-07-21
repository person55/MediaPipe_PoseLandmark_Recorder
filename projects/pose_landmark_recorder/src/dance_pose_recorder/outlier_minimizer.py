"""Visualization-oriented pose outlier minimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dance_pose_recorder.interpolation import contiguous_ranges
from dance_pose_recorder.outlier_report import write_frame_jsonl, write_json_report
from dance_pose_recorder.output_layout import (
    OUTLIER_MINIMIZED_DIR,
    OUTLIER_MINIMIZED_POSE_CSV,
    OUTLIER_MINIMIZED_POSE_JSONL,
    OUTLIER_MINIMIZED_REPORT_JSON,
    OUTLIER_MINIMIZED_TEMPORAL_SPIKE_REPORT_CSV,
    OUTLIER_MINIMIZED_TRAJECTORY_BREAKS_CSV,
    normalize_stage_output_dir,
)
from dance_pose_recorder.quality_flags import (
    OUTLIER_PROTECTED_FLAGS,
    RELIABLE_TRAJECTORY_FLAGS,
)
from dance_pose_recorder.temporal_features import compute_temporal_features
from dance_pose_recorder.trajectory_policy import (
    default_trajectory_policy,
    is_correctable_landmark,
    landmark_group,
)


RELIABLE_FLAGS = RELIABLE_TRAJECTORY_FLAGS

PROTECTED_FLAGS = OUTLIER_PROTECTED_FLAGS

UNAVAILABLE_STATUSES = {
    "crop_unavailable",
    "refined_unavailable",
}

# Interpolated rows are artificially smooth; keeping them in the baseline
# deflates the typical-motion scale and inflates every ratio.
BASELINE_EXCLUDED_FLAGS = {
    "interpolated_short_gap",
}

# Converts MAD to a robust standard deviation estimate for normal-ish data.
MAD_SIGMA_SCALE = 1.4826

SOURCE_POSITION_FIELDS: dict[str, tuple[str, str, str]] = {
    "pose": ("x", "y", "z"),
    "pose_world": ("tx", "ty", "tz"),
}

@dataclass(frozen=True)
class OutlierMinimizerOptions:
    source: str = "pose_world"
    position_fields: tuple[str, str, str] = ("tx", "ty", "tz")
    max_correction_gap_sec: float = 0.12
    max_break_gap_sec: float = 0.20
    velocity_threshold_multiplier: float = 6.0
    acceleration_threshold_multiplier: float = 6.0
    jerk_threshold_multiplier: float = 8.0
    # Floors are physical per-second values converted to per-frame units by the
    # session fps, so the effective spike thresholds stay fps-invariant. The
    # defaults are the 23.976fps-session equivalents of the Loop 2 per-frame
    # floors (0.02/0.02/0.03 m per frame).
    velocity_floor_m_per_s: float = 0.48
    acceleration_floor_m_per_s2: float = 11.5
    jerk_floor_m_per_s3: float = 414.0
    trim_feature_echo: bool = True
    min_stable_neighbors: int = 2
    landmark_policy: str = "visualization"
    preserve_quality_flags: bool = True
    sync_sources: bool = True
    save_csv: bool = True
    save_jsonl: bool = False
    save_report: bool = True
    save_trajectory_breaks: bool = True


@dataclass(frozen=True)
class OutlierMinimizationResult:
    outlier_minimized_csv: Path | None
    outlier_minimized_jsonl: Path | None
    outlier_report: Path
    temporal_spike_report: Path
    trajectory_breaks: Path


def minimize_pose_outliers(
    input_pose_csv: Path,
    metadata_path: Path,
    output_dir: Path,
    options: OutlierMinimizerOptions,
    crop_refine_report_path: Path | None = None,
    quality_report_path: Path | None = None,
) -> dict:
    """Minimize short temporal spikes and write trajectory display policy columns."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    df = pd.read_csv(input_pose_csv, low_memory=False)
    output_dir = normalize_stage_output_dir(output_dir, OUTLIER_MINIMIZED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(metadata.get("fps") or 30.0)
    frames_total = int(metadata.get("frame_count_written") or metadata.get("frame_count") or df["frame"].max() + 1)
    max_correction_gap_frames = max(1, int(round(options.max_correction_gap_sec * fps)))

    minimized = df.copy()
    _initialize_columns(minimized)
    features = compute_temporal_features(
        minimized,
        source=options.source,
        position_fields=options.position_fields,
        protected_flags=PROTECTED_FLAGS,
    )
    for column in ("velocity", "acceleration", "jerk"):
        minimized[column] = features[column]

    frame_floors = _frame_floors(options, fps)
    scales = _feature_scales(minimized, options, frame_floors)
    _apply_ratios(minimized, scales)
    spike_mask = _spike_mask(minimized, options)

    spike_rows: list[dict] = []
    break_rows: list[dict] = []
    trajectory_segment_id = 1

    mirror_sources: list[str] = []
    mirror_groups: dict[tuple[str, str], pd.DataFrame] = {}
    if options.sync_sources and options.source != "both":
        mirror_sources = sorted(set(SOURCE_POSITION_FIELDS) - _target_sources(options.source))
        mirror_rows = minimized[minimized["source"].astype(str).isin(mirror_sources)]
        for key, mirror_group in mirror_rows.groupby(["source", "landmark_name"], sort=False):
            mirror_groups[(str(key[0]), str(key[1]))] = mirror_group.sort_values("frame")

    for (source, landmark_name), group in minimized[minimized["source"].astype(str).isin(_target_sources(options.source))].groupby(
        ["source", "landmark_name"], sort=False
    ):
        group = group.sort_values("frame")
        group_spikes = group[spike_mask.loc[group.index]]
        spike_frames = group_spikes["frame"].tolist()
        if options.trim_feature_echo:
            spike_frames = _trim_echo_frames(group_spikes, options)
        spike_frame_set = set(spike_frames)
        for segment in contiguous_ranges(spike_frames):
            segment_indices = group[group["frame"].between(segment.start_frame, segment.end_frame)].index.tolist()
            if not segment_indices:
                continue
            spike_type = _spike_type(minimized.loc[segment_indices], options)
            protected = _segment_has_protected(minimized.loc[segment_indices])
            corrected = False
            reason = spike_type
            if (
                segment.length <= max_correction_gap_frames
                and not protected
                and is_correctable_landmark(str(landmark_name))
                and _has_stable_neighbors(group, segment.start_frame, segment.end_frame, options, spike_frame_set)
            ):
                corrected = _interpolate_segment(
                    minimized, group, segment_indices, options, fields=options.position_fields, spike_frames=spike_frame_set
                )
                if corrected:
                    _mark_corrected(minimized, segment_indices, reason)
            break_id_used = False
            if not corrected:
                break_reason = _break_reason(minimized.loc[segment_indices], spike_type, str(landmark_name))
                _mark_break(minimized, segment_indices, break_reason, trajectory_segment_id)
                break_rows.append(
                    _trajectory_break_row(
                        trajectory_segment_id,
                        str(source),
                        str(landmark_name),
                        segment.start_frame,
                        segment.end_frame,
                        segment.length,
                        fps,
                        break_reason,
                        minimized.loc[segment_indices],
                    )
                )
                break_id_used = True

            for mirror_source in mirror_sources:
                mirror_group = mirror_groups.get((mirror_source, str(landmark_name)))
                if mirror_group is None:
                    continue
                mirror_indices = mirror_group[
                    mirror_group["frame"].between(segment.start_frame, segment.end_frame)
                ].index.tolist()
                if not mirror_indices:
                    continue
                mirror_fields = SOURCE_POSITION_FIELDS.get(mirror_source)
                if corrected:
                    mirror_corrected = mirror_fields is not None and _interpolate_segment(
                        minimized, mirror_group, mirror_indices, options, fields=mirror_fields, spike_frames=spike_frame_set
                    )
                    if mirror_corrected:
                        _mark_corrected(minimized, mirror_indices, reason)
                        continue
                mirror_break_reason = _break_reason(minimized.loc[mirror_indices], spike_type, str(landmark_name))
                _mark_break(minimized, mirror_indices, mirror_break_reason, trajectory_segment_id)
                break_rows.append(
                    _trajectory_break_row(
                        trajectory_segment_id,
                        mirror_source,
                        str(landmark_name),
                        segment.start_frame,
                        segment.end_frame,
                        segment.length,
                        fps,
                        mirror_break_reason,
                        minimized.loc[mirror_indices],
                    )
                )
                break_id_used = True

            if break_id_used:
                trajectory_segment_id += 1

            spike_rows.append(
                _spike_report_row(
                    len(spike_rows) + 1,
                    str(source),
                    str(landmark_name),
                    segment.start_frame,
                    segment.end_frame,
                    segment.length,
                    fps,
                    spike_type,
                    minimized.loc[segment_indices],
                    corrected,
                )
            )

    _apply_quality_trajectory_policy(minimized, start_segment_id=trajectory_segment_id)
    _apply_outlier_scores(minimized)

    outlier_minimized_csv = output_dir / OUTLIER_MINIMIZED_POSE_CSV
    if options.save_csv:
        minimized.to_csv(outlier_minimized_csv, index=False)
    else:
        outlier_minimized_csv = None

    temporal_spike_report = output_dir / OUTLIER_MINIMIZED_TEMPORAL_SPIKE_REPORT_CSV
    pd.DataFrame(spike_rows, columns=_spike_report_columns()).to_csv(temporal_spike_report, index=False)

    trajectory_breaks = output_dir / OUTLIER_MINIMIZED_TRAJECTORY_BREAKS_CSV
    pd.DataFrame(break_rows, columns=_trajectory_break_columns()).to_csv(trajectory_breaks, index=False)

    outlier_minimized_jsonl = None
    if options.save_jsonl:
        outlier_minimized_jsonl = output_dir / OUTLIER_MINIMIZED_POSE_JSONL
        write_frame_jsonl(minimized, outlier_minimized_jsonl, session_id=str(metadata.get("session_id") or ""))

    report = _build_report(
        minimized=minimized,
        input_pose_csv=input_pose_csv,
        metadata=metadata,
        fps=fps,
        frames_total=frames_total,
        options=options,
        max_correction_gap_frames=max_correction_gap_frames,
        frame_floors=frame_floors,
        spike_rows=spike_rows,
        break_rows=break_rows,
        crop_refine_report_path=crop_refine_report_path,
        quality_report_path=quality_report_path,
    )
    outlier_report = output_dir / OUTLIER_MINIMIZED_REPORT_JSON
    write_json_report(report, outlier_report)

    return {
        "outlier_minimized_csv": outlier_minimized_csv,
        "outlier_minimized_jsonl": outlier_minimized_jsonl,
        "outlier_report": outlier_report,
        "temporal_spike_report": temporal_spike_report,
        "trajectory_breaks": trajectory_breaks,
        "report": report,
    }


def _initialize_columns(df: pd.DataFrame) -> None:
    df["outlier_status"] = "unchanged"
    df["outlier_action"] = "none"
    df["outlier_reason"] = "none"
    df["outlier_score"] = np.nan
    df["velocity"] = np.nan
    df["acceleration"] = np.nan
    df["jerk"] = np.nan
    df["velocity_ratio"] = np.nan
    df["acceleration_ratio"] = np.nan
    df["jerk_ratio"] = np.nan
    df["trajectory_visible"] = True
    df["trajectory_connect"] = True
    df["trajectory_alpha"] = 1.0
    df["trajectory_width"] = 1.0
    df["trajectory_reason"] = "stable"
    df["trajectory_segment_id"] = np.nan


def _target_sources(source: str) -> set[str]:
    if source == "both":
        return {"pose", "pose_world"}
    return {source}


def _frame_floors(options: OutlierMinimizerOptions, fps: float) -> dict[str, float]:
    """Convert per-second floors to the per-frame units the features use.

    Velocity is a first difference (1/fps), acceleration a second difference
    (1/fps^2), jerk a third difference (1/fps^3).
    """

    return {
        "velocity": options.velocity_floor_m_per_s / fps,
        "acceleration": options.acceleration_floor_m_per_s2 / (fps**2),
        "jerk": options.jerk_floor_m_per_s3 / (fps**3),
    }


def _feature_scales(
    df: pd.DataFrame, options: OutlierMinimizerOptions, floors: dict[str, float]
) -> dict[tuple[str, str], dict[str, float]]:
    scales: dict[tuple[str, str], dict[str, float]] = {}
    target = df[df["source"].astype(str).isin(_target_sources(options.source))]
    reliable = target[target.apply(_is_reliable_row, axis=1)]
    baseline = reliable[~reliable["quality_flag"].astype(str).isin(BASELINE_EXCLUDED_FLAGS)]
    for key, group in baseline.groupby(["source", "landmark_name"], sort=False):
        scales[(str(key[0]), str(key[1]))] = {
            feature: _robust_scale(group[feature], floors[feature]) for feature in ("velocity", "acceleration", "jerk")
        }
    return scales


def _apply_ratios(df: pd.DataFrame, scales: dict[tuple[str, str], dict[str, float]]) -> None:
    for index, row in df.iterrows():
        item = scales.get((str(row["source"]), str(row["landmark_name"])))
        if not item:
            continue
        for feature in ("velocity", "acceleration", "jerk"):
            scale = item.get(feature)
            value = row.get(feature)
            if scale is None or pd.isna(scale) or scale <= 0 or pd.isna(value):
                continue
            df.at[index, f"{feature}_ratio"] = float(value) / float(scale)


def _spike_mask(df: pd.DataFrame, options: OutlierMinimizerOptions) -> pd.Series:
    return (
        (df["velocity_ratio"] >= options.velocity_threshold_multiplier)
        | (df["acceleration_ratio"] >= options.acceleration_threshold_multiplier)
        | (df["jerk_ratio"] >= options.jerk_threshold_multiplier)
    ).fillna(False)


def _is_reliable_row(row: pd.Series) -> bool:
    if str(row.get("quality_flag", "")) not in RELIABLE_FLAGS:
        return False
    if str(row.get("crop_refine_status", "")) in UNAVAILABLE_STATUSES:
        return False
    if str(row.get("refine_status", "")) in UNAVAILABLE_STATUSES:
        return False
    visibility = row.get("visibility")
    presence = row.get("presence")
    if pd.notna(visibility) and float(visibility) < 0.05:
        return False
    if pd.notna(presence) and float(presence) < 0.05:
        return False
    return True


def _robust_scale(series: pd.Series, floor: float) -> float:
    values = series.dropna().astype(float)
    values = values[values > 0]
    if values.empty:
        return floor if floor > 0 else np.nan
    median = float(values.median())
    mad = float((values - median).abs().median())
    return max(median + MAD_SIGMA_SCALE * mad, floor)


def _trim_echo_frames(spike_rows: pd.DataFrame, options: OutlierMinimizerOptions) -> list[int]:
    """Drop acceleration/jerk-only echo frames from runs that contain a velocity spike.

    A single bad position spikes velocity on two frames but echoes through the
    second and third differences for up to two more frames, stretching the run
    past the correction limit. Runs with no velocity spike at all are kept
    whole so pure direction-change glitches remain detectable.
    """

    kept: list[int] = []
    for segment in contiguous_ranges(spike_rows["frame"].tolist()):
        run = spike_rows[spike_rows["frame"].between(segment.start_frame, segment.end_frame)]
        velocity_frames = run[
            run["velocity_ratio"].fillna(0.0) >= options.velocity_threshold_multiplier
        ]["frame"].tolist()
        kept.extend(velocity_frames if velocity_frames else run["frame"].tolist())
    return kept


def _segment_has_protected(segment_df: pd.DataFrame) -> bool:
    flags = set(segment_df["quality_flag"].astype(str).tolist())
    if flags & PROTECTED_FLAGS:
        return True
    for column in ("crop_refine_status", "refine_status"):
        if column in segment_df.columns and set(segment_df[column].astype(str).tolist()) & UNAVAILABLE_STATUSES:
            return True
    return False


def _has_stable_neighbors(
    group: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    options: OutlierMinimizerOptions,
    spike_frames: set[int],
) -> bool:
    before = group[group["frame"] < start_frame].tail(options.min_stable_neighbors)
    after = group[group["frame"] > end_frame].head(options.min_stable_neighbors)
    return len(before[before.apply(lambda row: _is_stable_neighbor(row, spike_frames), axis=1)]) >= options.min_stable_neighbors and len(
        after[after.apply(lambda row: _is_stable_neighbor(row, spike_frames), axis=1)]
    ) >= options.min_stable_neighbors


def _is_stable_neighbor(row: pd.Series, spike_frames: set[int]) -> bool:
    # Trimmed echo frames carry inflated acceleration/jerk ratios but their
    # positions are sound, so anchor eligibility keys off the final spike set
    # rather than the raw ratios.
    return _is_reliable_row(row) and int(row["frame"]) not in spike_frames


def _interpolate_segment(
    df: pd.DataFrame,
    group: pd.DataFrame,
    segment_indices: list[int],
    options: OutlierMinimizerOptions,
    fields: tuple[str, ...] | None = None,
    spike_frames: set[int] | None = None,
) -> bool:
    fields = fields if fields is not None else options.position_fields
    spike_frames = spike_frames if spike_frames is not None else set()
    start_frame = int(df.loc[segment_indices, "frame"].min())
    end_frame = int(df.loc[segment_indices, "frame"].max())
    before = group[(group["frame"] < start_frame) & group.apply(lambda row: _is_stable_neighbor(row, spike_frames), axis=1)].tail(1)
    after = group[(group["frame"] > end_frame) & group.apply(lambda row: _is_stable_neighbor(row, spike_frames), axis=1)].head(1)
    if before.empty or after.empty:
        return False
    prev = before.iloc[0]
    next_row = after.iloc[0]
    prev_frame = int(prev["frame"])
    next_frame = int(next_row["frame"])
    span = next_frame - prev_frame
    if span <= 0:
        return False
    for field in fields:
        if pd.isna(prev.get(field)) or pd.isna(next_row.get(field)):
            return False
    for index in segment_indices:
        frame = int(df.at[index, "frame"])
        ratio = (frame - prev_frame) / span
        for field in fields:
            df.at[index, field] = float(prev[field]) + (float(next_row[field]) - float(prev[field])) * ratio
    return True


def _mark_corrected(df: pd.DataFrame, indices: list[int], reason: str) -> None:
    df.loc[indices, "outlier_status"] = "outlier_corrected"
    df.loc[indices, "outlier_action"] = "interpolate_short_spike"
    df.loc[indices, "outlier_reason"] = reason
    df.loc[indices, "trajectory_visible"] = True
    df.loc[indices, "trajectory_connect"] = True
    df.loc[indices, "trajectory_alpha"] = 0.9
    df.loc[indices, "trajectory_width"] = 1.0
    df.loc[indices, "trajectory_reason"] = "short_spike_corrected"


def _mark_break(df: pd.DataFrame, indices: list[int], reason: str, segment_id: int) -> None:
    df.loc[indices, "outlier_status"] = "trajectory_break"
    df.loc[indices, "outlier_action"] = "break_trajectory"
    df.loc[indices, "outlier_reason"] = reason
    df.loc[indices, "trajectory_visible"] = True
    df.loc[indices, "trajectory_connect"] = False
    df.loc[indices, "trajectory_alpha"] = 0.25
    df.loc[indices, "trajectory_width"] = 0.6
    df.loc[indices, "trajectory_reason"] = f"break_due_to_{reason}"
    df.loc[indices, "trajectory_segment_id"] = segment_id


def _apply_quality_trajectory_policy(df: pd.DataFrame, start_segment_id: int) -> None:
    segment_id = start_segment_id
    for index, row in df.iterrows():
        if str(row["outlier_status"]) in {"outlier_corrected", "trajectory_break"}:
            continue
        policy = default_trajectory_policy(str(row.get("quality_flag", "")), str(row.get("landmark_name", "")))
        df.at[index, "trajectory_visible"] = policy.visible
        df.at[index, "trajectory_connect"] = policy.connect
        df.at[index, "trajectory_alpha"] = policy.alpha
        df.at[index, "trajectory_width"] = policy.width
        df.at[index, "trajectory_reason"] = policy.reason
        if str(row.get("quality_flag", "")) == "missing_long_gap":
            df.at[index, "outlier_status"] = "hidden_unreliable"
            df.at[index, "outlier_action"] = "hide"
            df.at[index, "outlier_reason"] = "missing_long_gap"
            df.at[index, "trajectory_segment_id"] = segment_id
            segment_id += 1
        elif str(row.get("quality_flag", "")) == "review_only":
            df.at[index, "outlier_status"] = "review_only"
            df.at[index, "outlier_action"] = "fade"
            df.at[index, "outlier_reason"] = "review_only"
        elif str(row.get("quality_flag", "")) in {"unreliable", "estimated_occluded_arm"}:
            df.at[index, "outlier_status"] = "trajectory_break"
            df.at[index, "outlier_action"] = "break_trajectory"
            df.at[index, "outlier_reason"] = "unreliable_quality_flag"


def _apply_outlier_scores(df: pd.DataFrame) -> None:
    max_ratio = df[["velocity_ratio", "acceleration_ratio", "jerk_ratio"]].max(axis=1, skipna=True)
    score = 1.0 / (1.0 + max_ratio.fillna(0.0))
    stable_mask = df["outlier_status"] == "unchanged"
    score.loc[stable_mask & max_ratio.isna()] = 1.0
    df["outlier_score"] = score.clip(0.0, 1.0)


def _spike_type(segment_df: pd.DataFrame, options: OutlierMinimizerOptions) -> str:
    types = []
    if (segment_df["velocity_ratio"] >= options.velocity_threshold_multiplier).any():
        types.append("velocity_spike")
    if (segment_df["acceleration_ratio"] >= options.acceleration_threshold_multiplier).any():
        types.append("acceleration_spike")
    if (segment_df["jerk_ratio"] >= options.jerk_threshold_multiplier).any():
        types.append("jerk_spike")
    if len(types) > 1:
        return "mixed_temporal_spike"
    return types[0] if types else "temporal_spike"


def _break_reason(segment_df: pd.DataFrame, spike_type: str, landmark_name: str) -> str:
    flags = set(segment_df["quality_flag"].astype(str).tolist())
    if "missing_long_gap" in flags:
        return "missing_long_gap"
    if "review_only" in flags:
        return "review_only"
    if landmark_group(landmark_name) == "hands_proxy":
        return "hands_proxy_unreliable"
    if _segment_has_protected(segment_df):
        return "unreliable_quality_flag"
    return f"{spike_type}_too_long"


def _spike_report_row(
    spike_id: int,
    source: str,
    landmark_name: str,
    start_frame: int,
    end_frame: int,
    length: int,
    fps: float,
    spike_type: str,
    segment_df: pd.DataFrame,
    corrected: bool,
) -> dict:
    return {
        "spike_id": spike_id,
        "source": source,
        "landmark_name": landmark_name,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "length_frames": int(length),
        "duration_sec": float(length / fps),
        "spike_type": spike_type,
        "max_velocity_ratio": _max_or_nan(segment_df["velocity_ratio"]),
        "max_acceleration_ratio": _max_or_nan(segment_df["acceleration_ratio"]),
        "max_jerk_ratio": _max_or_nan(segment_df["jerk_ratio"]),
        "quality_flags": ",".join(sorted(set(segment_df["quality_flag"].astype(str)))),
        "outlier_action": "interpolate_short_spike" if corrected else "break_trajectory",
        "outlier_reason": spike_type,
        "corrected": bool(corrected),
        "trajectory_break": not bool(corrected),
    }


def _trajectory_break_row(
    segment_id: int,
    source: str,
    landmark_name: str,
    start_frame: int,
    end_frame: int,
    length: int,
    fps: float,
    reason: str,
    segment_df: pd.DataFrame,
) -> dict:
    return {
        "trajectory_segment_id": int(segment_id),
        "source": source,
        "landmark_name": landmark_name,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "length_frames": int(length),
        "duration_sec": float(length / fps),
        "break_reason": reason,
        "visible_policy": bool(segment_df["trajectory_visible"].fillna(False).astype(bool).any()),
        "connect_policy": False,
        "alpha": float(segment_df["trajectory_alpha"].dropna().mean()) if not segment_df["trajectory_alpha"].dropna().empty else 0.0,
    }


def _build_report(
    minimized: pd.DataFrame,
    input_pose_csv: Path,
    metadata: dict,
    fps: float,
    frames_total: int,
    options: OutlierMinimizerOptions,
    max_correction_gap_frames: int,
    frame_floors: dict[str, float],
    spike_rows: list[dict],
    break_rows: list[dict],
    crop_refine_report_path: Path | None,
    quality_report_path: Path | None,
) -> dict:
    status_counts = minimized["outlier_status"].value_counts().to_dict()
    action_counts = minimized["outlier_action"].value_counts().to_dict()
    trajectory_reason_counts = minimized["trajectory_reason"].value_counts().to_dict()
    target_rows = minimized[minimized["source"].astype(str).isin(_target_sources(options.source))]
    return {
        "session_id": metadata.get("session_id"),
        "input_pose_csv": str(input_pose_csv),
        "crop_refine_report": str(crop_refine_report_path) if crop_refine_report_path else None,
        "quality_report": str(quality_report_path) if quality_report_path else None,
        "frames_total": int(frames_total),
        "fps": float(fps),
        "source": options.source,
        "settings": {
            "max_correction_gap_sec": options.max_correction_gap_sec,
            "max_correction_gap_frames": max_correction_gap_frames,
            "max_break_gap_sec": options.max_break_gap_sec,
            "velocity_threshold_multiplier": options.velocity_threshold_multiplier,
            "acceleration_threshold_multiplier": options.acceleration_threshold_multiplier,
            "jerk_threshold_multiplier": options.jerk_threshold_multiplier,
            "velocity_floor_m_per_s": options.velocity_floor_m_per_s,
            "acceleration_floor_m_per_s2": options.acceleration_floor_m_per_s2,
            "jerk_floor_m_per_s3": options.jerk_floor_m_per_s3,
            "velocity_floor_per_frame": frame_floors["velocity"],
            "acceleration_floor_per_frame": frame_floors["acceleration"],
            "jerk_floor_per_frame": frame_floors["jerk"],
            "trim_feature_echo": options.trim_feature_echo,
            "min_stable_neighbors": options.min_stable_neighbors,
            "landmark_policy": options.landmark_policy,
            "preserve_quality_flags": options.preserve_quality_flags,
            "sync_sources": options.sync_sources,
        },
        "counts": {
            "total_rows": int(len(minimized)),
            "target_rows": int(len(target_rows)),
            "flagged_spike_rows": int(sum(row["length_frames"] for row in spike_rows)),
            "corrected_rows": int(status_counts.get("outlier_corrected", 0)),
            "trajectory_break_rows": int(status_counts.get("trajectory_break", 0)),
            "hidden_rows": int((minimized["trajectory_visible"] == False).sum()),  # noqa: E712
            "faded_rows": int((minimized["trajectory_alpha"] < 0.5).sum()),
        },
        "status_counts": status_counts,
        "action_counts": action_counts,
        "trajectory_reason_counts": trajectory_reason_counts,
        "landmark_summary": _landmark_summary(minimized),
        "spike_count": int(len(spike_rows)),
        "trajectory_break_count": int(len(break_rows)),
        "notes": "Outlier minimization does not generate motion. It corrects only short temporal spikes and creates trajectory display policies for visualization.",
    }


def _landmark_summary(df: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for landmark_name, group in df.groupby("landmark_name", sort=True):
        summary[str(landmark_name)] = {
            "corrected_rows": int((group["outlier_status"] == "outlier_corrected").sum()),
            "trajectory_break_rows": int((group["outlier_status"] == "trajectory_break").sum()),
            "hidden_rows": int((group["trajectory_visible"] == False).sum()),  # noqa: E712
        }
    return summary


def _max_or_nan(series: pd.Series) -> float:
    values = series.dropna().astype(float)
    if values.empty:
        return np.nan
    return float(values.max())


def _spike_report_columns() -> list[str]:
    return [
        "spike_id",
        "source",
        "landmark_name",
        "start_frame",
        "end_frame",
        "length_frames",
        "duration_sec",
        "spike_type",
        "max_velocity_ratio",
        "max_acceleration_ratio",
        "max_jerk_ratio",
        "quality_flags",
        "outlier_action",
        "outlier_reason",
        "corrected",
        "trajectory_break",
    ]


def _trajectory_break_columns() -> list[str]:
    return [
        "trajectory_segment_id",
        "source",
        "landmark_name",
        "start_frame",
        "end_frame",
        "length_frames",
        "duration_sec",
        "break_reason",
        "visible_policy",
        "connect_policy",
        "alpha",
    ]
