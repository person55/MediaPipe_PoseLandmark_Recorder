"""Conservative skeleton optimization for refined pose data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dance_pose_recorder.bone_constraints import (
    DEFAULT_CONSTRAINTS,
    LANDMARK_IDS,
    bone_length_penalty,
    bone_length_ratio,
    bone_report_dataframe,
    build_frame_points,
    compute_bone_length_statistics,
    coordinate_fields_for_source,
    euclidean_distance,
    is_bone_length_violation,
    is_reliable_row,
    load_skeleton_constraints,
    point_from_row,
    reachability_status,
)
from dance_pose_recorder.joint_angle import joint_angle_deg
from dance_pose_recorder.optimization_report import build_optimization_report, write_json
from dance_pose_recorder.quality_flags import PROTECTED_QUALITY_FLAGS

OPTIMIZER_COLUMNS = [
    "optimizer_status",
    "optimizer_action",
    "optimizer_reason",
    "optimizer_score",
    "bone_penalty",
    "angle_penalty",
    "reachability_penalty",
    "temporal_penalty",
    "joint_angle_deg",
    "bone_length_ratio",
    "reachability_ratio",
    "optimization_segment_id",
    "optimization_source",
]

NON_CORRECTABLE_QUALITY_FLAGS = {"missing_long_gap", "review_only", "optimization_unreliable", "unreliable"}
REVIEW_PROBLEM_FLAGS = {
    "unreliable",
    "missing_long_gap",
    "interpolated_outlier_removed",
    "estimated_occluded_arm",
    "low_visibility_leg_kept",
    "review_only",
    "optimization_unreliable",
}
REFINE_PROBLEM_STATUSES = {"refined_rejected", "refined_unavailable"}
REASON_PRIORITY = [
    "missing_long_gap",
    "long_unreliable_run",
    "temporal_jump",
    "unreachable_chain",
    "bone_length_outlier",
    "joint_angle_outlier",
    "protected_quality_flag",
    "insufficient_neighbors",
]
EPSILON = 1e-9


@dataclass(frozen=True)
class SkeletonOptimizationOptions:
    max_correction_gap_sec: float = 0.10
    max_review_gap_sec: float = 1.50
    adaptive_percentile_low: float = 1.0
    adaptive_percentile_high: float = 99.0
    adaptive_margin_deg: float = 10.0
    bone_length_min_ratio: float = 0.45
    bone_length_max_ratio: float = 1.75
    reachability_margin_ratio: float = 0.10
    temporal_jump_multiplier: float = 6.0
    source: str = "pose_world"
    save_csv: bool = True
    save_jsonl: bool = True
    save_reports: bool = True
    save_preview: bool = False


@dataclass(frozen=True)
class SkeletonOptimizationResult:
    optimized_csv: Path | None
    optimized_jsonl: Path | None
    optimization_report: Path | None
    bone_length_report: Path
    joint_angle_report: Path
    optimization_segments: Path
    optimized_preview: Path | None


def optimize_pose_skeleton(
    input_refined_csv: Path,
    metadata_path: Path,
    output_dir: Path,
    constraints_path: Path | None,
    options: SkeletonOptimizationOptions,
    refine_report_path: Path | None = None,
) -> SkeletonOptimizationResult:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    refined = pd.read_csv(input_refined_csv, low_memory=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(metadata.get("fps") or 30.0)
    frames_total = int(metadata.get("frame_count_written") or refined["frame"].max() + 1)
    max_correction_gap_frames = max(1, int(round(options.max_correction_gap_sec * fps)))
    constraints = _effective_constraints(load_skeleton_constraints(constraints_path), options)
    reliable_flags = set(constraints.get("quality_policy", {}).get("reliable_flags", DEFAULT_CONSTRAINTS["quality_policy"]["reliable_flags"]))
    review_ranges = _load_refine_review_ranges(refine_report_path)

    optimized = refined.copy()
    _initialize_optimizer_columns(optimized)

    source_df = optimized[optimized["source"] == options.source].copy()
    coord_fields = coordinate_fields_for_source(options.source)
    row_reasons: dict[int, set[str]] = {}
    frame_details = _FrameDetails()

    bone_stats, lengths_by_bone = compute_bone_length_statistics(
        source_df,
        constraints["bone_constraints"],
        reliable_flags,
        coord_fields,
    )
    bone_report = bone_report_dataframe(bone_stats, constraints["bone_constraints"], lengths_by_bone)

    points_by_frame = build_frame_points(source_df, coord_fields)
    reliable_points_by_frame = build_frame_points(source_df, coord_fields, reliable_flags=reliable_flags)
    index_by_frame_landmark = _source_index_by_frame_landmark(source_df)

    _apply_bone_diagnostics(
        optimized,
        constraints,
        bone_stats,
        points_by_frame,
        index_by_frame_landmark,
        row_reasons,
        frame_details,
    )
    angle_report = _apply_angle_diagnostics(
        optimized,
        constraints,
        points_by_frame,
        reliable_points_by_frame,
        index_by_frame_landmark,
        row_reasons,
        frame_details,
        options,
    )
    _apply_reachability_diagnostics(
        optimized,
        constraints,
        bone_stats,
        points_by_frame,
        index_by_frame_landmark,
        row_reasons,
        frame_details,
        options,
    )
    _apply_temporal_diagnostics(
        optimized,
        source_df,
        coord_fields,
        reliable_flags,
        row_reasons,
        frame_details,
        options,
    )

    _assign_initial_optimizer_statuses(optimized, options.source, row_reasons, review_ranges)
    _correct_short_temporal_violations(
        optimized,
        options.source,
        coord_fields,
        max_correction_gap_frames,
    )
    _finalize_optimizer_scores(optimized)
    _propagate_pose_display_status(optimized, options.source)

    optimization_segments = _build_optimization_segments(
        optimized,
        source=options.source,
        fps=fps,
        max_correction_gap_sec=options.max_correction_gap_sec,
        review_ranges=review_ranges,
        frame_details=frame_details,
    )
    _assign_optimization_segment_ids(optimized, optimization_segments)
    _finalize_optimizer_scores(optimized)
    _propagate_pose_display_status(optimized, options.source)

    optimized_csv = None
    if options.save_csv:
        optimized_csv = output_dir / "optimized_pose.csv"
        optimized.to_csv(optimized_csv, index=False)

    optimized_jsonl = None
    if options.save_jsonl:
        optimized_jsonl = output_dir / "optimized_pose.jsonl"
        write_optimized_jsonl(optimized, optimized_jsonl, metadata)

    bone_length_report = output_dir / "bone_length_report.csv"
    bone_report.to_csv(bone_length_report, index=False)
    joint_angle_report = output_dir / "joint_angle_report.csv"
    angle_report.to_csv(joint_angle_report, index=False)
    optimization_segments_csv = output_dir / "optimization_segments.csv"
    optimization_segments.to_csv(optimization_segments_csv, index=False)

    optimization_report_path = None
    if options.save_reports:
        settings = {
            "max_correction_gap_sec": options.max_correction_gap_sec,
            "max_correction_gap_frames": max_correction_gap_frames,
            "max_review_gap_sec": options.max_review_gap_sec,
            "adaptive_percentile_low": options.adaptive_percentile_low,
            "adaptive_percentile_high": options.adaptive_percentile_high,
            "adaptive_margin_deg": options.adaptive_margin_deg,
            "bone_length_min_ratio": options.bone_length_min_ratio,
            "bone_length_max_ratio": options.bone_length_max_ratio,
            "reachability_margin_ratio": options.reachability_margin_ratio,
            "temporal_jump_multiplier": options.temporal_jump_multiplier,
        }
        report = build_optimization_report(
            metadata=metadata,
            input_refined_csv=input_refined_csv,
            frames_total=frames_total,
            fps=fps,
            source=options.source,
            settings=settings,
            optimized=optimized,
            optimization_segments=optimization_segments,
            bone_report=bone_report,
            angle_report=angle_report,
        )
        optimization_report_path = write_json(output_dir / "optimization_report.json", report)

    optimized_preview = None
    if options.save_preview:
        input_video = _resolve_input_video(metadata, Path(metadata_path))
        if input_video is not None and input_video.exists():
            from dance_pose_recorder.cleaned_preview_renderer import render_corrected_preview

            frame_status = _build_minimal_frame_status(optimized, frames_total, fps)
            optimized_preview = output_dir / "optimized_preview.mp4"
            render_corrected_preview(input_video, optimized_preview, optimized, frame_status, metadata)
        else:
            print("Skipping optimized preview: input video path could not be resolved from metadata.")

    return SkeletonOptimizationResult(
        optimized_csv=optimized_csv,
        optimized_jsonl=optimized_jsonl,
        optimization_report=optimization_report_path,
        bone_length_report=bone_length_report,
        joint_angle_report=joint_angle_report,
        optimization_segments=optimization_segments_csv,
        optimized_preview=optimized_preview,
    )


class _FrameDetails:
    def __init__(self) -> None:
        self.violation_types: dict[int, set[str]] = {}
        self.affected_landmarks: dict[int, set[str]] = {}
        self.affected_bones: dict[int, set[str]] = {}
        self.affected_angles: dict[int, set[str]] = {}

    def add(
        self,
        frame: int,
        reason: str,
        landmarks: list[str] | None = None,
        bone: str | None = None,
        angle: str | None = None,
    ) -> None:
        self.violation_types.setdefault(frame, set()).add(reason)
        if landmarks:
            self.affected_landmarks.setdefault(frame, set()).update(landmarks)
        if bone:
            self.affected_bones.setdefault(frame, set()).add(bone)
        if angle:
            self.affected_angles.setdefault(frame, set()).add(angle)


def _effective_constraints(constraints: dict, options: SkeletonOptimizationOptions) -> dict:
    result = json.loads(json.dumps(constraints))
    for spec in result.get("bone_constraints", {}).values():
        spec["min_ratio"] = options.bone_length_min_ratio
        spec["max_ratio"] = options.bone_length_max_ratio
    for spec in result.get("reachability_constraints", {}).values():
        spec["length_margin_ratio"] = options.reachability_margin_ratio
    return result


def _initialize_optimizer_columns(optimized: pd.DataFrame) -> None:
    optimized["optimizer_status"] = "unchanged"
    optimized["optimizer_action"] = "none"
    optimized["optimizer_reason"] = "none"
    optimized["optimizer_score"] = 1.0
    optimized["bone_penalty"] = 0.0
    optimized["angle_penalty"] = 0.0
    optimized["reachability_penalty"] = 0.0
    optimized["temporal_penalty"] = 0.0
    optimized["joint_angle_deg"] = np.nan
    optimized["bone_length_ratio"] = np.nan
    optimized["reachability_ratio"] = np.nan
    optimized["optimization_segment_id"] = np.nan
    optimized["optimization_source"] = "refined"


def _source_index_by_frame_landmark(source_df: pd.DataFrame) -> dict[tuple[int, str], int]:
    return {
        (int(row.frame), str(row.landmark_name)): int(row.Index)
        for row in source_df.itertuples()
    }


def _apply_bone_diagnostics(
    optimized: pd.DataFrame,
    constraints: dict,
    bone_stats: dict,
    points_by_frame: dict[int, dict[str, np.ndarray]],
    index_by_frame_landmark: dict[tuple[int, str], int],
    row_reasons: dict[int, set[str]],
    frame_details: _FrameDetails,
) -> None:
    for frame, points in points_by_frame.items():
        for bone_name, spec in constraints["bone_constraints"].items():
            point_a, point_b = spec["points"]
            if point_a not in points or point_b not in points:
                continue
            stats = bone_stats[bone_name]
            length = euclidean_distance(points[point_a], points[point_b])
            ratio = bone_length_ratio(length, stats.median_length)
            if not np.isfinite(ratio):
                continue
            min_ratio = float(spec.get("min_ratio", 0.45))
            max_ratio = float(spec.get("max_ratio", 1.75))
            penalty = bone_length_penalty(ratio, min_ratio, max_ratio)
            for landmark in [point_a, point_b]:
                index = index_by_frame_landmark.get((frame, landmark))
                if index is None:
                    continue
                _set_if_worse(optimized, index, "bone_penalty", penalty)
                _set_ratio_if_worse(optimized, index, "bone_penalty", "bone_length_ratio", penalty, ratio)
            if is_bone_length_violation(ratio, min_ratio, max_ratio):
                for landmark in [point_a, point_b]:
                    index = index_by_frame_landmark.get((frame, landmark))
                    if index is not None:
                        row_reasons.setdefault(index, set()).add("bone_length_outlier")
                frame_details.add(frame, "bone_length_outlier", [point_a, point_b], bone=bone_name)


def _apply_angle_diagnostics(
    optimized: pd.DataFrame,
    constraints: dict,
    points_by_frame: dict[int, dict[str, np.ndarray]],
    reliable_points_by_frame: dict[int, dict[str, np.ndarray]],
    index_by_frame_landmark: dict[tuple[int, str], int],
    row_reasons: dict[int, set[str]],
    frame_details: _FrameDetails,
    options: SkeletonOptimizationOptions,
) -> pd.DataFrame:
    rows = []
    for joint_name, spec in constraints["angle_constraints"].items():
        point_a, joint_center, point_c = spec["points"]
        hard_min = float(spec.get("hard_min_deg", 0.0))
        hard_max = float(spec.get("hard_max_deg", 180.0))
        reliable_angles = []
        for points in reliable_points_by_frame.values():
            if all(name in points for name in [point_a, joint_center, point_c]):
                angle = joint_angle_deg(points[point_a], points[joint_center], points[point_c])
                if np.isfinite(angle):
                    reliable_angles.append(angle)
        if reliable_angles:
            p01 = float(np.nanpercentile(reliable_angles, options.adaptive_percentile_low))
            p99 = float(np.nanpercentile(reliable_angles, options.adaptive_percentile_high))
            adaptive_min = max(hard_min, p01 - options.adaptive_margin_deg)
            adaptive_max = min(hard_max, p99 + options.adaptive_margin_deg)
            min_angle = float(np.nanmin(reliable_angles))
            max_angle = float(np.nanmax(reliable_angles))
        else:
            p01 = p99 = min_angle = max_angle = np.nan
            adaptive_min = hard_min
            adaptive_max = hard_max

        valid_sample_count = 0
        violation_count = 0
        hard_violation_count = 0
        adaptive_violation_count = 0
        for frame, points in points_by_frame.items():
            if not all(name in points for name in [point_a, joint_center, point_c]):
                continue
            angle = joint_angle_deg(points[point_a], points[joint_center], points[point_c])
            if not np.isfinite(angle):
                continue
            valid_sample_count += 1
            index = index_by_frame_landmark.get((frame, joint_center))
            if index is not None:
                optimized.at[index, "joint_angle_deg"] = angle
            hard_violation = angle < hard_min or angle > hard_max
            adaptive_violation = angle < adaptive_min or angle > adaptive_max
            if hard_violation or adaptive_violation:
                violation_count += 1
                if hard_violation:
                    hard_violation_count += 1
                if adaptive_violation:
                    adaptive_violation_count += 1
                penalty = _angle_penalty(angle, adaptive_min, adaptive_max, hard_min, hard_max)
                if index is not None:
                    _set_if_worse(optimized, index, "angle_penalty", penalty)
                    row_reasons.setdefault(index, set()).add("joint_angle_outlier")
                frame_details.add(frame, "joint_angle_outlier", [joint_center], angle=joint_name)
        rows.append(
            {
                "joint_name": joint_name,
                "point_a": point_a,
                "joint_center": joint_center,
                "point_c": point_c,
                "hard_min_deg": hard_min,
                "hard_max_deg": hard_max,
                "adaptive_min_deg": adaptive_min,
                "adaptive_max_deg": adaptive_max,
                "p01_deg": p01,
                "p99_deg": p99,
                "min_deg": min_angle,
                "max_deg": max_angle,
                "valid_sample_count": valid_sample_count,
                "violation_count": violation_count,
                "hard_violation_count": hard_violation_count,
                "adaptive_violation_count": adaptive_violation_count,
            }
        )
    return pd.DataFrame(rows)


def _angle_penalty(angle: float, adaptive_min: float, adaptive_max: float, hard_min: float, hard_max: float) -> float:
    if angle < hard_min or angle > hard_max:
        return 1.0
    if angle < adaptive_min:
        return float(min(1.0, (adaptive_min - angle) / max(adaptive_min - hard_min + 1.0, 1.0)))
    if angle > adaptive_max:
        return float(min(1.0, (angle - adaptive_max) / max(hard_max - adaptive_max + 1.0, 1.0)))
    return 0.0


def _apply_reachability_diagnostics(
    optimized: pd.DataFrame,
    constraints: dict,
    bone_stats: dict,
    points_by_frame: dict[int, dict[str, np.ndarray]],
    index_by_frame_landmark: dict[tuple[int, str], int],
    row_reasons: dict[int, set[str]],
    frame_details: _FrameDetails,
    options: SkeletonOptimizationOptions,
) -> None:
    pair_lengths = _bone_medians_by_pair(bone_stats)
    for frame, points in points_by_frame.items():
        for chain_name, spec in constraints["reachability_constraints"].items():
            root = spec["root"]
            mid = spec["mid"]
            end = spec["end"]
            if not all(name in points for name in [root, mid, end]):
                continue
            root_mid_length = pair_lengths.get(frozenset([root, mid]), np.nan)
            mid_end_length = pair_lengths.get(frozenset([mid, end]), np.nan)
            violation, _, ratio, penalty = reachability_status(
                points[root],
                points[mid],
                points[end],
                root_mid_length,
                mid_end_length,
                float(spec.get("length_margin_ratio", options.reachability_margin_ratio)),
            )
            for landmark in [root, mid, end]:
                index = index_by_frame_landmark.get((frame, landmark))
                if index is None:
                    continue
                _set_if_worse(optimized, index, "reachability_penalty", penalty)
                _set_ratio_if_worse(optimized, index, "reachability_penalty", "reachability_ratio", penalty, ratio)
                if violation:
                    row_reasons.setdefault(index, set()).add("unreachable_chain")
            if violation:
                frame_details.add(frame, "unreachable_chain", [root, mid, end], bone=chain_name)


def _bone_medians_by_pair(bone_stats: dict) -> dict[frozenset[str], float]:
    return {
        frozenset([stats.point_a, stats.point_b]): float(stats.median_length)
        for stats in bone_stats.values()
    }


def _apply_temporal_diagnostics(
    optimized: pd.DataFrame,
    source_df: pd.DataFrame,
    coord_fields: list[str],
    reliable_flags: set[str],
    row_reasons: dict[int, set[str]],
    frame_details: _FrameDetails,
    options: SkeletonOptimizationOptions,
) -> None:
    for landmark_id, group in source_df.groupby("landmark_id", sort=False):
        group = group.sort_values("frame")
        rows = list(group.itertuples())
        stable_distances = []
        for prev, current in zip(rows, rows[1:]):
            if int(current.frame) - int(prev.frame) != 1:
                continue
            if not (is_reliable_row(prev, reliable_flags, coord_fields) and is_reliable_row(current, reliable_flags, coord_fields)):
                continue
            distance = euclidean_distance(point_from_row(prev, coord_fields), point_from_row(current, coord_fields))
            if np.isfinite(distance):
                stable_distances.append(distance)
        if not stable_distances:
            continue
        median_motion = float(np.nanmedian(stable_distances))
        if not np.isfinite(median_motion) or median_motion <= EPSILON:
            continue
        threshold = median_motion * options.temporal_jump_multiplier
        for prev, current in zip(rows, rows[1:]):
            if int(current.frame) - int(prev.frame) != 1:
                continue
            distance = euclidean_distance(point_from_row(prev, coord_fields), point_from_row(current, coord_fields))
            if not np.isfinite(distance) or distance <= threshold:
                continue
            penalty = float(min(1.0, (distance - threshold) / max(threshold, EPSILON)))
            index = int(current.Index)
            _set_if_worse(optimized, index, "temporal_penalty", penalty)
            row_reasons.setdefault(index, set()).add("temporal_jump")
            frame_details.add(int(current.frame), "temporal_jump", [str(current.landmark_name)])


def _assign_initial_optimizer_statuses(
    optimized: pd.DataFrame,
    source: str,
    row_reasons: dict[int, set[str]],
    review_ranges: list[tuple[int, int]],
) -> None:
    source_mask = optimized["source"] == source
    for row in optimized[source_mask].itertuples():
        index = int(row.Index)
        frame = int(row.frame)
        quality_flag = str(row.quality_flag)
        refine_status = str(getattr(row, "refine_status", ""))
        reasons = set(row_reasons.get(index, set()))
        in_review_range = _frame_in_ranges(frame, review_ranges)

        if quality_flag == "missing_long_gap":
            _set_status(optimized, index, "optimization_unreliable", "hide_or_skip", "missing_long_gap")
            optimized.at[index, "quality_flag"] = "optimization_unreliable"
            continue
        if quality_flag in PROTECTED_QUALITY_FLAGS:
            _set_status(optimized, index, "review_only", "review_only", "protected_quality_flag")
            continue
        if in_review_range and (reasons or quality_flag in REVIEW_PROBLEM_FLAGS or refine_status in REFINE_PROBLEM_STATUSES):
            _set_status(optimized, index, "review_only", "review_only", "long_unreliable_run")
            if quality_flag in REVIEW_PROBLEM_FLAGS:
                optimized.at[index, "quality_flag"] = "review_only"
            continue
        if quality_flag == "unreliable" and reasons:
            _set_status(optimized, index, "optimization_unreliable", "hide_or_skip", _choose_reason(reasons))
            optimized.at[index, "quality_flag"] = "optimization_unreliable"
            continue
        if reasons:
            _set_status(optimized, index, "flagged", "flag_only", _choose_reason(reasons))


def _correct_short_temporal_violations(
    optimized: pd.DataFrame,
    source: str,
    coord_fields: list[str],
    max_correction_gap_frames: int,
) -> None:
    source_mask = optimized["source"] == source
    for landmark_id, group in optimized[source_mask].groupby("landmark_id", sort=False):
        group = group.sort_values("frame")
        temporal_group = group[
            (group["optimizer_status"] == "flagged")
            & (group["optimizer_reason"] == "temporal_jump")
            & (group["temporal_penalty"] > 0)
            & (~group["quality_flag"].isin(NON_CORRECTABLE_QUALITY_FLAGS))
        ]
        if temporal_group.empty:
            continue
        for segment_indices in _contiguous_index_segments(temporal_group):
            frames = optimized.loc[segment_indices, "frame"].astype(int).tolist()
            if not frames:
                continue
            if max(frames) - min(frames) + 1 > max_correction_gap_frames:
                continue
            prev_index, next_index = _stable_neighbors(optimized, group, segment_indices, coord_fields)
            if prev_index is None or next_index is None:
                for index in segment_indices:
                    if optimized.at[index, "optimizer_status"] == "flagged":
                        optimized.at[index, "optimizer_reason"] = "insufficient_neighbors"
                continue
            prev_frame = int(optimized.at[prev_index, "frame"])
            next_frame = int(optimized.at[next_index, "frame"])
            if next_frame <= prev_frame:
                continue
            for index in segment_indices:
                frame = int(optimized.at[index, "frame"])
                alpha = (frame - prev_frame) / (next_frame - prev_frame)
                for field in _correction_fields_for_source(source, coord_fields):
                    start = optimized.at[prev_index, field] if field in optimized.columns else np.nan
                    end = optimized.at[next_index, field] if field in optimized.columns else np.nan
                    if pd.isna(start) or pd.isna(end):
                        continue
                    optimized.at[index, field] = float(start) + alpha * (float(end) - float(start))
                optimized.at[index, "optimizer_status"] = "optimized_constrained"
                optimized.at[index, "optimizer_action"] = "interpolate_short_violation"
                optimized.at[index, "optimizer_reason"] = "temporal_jump"
                optimized.at[index, "optimization_source"] = "skeleton_optimizer"
                optimized.at[index, "quality_flag"] = "optimized_constrained"
                optimized.at[index, "is_valid"] = True
                optimized.at[index, "is_interpolated"] = True


def _contiguous_index_segments(group: pd.DataFrame) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    previous_frame: int | None = None
    for row in group.sort_values("frame").itertuples():
        frame = int(row.frame)
        if previous_frame is None or frame == previous_frame + 1:
            current.append(int(row.Index))
        else:
            if current:
                segments.append(current)
            current = [int(row.Index)]
        previous_frame = frame
    if current:
        segments.append(current)
    return segments


def _stable_neighbors(
    optimized: pd.DataFrame,
    landmark_group: pd.DataFrame,
    segment_indices: list[int],
    coord_fields: list[str],
) -> tuple[int | None, int | None]:
    sorted_group = landmark_group.sort_values("frame")
    segment_set = set(segment_indices)
    positions = {int(index): pos for pos, index in enumerate(sorted_group.index)}
    first_pos = min(positions[index] for index in segment_indices)
    last_pos = max(positions[index] for index in segment_indices)

    prev_index = None
    for pos in range(first_pos - 1, -1, -1):
        index = int(sorted_group.index[pos])
        if index in segment_set:
            continue
        if _is_stable_neighbor(optimized.loc[index], coord_fields):
            prev_index = index
            break

    next_index = None
    for pos in range(last_pos + 1, len(sorted_group)):
        index = int(sorted_group.index[pos])
        if index in segment_set:
            continue
        if _is_stable_neighbor(optimized.loc[index], coord_fields):
            next_index = index
            break
    return prev_index, next_index


def _is_stable_neighbor(row: pd.Series, coord_fields: list[str]) -> bool:
    if str(row.get("optimizer_status", "")) in {"review_only", "optimization_unreliable"}:
        return False
    if str(row.get("quality_flag", "")) in NON_CORRECTABLE_QUALITY_FLAGS:
        return False
    point = point_from_row(row, coord_fields)
    return bool(np.isfinite(point).all())


def _correction_fields_for_source(source: str, coord_fields: list[str]) -> list[str]:
    if source == "pose_world":
        return [field for field in ["x", "y", "z", "tx", "ty", "tz"] if field in coord_fields or field in ["x", "y", "z"]]
    return ["x", "y", "z"]


def _finalize_optimizer_scores(optimized: pd.DataFrame) -> None:
    weighted_penalty = (
        optimized["bone_penalty"].astype(float) * 0.30
        + optimized["angle_penalty"].astype(float) * 0.20
        + optimized["reachability_penalty"].astype(float) * 0.25
        + optimized["temporal_penalty"].astype(float) * 0.25
    )
    optimized["optimizer_score"] = (1.0 - weighted_penalty.clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
    optimized.loc[optimized["optimizer_status"].isin(["review_only", "optimization_unreliable"]), "optimizer_score"] = 0.0


def _propagate_pose_display_status(optimized: pd.DataFrame, source: str) -> None:
    if source == "pose":
        return
    source_rows = optimized[(optimized["source"] == source) & (optimized["optimizer_status"] != "unchanged")]
    if source_rows.empty:
        return
    pose_lookup = {
        (int(row.frame), int(row.landmark_id)): int(row.Index)
        for row in optimized[optimized["source"] == "pose"].itertuples()
    }
    display_columns = [
        "optimizer_status",
        "optimizer_action",
        "optimizer_reason",
        "optimizer_score",
        "bone_penalty",
        "angle_penalty",
        "reachability_penalty",
        "temporal_penalty",
        "joint_angle_deg",
        "bone_length_ratio",
        "reachability_ratio",
        "optimization_segment_id",
        "optimization_source",
    ]
    for row in source_rows.itertuples():
        pose_index = pose_lookup.get((int(row.frame), int(row.landmark_id)))
        if pose_index is None:
            continue
        for column in display_columns:
            optimized.at[pose_index, column] = optimized.at[int(row.Index), column]


def _build_optimization_segments(
    optimized: pd.DataFrame,
    *,
    source: str,
    fps: float,
    max_correction_gap_sec: float,
    review_ranges: list[tuple[int, int]],
    frame_details: _FrameDetails,
) -> pd.DataFrame:
    problem_rows = optimized[(optimized["source"] == source) & (optimized["optimizer_status"] != "unchanged")]
    columns = [
        "optimization_segment_id",
        "start_frame",
        "end_frame",
        "length_frames",
        "duration_sec",
        "violation_types",
        "affected_landmarks",
        "affected_bones",
        "affected_angles",
        "review_only",
        "recommended_action",
    ]
    if problem_rows.empty:
        return pd.DataFrame(columns=columns)

    frames = sorted(int(frame) for frame in problem_rows["frame"].unique())
    ranges = _contiguous_frame_ranges(frames)
    rows = []
    for segment_id, (start_frame, end_frame) in enumerate(ranges, start=1):
        segment_rows = problem_rows[problem_rows["frame"].between(start_frame, end_frame)]
        length_frames = end_frame - start_frame + 1
        duration_sec = length_frames / fps
        overlap_review = any(_ranges_overlap(start_frame, end_frame, start, end) for start, end in review_ranges)
        status_values = set(segment_rows["optimizer_status"].astype(str))
        review_only = bool(
            duration_sec > max_correction_gap_sec
            or overlap_review
            or status_values & {"review_only", "optimization_unreliable"}
        )
        if "optimized_constrained" in status_values and not review_only:
            recommended_action = "interpolate_short_violation"
        elif review_only:
            recommended_action = "review_only"
        else:
            recommended_action = "flag_only"
        detail_frames = range(start_frame, end_frame + 1)
        rows.append(
            {
                "optimization_segment_id": segment_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "length_frames": length_frames,
                "duration_sec": duration_sec,
                "violation_types": ",".join(sorted({item for frame in detail_frames for item in frame_details.violation_types.get(frame, set())})),
                "affected_landmarks": ",".join(sorted({item for frame in detail_frames for item in frame_details.affected_landmarks.get(frame, set())})),
                "affected_bones": ",".join(sorted({item for frame in detail_frames for item in frame_details.affected_bones.get(frame, set())})),
                "affected_angles": ",".join(sorted({item for frame in detail_frames for item in frame_details.affected_angles.get(frame, set())})),
                "review_only": review_only,
                "recommended_action": recommended_action,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _assign_optimization_segment_ids(optimized: pd.DataFrame, segments: pd.DataFrame) -> None:
    if segments.empty:
        return
    problem_mask = optimized["optimizer_status"] != "unchanged"
    for row in segments.itertuples(index=False):
        mask = problem_mask & optimized["frame"].between(int(row.start_frame), int(row.end_frame))
        optimized.loc[mask, "optimization_segment_id"] = int(row.optimization_segment_id)
        if bool(row.review_only):
            review_mask = mask & optimized["optimizer_status"].isin(["flagged"])
            optimized.loc[review_mask, "optimizer_status"] = "review_only"
            optimized.loc[review_mask, "optimizer_action"] = "review_only"
            optimized.loc[review_mask, "optimizer_reason"] = "long_unreliable_run"


def _set_status(optimized: pd.DataFrame, index: int, status: str, action: str, reason: str) -> None:
    optimized.at[index, "optimizer_status"] = status
    optimized.at[index, "optimizer_action"] = action
    optimized.at[index, "optimizer_reason"] = reason


def _choose_reason(reasons: set[str]) -> str:
    for reason in REASON_PRIORITY:
        if reason in reasons:
            return reason
    return sorted(reasons)[0] if reasons else "none"


def _set_if_worse(optimized: pd.DataFrame, index: int, column: str, value: float) -> None:
    if not np.isfinite(value):
        return
    current = optimized.at[index, column]
    if pd.isna(current) or float(value) > float(current):
        optimized.at[index, column] = float(value)


def _set_ratio_if_worse(
    optimized: pd.DataFrame,
    index: int,
    penalty_column: str,
    ratio_column: str,
    penalty: float,
    ratio: float,
) -> None:
    if not np.isfinite(ratio):
        return
    current_ratio = optimized.at[index, ratio_column]
    current_penalty = optimized.at[index, penalty_column]
    if pd.isna(current_ratio) or float(penalty) >= float(current_penalty):
        optimized.at[index, ratio_column] = float(ratio)


def _load_refine_review_ranges(refine_report_path: Path | None) -> list[tuple[int, int]]:
    if refine_report_path is None or not Path(refine_report_path).exists():
        return []
    data = json.loads(Path(refine_report_path).read_text(encoding="utf-8"))
    ranges = []
    for segment in data.get("segments", []):
        if bool(segment.get("review_only")):
            ranges.append((int(segment["start_frame"]), int(segment["end_frame"])))
    return ranges


def _frame_in_ranges(frame: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in ranges)


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def _contiguous_frame_ranges(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    ranges = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append((start, previous))
        start = previous = frame
    ranges.append((start, previous))
    return ranges


def _build_minimal_frame_status(optimized: pd.DataFrame, total_frames: int, fps: float) -> pd.DataFrame:
    rows = []
    for frame in range(total_frames):
        frame_df = optimized[optimized["frame"] == frame]
        pose = frame_df[frame_df["source"] == "pose"]
        long_missing = bool(pose["quality_flag"].isin(["missing_long_gap", "optimization_unreliable"]).all()) if not pose.empty else True
        rows.append(
            {
                "frame": frame,
                "time_sec": frame / fps,
                "has_pose": bool((~pose["quality_flag"].isin(["missing_long_gap", "optimization_unreliable"])).any()),
                "has_pose_world": bool((frame_df["source"] == "pose_world").any()),
                "pose_landmark_count": int(len(pose)),
                "pose_world_landmark_count": int((frame_df["source"] == "pose_world").sum()),
                "empty_frame": pose.empty,
                "is_inside_long_missing_range": long_missing,
                "interpolated_landmark_count": int(frame_df["is_interpolated"].sum()) if "is_interpolated" in frame_df else 0,
                "invalid_landmark_count": int((~frame_df["is_valid"]).sum()) if "is_valid" in frame_df else 0,
            }
        )
    return pd.DataFrame(rows)


def _resolve_input_video(metadata: dict, metadata_path: Path) -> Path | None:
    raw_path = metadata.get("source_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    project_root = metadata_path.resolve().parents[2] if len(metadata_path.resolve().parents) >= 3 else Path.cwd()
    candidates = [
        Path.cwd() / path,
        project_root / path,
        metadata_path.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_optimized_jsonl(optimized: pd.DataFrame, output_path: str | Path, metadata: dict) -> None:
    output = Path(output_path)
    session_id = metadata.get("session_id")
    with output.open("w", encoding="utf-8") as file:
        for frame, frame_df in optimized.groupby("frame", sort=True):
            record = {
                "session_id": session_id,
                "frame": int(frame),
                "time_sec": float(frame_df["time_sec"].iloc[0]),
                "pose_world_landmarks": _jsonl_landmarks(frame_df[frame_df["source"] == "pose_world"]),
                "quality_summary": frame_df["quality_flag"].value_counts().to_dict(),
                "optimization_summary": frame_df["optimizer_status"].value_counts().to_dict(),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jsonl_landmarks(frame_df: pd.DataFrame) -> list[dict]:
    landmarks = []
    fields = ["x", "y", "z", "visibility", "presence", "tx", "ty", "tz"]
    for row in frame_df.itertuples(index=False):
        if not bool(getattr(row, "is_valid", True)) and not bool(getattr(row, "is_interpolated", False)):
            continue
        item = {
            "id": int(row.landmark_id),
            "name": row.landmark_name,
            "quality_flag": row.quality_flag,
            "optimizer_status": row.optimizer_status,
            "optimizer_action": row.optimizer_action,
            "optimizer_reason": row.optimizer_reason,
        }
        for field in fields:
            value = getattr(row, field, np.nan)
            if pd.notna(value):
                item[field] = float(value)
        landmarks.append(item)
    return landmarks
