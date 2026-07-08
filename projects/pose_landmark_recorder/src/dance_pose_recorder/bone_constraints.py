"""Bone length and reachability helpers for skeleton optimization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dance_pose_recorder.landmark_schema import POSE_LANDMARK_NAMES

LANDMARK_IDS = {name: index for index, name in enumerate(POSE_LANDMARK_NAMES)}
EPSILON = 1e-9

DEFAULT_CONSTRAINTS = {
    "angle_constraints": {
        "left_elbow": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "hard_min_deg": 15,
            "hard_max_deg": 180,
            "adaptive_margin_deg": 10,
            "action": "flag_first",
        },
        "right_elbow": {
            "points": ["right_shoulder", "right_elbow", "right_wrist"],
            "hard_min_deg": 15,
            "hard_max_deg": 180,
            "adaptive_margin_deg": 10,
            "action": "flag_first",
        },
        "left_knee": {
            "points": ["left_hip", "left_knee", "left_ankle"],
            "hard_min_deg": 20,
            "hard_max_deg": 180,
            "adaptive_margin_deg": 10,
            "action": "flag_first",
        },
        "right_knee": {
            "points": ["right_hip", "right_knee", "right_ankle"],
            "hard_min_deg": 20,
            "hard_max_deg": 180,
            "adaptive_margin_deg": 10,
            "action": "flag_first",
        },
    },
    "bone_constraints": {
        "left_upper_arm": {"points": ["left_shoulder", "left_elbow"], "min_ratio": 0.45, "max_ratio": 1.75},
        "left_lower_arm": {"points": ["left_elbow", "left_wrist"], "min_ratio": 0.45, "max_ratio": 1.75},
        "right_upper_arm": {"points": ["right_shoulder", "right_elbow"], "min_ratio": 0.45, "max_ratio": 1.75},
        "right_lower_arm": {"points": ["right_elbow", "right_wrist"], "min_ratio": 0.45, "max_ratio": 1.75},
        "left_upper_leg": {"points": ["left_hip", "left_knee"], "min_ratio": 0.45, "max_ratio": 1.75},
        "left_lower_leg": {"points": ["left_knee", "left_ankle"], "min_ratio": 0.45, "max_ratio": 1.75},
        "right_upper_leg": {"points": ["right_hip", "right_knee"], "min_ratio": 0.45, "max_ratio": 1.75},
        "right_lower_leg": {"points": ["right_knee", "right_ankle"], "min_ratio": 0.45, "max_ratio": 1.75},
    },
    "reachability_constraints": {
        "left_arm": {"root": "left_shoulder", "mid": "left_elbow", "end": "left_wrist", "length_margin_ratio": 0.10},
        "right_arm": {"root": "right_shoulder", "mid": "right_elbow", "end": "right_wrist", "length_margin_ratio": 0.10},
        "left_leg": {"root": "left_hip", "mid": "left_knee", "end": "left_ankle", "length_margin_ratio": 0.10},
        "right_leg": {"root": "right_hip", "mid": "right_knee", "end": "right_ankle", "length_margin_ratio": 0.10},
    },
    "quality_policy": {
        "reliable_flags": ["measured", "refined_measured", "low_visibility_leg_kept"],
        "uncertain_flags": ["interpolated_short_gap", "interpolated_outlier_removed", "estimated_occluded_arm"],
        "protected_flags": ["unreliable", "missing_long_gap", "review_only"],
        "never_auto_correct_flags": ["missing_long_gap", "review_only"],
    },
}


@dataclass(frozen=True)
class BoneLengthStats:
    bone_name: str
    point_a: str
    point_b: str
    median_length: float
    p01_length: float
    p99_length: float
    min_length: float
    max_length: float
    valid_sample_count: int


def load_skeleton_constraints(path: str | Path | None = None) -> dict:
    """Load skeleton constraints, falling back to built-in defaults.

    PyYAML is optional in this project. When it is unavailable, the checked-in
    YAML file is treated as documentation and the equivalent built-in defaults
    are used.
    """

    defaults = deepcopy(DEFAULT_CONSTRAINTS)
    if path is None:
        return defaults
    constraint_path = Path(path)
    if not constraint_path.exists():
        return defaults
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return defaults
    loaded = yaml.safe_load(constraint_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(defaults, loaded)


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def coordinate_fields_for_source(source: str) -> list[str]:
    if source == "pose_world":
        return ["tx", "ty", "tz"]
    return ["x", "y", "z"]


def point_from_row(row: pd.Series | object, coord_fields: Iterable[str]) -> np.ndarray:
    values = [getattr(row, field, row[field] if isinstance(row, pd.Series) and field in row else np.nan) for field in coord_fields]
    point = np.asarray(values, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        return np.asarray([np.nan, np.nan, np.nan], dtype=float)
    return point


def euclidean_distance(a: Iterable[float], b: Iterable[float]) -> float:
    a_vec = np.asarray(list(a), dtype=float)
    b_vec = np.asarray(list(b), dtype=float)
    if a_vec.shape != (3,) or b_vec.shape != (3,):
        return float("nan")
    if not (np.isfinite(a_vec).all() and np.isfinite(b_vec).all()):
        return float("nan")
    return float(np.linalg.norm(a_vec - b_vec))


def is_reliable_row(row: pd.Series | object, reliable_flags: set[str], coord_fields: Iterable[str]) -> bool:
    quality_flag = getattr(row, "quality_flag", row["quality_flag"] if isinstance(row, pd.Series) else "")
    if str(quality_flag) not in reliable_flags:
        return False
    is_valid = getattr(row, "is_valid", row["is_valid"] if isinstance(row, pd.Series) and "is_valid" in row else True)
    if not bool(is_valid):
        return False
    point = point_from_row(row, coord_fields)
    if not np.isfinite(point).all():
        return False
    for field in ("visibility", "presence"):
        value = getattr(row, field, row[field] if isinstance(row, pd.Series) and field in row else np.nan)
        if pd.notna(value) and float(value) < 0.5:
            return False
    return True


def build_frame_points(
    source_df: pd.DataFrame,
    coord_fields: Iterable[str],
    reliable_flags: set[str] | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    points_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for row in source_df.itertuples(index=False):
        if reliable_flags is not None and not is_reliable_row(row, reliable_flags, coord_fields):
            continue
        point = point_from_row(row, coord_fields)
        if not np.isfinite(point).all():
            continue
        points_by_frame.setdefault(int(row.frame), {})[str(row.landmark_name)] = point
    return points_by_frame


def compute_bone_length_statistics(
    source_df: pd.DataFrame,
    bone_constraints: dict,
    reliable_flags: set[str],
    coord_fields: Iterable[str],
) -> tuple[dict[str, BoneLengthStats], dict[str, list[float]]]:
    points_by_frame = build_frame_points(source_df, coord_fields, reliable_flags=reliable_flags)
    lengths_by_bone: dict[str, list[float]] = {name: [] for name in bone_constraints}
    for points in points_by_frame.values():
        for bone_name, spec in bone_constraints.items():
            point_a, point_b = spec["points"]
            if point_a not in points or point_b not in points:
                continue
            length = euclidean_distance(points[point_a], points[point_b])
            if np.isfinite(length):
                lengths_by_bone[bone_name].append(length)

    stats: dict[str, BoneLengthStats] = {}
    for bone_name, lengths in lengths_by_bone.items():
        point_a, point_b = bone_constraints[bone_name]["points"]
        values = np.asarray(lengths, dtype=float)
        if len(values) == 0:
            stats[bone_name] = BoneLengthStats(bone_name, point_a, point_b, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
            continue
        stats[bone_name] = BoneLengthStats(
            bone_name=bone_name,
            point_a=point_a,
            point_b=point_b,
            median_length=float(np.nanmedian(values)),
            p01_length=float(np.nanpercentile(values, 1)),
            p99_length=float(np.nanpercentile(values, 99)),
            min_length=float(np.nanmin(values)),
            max_length=float(np.nanmax(values)),
            valid_sample_count=int(np.isfinite(values).sum()),
        )
    return stats, lengths_by_bone


def bone_length_ratio(length: float, median_length: float) -> float:
    if not np.isfinite(length) or not np.isfinite(median_length) or median_length <= EPSILON:
        return float("nan")
    return float(length / median_length)


def bone_length_penalty(ratio: float, min_ratio: float, max_ratio: float) -> float:
    if not np.isfinite(ratio):
        return 0.0
    if ratio < min_ratio:
        return float(min(1.0, (min_ratio - ratio) / max(min_ratio, EPSILON)))
    if ratio > max_ratio:
        return float(min(1.0, (ratio - max_ratio) / max(max_ratio, EPSILON)))
    return 0.0


def is_bone_length_violation(ratio: float, min_ratio: float, max_ratio: float) -> bool:
    return bool(np.isfinite(ratio) and (ratio < min_ratio or ratio > max_ratio))


def reachability_status(
    root: Iterable[float],
    mid: Iterable[float],
    end: Iterable[float],
    root_mid_length: float,
    mid_end_length: float,
    margin_ratio: float,
) -> tuple[bool, str, float, float]:
    """Return reachability violation, side, ratio, and penalty."""

    root_vec = np.asarray(list(root), dtype=float)
    mid_vec = np.asarray(list(mid), dtype=float)
    end_vec = np.asarray(list(end), dtype=float)
    if not (np.isfinite(root_vec).all() and np.isfinite(mid_vec).all() and np.isfinite(end_vec).all()):
        return False, "none", float("nan"), 0.0
    if not np.isfinite(root_mid_length) or not np.isfinite(mid_end_length):
        return False, "none", float("nan"), 0.0

    distance = euclidean_distance(root_vec, end_vec)
    max_distance = root_mid_length + mid_end_length
    min_distance = abs(root_mid_length - mid_end_length)
    margin = max_distance * margin_ratio
    upper = max_distance + margin
    lower = max(0.0, min_distance - margin)
    if upper <= EPSILON:
        return False, "none", float("nan"), 0.0
    ratio = float(distance / upper)
    if distance > upper:
        return True, "max", ratio, float(min(1.0, (distance - upper) / max(upper, EPSILON)))
    if distance < lower:
        penalty = float(min(1.0, (lower - distance) / max(lower, EPSILON))) if lower > EPSILON else 0.0
        return True, "min", ratio, penalty
    return False, "none", ratio, 0.0


def bone_report_dataframe(
    stats_by_bone: dict[str, BoneLengthStats],
    bone_constraints: dict,
    lengths_by_bone: dict[str, list[float]],
) -> pd.DataFrame:
    rows = []
    for bone_name, stats in stats_by_bone.items():
        spec = bone_constraints[bone_name]
        min_ratio = float(spec.get("min_ratio", 0.45))
        max_ratio = float(spec.get("max_ratio", 1.75))
        violation_count = 0
        if np.isfinite(stats.median_length) and stats.median_length > EPSILON:
            for length in lengths_by_bone.get(bone_name, []):
                ratio = bone_length_ratio(length, stats.median_length)
                if is_bone_length_violation(ratio, min_ratio, max_ratio):
                    violation_count += 1
        rows.append(
            {
                "bone_name": bone_name,
                "point_a": stats.point_a,
                "point_b": stats.point_b,
                "median_length": stats.median_length,
                "p01_length": stats.p01_length,
                "p99_length": stats.p99_length,
                "min_length": stats.min_length,
                "max_length": stats.max_length,
                "valid_sample_count": stats.valid_sample_count,
                "violation_count": violation_count,
                "min_ratio": min_ratio,
                "max_ratio": max_ratio,
            }
        )
    return pd.DataFrame(rows)
