"""Pose cleaning pipeline for raw MediaPipe recorder outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dance_pose_recorder.interpolation import GapSegment, contiguous_ranges, interpolate_group_linear
from dance_pose_recorder.landmark_schema import POSE_LANDMARK_NAMES, landmark_name
from dance_pose_recorder.quality_report import build_quality_report, write_json


CLEANED_EXTRA_COLUMNS = [
    "is_valid",
    "is_interpolated",
    "is_smoothed",
    "quality_flag",
    "invalid_reason",
    "interpolation_method",
    "gap_length",
    "source_frame_prev",
    "source_frame_next",
]

CLEANED_COLUMNS = [
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
    *CLEANED_EXTRA_COLUMNS,
]

FRAME_STATUS_COLUMNS = [
    "frame",
    "time_sec",
    "has_pose",
    "has_pose_world",
    "pose_landmark_count",
    "pose_world_landmark_count",
    "empty_frame",
    "is_inside_long_missing_range",
    "interpolated_landmark_count",
    "invalid_landmark_count",
]

SOURCES = ["pose", "pose_world"]
POSE_FIELDS = ["x", "y", "z", "visibility", "presence"]
POSE_WORLD_FIELDS = ["x", "y", "z", "visibility", "presence", "tx", "ty", "tz"]
POSE_WORLD_COORD_FIELDS = ["tx", "ty", "tz"]
POSE_COORD_FIELDS = ["x", "y"]
EPSILON = 1e-6

BONES = [
    (11, 13, "left_upper_arm"),
    (13, 15, "left_lower_arm"),
    (12, 14, "right_upper_arm"),
    (14, 16, "right_lower_arm"),
    (23, 25, "left_upper_leg"),
    (25, 27, "left_lower_leg"),
    (24, 26, "right_upper_leg"),
    (26, 28, "right_lower_leg"),
]

UPPER_BODY_SIDE_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
PELVIS_SIDE_PAIRS = [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
UPPER_BODY_COST_PAIRS = [(11, 12)]
PELVIS_COST_PAIRS = [(23, 24), (25, 26)]
TORSO_COST_LANDMARKS = [11, 12]
PELVIS_COST_LANDMARKS = [23, 24]
LEFT_ARM = {"side": "left", "shoulder": 11, "elbow": 13, "wrist": 15}
RIGHT_ARM = {"side": "right", "shoulder": 12, "elbow": 14, "wrist": 16}
ARM_SPECS = [LEFT_ARM, RIGHT_ARM]
LOWER_LEG_LANDMARK_IDS = {25, 26, 27, 28, 29, 30, 31, 32}
LEG_SALVAGE_BONES = [
    (23, 25, "left_upper_leg"),
    (25, 27, "left_lower_leg"),
    (27, 29, "left_heel"),
    (27, 31, "left_foot"),
    (24, 26, "right_upper_leg"),
    (26, 28, "right_lower_leg"),
    (28, 30, "right_heel"),
    (28, 32, "right_foot"),
]
LEG_SALVAGE_ANCHORS = {
    25: [23],
    27: [23, 25],
    29: [23, 25, 27],
    31: [23, 25, 27],
    26: [24],
    28: [24, 26],
    30: [24, 26, 28],
    32: [24, 26, 28],
}
LEG_SALVAGE_RELEVANT_BONES = {
    25: [(23, 25), (25, 27)],
    27: [(25, 27)],
    29: [(27, 29)],
    31: [(27, 31)],
    26: [(24, 26), (26, 28)],
    28: [(26, 28)],
    30: [(28, 30)],
    32: [(28, 32)],
}
BONE_LENGTH_MIN_RATIO = 0.4
BONE_LENGTH_MAX_RATIO = 1.8


@dataclass(frozen=True)
class CleaningOptions:
    input_csv: Path
    metadata: Path
    output: Path
    input_jsonl: Path | None = None
    input_video: Path | None = None
    max_interpolate_gap: int = 15
    visibility_threshold: float = 0.5
    presence_threshold: float = 0.5
    jump_threshold_multiplier: float = 6.0
    smoothing_window: int = 7
    save_csv: bool = True
    save_jsonl: bool = True
    save_preview: bool = False
    no_smoothing: bool = False
    enable_bone_check: bool = True
    interpolate_recoverable_outliers: bool = True
    interpolate_outliers: bool = False
    outlier_max_gap: int = 3
    enable_torso_side_lock: bool = True
    enable_pelvis_side_lock: bool = False
    torso_swap_cost_ratio: float = 0.65
    shoulder_hip_guard_ratio: float = 0.98
    arm_occlusion_max_gap: int = 55
    enable_leg_low_visibility_salvage: bool = True
    leg_salvage_min_visibility: float = 0.15


@dataclass(frozen=True)
class CleaningResult:
    cleaned_csv: Path | None
    cleaned_jsonl: Path | None
    frame_status_csv: Path
    quality_report: Path
    interpolation_report: Path
    corrected_preview: Path | None


def clean_pose_session(options: CleaningOptions) -> CleaningResult:
    metadata = json.loads(options.metadata.read_text(encoding="utf-8"))
    raw_df = pd.read_csv(options.input_csv)
    frames = _frame_range(metadata, options.input_jsonl)
    output_dir = options.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned = _build_full_grid(raw_df, metadata, frames)
    frames_with_pose_raw = _frames_with_raw_pose(cleaned, "pose_world")
    missing_ranges = _missing_frame_ranges(frames, frames_with_pose_raw)
    long_missing_ranges = [segment for segment in missing_ranges if segment.length > options.max_interpolate_gap]
    short_missing_ranges = [segment for segment in missing_ranges if segment.length <= options.max_interpolate_gap]
    long_missing_frames = _frames_from_segments(long_missing_ranges)

    cleaned = _mark_initial_quality(cleaned, long_missing_frames)
    cleaned = _apply_confidence_thresholds(cleaned, options)
    if options.enable_torso_side_lock:
        cleaned = _apply_torso_side_lock(cleaned, options)
    cleaned = _apply_jump_detection(cleaned, "pose_world", POSE_WORLD_COORD_FIELDS, options)
    cleaned = _apply_jump_detection(cleaned, "pose", POSE_COORD_FIELDS, options)
    if options.enable_bone_check:
        cleaned = _apply_bone_length_check(cleaned)
    cleaned = _apply_leg_low_visibility_salvage(cleaned, options)

    cleaned["was_invalid_before_interpolation"] = cleaned["has_raw"] & (~cleaned["is_valid"])
    cleaned, interpolation_items = _interpolate_short_gaps(cleaned, frames, options)
    cleaned = _finalize_missing_and_unreliable(cleaned, long_missing_frames)
    cleaned, arm_items = _estimate_occluded_arms(cleaned, options)
    interpolation_items.extend(arm_items)

    if not options.no_smoothing:
        cleaned = _apply_smoothing(cleaned, options.smoothing_window)

    frame_status = _build_frame_status(cleaned, frames, metadata, long_missing_frames)
    quality_payload = build_quality_report(
        metadata=metadata,
        frames_total=len(frames),
        frames_with_pose_raw=len(frames_with_pose_raw),
        long_missing_ranges=long_missing_ranges,
        short_missing_ranges=short_missing_ranges,
        interpolated_frame_count=int(cleaned.loc[cleaned["is_interpolated"], "frame"].nunique()),
        interpolated_landmark_count=int(cleaned["is_interpolated"].sum()),
        invalid_landmark_count=int(cleaned["was_invalid_before_interpolation"].sum()),
        options=options,
    )
    interpolation_payload = {
        "session_id": metadata.get("session_id"),
        "method": "linear",
        "items": interpolation_items,
    }

    cleaned_csv = None
    if options.save_csv:
        cleaned_csv = output_dir / "cleaned_pose.csv"
        cleaned[CLEANED_COLUMNS].to_csv(cleaned_csv, index=False)

    cleaned_jsonl = None
    if options.save_jsonl:
        cleaned_jsonl = output_dir / "cleaned_pose.jsonl"
        write_cleaned_jsonl(cleaned, cleaned_jsonl, metadata)

    frame_status_csv = output_dir / "frame_status.csv"
    frame_status.to_csv(frame_status_csv, index=False)
    quality_report = write_json(output_dir / "quality_report.json", quality_payload)
    interpolation_report = write_json(output_dir / "interpolation_report.json", interpolation_payload)

    corrected_preview = None
    if options.save_preview:
        if options.input_video is None:
            raise ValueError("--input-video is required when --save-preview is used")
        from dance_pose_recorder.cleaned_preview_renderer import render_corrected_preview

        corrected_preview = output_dir / "corrected_preview.mp4"
        render_corrected_preview(options.input_video, corrected_preview, cleaned, frame_status, metadata)

    return CleaningResult(
        cleaned_csv=cleaned_csv,
        cleaned_jsonl=cleaned_jsonl,
        frame_status_csv=frame_status_csv,
        quality_report=quality_report,
        interpolation_report=interpolation_report,
        corrected_preview=corrected_preview,
    )


def write_cleaned_jsonl(cleaned: pd.DataFrame, output_path: str | Path, metadata: dict) -> None:
    output = Path(output_path)
    session_id = metadata.get("session_id")
    with output.open("w", encoding="utf-8") as file:
        for frame, frame_df in cleaned.groupby("frame", sort=True):
            time_sec = float(frame_df["time_sec"].iloc[0])
            cleaned_landmarks = []
            for row in frame_df.itertuples(index=False):
                if not bool(row.is_valid) and not bool(row.is_interpolated):
                    continue
                item = {
                    "id": int(row.landmark_id),
                    "name": row.landmark_name,
                    "source": row.source,
                    "quality_flag": row.quality_flag,
                    "is_interpolated": bool(row.is_interpolated),
                    "is_smoothed": bool(row.is_smoothed),
                }
                for field in ("x", "y", "z", "visibility", "presence", "tx", "ty", "tz"):
                    value = getattr(row, field)
                    if pd.notna(value):
                        item[field] = float(value)
                if pd.notna(row.gap_length):
                    item["gap_length"] = int(row.gap_length)
                cleaned_landmarks.append(item)
            record = {
                "session_id": session_id,
                "frame": int(frame),
                "time_sec": time_sec,
                "cleaned_landmarks": cleaned_landmarks,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _frame_range(metadata: dict, input_jsonl: Path | None) -> list[int]:
    if input_jsonl and input_jsonl.exists():
        frames = []
        with input_jsonl.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    frames.append(int(json.loads(line)["frame"]))
        if frames:
            return list(range(min(frames), max(frames) + 1))

    frame_count = int(metadata.get("frame_count_written") or metadata.get("source_frame_count") or 0)
    return list(range(frame_count))


def _build_full_grid(raw_df: pd.DataFrame, metadata: dict, frames: list[int]) -> pd.DataFrame:
    frame_df = pd.DataFrame({"frame": frames})
    source_df = pd.DataFrame({"source": SOURCES})
    landmark_df = pd.DataFrame(
        {
            "landmark_id": list(range(len(POSE_LANDMARK_NAMES))),
            "landmark_name": [landmark_name(index) for index in range(len(POSE_LANDMARK_NAMES))],
        }
    )
    full = frame_df.merge(source_df, how="cross").merge(landmark_df, how="cross")
    merged = full.merge(raw_df, on=["frame", "source", "landmark_id"], how="left", suffixes=("", "_raw"))

    if "landmark_name_raw" in merged.columns:
        merged["landmark_name"] = merged["landmark_name_raw"].fillna(merged["landmark_name"])
        merged = merged.drop(columns=["landmark_name_raw"])

    fps = float(metadata.get("fps") or 30.0)
    merged["session_id"] = merged.get("session_id", pd.Series(index=merged.index)).fillna(metadata.get("session_id"))
    merged["time_sec"] = merged.get("time_sec", pd.Series(index=merged.index)).fillna(merged["frame"] / fps)
    for field in ("x", "y", "z", "visibility", "presence", "tx", "ty", "tz"):
        if field not in merged.columns:
            merged[field] = np.nan
    merged["has_raw"] = merged["x"].notna()
    return merged


def _frames_with_raw_pose(cleaned: pd.DataFrame, source: str) -> set[int]:
    source_df = cleaned[(cleaned["source"] == source) & cleaned["has_raw"]]
    counts = source_df.groupby("frame")["landmark_id"].nunique()
    return set(int(frame) for frame, count in counts.items() if count > 0)


def _missing_frame_ranges(frames: list[int], frames_with_pose: set[int]) -> list[GapSegment]:
    missing = [frame for frame in frames if frame not in frames_with_pose]
    return contiguous_ranges(missing)


def _frames_from_segments(segments: Iterable[GapSegment]) -> set[int]:
    frames: set[int] = set()
    for segment in segments:
        frames.update(range(segment.start_frame, segment.end_frame + 1))
    return frames


def _mark_initial_quality(cleaned: pd.DataFrame, long_missing_frames: set[int]) -> pd.DataFrame:
    result = cleaned.copy()
    result["is_valid"] = result["has_raw"]
    result["is_interpolated"] = False
    result["is_smoothed"] = False
    result["quality_flag"] = np.where(result["has_raw"], "measured", "missing_short_gap")
    result.loc[(~result["has_raw"]) & result["frame"].isin(long_missing_frames), "quality_flag"] = "missing_long_gap"
    result["invalid_reason"] = np.where(result["has_raw"], "", "missing_frame")
    result["interpolation_method"] = ""
    result["gap_length"] = pd.NA
    result["source_frame_prev"] = pd.NA
    result["source_frame_next"] = pd.NA
    result["is_recoverable_outlier"] = False
    return result


def _append_reason(current: object, reason: str) -> str:
    text = "" if pd.isna(current) else str(current)
    if not text:
        return reason
    if reason in text.split(";"):
        return text
    return f"{text};{reason}"


def _mark_invalid(result: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    if not mask.any():
        return
    result.loc[mask, "is_valid"] = False
    result.loc[mask, "quality_flag"] = "unreliable"
    result.loc[mask, "invalid_reason"] = result.loc[mask, "invalid_reason"].apply(_append_reason, reason=reason)


def _apply_confidence_thresholds(cleaned: pd.DataFrame, options: CleaningOptions) -> pd.DataFrame:
    result = cleaned.copy()
    measured = result["has_raw"]
    low_visibility = measured & result["visibility"].notna() & (result["visibility"] < options.visibility_threshold)
    low_presence = measured & result["presence"].notna() & (result["presence"] < options.presence_threshold)
    _mark_invalid(result, low_visibility, "low_visibility")
    _mark_invalid(result, low_presence, "low_presence")
    return result


def _apply_jump_detection(
    cleaned: pd.DataFrame,
    source: str,
    fields: list[str],
    options: CleaningOptions,
) -> pd.DataFrame:
    result = cleaned.copy()
    for landmark_id, group in result[(result["source"] == source) & result["has_raw"]].groupby("landmark_id"):
        group = group.sort_values("frame")
        coords = group[fields].astype(float)
        frame_diff = group["frame"].diff()
        distances = np.sqrt(((coords - coords.shift(1)) ** 2).sum(axis=1))
        adjacent_distances = distances[(frame_diff == 1) & distances.notna()]
        if adjacent_distances.empty:
            continue
        median_distance = float(adjacent_distances.median())
        threshold = max(median_distance * options.jump_threshold_multiplier, EPSILON)
        recoverable_index = _recoverable_jump_indices(group, coords, distances, threshold, options.outlier_max_gap)
        if recoverable_index:
            recoverable_mask = result.index.isin(recoverable_index)
            _mark_invalid(result, recoverable_mask, "jump_outlier")
            result.loc[recoverable_mask, "is_recoverable_outlier"] = True
    return result


def _recoverable_jump_indices(
    group: pd.DataFrame,
    coords: pd.DataFrame,
    distances: pd.Series,
    threshold: float,
    max_gap: int,
) -> set[int]:
    """Return rows that look like short spikes bounded by plausible motion."""

    if max_gap < 1 or len(group) < 3:
        return set()

    recoverable: set[int] = set()
    frames = group["frame"].astype(int).tolist()
    indices = group.index.tolist()
    cursor = 1
    max_end_exclusive = len(group) - 1
    while cursor < max_end_exclusive:
        matched_end: int | None = None
        max_end = min(cursor + max_gap - 1, len(group) - 2)
        for end in range(cursor, max_end + 1):
            prev_pos = cursor - 1
            next_pos = end + 1
            if frames[next_pos] - frames[prev_pos] != next_pos - prev_pos:
                continue

            left_distance = float(distances.iloc[cursor])
            right_distance = _coordinate_distance(coords.iloc[end], coords.iloc[next_pos])
            bridge_distance = _coordinate_distance(coords.iloc[prev_pos], coords.iloc[next_pos])
            if left_distance > threshold and right_distance > threshold and bridge_distance <= threshold:
                matched_end = end
                break

        if matched_end is None:
            cursor += 1
            continue

        recoverable.update(indices[cursor : matched_end + 1])
        cursor = matched_end + 1

    return recoverable


def _coordinate_distance(start: pd.Series, end: pd.Series) -> float:
    return float(np.sqrt(((end - start) ** 2).sum()))


def _apply_torso_side_lock(cleaned: pd.DataFrame, options: CleaningOptions) -> pd.DataFrame:
    result_groups = []
    for source, source_df in cleaned.groupby("source", sort=False):
        fields = POSE_WORLD_COORD_FIELDS if source == "pose_world" else POSE_COORD_FIELDS
        locked = _apply_side_lock_for_source(
            source_df,
            fields,
            side_pairs=UPPER_BODY_SIDE_PAIRS,
            cost_pairs=UPPER_BODY_COST_PAIRS,
            corrected_landmarks=TORSO_COST_LANDMARKS,
            quality_flag="shoulder_swap_corrected",
            reason="shoulder_swap_corrected",
            options=options,
        )
        if options.enable_pelvis_side_lock:
            locked = _apply_side_lock_for_source(
                locked,
                fields,
                side_pairs=PELVIS_SIDE_PAIRS,
                cost_pairs=PELVIS_COST_PAIRS,
                corrected_landmarks=PELVIS_COST_LANDMARKS,
                quality_flag="pelvis_swap_corrected",
                reason="pelvis_swap_corrected",
                options=options,
            )
        result_groups.append(locked)
    return pd.concat(result_groups, ignore_index=True).sort_values(["frame", "source", "landmark_id"])


def _apply_side_lock_for_source(
    source_df: pd.DataFrame,
    fields: list[str],
    side_pairs: list[tuple[int, int]],
    cost_pairs: list[tuple[int, int]],
    corrected_landmarks: list[int],
    quality_flag: str,
    reason: str,
    options: CleaningOptions,
) -> pd.DataFrame:
    result = source_df.sort_values(["frame", "landmark_id"]).copy()
    previous: dict[int, np.ndarray] = {}
    for frame, frame_df in result.groupby("frame", sort=True):
        current = _frame_positions(frame_df, fields, side_pairs, require_valid=True)
        if not previous:
            previous = current
            continue

        normal_cost, swapped_cost = _side_assignment_costs(current, previous, cost_pairs)
        if np.isfinite(normal_cost) and np.isfinite(swapped_cost):
            if swapped_cost < normal_cost * options.torso_swap_cost_ratio:
                frame_mask = result["frame"] == frame
                if quality_flag == "shoulder_swap_corrected" and not _shoulder_swap_preserves_torso(
                    result[frame_mask], fields, options.shoulder_hip_guard_ratio
                ):
                    if current:
                        previous.update(current)
                    continue
                result = _swap_side_pairs(result, frame_mask, side_pairs)
                corrected_mask = frame_mask & result["landmark_id"].isin(corrected_landmarks)
                result.loc[corrected_mask, "quality_flag"] = quality_flag
                result.loc[corrected_mask, "invalid_reason"] = result.loc[corrected_mask, "invalid_reason"].apply(
                    _append_reason, reason=reason
                )
                frame_df = result[frame_mask]
                current = _frame_positions(frame_df, fields, side_pairs, require_valid=True)

        if current:
            previous.update(current)

    return result


def _frame_positions(
    frame_df: pd.DataFrame,
    fields: list[str],
    side_pairs: list[tuple[int, int]],
    require_valid: bool,
) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    side_landmarks = {landmark for pair in side_pairs for landmark in pair}
    for row in frame_df.itertuples():
        if int(row.landmark_id) not in side_landmarks:
            continue
        if require_valid and not bool(row.is_valid):
            continue
        values = np.array([getattr(row, field) for field in fields], dtype=float)
        if np.isnan(values).any():
            continue
        positions[int(row.landmark_id)] = values
    return positions


def _side_assignment_costs(
    current: dict[int, np.ndarray],
    previous: dict[int, np.ndarray],
    cost_pairs: list[tuple[int, int]],
) -> tuple[float, float]:
    normal_cost = 0.0
    swapped_cost = 0.0
    count = 0
    for left_id, right_id in cost_pairs:
        if left_id not in current or right_id not in current or left_id not in previous or right_id not in previous:
            continue
        normal_cost += float(np.linalg.norm(current[left_id] - previous[left_id]))
        normal_cost += float(np.linalg.norm(current[right_id] - previous[right_id]))
        swapped_cost += float(np.linalg.norm(current[right_id] - previous[left_id]))
        swapped_cost += float(np.linalg.norm(current[left_id] - previous[right_id]))
        count += 2
    if count == 0:
        return float("inf"), float("inf")
    return normal_cost / count, swapped_cost / count


def _shoulder_swap_preserves_torso(frame_df: pd.DataFrame, fields: list[str], guard_ratio: float) -> bool:
    positions = _positions_for_landmarks(frame_df, fields, [11, 12, 23, 24])
    if not all(landmark_id in positions for landmark_id in [11, 12, 23, 24]):
        return True

    current_cost = _paired_distance(positions[11], positions[23]) + _paired_distance(positions[12], positions[24])
    swapped_cost = _paired_distance(positions[12], positions[23]) + _paired_distance(positions[11], positions[24])
    return swapped_cost <= current_cost * guard_ratio


def _positions_for_landmarks(
    frame_df: pd.DataFrame,
    fields: list[str],
    landmark_ids: list[int],
) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    for row in frame_df.itertuples():
        landmark_id = int(row.landmark_id)
        if landmark_id not in landmark_ids or not bool(row.is_valid):
            continue
        values = np.array([getattr(row, field) for field in fields], dtype=float)
        if np.isnan(values).any():
            continue
        positions[landmark_id] = values
    return positions


def _paired_distance(start: np.ndarray, end: np.ndarray) -> float:
    return float(np.linalg.norm(start - end))


def _swap_side_pairs(
    result: pd.DataFrame,
    frame_mask: pd.Series,
    side_pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    for left_id, right_id in side_pairs:
        left_mask = frame_mask & (result["landmark_id"] == left_id)
        right_mask = frame_mask & (result["landmark_id"] == right_id)
        if not left_mask.any() or not right_mask.any():
            continue
        left_index = result.index[left_mask][0]
        right_index = result.index[right_mask][0]
        swap_columns = [
            field
            for field in [
                "x",
                "y",
                "z",
                "visibility",
                "presence",
                "tx",
                "ty",
                "tz",
                "has_raw",
                "is_valid",
                "quality_flag",
                "invalid_reason",
            ]
            if field in result.columns
        ]
        left_values = result.loc[left_index, swap_columns].copy()
        result.loc[left_index, swap_columns] = result.loc[right_index, swap_columns].to_numpy()
        result.loc[right_index, swap_columns] = left_values.to_numpy()
    return result


def _apply_bone_length_check(cleaned: pd.DataFrame) -> pd.DataFrame:
    result = cleaned.copy()
    world = result[(result["source"] == "pose_world") & result["has_raw"]]
    for start_id, end_id, bone_name in BONES:
        start = world[world["landmark_id"] == start_id][["frame", "tx", "ty", "tz", "is_valid"]].rename(
            columns={"tx": "start_tx", "ty": "start_ty", "tz": "start_tz", "is_valid": "start_valid"}
        )
        end = world[world["landmark_id"] == end_id][["frame", "tx", "ty", "tz", "is_valid"]].rename(
            columns={"tx": "end_tx", "ty": "end_ty", "tz": "end_tz", "is_valid": "end_valid"}
        )
        merged = start.merge(end, on="frame")
        valid = merged["start_valid"] & merged["end_valid"]
        if not valid.any():
            continue
        lengths = np.sqrt(
            (merged["start_tx"] - merged["end_tx"]) ** 2
            + (merged["start_ty"] - merged["end_ty"]) ** 2
            + (merged["start_tz"] - merged["end_tz"]) ** 2
        )
        median_length = float(lengths[valid].median())
        if median_length <= EPSILON:
            continue
        outlier_frames = set(
            int(frame)
            for frame in merged.loc[
                valid & ((lengths > median_length * 1.8) | (lengths < median_length * 0.4)),
                "frame",
            ].tolist()
        )
        mask = (
            (result["source"] == "pose_world")
            & result["frame"].isin(outlier_frames)
            & result["landmark_id"].isin([start_id, end_id])
        )
        _mark_invalid(result, mask, f"bone_length_outlier:{bone_name}")
    return result


def _apply_leg_low_visibility_salvage(cleaned: pd.DataFrame, options: CleaningOptions) -> pd.DataFrame:
    if not options.enable_leg_low_visibility_salvage:
        return cleaned

    result = cleaned.copy()
    salvage_indices: list[int] = []
    for source in SOURCES:
        fields = POSE_WORLD_COORD_FIELDS if source == "pose_world" else POSE_COORD_FIELDS
        source_lookup = _source_lookup(result, source)
        motion_stable_keys = _stable_motion_keys(result, source, fields, options)
        bone_medians = _leg_bone_medians(result, source, fields)
        candidates = result[
            (result["source"] == source)
            & result["landmark_id"].isin(LOWER_LEG_LANDMARK_IDS)
            & result["has_raw"]
            & (~result["is_valid"])
            & (~result["is_interpolated"])
        ]
        for row in candidates.itertuples():
            frame = int(row.frame)
            landmark_id = int(row.landmark_id)
            reason = "" if pd.isna(row.invalid_reason) else str(row.invalid_reason)
            if "low_visibility" not in reason:
                continue
            if any(blocker in reason for blocker in ["low_presence", "jump_outlier", "bone_length_outlier"]):
                continue
            if pd.isna(row.visibility) or float(row.visibility) < options.leg_salvage_min_visibility:
                continue
            if pd.notna(row.presence) and float(row.presence) < options.presence_threshold:
                continue
            if (source, landmark_id, frame) not in motion_stable_keys:
                continue
            if not _leg_anchor_chain_is_usable(
                source_lookup,
                landmark_id,
                fields,
                frame,
                options,
            ):
                continue
            if not _leg_bone_lengths_are_plausible(
                source_lookup,
                landmark_id,
                fields,
                frame,
                bone_medians,
            ):
                continue
            salvage_indices.append(row.Index)

    if not salvage_indices:
        return result

    index_mask = result.index.isin(salvage_indices)
    result.loc[index_mask, "is_valid"] = True
    result.loc[index_mask, "quality_flag"] = "low_visibility_leg_kept"
    result.loc[index_mask, "invalid_reason"] = result.loc[index_mask, "invalid_reason"].apply(
        _append_reason, reason="low_visibility_leg_kept"
    )
    return result


def _source_lookup(cleaned: pd.DataFrame, source: str) -> pd.DataFrame:
    return cleaned[cleaned["source"] == source].set_index(["landmark_id", "frame"], drop=False)


def _lookup_row(lookup: pd.DataFrame, landmark_id: int, frame: int) -> pd.Series | None:
    try:
        row = lookup.loc[(landmark_id, frame)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        return row.iloc[0]
    return row


def _stable_motion_keys(
    cleaned: pd.DataFrame,
    source: str,
    fields: list[str],
    options: CleaningOptions,
) -> set[tuple[str, int, int]]:
    stable_keys: set[tuple[str, int, int]] = set()
    source_df = cleaned[(cleaned["source"] == source) & cleaned["has_raw"]].sort_values(["landmark_id", "frame"])
    for landmark_id, group in source_df.groupby("landmark_id", sort=True):
        coords = group[fields].astype(float)
        frames = group["frame"].astype(int)
        prev_frame_diff = frames.diff()
        next_frame_diff = frames.shift(-1) - frames
        prev_distances = np.sqrt(((coords - coords.shift(1)) ** 2).sum(axis=1))
        next_distances = np.sqrt(((coords.shift(-1) - coords) ** 2).sum(axis=1))
        adjacent_distances = prev_distances[(prev_frame_diff == 1) & prev_distances.notna()]
        if adjacent_distances.empty:
            continue
        threshold = max(float(adjacent_distances.median()) * options.jump_threshold_multiplier, EPSILON)
        for row_index, frame in zip(group.index, frames, strict=True):
            checks = []
            if prev_frame_diff.loc[row_index] == 1 and pd.notna(prev_distances.loc[row_index]):
                checks.append(float(prev_distances.loc[row_index]) <= threshold)
            if next_frame_diff.loc[row_index] == 1 and pd.notna(next_distances.loc[row_index]):
                checks.append(float(next_distances.loc[row_index]) <= threshold)
            if checks and all(checks):
                stable_keys.add((source, int(landmark_id), int(frame)))
    return stable_keys


def _leg_bone_medians(
    cleaned: pd.DataFrame,
    source: str,
    fields: list[str],
) -> dict[tuple[int, int], float]:
    medians: dict[tuple[int, int], float] = {}
    source_df = cleaned[(cleaned["source"] == source) & cleaned["has_raw"]]
    for start_id, end_id, _bone_name in LEG_SALVAGE_BONES:
        start = source_df[source_df["landmark_id"] == start_id][["frame", *fields, "is_valid"]].rename(
            columns={field: f"start_{field}" for field in fields} | {"is_valid": "start_valid"}
        )
        end = source_df[source_df["landmark_id"] == end_id][["frame", *fields, "is_valid"]].rename(
            columns={field: f"end_{field}" for field in fields} | {"is_valid": "end_valid"}
        )
        merged = start.merge(end, on="frame")
        valid = merged["start_valid"] & merged["end_valid"]
        if not valid.any():
            continue
        start_values = merged[[f"start_{field}" for field in fields]].astype(float).to_numpy()
        end_values = merged[[f"end_{field}" for field in fields]].astype(float).to_numpy()
        lengths = np.sqrt(((start_values - end_values) ** 2).sum(axis=1))
        valid_lengths = lengths[valid.to_numpy()]
        if len(valid_lengths) == 0:
            continue
        median_length = float(np.nanmedian(valid_lengths))
        if median_length > EPSILON:
            medians[(start_id, end_id)] = median_length
    return medians


def _leg_anchor_chain_is_usable(
    source_lookup: pd.DataFrame,
    landmark_id: int,
    fields: list[str],
    frame: int,
    options: CleaningOptions,
) -> bool:
    anchors = LEG_SALVAGE_ANCHORS.get(landmark_id, [])
    return all(_leg_support_row_is_usable(source_lookup, anchor_id, fields, frame, options) for anchor_id in anchors)


def _leg_support_row_is_usable(
    source_lookup: pd.DataFrame,
    landmark_id: int,
    fields: list[str],
    frame: int,
    options: CleaningOptions,
) -> bool:
    item = _lookup_row(source_lookup, landmark_id, frame)
    if item is None:
        return False
    if not bool(item["has_raw"]):
        return False
    values = item[fields].astype(float).to_numpy()
    if np.isnan(values).any():
        return False
    reason = "" if pd.isna(item["invalid_reason"]) else str(item["invalid_reason"])
    if any(blocker in reason for blocker in ["low_presence", "jump_outlier", "bone_length_outlier"]):
        return False
    if pd.notna(item["presence"]) and float(item["presence"]) < options.presence_threshold:
        return False
    if pd.notna(item["visibility"]) and float(item["visibility"]) < options.leg_salvage_min_visibility:
        return False
    return bool(item["is_valid"]) or "low_visibility" in reason


def _leg_bone_lengths_are_plausible(
    source_lookup: pd.DataFrame,
    landmark_id: int,
    fields: list[str],
    frame: int,
    bone_medians: dict[tuple[int, int], float],
) -> bool:
    for start_id, end_id in LEG_SALVAGE_RELEVANT_BONES.get(landmark_id, []):
        median_length = bone_medians.get((start_id, end_id))
        if median_length is None:
            continue
        start_values = _landmark_values_from_lookup(source_lookup, start_id, fields, frame, require_valid=False)
        end_values = _landmark_values_from_lookup(source_lookup, end_id, fields, frame, require_valid=False)
        if start_values is None or end_values is None:
            return False
        length = float(np.linalg.norm(start_values - end_values))
        if length < median_length * BONE_LENGTH_MIN_RATIO or length > median_length * BONE_LENGTH_MAX_RATIO:
            return False
    return True


def _fields_for_source(source: str) -> list[str]:
    return POSE_WORLD_FIELDS if source == "pose_world" else POSE_FIELDS


def _interpolate_short_gaps(
    cleaned: pd.DataFrame,
    frames: list[int],
    options: CleaningOptions,
) -> tuple[pd.DataFrame, list[dict]]:
    result_groups = []
    report_items: list[dict] = []
    for (source, landmark_id), group in cleaned.groupby(["source", "landmark_id"], sort=True):
        fields = _fields_for_source(source)
        interpolation_fields = [field for field in fields if field in group.columns]
        group = group.sort_values("frame").copy()
        group["stable_for_interpolation"] = group["has_raw"] & group["is_valid"]
        group["candidate_missing_interpolation"] = ~group["has_raw"]
        interpolated, segments = interpolate_group_linear(
            group,
            fields=interpolation_fields,
            stable_column="stable_for_interpolation",
            max_gap=options.max_interpolate_gap,
            candidate_column="candidate_missing_interpolation",
        )
        report_items.extend(
            _apply_interpolation_segments(
                interpolated,
                segments,
                source=source,
                landmark_id=int(landmark_id),
                fields=fields,
                reason="missing_short_gap",
            )
        )

        interpolated["stable_for_interpolation"] = interpolated["has_raw"] & interpolated["is_valid"]
        interpolated["candidate_outlier_interpolation"] = False
        measured_invalid = interpolated["has_raw"] & (~interpolated["is_valid"]) & (~interpolated["is_interpolated"])
        if options.interpolate_recoverable_outliers:
            interpolated["candidate_outlier_interpolation"] = (
                interpolated["candidate_outlier_interpolation"] | interpolated["is_recoverable_outlier"]
            )
        if options.interpolate_outliers:
            interpolated["candidate_outlier_interpolation"] = (
                interpolated["candidate_outlier_interpolation"] | measured_invalid
            )
        interpolated["candidate_outlier_interpolation"] = (
            interpolated["candidate_outlier_interpolation"] & measured_invalid
        )
        if options.outlier_max_gap >= 1:
            interpolated, outlier_segments = interpolate_group_linear(
                interpolated,
                fields=interpolation_fields,
                stable_column="stable_for_interpolation",
                max_gap=options.outlier_max_gap,
                candidate_column="candidate_outlier_interpolation",
            )
            report_items.extend(
                _apply_interpolation_segments(
                    interpolated,
                    outlier_segments,
                    source=source,
                    landmark_id=int(landmark_id),
                    fields=fields,
                    reason="invalid_short_gap",
                )
            )

        result_groups.append(
            interpolated.drop(
                columns=[
                    "stable_for_interpolation",
                    "candidate_missing_interpolation",
                    "candidate_outlier_interpolation",
                ]
            )
        )

    return pd.concat(result_groups, ignore_index=True).sort_values(["frame", "source", "landmark_id"]), report_items


def _apply_interpolation_segments(
    interpolated: pd.DataFrame,
    segments: list[GapSegment],
    source: str,
    landmark_id: int,
    fields: list[str],
    reason: str,
) -> list[dict]:
    report_items: list[dict] = []
    for segment in segments:
        frame_mask = interpolated["frame"].between(segment.start_frame, segment.end_frame)
        missing_mask = frame_mask & (~interpolated["has_raw"])
        invalid_mask = frame_mask & interpolated["has_raw"]
        interpolated.loc[frame_mask, "is_valid"] = True
        interpolated.loc[frame_mask, "is_interpolated"] = True
        interpolated.loc[frame_mask, "interpolation_method"] = "linear"
        interpolated.loc[frame_mask, "gap_length"] = segment.length
        interpolated.loc[frame_mask, "source_frame_prev"] = segment.prev_frame
        interpolated.loc[frame_mask, "source_frame_next"] = segment.next_frame
        interpolated.loc[missing_mask, "quality_flag"] = "interpolated_short_gap"
        interpolated.loc[invalid_mask, "quality_flag"] = "interpolated_outlier_removed"
        interpolated.loc[missing_mask, "invalid_reason"] = "missing_frame"

        report_items.append(
            {
                "source": source,
                "landmark_id": landmark_id,
                "landmark_name": landmark_name(landmark_id),
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "gap_length": segment.length,
                "prev_stable_frame": segment.prev_frame,
                "next_stable_frame": segment.next_frame,
                "interpolated_fields": fields,
                "reason": reason,
            }
        )
    return report_items


def _finalize_missing_and_unreliable(cleaned: pd.DataFrame, long_missing_frames: set[int]) -> pd.DataFrame:
    result = cleaned.copy()
    unfilled_missing = (~result["has_raw"]) & (~result["is_interpolated"])
    result.loc[unfilled_missing, "is_valid"] = False
    result.loc[unfilled_missing, "quality_flag"] = "missing_long_gap"
    result.loc[unfilled_missing & (~result["frame"].isin(long_missing_frames)), "quality_flag"] = "missing_long_gap"
    unfilled_invalid = result["has_raw"] & (~result["is_valid"]) & (~result["is_interpolated"])
    result.loc[unfilled_invalid, "quality_flag"] = "unreliable"
    return result


def _estimate_occluded_arms(cleaned: pd.DataFrame, options: CleaningOptions) -> tuple[pd.DataFrame, list[dict]]:
    if options.arm_occlusion_max_gap < 1:
        return cleaned, []

    result = cleaned.copy()
    report_items: list[dict] = []
    for source in SOURCES:
        fields = POSE_WORLD_COORD_FIELDS if source == "pose_world" else POSE_COORD_FIELDS
        for arm in ARM_SPECS:
            result, elbow_items = _estimate_occluded_joint(
                result,
                source=source,
                joint_id=int(arm["elbow"]),
                anchor_id=int(arm["shoulder"]),
                fields=fields,
                max_gap=options.arm_occlusion_max_gap,
                method="shoulder_local",
            )
            report_items.extend(elbow_items)
            result, wrist_items = _estimate_occluded_joint(
                result,
                source=source,
                joint_id=int(arm["wrist"]),
                anchor_id=int(arm["elbow"]),
                fields=fields,
                max_gap=options.arm_occlusion_max_gap,
                method="elbow_local",
            )
            report_items.extend(wrist_items)
    return result, report_items


def _estimate_occluded_joint(
    cleaned: pd.DataFrame,
    source: str,
    joint_id: int,
    anchor_id: int,
    fields: list[str],
    max_gap: int,
    method: str,
) -> tuple[pd.DataFrame, list[dict]]:
    result = cleaned
    source_lookup = _source_lookup(result, source)
    joint = result[(result["source"] == source) & (result["landmark_id"] == joint_id)].sort_values("frame")
    if joint.empty:
        return result, []
    joint_index_by_frame = {int(frame): index for index, frame in joint["frame"].items()}

    candidate = joint[
        joint["has_raw"]
        & (~joint["is_valid"])
        & (~joint["is_interpolated"])
        & joint["invalid_reason"].astype(str).str.contains("low_visibility|low_presence", regex=True)
    ]
    report_items: list[dict] = []
    for segment in contiguous_ranges(candidate["frame"].astype(int).tolist()):
        if segment.length > max_gap:
            continue
        prev_frame = _nearest_stable_joint_frame_from_lookup(
            source_lookup,
            joint,
            anchor_id,
            fields,
            segment.start_frame,
            -1,
        )
        next_frame = _nearest_stable_joint_frame_from_lookup(
            source_lookup,
            joint,
            anchor_id,
            fields,
            segment.end_frame,
            1,
        )
        if prev_frame is None or next_frame is None:
            continue

        prev_offset = _joint_anchor_offset_from_lookup(source_lookup, joint_id, anchor_id, fields, prev_frame)
        next_offset = _joint_anchor_offset_from_lookup(source_lookup, joint_id, anchor_id, fields, next_frame)
        if prev_offset is None or next_offset is None:
            continue

        estimated_count = 0
        span = next_frame - prev_frame
        for frame in range(segment.start_frame, segment.end_frame + 1):
            if not _arm_anchor_is_stable(source_lookup, anchor_id, fields, frame):
                continue
            anchor_values = _landmark_values_from_lookup(
                source_lookup,
                anchor_id,
                fields,
                frame,
                require_valid=True,
            )
            if anchor_values is None or span <= 0:
                continue
            ratio = (frame - prev_frame) / span
            offset = prev_offset + (next_offset - prev_offset) * ratio
            values = anchor_values + offset
            row_index = joint_index_by_frame.get(frame)
            if row_index is None:
                continue
            for field, value in zip(fields, values, strict=True):
                result.at[row_index, field] = float(value)
            result.at[row_index, "is_valid"] = True
            result.at[row_index, "is_interpolated"] = True
            result.at[row_index, "quality_flag"] = "estimated_occluded_arm"
            result.at[row_index, "interpolation_method"] = method
            result.at[row_index, "gap_length"] = segment.length
            result.at[row_index, "source_frame_prev"] = prev_frame
            result.at[row_index, "source_frame_next"] = next_frame
            result.at[row_index, "invalid_reason"] = _append_reason(
                result.at[row_index, "invalid_reason"], "estimated_occluded_arm"
            )
            estimated_count += 1

        if estimated_count:
            report_items.append(
                {
                    "source": source,
                    "landmark_id": joint_id,
                    "landmark_name": landmark_name(joint_id),
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "gap_length": segment.length,
                    "prev_stable_frame": prev_frame,
                    "next_stable_frame": next_frame,
                    "interpolated_fields": fields,
                    "reason": "estimated_occluded_arm",
                    "method": method,
                    "estimated_frame_count": estimated_count,
                }
            )
    return result, report_items


def _arm_anchor_is_stable(
    source_lookup: pd.DataFrame,
    anchor_id: int,
    fields: list[str],
    frame: int,
) -> bool:
    anchor_row = _lookup_row(source_lookup, anchor_id, frame)
    if anchor_row is None:
        return False

    quality = str(anchor_row["quality_flag"])
    if anchor_id in {11, 12} and quality == "shoulder_swap_corrected":
        try:
            frame_df = source_lookup.xs(frame, level="frame", drop_level=False)
        except KeyError:
            return False
        return _shoulder_hip_side_is_clear(frame_df, fields)
    return True


def _shoulder_hip_side_is_clear(frame_df: pd.DataFrame, fields: list[str]) -> bool:
    positions = _positions_for_landmarks(frame_df, fields, [11, 12, 23, 24])
    if not all(landmark_id in positions for landmark_id in [11, 12, 23, 24]):
        return True
    side_cost = _paired_distance(positions[11], positions[23]) + _paired_distance(positions[12], positions[24])
    cross_cost = _paired_distance(positions[11], positions[24]) + _paired_distance(positions[12], positions[23])
    return side_cost <= cross_cost


def _nearest_stable_joint_frame(
    cleaned: pd.DataFrame,
    source: str,
    joint_id: int,
    anchor_id: int,
    fields: list[str],
    frame: int,
    direction: int,
) -> int | None:
    joint = cleaned[(cleaned["source"] == source) & (cleaned["landmark_id"] == joint_id)]
    if direction < 0:
        candidates = joint[joint["frame"] < frame].sort_values("frame", ascending=False)
    else:
        candidates = joint[joint["frame"] > frame].sort_values("frame", ascending=True)

    for row in candidates.itertuples(index=False):
        candidate_frame = int(row.frame)
        if not bool(row.is_valid) or not all(pd.notna(getattr(row, field)) for field in fields):
            continue
        if _landmark_values(cleaned, source, anchor_id, fields, candidate_frame, require_valid=True) is not None:
            return candidate_frame
    return None


def _nearest_stable_joint_frame_from_lookup(
    source_lookup: pd.DataFrame,
    joint: pd.DataFrame,
    anchor_id: int,
    fields: list[str],
    frame: int,
    direction: int,
) -> int | None:
    if direction < 0:
        candidates = joint[joint["frame"] < frame].sort_values("frame", ascending=False)
    else:
        candidates = joint[joint["frame"] > frame].sort_values("frame", ascending=True)

    for row in candidates.itertuples(index=False):
        candidate_frame = int(row.frame)
        if not bool(row.is_valid) or not all(pd.notna(getattr(row, field)) for field in fields):
            continue
        if _landmark_values_from_lookup(source_lookup, anchor_id, fields, candidate_frame, require_valid=True) is not None:
            return candidate_frame
    return None


def _joint_anchor_offset(
    cleaned: pd.DataFrame,
    source: str,
    joint_id: int,
    anchor_id: int,
    fields: list[str],
    frame: int,
) -> np.ndarray | None:
    joint_values = _landmark_values(cleaned, source, joint_id, fields, frame, require_valid=True)
    anchor_values = _landmark_values(cleaned, source, anchor_id, fields, frame, require_valid=True)
    if joint_values is None or anchor_values is None:
        return None
    return joint_values - anchor_values


def _joint_anchor_offset_from_lookup(
    source_lookup: pd.DataFrame,
    joint_id: int,
    anchor_id: int,
    fields: list[str],
    frame: int,
) -> np.ndarray | None:
    joint_values = _landmark_values_from_lookup(source_lookup, joint_id, fields, frame, require_valid=True)
    anchor_values = _landmark_values_from_lookup(source_lookup, anchor_id, fields, frame, require_valid=True)
    if joint_values is None or anchor_values is None:
        return None
    return joint_values - anchor_values


def _landmark_values(
    cleaned: pd.DataFrame,
    source: str,
    landmark_id: int,
    fields: list[str],
    frame: int,
    require_valid: bool,
) -> np.ndarray | None:
    row = cleaned[
        (cleaned["source"] == source)
        & (cleaned["landmark_id"] == landmark_id)
        & (cleaned["frame"] == frame)
    ]
    if row.empty:
        return None
    item = row.iloc[0]
    if require_valid and not bool(item["is_valid"]):
        return None
    values = item[fields].astype(float).to_numpy()
    if np.isnan(values).any():
        return None
    return values


def _landmark_values_from_lookup(
    source_lookup: pd.DataFrame,
    landmark_id: int,
    fields: list[str],
    frame: int,
    require_valid: bool,
) -> np.ndarray | None:
    item = _lookup_row(source_lookup, landmark_id, frame)
    if item is None:
        return None
    if require_valid and not bool(item["is_valid"]):
        return None
    values = item[fields].astype(float).to_numpy()
    if np.isnan(values).any():
        return None
    return values


def _apply_smoothing(cleaned: pd.DataFrame, smoothing_window: int) -> pd.DataFrame:
    if smoothing_window <= 1:
        return cleaned
    if smoothing_window % 2 == 0:
        raise ValueError("--smoothing-window must be odd")

    result_groups = []
    for (_source, _landmark_id), group in cleaned.groupby(["source", "landmark_id"], sort=True):
        group = group.sort_values("frame").copy()
        fields = POSE_WORLD_COORD_FIELDS if group["source"].iloc[0] == "pose_world" else POSE_COORD_FIELDS
        valid = group["is_valid"] & group[fields].notna().all(axis=1)
        segment_id = (valid != valid.shift(fill_value=False)).cumsum()
        for _, segment in group[valid].groupby(segment_id[valid]):
            if segment.empty:
                continue
            smoothed = segment[fields].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            group.loc[segment.index, fields] = smoothed
            group.loc[segment.index, "is_smoothed"] = True
        result_groups.append(group)
    return pd.concat(result_groups, ignore_index=True).sort_values(["frame", "source", "landmark_id"])


def _build_frame_status(
    cleaned: pd.DataFrame,
    frames: list[int],
    metadata: dict,
    long_missing_frames: set[int],
) -> pd.DataFrame:
    fps = float(metadata.get("fps") or 30.0)
    rows = []
    raw_counts = cleaned[cleaned["has_raw"]].groupby(["frame", "source"])["landmark_id"].nunique()
    interpolated_counts = cleaned[cleaned["is_interpolated"]].groupby("frame")["landmark_id"].count()
    invalid_counts = cleaned[cleaned["was_invalid_before_interpolation"]].groupby("frame")["landmark_id"].count()
    for frame in frames:
        pose_count = int(raw_counts.get((frame, "pose"), 0))
        world_count = int(raw_counts.get((frame, "pose_world"), 0))
        rows.append(
            {
                "frame": frame,
                "time_sec": frame / fps,
                "has_pose": pose_count > 0,
                "has_pose_world": world_count > 0,
                "pose_landmark_count": pose_count,
                "pose_world_landmark_count": world_count,
                "empty_frame": pose_count == 0 and world_count == 0,
                "is_inside_long_missing_range": frame in long_missing_frames,
                "interpolated_landmark_count": int(interpolated_counts.get(frame, 0)),
                "invalid_landmark_count": int(invalid_counts.get(frame, 0)),
            }
        )
    return pd.DataFrame(rows, columns=FRAME_STATUS_COLUMNS)
