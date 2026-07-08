"""Quality report helpers for cleaned pose sessions."""

from __future__ import annotations

import json
from pathlib import Path

from dance_pose_recorder.interpolation import GapSegment


def segment_to_report(segment: GapSegment, fps: float) -> dict:
    return {
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "length": segment.length,
        "duration_sec": round(segment.length / fps, 6) if fps else 0.0,
    }


def write_json(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_quality_report(
    metadata: dict,
    frames_total: int,
    frames_with_pose_raw: int,
    long_missing_ranges: list[GapSegment],
    short_missing_ranges: list[GapSegment],
    interpolated_frame_count: int,
    interpolated_landmark_count: int,
    invalid_landmark_count: int,
    options: object,
) -> dict:
    fps = float(metadata.get("fps") or 0.0)
    frames_without_pose_raw = frames_total - frames_with_pose_raw
    detection_rate = frames_with_pose_raw / frames_total if frames_total else 0.0
    return {
        "session_id": metadata.get("session_id"),
        "fps": fps,
        "frames_total": frames_total,
        "frames_with_pose_raw": frames_with_pose_raw,
        "frames_without_pose_raw": frames_without_pose_raw,
        "raw_pose_detection_rate": round(detection_rate, 6),
        "max_interpolate_gap_frames": options.max_interpolate_gap,
        "visibility_threshold": options.visibility_threshold,
        "presence_threshold": options.presence_threshold,
        "jump_threshold_multiplier": options.jump_threshold_multiplier,
        "smoothing_window": 0 if options.no_smoothing else options.smoothing_window,
        "bone_check_enabled": options.enable_bone_check,
        "interpolate_recoverable_outliers": options.interpolate_recoverable_outliers,
        "interpolate_outliers": options.interpolate_outliers,
        "outlier_max_gap_frames": options.outlier_max_gap,
        "torso_side_lock_enabled": options.enable_torso_side_lock,
        "pelvis_side_lock_enabled": options.enable_pelvis_side_lock,
        "torso_swap_cost_ratio": options.torso_swap_cost_ratio,
        "shoulder_hip_guard_ratio": options.shoulder_hip_guard_ratio,
        "arm_occlusion_max_gap_frames": options.arm_occlusion_max_gap,
        "leg_low_visibility_salvage_enabled": options.enable_leg_low_visibility_salvage,
        "leg_salvage_min_visibility": options.leg_salvage_min_visibility,
        "interpolated_frame_count": interpolated_frame_count,
        "interpolated_landmark_count": interpolated_landmark_count,
        "invalid_landmark_count": invalid_landmark_count,
        "long_missing_ranges": [segment_to_report(segment, fps) for segment in long_missing_ranges],
        "short_missing_ranges": [segment_to_report(segment, fps) for segment in short_missing_ranges],
        "notes": (
            "Long missing ranges are not interpolated. "
            "Short missing-frame gaps up to max_interpolate_gap_frames are interpolated. "
            "Short recoverable spike-like outliers are interpolated when "
            "interpolate_recoverable_outliers is enabled. Other invalid measured outliers are not "
            "interpolated unless interpolate_outliers is enabled. Torso side-lock uses temporal "
            "left/right assignment cost. Pelvis side-lock is experimental and disabled by default "
            "to avoid cross-connecting legs. Bounded low-confidence elbow/wrist runs can be "
            "estimated from shoulder/elbow-local offsets. Stable low-visibility leg measurements "
            "can be kept when presence, motion, and bone-length checks pass."
        ),
    }
