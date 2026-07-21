"""Read-only statistical motion profile built from stable session frames.

Scope (Codex-agreed): observational statistics only. Nothing in the pipeline
reads this profile for thresholds or acceptance decisions. Promoting any value
here into a decision guard requires more diverse sessions plus the blind-label
validation protocol recorded in docs/claude_loop_progress.md.

All temporal statistics are expressed in physical per-second units (m/s,
m/s^2, m/s^3) derived from pose_world coordinates and the session fps, so
sessions with different frame rates are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dance_pose_recorder.pose_candidate_scorer import BONE_PAIRS
from dance_pose_recorder.quality_flags import STABLE_MEASUREMENT_FLAGS

PROFILE_NOTE = (
    "Read-only statistical profile. Not wired into any pipeline threshold or "
    "acceptance decision; values are observational."
)

TEMPORAL_FEATURES = ("velocity_m_per_s", "acceleration_m_per_s2", "jerk_m_per_s3")


def build_session_motion_samples(
    pose: pd.DataFrame, fps: float
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Collect per-landmark temporal samples and bone lengths from stable rows.

    Velocity/acceleration/jerk are vector-difference norms over consecutive
    frames only (chains reset at any frame gap), converted to per-second units
    by fps, fps^2 and fps^3.
    """
    stable = pose[
        (pose["source"] == "pose_world")
        & pose["quality_flag"].isin(STABLE_MEASUREMENT_FLAGS)
        & pose[["tx", "ty", "tz"]].notna().all(axis=1)
    ]

    landmark_samples: dict[str, dict[str, np.ndarray]] = {}
    for landmark_name, group in stable.groupby("landmark_name", sort=False):
        ordered = group.sort_values("frame")
        frames = ordered["frame"].to_numpy(dtype=np.int64)
        coords = ordered[["tx", "ty", "tz"]].to_numpy(dtype=float)
        velocity: list[np.ndarray] = []
        acceleration: list[np.ndarray] = []
        jerk: list[np.ndarray] = []
        if len(frames) >= 2:
            run_breaks = np.where(np.diff(frames) != 1)[0] + 1
            for run in np.split(np.arange(len(frames)), run_breaks):
                if len(run) < 2:
                    continue
                deltas = np.diff(coords[run], axis=0)
                velocity.append(np.linalg.norm(deltas, axis=1) * fps)
                if len(run) >= 3:
                    accel_vecs = np.diff(deltas, axis=0)
                    acceleration.append(np.linalg.norm(accel_vecs, axis=1) * fps * fps)
                    if len(run) >= 4:
                        jerk_vecs = np.diff(accel_vecs, axis=0)
                        jerk.append(np.linalg.norm(jerk_vecs, axis=1) * fps * fps * fps)
        landmark_samples[str(landmark_name)] = {
            "velocity_m_per_s": np.concatenate(velocity) if velocity else np.array([]),
            "acceleration_m_per_s2": np.concatenate(acceleration) if acceleration else np.array([]),
            "jerk_m_per_s3": np.concatenate(jerk) if jerk else np.array([]),
        }

    bone_samples: dict[str, list[float]] = {name: [] for _, _, name in BONE_PAIRS}
    by_frame = stable.set_index(["frame", "landmark_id"])[["tx", "ty", "tz"]]
    coords_lookup = {key: row.to_numpy(dtype=float) for key, row in by_frame.iterrows()}
    frames_present = stable["frame"].unique()
    for frame in frames_present:
        for start_id, end_id, bone_name in BONE_PAIRS:
            start = coords_lookup.get((frame, start_id))
            end = coords_lookup.get((frame, end_id))
            if start is None or end is None:
                continue
            bone_samples[bone_name].append(float(np.linalg.norm(end - start)))
    return landmark_samples, {name: np.asarray(values) for name, values in bone_samples.items()}


def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "median": round(float(np.median(values)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "p99": round(float(np.percentile(values, 99)), 4),
        "max": round(float(values.max()), 4),
    }


def build_motion_profile(
    sessions: list[tuple[str, pd.DataFrame, float]]
) -> dict:
    """Build the pooled profile plus per-session summaries.

    sessions: list of (session_id, pose dataframe, fps).
    """
    pooled_landmarks: dict[str, dict[str, list[np.ndarray]]] = {}
    pooled_bones: dict[str, list[np.ndarray]] = {}
    per_session: dict[str, dict] = {}

    for session_id, pose, fps in sessions:
        landmark_samples, bone_samples = build_session_motion_samples(pose, fps)
        session_summary: dict = {"fps": round(float(fps), 3), "landmarks": {}}
        for landmark, features in landmark_samples.items():
            session_summary["landmarks"][landmark] = {
                feature: summarize(values) for feature, values in features.items()
            }
            pooled = pooled_landmarks.setdefault(
                landmark, {feature: [] for feature in TEMPORAL_FEATURES}
            )
            for feature in TEMPORAL_FEATURES:
                pooled[feature].append(features[feature])
        for bone_name, values in bone_samples.items():
            pooled_bones.setdefault(bone_name, []).append(values)
        session_summary["bones"] = {
            name: summarize(values) for name, values in bone_samples.items()
        }
        per_session[session_id] = session_summary

    profile = {
        "profile_version": 1,
        "note": PROFILE_NOTE,
        "sessions": {sid: per_session[sid]["fps"] for sid in per_session},
        "landmarks": {
            landmark: {
                feature: summarize(
                    np.concatenate(chunks[feature]) if chunks[feature] else np.array([])
                )
                for feature in TEMPORAL_FEATURES
            }
            for landmark, chunks in sorted(pooled_landmarks.items())
        },
        "bones": {
            bone: summarize(np.concatenate(chunks) if chunks else np.array([]))
            for bone, chunks in sorted(pooled_bones.items())
        },
        "per_session": per_session,
    }
    return profile
