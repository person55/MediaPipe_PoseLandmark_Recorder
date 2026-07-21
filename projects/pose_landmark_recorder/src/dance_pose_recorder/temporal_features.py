"""Temporal feature extraction for pose landmark trajectories."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from dance_pose_recorder.quality_flags import PROTECTED_QUALITY_FLAGS


def compute_temporal_features(
    df: pd.DataFrame,
    source: str = "pose_world",
    position_fields: tuple[str, str, str] = ("tx", "ty", "tz"),
    protected_flags: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return velocity, acceleration, and jerk for each landmark row.

    Features are calculated per source/landmark in frame order. Velocity is
    the norm of the first position difference; acceleration and jerk are the
    norms of the second and third vector differences, so direction changes
    register even at constant speed. Frame gaps, protected quality flags, and
    invalid coordinates intentionally break the trajectory so false lines are
    not carried across unreliable regions.
    """

    features = pd.DataFrame(
        {
            "velocity": np.nan,
            "acceleration": np.nan,
            "jerk": np.nan,
        },
        index=df.index,
    )
    if df.empty:
        return features

    protected = set(PROTECTED_QUALITY_FLAGS if protected_flags is None else protected_flags)
    if source == "both":
        source_mask = pd.Series(True, index=df.index)
    else:
        source_mask = df["source"].astype(str) == source
    target = df[source_mask]

    for (_source, _landmark), group in target.groupby(["source", "landmark_name"], sort=False):
        ordered = group.sort_values("frame")
        previous_index: int | None = None
        previous_frame: int | None = None
        previous_coords: np.ndarray | None = None
        previous_velocity_vec: np.ndarray | None = None
        previous_acceleration_vec: np.ndarray | None = None

        for index, row in ordered.iterrows():
            frame = int(row["frame"])
            coords = _coords(row, position_fields)
            is_protected = str(row.get("quality_flag", "")) in protected
            velocity = np.nan
            acceleration = np.nan
            jerk = np.nan
            velocity_vec: np.ndarray | None = None
            acceleration_vec: np.ndarray | None = None

            if (
                previous_index is not None
                and previous_frame is not None
                and frame == previous_frame + 1
                and coords is not None
                and previous_coords is not None
                and not is_protected
            ):
                velocity_vec = coords - previous_coords
                velocity = float(np.linalg.norm(velocity_vec))
                if previous_velocity_vec is not None:
                    acceleration_vec = velocity_vec - previous_velocity_vec
                    acceleration = float(np.linalg.norm(acceleration_vec))
                    if previous_acceleration_vec is not None:
                        jerk = float(np.linalg.norm(acceleration_vec - previous_acceleration_vec))

            features.at[index, "velocity"] = velocity
            features.at[index, "acceleration"] = acceleration
            features.at[index, "jerk"] = jerk

            if is_protected or coords is None:
                previous_index = None
                previous_frame = None
                previous_coords = None
                previous_velocity_vec = None
                previous_acceleration_vec = None
                continue

            previous_index = int(index)
            previous_frame = frame
            previous_coords = coords
            previous_velocity_vec = velocity_vec
            previous_acceleration_vec = acceleration_vec

    return features


def _coords(row: pd.Series, fields: tuple[str, ...]) -> np.ndarray | None:
    values = []
    for field in fields:
        value = row.get(field)
        if value is None or pd.isna(value):
            return None
        values.append(float(value))
    coords = np.asarray(values, dtype=float)
    if not np.isfinite(coords).all():
        return None
    return coords
