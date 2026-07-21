"""Coordinate conversion helpers for Blender trajectory export."""

from __future__ import annotations

from math import isfinite

import pandas as pd


def to_screen_bottom_origin_position(
    row: pd.Series,
    screen_origin_x: float = 0.5,
    screen_origin_y: float = 1.0,
    screen_width_scale: float = 6.0,
    screen_height_scale: float = 6.0,
    depth_scale: float = 1.0,
    depth_mode: str = "pose_z",
) -> dict:
    """Convert MediaPipe normalized pose coordinates to Blender coordinates."""

    x = _finite_float(row.get("x"), "x")
    y = _finite_float(row.get("y"), "y")
    z = _finite_float(row.get("z"), "z")

    blender_x = (x - float(screen_origin_x)) * float(screen_width_scale)
    blender_z = (float(screen_origin_y) - y) * float(screen_height_scale)
    if depth_mode == "none":
        blender_y = 0.0
    elif depth_mode == "pose_z":
        # MediaPipe z decreases toward the camera; the Blender scene camera sits
        # at -Y looking toward +Y, so depth must keep the same sign as z for
        # closer landmarks to render closer to the camera.
        blender_y = z * float(depth_scale)
    else:
        raise ValueError(f"Unsupported depth mode for screen_bottom_origin: {depth_mode}")

    return {
        "blender_x": float(blender_x),
        "blender_y": float(blender_y),
        "blender_z": float(blender_z),
        "screen_x": float(x),
        "screen_y": float(y),
        "screen_z": float(z),
    }


def to_pose_world_direct_position(
    row: pd.Series,
    screen_width_scale: float = 6.0,
    screen_height_scale: float = 6.0,
    depth_scale: float = 1.0,
) -> dict:
    """Scale transformed pose_world coordinates directly for comparison/debug export."""

    tx = _finite_float(row.get("tx"), "tx")
    ty = _finite_float(row.get("ty"), "ty")
    tz = _finite_float(row.get("tz"), "tz")
    return {
        "blender_x": float(tx * float(screen_width_scale)),
        "blender_y": float(ty * float(depth_scale)),
        "blender_z": float(tz * float(screen_height_scale)),
        "screen_x": float(row.get("x")) if _is_finite(row.get("x")) else float("nan"),
        "screen_y": float(row.get("y")) if _is_finite(row.get("y")) else float("nan"),
        "screen_z": float(row.get("z")) if _is_finite(row.get("z")) else float("nan"),
    }


def to_pose_2d_flat_position(
    row: pd.Series,
    screen_origin_x: float = 0.5,
    screen_origin_y: float = 1.0,
    screen_width_scale: float = 6.0,
    screen_height_scale: float = 6.0,
) -> dict:
    """Convert normalized pose coordinates to a flat Blender X/Z plane."""

    return to_screen_bottom_origin_position(
        row,
        screen_origin_x=screen_origin_x,
        screen_origin_y=screen_origin_y,
        screen_width_scale=screen_width_scale,
        screen_height_scale=screen_height_scale,
        depth_scale=1.0,
        depth_mode="none",
    )


def _finite_float(value: object, field_name: str) -> float:
    if not _is_finite(value):
        raise ValueError(f"Invalid coordinate field {field_name}: {value}")
    return float(value)


def _is_finite(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return isfinite(float(value))
