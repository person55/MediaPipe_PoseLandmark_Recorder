"""Coordinate origin and axis transforms for downstream 3D tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

LEFT_HIP_ID = 23
RIGHT_HIP_ID = 24


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


class CoordinateTransformer:
    """Apply origin policy and Blender-oriented axis mapping."""

    def __init__(self, origin_policy: str = "raw", scale: float = 1.0) -> None:
        if origin_policy not in {"raw", "first_frame_pelvis", "per_frame_pelvis"}:
            raise ValueError(f"Unsupported origin policy: {origin_policy}")
        self.origin_policy = origin_policy
        self.scale = scale
        self._first_origin: Point3 | None = None

    def transform_landmarks(self, landmarks: list[dict]) -> list[dict]:
        if not landmarks:
            return []

        origin = self._origin_for_frame(landmarks)
        transformed = []
        for landmark in landmarks:
            point = Point3(
                float(landmark["x"]) - origin.x,
                float(landmark["y"]) - origin.y,
                float(landmark["z"]) - origin.z,
            )
            transformed.append(
                {
                    "id": landmark["id"],
                    "name": landmark["name"],
                    "tx": point.x * self.scale,
                    "ty": -point.z * self.scale,
                    "tz": -point.y * self.scale,
                }
            )
        return transformed

    def _origin_for_frame(self, landmarks: list[dict]) -> Point3:
        if self.origin_policy == "raw":
            return Point3(0.0, 0.0, 0.0)

        pelvis = pelvis_center(landmarks)
        if self.origin_policy == "per_frame_pelvis":
            return pelvis

        if self._first_origin is None:
            self._first_origin = pelvis
        return self._first_origin


def pelvis_center(landmarks: Iterable[dict]) -> Point3:
    by_id = {landmark["id"]: landmark for landmark in landmarks}
    try:
        left = by_id[LEFT_HIP_ID]
        right = by_id[RIGHT_HIP_ID]
    except KeyError as exc:
        raise ValueError("Pelvis origin requires left_hip and right_hip landmarks") from exc

    return Point3(
        (float(left["x"]) + float(right["x"])) / 2.0,
        (float(left["y"]) + float(right["y"])) / 2.0,
        (float(left["z"]) + float(right["z"])) / 2.0,
    )
