"""Joint angle utilities for skeleton optimization."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def joint_angle_deg(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """
    Return angle ABC in degrees.

    a, b, c are 3D points and b is the joint center. Returns np.nan when
    any point is invalid or either vector has zero length.
    """

    a_vec = np.asarray(list(a), dtype=float)
    b_vec = np.asarray(list(b), dtype=float)
    c_vec = np.asarray(list(c), dtype=float)
    if a_vec.shape != (3,) or b_vec.shape != (3,) or c_vec.shape != (3,):
        return float("nan")
    if not (np.isfinite(a_vec).all() and np.isfinite(b_vec).all() and np.isfinite(c_vec).all()):
        return float("nan")

    ba = a_vec - b_vec
    bc = c_vec - b_vec
    ba_norm = float(np.linalg.norm(ba))
    bc_norm = float(np.linalg.norm(bc))
    if ba_norm == 0.0 or bc_norm == 0.0:
        return float("nan")

    cosine = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))
