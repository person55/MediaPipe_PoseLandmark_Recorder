"""Per-frame marker fade events derived from exporter trajectory_alpha values.

The Blender importer keyframes each animated marker/halo's object color alpha
from these events so uncertain frames render faded, consuming the exporter's
fade contract (Loop 6) at the marker level. Pure logic lives here so it can be
unit-tested outside Blender.
"""

from __future__ import annotations

import math

ALPHA_EPSILON = 1e-6


def alpha_from_value(value) -> float:
    """Parse a trajectory_alpha cell into a clamped [0, 1] float (default 1.0)."""
    if value is None or value == "":
        return 1.0
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        return 1.0
    if math.isnan(alpha) or math.isinf(alpha):
        return 1.0
    return min(1.0, max(0.0, alpha))


def build_marker_alpha_events(frame_alphas: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Reduce per-frame alphas to change-point keyframe events.

    Input must be sorted by frame. Consecutive frames with the same alpha are
    collapsed to the first occurrence; the events are meant to be keyframed
    with CONSTANT interpolation so alpha holds until the next change.
    """
    events: list[tuple[int, float]] = []
    previous: float | None = None
    for frame, alpha in frame_alphas:
        clamped = alpha_from_value(alpha)
        if previous is None or abs(clamped - previous) >= ALPHA_EPSILON:
            events.append((int(frame), clamped))
            previous = clamped
    return events
