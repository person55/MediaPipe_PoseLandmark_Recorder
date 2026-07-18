"""Single source of truth for quality flag sets shared across pipeline stages."""

from __future__ import annotations


# Measured or measurement-equivalent rows: usable as scoring baselines and
# interpolation anchors.
STABLE_MEASUREMENT_FLAGS = {
    "measured",
    "low_visibility_leg_kept",
    "crop_refined_measured",
    "refined_measured",
}

# Stable measurements plus short interpolations: trusted for trajectory
# display and outlier judgment.
RELIABLE_TRAJECTORY_FLAGS = STABLE_MEASUREMENT_FLAGS | {"interpolated_short_gap"}

# Rows no later stage may correct, bridge, or differentiate across.
PROTECTED_QUALITY_FLAGS = {
    "missing_long_gap",
    "review_only",
    "optimization_unreliable",
}

# Protected rows plus estimated data: outlier correction must not rewrite these.
OUTLIER_PROTECTED_FLAGS = PROTECTED_QUALITY_FLAGS | {"estimated_occluded_arm"}
