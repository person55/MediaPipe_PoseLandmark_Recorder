"""Single source of truth for per-stage output column contracts.

Each stage appends its columns to the rows it receives; downstream stages and
tests read these lists instead of re-declaring them.
"""

from __future__ import annotations


# Landmark coordinate/confidence fields carried through every stage and copied
# from candidate rows on acceptance.
COORD_FIELDS = ["x", "y", "z", "visibility", "presence", "tx", "ty", "tz"]

# Columns added by the crop refinement stage.
CROP_STAGE_COLUMNS = [
    "crop_refine_status",
    "crop_refine_source",
    "crop_score_before",
    "crop_score_after",
    "crop_score_delta",
    "crop_segment_id",
    "crop_reason",
    "crop_x0",
    "crop_y0",
    "crop_w",
    "crop_h",
    "crop_margin_ratio",
    "crop_running_mode",
    "crop_rotation_deg",
    "crop_enhanced",
]

# Row layout of the crop stage candidate score report.
CROP_SCORE_COLUMNS = [
    "crop_segment_id",
    "frame",
    "source",
    "landmark_id",
    "landmark_name",
    "quality_flag_before",
    "crop_refine_status",
    "crop_reason",
    "score_before",
    "score_after",
    "score_delta",
]

# Columns added by the full-frame refinement stage.
REFINE_STAGE_COLUMNS = [
    "refine_status",
    "refine_source",
    "refine_score_before",
    "refine_score_after",
    "refine_score_delta",
    "refine_segment_id",
    "refine_reason",
]

# Row layout of the full-frame refinement candidate score report.
REFINE_SCORE_COLUMNS = [
    "segment_id",
    "frame",
    "source",
    "landmark_id",
    "landmark_name",
    "quality_flag_before",
    "refine_status",
    "refine_reason",
    "score_before",
    "score_after",
    "score_delta",
]

# Columns added by the outlier minimization stage.
OUTLIER_STAGE_COLUMNS = [
    "outlier_status",
    "outlier_action",
    "outlier_reason",
    "outlier_score",
    "velocity",
    "acceleration",
    "jerk",
    "velocity_ratio",
    "acceleration_ratio",
    "jerk_ratio",
    "trajectory_visible",
    "trajectory_connect",
    "trajectory_alpha",
    "trajectory_width",
    "trajectory_reason",
    "trajectory_segment_id",
]
