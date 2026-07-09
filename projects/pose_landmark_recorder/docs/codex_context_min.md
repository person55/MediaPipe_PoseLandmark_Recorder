# Codex Context Minimal

## Purpose

This is a lightweight MediaPipe-based pose landmark recorder for dance/movement video experiments, not a full motion-capture replacement. It preserves measured data, separates corrected/refined/optimized data, and keeps uncertainty visible.

## Scope

```text
projects/pose_landmark_recorder/
```

Do not modify upstream MediaPipe source files unless explicitly instructed.

## Pipeline

```text
record_from_video.py
-> raw_pose.csv / raw_pose.jsonl / metadata.json / preview.mp4

clean_pose_data.py
-> cleaned_pose.csv / quality_report.json / interpolation_report.json

crop_refine_pose.py
-> crop_refined_pose.csv / crop_refine_report.json / crop_segments.csv

minimize_pose_outliers.py
-> outlier_minimized_pose.csv / outlier_report.json
-> temporal_spike_report.csv / trajectory_breaks.csv

export_trajectory.py
-> blender_trajectory_points.csv / blender_trajectory_segments.csv
-> trajectory_export_report.json

planned:
build_motion_profile.py

optional:
refine_pose_segments.py
optimize_pose_skeleton.py
```

## Data Layers

```text
raw: direct MediaPipe output
cleaned: validation, short interpolation, smoothing, quality flags
crop_refined: torso-centered crop re-detection; accepts only better crop candidates
refined: full-frame segment re-detection; optional after crop refinement
outlier_minimized: default visualization layer for spike reduction and trajectory breaks
trajectory_export: Blender/TouchDesigner points and segments
optimized: optional diagnostic skeleton-constraint layer
generated: future optional layer only; never mix with measured data
```

## Default Path

```text
raw_pose
-> cleaned_pose
-> crop_refined_pose
-> outlier_minimized_pose
-> trajectory_export
-> Blender / TouchDesigner
```

## Diagnostic Path

```text
refined_pose or crop_refined_pose
-> optimize_pose_skeleton.py
-> optimization_report
-> diagnostic overlay
```

## Important Quality Flags

```text
measured
interpolated_short_gap
interpolated_outlier_removed
estimated_occluded_arm
low_visibility_leg_kept
unreliable
missing_long_gap
review_only
crop_refined_measured
refined_measured
optimized_constrained
generated_motion
```

Outlier minimization adds display/status columns:

```text
outlier_corrected
trajectory_break
trajectory_visible
trajectory_connect
trajectory_alpha
```

## Current Design Principle

Do not make uncertain motion look certain.

Long unreliable regions should not be automatically filled. They should be hidden, marked as review-only, disconnected with trajectory breaks, or handled later by a clearly separated generated-motion layer.

For visualization-first workflows, use crop-refined or outlier-minimized pose data as the primary coordinate source. Use skeleton optimization as a diagnostic overlay, not as the default final pose layer.

## Current Crop Policy

Crop refinement should be limited by default to selected short/mixed problem regions:

```text
target_segment_types: mixed_problem_segment
max_segment_length: 100
exclude_review_only: true
exclude_missing_long_gap: true
```

## Outlier Minimizer Direction

Outlier Minimizer v2:

```text
confidence-aware filtering
velocity / acceleration / jerk spike detection
landmark-group-specific policies
trajectory break handling
safe short-spike correction
no generated motion
```

## Trajectory Export Direction

Default Blender export uses `screen_bottom_origin`:

```text
screen_origin_x: 0.5
screen_origin_y: 1.0
head_proxy: nose
exclude: ears, hand index, thumb
keep: left_foot_index, right_foot_index
```

## Future Prior Strategy

Use motion profiles before learned generation: multiple stable sessions -> `motion_profile.json` -> adaptive thresholds. Keep heavy pretrained/generative models out of the core pipeline unless isolated as research backends.
