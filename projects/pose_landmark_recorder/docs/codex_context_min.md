# Codex Context Minimal

## Purpose

This project is a lightweight MediaPipe-based pose landmark recorder for dance/movement video experiments.

It is not a full motion-capture replacement.
It preserves measured data, separates corrected/refined/optimized data, and keeps uncertainty visible through quality flags.

## Folder Scope

Main project folder:

```text
projects/pose_landmark_recorder/
```

Do not modify upstream MediaPipe source files unless explicitly instructed.

## Pipeline

```text
record_from_video.py
-> raw_pose.csv / raw_pose.jsonl / metadata.json / preview.mp4

clean_pose_data.py
-> cleaned_pose.csv / cleaned_pose.jsonl
-> quality_report.json / interpolation_report.json / frame_status.csv
-> corrected_preview.mp4

refine_pose_segments.py
-> refined_pose.csv / refined_pose.jsonl
-> refine_report.json / candidate_segments.csv / candidate_scores.csv

optimize_pose_skeleton.py
-> optimized_pose.csv / optimized_pose.jsonl
-> optimization_report.json / bone_length_report.csv / joint_angle_report.csv

planned:
minimize_pose_outliers.py
-> outlier_minimized_pose.csv / outlier_report.json

planned:
build_motion_profile.py
-> motion_profile.json
```

## Data Layers

```text
raw:
  direct MediaPipe output

cleaned:
  validation, short interpolation, smoothing, quality flags

refined:
  segment re-detection result; accepts only better measurement candidates

optimized:
  conservative skeleton diagnostics and limited correction

outlier_minimized:
  planned layer for reducing visual/temporal spikes without generating motion

generated:
  future optional layer only; must never be mixed with measured data
```

## Important Quality Flags

```text
measured
interpolated_short_gap
interpolated_outlier_removed
estimated_occluded_arm
low_visibility_leg_kept
unreliable
refined_measured
optimized_constrained
review_only
generated_motion
```

## Current Design Principle

Do not make uncertain motion look certain.

Long unreliable regions should not be automatically filled.
They should be hidden, marked as review-only, or handled later by a clearly separated generated-motion layer.

## Current Best Next Direction

Skeleton optimization has limited reconstruction value.
The next core improvement should be Outlier Minimizer v2.

Outlier Minimizer v2 should focus on:

```text
confidence-aware filtering
velocity / acceleration / jerk outlier detection
landmark-group-specific policies
trajectory break handling
safe short-gap correction
no generated motion
```

## Future Prior Strategy

Use motion profiles before learned generation.

Preferred lightweight prior:

```text
multiple stable sessions
-> motion_profile.json
-> adaptive thresholds
-> safer filtering/interpolation rules
```

Avoid adding heavy pretrained/generative models to the core pipeline.
If used later, they should live in a separate research backend.
