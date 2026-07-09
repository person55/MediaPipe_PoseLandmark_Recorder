# Next Development Plan

## Current Direction Update

The project is currently visualization-oriented.

Skeleton optimization has been tested and found useful mainly as a diagnostic/reporting layer. It is not the preferred default path for final visualization because conservative flags such as `review_only` and `optimization_unreliable` can make the skeleton appear incomplete when hidden downstream.

Therefore, do not prioritize stronger skeleton optimization unless the project explicitly shifts toward learned motion reconstruction or generated interpolation.

The next implementation priority is:

```text
Outlier Minimizer v2
```

The default visualization-oriented path should be:

```text
crop_refined_pose.csv
-> outlier_minimized_pose.csv
-> blender_trajectory_points.csv / blender_trajectory_segments.csv
-> Blender importer
```

Before Outlier Minimizer v2, crop refinement should be tested as a lightweight way to improve MediaPipe input quality on problematic segments. The current crop refine default is limited to selected `mixed_problem_segment` regions and excludes long unreliable or `missing_long_gap` ranges.

## Current Assessment

The current pipeline has reached the limit of simple interpolation and conservative skeleton optimization.

Segment re-detection can recover only a small number of better candidates when the source video still contains usable visual evidence.

Skeleton optimization is useful as a diagnostic and flagging layer, but it should not be treated as a reconstruction engine.
It cannot reliably recover long occlusion, missing body parts, or frame-out motion.

Therefore, the next core improvement should be:

```text
Outlier Minimizer v2
```

After that, introduce:

```text
Motion Profile Builder
```

as a lightweight prior system.

## Skeleton Optimizer Role

Skeleton Optimizer should remain in the repository, but its role is diagnostic.

It can produce:

- `optimization_report.json`
- `bone_length_report.csv`
- `joint_angle_report.csv`
- `optimization_segments.csv`

It should not be the default final data source for visualization.

Recommended use:

```text
Use refined or outlier-minimized coordinates for visual continuity.
Use optimizer reports as overlays, warnings, or review guides.
```

## Priority 1 - Outlier Minimizer v2

### Purpose

Reduce visual and temporal outliers without generating new motion.

This module should improve downstream Blender visualization by minimizing spikes, breaking unreliable trajectories, and applying confidence-aware filtering.

Outlier Minimizer v2 should become the next core visualization-oriented improvement.

Its purpose is not to generate missing motion, but to reduce visual spikes and preserve readable trajectories.

It should support:

- confidence-aware filtering
- velocity / acceleration / jerk spike detection
- landmark-group-specific policy
- trajectory break handling
- short spike correction only when stable neighbors exist
- no generated motion

### Input

```text
crop_refined_pose.csv
or
refined_pose.csv
metadata.json
optional: crop_refine_report.json
optional: refine_report.json
optional: optimization_report.json
optional: motion_profile.json
```

### Output

```text
outlier_minimized_pose.csv
outlier_minimized_pose.jsonl
outlier_report.json
temporal_spike_report.csv
trajectory_breaks.csv
```

### Core Ideas

```text
1. confidence-aware filtering
2. velocity / acceleration / jerk outlier detection
3. landmark-group-specific policy
4. short spike correction
5. long unreliable run hiding / trajectory break
6. no generated motion
```

### Current Crop Refine Handoff

```text
cleaned_pose.csv
-> crop_refine_pose.py
-> crop_refined_pose.csv
-> minimize_pose_outliers.py
```

Crop refine should not spend default compute on:

```text
long_unreliable_run
review_only
missing_long_gap
segments longer than 100 frames
```

### Proposed Script

```text
scripts/minimize_pose_outliers.py
```

### Proposed Modules

```text
src/dance_pose_recorder/outlier_minimizer.py
src/dance_pose_recorder/temporal_features.py
src/dance_pose_recorder/trajectory_policy.py
src/dance_pose_recorder/outlier_report.py
```

### Quality Flag Policy

Do not erase uncertainty.

Suggested additional flags:

```text
outlier_minimized
trajectory_break
held_previous_stable
hidden_unreliable
```

Existing flags must remain distinguishable:

```text
measured
refined_measured
optimized_constrained
interpolated_short_gap
interpolated_outlier_removed
estimated_occluded_arm
low_visibility_leg_kept
unreliable
review_only
missing_long_gap
```

### Landmark Group Policy

```text
torso:
  preserve strongly; minimal filtering.

shoulder/hip:
  conservative filtering only.

elbow/knee:
  allow short spike correction.

wrist/ankle:
  allow confidence-aware smoothing and short interpolation.

thumb/index/pinky:
  do not force reconstruction; hide or show as uncertain unless confidence is high.

heel/foot_index:
  allow filtering, but preserve trajectory breaks when confidence is low.
```

### Temporal Features

Calculate:

```text
position
velocity
acceleration
jerk
```

Outlier candidates:

```text
temporal_position_jump
temporal_velocity_spike
temporal_acceleration_spike
temporal_jerk_spike
```

### Filtering Candidates

Use lightweight filters first:

```text
median filter
Hampel-style outlier detection
confidence-aware Savitzky-Golay smoothing
confidence-aware Kalman filter as optional later
```

Initial implementation should avoid heavy dependencies.
Prefer NumPy/Pandas-based logic.

### Acceptance Policy

Only apply correction when:

```text
- gap/spike is short
- stable neighbors exist
- correction improves temporal continuity
- correction does not cross review_only or missing_long_gap regions
```

Do not correct:

```text
- long unreliable runs
- missing_long_gap
- review_only
- frame-out regions
- hand proxy landmarks with long instability
```

## Implemented Trajectory Export

`export_trajectory.py` converts `outlier_minimized_pose.csv` into Blender/TouchDesigner-ready point and segment CSV files.

Default export policy:

```text
coordinate_mode: screen_bottom_origin
screen_origin_x: 0.5
screen_origin_y: 1.0
source: pose
depth_mode: pose_z
head_proxy: nose
```

Default Blender landmark preset excludes ears, hand index, and thumb while keeping `left_foot_index` and `right_foot_index`.

This export is a visualization coordinate system, not camera calibration or real-world 3D reconstruction.

## Priority 2 - Motion Profile Builder

### Purpose

Introduce a lightweight prior system without training a model.

Instead of using heavy pretrained generative models, build a project-specific statistical motion profile from stable frames across multiple sessions.

### Input

```text
cleaned_pose.csv
refined_pose.csv
optimized_pose.csv
outlier_minimized_pose.csv
```

from multiple sessions.

### Output

```text
configs/motion_profile_default.json
motion_profile_report.md
```

### Proposed Script

```text
scripts/build_motion_profile.py
```

### Proposed Modules

```text
src/dance_pose_recorder/motion_profile.py
src/dance_pose_recorder/temporal_features.py
```

### Motion Profile Content

For each landmark:

```text
median_velocity
p95_velocity
p99_velocity
median_acceleration
p95_acceleration
p99_acceleration
median_jerk
p95_jerk
p99_jerk
max_safe_gap_fill_sec
```

For each bone:

```text
median_length
p01_length
p99_length
safe_min_ratio
safe_max_ratio
```

For each quality group:

```text
recommended_filter_strength
recommended_visibility_threshold
recommended_presence_threshold
```

### Use

Future cleaning/outlier minimization can read:

```text
--motion-profile configs/motion_profile_default.json
```

and adapt thresholds by session type.

### Important Distinction

This is not generated motion.

It is a statistical prior used to guide outlier detection and safe interpolation thresholds.

## Priority 3 - Blender Importer

Blender importer should read the final selected layer:

```text
outlier_minimized_pose.csv
or
refined_pose.csv
```

and display uncertainty.

Required display policy:

```text
measured: solid
refined_measured: green
optimized_constrained: purple
outlier_minimized: cyan
interpolated_short_gap: yellow dotted
interpolated_outlier_removed: blue dotted
estimated_occluded_arm: translucent
low_visibility_leg_kept: dim
unreliable: hidden by default
review_only: hidden or translucent
trajectory_break: break line connection
generated_motion: separate color/layer only
```

## Deferred Research Backends

Do not add these to the core lightweight pipeline yet:

```text
VideoPose3D
PoseFormerV2
MotionBERT
HuMoR
WHAM
4DHumans
GVHMR
motion diffusion
```

These can be explored later as `research_backends/`.

Their outputs must be treated as candidates or generated layers, not measured data.

## Recommended Next Implementation Order

```text
1. Keep Skeleton Optimizer as optional diagnostic layer
2. Limit crop refinement to selected short/mixed problem segments
3. Implement Outlier Minimizer v2
4. Use screen-bottom-origin trajectory export for Blender/TouchDesigner points and segments
5. Add Motion Profile Builder for lightweight statistical prior
6. Consider learned or generated motion backends only as separate research modules
```

## Core Principle

Do not make uncertain motion look certain.

The project should prioritize trustworthy, inspectable, lightweight pose data over visually plausible but unmarked reconstruction.
