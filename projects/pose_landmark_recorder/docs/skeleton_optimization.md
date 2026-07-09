# Skeleton Optimization

## Current Status

Skeleton optimization is now treated as an optional diagnostic layer, not the default visualization path.

This decision is based on practical testing with `session_gpu_003`, where the optimized output preserved row counts but caused some pose_world skeletons to appear incomplete when `review_only` or `optimization_unreliable` rows were hidden downstream.

## Recommended Role

Use Skeleton Optimizer for:

- bone length diagnostics
- elbow/knee angle diagnostics
- reachability warnings
- temporal jump reports
- review-only segment identification
- diagnostic overlays

Do not use Skeleton Optimizer as the default final visualization layer unless a specific task requires conservative hiding of uncertain data.

For visualization-first workflows, use refined or outlier-minimized pose data as the primary coordinate source. Use skeleton optimization as a diagnostic overlay, not as the default final pose layer.

## Recommended Visualization Policy

If optimized data is used in Blender:

| status | Recommended display |
|---|---|
| `measured` | solid |
| `refined_measured` | solid or green marker |
| `optimized_constrained` | visible diagnostic color |
| `review_only` | translucent, not automatically deleted |
| `optimization_unreliable` | faint point or line break |
| `missing_long_gap` | hidden or explicit gap |
| `unreliable` | optional faint point |

## Purpose

Skeleton optimization is an optional post-refinement diagnostic step.

It does not generate new motion. It checks whether the refined pose sequence is structurally plausible under conservative human skeleton constraints.

## Pipeline

```text
raw_pose.csv
-> cleaned_pose.csv
-> refined_pose.csv
-> optimized_pose.csv
-> optimization reports / diagnostic overlay
```

`optimized_pose.csv` is an optional diagnostic output. For visualization, prefer `refined_pose.csv` or future `outlier_minimized_pose.csv` unless conservative hiding is desired.

## What It Checks

- bone length consistency
- elbow/knee angle range
- arm/leg reachability
- temporal jump
- long unreliable or review-only regions

## What It Does Not Do

- It does not fill long missing ranges.
- It does not generate motion.
- It does not replace raw, cleaned, or refined files.
- It does not apply strict medical ROM limits to dance motion.

## Policy

The optimizer uses hard guards only for clearly impossible cases.
For dance motion, most constraints are soft penalties and flags.

Short temporal violations may be corrected by interpolation when stable neighboring frames exist.
Long unreliable runs remain review-only.

## Recommended Blender Display

| quality_flag | Display |
|---|---|
| `measured` | solid |
| `refined_measured` | green marker |
| `optimized_constrained` | purple marker |
| `interpolated_short_gap` | yellow dotted |
| `interpolated_outlier_removed` | blue dotted |
| `estimated_occluded_arm` | translucent |
| `low_visibility_leg_kept` | dim |
| `unreliable` | hidden by default |
| `review_only` | hidden or translucent |

## Outputs

- `optimized_pose.csv`
- `optimized_pose.jsonl`
- `optimization_report.json`
- `bone_length_report.csv`
- `joint_angle_report.csv`
- `optimization_segments.csv`

`optimized_constrained` is only used when the optimizer changes coordinates. Flag-only rows keep their original `quality_flag` and use `optimizer_status` / `optimizer_reason` for downstream review.

## Limitation

Skeleton optimization should not be expected to reconstruct long occlusion or missing motion.

If visual continuity is the goal, use Outlier Minimizer v2 after refinement.
If data reliability review is the goal, use Skeleton Optimizer reports.
