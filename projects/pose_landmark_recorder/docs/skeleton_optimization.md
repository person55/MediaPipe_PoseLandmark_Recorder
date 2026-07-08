# Skeleton Optimization

Skeleton optimization is an optional post-refinement step.

It does not generate new motion. It checks whether the refined pose sequence is structurally plausible under conservative human skeleton constraints.

## Pipeline

```text
raw_pose.csv
-> cleaned_pose.csv
-> refined_pose.csv
-> optimized_pose.csv
-> Blender / downstream visualization
```

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
