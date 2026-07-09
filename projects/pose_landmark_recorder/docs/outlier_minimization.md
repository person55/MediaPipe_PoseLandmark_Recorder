# Outlier Minimization

Outlier minimization is a visualization-oriented post-processing step.

It does not generate missing motion. It detects velocity, acceleration, and jerk spikes, corrects only short temporal spikes, and creates trajectory display policies for Blender or TouchDesigner.

## Pipeline

```text
raw_pose.csv
-> cleaned_pose.csv
-> crop_refined_pose.csv
-> outlier_minimized_pose.csv
-> trajectory_export.csv
```

## Main Purpose

The main goal is not to fill all missing data.

The main goal is to prevent false trajectory lines from connecting through unreliable or spiking coordinates.

## Output Policy

The output keeps all rows and adds display columns:

- `trajectory_visible`
- `trajectory_connect`
- `trajectory_alpha`
- `trajectory_width`
- `trajectory_reason`

Points and lines should be controlled separately.

A point may remain visible while the trajectory line is disconnected.

## Correction Policy

Short spikes may be interpolated only when stable neighbors exist.

Long unreliable regions, `missing_long_gap`, `review_only` regions, and unavailable crop/refine regions are not corrected automatically.

## Spike Features

The v2 minimizer computes:

- velocity
- acceleration
- jerk

Rows are flagged when a landmark exceeds its own reliable median motion by configurable multipliers.

## Recommended Visualization Rule

Use `outlier_minimized_pose.csv` for visualization-first trajectory work.

Do not treat `trajectory_visible=false` and `trajectory_connect=false` as the same thing. Hidden points should disappear, while disconnected visible points can still show uncertain measurements without drawing false lines through them.
