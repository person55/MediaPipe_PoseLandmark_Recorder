# Current State

Last updated: 2026-07-09

## Project

MediaPipe Pose Landmark Recorder is a Python 3.11 project located under `projects/pose_landmark_recorder/` inside a MediaPipe fork.

The project extracts single-person pose landmarks from video, writes structured CSV/JSONL/metadata outputs, renders preview videos, cleans unstable landmarks, refines problematic segments, and keeps skeleton optimization as an optional diagnostic layer.

## Implemented scripts

```text
scripts/record_from_video.py
scripts/clean_pose_data.py
scripts/refine_pose_segments.py
scripts/optimize_pose_skeleton.py
```

## Main outputs

```text
raw_pose.csv / raw_pose.jsonl
cleaned_pose.csv / cleaned_pose.jsonl
refined_pose.csv / refined_pose.jsonl
optimized_pose.csv / optimized_pose.jsonl
metadata.json
quality_report.json
interpolation_report.json
refine_report.json
optimization_report.json / diagnostic reports
```

## Current decision

The project is currently visualization-oriented.

After comparing `refined_pose.csv` and `optimized_pose.csv` from `session_gpu_003`, skeleton optimization is no longer considered the default final output path for visualization. It remains useful as a diagnostic layer, but it can make pose_world skeletons look incomplete when `review_only` or `optimization_unreliable` rows are hidden downstream.

Default visualization path:

```text
raw_pose
-> cleaned_pose
-> refined_pose
-> outlier_minimized_pose
-> Blender
```

Optional diagnostic path:

```text
refined_pose
-> optimize_pose_skeleton.py
-> optimization_report / diagnostic overlay
```

## Current interpretation

- `refined_pose.csv` is better for visual continuity.
- `optimized_pose.csv` is better for diagnostics and constraint reporting.
- Skeleton optimization should not be strengthened as the main recovery method unless the project explicitly moves toward learned motion reconstruction or strong generated interpolation.
- The next practical step is Outlier Minimizer v2.

## Current baseline

```text
max_interpolate_gap: 15 frames
outlier_max_gap: 3 frames
arm_occlusion_max_gap: 55 frames
leg_low_visibility_salvage_enabled: true
leg_salvage_min_visibility: 0.15
smoothing_window: 7
```

## Current findings

- Raw detection can be stable, but hands, wrists, elbows, feet, and long occlusion segments remain difficult.
- Segment re-detection is useful for short problem regions, but it cannot recover long unreliable runs when the source video lacks reliable visual evidence.
- Skeleton optimization is useful for diagnostics and conservative flagging, but it is not the default final visualization layer.
- Long unreliable runs should remain `review_only`, `unreliable`, or hidden in Blender rather than being filled automatically.
- The next useful step is outlier minimization, not stronger automatic generation.

## Planned next step

Implement Outlier Minimizer v2 after refinement: confidence-aware filtering, velocity/acceleration/jerk outlier detection, landmark-group-specific policies, trajectory break handling, and `outlier_minimized_pose.csv`.

Then build Motion Profile Builder from multiple stable sessions to create `motion_profile.json` for adaptive cleaning and outlier-minimization thresholds.

## Known limitations

- No Hand Landmarker pipeline yet.
- No Blender importer yet.
- No learned temporal prior yet.
- No generated motion layer.
- Long occlusion and frame-out regions are not reconstructable in the current lightweight pipeline.
