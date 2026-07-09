# Current State

Last updated: 2026-07-09

## Project

MediaPipe Pose Landmark Recorder is a Python 3.11 project under `projects/pose_landmark_recorder/` inside a MediaPipe fork. It extracts single-person pose landmarks, writes CSV/JSONL/metadata, renders previews, cleans unstable landmarks, adds crop-based refinement, and keeps skeleton optimization as an optional diagnostic layer.

## Implemented scripts

```text
scripts/record_from_video.py
scripts/clean_pose_data.py
scripts/crop_refine_pose.py
scripts/refine_pose_segments.py
scripts/minimize_pose_outliers.py
scripts/optimize_pose_skeleton.py
```

## Main outputs

`raw_pose`, `cleaned_pose`, `crop_refined_pose`, `refined_pose`, `outlier_minimized_pose`, and optional diagnostic `optimized_pose` CSV/JSONL files, plus `metadata.json`, `quality_report.json`, `interpolation_report.json`, `crop_refine_report.json`, `refine_report.json`, `outlier_report.json`, and optimization diagnostic reports.

## Current decision

The project is visualization-oriented. Skeleton optimization is not the default final output path; it remains useful as a diagnostic layer because conservative flags can make pose_world skeletons look incomplete when hidden downstream.

```text
raw_pose
-> cleaned_pose
-> crop_refined_pose
-> outlier_minimized_pose
-> trajectory_export planned
-> Blender
```

Optional diagnostic path: `refined_pose -> optimize_pose_skeleton.py -> optimization_report / diagnostic overlay`.

## Interpretation

- `refined_pose.csv` is better for visual continuity; `optimized_pose.csv` is better for diagnostics.
- Crop refinement is the preferred lightweight improvement after cleaning.
- Full-frame segment refinement remains available, but the visualization-first path now moves from `crop_refined_pose.csv` to `outlier_minimized_pose.csv`.
- Skeleton optimization should not be strengthened as the main recovery method unless the project moves toward learned or generated reconstruction.
## Current baseline

```text
max_interpolate_gap: 15 frames
outlier_max_gap: 3 frames
arm_occlusion_max_gap: 55 frames
leg_low_visibility_salvage_enabled: true
leg_salvage_min_visibility: 0.15
smoothing_window: 7
crop_margin_ratio: 1.65
crop_full_body_margin_ratio: 1.45
crop_min_size: 480 px
crop_target_segment_types: mixed_problem_segment
crop_max_segment_length: 100 frames
crop_accept_score_margin: 0.06
segment_refine_accept_score_margin: 0.08
```

## Findings

- Raw detection can be stable, but hands, wrists, elbows, feet, and long occlusion segments remain difficult.
- MediaPipe Full remains the default model. Heavy model testing did not justify adoption for the current lightweight pipeline.
- Torso-centered crop refinement improves some short arm/hand proxy errors without generating motion.
- Crop bbox policy keeps torso as center, uses hands/feet only for size expansion, and avoids applying a full 1.8x margin to the full-body bbox.
- Crop refinement is now restricted by default to selected mixed problem segments; long unreliable runs and `missing_long_gap` are not crop-refined by default.
- Segment re-detection helps short problem regions but cannot recover long unreliable runs without visual evidence.
- Skeleton optimization is useful for diagnostics and conservative flagging, not final visualization.
- Long unreliable runs should remain `review_only`, `unreliable`, or hidden in Blender rather than being filled automatically.

## Latest session reference

`session_gpu_004` current path:

```text
cleaned_arm55_legkeep_outlier3
-> crop_refine_v1
-> refined_after_crop_v1
crop_margin_ratio: 1.65
full_body_margin_ratio: 1.45
crop_min_size: 480 px
average crop width: about 722.5 px
crop_refine_v1 accepted rows: 55
refined_after_crop_v1 accepted rows: 12
final flags before outlier minimization: crop_refined_measured 55, refined_measured 12
```

## Planned next step

Use Outlier Minimizer v2 after `crop_refine_v1`: confidence-aware filtering, velocity/acceleration/jerk outlier detection, landmark-group-specific policies, trajectory break handling, and `outlier_minimized_pose.csv`.

Then build Motion Profile Builder from multiple stable sessions to create `motion_profile.json` for adaptive cleaning and outlier-minimization thresholds.

## Known limitations

No Hand Landmarker, Blender importer, learned temporal prior, or generated motion layer yet. Long occlusion and frame-out regions are not reconstructable in the current lightweight pipeline.
