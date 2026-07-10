# Current State

Last updated: 2026-07-10

## Project

MediaPipe Pose Landmark Recorder is a Python 3.11 project under `projects/pose_landmark_recorder/` inside a MediaPipe fork. It extracts single-person pose landmarks, writes CSV/JSONL/metadata, renders previews, cleans unstable landmarks, adds crop-based refinement, and keeps skeleton optimization as an optional diagnostic layer.

## Implemented scripts

```text
scripts/record_from_video.py
scripts/clean_pose_data.py
scripts/crop_refine_pose.py
scripts/refine_pose_segments.py
scripts/minimize_pose_outliers.py
scripts/export_trajectory.py
scripts/open_blender_trajectory.py
scripts/optimize_pose_skeleton.py
scripts/run_full_pipeline.py
scripts/build_pipeline_app.py
src/dance_pose_recorder/pipeline_runner.py
```

## Main outputs

`raw_pose`, `cleaned_pose`, `crop_refine_pose`, `refined_pose`, `outlier_minimized_pose`, Blender trajectory points/segments, and optional diagnostic `optimized_pose` files, plus related metadata/report files.

## Current decision

The project is visualization-oriented. Skeleton optimization is not the default final output path; it remains useful as a diagnostic layer because conservative flags can make pose_world skeletons look incomplete when hidden downstream.

```text
raw_pose
-> cleaned_pose
-> crop_refine_pose
-> refined_pose
-> outlier_minimized_pose
-> trajectory_export
-> Blender
```

Optional diagnostic path: `refined_pose -> optimize_pose_skeleton.py -> optimization_report / diagnostic overlay`.

## Interpretation

- Visualization-first path: `crop_refine_pose.csv -> refined_pose.csv -> outlier_minimized_pose.csv -> trajectory export`.
- Full-frame segment refinement is part of the current default visualization handoff; skeleton optimization remains diagnostic.

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
trajectory_coordinate_mode: screen_bottom_origin
trajectory_origin: screen bottom center (x=0.5, y=1.0)
blender_auto_import_camera: location 0,-5,3.4 / rotation 90,0,0
blender_auto_import_scale: x_factor 2.2 / y_factor 0.36
blender_auto_import_scene_reset: load fresh startup scene, remove default Cube, then import CSV trajectory
blender_metadata_labels: hidden by default, optional camera lower-left summary
macos_gpu_sandbox_policy: crop/refine/Blender stages may require sandbox escalation for GPU/Metal context
```

## Findings

- Raw detection can be stable, but hands, wrists, elbows, feet, and long occlusion segments remain difficult.
- MediaPipe Full remains the default model. Heavy model testing did not justify adoption for the current lightweight pipeline.
- Torso-centered crop refinement improves some short arm/hand proxy errors without generating motion.
- Crop bbox policy keeps torso as center, uses hands/feet only for size expansion, and avoids applying a full 1.8x margin to the full-body bbox.
- Crop refinement is now restricted by default to selected mixed problem segments; long unreliable runs and `missing_long_gap` are not crop-refined by default.
- Segment re-detection helps short problem regions but cannot recover long unreliable runs without visual evidence.
- `pose-landmark-pipeline` is the installed local end-to-end runner, backed by `src/dance_pose_recorder/pipeline_runner.py`. `scripts/run_full_pipeline.py` remains as a compatibility wrapper. The runner cannot elevate sandbox permissions, but it logs each stage and reports macOS GPU/Metal context failures with the exact command to rerun outside the sandbox.
- `scripts/build_pipeline_app.py` creates the local PyInstaller `dist/pose-landmark-pipeline/pose-landmark-pipeline` build. The packaged runner dispatches Python stages through its own internal stage mode so it does not rely on `sys.executable` being a separate Python interpreter.
- Skeleton optimization is useful for diagnostics and conservative flagging, not final visualization.
- Long unreliable runs should remain `review_only`, `unreliable`, or hidden in Blender rather than being filled automatically.
- Blender default trajectory export excludes ears, hand index, and thumb, keeps `foot_index`, and uses `nose` as the head proxy.
- Blender auto import starts from a fresh Blender startup scene, removes the default `Cube`, and then imports trajectory CSV data.
- Blender auto import now keeps metadata/debug text hidden by default. Use the optional camera summary only when a compact lower-left camera-view label is needed.

## Latest session reference

Latest local PyInstaller build:

```text
dist/pose-landmark-pipeline/pose-landmark-pipeline
dist/pose-landmark-pipeline/_internal/models/pose_landmarker.task
dist/pose-landmark-pipeline/_internal/scripts/record_from_video.py
```

Verified build checks:

```text
pose-landmark-pipeline --help
internal record_from_video.py --help dispatch
session_gpu_006 continue-on-existing pipeline call
```

`session_gpu_006` full pipeline smoke path:

```text
dance_take_006.mp4
-> raw
-> cleaned
-> crop_refine
-> refined_after_crop_v1
-> outlier_minimized
-> trajectory_export
-> blender/blender_session_gpu_006_trajectory.blend
```

`session_gpu_004` earlier reference:

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

## Current export step

Use `export_trajectory.py` after `outlier_minimized`, which is produced from `refined_pose.csv`. Default coordinate mode is `screen_bottom_origin`, using screen bottom center as Blender origin.

Use `open_blender_trajectory.py` to open the exported CSV in Blender and save `blender/blender_<session_id>_trajectory.blend`. The importer resets to a fresh startup scene, deletes the default `Cube`, then imports the trajectory. The current default camera is the fixed `-Y` view, marker/halo visibility is tuned for playback, overview trails are shown while paused, progressive trails draw during playback, and metadata labels are hidden unless `--show-camera-summary` is used.

Next: persistent Blender/TouchDesigner importer and Motion Profile Builder.

## Known limitations

No Hand Landmarker, persistent Blender add-on, TouchDesigner importer, learned temporal prior, or generated motion layer yet. Long occlusion and frame-out regions are not reconstructable in the current lightweight pipeline.
