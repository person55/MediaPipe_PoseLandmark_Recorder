# Current State

Last updated: 2026-07-21

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
crop_accept_score_margin: 0.04
segment_refine_accept_score_margin: 0.05
temporal_scoring: per-frame rate (anchor distance normalized by frame gap)
crop_z_restore: z scaled by bbox.w / frame_width
outlier_spike_scale: max(median + 1.4826 * MAD, floor)
outlier_spike_floors: velocity 0.48 m/s / acceleration 11.5 m/s^2 / jerk 414 m/s^3 (converted to per-frame units by metadata fps; 23.976fps equivalents of the Loop 2 per-frame values)
outlier_accel_jerk: vector 2nd/3rd difference norms
outlier_echo_trimming: true (velocity-spike runs only)
outlier_baseline_excludes: interpolated_short_gap
outlier_correctable_hips: true
outlier_sync_sources: true (pose_world decisions propagated to pose source)
trajectory_coordinate_mode: screen_bottom_origin
trajectory_origin: screen bottom center (x=0.5, y=1.0)
trajectory_apply_aspect_ratio: true (screen_width_scale = height_scale * W/H)
blender_auto_import_camera: location 0,-5,3.4 / rotation 90,0,0
blender_auto_import_scale: x_factor 1.0 (aspect applied at export) / y_factor 0.36
blender_auto_import_scene_reset: load fresh startup scene, remove default Cube, then import CSV trajectory
blender_metadata_labels: hidden by default, optional camera lower-left summary
macos_gpu_sandbox_policy: crop/refine/Blender stages may require sandbox escalation for GPU/Metal context
```

## Claude Loop improvements (2026-07-19, branch feat/claude-loop-pose-landmark-improvement)

Three verified loops fixed stage-wiring defects found in the structural review. Full records: `docs/claude_loop_progress.md` and the Notion page "Claude Loop — Pose Landmark 개선".

- Loop 1: outlier minimizer decisions (corrections/breaks) now propagate to the `pose` source that export reads (`sync_sources`); export applies the 16:9 aspect ratio (X/Z span ratio increase exactly 1.778).
- Loop 2: spike thresholds rebuilt as median + 1.4826*MAD with absolute floors, vector-based acceleration/jerk, echo trimming, hips correctable. False spike breaks on hip/ankle reduced 90-97% while real glitches (5-7 m/s) stayed detected; confirmed by visual frame comparison.
- Loop 3: temporal scoring normalized to per-frame rate so re-detected candidates compete fairly with interpolation; crop candidate z restored to frame scale; margins recalibrated (crop 0.04, full-frame 0.05). First crop acceptances occurred (session 006: 11, all in target categories).
- Loop 4 (2026-07-20): spike floors redefined in physical per-second units (0.48 m/s / 11.5 m/s^2 / 414 m/s^3) and converted to per-frame values by metadata fps, making spike judgments fps-invariant. Baseline decisions preserved (007_v2 byte-identical; 006_v2 4 borderline rows of 219,384).
- Loops 8-10 (2026-07-20): crop re-detection gained rotation-augmented candidates for inverted poses, a CLAHE low-light rescue pass, and mirror/reverse passes — all real detections competing under the unchanged scorer/guards on isolated per-pass trackers (zero regression; acceptances 11→1,284 on 006_v2 and 0→789 on 007_v2 with full crop_pass/rotation/enhanced provenance). Forward/mirror disagreement now feeds `crop_confusion_diagnostics.csv` (target-switch indicator; flags the known cartwheel confusion, metadata only). Codex P0 items done: export-integrity regression tests, per-session reproducibility manifest, overlay review of the 267 changed points / 662 removed segments (all judged correct).
- Loop 7 (2026-07-20): One-Euro visualization smoothing layer at export — separate `*_smooth` columns (raw kept byte-identical), filters reset at breaks/gaps, importer prefers smoothed. Depth jitter −88%; fast-motion lag ~1-2 frames. Defaults: x/z 1.2Hz/beta 1.5, depth 0.4Hz/beta 0.4.
- Loop 6 (2026-07-20): export depth sign corrected (`blender_y = z`, verified against MediaPipe convention and video frames) and the Blender importer now consumes the exporter's trajectory_alpha/trajectory_width fade policy (0.1 buckets, tiered curve objects) instead of rendering everything at fixed alpha. Loop 5 (holdout) is on hold until third-video footage exists.
- Cleanup: `quality_flags.py` and `stage_schema.py` single sources; merge/accept logic moved from scripts to `src/crop_apply.py` / `src/refine_apply.py` with contract tests (123 tests passing).
- Final integrated rerun (`session_cpu_006_v2` / `session_cpu_007_v2`) reproduced all loop numbers end to end and produced `.blend` files.
- v3 integrated rerun (2026-07-20): Loop 8-10 crop outputs carried through refine→outlier→export→Blender (`refined_v3`, `outlier_minimized_v3`, `trajectory_export_v3`, `blender_*_v3.blend` on both sessions). Full-frame refine acceptances fell naturally (76→28 / 93→32) as the crop stage improved; spike segments rose slightly (664→676 / 308→324, mostly handled as corrected boundary transitions); final connected export segments increased (58,408→58,657 / 25,434→25,591). Of the 412 accepted crop pose rows exposed to export, 93% (382) reach the final output with coordinates intact; the rest were legitimately re-competed downstream.
- Stratified positional-accuracy verification (Codex P1, 2026-07-20): cross-pass agreement over all acceptances — 007 median nearest-independent-pass distance 0.0086 (≈16px) with 77% within 0.02; 006 median 0.0233 with 45% agreement (hard inverted/confusion segments). Both sessions' accepted coordinates sit closer to independent detections than the cleaned values they replaced. All 12 stratified visual samples per session were on-body and valid; zero off-body or wrong-side acceptances observed.

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

Latest verified sessions (Windows 11, CPU delegate, Blender 5.2), v3 layers: `session_cpu_006_v2` (3,324 frames, crop accept 1,284, full-frame accept 28, spike segments 676, export segments 58,657, `blender_session_cpu_006_v2_trajectory` v3 .blend) and `session_cpu_007_v2` (1,570 frames, crop accept 789, full-frame accept 32, spike segments 324, export segments 25,591, v3 .blend). All layers row-consistent (219,384 / 103,620). Earlier v2 layers (crop accept 11/0, full-frame 76/93) remain on disk as the pre-Loop-8-10 baseline.

## Earlier build reference

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

Next (validation-first order): holdout validation on a third video (awaiting footage) — margins/floors, smoothing parameters, and the 006 low-agreement acceptances all get re-checked there. Then: Motion Profile Builder, persistent Blender/TouchDesigner importer, per-frame marker/halo fade. Done as of 2026-07-20: v3 integrated rerun, stratified accuracy verification (Codex P1), target-switch diagnostics (Loop 10).

## Known limitations

No Hand Landmarker, persistent Blender add-on, TouchDesigner importer, learned temporal prior, or generated motion layer yet. Long occlusion and frame-out regions are not reconstructable in the current lightweight pipeline.

Open verification reservations (updated 2026-07-21):

- margins/floors were derived from the same two sessions used for validation (no holdout yet)
- positional accuracy of accepted re-detections now has cross-pass agreement metrics plus stratified visual samples (Codex P1, 2026-07-20), but 006's 45% cross-pass agreement means some acceptances sit in segments where independent detections scatter — re-examine alongside the holdout
- animated markers/halos still render at fixed alpha (only trails consume the fade policy since Loop 6); frame-level body depth remains an importer heuristic by design (pose_z carries only hip-relative local depth)
