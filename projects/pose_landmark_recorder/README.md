# MediaPipe Pose Landmark Recorder

Python 3.11 tool for extracting MediaPipe Tasks Pose Landmarker data from a video file and writing structured JSONL, CSV, and metadata outputs.

This project lives under `projects/pose_landmark_recorder/` so the upstream MediaPipe fork remains mostly unchanged.

## Setup

```bash
cd projects/pose_landmark_recorder
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

After editable install, the full pipeline command is available as:

```bash
pose-landmark-pipeline --help
```

To create a local PyInstaller build:

```bash
python scripts/build_pipeline_app.py
```

The build output is written to `dist/pose-landmark-pipeline/pose-landmark-pipeline`.

### Run the standalone executable

`build_pipeline_app.py` creates a PyInstaller `onedir` build. Keep the complete
`dist/pose-landmark-pipeline/` directory together when copying or distributing
the app; do not move only the `pose-landmark-pipeline` executable because it
uses the adjacent `_internal/` directory for MediaPipe, the pipeline scripts,
and the bundled pose model.

Run the executable from the project directory, or use absolute paths for the
input and output locations:

```bash
./dist/pose-landmark-pipeline/pose-landmark-pipeline --help

./dist/pose-landmark-pipeline/pose-landmark-pipeline \
  --input-video /absolute/path/to/dance_take.mp4 \
  --output-root /absolute/path/to/output \
  --delegate gpu \
  --blender-mode background
```

The build bundles `models/pose_landmarker.task`, so no separate `--model`
argument is needed for the default model. Blender remains an external
dependency. Use `--blender-mode skip` when only the pose and trajectory CSV
outputs are needed, or pass `--blender-bin /path/to/Blender` for a non-default
Blender installation. Pipeline logs are written to
`<output-root>/<session-id>/pipeline_logs/`; use `--continue-on-existing` to
resume a prior session without rerunning completed stages.

The default MediaPipe Pose Landmarker Full model is tracked at:

```plain
models/pose_landmarker.task
```

It is ready after a fresh clone. See [`models/README.md`](models/README.md) for
the exact model version, source, license, and checksum. Pass `--model` only
when using a compatible custom model.

## File acquisition order

The main local workflow is:

```text
1. raw data
2. cleaned data
3. crop-refined data
4. refined data
5. Blender CSV data: add trajectory display policy, export trajectory CSV
6. Blender handoff: open Blender, adjust by hand, save .blend
```

Trajectory export belongs after refined data because it creates the CSV files that Blender imports. The exporter requires trajectory display columns, so run outlier minimization first to add `trajectory_visible`, `trajectory_connect`, `trajectory_alpha`, `trajectory_width`, and `trajectory_reason`.

## Full Pipeline Runner

For end-to-end local tests, use the installed runner:

```bash
pose-landmark-pipeline \
  --input-video examples/input/dance_take_006.mp4 \
  --delegate gpu \
  --blender-mode background
```

The legacy script path is kept as a compatibility wrapper:

```bash
python scripts/run_full_pipeline.py \
  --input-video examples/input/dance_take_006.mp4 \
  --delegate gpu \
  --blender-mode background
```

The runner executes the same stages documented below and writes per-stage logs under `examples/output/<session_id>/pipeline_logs/`. On macOS, `crop_refine_pose.py`, `refine_pose_segments.py`, and Blender creation may need GPU/Metal context access outside the Codex sandbox. Project code cannot elevate itself; if a stage fails with `kGpuService`, `Could not create an NSOpenGLPixelFormat`, or `DrishtiMetalHelper`, rerun the failed command outside the sandbox or ask Codex to rerun that stage with escalation.

## 1. Raw Data

```bash
python scripts/record_from_video.py \
  --input examples/input/dance_take_001.mp4 \
  --output examples/output/session_gpu_001 \
  --delegate gpu \
  --origin first_frame_pelvis \
  --save-jsonl \
  --save-csv \
  --save-preview
```

Use `--delegate gpu` on macOS to run through the TensorFlow Lite Metal delegate. The recorder uses `SRGBA` input frames for the GPU path because MediaPipe's macOS GPU conversion path does not accept `SRGB` image frames.

Outputs:

- `raw/raw_pose.jsonl`
- `raw/raw_pose.csv`
- `raw/raw_metadata.json`
- `raw/raw_preview.mp4` when `--save-preview` is used

Most downstream scripts accept the session root as `--output` and automatically write into their stage folder. For example, `--output examples/output/session_gpu_001` writes cleaned files to `examples/output/session_gpu_001/cleaned/`. Explicit experiment folders such as `cleaned_test_v1` or `crop_refine_v1` are still respected when their names start with the stage prefix.

## 2. Cleaned Data

Raw extraction and pose cleaning are separate. `record_from_video.py` preserves MediaPipe's direct measurements in `raw/raw_pose.csv` and `raw/raw_pose.jsonl`. After reviewing `raw/raw_preview.mp4`, run `clean_pose_data.py` to mark unstable landmarks, interpolate only short gaps, smooth valid runs, and write corrected outputs under a separate `cleaned/` folder.

```bash
python scripts/clean_pose_data.py \
  --input-video examples/input/dance_take_001.mp4 \
  --input-csv examples/output/session_gpu_001/raw/raw_pose.csv \
  --input-jsonl examples/output/session_gpu_001/raw/raw_pose.jsonl \
  --metadata examples/output/session_gpu_001/raw/raw_metadata.json \
  --output examples/output/session_gpu_001 \
  --max-interpolate-gap 15 \
  --visibility-threshold 0.5 \
  --presence-threshold 0.5 \
  --jump-threshold-multiplier 6.0 \
  --smoothing-window 7 \
  --save-csv \
  --save-jsonl \
  --save-preview
```

Cleaned outputs:

- `cleaned_pose.csv`
- `cleaned_pose.jsonl`
- `cleaned_frame_status.csv`
- `cleaned_quality_report.json`
- `cleaned_interpolation_report.json`
- `cleaned_corrected_preview.mp4` when `--save-preview` is used

Long missing ranges are not interpolated. Short gaps up to `--max-interpolate-gap` are linearly interpolated and marked with `quality_flag` so downstream tools can distinguish measured values from corrected values.

See [`docs/quality_flags.md`](docs/quality_flags.md) for the meaning of each `quality_flag` and suggested downstream visualization behavior.

## 3. Crop-Refined Data

Crop refinement is the post-cleaning step before full-frame segment refinement. It detects problematic cleaned segments, creates torso-centered person crops from the original video, runs MediaPipe again on those crops, restores crop coordinates to the original frame, and accepts only crop candidates that score better than cleaned values.

The current crop baseline keeps the torso as the crop center and uses hands/feet only to expand crop size. It does not generate motion or fill long missing ranges. By default, it only attempts selected `mixed_problem_segment` ranges and skips `review_only`, `missing_long_gap`, and overlong segments.

```bash
python scripts/crop_refine_pose.py \
  --input-video examples/input/dance_take_001.mp4 \
  --input-cleaned-csv examples/output/session_gpu_001/cleaned/cleaned_pose.csv \
  --metadata examples/output/session_gpu_001/raw/raw_metadata.json \
  --quality-report examples/output/session_gpu_001/cleaned/cleaned_quality_report.json \
  --output examples/output/session_gpu_001 \
  --model models/pose_landmarker.task \
  --delegate gpu \
  --running-mode video \
  --crop-source torso \
  --target-flags unreliable,interpolated_outlier_removed,estimated_occluded_arm \
  --target-segment-types mixed_problem_segment \
  --max-segment-length 100 \
  --segment-margin 12 \
  --accept-score-margin 0.06 \
  --save-candidates \
  --save-refined \
  --save-report \
  --save-jsonl \
  --save-preview \
  --save-debug-images
```

Default crop settings:

```text
crop_margin_ratio: 1.65
full_body_margin_ratio: 1.45
crop_min_size: 480 px
target_segment_types: mixed_problem_segment
max_segment_length: 100 frames
```

Crop outputs:

- `crop_refine_pose.csv`
- `crop_refine_pose.jsonl`
- `crop_refine_report.json`
- `crop_refine_candidates.csv`
- `crop_refine_candidate_scores.csv`
- `crop_refine_segments.csv`
- `crop_refine_preview.mp4`
- `crop_refine_debug_images/`

See [`docs/crop_refinement.md`](docs/crop_refinement.md) for the crop-based post-cleaning refinement workflow.

## 4. Refined Data

Full-frame segment refinement runs after crop refinement. Use `crop_refine_pose.csv` as the input cleaned CSV so crop-accepted rows remain part of the downstream scoring context.

```bash
python scripts/refine_pose_segments.py \
  --input-video examples/input/dance_take_001.mp4 \
  --input-cleaned-csv examples/output/session_gpu_001/crop_refine/crop_refine_pose.csv \
  --input-raw-csv examples/output/session_gpu_001/raw/raw_pose.csv \
  --metadata examples/output/session_gpu_001/raw/raw_metadata.json \
  --frame-status examples/output/session_gpu_001/cleaned/cleaned_frame_status.csv \
  --quality-report examples/output/session_gpu_001/cleaned/cleaned_quality_report.json \
  --output examples/output/session_gpu_001/refined_after_crop_v1 \
  --delegate gpu \
  --target-landmarks arms,hands,feet \
  --min-cluster-length 2 \
  --max-cluster-length 90 \
  --segment-margin 12 \
  --accept-score-margin 0.08 \
  --save-csv \
  --save-jsonl \
  --save-preview
```

Refined outputs:

- `refined_after_crop_v1/refined_pose.csv`
- `refined_after_crop_v1/refined_pose.jsonl`
- `refined_after_crop_v1/refine_report.json`
- `refined_after_crop_v1/refined_preview.mp4` when `--save-preview` is used

See [`docs/segment_refinement.md`](docs/segment_refinement.md) for the optional second-pass segment re-detection workflow.

See [`docs/skeleton_optimization.md`](docs/skeleton_optimization.md) for the optional skeleton constraint and optimization workflow.

## 5. Blender CSV Data

Before opening Blender, write trajectory display policy and export Blender-ready CSV files from the refined pose data.

### Outlier Minimization

Outlier minimization is the visualization-oriented step after full-frame refinement. It keeps all rows, corrects only short temporal spikes when stable neighbors exist, and writes trajectory display columns so Blender or TouchDesigner can avoid drawing false lines through unreliable coordinates.

```bash
python scripts/minimize_pose_outliers.py \
  --input-pose-csv examples/output/session_gpu_001/refined_after_crop_v1/refined_pose.csv \
  --metadata examples/output/session_gpu_001/raw/raw_metadata.json \
  --crop-refine-report examples/output/session_gpu_001/crop_refine/crop_refine_report.json \
  --output examples/output/session_gpu_001 \
  --source pose_world \
  --position-fields tx,ty,tz \
  --max-correction-gap-sec 0.12 \
  --max-break-gap-sec 0.20 \
  --velocity-threshold-multiplier 6.0 \
  --acceleration-threshold-multiplier 6.0 \
  --jerk-threshold-multiplier 8.0 \
  --min-stable-neighbors 2 \
  --preserve-quality-flags \
  --save-csv \
  --save-report \
  --save-trajectory-breaks
```

Outlier minimization outputs:

- `outlier_minimized_pose.csv`
- `outlier_minimized_report.json`
- `outlier_minimized_temporal_spike_report.csv`
- `outlier_minimized_trajectory_breaks.csv`

See [`docs/outlier_minimization.md`](docs/outlier_minimization.md) for the visualization-oriented outlier minimization workflow.

### Trajectory Export

Trajectory export converts `outlier_minimized_pose.csv` into Blender/TouchDesigner-ready points and line segments. It does not correct pose data; it only translates existing trajectory visibility/connect policy into export files.

```bash
python scripts/export_trajectory.py \
  --input-pose-csv examples/output/session_gpu_001/outlier_minimized/outlier_minimized_pose.csv \
  --metadata examples/output/session_gpu_001/raw/raw_metadata.json \
  --output examples/output/session_gpu_001 \
  --coordinate-mode screen_bottom_origin \
  --source pose \
  --depth-mode pose_z \
  --landmark-preset blender_default \
  --screen-origin-x 0.5 \
  --screen-origin-y 1.0 \
  --screen-width-scale 6.0 \
  --screen-height-scale 6.0 \
  --depth-scale 1.0 \
  --save-points \
  --save-segments \
  --save-report
```

Default export policy:

```text
coordinate_mode: screen_bottom_origin
screen origin: x=0.5, y=1.0
head proxy: nose
excluded: ears, hand index, thumb
included: left_foot_index, right_foot_index
```

Trajectory export outputs:

- `trajectory_export_points.csv`
- `trajectory_export_segments.csv`
- `trajectory_export_report.json`

## 6. Open Blender and Save

Open the exported trajectory directly in Blender:

```bash
python scripts/open_blender_trajectory.py \
  --trajectory-dir examples/output/session_gpu_001/trajectory_export
```

The Blender importer defaults to the current fixed-camera visualization setup:

```text
scene: fresh Blender startup scene with the default Cube removed before CSV import
camera: location 0,-5,3.4 / rotation 90,0,0
x_factor: 2.2
y_factor: 0.36
depth: approximate 2D body-scale depth, displayed as about 0-1.8m
markers: small emissive cores with smaller high-emission halo spheres
left/right: left landmarks orange, right landmarks cyan
face markers: white nose, normal-size left/right eyes, inner/outer eyes and mouth hidden
paused: full overview trails
playing: progressive draw trails
metadata labels: hidden by default
```

Use `--show-camera-summary` only when a compact session summary should appear in the lower-left camera view.

The script saves the generated `.blend` file under `blender/blender_<session_id>_trajectory.blend` by default. After Blender opens, adjust camera, materials, timing, and visibility by hand, then save the `.blend` from Blender. Video rendering is intentionally left out of the default workflow; the handoff point is a saved Blender scene.

See [`docs/trajectory_export.md`](docs/trajectory_export.md) for Blender/TouchDesigner trajectory export.

## Optional Skeleton Optimization

Skeleton optimization is a diagnostic option after `refined_pose.csv`. It checks bone length, joint angle, reachability, and temporal jumps under conservative skeleton constraints. It does not replace the main visualization path and should not be used to fill long missing motion.

```bash
python scripts/optimize_pose_skeleton.py \
  --input-refined-csv examples/output/session_gpu_001/refined_after_crop_v1/refined_pose.csv \
  --metadata examples/output/session_gpu_001/raw/raw_metadata.json \
  --refine-report examples/output/session_gpu_001/refined_after_crop_v1/refine_report.json \
  --output examples/output/session_gpu_001/skeleton_optimization_v1 \
  --constraints configs/skeleton_constraints.yaml \
  --source pose_world \
  --max-correction-gap-sec 0.10 \
  --max-review-gap-sec 1.50 \
  --adaptive-percentile-low 1.0 \
  --adaptive-percentile-high 99.0 \
  --adaptive-margin-deg 10.0 \
  --bone-length-min-ratio 0.45 \
  --bone-length-max-ratio 1.75 \
  --reachability-margin-ratio 0.10 \
  --temporal-jump-multiplier 6.0 \
  --save-csv \
  --save-jsonl \
  --save-reports
```

Skeleton optimization outputs:

- `skeleton_optimization_v1/optimized_pose.csv`
- `skeleton_optimization_v1/optimized_pose.jsonl`
- `skeleton_optimization_v1/optimization_report.json`
- `skeleton_optimization_v1/bone_length_report.csv`
- `skeleton_optimization_v1/joint_angle_report.csv`
- `skeleton_optimization_v1/optimization_segments.csv`

## Recommended pipeline

### Visualization-first path

```text
record_from_video.py
-> clean_pose_data.py
-> crop_refine_pose.py
-> refine_pose_segments.py
-> minimize_pose_outliers.py
-> export_trajectory.py
-> open_blender_trajectory.py
-> adjust and save in Blender
```

### Optional diagnostic path

```text
refine_pose_segments.py
-> optimize_pose_skeleton.py
-> optimization reports
```

Skeleton optimization is useful for diagnostics, but it is not the default final visualization layer.
For current visualization-first work, prefer `refined_pose.csv` followed by `outlier_minimized_pose.csv`.

## Current cleaning baseline

The current baseline preset is conservative. It keeps raw extraction and cleaned data separate, interpolates short missing-frame gaps, and avoids turning long or uncertain outlier runs into plausible-looking motion.

Recommended baseline options:

```bash
python scripts/clean_pose_data.py \
  --input-video examples/input/dance_take_002.mov \
  --input-csv examples/output/session_gpu_002/raw/raw_pose.csv \
  --input-jsonl examples/output/session_gpu_002/raw/raw_pose.jsonl \
  --metadata examples/output/session_gpu_002/raw/raw_metadata.json \
  --output examples/output/session_gpu_002/cleaned \
  --max-interpolate-gap 15 \
  --visibility-threshold 0.5 \
  --presence-threshold 0.5 \
  --jump-threshold-multiplier 6.0 \
  --smoothing-window 7 \
  --outlier-max-gap 3 \
  --arm-occlusion-max-gap 55 \
  --leg-salvage-min-visibility 0.15 \
  --save-csv \
  --save-jsonl \
  --save-preview
```

Baseline policy:

- Missing-frame gaps up to 15 frames are linearly interpolated.
- Recoverable spike-like outliers are interpolated only up to 3 frames.
- Longer outlier runs are left as `unreliable`.
- Bounded elbow/wrist occlusion runs can be estimated as `estimated_occluded_arm`.
- Stable low-visibility leg measurements can be kept as `low_visibility_leg_kept`.
- Raw files are never overwritten.

MediaPipe `pose_world_landmarks` are model-estimated 3D coordinates from a single RGB source. They are useful for trajectory and visual experiments, but should not be treated as calibrated stage coordinates.

## Documentation

This project uses lightweight documentation files to reduce Codex CLI context usage.

Recommended context files:

- [`AGENTS.md`](AGENTS.md)
- [`CURRENT_STATE.md`](CURRENT_STATE.md)
- [`docs/codex_context_min.md`](docs/codex_context_min.md)
- [`docs/README.md`](docs/README.md)

For next development priorities, see:

- [`docs/next_development_plan.md`](docs/next_development_plan.md)

Long notes and archived planning documents should live under `docs/archive/` and should not be read by Codex unless explicitly needed.
