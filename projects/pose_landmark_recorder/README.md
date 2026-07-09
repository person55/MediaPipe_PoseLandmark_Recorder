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

Place a compatible MediaPipe Pose Landmarker model at:

```plain text
models/pose_landmarker.task
```

## Run

```bash
python scripts/record_from_video.py \
  --input examples/input/dance_take_001.mp4 \
  --output examples/output/session_001 \
  --delegate cpu \
  --origin first_frame_pelvis \
  --save-jsonl \
  --save-csv \
  --save-preview
```

Use `--delegate gpu` on macOS to run through the TensorFlow Lite Metal delegate. The recorder uses `SRGBA` input frames for the GPU path because MediaPipe's macOS GPU conversion path does not accept `SRGB` image frames.

Outputs:

- `raw_pose.jsonl`
- `raw_pose.csv`
- `metadata.json`
- `preview.mp4` when `--save-preview` is used

## Clean

Raw extraction and pose cleaning are separate. `record_from_video.py` preserves MediaPipe's direct measurements in `raw_pose.csv` and `raw_pose.jsonl`. After reviewing `preview.mp4`, run `clean_pose_data.py` to mark unstable landmarks, interpolate only short gaps, smooth valid runs, and write corrected outputs under a separate `cleaned/` folder.

```bash
python scripts/clean_pose_data.py \
  --input-video examples/input/dance_take_001.mp4 \
  --input-csv examples/output/session_gpu_001/raw_pose.csv \
  --input-jsonl examples/output/session_gpu_001/raw_pose.jsonl \
  --metadata examples/output/session_gpu_001/metadata.json \
  --output examples/output/session_gpu_001/cleaned \
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
- `frame_status.csv`
- `quality_report.json`
- `interpolation_report.json`
- `corrected_preview.mp4` when `--save-preview` is used

Long missing ranges are not interpolated. Short gaps up to `--max-interpolate-gap` are linearly interpolated and marked with `quality_flag` so downstream tools can distinguish measured values from corrected values.

See [`docs/quality_flags.md`](docs/quality_flags.md) for the meaning of each `quality_flag` and suggested downstream visualization behavior.

## Crop refine

Crop refinement is an optional post-cleaning step before full-frame segment refinement. It detects problematic cleaned segments, creates torso-centered person crops from the original video, runs MediaPipe again on those crops, restores crop coordinates to the original frame, and accepts only crop candidates that score better than cleaned values.

The current crop baseline keeps the torso as the crop center and uses hands/feet only to expand crop size. It does not generate motion or fill long missing ranges. By default, it only attempts selected `mixed_problem_segment` ranges and skips `review_only`, `missing_long_gap`, and overlong segments.

```bash
python scripts/crop_refine_pose.py \
  --input-video examples/input/dance_take_004.mp4 \
  --input-cleaned-csv examples/output/session_gpu_004/cleaned_arm55_legkeep_outlier3/cleaned_pose.csv \
  --metadata examples/output/session_gpu_004/metadata.json \
  --quality-report examples/output/session_gpu_004/cleaned_arm55_legkeep_outlier3/quality_report.json \
  --output examples/output/session_gpu_004/crop_refine_v1 \
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

- `crop_refined_pose.csv`
- `crop_refined_pose.jsonl`
- `crop_refine_report.json`
- `crop_candidates.csv`
- `crop_candidate_scores.csv`
- `crop_segments.csv`
- `crop_refined_preview.mp4`
- `crop_debug_images/`

See [`docs/crop_refinement.md`](docs/crop_refinement.md) for the crop-based post-cleaning refinement workflow.

## Outlier minimization

Outlier minimization is the visualization-oriented step after crop refinement. It keeps all rows, corrects only short temporal spikes when stable neighbors exist, and writes trajectory display columns so Blender or TouchDesigner can avoid drawing false lines through unreliable coordinates.

```bash
python scripts/minimize_pose_outliers.py \
  --input-pose-csv examples/output/session_gpu_004/crop_refine_v1/crop_refined_pose.csv \
  --metadata examples/output/session_gpu_004/metadata.json \
  --crop-refine-report examples/output/session_gpu_004/crop_refine_v1/crop_refine_report.json \
  --output examples/output/session_gpu_004/outlier_minimized_v1 \
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
- `outlier_report.json`
- `temporal_spike_report.csv`
- `trajectory_breaks.csv`

See [`docs/outlier_minimization.md`](docs/outlier_minimization.md) for the visualization-oriented outlier minimization workflow.

## Segment refine

Full-frame segment refinement can be run after crop refinement. Use `crop_refined_pose.csv` as the input cleaned CSV so crop-accepted rows remain part of the downstream scoring context.

```bash
python scripts/refine_pose_segments.py \
  --input-video examples/input/dance_take_004.mp4 \
  --input-cleaned-csv examples/output/session_gpu_004/crop_refine_v1/crop_refined_pose.csv \
  --input-raw-csv examples/output/session_gpu_004/raw_pose.csv \
  --metadata examples/output/session_gpu_004/metadata.json \
  --frame-status examples/output/session_gpu_004/cleaned_arm55_legkeep_outlier3/frame_status.csv \
  --quality-report examples/output/session_gpu_004/cleaned_arm55_legkeep_outlier3/quality_report.json \
  --output examples/output/session_gpu_004/refined_after_crop_v1 \
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

See [`docs/segment_refinement.md`](docs/segment_refinement.md) for the optional second-pass segment re-detection workflow.

See [`docs/skeleton_optimization.md`](docs/skeleton_optimization.md) for the optional skeleton constraint and optimization workflow.

## Recommended pipeline

### Visualization-first path

```text
record_from_video.py
-> clean_pose_data.py
-> crop_refine_pose.py
-> minimize_pose_outliers.py
-> trajectory_export.py planned
-> Blender / TouchDesigner importer planned
```

### Optional diagnostic path

```text
refine_pose_segments.py
-> optimize_pose_skeleton.py
-> optimization reports
```

Skeleton optimization is useful for diagnostics, but it is not the default final visualization layer.
For current visualization-first work, prefer `crop_refined_pose.csv` followed by `outlier_minimized_pose.csv`.

## Current cleaning baseline

The current baseline preset is conservative. It keeps raw extraction and cleaned data separate, interpolates short missing-frame gaps, and avoids turning long or uncertain outlier runs into plausible-looking motion.

Recommended baseline options:

```bash
python scripts/clean_pose_data.py \
  --input-video examples/input/dance_take_002.mov \
  --input-csv examples/output/session_gpu_002/raw_pose.csv \
  --input-jsonl examples/output/session_gpu_002/raw_pose.jsonl \
  --metadata examples/output/session_gpu_002/metadata.json \
  --output examples/output/session_gpu_002/cleaned_arm55_legkeep_outlier3 \
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
