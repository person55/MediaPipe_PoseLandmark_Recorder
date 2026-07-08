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

MediaPipe `pose_world_landmarks` are model-estimated 3D coordinates from a single RGB source. They are useful for trajectory and visual experiments, but should not be treated as calibrated stage coordinates.
