# AGENTS.md

## Scope

Work only inside `projects/pose_landmark_recorder/` unless explicitly instructed otherwise.

Do not modify upstream MediaPipe source files.

## Safety

Never commit:

- input videos
- output sessions
- raw/cleaned/refined/optimized CSV files
- raw/cleaned/refined/optimized JSONL files
- preview videos
- corrected/refined/optimized preview videos
- `.venv`
- build, dist, cache files
- `.DS_Store`

Do not use `git add .`.

Push only to `origin master`. Never push to `upstream`.

## Model policy

`models/pose_landmarker.task` is an intentional tracked runtime dependency:
the official MediaPipe Pose Landmarker Full (float16, v1) bundle. It is not a
generated session artifact.

Do not replace it or add another model binary unless `models/README.md` records
the official source URL, model/version, applicable license, and SHA-256.

## Current pipeline

```text
record_from_video.py
-> clean_pose_data.py
-> refine_pose_segments.py
-> minimize_pose_outliers.py planned
-> Blender importer planned
```

Optional diagnostic branch:

```text
refined_pose.csv
-> optimize_pose_skeleton.py
-> optimization reports / diagnostic overlay
```

## Data policy

- `raw_pose` preserves direct MediaPipe measurements.
- `cleaned_pose` applies validation, short interpolation, smoothing, and quality flags.
- `refined_pose` accepts only better re-detected candidates.
- future `outlier_minimized_pose` should reduce visual/temporal spikes without generating motion.
- `optimized_pose` is an optional diagnostic layer for conservative skeleton checks, not the default final visualization layer.
- generated motion must be explicitly separated from measured/refined/optimized data.

## Default reading policy

Before each task, read only:

- `AGENTS.md`
- `CURRENT_STATE.md`
- `docs/codex_context_min.md`
- task-specific files

Do not read `examples/output/**`, `examples/input/**`, `.venv/**`, `docs/archive/**`, or upstream MediaPipe source files unless explicitly instructed.

## Testing

Run tests from:

```bash
cd projects/pose_landmark_recorder
source .venv/bin/activate
python -m pytest
```
