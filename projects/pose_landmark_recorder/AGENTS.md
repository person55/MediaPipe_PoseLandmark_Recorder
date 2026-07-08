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
- `.task` model files
- `.venv`
- build, dist, cache files
- `.DS_Store`

Do not use `git add .`.

Push only to `origin master`. Never push to `upstream`.

## Current pipeline

```text
record_from_video.py
-> clean_pose_data.py
-> refine_pose_segments.py
-> optimize_pose_skeleton.py
-> outlier minimization planned
-> Blender importer planned
```

## Data policy

- `raw_pose` preserves direct MediaPipe measurements.
- `cleaned_pose` applies validation, short interpolation, smoothing, and quality flags.
- `refined_pose` accepts only better re-detected candidates.
- `optimized_pose` applies conservative skeleton diagnostics and very limited correction.
- future `outlier_minimized_pose` should reduce visual/temporal spikes without generating motion.
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
