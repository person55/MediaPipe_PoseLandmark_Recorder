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

Push only to `origin`. Never push to `upstream`.

Improvement work happens on feature branches. Do not push to `origin master` directly unless explicitly instructed.

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
-> crop_refine_pose.py      (multi-pass re-detection + crop_crosspass_agreement.csv diagnostic)
-> refine_pose_segments.py
-> minimize_pose_outliers.py (fps-normalized physical spike floors)
-> export_trajectory.py      (aspect ratio, One-Euro *_smooth columns)
-> open_blender_trajectory.py (fade-policy consumption incl. per-frame marker fade)
-> write_session_manifest.py  (reproducibility manifest)
```

One-command runner: `pose-landmark-pipeline` (`src/dance_pose_recorder/pipeline_runner.py`).

Standalone diagnostics: `report_crosspass_agreement.py` (backfill the acceptance
agreement diagnostic for old sessions), `build_motion_profile.py` (read-only
statistical motion profile in `configs/`; never wired into thresholds).

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
- `outlier_minimized_pose` reduces visual/temporal spikes without generating motion; its corrections/breaks propagate to the export source (`sync_sources`).
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

On Windows (Git Bash), run the venv interpreter directly instead of activating:

```bash
cd projects/pose_landmark_recorder
./.venv/Scripts/python.exe -m pytest -q
```
