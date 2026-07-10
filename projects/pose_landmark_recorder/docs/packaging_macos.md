# macOS Packaging Notes

Packaging is not part of the initial Phase 1 verification, but the planned tool is PyInstaller.

The current one-command local runner is installed through the editable package:

```bash
python -m pip install -e .
pose-landmark-pipeline \
  --input-video examples/input/dance_take_006.mp4 \
  --delegate gpu \
  --blender-mode background
```

`pose-landmark-pipeline` runs the full raw -> cleaned -> crop-refined -> refined -> outlier-minimized -> trajectory export -> Blender handoff flow. The legacy `scripts/run_full_pipeline.py` path remains as a compatibility wrapper for the same code path.

Build the current full-pipeline executable with:

```bash
python scripts/build_pipeline_app.py
```

Output:

```plain text
dist/pose-landmark-pipeline/pose-landmark-pipeline
```

The build script uses `--onedir`, because MediaPipe and the `.task` model file need explicit inclusion. Earlier single-script recorder packaging used:

```bash
pyinstaller scripts/record_from_video.py \
  --name record_from_video \
  --onedir \
  --add-data "models/pose_landmarker.task:models" \
  --collect-all mediapipe \
  --hidden-import mediapipe.tasks.c
```

Before distribution, verify:

- `models/pose_landmarker.task` is included.
- MediaPipe runtime resources are included.
- OpenCV video decoding works on the target Mac.
- Apple Silicon and Intel targets are handled intentionally.
- Code signing and Gatekeeper requirements are reviewed.

For a standalone full-pipeline build, also verify that the packaged app can find the `scripts/` helpers, the `models/pose_landmarker.task` file, writable `examples/output` or user-selected output directory, and the external Blender executable. Blender itself should remain an external dependency unless a dedicated Blender add-on/package is built later.
