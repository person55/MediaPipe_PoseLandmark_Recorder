# macOS Packaging Notes

Packaging is not part of the initial Phase 1 verification, but the planned tool is PyInstaller.

Initial packaging should use `--onedir`, because MediaPipe and the `.task` model file need explicit inclusion:

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
