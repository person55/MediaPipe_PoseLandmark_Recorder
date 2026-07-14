# Bundled Model

`pose_landmarker.task` is a tracked runtime dependency, so a fresh clone can
run the default pipeline without a separate model download.

- **Bundle:** MediaPipe Pose Landmarker Full, float16, version 1
- **Official source:** <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task>
- **License:** Apache License 2.0. The official [BlazePose GHUM 3D model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) identifies this model family as Apache-2.0.
- **SHA-256 of this tracked bundle:** `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad`

The bundle contains the official `pose_detector.tflite` and
`pose_landmarks_detector.tflite` payloads. The source archive can differ at the
ZIP timestamp metadata level; use the SHA-256 above to verify the exact bundle
tracked by this repository.

Pass `--model /path/to/other.task` to use a compatible custom model instead.
