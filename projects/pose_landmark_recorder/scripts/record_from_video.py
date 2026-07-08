#!/usr/bin/env python
"""Record MediaPipe pose landmarks from a video file."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.main import main


if __name__ == "__main__":
    main()
