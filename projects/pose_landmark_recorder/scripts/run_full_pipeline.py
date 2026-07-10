#!/usr/bin/env python
"""Compatibility wrapper for the full pose-to-Blender pipeline runner."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.pipeline_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
