#!/usr/bin/env python
"""Build the full pipeline runner with PyInstaller."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_DIR / "dist"
WORK_DIR = PROJECT_DIR / "build/pyinstaller-work"
SPEC_DIR = PROJECT_DIR / "build/pyinstaller-spec"
CONFIG_DIR = PROJECT_DIR / "build/pyinstaller-config"
MPLCONFIG_DIR = PROJECT_DIR / "build/matplotlib-config"
MODEL_PATH = PROJECT_DIR / "models/pose_landmarker.task"
ENTRY_SCRIPT = PROJECT_DIR / "src/dance_pose_recorder/pipeline_runner.py"


def add_data_arg(source: Path, target: str) -> str:
    return f"{source}{os.pathsep}{target}"


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Missing model file: {MODEL_PATH}", file=sys.stderr)
        return 2

    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(ENTRY_SCRIPT),
        "--name",
        "pose-landmark-pipeline",
        "--onedir",
        "--noconfirm",
        "--clean",
        "--paths",
        str(PROJECT_DIR / "src"),
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(WORK_DIR),
        "--specpath",
        str(SPEC_DIR),
        "--add-data",
        add_data_arg(PROJECT_DIR / "scripts", "scripts"),
        "--add-data",
        add_data_arg(MODEL_PATH, "models"),
        "--collect-submodules",
        "dance_pose_recorder",
        "--collect-all",
        "mediapipe",
        "--hidden-import",
        "mediapipe.tasks.python",
        "--hidden-import",
        "mediapipe.tasks.python.vision",
    ]

    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(CONFIG_DIR)
    env["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)

    print("Building pose-landmark-pipeline with PyInstaller", flush=True)
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=PROJECT_DIR, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
