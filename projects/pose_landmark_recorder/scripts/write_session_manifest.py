#!/usr/bin/env python
"""Write a reproducibility manifest (manifest.json) for a pipeline session."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.session_manifest import write_session_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a session reproducibility manifest.")
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-input-hash",
        action="store_true",
        help="Skip hashing the source video (useful when the input is unavailable or very large).",
    )
    args = parser.parse_args()

    output = write_session_manifest(args.session_dir, include_input_hash=not args.skip_input_hash)
    print(f"Wrote session manifest to {output}")


if __name__ == "__main__":
    main()
