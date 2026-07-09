#!/usr/bin/env python
"""Visualization-oriented temporal outlier minimization CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.outlier_minimizer import (  # noqa: E402
    OutlierMinimizerOptions,
    minimize_pose_outliers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimize short pose outliers and write trajectory display policy.")
    parser.add_argument("--input-pose-csv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crop-refine-report", type=Path, default=None)
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--source", default="pose_world", choices=["pose_world", "pose", "both"])
    parser.add_argument("--position-fields", default="tx,ty,tz")
    parser.add_argument("--max-correction-gap-sec", type=float, default=0.12)
    parser.add_argument("--max-break-gap-sec", type=float, default=0.20)
    parser.add_argument("--velocity-threshold-multiplier", type=float, default=6.0)
    parser.add_argument("--acceleration-threshold-multiplier", type=float, default=6.0)
    parser.add_argument("--jerk-threshold-multiplier", type=float, default=8.0)
    parser.add_argument("--min-stable-neighbors", type=int, default=2)
    parser.add_argument("--landmark-policy", default="visualization", choices=["default", "visualization"])
    parser.add_argument("--preserve-quality-flags", action="store_true", default=True)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-jsonl", action="store_true")
    parser.add_argument("--save-report", action="store_true")
    parser.add_argument("--save-trajectory-breaks", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    position_fields = tuple(item.strip() for item in args.position_fields.split(",") if item.strip())
    if len(position_fields) not in {2, 3}:
        raise SystemExit("--position-fields must contain two or three comma-separated fields")

    result = minimize_pose_outliers(
        input_pose_csv=args.input_pose_csv,
        metadata_path=args.metadata,
        output_dir=args.output,
        crop_refine_report_path=args.crop_refine_report,
        quality_report_path=args.quality_report,
        options=OutlierMinimizerOptions(
            source=args.source,
            position_fields=position_fields,  # type: ignore[arg-type]
            max_correction_gap_sec=args.max_correction_gap_sec,
            max_break_gap_sec=args.max_break_gap_sec,
            velocity_threshold_multiplier=args.velocity_threshold_multiplier,
            acceleration_threshold_multiplier=args.acceleration_threshold_multiplier,
            jerk_threshold_multiplier=args.jerk_threshold_multiplier,
            min_stable_neighbors=args.min_stable_neighbors,
            landmark_policy=args.landmark_policy,
            preserve_quality_flags=args.preserve_quality_flags,
            save_csv=args.save_csv,
            save_jsonl=args.save_jsonl,
            save_report=args.save_report,
            save_trajectory_breaks=args.save_trajectory_breaks,
        ),
    )
    print(f"Wrote outlier minimization outputs to {args.output}")
    for key in (
        "outlier_minimized_csv",
        "outlier_minimized_jsonl",
        "outlier_report",
        "temporal_spike_report",
        "trajectory_breaks",
    ):
        path = result.get(key)
        if path:
            print(path)


if __name__ == "__main__":
    main()
