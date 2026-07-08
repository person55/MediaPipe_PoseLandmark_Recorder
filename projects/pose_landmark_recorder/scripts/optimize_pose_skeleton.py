#!/usr/bin/env python
"""Run conservative skeleton optimization on refined pose data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.skeleton_optimizer import (  # noqa: E402
    SkeletonOptimizationOptions,
    optimize_pose_skeleton,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose and conservatively optimize refined pose skeleton data.")
    parser.add_argument("--input-refined-csv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refine-report", type=Path, default=None)
    parser.add_argument("--constraints", type=Path, default=Path("configs/skeleton_constraints.yaml"))
    parser.add_argument("--source", default="pose_world")
    parser.add_argument("--max-correction-gap-sec", type=float, default=0.10)
    parser.add_argument("--max-review-gap-sec", type=float, default=1.50)
    parser.add_argument("--adaptive-percentile-low", type=float, default=1.0)
    parser.add_argument("--adaptive-percentile-high", type=float, default=99.0)
    parser.add_argument("--adaptive-margin-deg", type=float, default=10.0)
    parser.add_argument("--bone-length-min-ratio", type=float, default=0.45)
    parser.add_argument("--bone-length-max-ratio", type=float, default=1.75)
    parser.add_argument("--reachability-margin-ratio", type=float, default=0.10)
    parser.add_argument("--temporal-jump-multiplier", type=float, default=6.0)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-jsonl", action="store_true")
    parser.add_argument("--save-reports", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    save_csv = args.save_csv or not args.save_jsonl
    save_jsonl = args.save_jsonl or not args.save_csv
    options = SkeletonOptimizationOptions(
        max_correction_gap_sec=args.max_correction_gap_sec,
        max_review_gap_sec=args.max_review_gap_sec,
        adaptive_percentile_low=args.adaptive_percentile_low,
        adaptive_percentile_high=args.adaptive_percentile_high,
        adaptive_margin_deg=args.adaptive_margin_deg,
        bone_length_min_ratio=args.bone_length_min_ratio,
        bone_length_max_ratio=args.bone_length_max_ratio,
        reachability_margin_ratio=args.reachability_margin_ratio,
        temporal_jump_multiplier=args.temporal_jump_multiplier,
        source=args.source,
        save_csv=save_csv,
        save_jsonl=save_jsonl,
        save_reports=args.save_reports or not args.save_csv,
        save_preview=args.save_preview,
    )
    result = optimize_pose_skeleton(
        input_refined_csv=args.input_refined_csv,
        metadata_path=args.metadata,
        output_dir=args.output,
        constraints_path=args.constraints,
        options=options,
        refine_report_path=args.refine_report,
    )
    print(f"Wrote optimized outputs to {args.output}")
    for path in (
        result.optimized_csv,
        result.optimized_jsonl,
        result.optimization_report,
        result.bone_length_report,
        result.joint_angle_report,
        result.optimization_segments,
        result.optimized_preview,
    ):
        if path:
            print(path)


if __name__ == "__main__":
    main()
