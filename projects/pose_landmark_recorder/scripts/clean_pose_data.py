#!/usr/bin/env python
"""Clean, interpolate, and smooth raw MediaPipe pose recorder data."""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.pose_cleaner import CleaningOptions, clean_pose_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean raw pose CSV/JSONL data.")
    parser.add_argument("--input-csv", required=True, type=Path, help="Input raw_pose.csv path.")
    parser.add_argument("--metadata", required=True, type=Path, help="Input raw_metadata.json path.")
    parser.add_argument("--output", required=True, type=Path, help="Output cleaned directory.")
    parser.add_argument("--input-video", type=Path, default=None, help="Input video path for corrected preview.")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Optional input raw_pose.jsonl path.")
    parser.add_argument("--max-interpolate-gap", type=int, default=15)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--jump-threshold-multiplier", type=float, default=6.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--save-csv", action="store_true", help="Write cleaned_pose.csv.")
    parser.add_argument("--save-jsonl", action="store_true", help="Write cleaned_pose.jsonl.")
    parser.add_argument("--save-preview", action="store_true", help="Write cleaned_corrected_preview.mp4.")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable rolling mean smoothing.")
    parser.add_argument(
        "--interpolate-outliers",
        action="store_true",
        help="Also interpolate all short invalid measured runs such as jump or low-confidence outliers.",
    )
    parser.add_argument(
        "--disable-recoverable-outlier-interpolation",
        dest="interpolate_recoverable_outliers",
        action="store_false",
        default=True,
        help="Disable interpolation of short spike-like jump outliers.",
    )
    parser.add_argument("--outlier-max-gap", type=int, default=3, help="Maximum spike-like outlier run to interpolate.")
    parser.add_argument(
        "--disable-torso-side-lock",
        dest="enable_torso_side_lock",
        action="store_false",
        default=True,
        help="Disable temporal left/right torso identity correction.",
    )
    parser.add_argument(
        "--enable-pelvis-side-lock",
        action="store_true",
        help="Enable experimental temporal left/right pelvis and leg-chain identity correction.",
    )
    parser.add_argument(
        "--torso-swap-cost-ratio",
        type=float,
        default=0.65,
        help="Swap left/right torso assignment when swapped temporal cost is below this ratio.",
    )
    parser.add_argument(
        "--shoulder-hip-guard-ratio",
        type=float,
        default=0.98,
        help="Reject shoulder swaps that worsen shoulder-hip side cost beyond this ratio.",
    )
    parser.add_argument(
        "--arm-occlusion-max-gap",
        type=int,
        default=55,
        help="Maximum low-confidence arm run to estimate from shoulder/elbow-local offsets.",
    )
    parser.add_argument(
        "--disable-leg-low-visibility-salvage",
        dest="enable_leg_low_visibility_salvage",
        action="store_false",
        default=True,
        help="Disable conservative keeping of stable low-visibility leg measurements.",
    )
    parser.add_argument(
        "--leg-salvage-min-visibility",
        type=float,
        default=0.15,
        help="Minimum visibility for keeping stable low-visibility leg measurements.",
    )
    bone_group = parser.add_mutually_exclusive_group()
    bone_group.add_argument("--enable-bone-check", dest="enable_bone_check", action="store_true", default=True)
    bone_group.add_argument("--disable-bone-check", dest="enable_bone_check", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.save_preview and args.input_video is None:
        raise SystemExit("--input-video is required when --save-preview is used")
    if args.smoothing_window % 2 == 0:
        raise SystemExit("--smoothing-window must be odd")

    save_csv = args.save_csv or not args.save_jsonl
    save_jsonl = args.save_jsonl or not args.save_csv
    result = clean_pose_session(
        CleaningOptions(
            input_csv=args.input_csv,
            input_jsonl=args.input_jsonl,
            input_video=args.input_video,
            metadata=args.metadata,
            output=args.output,
            max_interpolate_gap=args.max_interpolate_gap,
            visibility_threshold=args.visibility_threshold,
            presence_threshold=args.presence_threshold,
            jump_threshold_multiplier=args.jump_threshold_multiplier,
            smoothing_window=args.smoothing_window,
            save_csv=save_csv,
            save_jsonl=save_jsonl,
            save_preview=args.save_preview,
            no_smoothing=args.no_smoothing,
            enable_bone_check=args.enable_bone_check,
            interpolate_recoverable_outliers=args.interpolate_recoverable_outliers,
            interpolate_outliers=args.interpolate_outliers,
            outlier_max_gap=args.outlier_max_gap,
            enable_torso_side_lock=args.enable_torso_side_lock,
            enable_pelvis_side_lock=args.enable_pelvis_side_lock,
            torso_swap_cost_ratio=args.torso_swap_cost_ratio,
            shoulder_hip_guard_ratio=args.shoulder_hip_guard_ratio,
            arm_occlusion_max_gap=args.arm_occlusion_max_gap,
            enable_leg_low_visibility_salvage=args.enable_leg_low_visibility_salvage,
            leg_salvage_min_visibility=args.leg_salvage_min_visibility,
        )
    )
    print(f"Wrote cleaned outputs to {result.frame_status_csv.parent}")
    for path in (
        result.cleaned_csv,
        result.cleaned_jsonl,
        result.frame_status_csv,
        result.quality_report,
        result.interpolation_report,
        result.corrected_preview,
    ):
        if path:
            print(path)


if __name__ == "__main__":
    main()
