#!/usr/bin/env python
"""Export outlier-minimized pose rows as Blender trajectory CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.trajectory_exporter import (  # noqa: E402
    TrajectoryExportOptions,
    export_trajectory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Blender/TouchDesigner trajectory CSV files.")
    parser.add_argument("--input-pose-csv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--coordinate-mode",
        default="screen_bottom_origin",
        choices=["screen_bottom_origin", "pose_world_direct", "pose_2d_flat"],
    )
    parser.add_argument("--source", default="pose", choices=["pose", "pose_world"])
    parser.add_argument("--depth-mode", default="pose_z", choices=["none", "pose_z", "pose_world_y"])
    parser.add_argument("--landmark-preset", default="blender_default", choices=["blender_default", "all", "body_core", "custom"])
    parser.add_argument("--include-landmarks", default="")
    parser.add_argument("--exclude-landmarks", default="")
    parser.add_argument("--screen-origin-x", type=float, default=0.5)
    parser.add_argument("--screen-origin-y", type=float, default=1.0)
    parser.add_argument("--screen-width-scale", type=float, default=6.0)
    parser.add_argument("--screen-height-scale", type=float, default=6.0)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--apply-aspect-ratio", action="store_true", default=True)
    parser.add_argument("--no-aspect-ratio", action="store_false", dest="apply_aspect_ratio")
    parser.add_argument("--smooth-trajectory", action="store_true", default=True)
    parser.add_argument("--no-smooth-trajectory", action="store_false", dest="smooth_trajectory")
    parser.add_argument("--smooth-min-cutoff-hz", type=float, default=1.2)
    parser.add_argument("--smooth-beta", type=float, default=1.5)
    parser.add_argument("--smooth-depth-min-cutoff-hz", type=float, default=0.4)
    parser.add_argument("--smooth-depth-beta", type=float, default=0.4)
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--include-disconnected-points", action="store_true", default=True)
    parser.add_argument("--no-include-disconnected-points", action="store_false", dest="include_disconnected_points")
    parser.add_argument("--save-points", action="store_true", default=True)
    parser.add_argument("--no-save-points", action="store_false", dest="save_points")
    parser.add_argument("--save-segments", action="store_true", default=True)
    parser.add_argument("--no-save-segments", action="store_false", dest="save_segments")
    parser.add_argument("--save-report", action="store_true", default=True)
    parser.add_argument("--no-save-report", action="store_false", dest="save_report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_trajectory(
        input_pose_csv=args.input_pose_csv,
        metadata_path=args.metadata,
        output_dir=args.output,
        options=TrajectoryExportOptions(
            coordinate_mode=args.coordinate_mode,
            source=args.source,
            depth_mode=args.depth_mode,
            landmark_preset=args.landmark_preset,
            include_landmarks=_parse_names(args.include_landmarks),
            exclude_landmarks=_parse_names(args.exclude_landmarks),
            screen_origin_x=args.screen_origin_x,
            screen_origin_y=args.screen_origin_y,
            screen_width_scale=args.screen_width_scale,
            screen_height_scale=args.screen_height_scale,
            depth_scale=args.depth_scale,
            apply_aspect_ratio=args.apply_aspect_ratio,
            smooth_trajectory=args.smooth_trajectory,
            smooth_min_cutoff_hz=args.smooth_min_cutoff_hz,
            smooth_beta=args.smooth_beta,
            smooth_depth_min_cutoff_hz=args.smooth_depth_min_cutoff_hz,
            smooth_depth_beta=args.smooth_depth_beta,
            include_hidden=args.include_hidden,
            include_disconnected_points=args.include_disconnected_points,
            save_points=args.save_points,
            save_segments=args.save_segments,
            save_report=args.save_report,
        ),
    )
    first_output = result.get("report_json") or result.get("points_csv") or result.get("segments_csv")
    if first_output:
        print(f"Wrote trajectory export outputs to {first_output.parent}")
    else:
        print(f"Trajectory export completed without file output for {args.output}")
    for key in ("points_csv", "segments_csv", "report_json"):
        path = result.get(key)
        if path:
            print(path)


def _parse_names(value: str) -> list[str] | None:
    names = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return names or None


if __name__ == "__main__":
    main()
