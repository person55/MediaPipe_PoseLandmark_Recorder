"""Run the full pose-to-Blender output pipeline."""

from __future__ import annotations

import argparse
import re
import runpy
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


INTERNAL_STAGE_FLAG = "--internal-run-stage"
IS_FROZEN = bool(getattr(sys, "frozen", False))
PROJECT_DIR = (
    Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    if IS_FROZEN
    else Path(__file__).resolve().parents[2]
)
RUN_BASE_DIR = Path.cwd() if IS_FROZEN else PROJECT_DIR
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DEFAULT_MODEL = Path("models/pose_landmarker.task")
DEFAULT_OUTPUT_ROOT = Path("examples/output")
DEFAULT_BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")

METAL_FAILURE_PATTERNS = (
    "Could not create an NSOpenGLPixelFormat",
    "kGpuService",
    "DrishtiMetalHelper",
    "Service is unavailable",
    "failed to create pixel format",
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    requires_metal_context: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run raw -> cleaned -> crop -> refine -> Blender trajectory outputs."
    )
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--delegate", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--origin", default="first_frame_pelvis")
    parser.add_argument(
        "--blender-mode",
        choices=["background", "gui", "skip"],
        default="background",
        help="Use background for automated .blend creation, gui for manual Blender open, or skip.",
    )
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--show-camera-summary", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--continue-on-existing",
        action="store_true",
        help="Skip a stage when its primary output already exists.",
    )
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == INTERNAL_STAGE_FLAG:
        return run_internal_stage(sys.argv[2:])

    args = build_parser().parse_args()
    input_video = resolve_input_path(args.input_video)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    model = resolve_existing_path(args.model)
    if not model.exists():
        raise FileNotFoundError(f"Pose model not found: {model}")

    session_id = args.session_id or derive_session_id(input_video, args.delegate)
    output_root = args.output_root if args.output_root.is_absolute() else RUN_BASE_DIR / args.output_root
    session_dir = output_root / session_id
    log_dir = session_dir / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stages = build_stages(args, input_video, model, session_id, session_dir)
    for stage in stages:
        primary = primary_output_for_stage(stage.name, session_dir)
        if args.continue_on_existing and primary and primary.exists():
            print(f"[skip] {stage.name}: existing {primary}")
            continue
        result = run_stage(stage, log_dir)
        if result != 0:
            return result
    print(f"Pipeline complete: {session_dir}")
    return 0


def run_internal_stage(argv: list[str]) -> int:
    if not argv:
        print(f"{INTERNAL_STAGE_FLAG} requires a script name", file=sys.stderr)
        return 2
    script_name = argv[0]
    script_args = argv[2:] if len(argv) > 1 and argv[1] == "--" else argv[1:]
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"Stage script not found: {script_path}", file=sys.stderr)
        return 2

    original_argv = sys.argv
    sys.argv = [str(script_path), *script_args]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    finally:
        sys.argv = original_argv
    return 0


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [RUN_BASE_DIR / path]
    if RUN_BASE_DIR != PROJECT_DIR:
        candidates.append(PROJECT_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_existing_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [RUN_BASE_DIR / path]
    if RUN_BASE_DIR != PROJECT_DIR:
        candidates.append(PROJECT_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_stages(
    args: argparse.Namespace,
    input_video: Path,
    model: Path,
    session_id: str,
    session_dir: Path,
) -> list[Stage]:
    preview_flags = [] if args.no_preview else ["--save-preview"]
    cleaned_dir = session_dir / "cleaned"
    crop_dir = session_dir / "crop_refine"
    refined_dir = session_dir / "refined_after_crop_v1"
    outlier_dir = session_dir / "outlier_minimized"
    trajectory_dir = session_dir / "trajectory_export"

    stages = [
        Stage(
            "raw",
            python_stage_command(
                "record_from_video.py",
                [
                    "--input",
                    str(input_video),
                    "--output",
                    str(session_dir),
                    "--model",
                    str(model),
                    "--delegate",
                    args.delegate,
                    "--origin",
                    args.origin,
                    "--save-jsonl",
                    "--save-csv",
                    *preview_flags,
                ],
            ),
        ),
        Stage(
            "cleaned",
            python_stage_command(
                "clean_pose_data.py",
                [
                    "--input-video",
                    str(input_video),
                    "--input-csv",
                    str(session_dir / "raw/raw_pose.csv"),
                    "--input-jsonl",
                    str(session_dir / "raw/raw_pose.jsonl"),
                    "--metadata",
                    str(session_dir / "raw/raw_metadata.json"),
                    "--output",
                    str(session_dir),
                    "--max-interpolate-gap",
                    "15",
                    "--visibility-threshold",
                    "0.5",
                    "--presence-threshold",
                    "0.5",
                    "--jump-threshold-multiplier",
                    "6.0",
                    "--smoothing-window",
                    "7",
                    "--outlier-max-gap",
                    "3",
                    "--arm-occlusion-max-gap",
                    "55",
                    "--leg-salvage-min-visibility",
                    "0.15",
                    "--save-csv",
                    "--save-jsonl",
                    *preview_flags,
                ],
            ),
        ),
        Stage(
            "crop_refine",
            python_stage_command(
                "crop_refine_pose.py",
                [
                    "--input-video",
                    str(input_video),
                    "--input-cleaned-csv",
                    str(cleaned_dir / "cleaned_pose.csv"),
                    "--metadata",
                    str(session_dir / "raw/raw_metadata.json"),
                    "--quality-report",
                    str(cleaned_dir / "cleaned_quality_report.json"),
                    "--output",
                    str(session_dir),
                    "--model",
                    str(model),
                    "--delegate",
                    args.delegate,
                    "--running-mode",
                    "video",
                    "--crop-source",
                    "torso",
                    "--target-flags",
                    "unreliable,interpolated_outlier_removed,estimated_occluded_arm",
                    "--target-segment-types",
                    "mixed_problem_segment",
                    "--max-segment-length",
                    "100",
                    "--segment-margin",
                    "12",
                    "--accept-score-margin",
                    "0.06",
                    "--save-candidates",
                    "--save-refined",
                    "--save-report",
                    "--save-jsonl",
                    *preview_flags,
                    "--save-debug-images",
                ],
            ),
            requires_metal_context=True,
        ),
        Stage(
            "refined",
            python_stage_command(
                "refine_pose_segments.py",
                [
                    "--input-video",
                    str(input_video),
                    "--input-cleaned-csv",
                    str(crop_dir / "crop_refine_pose.csv"),
                    "--input-raw-csv",
                    str(session_dir / "raw/raw_pose.csv"),
                    "--metadata",
                    str(session_dir / "raw/raw_metadata.json"),
                    "--frame-status",
                    str(cleaned_dir / "cleaned_frame_status.csv"),
                    "--quality-report",
                    str(cleaned_dir / "cleaned_quality_report.json"),
                    "--output",
                    str(refined_dir),
                    "--delegate",
                    args.delegate,
                    "--target-landmarks",
                    "arms,hands,feet",
                    "--min-cluster-length",
                    "2",
                    "--max-cluster-length",
                    "90",
                    "--segment-margin",
                    "12",
                    "--accept-score-margin",
                    "0.08",
                    "--save-csv",
                    "--save-jsonl",
                    *preview_flags,
                ],
            ),
            requires_metal_context=True,
        ),
        Stage(
            "outlier_minimized",
            python_stage_command(
                "minimize_pose_outliers.py",
                [
                    "--input-pose-csv",
                    str(refined_dir / "refined_pose.csv"),
                    "--metadata",
                    str(session_dir / "raw/raw_metadata.json"),
                    "--crop-refine-report",
                    str(crop_dir / "crop_refine_report.json"),
                    "--output",
                    str(session_dir),
                    "--source",
                    "pose_world",
                    "--position-fields",
                    "tx,ty,tz",
                    "--max-correction-gap-sec",
                    "0.12",
                    "--max-break-gap-sec",
                    "0.20",
                    "--velocity-threshold-multiplier",
                    "6.0",
                    "--acceleration-threshold-multiplier",
                    "6.0",
                    "--jerk-threshold-multiplier",
                    "8.0",
                    "--min-stable-neighbors",
                    "2",
                    "--preserve-quality-flags",
                    "--save-csv",
                    "--save-report",
                    "--save-trajectory-breaks",
                ],
            ),
        ),
        Stage(
            "trajectory_export",
            python_stage_command(
                "export_trajectory.py",
                [
                    "--input-pose-csv",
                    str(outlier_dir / "outlier_minimized_pose.csv"),
                    "--metadata",
                    str(session_dir / "raw/raw_metadata.json"),
                    "--output",
                    str(session_dir),
                    "--coordinate-mode",
                    "screen_bottom_origin",
                    "--source",
                    "pose",
                    "--depth-mode",
                    "pose_z",
                    "--landmark-preset",
                    "blender_default",
                    "--screen-origin-x",
                    "0.5",
                    "--screen-origin-y",
                    "1.0",
                    "--screen-width-scale",
                    "6.0",
                    "--screen-height-scale",
                    "6.0",
                    "--depth-scale",
                    "1.0",
                    "--save-points",
                    "--save-segments",
                    "--save-report",
                ],
            ),
        ),
    ]

    if args.blender_mode != "skip":
        stages.append(blender_stage(args, trajectory_dir))
    return stages


def python_stage_command(script_name: str, arguments: list[str]) -> list[str]:
    if IS_FROZEN:
        return [sys.executable, INTERNAL_STAGE_FLAG, script_name, "--", *arguments]
    return [sys.executable, str(SCRIPTS_DIR / script_name), *arguments]


def blender_stage(args: argparse.Namespace, trajectory_dir: Path) -> Stage:
    script = SCRIPTS_DIR / "open_blender_trajectory.py"
    summary_flag = ["--show-camera-summary"] if args.show_camera_summary else []
    if args.blender_mode == "background":
        return Stage(
            "blender_background",
            [
                str(args.blender_bin),
                "--background",
                "--python",
                str(script),
                "--",
                "--inside-blender",
                "--trajectory-dir",
                str(trajectory_dir),
                *summary_flag,
            ],
            requires_metal_context=True,
        )
    command = python_stage_command(
        "open_blender_trajectory.py",
        [
            "--trajectory-dir",
            str(trajectory_dir),
            *summary_flag,
        ],
    )
    return Stage(
        "blender_gui",
        command,
        requires_metal_context=True,
    )


def run_stage(stage: Stage, log_dir: Path) -> int:
    log_path = log_dir / f"{stage.name}.log"
    print(f"[run] {stage.name}")
    print(format_command(stage.command))
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {format_command(stage.command)}\n\n")
        proc = subprocess.run(
            stage.command,
            cwd=RUN_BASE_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode == 0:
        print(f"[ok] {stage.name} log={log_path}")
        return 0

    tail = tail_text(log_path)
    print(f"[fail] {stage.name} exit={proc.returncode} log={log_path}")
    print(tail)
    if stage.requires_metal_context and is_metal_context_failure(tail, proc.returncode):
        print_metal_policy(stage)
    return proc.returncode


def print_metal_policy(stage: Stage) -> None:
    print()
    print("Detected macOS GPU/Metal context failure.")
    print("Project code cannot escalate itself outside the Codex sandbox.")
    print("Rerun this exact stage outside the sandbox, or ask Codex to run it with escalation:")
    print(format_command(stage.command))


def is_metal_context_failure(text: str, returncode: int) -> bool:
    if returncode in {134, -6}:
        return True
    return any(pattern in text for pattern in METAL_FAILURE_PATTERNS)


def tail_text(path: Path, line_count: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def primary_output_for_stage(stage_name: str, session_dir: Path) -> Path | None:
    mapping = {
        "raw": session_dir / "raw/raw_pose.csv",
        "cleaned": session_dir / "cleaned/cleaned_pose.csv",
        "crop_refine": session_dir / "crop_refine/crop_refine_pose.csv",
        "refined": session_dir / "refined_after_crop_v1/refined_pose.csv",
        "outlier_minimized": session_dir / "outlier_minimized/outlier_minimized_pose.csv",
        "trajectory_export": session_dir / "trajectory_export/trajectory_export_points.csv",
        "blender_background": session_dir / "blender",
        "blender_gui": session_dir / "blender",
    }
    return mapping.get(stage_name)


def derive_session_id(input_video: Path, delegate: str) -> str:
    match = re.search(r"(\d+)(?!.*\d)", input_video.stem)
    suffix = match.group(1) if match else input_video.stem.replace(" ", "_")
    return f"session_{delegate}_{suffix}"


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
