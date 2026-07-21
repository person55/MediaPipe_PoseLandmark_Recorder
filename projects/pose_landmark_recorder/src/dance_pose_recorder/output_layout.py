"""Output folder and file naming conventions."""

from __future__ import annotations

import json
from pathlib import Path


RAW_DIR = "raw"
CLEANED_DIR = "cleaned"
CROP_REFINE_DIR = "crop_refine"
OUTLIER_MINIMIZED_DIR = "outlier_minimized"
TRAJECTORY_EXPORT_DIR = "trajectory_export"
BLENDER_DIR = "blender"

RAW_POSE_CSV = "raw_pose.csv"
RAW_POSE_JSONL = "raw_pose.jsonl"
RAW_METADATA_JSON = "raw_metadata.json"
RAW_PREVIEW_MP4 = "raw_preview.mp4"

CLEANED_POSE_CSV = "cleaned_pose.csv"
CLEANED_POSE_JSONL = "cleaned_pose.jsonl"
CLEANED_FRAME_STATUS_CSV = "cleaned_frame_status.csv"
CLEANED_QUALITY_REPORT_JSON = "cleaned_quality_report.json"
CLEANED_INTERPOLATION_REPORT_JSON = "cleaned_interpolation_report.json"
CLEANED_CORRECTED_PREVIEW_MP4 = "cleaned_corrected_preview.mp4"

CROP_REFINE_POSE_CSV = "crop_refine_pose.csv"
CROP_REFINE_POSE_JSONL = "crop_refine_pose.jsonl"
CROP_REFINE_CANDIDATES_CSV = "crop_refine_candidates.csv"
CROP_REFINE_CANDIDATE_SCORES_CSV = "crop_refine_candidate_scores.csv"
CROP_REFINE_SEGMENTS_CSV = "crop_refine_segments.csv"
CROP_REFINE_CROSSPASS_CSV = "crop_crosspass_agreement.csv"
CROP_REFINE_REPORT_JSON = "crop_refine_report.json"
CROP_REFINE_PREVIEW_MP4 = "crop_refine_preview.mp4"
CROP_REFINE_DEBUG_IMAGES_DIR = "crop_refine_debug_images"

OUTLIER_MINIMIZED_POSE_CSV = "outlier_minimized_pose.csv"
OUTLIER_MINIMIZED_POSE_JSONL = "outlier_minimized_pose.jsonl"
OUTLIER_MINIMIZED_REPORT_JSON = "outlier_minimized_report.json"
OUTLIER_MINIMIZED_TEMPORAL_SPIKE_REPORT_CSV = (
    "outlier_minimized_temporal_spike_report.csv"
)
OUTLIER_MINIMIZED_TRAJECTORY_BREAKS_CSV = "outlier_minimized_trajectory_breaks.csv"

TRAJECTORY_EXPORT_POINTS_CSV = "trajectory_export_points.csv"
TRAJECTORY_EXPORT_SEGMENTS_CSV = "trajectory_export_segments.csv"
TRAJECTORY_EXPORT_REPORT_JSON = "trajectory_export_report.json"


def stage_dir(base_dir: str | Path, stage_name: str) -> Path:
    """Return a stage folder under a session/output directory."""
    return Path(base_dir) / stage_name


def normalize_stage_output_dir(output_dir: str | Path, stage_name: str) -> Path:
    """Return a stage output folder while preserving explicit stage variants."""
    output = Path(output_dir)
    name = output.name
    if name == stage_name or name.startswith(f"{stage_name}_"):
        return output
    return output / stage_name


def resolve_existing_file(
    base_dir: str | Path,
    preferred_name: str,
    legacy_names: tuple[str, ...] = (),
) -> Path:
    """Prefer the new name, but allow reading existing legacy outputs."""
    base = Path(base_dir)
    preferred = base / preferred_name
    if preferred.exists():
        return preferred
    for legacy_name in legacy_names:
        legacy = base / legacy_name
        if legacy.exists():
            return legacy
    return preferred


def session_id_from_report(report_json: Path, default: str) -> str:
    if report_json.exists():
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            return str(payload.get("session_id") or default)
        except Exception:
            pass
    return default


def blender_blend_filename(session_id: str) -> str:
    return f"blender_{session_id}_trajectory.blend"
