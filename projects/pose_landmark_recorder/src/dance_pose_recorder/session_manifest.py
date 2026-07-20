"""Reproducibility manifest for a pipeline session.

Records everything needed to re-verify reported numbers later even when the
session outputs are not in the work tree: input checksum, git revision, stage
settings and counts (from stage reports), and content hashes plus row counts
of the key CSV outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

MANIFEST_FILENAME = "manifest.json"

# stage name -> (report path, key CSV paths), all relative to the session dir.
STAGE_LAYOUT: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "raw": (None, ("raw/raw_pose.csv",)),
    "cleaned": ("cleaned/cleaned_interpolation_report.json", ("cleaned/cleaned_pose.csv",)),
    "crop_refine": ("crop_refine/crop_refine_report.json", ("crop_refine/crop_refine_pose.csv",)),
    "refined": ("refined_after_crop_v1/refine_report.json", ("refined_after_crop_v1/refined_pose.csv",)),
    "outlier_minimized": (
        "outlier_minimized/outlier_minimized_report.json",
        ("outlier_minimized/outlier_minimized_pose.csv",),
    ),
    "trajectory_export": (
        "trajectory_export/trajectory_export_report.json",
        (
            "trajectory_export/trajectory_export_points.csv",
            "trajectory_export/trajectory_export_segments.csv",
        ),
    ),
}

REPORT_COUNT_KEYS = (
    "counts",
    "status_counts",
    "spike_count",
    "trajectory_break_count",
    "accepted_rows",
    "rejected_rows",
)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - 1)


def git_revision(repo_hint: Path) -> dict:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=repo_hint, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")) if commit else None,
    }


def _file_entry(path: Path) -> dict:
    entry: dict = {"path": str(path), "exists": path.exists()}
    if path.exists():
        entry["sha256"] = sha256_of_file(path)
        entry["bytes"] = path.stat().st_size
        if path.suffix == ".csv":
            entry["rows"] = csv_row_count(path)
    return entry


def _report_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "unreadable report"}
    summary: dict = {"settings": report.get("settings")}
    for key in REPORT_COUNT_KEYS:
        if key in report:
            summary[key] = report[key]
    return summary


def build_session_manifest(session_dir: Path, include_input_hash: bool = True) -> dict:
    session_dir = Path(session_dir)
    metadata_path = session_dir / "raw/raw_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    manifest: dict = {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": metadata.get("session_id") or session_dir.name,
        "session_dir": str(session_dir),
        "git": git_revision(session_dir),
        "environment": {
            "model": metadata.get("model"),
            "model_path": metadata.get("model_path"),
            "delegate": metadata.get("delegate"),
            "fps": metadata.get("fps"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "python_version": metadata.get("python_version"),
            "platform": metadata.get("platform"),
        },
    }

    source_path = metadata.get("source_path")
    input_entry: dict = {"path": source_path}
    if source_path and include_input_hash:
        source = Path(source_path)
        input_entry["exists"] = source.exists()
        if source.exists():
            input_entry["sha256"] = sha256_of_file(source)
            input_entry["bytes"] = source.stat().st_size
    manifest["input_video"] = input_entry

    stages: dict = {}
    for stage_name, (report_rel, csv_rels) in STAGE_LAYOUT.items():
        stage: dict = {}
        if report_rel:
            stage["report"] = _report_summary(session_dir / report_rel)
        stage["files"] = [_file_entry(session_dir / rel) for rel in csv_rels]
        stages[stage_name] = stage
    manifest["stages"] = stages
    return manifest


def write_session_manifest(session_dir: Path, include_input_hash: bool = True) -> Path:
    manifest = build_session_manifest(session_dir, include_input_hash=include_input_hash)
    output_path = Path(session_dir) / MANIFEST_FILENAME
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
