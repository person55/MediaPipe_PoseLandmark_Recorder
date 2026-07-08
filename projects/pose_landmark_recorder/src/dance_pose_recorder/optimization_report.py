"""Optimization report helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_optimization_report(
    *,
    metadata: dict,
    input_refined_csv: Path,
    frames_total: int,
    fps: float,
    source: str,
    settings: dict,
    optimized: pd.DataFrame,
    optimization_segments: pd.DataFrame,
    bone_report: pd.DataFrame,
    angle_report: pd.DataFrame,
) -> dict:
    return {
        "session_id": metadata.get("session_id"),
        "input_refined_csv": str(input_refined_csv),
        "frames_total": int(frames_total),
        "fps": float(fps),
        "source": source,
        "settings": settings,
        "counts": {
            "total_rows": int(len(optimized)),
            "pose_world_rows": int((optimized["source"] == "pose_world").sum()),
            "flagged_rows": int((optimized["optimizer_status"] == "flagged").sum()),
            "optimized_constrained_rows": int((optimized["optimizer_status"] == "optimized_constrained").sum()),
            "review_only_rows": int((optimized["optimizer_status"] == "review_only").sum()),
            "optimization_unreliable_rows": int((optimized["optimizer_status"] == "optimization_unreliable").sum()),
        },
        "segments": {
            "optimization_segment_count": int(len(optimization_segments)),
            "review_only_segment_count": int(optimization_segments["review_only"].sum()) if "review_only" in optimization_segments else 0,
            "corrected_segment_count": int(
                (optimization_segments.get("recommended_action", pd.Series(dtype=str)) == "interpolate_short_violation").sum()
            ),
        },
        "bone_summary": _report_summary(bone_report, "bone_name", "violation_count"),
        "angle_summary": _report_summary(angle_report, "joint_name", "violation_count"),
        "notes": "Skeleton optimization uses conservative constraints. It does not generate motion or fill long unreliable runs.",
    }


def write_json(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _report_summary(report: pd.DataFrame, name_column: str, count_column: str) -> dict:
    if report.empty or name_column not in report or count_column not in report:
        return {}
    return {
        str(row[name_column]): int(row[count_column])
        for _, row in report.iterrows()
    }
