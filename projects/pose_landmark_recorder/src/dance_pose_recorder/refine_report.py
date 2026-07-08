"""Report helpers for segment re-detection refinement."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_refine_report(
    metadata: dict,
    input_cleaned_csv: str | Path,
    input_video: str | Path,
    frames_total: int,
    target_landmarks: list[str],
    settings: dict,
    segments: list[dict],
    refined: pd.DataFrame,
) -> dict:
    status_counts = refined["refine_status"].value_counts().to_dict()
    redetected_segments = sum(1 for segment in segments if segment.get("redetected"))
    return {
        "session_id": metadata.get("session_id"),
        "input_cleaned_csv": str(input_cleaned_csv),
        "input_video": str(input_video),
        "frames_total": int(frames_total),
        "candidate_segment_count": len(segments),
        "redetected_segment_count": int(redetected_segments),
        "accepted_row_count": int(status_counts.get("refined_accepted", 0)),
        "rejected_row_count": int(status_counts.get("refined_rejected", 0)),
        "unavailable_row_count": int(status_counts.get("refined_unavailable", 0)),
        "target_landmarks": target_landmarks,
        "settings": settings,
        "segments": segments,
        "notes": (
            "Segment re-detection does not generate motion. It only accepts re-detected candidates "
            "when they score better than the cleaned value."
        ),
    }


def write_json(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
