"""Report helpers for visualization-oriented outlier minimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json_report(report: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")


def write_frame_jsonl(df: pd.DataFrame, output_path: str | Path, session_id: str | None = None) -> None:
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as file:
        for frame, frame_df in df.groupby("frame", sort=True):
            record = {
                "session_id": session_id or str(frame_df["session_id"].iloc[0]),
                "frame": int(frame),
                "time_sec": float(frame_df["time_sec"].iloc[0]),
                "quality_summary": frame_df["quality_flag"].value_counts().to_dict(),
                "outlier_summary": frame_df["outlier_status"].value_counts().to_dict(),
                "trajectory_summary": {
                    "visible": int(frame_df["trajectory_visible"].fillna(False).astype(bool).sum()),
                    "connected": int(frame_df["trajectory_connect"].fillna(False).astype(bool).sum()),
                },
            }
            file.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
