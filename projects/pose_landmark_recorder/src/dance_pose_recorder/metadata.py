"""Session metadata helpers."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from dance_pose_recorder.output_layout import RAW_METADATA_JSON


def build_metadata(
    session_id: str,
    source_type: str,
    source_path: str,
    video_info: object,
    model_path: str,
    delegate: str,
    origin_policy: str,
    output_formats: list[str],
    frame_count_written: int,
) -> dict:
    return {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "source_path": source_path,
        "fps": video_info.fps,
        "width": video_info.width,
        "height": video_info.height,
        "source_frame_count": video_info.frame_count,
        "frame_count_written": frame_count_written,
        "model": "mediapipe_tasks_pose_landmarker",
        "model_path": model_path,
        "delegate": delegate,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "origin_policy": origin_policy,
        "axis_policy": "mediapipe_to_blender_v1",
        "output_formats": output_formats,
        "notes": "Single RGB camera; 3D world coordinates are model-estimated, not calibrated stage coordinates.",
    }


def write_metadata(output_dir: str | Path, metadata: dict) -> Path:
    path = Path(output_dir) / RAW_METADATA_JSON
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
