"""JSONL and CSV session writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dance_pose_recorder.output_layout import RAW_POSE_CSV, RAW_POSE_JSONL


CSV_FIELDS = [
    "session_id",
    "frame",
    "time_sec",
    "landmark_id",
    "landmark_name",
    "source",
    "x",
    "y",
    "z",
    "visibility",
    "presence",
    "tx",
    "ty",
    "tz",
]


class SessionWriters:
    def __init__(
        self,
        output_dir: str | Path,
        session_id: str,
        save_jsonl: bool = True,
        save_csv: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._jsonl_file = None
        self._csv_file = None
        self._csv_writer = None

        if save_jsonl:
            self._jsonl_file = (self.output_dir / RAW_POSE_JSONL).open("w", encoding="utf-8")
        if save_csv:
            self._csv_file = (self.output_dir / RAW_POSE_CSV).open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
            self._csv_writer.writeheader()

    def write_frame(self, frame_record: dict) -> None:
        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
        if self._csv_writer:
            for row in frame_to_csv_rows(frame_record):
                self._csv_writer.writerow(row)

    def close(self) -> None:
        if self._jsonl_file:
            self._jsonl_file.close()
        if self._csv_file:
            self._csv_file.close()

    def __enter__(self) -> "SessionWriters":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def make_frame_record(
    session_id: str,
    frame_index: int,
    timestamp_ms: int,
    pose_landmarks: list[dict],
    pose_world_landmarks: list[dict],
    transformed_landmarks: list[dict],
) -> dict:
    return {
        "session_id": session_id,
        "frame": frame_index,
        "time_sec": timestamp_ms / 1000.0,
        "pose_landmarks": pose_landmarks,
        "pose_world_landmarks": pose_world_landmarks,
        "transformed_landmarks": transformed_landmarks,
    }


def frame_to_csv_rows(frame_record: dict) -> list[dict]:
    transformed_by_id = {
        landmark["id"]: landmark for landmark in frame_record.get("transformed_landmarks", [])
    }
    rows = []
    for source, landmarks in (
        ("pose", frame_record.get("pose_landmarks", [])),
        ("pose_world", frame_record.get("pose_world_landmarks", [])),
    ):
        for landmark in landmarks:
            transformed = transformed_by_id.get(landmark["id"], {}) if source == "pose_world" else {}
            rows.append(
                {
                    "session_id": frame_record["session_id"],
                    "frame": frame_record["frame"],
                    "time_sec": frame_record["time_sec"],
                    "landmark_id": landmark["id"],
                    "landmark_name": landmark["name"],
                    "source": source,
                    "x": landmark["x"],
                    "y": landmark["y"],
                    "z": landmark["z"],
                    "visibility": landmark.get("visibility"),
                    "presence": landmark.get("presence"),
                    "tx": transformed.get("tx"),
                    "ty": transformed.get("ty"),
                    "tz": transformed.get("tz"),
                }
            )
    return rows
