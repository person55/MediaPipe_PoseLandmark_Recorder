import csv
import json

from dance_pose_recorder.data_writer import SessionWriters, make_frame_record


def test_session_writers_write_jsonl_and_csv(tmp_path):
    record = make_frame_record(
        session_id="session_test",
        frame_index=0,
        timestamp_ms=0,
        pose_landmarks=[
            {"id": 0, "name": "nose", "x": 0.1, "y": 0.2, "z": 0.3, "visibility": 0.9, "presence": 0.8}
        ],
        pose_world_landmarks=[
            {"id": 0, "name": "nose", "x": 1.0, "y": 2.0, "z": 3.0, "visibility": 0.9, "presence": 0.8}
        ],
        transformed_landmarks=[{"id": 0, "name": "nose", "tx": 1.0, "ty": -3.0, "tz": -2.0}],
    )

    with SessionWriters(tmp_path, session_id="session_test") as writers:
        writers.write_frame(record)

    jsonl_record = json.loads((tmp_path / "raw_pose.jsonl").read_text(encoding="utf-8").strip())
    assert jsonl_record["session_id"] == "session_test"
    assert jsonl_record["pose_world_landmarks"][0]["name"] == "nose"

    with (tmp_path / "raw_pose.csv").open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["source"] == "pose"
    assert rows[1]["source"] == "pose_world"
    assert rows[1]["tx"] == "1.0"
