import hashlib
import json

from dance_pose_recorder.session_manifest import build_session_manifest, write_session_manifest


def _make_session(tmp_path):
    (tmp_path / "raw").mkdir()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake-video-bytes")
    (tmp_path / "raw/raw_metadata.json").write_text(
        json.dumps(
            {
                "session_id": "test_session",
                "source_path": str(video),
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "model": "pose_landmarker_full",
                "delegate": "cpu",
            }
        )
    )
    (tmp_path / "raw/raw_pose.csv").write_text("frame,x\n0,0.1\n1,0.2\n")
    (tmp_path / "outlier_minimized").mkdir()
    (tmp_path / "outlier_minimized/outlier_minimized_report.json").write_text(
        json.dumps({"settings": {"source": "pose_world"}, "spike_count": 3, "counts": {"total_rows": 2}})
    )
    return video


def test_manifest_records_hashes_counts_and_settings(tmp_path):
    video = _make_session(tmp_path)

    manifest = build_session_manifest(tmp_path)

    assert manifest["session_id"] == "test_session"
    assert manifest["input_video"]["sha256"] == hashlib.sha256(b"fake-video-bytes").hexdigest()
    raw_entry = manifest["stages"]["raw"]["files"][0]
    assert raw_entry["exists"] and raw_entry["rows"] == 2
    outlier_report = manifest["stages"]["outlier_minimized"]["report"]
    assert outlier_report["spike_count"] == 3
    assert outlier_report["settings"] == {"source": "pose_world"}
    assert manifest["environment"]["delegate"] == "cpu"
    assert video.exists()


def test_manifest_tolerates_missing_stages_and_writes_file(tmp_path):
    _make_session(tmp_path)

    output = write_session_manifest(tmp_path, include_input_hash=False)

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert output.name == "manifest.json"
    assert "sha256" not in manifest["input_video"]
    assert manifest["stages"]["cleaned"]["files"][0]["exists"] is False
    assert manifest["stages"]["cleaned"]["report"] is None
