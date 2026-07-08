import pandas as pd

from dance_pose_recorder.segment_refiner import detect_candidate_segments, parse_target_landmarks


def _row(frame, landmark_name="left_wrist", flag="measured", source="pose"):
    return {
        "frame": frame,
        "source": source,
        "landmark_name": landmark_name,
        "quality_flag": flag,
    }


def test_unreliable_rows_create_segment():
    cleaned = pd.DataFrame([_row(2, flag="unreliable"), _row(3, flag="unreliable")])

    segments = detect_candidate_segments(cleaned, total_frames=10, segment_margin=0)

    assert len(segments) == 1
    assert segments[0].start_frame == 2
    assert segments[0].end_frame == 3
    assert segments[0].target_landmarks == {"left_wrist"}


def test_segment_margin_is_applied_and_clamped():
    cleaned = pd.DataFrame([_row(1, flag="unreliable"), _row(2, flag="unreliable")])

    segments = detect_candidate_segments(cleaned, total_frames=5, segment_margin=3)

    assert len(segments) == 1
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == 4


def test_long_segment_is_review_only():
    cleaned = pd.DataFrame([_row(frame, flag="unreliable") for frame in range(20)])

    segments = detect_candidate_segments(cleaned, total_frames=20, segment_margin=0, max_cluster_length=10)

    assert len(segments) == 1
    assert segments[0].review_only is True
    assert segments[0].segment_type == "long_unreliable_run"


def test_missing_long_gap_is_review_only():
    cleaned = pd.DataFrame([_row(frame, flag="missing_long_gap") for frame in range(2, 4)])

    segments = detect_candidate_segments(cleaned, total_frames=10, segment_margin=0, max_cluster_length=90)

    assert len(segments) == 1
    assert segments[0].review_only is True


def test_target_landmark_filter_works():
    cleaned = pd.DataFrame(
        [
            _row(2, landmark_name="left_wrist", flag="unreliable"),
            _row(3, landmark_name="left_wrist", flag="unreliable"),
            _row(2, landmark_name="left_ankle", flag="unreliable"),
            _row(3, landmark_name="left_ankle", flag="unreliable"),
        ]
    )

    segments = detect_candidate_segments(cleaned, total_frames=10, target_landmarks="feet", segment_margin=0)

    assert len(segments) == 1
    assert segments[0].target_landmarks == {"left_ankle"}


def test_measured_rows_do_not_create_segment():
    cleaned = pd.DataFrame([_row(frame, flag="measured") for frame in range(5)])

    assert detect_candidate_segments(cleaned, total_frames=5) == []


def test_parse_target_landmarks_expands_groups():
    names = parse_target_landmarks("arms,hands")

    assert "left_elbow" in names
    assert "right_thumb" in names
    assert "left_ankle" not in names
