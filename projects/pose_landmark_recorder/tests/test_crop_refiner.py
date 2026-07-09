import pandas as pd

from dance_pose_recorder.crop_refiner import crop_segments_to_dataframe, detect_crop_segments, parse_crop_target_landmarks


def _row(frame, landmark_name="left_wrist", flag="measured", source="pose"):
    return {
        "frame": frame,
        "source": source,
        "landmark_name": landmark_name,
        "quality_flag": flag,
    }


def test_cleaned_pose_rows_create_crop_segment():
    cleaned = pd.DataFrame([_row(2, flag="unreliable"), _row(3, flag="unreliable")])

    segments = detect_crop_segments(cleaned, total_frames=10, segment_margin=0)

    assert len(segments) == 1
    assert segments[0].start_frame == 2
    assert segments[0].end_frame == 3
    assert segments[0].segment_type == "short_invalid_cluster"
    assert segments[0].crop_attempted is False
    assert segments[0].selected_for_crop is False
    assert segments[0].selection_reason == "excluded_segment_type_short_invalid_cluster"


def test_mixed_problem_segment_is_selected_by_default():
    cleaned = pd.DataFrame(
        [
            _row(2, landmark_name="left_wrist", flag="unreliable"),
            _row(3, landmark_name="left_wrist", flag="unreliable"),
            _row(2, landmark_name="left_elbow", flag="estimated_occluded_arm"),
            _row(3, landmark_name="left_elbow", flag="estimated_occluded_arm"),
        ]
    )

    segments = detect_crop_segments(cleaned, total_frames=10, segment_margin=0)

    assert len(segments) == 1
    assert segments[0].segment_type == "mixed_problem_segment"
    assert segments[0].selected_for_crop is True
    assert segments[0].crop_attempted is True
    assert segments[0].selection_reason == "selected_mixed_problem_segment"


def test_missing_long_gap_is_excluded_by_default():
    cleaned = pd.DataFrame([_row(frame, flag="missing_long_gap") for frame in range(2, 6)])

    segments = detect_crop_segments(
        cleaned,
        total_frames=10,
        segment_margin=0,
        target_flags="missing_long_gap",
    )

    assert len(segments) == 1
    assert segments[0].selected_for_crop is False
    assert segments[0].crop_attempted is False
    assert segments[0].selection_reason == "excluded_missing_long_gap"


def test_long_crop_segment_is_review_only_and_not_attempted():
    cleaned = pd.DataFrame(
        [
            _row(frame, landmark_name="left_wrist", flag="unreliable")
            for frame in range(20)
        ]
        + [
            _row(frame, landmark_name="left_elbow", flag="estimated_occluded_arm")
            for frame in range(20)
        ]
    )

    segments = detect_crop_segments(cleaned, total_frames=20, segment_margin=0, max_segment_length=10)

    assert len(segments) == 1
    assert segments[0].review_only is True
    assert segments[0].crop_attempted is False
    assert segments[0].selected_for_crop is False
    assert segments[0].selection_reason == "excluded_too_long"


def test_segment_margin_is_clamped():
    cleaned = pd.DataFrame([_row(1, flag="unreliable"), _row(2, flag="unreliable")])

    segments = detect_crop_segments(cleaned, total_frames=5, segment_margin=3)

    assert len(segments) == 1
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == 4


def test_target_landmarks_filter_works():
    cleaned = pd.DataFrame(
        [
            _row(2, landmark_name="left_wrist", flag="unreliable"),
            _row(3, landmark_name="left_wrist", flag="unreliable"),
            _row(2, landmark_name="left_ankle", flag="unreliable"),
            _row(3, landmark_name="left_ankle", flag="unreliable"),
        ]
    )

    segments = detect_crop_segments(cleaned, total_frames=10, target_landmarks="feet", segment_margin=0)

    assert len(segments) == 1
    assert segments[0].target_landmarks == {"left_ankle"}
    assert segments[0].selected_for_crop is False


def test_include_short_invalid_cluster_allows_short_segments():
    cleaned = pd.DataFrame([_row(2, flag="unreliable"), _row(3, flag="unreliable")])

    segments = detect_crop_segments(
        cleaned,
        total_frames=10,
        segment_margin=0,
        include_short_invalid_cluster=True,
    )

    assert len(segments) == 1
    assert segments[0].selected_for_crop is True
    assert segments[0].selection_reason == "selected_short_invalid_cluster"


def test_parse_crop_target_landmarks_expands_groups():
    names = parse_crop_target_landmarks("arms,hands_proxy")

    assert "left_elbow" in names
    assert "right_thumb" in names
    assert "left_ankle" not in names


def test_crop_segments_dataframe_includes_selection_policy():
    cleaned = pd.DataFrame([_row(2, flag="unreliable"), _row(3, flag="unreliable")])
    segments = detect_crop_segments(cleaned, total_frames=10, segment_margin=0)

    df = crop_segments_to_dataframe(segments)

    assert "selected_for_crop" in df.columns
    assert "selection_reason" in df.columns
    assert df["selected_for_crop"].iloc[0] == False  # noqa: E712
