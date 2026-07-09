import pandas as pd

from dance_pose_recorder.crop_bbox import (
    compute_crop_bbox,
    compute_torso_center,
    crop_to_original_norm,
    is_near_crop_edge,
)


def _row(frame, landmark_id, x, y):
    return {
        "frame": frame,
        "landmark_id": landmark_id,
        "x": x,
        "y": y,
    }


def test_torso_center_uses_shoulders_and_hips():
    rows = pd.DataFrame(
        [
            _row(0, 11, 0.4, 0.3),
            _row(0, 12, 0.6, 0.3),
            _row(0, 23, 0.45, 0.7),
            _row(0, 24, 0.55, 0.7),
        ]
    )

    center = compute_torso_center(rows, frame_width=1000, frame_height=1000)

    assert center == (500.0, 500.0)


def test_hand_points_do_not_control_crop_center():
    torso = [
        _row(0, 11, 0.45, 0.4),
        _row(0, 12, 0.55, 0.4),
        _row(0, 23, 0.46, 0.6),
        _row(0, 24, 0.54, 0.6),
    ]
    rows_without_hand = pd.DataFrame(torso)
    rows_with_hand = pd.DataFrame(torso + [_row(0, 19, 0.02, 0.02), _row(0, 20, 0.98, 0.98)])

    center_without_hand = compute_torso_center(rows_without_hand, 1000, 1000)
    center_with_hand = compute_torso_center(rows_with_hand, 1000, 1000)

    assert center_with_hand == center_without_hand


def test_crop_margin_and_square_bbox_are_applied():
    rows = pd.DataFrame(
        [
            _row(0, 11, 0.45, 0.4),
            _row(0, 12, 0.55, 0.4),
            _row(0, 23, 0.46, 0.6),
            _row(0, 24, 0.54, 0.6),
        ]
    )

    small = compute_crop_bbox(rows, 1000, 1000, crop_margin_ratio=1.0, crop_min_size=100)
    large = compute_crop_bbox(rows, 1000, 1000, crop_margin_ratio=2.0, crop_min_size=100)

    assert small is not None
    assert large is not None
    assert small.w == small.h
    assert large.w > small.w


def test_full_body_margin_can_be_smaller_than_torso_margin():
    rows = pd.DataFrame(
        [
            _row(0, 11, 0.49, 0.45),
            _row(0, 12, 0.51, 0.45),
            _row(0, 23, 0.49, 0.55),
            _row(0, 24, 0.51, 0.55),
            _row(0, 27, 0.2, 0.7),
            _row(0, 28, 0.8, 0.7),
        ]
    )

    default = compute_crop_bbox(rows, 1200, 1000, crop_margin_ratio=1.8, crop_min_size=100)
    tight = compute_crop_bbox(
        rows,
        1200,
        1000,
        crop_margin_ratio=1.8,
        full_body_margin_ratio=1.15,
        crop_min_size=100,
    )

    assert default is not None
    assert tight is not None
    assert tight.w < default.w


def test_crop_bbox_is_clamped_to_frame_boundary():
    rows = pd.DataFrame(
        [
            _row(0, 11, 0.01, 0.01),
            _row(0, 12, 0.02, 0.01),
            _row(0, 23, 0.01, 0.02),
            _row(0, 24, 0.02, 0.02),
        ]
    )

    bbox = compute_crop_bbox(rows, 640, 480, crop_margin_ratio=2.0, crop_min_size=512)

    assert bbox is not None
    assert bbox.x0 == 0
    assert bbox.y0 == 0
    assert bbox.w <= 640
    assert bbox.h <= 480


def test_crop_coordinate_restore_and_edge_guard():
    rows = pd.DataFrame([_row(0, 11, 0.5, 0.5), _row(0, 12, 0.6, 0.5)])
    bbox = compute_crop_bbox(rows, 1000, 1000, crop_min_size=200)

    assert bbox is not None
    x, y = crop_to_original_norm(0.5, 0.5, bbox)

    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert is_near_crop_edge(0.01, 0.5) is True
    assert is_near_crop_edge(0.5, 0.5) is False
