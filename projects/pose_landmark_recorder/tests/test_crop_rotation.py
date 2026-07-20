import numpy as np
import pandas as pd

from dance_pose_recorder.crop_refiner import CropSegment
from dance_pose_recorder.crop_rotation import (
    body_axis_angle_deg,
    build_inverted_segments,
    detect_width_for_z,
    rotate_crop,
    snap_rotation,
    unrotate_direction,
    unrotate_norm,
)


def _pose_frame(shoulder_mid, hip_mid):
    rows = []
    for landmark_id, (x, y) in (
        (11, shoulder_mid),
        (12, shoulder_mid),
        (23, hip_mid),
        (24, hip_mid),
    ):
        rows.append({"landmark_id": landmark_id, "x": x, "y": y})
    return pd.DataFrame(rows)


def test_body_axis_angle_upright_and_inverted():
    upright = body_axis_angle_deg(_pose_frame((0.5, 0.3), (0.5, 0.6)), 100, 100)
    inverted = body_axis_angle_deg(_pose_frame((0.5, 0.6), (0.5, 0.3)), 100, 100)
    head_right = body_axis_angle_deg(_pose_frame((0.7, 0.5), (0.4, 0.5)), 100, 100)

    assert abs(upright) < 1e-6
    assert abs(abs(inverted) - 180.0) < 1e-6
    assert abs(head_right - 90.0) < 1e-6


def test_snap_rotation_thresholds():
    assert snap_rotation(30.0, 60.0) == 0
    assert snap_rotation(170.0, 60.0) == 180
    assert snap_rotation(-170.0, 60.0) == 180
    assert snap_rotation(95.0, 60.0) == 90
    assert snap_rotation(-95.0, 60.0) == 270
    assert snap_rotation(None, 60.0) == 0


def test_unrotate_norm_round_trips_against_cv2():
    # Mark one bright pixel, rotate the image with cv2, then check that the
    # inverse mapping recovers the original normalized position.
    height, width = 40, 60
    py, px = 8, 45
    image = np.zeros((height, width), dtype=np.uint8)
    image[py, px] = 255
    for snap in (90, 180, 270):
        rotated = rotate_crop(image, snap)
        yr, xr = np.argwhere(rotated == 255)[0]
        x_norm_rot = (xr + 0.5) / rotated.shape[1]
        y_norm_rot = (yr + 0.5) / rotated.shape[0]
        x_norm, y_norm = unrotate_norm(x_norm_rot, y_norm_rot, snap)
        assert abs(x_norm - (px + 0.5) / width) < 1e-9, snap
        assert abs(y_norm - (py + 0.5) / height) < 1e-9, snap


def test_unrotate_direction_restores_up_vector():
    # A body pointing head-right (snap 90) appears upright after CCW rotation;
    # the detected "up" (0, -1) must map back to screen-right (+x).
    assert unrotate_direction(0.0, -1.0, 90) == (1.0, 0.0)
    assert unrotate_direction(0.0, -1.0, 270) == (-1.0, 0.0)
    assert unrotate_direction(0.0, -1.0, 180) == (0.0, 1.0)


def test_detect_width_for_z_uses_rotated_width():
    assert detect_width_for_z(480.0, 640.0, 0) == 480.0
    assert detect_width_for_z(480.0, 640.0, 180) == 480.0
    assert detect_width_for_z(480.0, 640.0, 90) == 640.0


def test_build_inverted_segments_skips_covered_and_upright():
    pose_frames = {}
    for frame in range(20):
        if 5 <= frame <= 9 or 14 <= frame <= 16:
            pose_frames[frame] = _pose_frame((0.5, 0.6), (0.5, 0.3))  # inverted
        else:
            pose_frames[frame] = _pose_frame((0.5, 0.3), (0.5, 0.6))  # upright
    existing = [
        CropSegment(crop_segment_id=1, start_frame=14, end_frame=18)
    ]

    segments = build_inverted_segments(
        pose_frames, 20, 100, 100, 60.0, existing, max_segment_length=100
    )

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.start_frame, segment.end_frame) == (5, 9)
    assert segment.segment_type == "inverted_pose_segment"
    assert segment.crop_segment_id == 2
    assert "measured" in segment.problem_flags
