import pandas as pd

from dance_pose_recorder.blender_coordinate import to_screen_bottom_origin_position


def _row(x=0.5, y=1.0, z=0.25):
    return pd.Series({"x": x, "y": y, "z": z})


def test_default_origin_center_bottom_maps_to_zero_xz():
    result = to_screen_bottom_origin_position(_row())

    assert result["blender_x"] == 0.0
    assert result["blender_z"] == 0.0


def test_top_of_screen_maps_to_positive_blender_z():
    result = to_screen_bottom_origin_position(_row(y=0.0))

    assert result["blender_z"] > 0.0


def test_left_side_maps_to_negative_blender_x():
    result = to_screen_bottom_origin_position(_row(x=0.0))

    assert result["blender_x"] < 0.0


def test_right_side_maps_to_positive_blender_x():
    result = to_screen_bottom_origin_position(_row(x=1.0))

    assert result["blender_x"] > 0.0


def test_depth_mode_none_sets_blender_y_to_zero():
    result = to_screen_bottom_origin_position(_row(z=0.5), depth_mode="none")

    assert result["blender_y"] == 0.0


def test_depth_mode_pose_z_keeps_z_sign_scaled():
    # z < 0 is closer to the camera; the Blender camera looks from -Y, so a
    # closer landmark must map to a smaller blender_y.
    result = to_screen_bottom_origin_position(_row(z=0.5), depth_mode="pose_z", depth_scale=2.0)

    assert result["blender_y"] == 1.0
