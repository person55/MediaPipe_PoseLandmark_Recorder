import math

import numpy as np

from dance_pose_recorder.joint_angle import joint_angle_deg


def test_straight_angle_is_180_degrees():
    angle = joint_angle_deg((0, 0, 0), (1, 0, 0), (2, 0, 0))

    assert math.isclose(angle, 180.0, abs_tol=1e-6)


def test_right_angle_is_90_degrees():
    angle = joint_angle_deg((1, 0, 0), (0, 0, 0), (0, 1, 0))

    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_zero_vector_returns_nan():
    angle = joint_angle_deg((0, 0, 0), (0, 0, 0), (0, 1, 0))

    assert math.isnan(angle)


def test_nan_input_returns_nan():
    angle = joint_angle_deg((np.nan, 0, 0), (0, 0, 0), (0, 1, 0))

    assert math.isnan(angle)
