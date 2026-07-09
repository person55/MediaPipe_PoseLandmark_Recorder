from dance_pose_recorder.trajectory_policy import default_trajectory_policy


def test_measured_is_visible_and_connected():
    policy = default_trajectory_policy("measured", "left_wrist")

    assert policy.visible is True
    assert policy.connect is True
    assert policy.alpha == 1.0


def test_missing_long_gap_is_hidden_and_disconnected():
    policy = default_trajectory_policy("missing_long_gap", "left_wrist")

    assert policy.visible is False
    assert policy.connect is False
    assert policy.alpha == 0.0


def test_unreliable_is_visible_but_disconnected_with_low_alpha():
    policy = default_trajectory_policy("unreliable", "left_wrist")

    assert policy.visible is True
    assert policy.connect is False
    assert policy.alpha < 0.5


def test_hands_proxy_unreliable_disconnects():
    policy = default_trajectory_policy("unreliable", "left_thumb")

    assert policy.connect is False
    assert policy.reason == "hands_proxy_unreliable"


def test_estimated_occluded_arm_is_faded():
    policy = default_trajectory_policy("estimated_occluded_arm", "left_elbow")

    assert policy.visible is True
    assert policy.connect is False
    assert policy.reason == "faded_occluded_arm"
