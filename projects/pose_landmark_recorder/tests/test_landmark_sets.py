from dance_pose_recorder.landmark_sets import get_landmark_names


def test_blender_default_excludes_ears():
    names = get_landmark_names("blender_default")

    assert "left_ear" not in names
    assert "right_ear" not in names


def test_blender_default_excludes_thumbs():
    names = get_landmark_names("blender_default")

    assert "left_thumb" not in names
    assert "right_thumb" not in names


def test_blender_default_excludes_hand_index():
    names = get_landmark_names("blender_default")

    assert "left_index" not in names
    assert "right_index" not in names


def test_blender_default_keeps_foot_index():
    names = get_landmark_names("blender_default")

    assert "left_foot_index" in names
    assert "right_foot_index" in names


def test_blender_default_keeps_pinky_and_nose():
    names = get_landmark_names("blender_default")

    assert "left_pinky" in names
    assert "right_pinky" in names
    assert "nose" in names


def test_custom_include_and_exclude_work():
    names = get_landmark_names("custom", include_landmarks=["nose", "left_thumb"], exclude_landmarks=["left_thumb"])

    assert names == ["nose"]
