from dance_pose_recorder.coordinate_transform import CoordinateTransformer, pelvis_center


def _landmark(landmark_id, x, y, z):
    return {"id": landmark_id, "name": str(landmark_id), "x": x, "y": y, "z": z}


def test_pelvis_center():
    landmarks = [_landmark(23, 1.0, 2.0, 3.0), _landmark(24, 3.0, 4.0, 5.0)]
    center = pelvis_center(landmarks)
    assert center.x == 2.0
    assert center.y == 3.0
    assert center.z == 4.0


def test_raw_axis_mapping():
    transformer = CoordinateTransformer(origin_policy="raw")
    result = transformer.transform_landmarks([_landmark(0, 1.0, 2.0, -3.0)])
    assert result[0]["tx"] == 1.0
    assert result[0]["ty"] == 3.0
    assert result[0]["tz"] == -2.0


def test_first_frame_pelvis_origin_is_reused():
    transformer = CoordinateTransformer(origin_policy="first_frame_pelvis")
    first = [_landmark(23, 1, 1, 1), _landmark(24, 3, 1, 1), _landmark(0, 4, 2, -1)]
    second = [_landmark(23, 10, 1, 1), _landmark(24, 12, 1, 1), _landmark(0, 4, 2, -1)]

    first_result = transformer.transform_landmarks(first)
    second_result = transformer.transform_landmarks(second)

    assert first_result[2]["tx"] == 2.0
    assert second_result[2]["tx"] == 2.0
