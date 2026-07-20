import numpy as np

from dance_pose_recorder.crop_enhancement import detection_is_weak, enhance_crop, mean_visibility


def test_enhance_crop_raises_contrast_on_dark_image():
    rng = np.random.default_rng(3)
    dark = (rng.integers(10, 50, size=(64, 64, 3))).astype(np.uint8)

    enhanced = enhance_crop(dark)

    assert enhanced.shape == dark.shape
    assert float(enhanced.std()) > float(dark.std())


def test_detection_is_weak_on_missing_or_low_visibility():
    assert detection_is_weak([], 0.5)
    weak = [{"visibility": 0.2}, {"visibility": 0.3}]
    strong = [{"visibility": 0.9}, {"visibility": 0.8}]
    assert detection_is_weak(weak, 0.5)
    assert not detection_is_weak(strong, 0.5)


def test_mean_visibility_ignores_missing_values():
    landmarks = [{"visibility": 0.4}, {"visibility": None}, {"visibility": 0.8}]
    assert abs(mean_visibility(landmarks) - 0.6) < 1e-9
