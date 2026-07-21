import math

from dance_pose_recorder.marker_fade import alpha_from_value, build_marker_alpha_events


def test_alpha_parsing_defaults_and_clamping():
    assert alpha_from_value(None) == 1.0
    assert alpha_from_value("") == 1.0
    assert alpha_from_value("not-a-number") == 1.0
    assert alpha_from_value(float("nan")) == 1.0
    assert alpha_from_value(math.inf) == 1.0
    assert alpha_from_value("-0.5") == 0.0
    assert alpha_from_value(1.7) == 1.0
    assert alpha_from_value("0.35") == 0.35


def test_events_collapse_constant_runs_to_change_points():
    frame_alphas = [(0, 1.0), (1, 1.0), (2, 0.5), (3, 0.5), (4, 0.5), (5, 1.0)]
    assert build_marker_alpha_events(frame_alphas) == [(0, 1.0), (2, 0.5), (5, 1.0)]


def test_all_solid_yields_single_event():
    frame_alphas = [(frame, 1.0) for frame in range(50)]
    assert build_marker_alpha_events(frame_alphas) == [(0, 1.0)]


def test_invalid_values_read_as_solid():
    frame_alphas = [(0, 0.4), (1, ""), (2, 0.4)]
    assert build_marker_alpha_events(frame_alphas) == [(0, 0.4), (1, 1.0), (2, 0.4)]


def test_empty_input():
    assert build_marker_alpha_events([]) == []
