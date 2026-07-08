import pandas as pd

from dance_pose_recorder.interpolation import find_short_gaps, interpolate_group_linear


def test_one_frame_gap_is_interpolated():
    group = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "tx": [0.0, None, 2.0],
            "stable": [True, False, True],
        }
    )
    result, segments = interpolate_group_linear(group, ["tx"], "stable", max_gap=2)

    assert len(segments) == 1
    assert result.loc[result["frame"] == 1, "tx"].iloc[0] == 1.0


def test_two_frame_gap_is_interpolated():
    group = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "tx": [0.0, None, None, 3.0],
            "stable": [True, False, False, True],
        }
    )
    result, segments = interpolate_group_linear(group, ["tx"], "stable", max_gap=2)

    assert len(segments) == 1
    assert result.loc[result["frame"] == 1, "tx"].iloc[0] == 1.0
    assert result.loc[result["frame"] == 2, "tx"].iloc[0] == 2.0


def test_gap_longer_than_max_is_not_interpolated():
    group = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "tx": [0.0, None, None, 3.0],
            "stable": [True, False, False, True],
        }
    )
    result, segments = interpolate_group_linear(group, ["tx"], "stable", max_gap=1)

    assert segments == []
    assert pd.isna(result.loc[result["frame"] == 1, "tx"].iloc[0])
    assert pd.isna(result.loc[result["frame"] == 2, "tx"].iloc[0])


def test_start_gap_is_not_interpolated():
    segments = find_short_gaps([0, 1, 2], [False, False, True], max_gap=2)
    assert segments == []


def test_end_gap_is_not_interpolated():
    segments = find_short_gaps([0, 1, 2], [True, False, False], max_gap=2)
    assert segments == []


def test_non_candidate_gap_is_not_interpolated():
    group = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "tx": [0.0, 10.0, 2.0],
            "stable": [True, False, True],
            "candidate": [False, False, False],
        }
    )
    result, segments = interpolate_group_linear(group, ["tx"], "stable", max_gap=2, candidate_column="candidate")

    assert segments == []
    assert result.loc[result["frame"] == 1, "tx"].iloc[0] == 10.0


def test_candidate_gap_is_interpolated():
    group = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "tx": [0.0, None, 2.0],
            "stable": [True, False, True],
            "candidate": [False, True, False],
        }
    )
    result, segments = interpolate_group_linear(group, ["tx"], "stable", max_gap=2, candidate_column="candidate")

    assert len(segments) == 1
    assert result.loc[result["frame"] == 1, "tx"].iloc[0] == 1.0
