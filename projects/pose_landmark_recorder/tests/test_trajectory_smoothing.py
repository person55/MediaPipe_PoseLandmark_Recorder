import numpy as np

from dance_pose_recorder.trajectory_smoothing import OneEuroFilter, OneEuroParams, smooth_chain


def test_constant_signal_passes_through():
    values = [0.5] * 20
    result = smooth_chain(values, fps=30.0, params=OneEuroParams())

    assert np.allclose(result, values)


def test_noisy_static_signal_jitter_is_reduced():
    rng = np.random.default_rng(7)
    values = (0.002 * rng.standard_normal(200)).tolist()
    result = smooth_chain(values, fps=30.0, params=OneEuroParams(min_cutoff_hz=1.2, beta=0.6))

    raw_jitter = np.abs(np.diff(values, n=2)).mean()
    smooth_jitter = np.abs(np.diff(result, n=2)).mean()
    assert smooth_jitter < raw_jitter * 0.5


def test_fast_motion_lag_stays_bounded():
    # 3 units/sec ramp at 30fps moves 0.1 units per frame; adaptive cutoff must
    # keep steady-state lag within about one frame of motion.
    values = [3.0 * frame / 30.0 for frame in range(60)]
    result = smooth_chain(values, fps=30.0, params=OneEuroParams(min_cutoff_hz=1.2, beta=1.5))

    lag = max(abs(raw - smooth) for raw, smooth in zip(values[10:], result[10:]))
    assert lag < 0.1


def test_first_sample_after_reset_passes_through():
    one_euro = OneEuroFilter(OneEuroParams(), fps=30.0)
    for value in (0.0, 0.1, 0.2):
        one_euro.filter(value)
    one_euro.reset()

    assert one_euro.filter(5.0) == 5.0
