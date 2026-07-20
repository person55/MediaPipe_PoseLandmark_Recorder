"""One-Euro smoothing for the trajectory export visualization layer.

The filter adapts its cutoff to speed: strong smoothing at low speed (kills
sub-spike jitter) and light smoothing at high speed (no lag on fast motion).
Smoothed values are a derived visualization layer; raw coordinates are kept
alongside them, and filters reset at every trajectory break or frame gap so
smoothing never bridges disconnected chains.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OneEuroParams:
    min_cutoff_hz: float = 1.2
    beta: float = 1.5
    d_cutoff_hz: float = 1.0


class OneEuroFilter:
    """Standard One-Euro filter (Casiez et al. 2012) over a fixed-rate signal."""

    def __init__(self, params: OneEuroParams, fps: float) -> None:
        self._params = params
        self._dt = 1.0 / float(fps) if fps > 0 else 1.0 / 30.0
        self._prev_value: float | None = None
        self._prev_derivative = 0.0

    def reset(self) -> None:
        self._prev_value = None
        self._prev_derivative = 0.0

    def _alpha(self, cutoff_hz: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / self._dt)

    def filter(self, value: float) -> float:
        if self._prev_value is None:
            self._prev_value = value
            self._prev_derivative = 0.0
            return value
        derivative = (value - self._prev_value) / self._dt
        d_alpha = self._alpha(self._params.d_cutoff_hz)
        smoothed_derivative = d_alpha * derivative + (1.0 - d_alpha) * self._prev_derivative
        cutoff = self._params.min_cutoff_hz + self._params.beta * abs(smoothed_derivative)
        alpha = self._alpha(cutoff)
        smoothed = alpha * value + (1.0 - alpha) * self._prev_value
        self._prev_value = smoothed
        self._prev_derivative = smoothed_derivative
        return smoothed


def smooth_chain(
    values: list[float],
    fps: float,
    params: OneEuroParams,
) -> list[float]:
    """Smooth one connected chain of samples; the first sample passes through."""

    one_euro = OneEuroFilter(params, fps)
    return [one_euro.filter(value) for value in values]
