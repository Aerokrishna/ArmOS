"""Signal filters. Source: armos_control/cartesian_admittance.py (wrench low-pass)."""

from __future__ import annotations

import math

import numpy as np

# self.y = self.y + alpha * (x - self.y)
# x is the new measurement, we update the new measurement base on alpha
# Lower cutoff → more smoothing but more delay. Higher cutoff → less smoothing but faster response.
class LowPass1:
    """First-order low-pass with a dt-aware gain: y += alpha (x - y), alpha = 1 - exp(-2 pi f dt).

    The first sample passes through unchanged. A cutoff <= 0 disables filtering.
    """

    def __init__(self, cutoff_hz: float) -> None:
        self.cutoff_hz = float(cutoff_hz)
        self.y: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the state; the next sample passes through."""
        self.y = None

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Filter one sample taken dt [s] after the previous one."""
        x = np.asarray(x, dtype=float)
        if self.y is None or self.cutoff_hz <= 0.0:
            self.y = x.copy()
        else:
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt)
            self.y = self.y + alpha * (x - self.y)
        return self.y
