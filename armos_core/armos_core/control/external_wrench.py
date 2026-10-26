"""External wrench estimate (Franka `O_F_ext_hat_K` style): sign, tare, low-pass, moment transport.

Source: franka_custom_control/bimanual/admittance_controller.py (`_on_wrench`, `_to_tcp`), unchanged.

Input: an external wrench in the base frame with the payload already removed (Franka removes the
configured end-effector load; the armos sim bridge publishes the same from its wrist F/T sensor),
torque about a reference point (Franka: O_T_EE). Output: the tared, filtered base-frame wrench, which
`to_axes` moves to the TCP and rotates into the admittance law's axes.

Convention (as in armos): the wrench the robot exerts ON THE ENVIRONMENT. `sign` flips an input that
uses the opposite convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armos_core.math.filters import LowPass1


@dataclass
class ExternalWrenchCfg:
    """Attributes:
        sign: +1 if the input is robot-on-environment, -1 if environment-on-robot.
        tare_samples: Readings averaged for the bias while the robot holds still, untouched.
        tare_max_std: Max std (force [N], torque [Nm]) during the tare; above it the tare restarts
            (robot touched or moving).
        filter_hz: Low-pass cutoff on the tared wrench; <= 0 disables it.
    """

    sign: float = 1.0
    tare_samples: int = 500
    tare_max_std: tuple[float, float] = (1.0, 0.15)
    filter_hz: float = 15.0


class ExternalWrench:
    """Tares and filters a base-frame external wrench.

    After each `update`, `tare_event` is None, "done" (the bias was set) or "rejected" (the readings
    varied too much; `tare_std` holds their std and the tare restarts).
    """

    def __init__(self, cfg: ExternalWrenchCfg | None = None) -> None:
        self.cfg = cfg or ExternalWrenchCfg()
        c = self.cfg
        self.sign = float(c.sign)
        self.tare_samples = int(c.tare_samples)
        self.tare_max_std = np.asarray(c.tare_max_std, dtype=float)
        self.bias: np.ndarray | None = None      # base frame, from the tare (sign applied)
        self.tare_event: str | None = None
        self.tare_std: np.ndarray | None = None
        self._tare_buffer: list[np.ndarray] = []
        self._filter = LowPass1(c.filter_hz)
        self._last_stamp: float | None = None

    @property
    def tared(self) -> bool:
        return self.bias is not None

    @property
    def wrench(self) -> np.ndarray | None:
        """Last tared, filtered base-frame wrench (None before the first one)."""
        return self._filter.y

    def reset(self) -> None:
        """Drop the bias and the filter state; the next readings tare again."""
        self.bias = None
        self._tare_buffer = []
        self._filter.reset()
        self._last_stamp = None

    def update(self, raw: np.ndarray, stamp: float, nominal_dt: float) -> np.ndarray | None:
        """Process one reading.

        Args:
            raw: External wrench [f, tau], base frame.
            stamp: Reading time [s] (message header); sets the filter dt.
            nominal_dt: Filter dt when the stamps give none [s].

        Returns:
            The tared, filtered base-frame wrench, or None while the tare is collecting.
        """
        self.tare_event = None
        raw = self.sign * np.asarray(raw, dtype=float)
        if self.bias is None:
            self._tare_buffer.append(raw)
            if len(self._tare_buffer) < max(1, self.tare_samples):
                return None
            buf = np.asarray(self._tare_buffer)
            self._tare_buffer = []
            std = buf.std(axis=0)
            if std[:3].max() > self.tare_max_std[0] or std[3:].max() > self.tare_max_std[1]:
                self.tare_event, self.tare_std = "rejected", std
                return None
            self.bias = buf.mean(axis=0)
            self._filter.reset()
            self._last_stamp = None
            self.tare_event = "done"
            return None

        dt = nominal_dt
        if self._last_stamp is not None and stamp > self._last_stamp:
            dt = min(stamp - self._last_stamp, 0.1)
        self._last_stamp = stamp
        return self._filter.update(raw - self.bias, dt)


def to_axes(
    wrench_base: np.ndarray,
    tcp_pos: np.ndarray,
    R_axes: np.ndarray | None,
    point: np.ndarray | None = None,
) -> np.ndarray:
    """Base-frame wrench about `point` -> wrench about the TCP, in the given axes.

    Args:
        wrench_base: [f, tau] in the base frame, torque about `point`.
        tcp_pos: TCP position, base frame.
        R_axes: Axes -> base rotation (the TCP orientation for TCP axes); None = base axes.
        point: Torque reference point (base frame); None = already about the TCP.
    """
    f, t = wrench_base[:3], wrench_base[3:]
    if point is not None:
        t = t + np.cross(point - tcp_pos, f)       # t_tcp = t_p + (p - p_tcp) x f
    if R_axes is None:
        return np.concatenate((f, t))
    R_T = R_axes.T
    return np.concatenate((R_T @ f, R_T @ t))
