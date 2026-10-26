"""Wrist F/T compensation: payload gravity, tare, low-pass, conversion to the base frame.

Source: armos_control/cartesian_admittance.py (`_compensate_wrench`, tare and filter in `_on_telemetry`),
unchanged.

Input: the raw sensor wrench in the sensor frame, MuJoCo convention (force/torque the arm exerts on the
payload side). Output: the wrench the robot exerts ON THE ENVIRONMENT, in the base frame, torque about
the sensor origin. At rest the sensor reads -(m g + F_ext); adding the payload weight back leaves -F_ext.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from armos_core.math.filters import LowPass1
from armos_core.math.so3 import quat_to_rot


@dataclass
class WrenchCompensatorCfg:
    """Attributes:
        payload_mass: Mass below the sensor [kg] (default: Franka Hand incl. fingers).
        payload_com: Payload COM in the sensor frame [m].
        gravity: Gravity in the base frame [m/s^2].
        sensor_in_tcp_quat: Sensor orientation in the TCP frame (w, x, y, z).
        tare_samples: Samples averaged for the bias while holding still at start-up.
        filter_hz: Low-pass cutoff on the compensated wrench; <= 0 disables it.
    """

    payload_mass: float = 0.76
    payload_com: np.ndarray = field(default_factory=lambda: np.array([-0.0096, 0.0, 0.0311]))
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    sensor_in_tcp_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    tare_samples: int = 50
    filter_hz: float = 15.0


class WrenchCompensator:
    """Turns raw F/T readings into the compensated, filtered base-frame wrench."""

    def __init__(self, cfg: WrenchCompensatorCfg | None = None) -> None:
        self.cfg = cfg or WrenchCompensatorCfg()
        c = self.cfg
        self.payload_mass = float(c.payload_mass)
        self.payload_com = np.asarray(c.payload_com, dtype=float)
        self.gravity = np.asarray(c.gravity, dtype=float)
        self.R_tcp_sensor = quat_to_rot(np.asarray(c.sensor_in_tcp_quat, dtype=float))
        self.tare_samples = int(c.tare_samples)
        self.bias: np.ndarray | None = None       # sensor frame, from tare
        self._tare_buffer: list[np.ndarray] = []
        self._filter = LowPass1(c.filter_hz)

    @property
    def tared(self) -> bool:
        """True once the bias has been estimated."""
        return self.bias is not None

    @property
    def wrench(self) -> np.ndarray | None:
        """Last compensated, filtered wrench (None before the first one)."""
        return self._filter.y

    def reset(self) -> None:
        """Drop the bias and the filter state; the next readings tare again."""
        self.bias = None
        self._tare_buffer = []
        self._filter.reset()

    def sensor_rotation(self, R_tcp: np.ndarray) -> np.ndarray:
        """Sensor frame -> base frame rotation, from the TCP orientation in the base frame."""
        return R_tcp @ self.R_tcp_sensor

    def compensate(self, raw: np.ndarray, R_sensor: np.ndarray) -> np.ndarray:
        """Raw sensor wrench -> wrench the robot exerts on the environment, base frame (no tare/filter)."""
        mg = self.payload_mass * self.gravity
        force = R_sensor @ raw[:3] + mg
        torque = R_sensor @ raw[3:] + np.cross(R_sensor @ self.payload_com, mg)
        return np.concatenate((force, torque))

    def update(self, raw: np.ndarray, R_tcp: np.ndarray, dt: float) -> np.ndarray | None:
        """Process one reading taken dt [s] after the previous one.

        Args:
            raw: Raw sensor wrench [f, tau], sensor frame.
            R_tcp: TCP orientation in the base frame (3x3).
            dt: Time since the previous reading [s].

        Returns:
            The compensated, filtered base-frame wrench, or None while the tare is still collecting.
        """
        R_sensor = self.sensor_rotation(R_tcp)
        compensated = self.compensate(np.asarray(raw, dtype=float), R_sensor)

        # Tare: average the residual in the sensor frame (where a sensor bias lives).
        if self.bias is None:
            self._tare_buffer.append(np.concatenate((
                R_sensor.T @ compensated[:3], R_sensor.T @ compensated[3:]
            )))
            if len(self._tare_buffer) < max(1, self.tare_samples):
                return None
            self.bias = np.mean(self._tare_buffer, axis=0)
            self._tare_buffer = []
        compensated[:3] -= R_sensor @ self.bias[:3]
        compensated[3:] -= R_sensor @ self.bias[3:]

        return self._filter.update(compensated, dt)
