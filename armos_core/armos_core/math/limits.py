"""Vector limits. Source: armos_control/utils/kinematics.py."""

from __future__ import annotations

import numpy as np

# limit the maximum magnitude of any given vector
def limit_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale v down so that |v| <= max_norm (no-op if max_norm <= 0)."""
    norm = float(np.linalg.norm(v))
    if norm > max_norm > 0.0:
        return v * (max_norm / norm)
    return v


def deadband(v: np.ndarray, band: float) -> np.ndarray:
    """Per component: zero inside +-band, shifted towards zero by band outside (no-op if band <= 0).

    Source: franka_custom_control/bimanual/admittance_controller.py.
    """
    if band <= 0.0:
        return v
    return np.sign(v) * np.maximum(np.abs(v) - band, 0.0)
