# armos_core tests

- `test_admittance.py`: `HybridAdmittance` (base / TCP axes, deadband, max deviation) and `ExternalWrench`
  (tare, sign, moment transport). Pure numpy, no simulator.

The controller tests moved from `mujoco_rl_bed/tests/test_controller.py` need a simulated plant, so they
live in `armos_sim/tests/test_controller.py`. Tests for `armos_core.math` and the other laws: placeholder.
