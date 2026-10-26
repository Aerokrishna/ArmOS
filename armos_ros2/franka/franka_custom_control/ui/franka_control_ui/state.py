"""Thread-safe robot state shared between the ROS executor thread and the Qt thread."""

import math
import threading
import time

import numpy as np

from .transforms import quat_to_euler_xyz, unwrap_to


class Ring:
    """Fixed-capacity time series buffer."""

    def __init__(self, capacity, width):
        self.t = np.full(capacity, np.nan)
        self.v = np.full((capacity, width), np.nan)
        self.cap = capacity
        self.i = 0
        self.n = 0

    def append(self, t, row):
        self.t[self.i] = t
        self.v[self.i] = row
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def snapshot(self):
        if self.n < self.cap:
            return self.t[:self.n].copy(), self.v[:self.n].copy()
        idx = np.r_[self.i:self.cap, 0:self.i]
        return self.t[idx], self.v[idx]

    def clear(self):
        self.t[:] = np.nan
        self.v[:] = np.nan
        self.i = 0
        self.n = 0


class RateMeter:
    def __init__(self):
        self.count = 0
        self.last_reset = time.monotonic()
        self.hz = 0.0

    def tick(self):
        self.count += 1

    def update(self, now):
        dt = now - self.last_reset
        if dt >= 1.0:
            self.hz = self.count / dt
            self.count = 0
            self.last_reset = now


def _euler_continuous(quat, previous):
    """Euler XYZ (rad) unwrapped against a previous sample so plots never jump at +-pi."""
    e = quat_to_euler_xyz(quat)
    if previous is not None and not np.any(np.isnan(previous)):
        e = np.array([unwrap_to(a, p) for a, p in zip(e, previous)])
    return e


class RobotState:
    STREAMS = ('pose', 'wrench', 'joints')

    def __init__(self, buffer_seconds=60.0, sample_hz=100.0, n_joints=7):
        self.lock = threading.Lock()
        self.n_joints = n_joints
        self.sample_dt = 1.0 / sample_hz
        cap = int(buffer_seconds * sample_hz)
        self.t0 = time.monotonic()

        self.pose_pos = None
        self.pose_quat = None
        self.wrench = None
        self.q = None
        self.last_rx = {s: 0.0 for s in self.STREAMS}
        self.rates = {s: RateMeter() for s in self.STREAMS}

        # Last values this UI published (None until first publish).
        self.target_pos = None
        self.target_quat = None
        self.target_q = None
        self.target_wrench = np.zeros(6)
        self.target_stiffness = None

        self.buf = {
            'pose': Ring(cap, 6),
            'pose_target': Ring(cap, 6),
            'wrench': Ring(cap, 6),
            'joints': Ring(cap, n_joints),
            'joints_target': Ring(cap, n_joints),
        }
        self._last_sample = {k: -1.0 for k in self.buf}
        self._last_euler = None
        self._last_target_euler = None

    def now(self):
        return time.monotonic() - self.t0

    def _due(self, key, t):
        if t - self._last_sample[key] >= self.sample_dt:
            self._last_sample[key] = t
            return True
        return False

    # ----- writers (ROS thread) -----
    def update_pose(self, pos, quat):
        t = self.now()
        with self.lock:
            self.pose_pos = pos
            self.pose_quat = quat
            self.last_rx['pose'] = t
            self.rates['pose'].tick()
            if self._due('pose', t):
                e = _euler_continuous(quat, self._last_euler)
                self._last_euler = e
                self.buf['pose'].append(t, np.r_[pos, np.degrees(e)])
                if self.target_pos is not None:
                    te = _euler_continuous(self.target_quat, e)
                    self.buf['pose_target'].append(
                        t, np.r_[self.target_pos, np.degrees(te)])
                else:
                    self.buf['pose_target'].append(t, np.full(6, np.nan))

    def update_wrench(self, w):
        t = self.now()
        with self.lock:
            self.wrench = w
            self.last_rx['wrench'] = t
            self.rates['wrench'].tick()
            if self._due('wrench', t):
                self.buf['wrench'].append(t, w)

    def update_joints(self, q):
        t = self.now()
        with self.lock:
            self.q = q
            self.last_rx['joints'] = t
            self.rates['joints'].tick()
            if self._due('joints', t):
                self.buf['joints'].append(t, np.degrees(q))
                tq = self.target_q if self.target_q is not None else np.full(self.n_joints, np.nan)
                self.buf['joints_target'].append(t, np.degrees(tq))

    # ----- writers (Qt thread) -----
    def set_target_pose(self, pos, quat):
        with self.lock:
            self.target_pos = np.array(pos, dtype=float)
            self.target_quat = np.array(quat, dtype=float)

    def set_target_joints(self, q):
        with self.lock:
            self.target_q = np.array(q, dtype=float)

    def set_target_wrench(self, w):
        with self.lock:
            self.target_wrench = np.array(w, dtype=float)

    def set_target_stiffness(self, k):
        with self.lock:
            self.target_stiffness = np.array(k, dtype=float)

    def clear_buffers(self):
        with self.lock:
            for b in self.buf.values():
                b.clear()
            self._last_euler = None

    # ----- readers (Qt thread) -----
    def latest(self):
        with self.lock:
            now = self.now()
            for r in self.rates.values():
                r.update(time.monotonic())
            return {
                'pose_pos': None if self.pose_pos is None else self.pose_pos.copy(),
                'pose_quat': None if self.pose_quat is None else self.pose_quat.copy(),
                'wrench': None if self.wrench is None else self.wrench.copy(),
                'q': None if self.q is None else self.q.copy(),
                'age': {s: (now - self.last_rx[s]) if self.last_rx[s] > 0 else math.inf
                        for s in self.STREAMS},
                'hz': {s: self.rates[s].hz for s in self.STREAMS},
                'target_pos': None if self.target_pos is None else self.target_pos.copy(),
                'target_quat': None if self.target_quat is None else self.target_quat.copy(),
                'target_q': None if self.target_q is None else self.target_q.copy(),
                'now': now,
            }

    def snapshot(self, key):
        with self.lock:
            return self.buf[key].snapshot()
