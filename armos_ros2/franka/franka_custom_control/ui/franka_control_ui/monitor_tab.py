"""MONITOR tab: numeric dashboard of EE pose, tracking error, external wrench and joints."""

import math

import numpy as np
from PyQt5.QtWidgets import QGridLayout, QScrollArea, QVBoxLayout, QWidget

from . import theme
from .transforms import quat_angle_between, quat_to_euler_xyz, unwrap_to
from .widgets import Card, StatTile

NEAR_LIMIT_DEG = 5.0


class MonitorTab(QWidget):

    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self.cfg = bridge.cfg
        self.joint_lo = np.array(self.cfg['joint_lower'])
        self.joint_hi = np.array(self.cfg['joint_upper'])
        self.center = np.radians(self.cfg['ranges.euler_center_deg'])

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QGridLayout(inner)
        lay.setSpacing(12)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        pose = Card('End-effector pose', 'Measured, base frame. Orientation as extrinsic XYZ.')
        g = QGridLayout()
        g.setSpacing(8)
        self.pos_tiles = [StatTile(t, 'm', theme.AXIS[i]) for i, t in enumerate('xyz')]
        self.rot_tiles = [StatTile(f'r{t}', 'deg', theme.AXIS[i]) for i, t in enumerate('XYZ')]
        for i in range(3):
            g.addWidget(self.pos_tiles[i], 0, i)
            g.addWidget(self.rot_tiles[i], 1, i)
        pose.body.addLayout(g)
        pose.layout().addStretch(1)
        lay.addWidget(pose, 0, 0)

        track = Card('Tracking', 'Difference between the last target this UI published and '
                                 'the measured pose.')
        g = QGridLayout()
        self.err_pos = StatTile('position error', 'cm')
        self.err_rot = StatTile('orientation error', 'deg')
        g.addWidget(self.err_pos, 0, 0)
        g.addWidget(self.err_rot, 0, 1)
        track.body.addLayout(g)
        track.layout().addStretch(1)
        lay.addWidget(track, 1, 0)

        fa, ta = self.cfg['limits.wrench_alert']
        wrench = Card('External wrench', f'Estimated by the robot, base frame. Alert above '
                                         f'{fa:.0f} N / {ta:.1f} Nm.')
        g = QGridLayout()
        g.setSpacing(8)
        self.f_tiles = [StatTile(f'F{t}', 'N', theme.AXIS[i]) for i, t in enumerate('xyz')]
        self.t_tiles = [StatTile(f'T{t}', 'Nm', theme.AXIS[i]) for i, t in enumerate('xyz')]
        for i in range(3):
            g.addWidget(self.f_tiles[i], 0, i)
            g.addWidget(self.t_tiles[i], 1, i)
        self.f_mag = StatTile('|F|', 'N', with_bar=True)
        self.t_mag = StatTile('|T|', 'Nm', with_bar=True)
        g.addWidget(self.f_mag, 0, 3)
        g.addWidget(self.t_mag, 1, 3)
        wrench.body.addLayout(g)
        wrench.layout().addStretch(1)
        lay.addWidget(wrench, 0, 1)

        joints = Card('Joint positions', 'Bar shows position within the joint range. '
                                         f'Flags within {NEAR_LIMIT_DEG:.0f}° of a limit.')
        g = QGridLayout()
        g.setSpacing(8)
        self.j_tiles = []
        for i in range(len(bridge.joint_names)):
            t = StatTile(f'q{i + 1}', 'deg', theme.JOINT[i % len(theme.JOINT)], with_bar=True)
            g.addWidget(t, i // 4, i % 4)
            self.j_tiles.append(t)
        joints.body.addLayout(g)
        joints.layout().addStretch(1)
        lay.addWidget(joints, 1, 1)
        lay.setColumnStretch(0, 1)
        lay.setColumnStretch(1, 1)
        lay.setRowStretch(2, 1)

    def update_view(self, lat):
        if lat['pose_pos'] is not None:
            e = quat_to_euler_xyz(lat['pose_quat'])
            tgt_e = None
            if lat['target_quat'] is not None:
                tgt_e = quat_to_euler_xyz(lat['target_quat'])
            for i in range(3):
                sub = '' if lat['target_pos'] is None else f"target {lat['target_pos'][i]:+.4f}"
                self.pos_tiles[i].set(f"{lat['pose_pos'][i]:+.4f}", sub)
                a = math.degrees(unwrap_to(e[i], self.center[i]))
                sub = '' if tgt_e is None else \
                    f'target {math.degrees(unwrap_to(tgt_e[i], self.center[i])):+.1f}'
                self.rot_tiles[i].set(f'{a:+.2f}', sub)
            if lat['target_pos'] is not None:
                d = np.linalg.norm(lat['target_pos'] - lat['pose_pos']) * 100
                a = math.degrees(quat_angle_between(lat['target_quat'], lat['pose_quat']))
                self.err_pos.set(f'{d:.2f}', '', alert='warn' if d > 5 else None)
                self.err_rot.set(f'{a:.2f}', '', alert='warn' if a > 10 else None)

        if lat['wrench'] is not None:
            w = lat['wrench']
            fa, ta = self.cfg['limits.wrench_alert']
            for i in range(3):
                self.f_tiles[i].set(f'{w[i]:+.2f}')
                self.t_tiles[i].set(f'{w[3 + i]:+.3f}')
            fm, tm = float(np.linalg.norm(w[:3])), float(np.linalg.norm(w[3:]))
            self.f_mag.set(f'{fm:.2f}', '⚠ above alert threshold' if fm > fa else '', fm / fa,
                           'crit' if fm > fa else 'warn' if fm > 0.7 * fa else None)
            self.t_mag.set(f'{tm:.3f}', '⚠ above alert threshold' if tm > ta else '', tm / ta,
                           'crit' if tm > ta else 'warn' if tm > 0.7 * ta else None)

        if lat['q'] is not None:
            for i, (t, q) in enumerate(zip(self.j_tiles, lat['q'])):
                lo, hi = self.joint_lo[i], self.joint_hi[i]
                margin = math.degrees(min(q - lo, hi - q))
                near = margin < NEAR_LIMIT_DEG
                t.set(f'{math.degrees(q):+.2f}',
                      f'{q:+.4f} rad' + (f'   ⚠ {margin:.1f}° to limit' if near else ''),
                      (q - lo) / (hi - lo), 'crit' if near else None)
