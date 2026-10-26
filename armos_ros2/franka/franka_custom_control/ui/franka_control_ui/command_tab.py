"""COMMAND tab: pose / jog / stiffness / wrench / nullspace-joint targets."""

import math

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .transforms import (
    euler_xyz_to_quat,
    quat_angle_between,
    quat_from_rotvec,
    quat_mul,
    quat_to_euler_xyz,
    quat_to_matrix,
    unwrap_to,
)
from .widgets import Card, SafeSpinBox, Segmented, ValueSlider, dot

STIFFNESS_PRESETS = {'SOFT': (200.0, 10.0), 'MEDIUM': (500.0, 30.0), 'STIFF': (800.0, 50.0)}


class CommandTab(QWidget):
    log = pyqtSignal(str, str)

    def __init__(self, bridge, state, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.state = state
        cfg = self.cfg = bridge.cfg
        self.enabled = False
        self.live = True
        self.dirty = {'pose': False, 'joints': False, 'stiffness': False}
        self.stiffness_published = False
        self._last_refusal = ''

        # Exact (unquantised) pose target; sliders only display it.
        self.t_pos = np.zeros(3)
        self.t_eul = np.zeros(3)  # rad, extrinsic XYZ, inside the slider ranges
        self.have_target = False

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        bar = QHBoxLayout()
        self.btn_sync = QPushButton('⟳  SYNC SLIDERS TO ROBOT')
        self.btn_sync.setToolTip('Copy the current EE pose and joint positions into the sliders '
                                 '(does not publish).')
        self.btn_sync.clicked.connect(self.sync_from_robot)
        bar.addWidget(self.btn_sync)
        bar.addSpacing(16)
        lab = QLabel('PUBLISH')
        lab.setObjectName('dim')
        bar.addWidget(lab)
        self.mode = Segmented([('live', 'WHILE DRAGGING'), ('release', 'ON RELEASE')], 'live')
        self.mode.changed.connect(lambda k: setattr(self, 'live', k == 'live'))
        bar.addWidget(self.mode)
        bar.addStretch(1)
        self.banner = QLabel('UI DISABLED: press ENABLE (Ctrl+E) to send commands')
        self.banner.setObjectName('banner')
        bar.addWidget(self.banner)
        root.addLayout(bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(0, 0, 4, 0)
        grid.setSpacing(12)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        col1, col2, col3 = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        for c in (col1, col2, col3):
            c.setSpacing(12)
        grid.addLayout(col1, 0, 0)
        grid.addLayout(col2, 0, 1)
        grid.addLayout(col3, 0, 2)
        for i in range(3):
            grid.setColumnStretch(i, 1)

        self.cards = []
        col1.addWidget(self._build_pose_card())
        col1.addWidget(self._build_jog_card())
        col1.addStretch(1)
        col2.addWidget(self._build_stiffness_card())
        col2.addWidget(self._build_wrench_card())
        col2.addStretch(1)
        col3.addWidget(self._build_joint_card())
        col3.addStretch(1)

        self.flush_timer = QTimer(self)
        self.flush_timer.timeout.connect(self._flush)
        self.flush_timer.start(int(1000 / cfg['publish_rate_limit']))
        self.jog_timer = QTimer(self)
        self.jog_timer.timeout.connect(self._jog_tick)

        self.set_enabled(False)

    # ================================================================== cards
    def _build_pose_card(self):
        card = Card('Target pose', 'Base frame · orientation as extrinsic XYZ (fixed axes), '
                                   'sent as a quaternion. ▲ marks the robot\'s actual value.')
        lo, hi = self.cfg['ranges.position_min'], self.cfg['ranges.position_max']
        center, half = self.cfg['ranges.euler_center_deg'], self.cfg['ranges.euler_half_span_deg']
        self.pose_sliders = []
        for i, tag in enumerate('xyz'):
            s = ValueSlider(tag, lo[i], hi[i], 0.0005, 4, 'm', theme.AXIS[i])
            s.edited.connect(lambda v, final, i=i: self._pose_edited(i, v, final))
            card.body.addWidget(s)
            self.pose_sliders.append(s)
        for i, tag in enumerate('XYZ'):
            s = ValueSlider(f'r{tag}', center[i] - half[i], center[i] + half[i], 0.1, 1, 'deg',
                            theme.AXIS[i])
            s.edited.connect(lambda v, final, i=i: self._pose_edited(3 + i, v, final))
            card.body.addWidget(s)
            self.pose_sliders.append(s)
        self.pose_info = QLabel('quat  --')
        self.pose_info.setObjectName('valueSmall')
        card.body.addWidget(self.pose_info)
        self.pose_status = QLabel('')
        self.pose_status.setObjectName('valueSmall')
        card.body.addWidget(self.pose_status)
        self.cards.append(card)
        return card

    def _build_jog_card(self):
        card = Card('Cartesian velocity jog',
                    'Integrates the velocity into the pose target and publishes at '
                    f"{self.cfg['jog.rate']:.0f} Hz while JOG is on. Pauses while the target "
                    'leads the robot by more than the configured limit.')
        self.jog_frame = Segmented([('base', 'BASE'), ('tool', 'TOOL')], 'base')
        card.header.addWidget(self.jog_frame)
        vmax, wmax = self.cfg['jog.max_linear'], self.cfg['jog.max_angular_deg']
        self.jog_sliders = []
        for i, tag in enumerate('xyz'):
            s = ValueSlider(f'v{tag}', -vmax, vmax, 0.001, 3, 'm/s', theme.AXIS[i], centered=True)
            card.body.addWidget(s)
            self.jog_sliders.append(s)
        for i, tag in enumerate('xyz'):
            s = ValueSlider(f'w{tag}', -wmax, wmax, 0.5, 1, 'deg/s', theme.AXIS[i], centered=True)
            card.body.addWidget(s)
            self.jog_sliders.append(s)
        row = QHBoxLayout()
        self.btn_jog = QPushButton('▶  JOG')
        self.btn_jog.setCheckable(True)
        self.btn_jog.toggled.connect(self._jog_toggled)
        row.addWidget(self.btn_jog)
        stop = QPushButton('■  STOP + ZERO')
        stop.clicked.connect(self.stop_jog)
        row.addWidget(stop)
        row.addStretch(1)
        card.body.addLayout(row)
        self.jog_status = QLabel('idle')
        self.jog_status.setObjectName('valueSmall')
        card.body.addWidget(self.jog_status)
        self.cards.append(card)
        return card

    def _build_stiffness_card(self):
        card = Card('Cartesian stiffness',
                    'Published to the variable-stiffness topic. Clamped by the controller to '
                    f"[0, {self.cfg['limits.stiffness_max'][0]:.0f}] N/m and "
                    f"[0, {self.cfg['limits.stiffness_max'][1]:.0f}] Nm/rad. Once received, "
                    'it overrides the yaml stiffness until the controller is reconfigured.')
        read = QPushButton('⟳ READ')
        read.setToolTip('Read task.k_* parameters from the command controller')
        read.clicked.connect(self.bridge.fetch_stiffness)
        card.header.addWidget(read)
        kt, kr = self.cfg['limits.stiffness_max']
        self.link_t = QCheckBox('link xyz')
        self.link_r = QCheckBox('link rot')
        self.k_sliders = []
        for i, tag in enumerate('xyz'):
            s = ValueSlider(f'K{tag}', 0.0, kt, 1.0, 0, 'N/m', theme.AXIS[i])
            s.edited.connect(lambda v, final, i=i: self._stiffness_edited(i, v, final))
            card.body.addWidget(s)
            self.k_sliders.append(s)
        for i, tag in enumerate('xyz'):
            s = ValueSlider(f'Kr{tag}', 0.0, kr, 0.5, 1, 'Nm/rad', theme.AXIS[i])
            s.edited.connect(lambda v, final, i=i: self._stiffness_edited(3 + i, v, final))
            card.body.addWidget(s)
            self.k_sliders.append(s)
        row = QHBoxLayout()
        row.addWidget(self.link_t)
        row.addWidget(self.link_r)
        row.addStretch(1)
        for name, (t, r) in STIFFNESS_PRESETS.items():
            b = QPushButton(name)
            b.clicked.connect(lambda _, t=t, r=r: self._apply_preset(t, r))
            row.addWidget(b)
        card.body.addLayout(row)
        self.k_source = QLabel('source: not read yet')
        self.k_source.setObjectName('valueSmall')
        card.body.addWidget(self.k_source)
        self.cards.append(card)
        return card

    def _build_wrench_card(self):
        fmax, tmax = self.cfg['limits.wrench_max']
        card = Card('Target wrench', 'Feed-forward wrench in the base frame, applied as Jᵀ·w. '
                                     f'Limited to ±{fmax:.0f} N / ±{tmax:.1f} Nm.')
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        self.w_spins = []
        for i, tag in enumerate(['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
            r, c = divmod(i, 3)
            cell = QHBoxLayout()
            cell.setSpacing(6)
            cell.addWidget(dot(theme.AXIS[c]))
            lab = QLabel(tag)
            lab.setObjectName('axisTag')
            cell.addWidget(lab)
            sp = SafeSpinBox()
            lim = fmax if r == 0 else tmax
            sp.setRange(-lim, lim)
            sp.setDecimals(2)
            sp.setSingleStep(0.5 if r == 0 else 0.1)
            sp.setSuffix(' N' if r == 0 else ' Nm')
            cell.addWidget(sp, 1)
            grid.addLayout(cell, r, c)
            self.w_spins.append(sp)
        card.body.addLayout(grid)
        row = QHBoxLayout()
        apply = QPushButton('APPLY WRENCH')
        apply.clicked.connect(self._apply_wrench)
        zero = QPushButton('ZERO')
        zero.clicked.connect(self.zero_wrench)
        row.addWidget(apply)
        row.addWidget(zero)
        row.addStretch(1)
        card.body.addLayout(row)
        self.w_status = QLabel('active: 0 (nothing sent)')
        self.w_status.setObjectName('valueSmall')
        card.body.addWidget(self.w_status)
        self.cards.append(card)
        return card

    def _build_joint_card(self):
        card = Card('Joint target (nullspace)',
                    'cartesian_controller uses this only as the nullspace posture reference '
                    '(q_ref), so it biases the elbow without moving the EE. Under '
                    'joint_impedance_controller it is the full joint target.')
        sync = QPushButton('⟳')
        sync.setToolTip('Sync joint sliders to the robot')
        sync.clicked.connect(self.sync_joints)
        card.header.addWidget(sync)
        lo, hi = self.cfg['joint_lower'], self.cfg['joint_upper']
        self.j_sliders = []
        for i in range(len(self.bridge.joint_names)):
            s = ValueSlider(f'q{i + 1}', math.degrees(lo[i]), math.degrees(hi[i]), 0.1, 1, 'deg',
                            theme.JOINT[i % len(theme.JOINT)])
            s.edited.connect(lambda v, final: self._mark('joints', final))
            card.body.addWidget(s)
            self.j_sliders.append(s)
        self.cards.append(card)
        return card

    # ================================================================== enable / sync
    def set_enabled(self, on):
        self.enabled = on
        for c in self.cards:
            for w in c.findChildren(QWidget):
                w.setEnabled(on)
            c.set_glow(on)
        self.banner.setVisible(not on)
        if on:
            self.sync_from_robot()
            if not self.stiffness_published:
                self.bridge.fetch_stiffness()
        else:
            self.stop_jog()
            self.dirty = dict.fromkeys(self.dirty, False)

    def sync_from_robot(self):
        lat = self.state.latest()
        if lat['pose_pos'] is None:
            self.log.emit('warn', 'sync: no pose received yet')
        else:
            self.t_pos = lat['pose_pos'].copy()
            e = quat_to_euler_xyz(lat['pose_quat'])
            center = np.radians(self.cfg['ranges.euler_center_deg'])
            self.t_eul = np.array([unwrap_to(a, c) for a, c in zip(e, center)])
            self.have_target = True
            self._show_pose_target()
        self.sync_joints()

    def sync_joints(self):
        q = self.state.latest()['q']
        if q is None:
            self.log.emit('warn', 'sync: no joint state received yet')
            return
        for s, v in zip(self.j_sliders, q):
            s.set_value(math.degrees(v))

    def _show_pose_target(self):
        for i in range(3):
            self.pose_sliders[i].set_value(self.t_pos[i])
            self.pose_sliders[3 + i].set_value(math.degrees(self.t_eul[i]))
        q = self.target_quat()
        self.pose_info.setText('quat  x {:+.4f}  y {:+.4f}  z {:+.4f}  w {:+.4f}'.format(*q))

    def target_quat(self):
        return euler_xyz_to_quat(*self.t_eul)

    # ================================================================== edits
    def _mark(self, key, final):
        if self.enabled and (self.live or final):
            self.dirty[key] = True

    def _pose_edited(self, idx, v, final):
        if idx < 3:
            self.t_pos[idx] = v
        else:
            self.t_eul[idx - 3] = math.radians(v)
        self.have_target = True
        self.pose_info.setText(
            'quat  x {:+.4f}  y {:+.4f}  z {:+.4f}  w {:+.4f}'.format(*self.target_quat()))
        self._mark('pose', final)

    def _stiffness_edited(self, idx, v, final):
        link = self.link_t if idx < 3 else self.link_r
        if link.isChecked():
            base = 0 if idx < 3 else 3
            for j in range(base, base + 3):
                self.k_sliders[j].set_value(v)
        self._mark('stiffness', final)

    def _apply_preset(self, t, r):
        kt, kr = self.cfg['limits.stiffness_max']
        for i, s in enumerate(self.k_sliders):
            s.set_value(min(t, kt) if i < 3 else min(r, kr))
        self.dirty['stiffness'] = self.enabled

    def on_params(self, tag, vals):
        if tag != 'stiffness':
            return
        for s, v in zip(self.k_sliders, vals.values()):
            s.set_value(v)
        self.k_source.setText(f"source: {self.cfg['command_controller']} parameters "
                              '(topic override may differ if another node sent stiffness)')

    # ================================================================== publishing
    def _flush(self):
        if not self.enabled:
            return
        if self.dirty['pose']:
            self.dirty['pose'] = False
            self.publish_pose_target()
        if self.dirty['joints']:
            self.dirty['joints'] = False
            self.bridge.publish_joints([math.radians(s.value()) for s in self.j_sliders])
        if self.dirty['stiffness']:
            self.dirty['stiffness'] = False
            k = [s.value() for s in self.k_sliders]
            if self.bridge.publish_stiffness(k):
                self.stiffness_published = True
                self.k_source.setText('source: published by this UI  [' +
                                      ', '.join(f'{v:.0f}' for v in k) + ']')

    def publish_pose_target(self):
        if not self.have_target:
            return False
        quat = self.target_quat()
        lat = self.state.latest()
        if lat['pose_pos'] is not None:
            d = float(np.linalg.norm(self.t_pos - lat['pose_pos']))
            a = math.degrees(quat_angle_between(quat, lat['pose_quat']))
            dmax = self.cfg['limits.max_target_offset_pos']
            amax = self.cfg['limits.max_target_offset_rot_deg']
            if d > dmax or a > amax:
                msg = (f'REFUSED: target is {d * 100:.1f} cm / {a:.1f}° from the robot '
                       f'(limit {dmax * 100:.0f} cm / {amax:.0f}°)')
                self.pose_status.setText(msg)
                self.pose_status.setStyleSheet(f'color:{theme.CRIT};')
                if msg != self._last_refusal:
                    self.log.emit('warn', msg)
                    self._last_refusal = msg
                return False
        self._last_refusal = ''
        return self.bridge.publish_pose(self.t_pos.copy(), quat)

    def _apply_wrench(self):
        w = [s.value() for s in self.w_spins]
        if self.bridge.publish_wrench(w):
            self.w_status.setText('active: [' + ', '.join(f'{v:+.2f}' for v in w) + ']')
            self.log.emit('info', 'wrench applied ' + ' '.join(f'{v:+.2f}' for v in w))

    def zero_wrench(self):
        for s in self.w_spins:
            s.setValue(0.0)
        if self.bridge.publish_wrench([0.0] * 6):
            self.w_status.setText('active: 0')

    def hold_current_pose(self):
        """E-stop stage 1: target := measured pose, wrench := 0."""
        self.stop_jog()
        lat = self.state.latest()
        if lat['pose_pos'] is not None:
            self.t_pos = lat['pose_pos'].copy()
            e = quat_to_euler_xyz(lat['pose_quat'])
            center = np.radians(self.cfg['ranges.euler_center_deg'])
            self.t_eul = np.array([unwrap_to(a, c) for a, c in zip(e, center)])
            self.have_target = True
            self._show_pose_target()
            self.bridge.publish_pose(self.t_pos.copy(), lat['pose_quat'])
        self.zero_wrench()

    # ================================================================== jog
    def _jog_toggled(self, on):
        if on:
            if not self.have_target:
                self.sync_from_robot()
            self.jog_timer.start(int(1000 / self.cfg['jog.rate']))
            self.btn_jog.setText('❚❚  JOGGING')
        else:
            self.jog_timer.stop()
            self.btn_jog.setText('▶  JOG')
            self.jog_status.setText('idle')

    def stop_jog(self):
        for s in self.jog_sliders:
            s.set_value(0.0)
        self.btn_jog.setChecked(False)

    def _jog_tick(self):
        if not self.enabled:
            self.stop_jog()
            return
        v = np.array([s.value() for s in self.jog_sliders[:3]])
        w = np.radians([s.value() for s in self.jog_sliders[3:]])
        if not v.any() and not w.any():
            self.jog_status.setText('armed: set a velocity')
            return
        lat = self.state.latest()
        if lat['pose_pos'] is None or lat['age']['pose'] > self.cfg['stale_timeout']:
            self.jog_status.setText('paused: pose feedback stale')
            return
        q_t = self.target_quat()
        lead_p = float(np.linalg.norm(self.t_pos - lat['pose_pos']))
        lead_r = math.degrees(quat_angle_between(q_t, lat['pose_quat']))
        if lead_p > self.cfg['jog.max_lead_pos'] or lead_r > self.cfg['jog.max_lead_rot_deg']:
            self.jog_status.setText(f'paused: target leads robot by {lead_p * 100:.1f} cm / '
                                    f'{lead_r:.1f}°')
            return

        dt = 1.0 / self.cfg['jog.rate']
        if self.jog_frame.value() == 'tool':
            v = quat_to_matrix(q_t) @ v
            q_new = quat_mul(q_t, quat_from_rotvec(w * dt))
        else:
            q_new = quat_mul(quat_from_rotvec(w * dt), q_t)

        notes = []
        lo, hi = np.array(self.cfg['ranges.position_min']), np.array(self.cfg['ranges.position_max'])
        new_pos = self.t_pos + v * dt
        clipped = np.clip(new_pos, lo, hi)
        if not np.allclose(clipped, new_pos):
            notes.append('workspace limit')
        self.t_pos = clipped

        if w.any():
            e = quat_to_euler_xyz(q_new)
            e = np.array([unwrap_to(a, p) for a, p in zip(e, self.t_eul)])
            center = np.radians(self.cfg['ranges.euler_center_deg'])
            half = np.radians(self.cfg['ranges.euler_half_span_deg'])
            if np.all(np.abs(e - center) <= half):
                self.t_eul = e
            else:
                notes.append('rotation range limit')

        self._show_pose_target()
        if self.publish_pose_target():
            self.jog_status.setText('jogging' + (f'  ({", ".join(notes)})' if notes else ''))

    # ================================================================== feedback
    def update_feedback(self, lat):
        if lat['pose_pos'] is not None:
            e = quat_to_euler_xyz(lat['pose_quat'])
            for i in range(3):
                self.pose_sliders[i].set_actual(lat['pose_pos'][i])
                a = unwrap_to(e[i], math.radians(self.pose_sliders[3 + i].value()))
                self.pose_sliders[3 + i].set_actual(math.degrees(a))
            if self.have_target and not self._last_refusal:
                d = float(np.linalg.norm(self.t_pos - lat['pose_pos']))
                a = math.degrees(quat_angle_between(self.target_quat(), lat['pose_quat']))
                self.pose_status.setText(f'slider target − robot:  {d * 100:6.2f} cm   {a:6.2f}°')
                self.pose_status.setStyleSheet('')
        if lat['q'] is not None:
            for s, v in zip(self.j_sliders, lat['q']):
                s.set_actual(math.degrees(v))
