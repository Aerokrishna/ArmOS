"""Main window: header (status, ENABLE, E-STOP), tabs, and the system sidebar."""

import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .command_tab import CommandTab
from .monitor_tab import MonitorTab
from .plot_tab import PlotTab
from .widgets import Card, Led, SafeSpinBox

LOG_COLORS = {'info': theme.TEXT_DIM, 'ok': theme.OK, 'warn': theme.WARN, 'error': theme.CRIT}


class MainWindow(QMainWindow):

    def __init__(self, bridge, state):
        super().__init__()
        self.bridge = bridge
        self.state = state
        self.cfg = bridge.cfg
        self.estop_latched = False
        self.controllers = []
        self.configured = []

        self.setWindowTitle('Franka Control Deck')
        self.resize(1680, 980)
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 12, 12)
        body.setSpacing(0)
        self.tabs = QTabWidget()
        self.command = CommandTab(bridge, state)
        self.monitor = MonitorTab(bridge)
        self.plots = PlotTab(state, len(bridge.joint_names))
        self.tabs.addTab(self.command, 'COMMAND')
        self.tabs.addTab(self.monitor, 'MONITOR')
        self.tabs.addTab(self.plots, 'GRAPH')
        body.addWidget(self.tabs, 1)
        body.addWidget(self._build_sidebar())
        v.addLayout(body, 1)

        self.command.log.connect(self.log)
        bridge.sig.log.connect(self.log)
        bridge.sig.controllers.connect(self._on_controllers)
        bridge.sig.configured.connect(self._on_configured)
        bridge.sig.switch_done.connect(self._on_switch_done)
        bridge.sig.params.connect(self.command.on_params)

        QShortcut(QKeySequence(Qt.Key_Escape), self, self.trigger_estop,
                  context=Qt.ApplicationShortcut)
        QShortcut(QKeySequence('Ctrl+E'), self, self.btn_enable.toggle,
                  context=Qt.ApplicationShortcut)

        self.fast = QTimer(self)
        self.fast.timeout.connect(self._tick_fast)
        self.fast.start(66)
        self.slow = QTimer(self)
        self.slow.timeout.connect(self._tick_slow)
        self.slow.start(1000)
        self.ctrl_timer = QTimer(self)
        self.ctrl_timer.timeout.connect(self.bridge.refresh_controllers)
        self.ctrl_timer.start(2000)
        QTimer.singleShot(500, self._refresh_all_controllers)

        self.log('info', f'UI node {bridge.get_namespace().rstrip("/")}/{bridge.get_name()} up. '
                         'Esc = E-STOP, Ctrl+E = toggle ENABLE.')

    # ================================================================== layout
    def _build_header(self):
        h = QFrame()
        h.setObjectName('header')
        lay = QHBoxLayout(h)
        lay.setContentsMargins(20, 10, 20, 10)
        lay.setSpacing(18)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        t = QLabel('FRANKA // CONTROL DECK')
        t.setObjectName('title')
        ns = self.bridge.get_namespace()
        s = QLabel(f"CRISP {self.cfg['command_controller'].upper()}  ·  NS {ns}  ·  "
                   f"FRAME {self.cfg['base_frame']}")
        s.setObjectName('subtitle')
        titles.addWidget(t)
        titles.addWidget(s)
        lay.addLayout(titles)
        lay.addSpacing(20)
        self.leds = {k: Led(k.upper()) for k in ('pose', 'wrench', 'joints')}
        for led in self.leds.values():
            lay.addWidget(led)
        lay.addSpacing(12)
        self.active_chip = Led('CTRL')
        lay.addWidget(self.active_chip)
        lay.addStretch(1)
        self.btn_enable = QPushButton('ENABLE UI')
        self.btn_enable.setObjectName('enable')
        self.btn_enable.setCheckable(True)
        self.btn_enable.toggled.connect(self._on_enable_toggled)
        lay.addWidget(self.btn_enable)
        self.btn_estop = QPushButton('■  E-STOP')
        self.btn_estop.setObjectName('estop')
        self.btn_estop.setToolTip('Software stop (Esc): hold current pose, zero wrench, then '
                                  'deactivate all command controllers. Not a substitute for '
                                  'the hardware stop.')
        self.btn_estop.clicked.connect(self._on_estop_clicked)
        lay.addWidget(self.btn_estop)
        return h

    def _build_sidebar(self):
        side = QWidget()
        side.setFixedWidth(340)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(12, 12, 0, 0)
        lay.setSpacing(12)

        ctrl = Card('Controllers', 'Activating a controller deactivates every other controller '
                                   'that holds command interfaces. Broadcasters stay up.')
        refresh = QPushButton('⟳')
        refresh.setToolTip('Re-read configured and loaded controllers')
        refresh.clicked.connect(self._refresh_all_controllers)
        ctrl.header.addWidget(refresh)
        self.combo = QComboBox()
        ctrl.body.addWidget(self.combo)
        row = QHBoxLayout()
        self.btn_activate = QPushButton('ACTIVATE')
        self.btn_activate.clicked.connect(self._activate_selected)
        self.btn_deactivate = QPushButton('DEACTIVATE')
        self.btn_deactivate.clicked.connect(self._deactivate_selected)
        row.addWidget(self.btn_activate)
        row.addWidget(self.btn_deactivate)
        ctrl.body.addLayout(row)
        lab = QLabel('LOADED CONTROLLERS')
        lab.setObjectName('dim')
        ctrl.body.addWidget(lab)
        self.ctrl_list = QListWidget()
        self.ctrl_list.setMinimumHeight(150)
        self.ctrl_list.itemClicked.connect(
            lambda it: self._select_in_combo(it.data(Qt.UserRole)))
        ctrl.body.addWidget(self.ctrl_list)
        lay.addWidget(ctrl)

        link = Card('Command topics', 'The controller ignores a topic that has more than one '
                                      'publisher, so ENABLE is refused while another node '
                                      'publishes on one.')
        self.topic_labels = {}
        for key, topic in self.bridge.topics.items():
            led = Led(topic)
            link.body.addWidget(led)
            self.topic_labels[key] = led
        lay.addWidget(link)

        robot = Card('Robot')
        rec = QPushButton('ERROR RECOVERY')
        rec.setToolTip(f"franka_msgs/ErrorRecovery on {self.cfg['error_recovery_action']}")
        rec.clicked.connect(self.bridge.error_recovery)
        robot.body.addWidget(rec)
        lay.addWidget(robot)
        lay.addWidget(self._build_gripper_card())

        logc = Card('Event log')
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        logc.body.addWidget(self.log_view)
        lay.addWidget(logc, 1)
        return side

    def _build_gripper_card(self):
        card = Card('Gripper', 'franka_gripper actions; independent of ENABLE and of the arm '
                               'controllers. CLOSE grasps whatever is between the fingers.')
        self.gripper_led = Led('WIDTH')
        card.body.addWidget(self.gripper_led)

        def spin(lo, hi, step, decimals, suffix, value):
            sp = SafeSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setSuffix(suffix)
            sp.setValue(value)
            return sp
        open_mm = self.cfg['gripper.open_width'] * 1000.0
        self.g_speed = spin(0.01, 0.1, 0.01, 2, ' m/s', self.cfg['gripper.speed'])
        self.g_force = spin(1.0, 70.0, 5.0, 0, ' N', self.cfg['gripper.force'])
        self.g_width = spin(0.0, open_mm, 5.0, 0, ' mm', open_mm)
        row = QHBoxLayout()
        row.addWidget(self.g_speed)
        row.addWidget(self.g_force)
        card.body.addLayout(row)

        row = QHBoxLayout()
        b_open = QPushButton('OPEN')
        b_open.clicked.connect(lambda: self._gripper(
            self.bridge.gripper_move, self.cfg['gripper.open_width'], self.g_speed.value()))
        b_close = QPushButton('CLOSE')
        b_close.clicked.connect(lambda: self._gripper(
            self.bridge.gripper_grasp, self.g_speed.value(), self.g_force.value()))
        row.addWidget(b_open)
        row.addWidget(b_close)
        card.body.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(self.g_width, 1)
        b_move = QPushButton('MOVE TO')
        b_move.clicked.connect(lambda: self._gripper(
            self.bridge.gripper_move, self.g_width.value() / 1000.0, self.g_speed.value()))
        row.addWidget(b_move)
        card.body.addLayout(row)

        row = QHBoxLayout()
        b_home = QPushButton('HOME')
        b_home.setToolTip('Homing: full open/close to calibrate the width')
        b_home.clicked.connect(lambda: self._gripper(self.bridge.gripper_home))
        b_stop = QPushButton('STOP')
        b_stop.setToolTip('Stop the current gripper motion (also releases a grasp)')
        b_stop.clicked.connect(self.bridge.gripper_stop)
        row.addWidget(b_home)
        row.addWidget(b_stop)
        card.body.addLayout(row)
        return card

    def _gripper(self, fn, *args):
        if self.estop_latched:
            self.log('error', 'gripper command refused: E-STOP is latched')
            return
        fn(*args)

    # ================================================================== log
    def log(self, level, msg):
        color = LOG_COLORS.get(level, theme.TEXT_DIM)
        stamp = time.strftime('%H:%M:%S')
        self.log_view.appendHtml(f'<span style="color:{theme.TEXT_MUTED}">{stamp}</span> '
                                 f'<span style="color:{color}">{msg}</span>')
        # rclpy pins one severity per call site, so each level needs its own call.
        logger = self.bridge.get_logger()
        if level == 'error':
            logger.error(msg)
        elif level == 'warn':
            logger.warning(msg)
        else:
            logger.info(msg)

    # ================================================================== timers
    def _tick_fast(self):
        lat = self.state.latest()
        stale = self.cfg['stale_timeout']
        for k, led in self.leds.items():
            age = lat['age'][k]
            if age == float('inf'):
                led.set('off', 'no data')
            elif age > stale:
                led.set('crit', f'stale {age:.1f}s')
            else:
                led.set('ok', f"{lat['hz'][k]:.0f} Hz")
        self.command.update_feedback(lat)
        w, age = self.bridge.gripper_width, time.monotonic() - self.bridge.gripper_time
        if w is None:
            self.gripper_led.set('off', 'no gripper')
        elif age > 1.0:
            self.gripper_led.set('crit', f'{w * 1000:.1f} mm  stale')
        else:
            self.gripper_led.set('ok', f'{w * 1000:.1f} mm')
        if self.monitor.isVisible():
            self.monitor.update_view(lat)

    def _tick_slow(self):
        counts = self.bridge.foreign_publishers()
        own = self.bridge.enabled
        for key, led in self.topic_labels.items():
            n = counts[key]
            if n > 0:
                led.set('crit', f'· {n} other publisher(s)' + (' · CONFLICT' if own else ''))
            elif own:
                led.set('ok', '· this UI')
            else:
                led.set('off', '· idle')
        if own and self.state.latest()['age']['pose'] > 1.0:
            self.log('warn', 'pose feedback lost while enabled')

    # ================================================================== enable / estop
    def _set_enable_checked(self, on):
        self.btn_enable.blockSignals(True)
        self.btn_enable.setChecked(on)
        self.btn_enable.blockSignals(False)
        self.btn_enable.setText('UI ENABLED' if on else 'ENABLE UI')

    def _on_enable_toggled(self, on):
        if on:
            reason = self._enable_blocker()
            if reason:
                self._set_enable_checked(False)
                self.log('error', f'ENABLE refused: {reason}')
                return
            self.bridge.enable()
            self.command.set_enabled(True)
            self._set_enable_checked(True)
            self.log('ok', 'UI enabled: sliders synced to robot, commands will be published')
            cmd = self.cfg['command_controller']
            if not any(c['name'] == cmd and c['state'] == 'active' for c in self.controllers):
                self.log('warn', f'{cmd} is not active: commands have no effect until it is')
        else:
            self.command.set_enabled(False)
            self.bridge.disable()
            self._set_enable_checked(False)
            self.log('info', 'UI disabled: command publishers removed')

    def _enable_blocker(self):
        if self.estop_latched:
            return 'E-STOP is latched (reset it first)'
        lat = self.state.latest()
        if lat['age']['pose'] > self.cfg['stale_timeout']:
            return f"no fresh pose on {self.cfg['topics.current_pose']} (is pose_broadcaster active?)"
        if lat['age']['joints'] > 1.0:
            return f"no fresh joint states on {self.cfg['topics.joint_states']}"
        busy = [self.bridge.topics[k] for k, n in self.bridge.foreign_publishers().items() if n > 0]
        if busy:
            return 'other publishers on ' + ', '.join(busy)
        return None

    def _on_estop_clicked(self):
        if self.estop_latched:
            if QMessageBox.question(self, 'Reset E-STOP',
                                    'Release the software E-STOP latch?\nControllers stay '
                                    'deactivated until you activate one.') == QMessageBox.Yes:
                self.estop_latched = False
                self._style_estop()
                self.log('ok', 'E-STOP latch released')
        else:
            self.trigger_estop()

    def trigger_estop(self):
        if self.estop_latched:
            return
        self.estop_latched = True
        # Safety actions first; logging and styling can never be allowed to block them.
        if self.bridge.enabled:
            self.command.hold_current_pose()

        def stage2():
            self.bridge.deactivate_command_controllers()
            if self.bridge.enabled:
                self.command.set_enabled(False)
                self.bridge.disable()
            self._set_enable_checked(False)
        QTimer.singleShot(150, stage2)
        self._style_estop()
        self.log('error', 'E-STOP: holding current pose, zero wrench, deactivating controllers')

    def _style_estop(self):
        self.btn_estop.setText('RESET E-STOP' if self.estop_latched else '■  E-STOP')
        self.btn_estop.setProperty('latched', 'true' if self.estop_latched else 'false')
        self.btn_estop.style().unpolish(self.btn_estop)
        self.btn_estop.style().polish(self.btn_estop)

    # ================================================================== controllers
    def _refresh_all_controllers(self):
        self.bridge.refresh_configured()
        self.bridge.refresh_controllers()

    def _on_configured(self, names):
        self.configured = names
        self._fill_combo()

    def _on_controllers(self, ctrls):
        self.controllers = ctrls
        order = {'active': 0, 'inactive': 1, 'unconfigured': 2}
        self.ctrl_list.clear()
        for c in sorted(ctrls, key=lambda c: (order.get(c['state'], 3), c['name'])):
            mark = {'active': '●', 'inactive': '○'}.get(c['state'], '◌')
            kind = 'cmd' if c['claimed'] else ''
            it = QListWidgetItem(f"{mark} {c['name']:<30} {c['state']:<9} {kind}")
            it.setData(Qt.UserRole, c['name'])
            it.setForeground(QColor(theme.OK if c['state'] == 'active' else theme.TEXT_DIM))
            self.ctrl_list.addItem(it)
        cmd = [c['name'] for c in ctrls if c['state'] == 'active' and c['claimed']]
        if cmd:
            self.active_chip.set('ok', ' + '.join(cmd))
        else:
            self.active_chip.set('warn', 'no command controller active')
        self._fill_combo()

    def _fill_combo(self):
        current = self.combo.currentData()
        states = {c['name']: c['state'] for c in self.controllers}
        names = sorted(set(self.configured) | set(states),
                       key=lambda n: (n not in states, n))
        self.combo.blockSignals(True)
        self.combo.clear()
        for n in names:
            self.combo.addItem(f"{n}   [{states.get(n, 'not loaded')}]", n)
        idx = self.combo.findData(current if current else self.cfg['command_controller'])
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)

    def _select_in_combo(self, name):
        idx = self.combo.findData(name)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _activate_selected(self):
        name = self.combo.currentData()
        if not name:
            return
        if self.estop_latched:
            self.log('error', 'activation refused: E-STOP is latched')
            return
        others = [c['name'] for c in self.controllers
                  if c['state'] == 'active' and c['claimed'] and c['name'] != name]
        text = f'Activate {name}?'
        if others:
            text += f"\n\nThis deactivates: {', '.join(others)}"
        if QMessageBox.question(self, 'Activate controller', text) != QMessageBox.Yes:
            return
        self.log('info', f'activating {name} ...')
        self.bridge.activate_exclusive(name)

    def _deactivate_selected(self):
        name = self.combo.currentData()
        if name:
            self.bridge.deactivate([name])

    def _on_switch_done(self, ok, msg):
        self.log('ok' if ok else 'error', msg)
        if ok and self.bridge.enabled:
            # A freshly activated controller targets the current pose: re-sync so the next
            # slider move starts from where the robot is.
            self.command.sync_from_robot()

    def closeEvent(self, event):
        self.command.set_enabled(False)
        self.bridge.disable()
        super().closeEvent(event)
