"""GRAPH tab: live time-series of pose, wrench and joints with target-vs-actual overlay."""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import theme
from .widgets import Card, Segmented

pg.setConfigOptions(antialias=True, background=theme.PLOT_BG, foreground=theme.TEXT_DIM)


class LivePlot:
    """One plot: solid actual curves, optional dashed target curves, hover crosshair."""

    def __init__(self, title, unit, names, colors, with_target=False):
        self.names = names
        self.colors = colors
        self.unit = unit
        self.widget = pg.PlotWidget()
        pi = self.widget.getPlotItem()
        pi.setTitle(f'<span style="color:{theme.TEXT_DIM}; font-size:9pt">{title}</span>')
        pi.showGrid(x=True, y=True, alpha=0.12)
        pi.setLabel('bottom', 'time', 's')
        pi.setLabel('left', unit)
        pi.setMenuEnabled(False)
        pi.setClipToView(True)
        pi.setDownsampling(auto=True, mode='peak')
        for ax in ('left', 'bottom'):
            pi.getAxis(ax).setPen(pg.mkPen(theme.BORDER))
            pi.getAxis(ax).setTextPen(pg.mkPen(theme.TEXT_DIM))
        self.curves = [pi.plot(pen=pg.mkPen(c, width=2)) for c in colors]
        self.show_targets = with_target
        self.targets = []
        if with_target:
            self.targets = [pi.plot(pen=pg.mkPen(c, width=1.5, style=Qt.DashLine))
                            for c in colors]

        # Legend lives in a row above the plot so it never covers data.
        items = [f'<span style="color:{c}">━━</span>&nbsp;<span style="color:{theme.TEXT}">{n}</span>'
                 for n, c in zip(names, colors)]
        if with_target:
            items.append(f'<span style="color:{theme.TEXT_DIM}">╍╍&nbsp;target</span>')
        self.legend = QLabel('&nbsp;&nbsp;&nbsp;'.join(items))
        self.legend.setObjectName('valueSmall')
        self.legend.setTextFormat(Qt.RichText)
        self.container = QWidget()
        box = QVBoxLayout(self.container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(self.legend)
        box.addWidget(self.widget, 1)
        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.TEXT_MUTED, width=1))
        self.hover = pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(12, 19, 30, 230),
                                 border=pg.mkPen(theme.BORDER))
        self.vline.setZValue(50)
        self.hover.setZValue(51)
        pi.addItem(self.vline, ignoreBounds=True)
        pi.addItem(self.hover, ignoreBounds=True)
        self.vline.hide()
        self.hover.hide()
        self._x = None
        self._y = None
        self._ty = None
        self._proxy = pg.SignalProxy(self.widget.scene().sigMouseMoved, rateLimit=30,
                                     slot=self._mouse)

    def set_targets_visible(self, on):
        for c in self.targets:
            c.setVisible(on)
        self.show_targets = on

    def set_data(self, x, y, ty=None):
        self._x, self._y, self._ty = x, y, ty
        for i, c in enumerate(self.curves):
            c.setData(x, y[:, i]) if len(x) else c.clear()
        if self.targets and ty is not None:
            for i, c in enumerate(self.targets):
                c.setData(x, ty[:, i], connect='finite') if len(x) else c.clear()

    def set_window(self, seconds):
        self.widget.getPlotItem().setXRange(-seconds, 0, padding=0)

    def _mouse(self, evt):
        pos = evt[0]
        vb = self.widget.getPlotItem().vb
        if self._x is None or not len(self._x) or not self.widget.sceneBoundingRect().contains(pos):
            self.vline.hide()
            self.hover.hide()
            return
        p = vb.mapSceneToView(pos)
        i = int(np.clip(np.searchsorted(self._x, p.x()), 0, len(self._x) - 1))
        self.vline.setPos(self._x[i])
        rows = [f'<span style="color:{theme.TEXT_DIM}">t {self._x[i]:+.2f} s</span>']
        for k, (n, c) in enumerate(zip(self.names, self.colors)):
            tgt = ''
            if self.show_targets and self._ty is not None and np.isfinite(self._ty[i, k]):
                tgt = f'<span style="color:{theme.TEXT_DIM}">  tgt {self._ty[i, k]:+.4g}</span>'
            rows.append(f'<span style="color:{c}">●</span> '
                        f'<span style="color:{theme.TEXT}">{n} {self._y[i, k]:+.4g}</span>{tgt}')
        self.hover.setHtml('<div style="font-family:monospace; font-size:9pt">' +
                           '<br>'.join(rows) + '</div>')
        (x0, x1), (y0, y1) = vb.viewRange()
        self.hover.setPos(x0 + 0.02 * (x1 - x0), y1)
        self.vline.show()
        self.hover.show()


class PlotRow(Card):
    """A card holding one or two plots with a view selector."""

    def __init__(self, title, plots, views, initial):
        super().__init__(title)
        self.plots = plots
        self.views = views
        self.sel = Segmented([(k, k.upper()) for k in views], initial)
        self.sel.changed.connect(self.show_view)
        self.header.addWidget(self.sel)
        row = QHBoxLayout()
        row.setSpacing(8)
        for p in plots.values():
            row.addWidget(p.container)
        self.body.addLayout(row)
        self.show_view(initial)

    def show_view(self, key):
        for name, p in self.plots.items():
            p.container.setVisible(name in self.views[key])


class PlotTab(QWidget):

    def __init__(self, state, n_joints, parent=None):
        super().__init__(parent)
        self.state = state
        self.window = 10.0

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        bar = QHBoxLayout()
        self.btn_plot = QPushButton('▶  PLOT')
        self.btn_plot.setCheckable(True)
        self.btn_plot.toggled.connect(self._toggle)
        bar.addWidget(self.btn_plot)
        clear = QPushButton('CLEAR')
        clear.clicked.connect(self._clear)
        bar.addWidget(clear)
        bar.addSpacing(16)
        lab = QLabel('WINDOW')
        lab.setObjectName('dim')
        bar.addWidget(lab)
        win = Segmented([('5', '5 s'), ('10', '10 s'), ('30', '30 s'), ('60', '60 s')], '10')
        win.changed.connect(self._set_window)
        bar.addWidget(win)
        bar.addSpacing(16)
        self.chk_target = QCheckBox('target overlay (dashed)')
        self.chk_target.setChecked(True)
        self.chk_target.toggled.connect(self._overlay)
        bar.addWidget(self.chk_target)
        bar.addStretch(1)
        hint = QLabel('Data buffers continuously in the background. Hover a plot for values.')
        hint.setObjectName('cardNote')
        bar.addWidget(hint)
        root.addLayout(bar)

        self.p_pos = LivePlot('Position', 'm', ['x', 'y', 'z'], theme.AXIS, with_target=True)
        self.p_rot = LivePlot('Orientation (extrinsic XYZ)', 'deg', ['rX', 'rY', 'rZ'],
                              theme.AXIS, with_target=True)
        self.p_force = LivePlot('External force', 'N', ['Fx', 'Fy', 'Fz'], theme.AXIS)
        self.p_torque = LivePlot('External torque', 'Nm', ['Tx', 'Ty', 'Tz'], theme.AXIS)
        jcolors = [theme.JOINT[i % len(theme.JOINT)] for i in range(n_joints)]
        self.p_joint = LivePlot('Joint positions', 'deg', [f'q{i + 1}' for i in range(n_joints)],
                                jcolors, with_target=True)
        self.all = [self.p_pos, self.p_rot, self.p_force, self.p_torque, self.p_joint]

        root.addWidget(PlotRow('Pose', {'pos': self.p_pos, 'rot': self.p_rot},
                               {'xyz': ['pos'], 'rot': ['rot'], 'both': ['pos', 'rot']},
                               'both'), 1)
        root.addWidget(PlotRow('Wrench', {'f': self.p_force, 't': self.p_torque},
                               {'force': ['f'], 'torque': ['t'], 'both': ['f', 't']},
                               'both'), 1)
        root.addWidget(PlotRow('Joints', {'q': self.p_joint}, {'all': ['q']}, 'all'), 1)

        self._set_window('10')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)

    def _toggle(self, on):
        self.btn_plot.setText('❚❚  PAUSE' if on else '▶  PLOT')
        if on:
            self.timer.start(33)
            self.refresh()
        else:
            self.timer.stop()

    def _set_window(self, key):
        self.window = float(key)
        for p in self.all:
            p.set_window(self.window)

    def _overlay(self, on):
        for p in (self.p_pos, self.p_rot, self.p_joint):
            p.set_targets_visible(on)

    def _clear(self):
        self.state.clear_buffers()
        for p in self.all:
            p.set_data(np.empty(0), np.empty((0, len(p.names))))

    def refresh(self):
        if not self.isVisible():
            return
        now = self.state.now()

        def window(key):
            t, v = self.state.snapshot(key)
            m = t >= now - self.window - 0.5
            return t[m] - now, v[m]

        x, v = window('pose')
        _, tv = window('pose_target')
        n = min(len(x), len(tv))
        x, v, tv = x[-n:], v[-n:], tv[-n:]
        self.p_pos.set_data(x, v[:, :3], tv[:, :3])
        self.p_rot.set_data(x, v[:, 3:], tv[:, 3:])
        x, v = window('wrench')
        self.p_force.set_data(x, v[:, :3])
        self.p_torque.set_data(x, v[:, 3:])
        x, v = window('joints')
        _, tv = window('joints_target')
        n = min(len(x), len(tv))
        self.p_joint.set_data(x[-n:], v[-n:], tv[-n:])
