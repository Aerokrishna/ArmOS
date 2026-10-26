"""Reusable widgets: cards, value sliders with an actual-value marker, stat tiles, LEDs."""

from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPolygonF
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from . import theme


class Card(QFrame):
    def __init__(self, title, note=None, parent=None):
        super().__init__(parent)
        self.setObjectName('card')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(6)
        self.header = QHBoxLayout()
        self.header.setSpacing(6)
        t = QLabel(f'▍{title.upper()}')
        t.setObjectName('cardTitle')
        self.header.addWidget(t)
        self.header.addStretch(1)
        outer.addLayout(self.header)
        if note:
            n = QLabel(note)
            n.setObjectName('cardNote')
            n.setWordWrap(True)
            outer.addWidget(n)
        self.body = QVBoxLayout()
        self.body.setSpacing(4)
        outer.addLayout(self.body)

    def set_glow(self, on):
        self.setProperty('glow', 'true' if on else 'false')
        self.style().unpolish(self)
        self.style().polish(self)


def dot(color, size=8):
    d = QLabel()
    d.setFixedSize(size, size)
    d.setStyleSheet(f'background:{color}; border-radius:{size // 2}px;')
    return d


class MarkerSlider(QSlider):
    """Horizontal slider that also draws a marker at the robot's actual value."""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._marker = None
        self.setMinimumHeight(22)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        # Scrolling the page must never move the robot: only a focused slider takes the wheel.
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def set_marker(self, v):
        if v != self._marker:
            self._marker = v
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._marker is None:
            return
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
        span = self.width() - handle.width()
        v = min(max(self._marker, self.minimum()), self.maximum())
        x = handle.width() / 2 + QStyle.sliderPositionFromValue(
            self.minimum(), self.maximum(), v, span)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.ACCENT_2))
        y = self.height() / 2 + 6
        p.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x - 4, y + 6), QPointF(x + 4, y + 6)]))
        p.end()


class SafeSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ValueSlider(QWidget):
    """[tag] [slider ▲actual] [spinbox] [unit].

    edited(value, final) fires only on user interaction. `final` is False while the
    slider is being dragged and True on release / spinbox entry / click-step.
    """

    edited = pyqtSignal(float, bool)

    def __init__(self, tag, lo, hi, step, decimals, unit='', color=None, centered=False,
                 parent=None):
        super().__init__(parent)
        self.step = step
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(dot(color or theme.TEXT_DIM))
        tl = QLabel(tag)
        tl.setObjectName('axisTag')
        tl.setFixedWidth(34)
        lay.addWidget(tl)
        self.slider = MarkerSlider()
        if centered:
            self.slider.setProperty('centered', 'true')
        lay.addWidget(self.slider, 1)
        self.spin = SafeSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(84)
        self.spin.setAlignment(Qt.AlignRight)
        lay.addWidget(self.spin)
        u = QLabel(unit)
        u.setObjectName('unit')
        u.setFixedWidth(40)
        lay.addWidget(u)
        self.set_range(lo, hi)

        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderReleased.connect(lambda: self.edited.emit(self.value(), True))
        self.spin.valueChanged.connect(self._spin_changed)

    def set_range(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setRange(round(lo / self.step), round(hi / self.step))
        self.spin.setRange(lo, hi)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def value(self):
        return self.spin.value()

    def set_value(self, v):
        """Programmatic update: never emits edited."""
        v = min(max(v, self.lo), self.hi)
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(round(v / self.step))
        self.spin.setValue(v)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def set_actual(self, v):
        self.slider.set_marker(None if v is None else round(v / self.step))

    def _slider_changed(self, iv):
        v = iv * self.step
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.edited.emit(v, not self.slider.isSliderDown())

    def _spin_changed(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round(v / self.step))
        self.slider.blockSignals(False)
        self.edited.emit(v, True)


class StatTile(QFrame):
    def __init__(self, label, unit='', color=None, with_bar=False, parent=None):
        super().__init__(parent)
        self.setObjectName('card')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(6)
        if color:
            top.addWidget(dot(color))
        lab = QLabel(label)
        lab.setObjectName('dim')
        top.addWidget(lab)
        top.addStretch(1)
        un = QLabel(unit)
        un.setObjectName('unit')
        top.addWidget(un)
        lay.addLayout(top)
        self.value = QLabel('--')
        self.value.setObjectName('value')
        lay.addWidget(self.value)
        self.sub = QLabel('')
        self.sub.setObjectName('valueSmall')
        lay.addWidget(self.sub)
        self.bar = None
        if with_bar:
            self.bar = QProgressBar()
            self.bar.setRange(0, 1000)
            self.bar.setTextVisible(False)
            lay.addWidget(self.bar)

    def set(self, text, sub=None, frac=None, alert=None):
        self.value.setText(text)
        if sub is not None:
            self.sub.setText(sub)
        if self.bar is not None and frac is not None:
            self.bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        color = {'warn': theme.WARN, 'crit': theme.CRIT}.get(alert, theme.TEXT)
        self.value.setStyleSheet(f'color:{color};')


class Led(QWidget):
    """Status light with a text label: ● POSE 250 Hz."""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._dot = QLabel()
        self._dot.setFixedSize(9, 9)
        lay.addWidget(self._dot)
        self._label = label
        self._text = QLabel(label)
        self._text.setObjectName('valueSmall')
        lay.addWidget(self._text)
        self.set('off', '')

    def set(self, status, detail):
        color = {'ok': theme.OK, 'warn': theme.WARN, 'crit': theme.CRIT}.get(status, theme.TEXT_MUTED)
        glow = f'border: 2px solid {color}55;' if status == 'ok' else ''
        self._dot.setStyleSheet(f'background:{color}; border-radius:4px; {glow}')
        self._text.setText(f'{self._label} {detail}'.strip())


class Segmented(QWidget):
    """Exclusive segmented button group; emits the selected key."""

    changed = pyqtSignal(str)

    def __init__(self, options, initial, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for i, (key, label) in enumerate(options):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setObjectName('segFirst' if i == 0 else 'segLast' if i == len(options) - 1 else 'seg')
            b.setProperty('key', key)
            b.setChecked(key == initial)
            self.group.addButton(b)
            lay.addWidget(b)
        self.group.buttonClicked.connect(lambda b: self.changed.emit(b.property('key')))

    def value(self):
        b = self.group.checkedButton()
        return b.property('key') if b else None
