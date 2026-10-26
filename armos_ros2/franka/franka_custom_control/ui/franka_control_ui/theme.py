"""Colours and Qt stylesheet for the control deck."""

# Chrome
BG = '#070b12'
PANEL = '#0c131e'
PANEL_HI = '#111b2a'
BORDER = '#1c2b3f'
TEXT = '#d3e4f5'
TEXT_DIM = '#7d93ab'
TEXT_MUTED = '#4c5f75'
ACCENT = '#00e5ff'      # neon cyan: UI chrome, focus, enabled state
ACCENT_2 = '#ff2bd6'    # neon magenta: "actual robot value" markers

# Status (reserved; never used for data series)
OK = '#29f19c'
WARN = '#ffb020'
CRIT = '#ff3b5c'

# Data series. Validated dark-mode categorical slots 1..7 (dataviz reference palette).
# x/y/z take slots 1-3, which validate all-pairs; ordered to echo RViz's red/green/blue.
AXIS = ['#d95926', '#199e70', '#3987e5']           # x, y, z   (orange, aqua, blue)
JOINT = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9']
PLOT_BG = '#0a1019'
PLOT_GRID = '#1a2636'

MONO_CANDIDATES = ['JetBrains Mono', 'Ubuntu Mono', 'DejaVu Sans Mono', 'Noto Sans Mono']
SANS_CANDIDATES = ['Ubuntu', 'Noto Sans', 'DejaVu Sans']


def pick_font(candidates):
    """First installed family from candidates (Qt stylesheets take one family, no fallbacks)."""
    from PyQt5.QtGui import QFontDatabase
    installed = set(QFontDatabase().families())
    return next((f for f in candidates if f in installed), candidates[-1])


def build_qss():
    return _QSS_TEMPLATE.replace('{MONO}', pick_font(MONO_CANDIDATES)).replace(
        '{SANS}', pick_font(SANS_CANDIDATES))


MONO = '{MONO}'
SANS = '{SANS}'

_QSS_TEMPLATE = f"""
* {{ font-family: '{SANS}'; font-size: 12px; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QToolTip {{ background: {PANEL_HI}; color: {TEXT}; border: 1px solid {ACCENT}; padding: 4px; }}

QFrame#header {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0a1626, stop:0.5 {PANEL}, stop:1 #140a1f);
    border-bottom: 1px solid {BORDER};
}}
QLabel#title {{ font-size: 17px; font-weight: 700; letter-spacing: 4px; color: {ACCENT}; }}
QLabel#subtitle {{ color: {TEXT_DIM}; letter-spacing: 2px; font-size: 10px; }}

QFrame#card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#card[glow="true"] {{ border: 1px solid {ACCENT}; }}
QLabel#cardTitle {{ color: {ACCENT}; font-weight: 700; letter-spacing: 3px; font-size: 11px; }}
QLabel#cardNote {{ color: {TEXT_MUTED}; font-size: 10px; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#axisTag {{ font-family: '{MONO}'; font-weight: 700; font-size: 12px; }}
QLabel#unit {{ color: {TEXT_MUTED}; font-family: '{MONO}'; font-size: 10px; }}
QLabel#value {{ font-family: '{MONO}'; font-size: 20px; font-weight: 600; }}
QLabel#valueSmall {{ font-family: '{MONO}'; font-size: 11px; color: {TEXT_DIM}; }}
QLabel#banner {{
    background: rgba(255, 176, 32, 0.08); border: 1px solid {WARN}; border-radius: 6px;
    color: {WARN}; padding: 6px 10px; letter-spacing: 1px;
}}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_DIM}; padding: 9px 22px; margin-right: 4px;
    border: none; border-bottom: 2px solid transparent; min-width: 90px;
    font-weight: 700; font-size: 12px;
}}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QPushButton {{
    background: {PANEL_HI}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 12px; font-weight: 600; letter-spacing: 1px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #0b2a33; }}
QPushButton:checked {{ background: rgba(0, 229, 255, 0.14); border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: #141f2e; }}
QPushButton#seg {{ border-radius: 0; padding: 4px 10px; font-size: 11px; }}
QPushButton#segFirst {{ border-top-right-radius: 0; border-bottom-right-radius: 0; padding: 4px 10px; font-size: 11px; }}
QPushButton#segLast {{ border-top-left-radius: 0; border-bottom-left-radius: 0; padding: 4px 10px; font-size: 11px; }}

QPushButton#enable {{
    min-width: 150px; min-height: 34px; border-radius: 17px; font-size: 13px; letter-spacing: 3px;
    border: 2px solid {TEXT_MUTED}; color: {TEXT_DIM};
}}
QPushButton#enable:checked {{
    border: 2px solid {ACCENT}; color: {BG}; background: {ACCENT};
}}
QPushButton#estop {{
    min-width: 150px; min-height: 34px; border-radius: 17px; font-size: 14px; font-weight: 800;
    letter-spacing: 3px; color: white;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.4, stop:0 #ff5a74, stop:1 #b3001e);
    border: 2px solid #ff8095;
}}
QPushButton#estop:hover {{ border: 2px solid white; }}
QPushButton#estop[latched="true"] {{
    background: {PANEL_HI}; color: {CRIT}; border: 2px dashed {CRIT};
}}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #004e5c, stop:1 {ACCENT});
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {BG}; border: 2px solid {ACCENT}; width: 12px; height: 12px;
    margin: -6px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QSlider::sub-page:horizontal:disabled {{ background: #1d2d3d; }}
QSlider::handle:horizontal:disabled {{ border-color: {TEXT_MUTED}; }}
QSlider[centered="true"]::sub-page:horizontal {{ background: {BORDER}; }}

QDoubleSpinBox, QSpinBox, QComboBox {{
    background: #08101a; border: 1px solid {BORDER}; border-radius: 5px; padding: 3px 6px;
    font-family: '{MONO}'; selection-background-color: #00525c;
}}
QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QDoubleSpinBox:disabled {{ color: {TEXT_MUTED}; }}
QComboBox QAbstractItemView {{ background: {PANEL_HI}; border: 1px solid {ACCENT}; selection-background-color: #00525c; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QListWidget, QPlainTextEdit {{
    background: #08101a; border: 1px solid {BORDER}; border-radius: 6px; font-family: '{MONO}';
    font-size: 11px;
}}
QCheckBox {{ color: {TEXT_DIM}; spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {TEXT_MUTED}; border-radius: 3px; background: #08101a; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QProgressBar {{ background: {BORDER}; border: none; border-radius: 2px; max-height: 4px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
