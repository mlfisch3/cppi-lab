"""High-contrast dark institutional theme (PRD section 6.3, TSD section 4.1).

Every colour used by the application is named here so the stylesheet, the
chart layers and the documentation pages stay in step.
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

# Surfaces (PRD section 6.3)
CANVAS = "#0B0F19"
PANEL = "#111827"
GRID = "#1E293B"
PANEL_RAISED = "#162033"
BORDER = "#243044"
HOVER = "#1B2740"
#: Outline of things that float above the workbench: popovers and floating windows.
FLOAT_BORDER = "#33506E"

# Text
TEXT = "#E5E7EB"
TEXT_MUTED = "#94A3B8"
TEXT_FAINT = "#64748B"
ACCENT = "#38BDF8"
GOOD = "#34D399"
WARN = "#F59E0B"
BAD = "#EF4444"

# Chart layers (TSD section 4.1 visual-encoding matrix)
PRICE = "#F8FAFC"
PRICE_BAND = "#33415540"
P_LIQ = "#DC2626"
P_FLOOR = "#F97316"
P_FLOOR_ZONE = "#F9731622"
EQUITY = "#38BDF8"
EQUITY_GLOW = "#38BDF830"
MC_FAN = "#38BDF825"
MAKER_SELL = "#06B6D4"
TAKER_SELL = "#EC4899"
MAKER_BUY = "#10B981"
TAKER_BUY = "#84CC16"
MISSED = "#F59E0B"
LIQUIDATION = "#EF4444"
N_CEIL = "#FB7185"
N_REBUY = "#34D399"
DEADBAND = "#64748B28"
N_HELD = "#FACC15"
N_MAX = "#94A3B8"
S_DOWN = "#F43F5E"
S_DOWN_FILL = "#F43F5E30"
S_UP = "#10B981"
S_UP_FILL = "#10B98130"
FILL_PROB = "#A855F7"
CROSSHAIR = "#94A3B8"
SELECTION = "#38BDF8"

FONT_FAMILY = "Inter, 'Segoe UI', 'SF Pro Text', 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif"
MONO_FAMILY = (
    "'JetBrains Mono', 'Cascadia Mono', 'SF Mono', 'DejaVu Sans Mono', Consolas, monospace"
)

STYLESHEET = f"""
* {{
    font-family: {FONT_FAMILY};
    font-size: 12px;
    color: {TEXT};
}}
QMainWindow, QDialog, #workbench {{ background: {CANVAS}; }}
QToolTip {{
    background: {PANEL_RAISED}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px;
}}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; }}
QLabel[role="faint"] {{ color: {TEXT_FAINT}; }}

/* ---- top bar ---------------------------------------------------------- */
#topBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
#brand {{ font-size: 14px; font-weight: 700; letter-spacing: 0.4px; color: {TEXT}; }}
#brandAccent {{ font-size: 14px; font-weight: 700; color: {ACCENT}; }}
#statusChip {{
    color: {TEXT_MUTED}; font-family: {MONO_FAMILY}; font-size: 11px; padding: 0 2px;
}}
#normChip {{
    color: {CANVAS}; background: {WARN}; border-radius: 3px; padding: 2px 7px;
    font-size: 11px; font-weight: 700;
}}

/* ---- sidebar ---------------------------------------------------------- */
#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
#sidebarScroll {{ background: {PANEL}; border: none; }}
#sidebarScroll QScrollBar:vertical {{ background: {PANEL}; }}
#sidebarScroll QScrollBar::handle:vertical:disabled {{ background: transparent; }}
#sidebarBody {{ background: {PANEL}; }}
#groupHeader {{
    color: {ACCENT}; font-size: 10px; font-weight: 700; letter-spacing: 1.1px;
    padding: 0; border-bottom: 1px solid {GRID};
}}
#paramLabel {{ color: {TEXT}; font-size: 11px; }}
#paramSymbol {{ color: {TEXT_FAINT}; font-size: 11px; font-style: italic; }}
#paramRow[modified="true"] #paramLabel {{ color: {ACCENT}; }}
#sidebarFooter {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}

QPushButton {{
    background: {PANEL_RAISED}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 4px 10px; color: {TEXT};
}}
QPushButton:hover {{ background: {HOVER}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {GRID}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {GRID}; }}
QPushButton:checked {{ background: #0C2A3D; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#primary {{ background: #0C2A3D; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#primary:hover {{ background: #103850; }}
QPushButton#resetAll {{ font-weight: 600; }}
QToolButton#resetOne {{
    background: transparent; border: none; color: {TEXT_FAINT}; font-size: 13px; padding: 0;
}}
QToolButton#resetOne:hover {{ color: {ACCENT}; }}
QToolButton#resetOne:disabled {{ color: #2A3549; }}

#helpBadge {{
    color: {TEXT_MUTED}; background: {PANEL_RAISED}; border: 1px solid {BORDER};
    border-radius: 7px; font-size: 9px; font-weight: 700;
}}
#helpBadge:hover, #helpBadge[active="true"] {{
    color: {CANVAS}; background: {ACCENT}; border-color: {ACCENT};
}}

QAbstractSpinBox {{
    background: {CANVAS}; border: 1px solid {BORDER}; border-radius: 3px;
    padding: 1px 3px; font-family: {MONO_FAMILY}; font-size: 11px;
    selection-background-color: {ACCENT}; selection-color: {CANVAS};
}}
QAbstractSpinBox:focus {{ border-color: {ACCENT}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0; border: none; }}
QComboBox {{
    background: {CANVAS}; border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 8px;
}}
QComboBox:hover, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_RAISED}; border: 1px solid {BORDER}; selection-background-color: {HOVER};
    selection-color: {TEXT}; outline: none;
}}
QLineEdit {{
    background: {CANVAS}; border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 6px;
    selection-background-color: {ACCENT}; selection-color: {CANVAS};
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 3px; background: {GRID}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: #2B6E91; border-radius: 1px; }}
QSlider::handle:horizontal {{
    width: 10px; height: 10px; margin: -4px 0; border-radius: 5px;
    background: {TEXT_MUTED}; border: 1px solid {CANVAS};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}

QCheckBox {{ spacing: 6px; font-size: 11px; }}
QCheckBox::indicator {{
    width: 26px; height: 13px; border-radius: 7px; background: {GRID}; border: 1px solid {BORDER};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

/* ---- KPI ribbon ------------------------------------------------------- */
#kpiRibbon {{ background: {CANVAS}; }}
#kpiCard {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 5px; }}
#kpiTitle {{
    color: {TEXT_MUTED}; font-size: 9px; font-weight: 700; letter-spacing: 0.9px;
}}
#kpiTitle[layout="compact"], #kpiTitle[layout="grid"] {{ letter-spacing: 0.6px; }}
#kpiValue {{ font-family: {MONO_FAMILY}; font-size: 15px; font-weight: 700; color: {TEXT}; }}
#kpiValue[layout="compact"] {{ font-size: 12px; }}
#kpiValue[layout="grid"] {{ font-size: 11px; }}
#kpiSub[layout="compact"] {{ font-size: 9px; }}
#kpiValue[tone="good"] {{ color: {GOOD}; }}
#kpiValue[tone="warn"] {{ color: {WARN}; }}
#kpiValue[tone="bad"] {{ color: {BAD}; }}
#kpiSub {{ color: {TEXT_FAINT}; font-family: {MONO_FAMILY}; font-size: 10px; }}

/* ---- chart overlays ---------------------------------------------------- */
#paneTitle {{ color: {TEXT}; font-size: 11px; font-weight: 700; }}
#paneLegend {{ color: {TEXT_MUTED}; font-size: 10px; }}
#paneReadout {{ color: {TEXT_MUTED}; font-family: {MONO_FAMILY}; font-size: 10px; }}
#emptyBanner {{
    background: #101A2CF0; border: 1px solid {BORDER}; border-radius: 6px;
}}
#emptyBanner QLabel {{ color: {TEXT_MUTED}; font-size: 11px; }}
#emptyBanner QLabel#bannerTitle {{ color: {TEXT}; font-size: 12px; font-weight: 700; }}

/* ---- tables ------------------------------------------------------------ */
QTableView {{
    background: {CANVAS}; alternate-background-color: #0E1422; gridline-color: {GRID};
    border: none; font-family: {MONO_FAMILY}; font-size: 11px;
    selection-background-color: #12344A; selection-color: {TEXT};
}}
QHeaderView::section {{
    background: {PANEL}; color: {TEXT_MUTED}; border: none; border-right: 1px solid {GRID};
    border-bottom: 1px solid {BORDER}; padding: 4px 6px; font-size: 10px; font-weight: 700;
}}
QHeaderView::section:hover {{ color: {TEXT}; }}
QTableCornerButton::section {{ background: {PANEL}; border: none; }}
#ledgerBar, #docBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}

QScrollBar:vertical {{ background: {CANVAS}; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: {CANVAS}; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: #2A3549; border-radius: 4px; min-width: 24px; min-height: 24px; }}
QScrollBar::handle:hover {{ background: #3B4A63; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {TEXT_MUTED}; }}
QStatusBar::item {{ border: none; }}
QMenu {{ background: {PANEL_RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 3px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QProgressBar {{
    background: {CANVAS}; border: 1px solid {BORDER}; border-radius: 3px; text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

/* ---- popovers ----------------------------------------------------------- */
#popover {{ background: {PANEL_RAISED}; border: 1px solid {FLOAT_BORDER}; border-radius: 6px; }}
#popoverTitleBar {{ background: transparent; }}
#popoverTitle {{ font-size: 12px; font-weight: 700; color: {TEXT}; }}
QToolButton#popoverButton {{
    background: transparent; border: 1px solid transparent; border-radius: 3px;
    color: {TEXT_MUTED}; font-size: 11px; padding: 1px 5px;
}}
QToolButton#popoverButton:hover {{ border-color: {BORDER}; color: {TEXT}; }}
QToolButton#popoverButton:checked {{ color: {ACCENT}; border-color: {ACCENT}; }}

/* ---- Qt Advanced Docking System ---------------------------------------- */
ads--CDockManager, ads--CDockContainerWidget, ads--CDockAreaWidget {{ background: {CANVAS}; }}
ads--CDockAreaTitleBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; min-height: 24px; }}
ads--CDockWidgetTab {{ background: {PANEL}; border: none; padding: 0 6px; }}
ads--CDockWidgetTab[activeTab="true"] {{ background: {PANEL_RAISED}; }}
ads--CDockWidgetTab QLabel {{ color: {TEXT_MUTED}; font-size: 11px; font-weight: 600; }}
ads--CDockWidgetTab[activeTab="true"] QLabel {{ color: {TEXT}; }}
ads--CDockWidget {{ background: {CANVAS}; border: none; }}
ads--CTitleBarButton, #tabCloseButton, #floatingTitleCloseButton, #floatingTitleMaximizeButton {{
    background: transparent; border: none; padding: 2px;
}}
ads--CTitleBarButton:hover, #tabCloseButton:hover {{ background: {HOVER}; border-radius: 3px; }}
ads--CDockSplitter::handle {{ background: {BORDER}; }}
ads--CFloatingDockContainer {{ background: {CANVAS}; border: 1px solid {FLOAT_BORDER}; }}
ads--CFloatingWidgetTitleBar {{ background: {PANEL}; }}
#floatingTitleLabel {{ color: {TEXT}; font-size: 11px; font-weight: 600; }}
ads--CAutoHideTab {{ background: {PANEL}; color: {TEXT_MUTED}; padding: 4px; }}
ads--CAutoHideSideBar {{ background: {PANEL}; border: none; }}
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    """Install the dark palette and stylesheet on the application."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    for key, colour in (
        (role.Window, CANVAS),
        (role.WindowText, TEXT),
        (role.Base, CANVAS),
        (role.AlternateBase, PANEL),
        (role.ToolTipBase, PANEL_RAISED),
        (role.ToolTipText, TEXT),
        (role.Text, TEXT),
        (role.Button, PANEL_RAISED),
        (role.ButtonText, TEXT),
        (role.BrightText, "#FFFFFF"),
        (role.Highlight, ACCENT),
        (role.HighlightedText, CANVAS),
        (role.Link, ACCENT),
        (role.PlaceholderText, TEXT_FAINT),
    ):
        palette.setColor(key, QtGui.QColor(colour))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
