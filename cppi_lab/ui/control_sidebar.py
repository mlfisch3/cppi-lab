"""Parameter sidebar: sliders, spin boxes, reset buttons and ``?`` badges (PRD FR-2).

Every row is generated from :data:`cppi_lab.config.defaults.PARAM_SPECS`, so a
control can never disagree with the engine about a default, a range or a
step.  Each numeric row pairs a slider with a spin box that stay in step,
carries a ``?`` badge and a per-row *Reset to Default* micro-button; the
footer holds *Reset All Parameters to Defaults*.

The whole column fits a 1080-pixel-high screen without scrolling; on shorter
screens the body becomes scrollable while the footer stays pinned.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from cppi_lab.config.defaults import (
    GROUP_SYMBOLS,
    GROUPS,
    PARAM_SPECS,
    SPEC_BY_KEY,
    ParamSpec,
    ParamValue,
    SimParams,
    default_params,
    p_ref_minimum,
)
from cppi_lab.ui import theme
from cppi_lab.ui.popover_manager import HelpBadge

SIDEBAR_WIDTH = 468
ROW_HEIGHT = 25
GROUP_HEADER_HEIGHT = 21
#: Width of the scrollbar gutter that is always kept free at the right of the rows.
SCROLL_GUTTER = 10
#: Room for the longest caption with its symbol ("Momentum vs. Imbalance Weight w_m,d").
_LABEL_WIDTH = 228
_SPIN_WIDTH = 92
_SLIDER_WIDTH = 70
_DATETIME_WIDTH = 124

#: Row captions that differ from the PRD table label (paired rows are split).
ROW_LABELS: dict[str, str] = {
    "w_fast_d": "Fast EMA Window",
    "w_slow_d": "Slow EMA Window",
    "w_fast_u": "Fast EMA Window",
    "w_slow_u": "Slow EMA Window",
    "normalize": "Normalize Start Price to Anchor",
    "mc_fan": "Monte Carlo Fan Overlay (M = 100)",
}

#: Plain-text rendering of each math symbol for the compact sidebar.
PLAIN_SYMBOLS: dict[str, str] = {
    "start_ms": "t₀",
    "duration_days": "ΔT",
    "n0": "N₀",
    "c0": "c₀",
    "p_liq": "P_liq",
    "p_ref": "P_ref",
    "dn_min_pct": "Δn_min",
    "f_maker": "f_maker",
    "f_taker": "f_taker",
    "delta_slip": "δ_slip",
    "rho": "ρ",
    "lambda_liq": "λ_liq",
    "beta_vel": "β_vel",
    "alpha_size": "α_size",
    "seed": "s_rng",
    "mc_fan": "",
    "w_fast_d": "W_f,d",
    "w_slow_d": "W_s,d",
    "w_atr_d": "W_atr,d",
    "w_imb_d": "W_imb,d",
    "w_mom_d": "w_m,d",
    "k_sell": "k_sell",
    "w_fast_u": "W_f,u",
    "w_slow_u": "W_s,u",
    "w_atr_u": "W_atr,u",
    "w_brk_u": "W_brk,u",
    "tau_veto": "τ_veto",
    "normalize": "",
}

GROUP_PLAIN_SYMBOLS: dict[str, str] = {"S_{\\text{down}}": "S_down", "S_{\\text{up}}": "S_up"}

_UTC = QtCore.QTimeZone.utc()


class ToggleSwitch(QtWidgets.QAbstractButton):
    """A compact on/off switch with a sliding knob."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(30, 16)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(30, 16)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        on = self.isChecked()
        track = QtGui.QColor(theme.ACCENT if on else theme.GRID)
        if not self.isEnabled():
            track.setAlpha(110)
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.ACCENT if on else theme.BORDER), 1.0))
        painter.setBrush(track)
        painter.drawRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)
        diameter = rect.height() - 4.0
        x = rect.right() - diameter - 2.0 if on else rect.left() + 2.0
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(theme.CANVAS if on else theme.TEXT_MUTED))
        painter.drawEllipse(QtCore.QRectF(x, rect.top() + 2.0, diameter, diameter))
        if self.hasFocus():
            painter.setPen(QtGui.QPen(QtGui.QColor(theme.TEXT), 1.0, QtCore.Qt.PenStyle.DotLine))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)


def _to_qdatetime(ms: int) -> QtCore.QDateTime:
    return QtCore.QDateTime.fromMSecsSinceEpoch(int(ms), _UTC)


class ParamRow(QtWidgets.QWidget):
    """One sidebar row bound to a :class:`ParamSpec`."""

    valueChanged = QtCore.Signal(str, object)
    resetRequested = QtCore.Signal(str)

    def __init__(self, spec: ParamSpec, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._value: ParamValue = spec.default
        self._default: ParamValue = spec.default
        self._minimum = spec.minimum
        self._maximum = spec.maximum
        self._guard = False
        self.setObjectName("paramRow")
        self.setFixedHeight(ROW_HEIGHT)

        caption = ROW_LABELS.get(spec.key, spec.label)
        symbol = PLAIN_SYMBOLS.get(spec.key, "")
        self.label = QtWidgets.QLabel(caption, self)
        self.label.setObjectName("paramLabel")
        self.label.setToolTip(f"{spec.label}" + (f"  ({symbol})" if symbol else ""))
        self.symbol = QtWidgets.QLabel(symbol, self)
        self.symbol.setObjectName("paramSymbol")
        self.badge = HelpBadge(spec.popover, self)
        self.reset_button = QtWidgets.QToolButton(self)
        self.reset_button.setObjectName("resetOne")
        self.reset_button.setText("↺")
        self.reset_button.setFixedSize(16, 18)
        self.reset_button.setToolTip("Reset to Default")
        self.reset_button.setAccessibleName(f"Reset {caption} to default")
        self.reset_button.clicked.connect(lambda: self.resetRequested.emit(self.spec.key))

        self.slider: QtWidgets.QSlider | None = None
        self.spin: QtWidgets.QAbstractSpinBox | None = None
        self.toggle: ToggleSwitch | None = None

        name_box = QtWidgets.QHBoxLayout()
        name_box.setContentsMargins(0, 0, 0, 0)
        name_box.setSpacing(5)
        name_box.addWidget(self.label)
        name_box.addWidget(self.symbol)
        name_box.addStretch(1)
        name_holder = QtWidgets.QWidget(self)
        name_holder.setLayout(name_box)
        name_holder.setFixedWidth(_LABEL_WIDTH)

        # 10 + 228 + 14 + 70 + 92 + 16 + 4 and five gaps of 5 make 454: the sidebar
        # width less its border, the scrollbar gutter and the style's gap beside
        # it, so a row is never wider than the viewport it scrolls in.
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 4, 0)
        layout.setSpacing(5)
        layout.addWidget(name_holder)
        layout.addWidget(self.badge)
        if spec.kind == "bool":
            self.toggle = ToggleSwitch(self)
            self.toggle.setAccessibleName(caption)
            self.toggle.toggled.connect(self._on_toggled)
            layout.addWidget(self.toggle)
            layout.addStretch(1)
        else:
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
            is_time = spec.kind == "datetime"
            self.slider.setFixedWidth(
                _SLIDER_WIDTH - (_DATETIME_WIDTH - _SPIN_WIDTH) if is_time else _SLIDER_WIDTH
            )
            self.slider.setAccessibleName(f"{caption} slider")
            self.slider.valueChanged.connect(self._on_slider)
            self.spin = self._make_spin()
            self.spin.setFixedWidth(_DATETIME_WIDTH if is_time else _SPIN_WIDTH)
            self.spin.setAccessibleName(caption)
            self.spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self.slider)
            layout.addWidget(self.spin)
        layout.addWidget(self.reset_button)
        self._apply_range()
        self.set_value(spec.default)

    # -- construction ------------------------------------------------------------
    def _make_spin(self) -> QtWidgets.QAbstractSpinBox:
        spec = self.spec
        if spec.kind == "datetime":
            edit = QtWidgets.QDateTimeEdit(self)
            edit.setTimeZone(_UTC)
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
            edit.setKeyboardTracking(False)
            edit.dateTimeChanged.connect(self._on_datetime)
            return edit
        if spec.kind == "int":
            ispin = QtWidgets.QSpinBox(self)
            ispin.setKeyboardTracking(False)
            ispin.setSingleStep(max(1, round(spec.step)))
            ispin.setSuffix(spec.suffix)
            ispin.setGroupSeparatorShown(spec.key == "seed")
            ispin.valueChanged.connect(self._on_spin)
            return ispin
        dspin = QtWidgets.QDoubleSpinBox(self)
        dspin.setKeyboardTracking(False)
        dspin.setDecimals(spec.decimals)
        dspin.setSingleStep(spec.step * spec.scale)
        dspin.setPrefix(spec.prefix)
        dspin.setSuffix(spec.suffix)
        dspin.setGroupSeparatorShown(bool(spec.prefix))
        dspin.valueChanged.connect(self._on_spin)
        return dspin

    def _apply_range(self) -> None:
        spec = self.spec
        self._guard = True
        try:
            if isinstance(self.spin, QtWidgets.QDateTimeEdit):
                self.spin.setDateTimeRange(
                    _to_qdatetime(int(self._minimum)), _to_qdatetime(int(self._maximum))
                )
            elif isinstance(self.spin, QtWidgets.QSpinBox):
                self.spin.setRange(round(self._minimum), round(self._maximum))
            elif isinstance(self.spin, QtWidgets.QDoubleSpinBox):
                self.spin.setRange(self._minimum * spec.scale, self._maximum * spec.scale)
            if self.slider is not None:
                steps = max(1, round((self._maximum - self._minimum) / spec.step))
                self.slider.setRange(0, steps)
                self.slider.setPageStep(max(1, steps // 20))
        finally:
            self._guard = False

    # -- state ---------------------------------------------------------------------
    @property
    def value(self) -> ParamValue:
        return self._value

    def set_range(self, minimum: float, maximum: float) -> None:
        """Change the allowed range (dataset bounds, the ``P_ref`` coupling)."""
        if minimum == self._minimum and maximum == self._maximum:
            return
        self._minimum, self._maximum = minimum, maximum
        self._apply_range()
        self._show(self._value)

    def set_default(self, default: ParamValue) -> None:
        self._default = default
        self._mark()

    def set_value(self, value: ParamValue) -> None:
        """Display ``value`` without emitting :attr:`valueChanged`."""
        self._value = value
        self._show(value)
        self._mark()

    def _show(self, value: ParamValue) -> None:
        spec = self.spec
        self._guard = True
        try:
            if self.toggle is not None:
                self.toggle.setChecked(bool(value))
            if isinstance(self.spin, QtWidgets.QDateTimeEdit):
                self.spin.setDateTime(_to_qdatetime(int(value)))
            elif isinstance(self.spin, QtWidgets.QSpinBox):
                self.spin.setValue(int(value))
            elif isinstance(self.spin, QtWidgets.QDoubleSpinBox):
                self.spin.setValue(float(value) * spec.scale)
            if self.slider is not None:
                self.slider.setValue(round((float(value) - self._minimum) / spec.step))
        finally:
            self._guard = False

    def _mark(self) -> None:
        modified = self._value != self._default
        self.reset_button.setEnabled(modified)
        flag = "true" if modified else "false"
        if self.property("modified") != flag:
            self.setProperty("modified", flag)
            self.style().unpolish(self.label)
            self.style().polish(self.label)

    def _commit(self, value: ParamValue) -> None:
        clamped: ParamValue
        if self.spec.kind == "bool":
            clamped = bool(value)
        else:
            bounded = min(max(float(value), self._minimum), self._maximum)
            clamped = round(bounded) if self.spec.kind in ("int", "datetime") else bounded
        if clamped == self._value:
            self._show(clamped)
            return
        self._value = clamped
        self._show(clamped)
        self._mark()
        self.valueChanged.emit(self.spec.key, clamped)

    # -- widget signals ------------------------------------------------------------
    def _on_slider(self, position: int) -> None:
        if self._guard:
            return
        raw = self._minimum + position * self.spec.step
        if self.spec.kind == "float":
            # Snap to the step grid without accumulating binary noise.
            raw = round(raw / self.spec.step) * self.spec.step
            raw = round(raw, 10)
        self._commit(raw)

    def _on_spin(self, shown: float) -> None:
        if not self._guard:
            self._commit(self.spec.from_display(float(shown)))

    def _on_datetime(self, moment: QtCore.QDateTime) -> None:
        if not self._guard:
            self._commit(int(moment.toMSecsSinceEpoch()))

    def _on_toggled(self, checked: bool) -> None:
        if not self._guard:
            self._commit(checked)


class ControlSidebar(QtWidgets.QWidget):
    """The full parameter column."""

    #: ``(key, value)`` of a parameter edited by the user.
    paramChanged = QtCore.Signal(str, object)
    resetRequested = QtCore.Signal(str)
    resetAllRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.rows: dict[str, ParamRow] = {}
        self.badges: list[HelpBadge] = []

        body = QtWidgets.QWidget(self)
        body.setObjectName("sidebarBody")
        column = QtWidgets.QVBoxLayout(body)
        column.setContentsMargins(0, 4, 0, 4)
        column.setSpacing(0)
        for group in GROUPS:
            column.addWidget(self._group_header(group))
            for spec in PARAM_SPECS:
                if spec.group != group or spec.key == "anchor_price":
                    continue
                row = ParamRow(spec, body)
                row.valueChanged.connect(self.paramChanged)
                row.resetRequested.connect(self.resetRequested)
                self.rows[spec.key] = row
                self.badges.append(row.badge)
                column.addWidget(row)
        column.addStretch(1)
        self._attach_anchor_spin()

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setObjectName("sidebarScroll")
        self.scroll_area.setWidget(body)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The gutter is reserved whether or not there is anything to scroll (the
        # empty track has the panel colour), so the scrollbar that appears on a
        # short screen never covers or shifts the reset buttons.
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setFixedWidth(SCROLL_GUTTER)
        scrollbar.rangeChanged.connect(lambda low, high: scrollbar.setEnabled(high > low))
        scrollbar.setEnabled(scrollbar.maximum() > scrollbar.minimum())

        footer = QtWidgets.QWidget(self)
        footer.setObjectName("sidebarFooter")
        footer.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.reset_all_button = QtWidgets.QPushButton("Reset All Parameters to Defaults", footer)
        self.reset_all_button.setObjectName("resetAll")
        self.reset_all_button.clicked.connect(self.resetAllRequested)
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 6, 10, 6)
        footer_layout.addWidget(self.reset_all_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(footer)
        self.body = body

    def _group_header(self, group: str) -> QtWidgets.QWidget:
        symbol = GROUP_PLAIN_SYMBOLS.get(GROUP_SYMBOLS.get(group, ""), "")
        text = group.upper() + (f"  ({symbol})" if symbol else "")
        label = QtWidgets.QLabel(text, self)
        label.setObjectName("groupHeader")
        label.setFixedHeight(GROUP_HEADER_HEIGHT)
        label.setContentsMargins(10, 5, 0, 0)
        return label

    def _attach_anchor_spin(self) -> None:
        """Place the anchor-price spin box inside the normalisation row."""
        spec = SPEC_BY_KEY["anchor_price"]
        row = self.rows["normalize"]
        self.anchor_spin = QtWidgets.QDoubleSpinBox(row)
        self.anchor_spin.setKeyboardTracking(False)
        self.anchor_spin.setDecimals(spec.decimals)
        self.anchor_spin.setRange(spec.minimum, spec.maximum)
        self.anchor_spin.setSingleStep(spec.step)
        self.anchor_spin.setPrefix(spec.prefix)
        self.anchor_spin.setGroupSeparatorShown(True)
        self.anchor_spin.setFixedWidth(_SPIN_WIDTH)
        self.anchor_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.anchor_spin.setAccessibleName("Reference Anchor Price")
        self.anchor_spin.setToolTip("Reference anchor price the window's first open is rescaled to")
        self.anchor_spin.valueChanged.connect(self._on_anchor)
        layout = row.layout()
        assert isinstance(layout, QtWidgets.QHBoxLayout)
        layout.insertWidget(layout.count() - 1, self.anchor_spin)
        self._anchor_guard = False

    def _on_anchor(self, value: float) -> None:
        if not self._anchor_guard:
            self.paramChanged.emit("anchor_price", float(value))

    # -- public API ----------------------------------------------------------------
    def set_params(self, params: SimParams) -> None:
        """Mirror a parameter struct into every control without emitting signals."""
        self.rows["p_ref"].set_range(p_ref_minimum(params.p_liq), SPEC_BY_KEY["p_ref"].maximum)
        for key, row in self.rows.items():
            row.set_value(getattr(params, key))
        self._anchor_guard = True
        try:
            self.anchor_spin.setValue(params.anchor_price)
        finally:
            self._anchor_guard = False
        self.anchor_spin.setEnabled(params.normalize)
        self.reset_all_button.setEnabled(params != self._defaults(params))

    def set_start_bounds(self, lo_ms: int, hi_ms: int, default_ms: int) -> None:
        """Bound the start-time controls to the active dataset (PRD section 5)."""
        row = self.rows["start_ms"]
        row.set_range(float(lo_ms), float(hi_ms))
        row.set_default(default_ms)
        self._start_default = default_ms

    def _defaults(self, params: SimParams) -> SimParams:
        from dataclasses import replace

        return replace(default_params(), start_ms=getattr(self, "_start_default", params.start_ms))

    def required_height(self) -> int:
        """Height at which the whole column is visible without scrolling."""
        return self.body.sizeHint().height() + self.reset_all_button.sizeHint().height() + 14

    def scrolls(self) -> bool:
        """Whether the column is taller than the room it has (short screens only)."""
        return self.scroll_area.verticalScrollBar().maximum() > 0

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)
