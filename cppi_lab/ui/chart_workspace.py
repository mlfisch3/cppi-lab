"""Three-pane synchronised chart canvas (PRD FR-4, TSD section 4.1).

One pyqtgraph ``GraphicsLayoutWidget`` holds three plots with stretch ratios
``55 : 25 : 20`` and a shared, locked time axis:

* pane 1 - price, floor levels, net equity and execution markers;
* pane 2 - BTC holdings inside the hysteresis deadband;
* pane 3 - decline and growth signals and the maker fill diagnostics.

pyqtgraph provides the layout, the view ranges and the mouse interaction
(drag to pan, wheel to zoom).  Everything visible - grid, layers, markers,
axes, tick labels, liquidation flashes, crosshair and read-outs - is drawn
by this module as two :class:`~cppi_lab.ui.chart_gl.Frame` display lists that
:class:`~cppi_lab.ui.chart_gl.ChartCanvas` replays in one pass:

* the *scene*, rebuilt when the visible range, the data or the size changes;
* the *overlay* (crosshair lines and read-outs), rebuilt when the cursor
  moves, which is why moving the crosshair costs almost nothing.

A thin header strip floats over the top of each pane and carries the title,
the ``?`` badge and a colour legend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets

from cppi_lab.engine.controller import RunOutput
from cppi_lab.engine.kernel_numba import (
    EVENT_HARD_BUFFER_LIQUIDATION,
    EVENT_MAKER_BUY,
    EVENT_MAKER_SELL,
    EVENT_MISSED_MAKER_BUY,
    EVENT_MISSED_MAKER_SELL,
    EVENT_TAKER_BUY,
    EVENT_TAKER_SELL,
    event_bit,
)
from cppi_lab.engine.monte_carlo import EnsembleResult
from cppi_lab.ui import theme
from cppi_lab.ui.chart_geometry import (
    time_tick_label,
    time_ticks,
    value_tick_label,
    value_ticks,
)
from cppi_lab.ui.chart_gl import (
    ChartCanvas,
    Frame,
    Layer,
    MarkerSet,
    Rgba,
    Series,
    SpriteAtlas,
    Transform,
    premultiplied,
)
from cppi_lab.ui.popover_manager import HelpBadge

FloatArray = NDArray[np.float64]

PANE_STRETCH: tuple[int, int, int] = (55, 25, 20)
HEADER_HEIGHT = 20
AXIS_WIDTH = 66
TIME_AXIS_HEIGHT = 22
#: Markers shrink once more than this many are inside the visible range.
MARKER_DENSITY_SOFT = 250
_STEM_PX = 16.0
#: Share of the visible value span added above and below an auto-ranged pane.
Y_PADDING = 0.04
#: Fixed value range of the signal pane: both signals live in [0, 1].
SIGNAL_RANGE: tuple[float, float] = (-0.03, 1.05)
#: Gap between tick labels and the axis line.
TICK_TEXT_GAP = 5.0
#: Free pixels kept between two liquidation badges.
BADGE_GAP_PX = 6.0
BADGE_PAD_X = 4.0
BADGE_PAD_Y = 2.0
#: Dash patterns ``(on, off)`` in pixels of the hover and selection lines.
HOVER_DASH: tuple[float, float] = (1.0, 2.0)
SELECT_DASH: tuple[float, float] = (4.0, 2.0)
#: Label crowding limits of a value axis: ``(labels already drawn, fill ratio)``.
_TEXT_FILL_LIMITS: tuple[tuple[int, float], ...] = ((0, 0.8), (2, 0.6), (4, 0.4), (6, 0.2))


@dataclass(frozen=True)
class LegendEntry:
    label: str
    colour: str
    dashed: bool = False


PANE_TITLES: tuple[str, str, str] = (
    "Price, Floor & Net Portfolio Equity",
    "BTC Holdings & Hysteresis Deadband",
    "Signals & Fill Probability Diagnostics",
)
PANE_POPOVERS: tuple[str, str, str] = ("pane-price", "pane-holdings", "pane-signals")
PANE_LEGENDS: tuple[tuple[LegendEntry, ...], ...] = (
    (
        LegendEntry("Close", theme.PRICE),
        LegendEntry("P_liq", theme.P_LIQ, dashed=True),
        LegendEntry("P_floor", theme.P_FLOOR),
        LegendEntry("V/N₀", theme.EQUITY),
        LegendEntry("▼ maker sell", theme.MAKER_SELL),
        LegendEntry("◆ taker sell", theme.TAKER_SELL),
        LegendEntry("▲ maker buy", theme.MAKER_BUY),
        LegendEntry("◆ taker buy", theme.TAKER_BUY),
        LegendEntry("○ missed", theme.MISSED),
    ),
    (
        LegendEntry("n held", theme.N_HELD),
        LegendEntry("n_max", theme.N_MAX),
        LegendEntry("n_ceil", theme.N_CEIL, dashed=True),
        LegendEntry("n_floor", theme.N_REBUY, dashed=True),
    ),
    (
        LegendEntry("S_down", theme.S_DOWN),
        LegendEntry("S_up", theme.S_UP),
        LegendEntry("P_fill stem · U dot", theme.FILL_PROB),
    ),
)


def _legend_html(entries: tuple[LegendEntry, ...]) -> str:
    parts = []
    for entry in entries:
        mark = "┄" if entry.dashed else "━"
        glyph = "" if entry.label[0] in "▼▲◆○" else f"{mark} "
        parts.append(f'<span style="color:{entry.colour}">{glyph}{entry.label}</span>')
    return "&nbsp;&nbsp;".join(parts)


def _font_families(css: str) -> list[str]:
    """Family names of a CSS ``font-family`` value."""
    return [name.strip().strip("'\"") for name in css.split(",") if name.strip()]


class _LayoutAxis(pg.AxisItem):
    """Reserves an axis strip in the plot layout; the canvas draws the axis itself."""

    def paint(
        self,
        p: QtGui.QPainter,
        opt: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        return

    def linkedViewChanged(self, view: pg.ViewBox, newRange: object = None) -> None:
        return


@dataclass(frozen=True, slots=True)
class Pane:
    """Geometry and visible ranges of one pane, in viewport pixels and data units."""

    left: float
    top: float
    width: float
    height: float
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def transform(self) -> Transform:
        """``(m11, dx, m22, dy)`` mapping data to pixels; values grow upwards."""
        m11 = self.width / (self.x1 - self.x0)
        m22 = -self.height / (self.y1 - self.y0)
        return m11, self.left - self.x0 * m11, m22, self.top - self.y1 * m22

    @property
    def drawable(self) -> bool:
        return self.width >= 2.0 and self.height >= 2.0 and self.x1 > self.x0 and self.y1 > self.y0


class PaneHeader(QtWidgets.QWidget):
    """Title, ``?`` badge and legend strip floating over a pane.

    The crosshair read-out is drawn by the canvas at the right end of the same
    strip; the legend gives way while both would not fit.
    """

    def __init__(self, index: int, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.title = QtWidgets.QLabel(PANE_TITLES[index], self)
        self.title.setObjectName("paneTitle")
        self.badge = HelpBadge(PANE_POPOVERS[index], self)
        self.legend = QtWidgets.QLabel(_legend_html(PANE_LEGENDS[index]), self)
        self.legend.setObjectName("paneLegend")
        self.legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(AXIS_WIDTH + 4, 0, 8, 0)
        layout.setSpacing(6)
        layout.addWidget(self.title)
        layout.addWidget(self.badge)
        layout.addSpacing(6)
        layout.addWidget(self.legend)
        layout.addStretch(1)
        self._readout_width = 0.0
        self._legend_width: int | None = None

    def set_readout_width(self, width: float) -> None:
        """Tell the strip how much room the read-out drawn by the canvas needs."""
        if width != self._readout_width:
            self._readout_width = width
            self.fit()

    def fit(self) -> None:
        """Show the legend only while it and the read-out both fit on the strip."""
        if self._legend_width is None:
            self._legend_width = self.legend.sizeHint().width()
        fixed = AXIS_WIDTH + 12 + self.title.sizeHint().width() + 60
        needed = fixed + self._legend_width + self._readout_width
        visible = needed <= self.width()
        if visible != self.legend.isVisibleTo(self):
            self.legend.setVisible(visible)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.fit()


class EmptyStateBanner(QtWidgets.QFrame):
    """Explains a back-test without any order and offers one-click remedies."""

    normalizeRequested = QtCore.Signal()
    scenarioRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("emptyBanner")
        title = QtWidgets.QLabel("No rebalancing in this window", self)
        title.setObjectName("bannerTitle")
        self.message = QtWidgets.QLabel("", self)
        self.message.setWordWrap(True)
        self.normalize_button = QtWidgets.QPushButton("Normalize start price to anchor", self)
        self.normalize_button.setObjectName("primary")
        self.normalize_button.clicked.connect(self.normalizeRequested)
        self.scenario_button = QtWidgets.QPushButton("Load a crash scenario", self)
        self.scenario_button.clicked.connect(self.scenarioRequested)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.normalize_button)
        buttons.addWidget(self.scenario_button)
        buttons.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(self.message)
        layout.addLayout(buttons)
        self.setFixedWidth(470)
        self.hide()


def _fmt_usd(value: float) -> str:
    return f"{value:,.2f}"


@dataclass(frozen=True, slots=True)
class Badge:
    """One liquidation label: where it sits and how many liquidations it stands for."""

    x: float
    y: float
    width: float
    height: float
    text: str
    count: int


class ChartWorkspace(QtWidgets.QWidget):
    """The three-pane canvas with crosshair and ledger synchronisation."""

    #: Bar index under the cursor (``-1`` when the cursor leaves the canvas).
    barHovered = QtCore.Signal(int)
    #: Bar index the user clicked; drives the ledger selection.
    barSelected = QtCore.Signal(int)
    normalizeRequested = QtCore.Signal()
    scenarioRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._run: RunOutput | None = None
        self._x: FloatArray = np.empty(0)
        self._n0 = 1.0
        self._hover_bar = -1
        self._hover_y: float | None = None
        self._selected_bar = -1
        self._marker_scale = 1.0
        self._pane_heights: list[int] = []
        #: Crosshair read-out of each pane ("" while nothing is hovered or selected).
        self.readouts: list[str] = ["", "", ""]
        #: Timestamps of the hard-buffer liquidations, and which of them gapped.
        self.liquidation_x: FloatArray = np.empty(0)
        self._liquidation_gap: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)
        #: Liquidation labels of the scene frame built last.
        self.badges: list[Badge] = []

        self._atlas: SpriteAtlas | None = None
        self._font_tick = self._font_badge = self._font_readout = 0
        self._scene = Frame()
        self._overlay = Frame()
        self._scene_key: tuple[object, ...] | None = None
        self._overlay_key: tuple[object, ...] | None = None
        self._data_version = 0
        self._y_targets: list[tuple[float, float] | None] = [None, None]
        self._value_tick_cache: dict[
            int, tuple[tuple[float, float, float], list[tuple[float, FloatArray, float]]]
        ] = {}

        self.canvas = ChartCanvas(self, self._frames)
        self.canvas.setBackground(theme.CANVAS)
        self.canvas.ci.layout.setContentsMargins(0, 2, 6, 0)
        self.canvas.ci.layout.setSpacing(2)
        self.canvas.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.canvas.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.canvas.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.canvas)

        self.plots: list[pg.PlotItem] = []
        for row in range(3):
            plot = self._make_plot(row)
            self.canvas.ci.layout.setRowStretchFactor(row, PANE_STRETCH[row])
            self.plots.append(plot)
        self.plots[1].setXLink(self.plots[0])
        self.plots[2].setXLink(self.plots[0])
        self.plots[2].getViewBox().setYRange(*SIGNAL_RANGE, padding=0)

        self._build_layers()

        self.headers = [PaneHeader(i, self.canvas) for i in range(3)]
        self.banner = EmptyStateBanner(self.canvas)
        self.banner.normalizeRequested.connect(self.normalizeRequested)
        self.banner.scenarioRequested.connect(self.scenarioRequested)

        self.canvas.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.canvas.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.plots[0].getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)
        for plot in self.plots:
            plot.getViewBox().sigResized.connect(self._schedule_layout)
        self._layout_timer = QtCore.QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(0)
        self._layout_timer.timeout.connect(self._layout_overlays)
        self._schedule_layout()

    # -- construction --------------------------------------------------------------
    def _make_plot(self, row: int) -> pg.PlotItem:
        plot: pg.PlotItem = self.canvas.addPlot(
            row=row,
            col=0,
            axisItems={side: _LayoutAxis(side) for side in ("left", "bottom", "top", "right")},
        )
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setTitle(None)
        plot.layout.setRowFixedHeight(0, HEADER_HEIGHT)
        plot.layout.setRowMinimumHeight(0, HEADER_HEIGHT)
        view = plot.getViewBox()
        view.setMouseEnabled(x=True, y=False)
        # The value ranges are set by _auto_range from the layers in view.
        view.enableAutoRange(enable=False)
        plot.getAxis("left").setWidth(AXIS_WIDTH)
        if row < 2:
            plot.hideAxis("bottom")  # the three panes share the time axis of the last one
        else:
            plot.getAxis("bottom").setHeight(TIME_AXIS_HEIGHT)
        return plot

    def _layer(self, pane: int, name: str, *args: object, **kwargs: object) -> Layer:
        layer = Layer(*args, **kwargs)  # type: ignore[arg-type]
        self._layers[pane].append((f"layer:{name}", layer))
        return layer

    def _marker(
        self, pane: int, name: str, symbol: str, size: float, colour: str, hollow: bool = False
    ) -> MarkerSet:
        markers = MarkerSet(symbol, size, colour, hollow)
        self._markers[pane].append((f"marker:{name}", markers))
        return markers

    def _build_layers(self) -> None:
        """Every layer of the TSD visual-encoding matrix, back to front within its pane."""
        self._layers: list[list[tuple[str, Layer]]] = [[], [], []]
        self._markers: list[list[tuple[str, MarkerSet]]] = [[], [], []]
        # Pane 1
        self.mc_fan = self._layer(0, "mc_fan", "band", theme.MC_FAN)
        self.price_band = self._layer(0, "price_band", "band", theme.PRICE_BAND)
        self.floor_zone = self._layer(0, "floor_zone", "band", theme.P_FLOOR_ZONE)
        self.p_liq = self._layer(0, "p_liq", "line", theme.P_LIQ, 1.5, (6.0, 4.0))
        self.p_floor = self._layer(0, "p_floor", "step", theme.P_FLOOR, 2.2)
        self.price = self._layer(0, "price", "line", theme.PRICE, 1.6)
        self.equity_glow = self._layer(0, "equity_glow", "line", theme.EQUITY_GLOW, 7.0)
        self.equity = self._layer(0, "equity", "line", theme.EQUITY, 2.4)
        self.taker_sell_stems = self._layer(
            0, "taker_sell_stems", "stems", theme.TAKER_SELL, 1.4, stem_px=_STEM_PX
        )
        self.markers: dict[int, MarkerSet] = {
            EVENT_MISSED_MAKER_SELL: self._marker(0, "missed", "o", 13, theme.MISSED, hollow=True),
            EVENT_MAKER_SELL: self._marker(0, "maker_sell", "t", 11, theme.MAKER_SELL),
            EVENT_MAKER_BUY: self._marker(0, "maker_buy", "t1", 11, theme.MAKER_BUY),
            EVENT_TAKER_BUY: self._marker(0, "taker_buy", "d", 12, theme.TAKER_BUY),
            EVENT_TAKER_SELL: self._marker(0, "taker_sell", "d", 12, theme.TAKER_SELL),
        }
        # Pane 2
        self.deadband = self._layer(1, "deadband", "band", theme.DEADBAND, reduce="mean")
        self.n_max = self._layer(1, "n_max", "line", theme.N_MAX, 1.2, reduce="mean")
        self.n_rebuy = self._layer(
            1, "n_rebuy", "line", theme.N_REBUY, 1.5, (6.0, 4.0), reduce="mean"
        )
        self.n_ceil = self._layer(1, "n_ceil", "line", theme.N_CEIL, 1.5, (6.0, 4.0), reduce="mean")
        self.n_held = self._layer(1, "n_held", "step", theme.N_HELD, 2.5)
        # Pane 3
        self.s_down_fill = self._layer(2, "s_down_fill", "band", theme.S_DOWN_FILL, reduce="mean")
        self.s_up_fill = self._layer(2, "s_up_fill", "band", theme.S_UP_FILL, reduce="mean")
        self.s_down = self._layer(2, "s_down", "line", theme.S_DOWN, 1.5, reduce="mean")
        self.s_up = self._layer(2, "s_up", "line", theme.S_UP, 1.5, reduce="mean")
        self.fill_stems = self._layer(2, "fill_stems", "segments", theme.FILL_PROB, 1.6)
        self.draw_hit = self._marker(2, "draw_hit", "o", 6, theme.FILL_PROB)
        self.draw_miss = self._marker(2, "draw_miss", "o", 7, theme.MISSED, hollow=True)

    # -- data ----------------------------------------------------------------------
    def set_run(self, run: RunOutput, keep_view: bool = False) -> None:
        """Display a back-test.  ``keep_view`` preserves the current zoom."""
        tel = run.result.telemetry
        params = run.params
        previous = self._run
        self._run = run
        self._n0 = params.n0
        x = tel["ts_utc_ms"].astype(np.float64) / 1000.0
        same_axis = (
            previous is not None
            and keep_view
            and self._x.shape == x.shape
            and self._x[0] == x[0]
            and self._x[-1] == x[-1]
        )
        self._x = x

        # One pyramid per series, shared by every layer that draws it.
        p_floor = Series(tel["p_floor"])
        equity = Series(tel["net_equity_usd"] / params.n0)
        liq = Series(np.full(x.shape[0], params.p_liq))
        self.price_band.set_band(x, tel["low"], tel["high"])
        self.floor_zone.set_band(x, liq, p_floor)
        self.p_liq.set_polyline(x, liq)
        self.p_floor.set_polyline(x, p_floor)
        self.price.set_polyline(x, tel["close"])
        self.equity_glow.set_polyline(x, equity)
        self.equity.set_polyline(x, equity)

        n_rebuy, n_ceil = Series(tel["n_rebuy"]), Series(tel["n_ceil"])
        self.deadband.set_band(x, n_rebuy, n_ceil)
        self.n_max.set_polyline(x, tel["n_max"])
        self.n_rebuy.set_polyline(x, n_rebuy)
        self.n_ceil.set_polyline(x, n_ceil)
        self.n_held.set_polyline(x, tel["btc_held"])

        zeros = Series(np.zeros(x.shape[0]))
        s_down, s_up = Series(tel["s_down"]), Series(tel["s_up"])
        self.s_down_fill.set_band(x, zeros, s_down)
        self.s_up_fill.set_band(x, zeros, s_up)
        self.s_down.set_polyline(x, s_down)
        self.s_up.set_polyline(x, s_up)

        self._set_markers(run)
        bars = np.nonzero(run.result.flags & event_bit(EVENT_HARD_BUFFER_LIQUIDATION))[0]
        self.liquidation_x = x[bars]
        self._liquidation_gap = bars == run.result.stats.gap_bar
        self._data_version += 1
        if not same_axis:
            self.mc_fan.clear()
            self.reset_view()
        self._auto_range()
        self._update_banner(run)
        self._update_marker_density()
        self._apply_selection()
        self._refresh_readout()
        self._repaint()

    def _set_markers(self, run: RunOutput) -> None:
        tel = run.result.telemetry
        flags = run.result.flags
        x = self._x
        close = tel["close"]
        prev_close = np.concatenate(([close[0]], close[:-1]))
        slip = run.params.delta_slip
        levels = {
            EVENT_MAKER_SELL: prev_close,
            EVENT_MAKER_BUY: prev_close,
            EVENT_TAKER_SELL: close * (1.0 - slip),
            EVENT_TAKER_BUY: close * (1.0 + slip),
        }
        for code, markers in self.markers.items():
            if code == EVENT_MISSED_MAKER_SELL:
                mask = (flags & (event_bit(code) | event_bit(EVENT_MISSED_MAKER_BUY))) != 0
                y = prev_close
            else:
                mask = (flags & event_bit(code)) != 0
                y = levels[code]
            markers.set_data(x[mask], y[mask])
        sell_mask = (flags & event_bit(EVENT_TAKER_SELL)) != 0
        self.taker_sell_stems.set_stems(x[sell_mask], levels[EVENT_TAKER_SELL][sell_mask])

        p_fill = tel["p_fill_prob"]
        attempt = np.isfinite(p_fill)
        ax, ap, au = x[attempt], p_fill[attempt], tel["u_rand_draw"][attempt]
        self.fill_stems.set_segments(ax, np.zeros(ax.shape[0]), ax, ap)
        hit = (au <= ap) & (ap > 0.0)
        self.draw_hit.set_data(ax[hit], au[hit])
        self.draw_miss.set_data(ax[~hit], au[~hit])

    def set_ensemble(self, ensemble: EnsembleResult | None) -> None:
        """Show or clear the Monte Carlo 5-95 % equity fan (FR-3.3)."""
        if ensemble is None or self._x.shape[0] == 0:
            self.mc_fan.clear()
        else:
            idx = ensemble.sample_idx
            idx = idx[idx < self._x.shape[0]]
            count = idx.shape[0]
            self.mc_fan.set_band(
                self._x[idx], ensemble.p05[:count] / self._n0, ensemble.p95[:count] / self._n0
            )
        self._data_version += 1
        self._auto_range()
        self._repaint()

    def _update_banner(self, run: RunOutput) -> None:
        stats = run.result.stats
        active = stats.maker_attempts + stats.taker_fills + stats.liquidations
        if active > 0:
            self.banner.hide()
            return
        p = run.params
        low = float(run.window.low.min())
        self.banner.message.setText(
            f"BTC traded between {_fmt_usd(low)} and {_fmt_usd(float(run.window.high.max()))} "
            f"USD here, always above the full-allocation reference P_ref = "
            f"{_fmt_usd(p.p_ref)}, so the envelope never asks for a trade and the strategy "
            f"holds its whole allocation.  Rescale the window to start near the floor, or "
            f"jump to a period that tests it."
        )
        self.banner.normalize_button.setVisible(not p.normalize)
        self.banner.adjustSize()
        self.banner.show()
        self.banner.raise_()
        self._schedule_layout()

    # -- view ----------------------------------------------------------------------
    def reset_view(self) -> None:
        """Zoom out to the whole back-test."""
        if self._x.shape[0] < 2:
            return
        view = self.plots[0].getViewBox()
        x0, x1 = float(self._x[0]), float(self._x[-1])
        view.setLimits(xMin=x0, xMax=x1, minXRange=600.0)
        view.setXRange(x0, x1, padding=0)
        self._auto_range()

    def visible_range(self) -> tuple[float, float]:
        (x0, x1), _ = self.plots[0].getViewBox().viewRange()
        return float(x0), float(x1)

    def set_visible_range(self, x0: float, x1: float) -> None:
        self.plots[0].getViewBox().setXRange(x0, x1, padding=0)

    def centre_on_bar(self, bar: int) -> None:
        """Pan (without changing zoom) so ``bar`` is inside the visible range."""
        if not 0 <= bar < self._x.shape[0]:
            return
        x0, x1 = self.visible_range()
        x = float(self._x[bar])
        if x0 <= x <= x1:
            return
        half = 0.5 * (x1 - x0)
        lo = max(float(self._x[0]), x - half)
        hi = min(float(self._x[-1]), lo + 2.0 * half)
        self.set_visible_range(hi - 2.0 * half, hi)

    def _on_x_range_changed(self) -> None:
        self._auto_range()
        self._update_marker_density()
        self._repaint()

    def _auto_range(self) -> None:
        """Fit the value axes of panes 1 and 2 to the samples inside the visible range.

        Every series is asked once even when several layers draw it, and the
        answer comes from its pyramid, so this costs microseconds per frame.
        """
        x0, x1 = self.visible_range()
        spans: dict[int, tuple[int, int]] = {}
        for index in (0, 1):
            low, high = math.inf, -math.inf
            seen: set[int] = set()
            for _, layer in self._layers[index]:
                for x, series in layer.ranged():
                    if id(series) in seen:
                        continue
                    seen.add(id(series))
                    span = spans.get(id(x))
                    if span is None:
                        span = (
                            max(0, int(np.searchsorted(x, x0, side="left")) - 1),
                            min(x.shape[0], int(np.searchsorted(x, x1, side="right")) + 1),
                        )
                        spans[id(x)] = span
                    if span[1] <= span[0]:
                        continue
                    series_low, series_high = series.min_max(*span)
                    if series_low == series_low:  # not NaN: the range holds finite samples
                        low, high = min(low, series_low), max(high, series_high)
            if low > high:
                continue
            extent = high - low
            pad = extent * Y_PADDING if extent > 0.0 else max(abs(high) * 0.01, 1e-6)
            target = (low - pad, high + pad)
            if target != self._y_targets[index]:
                self._y_targets[index] = target
                self.plots[index].getViewBox().setYRange(*target, padding=0)

    def _update_marker_density(self) -> None:
        if self._run is None or self._x.shape[0] == 0:
            return
        x0, x1 = self.visible_range()
        lo = int(np.searchsorted(self._x, x0))
        hi = int(np.searchsorted(self._x, x1))
        visible = int(np.count_nonzero(self._run.result.flags[lo:hi] & 0x3F))
        scale = 1.0
        if visible > MARKER_DENSITY_SOFT:
            scale = max(0.45, (MARKER_DENSITY_SOFT / visible) ** 0.5)
        scale = round(scale * 20.0) / 20.0
        if scale == self._marker_scale:
            return
        self._marker_scale = scale
        self.taker_sell_stems.stem_px = _STEM_PX * scale
        self.fill_stems.width = 1.6 * max(0.6, scale)
        self.fill_stems.set_alpha(0.55 + 0.45 * (scale - 0.45) / 0.55)

    @property
    def marker_scale(self) -> float:
        """Current size factor of the markers (1 unless very many are in view)."""
        return self._marker_scale

    # -- overlays ------------------------------------------------------------------
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._schedule_layout()

    def _schedule_layout(self) -> None:
        self._layout_timer.start()

    def _apply_pane_heights(self) -> None:
        """Give the three panes exactly the 55 : 25 : 20 share of the canvas height.

        Layout stretch factors only distribute the space left over after each
        row's own size hint, which skews the split; fixed row heights do not.
        """
        layout = self.canvas.ci.layout
        _, top, _, bottom = layout.getContentsMargins()
        available = self.canvas.height() - top - bottom - 2.0 * layout.verticalSpacing()
        if available < 150:
            return
        heights = [round(available * share / sum(PANE_STRETCH)) for share in PANE_STRETCH]
        heights[2] = int(available) - heights[0] - heights[1]
        if heights != self._pane_heights:
            self._pane_heights = heights
            for row, height in enumerate(heights):
                layout.setRowFixedHeight(row, height)

    def _layout_overlays(self) -> None:
        self._apply_pane_heights()
        for plot, header in zip(self.plots, self.headers, strict=True):
            rect = plot.sceneBoundingRect()
            header.setGeometry(int(rect.left()), int(rect.top()), int(rect.width()), HEADER_HEIGHT)
            header.raise_()
        if self.banner.isVisible():
            area = self.plots[0].getViewBox().sceneBoundingRect()
            x = int(area.left() + (area.width() - self.banner.width()) / 2)
            y = int(area.top() + max(8.0, (area.height() - self.banner.height()) * 0.42))
            self.banner.move(max(0, x), max(0, y))
            self.banner.raise_()
        self._repaint()

    def _repaint(self) -> None:
        self.canvas.viewport().update()

    # -- frames --------------------------------------------------------------------
    def pane(self, index: int) -> Pane:
        """Geometry and visible ranges of one pane."""
        view = self.plots[index].getViewBox()
        rect = view.sceneBoundingRect()
        (x0, x1), (y0, y1) = view.viewRange()
        return Pane(rect.left(), rect.top(), rect.width(), rect.height(), x0, x1, y0, y1)

    def _ensure_atlas(self) -> SpriteAtlas:
        dpr = float(self.canvas.devicePixelRatioF())
        if self._atlas is None or self._atlas.dpr != dpr:
            atlas = SpriteAtlas(dpr)
            tick = QtGui.QFont()
            tick.setPixelSize(10)
            badge = QtGui.QFont()
            badge.setPixelSize(9)
            badge.setBold(True)
            readout = QtGui.QFont()
            readout.setFamilies(_font_families(theme.MONO_FAMILY))
            readout.setStyleHint(QtGui.QFont.StyleHint.Monospace)
            readout.setPixelSize(10)
            self._font_tick = atlas.add_font(tick)
            self._font_badge = atlas.add_font(badge)
            self._font_readout = atlas.add_font(readout)
            self._atlas = atlas
        return self._atlas

    def _frames(self) -> tuple[tuple[Frame, Frame], SpriteAtlas]:
        """The scene and overlay frames, rebuilt only when what they show has changed."""
        atlas = self._ensure_atlas()
        panes = tuple(self.pane(i) for i in range(3))
        size = (self.canvas.width(), self.canvas.height(), atlas.dpr)
        while True:
            generation = atlas.generation
            scene_key = (self._data_version, size, panes, self._marker_scale, generation)
            if scene_key != self._scene_key:
                self._build_scene(atlas, panes)
                self._scene_key = scene_key
                self._overlay_key = None
            overlay_key = (self._hover_bar, self._hover_y, self._selected_bar, *self.readouts)
            if overlay_key != self._overlay_key:
                self._build_overlay(atlas, panes)
                self._overlay_key = overlay_key
            if atlas.generation == generation:
                break
            self._scene_key = None  # the atlas started over: its coordinates changed
        return (self._scene, self._overlay), atlas

    def frames(self) -> tuple[Frame, Frame]:
        """The up-to-date ``(scene, overlay)`` display lists (for tests and diagnostics)."""
        return self._frames()[0]

    def _value_ticks(self, index: int, pane: Pane) -> list[tuple[float, FloatArray, float]]:
        key = (pane.y0, pane.y1, pane.height)
        cached = self._value_tick_cache.get(index)
        if cached is not None and cached[0] == key:
            return cached[1]
        levels = value_ticks(pane.y0, pane.y1, pane.height)
        self._value_tick_cache[index] = (key, levels)
        return levels

    def _build_scene(self, atlas: SpriteAtlas, panes: tuple[Pane, ...]) -> None:
        frame = self._scene
        frame.begin(self.canvas.width(), self.canvas.height(), atlas.dpr)
        grid = pg.mkColor(theme.GRID)
        grid_rgba = premultiplied(grid)
        flash_rgba = premultiplied(pg.mkColor(theme.LIQUIDATION))
        text_rgba = premultiplied(pg.mkColor(theme.TEXT_MUTED))
        scale = self._marker_scale
        self.badges = []
        step = 0
        time_values: list[float] = []
        if panes[2].drawable:
            step, time_values = time_ticks(panes[2].x0, panes[2].x1, panes[2].width)
        tick_x = np.array(time_values, dtype=np.float64)

        for index, pane in enumerate(panes):
            if not pane.drawable:
                continue
            m11, dx, m22, dy = transform = pane.transform
            clip = frame.clip(pane.left, pane.top, pane.width, pane.height)
            pixels = max(16, int(pane.width))
            levels = self._value_ticks(index, pane)

            # Grid: faint levels first, then the major lines, then the time grid.
            for _, values, opacity in reversed(levels):
                if values.shape[0] == 0:
                    continue
                ys = np.floor(values * m22 + dy) + 0.5
                frame.add_segments(
                    "grid-h",
                    clip,
                    np.full(ys.shape[0], pane.left),
                    ys,
                    np.full(ys.shape[0], pane.right),
                    ys,
                    premultiplied(grid, opacity),
                )
            if tick_x.shape[0]:
                xs = np.floor(tick_x * m11 + dx) + 0.5
                frame.add_segments(
                    "grid-v",
                    clip,
                    xs,
                    np.full(xs.shape[0], pane.top),
                    xs,
                    np.full(xs.shape[0], pane.bottom),
                    grid_rgba,
                )

            spans: dict[int, tuple[int, int]] = {}
            for tag, layer in self._layers[index]:
                span = None
                if layer.kind in ("line", "step", "band") and layer.size:
                    span = spans.get(id(layer._x))
                    if span is None:
                        span = spans[id(layer._x)] = layer.visible(pane.x0, pane.x1)
                frame.add_layer(tag, clip, layer, transform, pane.x0, pane.x1, pixels, span)

            for tag, markers in self._markers[index]:
                if markers.x.shape[0] == 0:
                    continue
                sprite = atlas.marker(
                    markers.symbol, markers.base_size * scale, markers.colour, markers.hollow
                )
                margin = sprite.width / m11
                lo = int(np.searchsorted(markers.x, pane.x0 - margin, side="left"))
                hi = int(np.searchsorted(markers.x, pane.x1 + margin, side="right"))
                if hi > lo:
                    frame.add_markers(
                        tag,
                        clip,
                        sprite,
                        markers.x[lo:hi] * m11 + dx,
                        markers.y[lo:hi] * m22 + dy,
                    )

            # Hard-buffer liquidations flash through all three panes.
            if self.liquidation_x.shape[0]:
                lo = int(np.searchsorted(self.liquidation_x, pane.x0, side="left"))
                hi = int(np.searchsorted(self.liquidation_x, pane.x1, side="right"))
                if hi > lo:
                    xs = np.floor(self.liquidation_x[lo:hi] * m11 + dx) + 0.5
                    frame.add_segments(
                        "liquidation",
                        clip,
                        xs,
                        np.full(xs.shape[0], pane.top),
                        xs,
                        np.full(xs.shape[0], pane.bottom),
                        flash_rgba,
                        1.6,
                    )
                    if index == 0:
                        self._add_badges(frame, atlas, clip, pane, lo, hi)

        # Axes are drawn outside the clip rectangles of the panes.
        ascent = atlas.ascent(self._font_tick)
        descent = atlas.descent(self._font_tick)
        text_height = ascent + descent
        for index, pane in enumerate(panes):
            if not pane.drawable:
                continue
            _, _, m22, dy = pane.transform
            edge = np.array([pane.left - 0.5])
            frame.add_segments(
                "axis",
                -1,
                edge,
                np.array([pane.top]),
                edge,
                np.array([pane.bottom + (1.0 if index == 2 else 0.0)]),
                grid_rgba,
            )
            labels: list[tuple[str, float, float]] = []
            stacked = 0.0
            for level, (spacing, values, _) in enumerate(self._value_ticks(index, pane)):
                if values.shape[0] == 0:
                    continue
                stacked += values.shape[0] * text_height
                if level > 0:
                    fill = stacked / pane.height
                    drawn = len(labels)
                    if any(drawn >= count and fill >= limit for count, limit in _TEXT_FILL_LIMITS):
                        break
                x = pane.left - TICK_TEXT_GAP
                shift = dy + 0.5 * (ascent - descent)
                labels.extend(
                    (value_tick_label(value, spacing), x, value * m22 + shift)
                    for value in values.tolist()
                )
            frame.add_texts("tick-y", -1, atlas, self._font_tick, labels, text_rgba, align=1.0)
        last = panes[2]
        if last.drawable:
            m11, dx, _, _ = last.transform
            frame.add_segments(
                "axis",
                -1,
                np.array([last.left - 1.0]),
                np.array([last.bottom + 0.5]),
                np.array([last.right]),
                np.array([last.bottom + 0.5]),
                grid_rgba,
            )
            labels = []
            baseline = last.bottom + 4.0 + ascent
            for value in time_values:
                label = time_tick_label(value, float(step))
                half = 0.5 * atlas.text_width(self._font_tick, label)
                x = value * m11 + dx
                if last.left <= x - half and x + half <= last.right:  # only where it fits whole
                    labels.append((label, x, baseline))
            frame.add_texts("tick-x", -1, atlas, self._font_tick, labels, text_rgba, align=0.5)

    def _badge_text(self, count: int, gap: bool) -> str:
        if count == 1:
            return "GAP LIQUIDATION" if gap else "LIQUIDATION"
        return f"LIQUIDATION ×{count}" + (" · GAP" if gap else "")

    def _add_badges(
        self, frame: Frame, atlas: SpriteAtlas, clip: int, pane: Pane, lo: int, hi: int
    ) -> None:
        """Label the liquidations in view, one badge per cluster of nearby flashes.

        Liquidations whose flash lines would fall inside one label are counted
        on that label instead of stacking unreadable ones on top of each other.
        Badges are opaque and drawn after every flash line, so no line can
        strike through a label; a badge that would leave the pane on the right
        is placed to the left of its line.
        """
        font = self._font_badge
        m11, dx, _, _ = pane.transform
        px = self.liquidation_x[lo:hi] * m11 + dx
        gaps = self._liquidation_gap[lo:hi]
        height = atlas.ascent(font) + atlas.descent(font) + 2.0 * BADGE_PAD_Y
        fill = premultiplied(pg.mkColor(theme.LIQUIDATION))
        ink = premultiplied(pg.mkColor(theme.CANVAS))
        row_ends: list[float] = []  # right edge of the last badge placed on each row
        start = 0
        total = px.shape[0]
        while start < total:
            stop = start + 1
            while stop < total:
                text = self._badge_text(stop - start, bool(gaps[start:stop].any()))
                reach = atlas.text_width(font, text) + 2.0 * BADGE_PAD_X + BADGE_GAP_PX
                if px[stop] >= px[start] + reach:
                    break
                stop += 1
            count = stop - start
            text = self._badge_text(count, bool(gaps[start:stop].any()))
            width = atlas.text_width(font, text) + 2.0 * BADGE_PAD_X
            x = float(px[start]) + 1.0
            if x + width > pane.right:
                x = max(pane.left, float(px[start]) - width)
            row = next(
                (i for i, end in enumerate(row_ends) if x >= end + BADGE_GAP_PX), len(row_ends)
            )
            if row == len(row_ends):
                row_ends.append(x + width)
            else:
                row_ends[row] = x + width
            y = pane.top + 1.0 + row * (height + 2.0)
            frame.add_rect("badge", clip, x, y, width, height, fill)
            frame.add_text(
                "badge-text",
                clip,
                atlas,
                font,
                text,
                x + BADGE_PAD_X,
                y + BADGE_PAD_Y + atlas.ascent(font),
                ink,
            )
            self.badges.append(Badge(x, y, width, height, text, count))
            start = stop

    def _build_overlay(self, atlas: SpriteAtlas, panes: tuple[Pane, ...]) -> None:
        frame = self._overlay
        frame.begin(self.canvas.width(), self.canvas.height(), atlas.dpr)
        hover: Rgba = premultiplied(pg.mkColor(theme.CROSSHAIR))
        select: Rgba = premultiplied(pg.mkColor(theme.SELECTION))
        text_rgba = premultiplied(pg.mkColor(theme.TEXT_MUTED))
        lines = (
            ("crosshair-select", self._selected_bar, select, SELECT_DASH),
            ("crosshair-hover", self._hover_bar, hover, HOVER_DASH),
        )
        for index, pane in enumerate(panes):
            if not pane.drawable:
                continue
            m11, dx, m22, dy = pane.transform
            clip = frame.clip(pane.left, pane.top, pane.width, pane.height)
            for tag, bar, rgba, dash in lines:
                if not 0 <= bar < self._x.shape[0]:
                    continue
                x = math.floor(float(self._x[bar]) * m11 + dx) + 0.5
                if pane.left <= x <= pane.right:
                    frame.add_segments(
                        tag,
                        clip,
                        np.array([x]),
                        np.array([pane.top]),
                        np.array([x]),
                        np.array([pane.bottom]),
                        rgba,
                        1.0,
                        dash,
                    )
            if index == 0 and self._hover_bar >= 0 and self._hover_y is not None:
                y = math.floor(self._hover_y * m22 + dy) + 0.5
                frame.add_segments(
                    "crosshair-level",
                    clip,
                    np.array([pane.left]),
                    np.array([y]),
                    np.array([pane.right]),
                    np.array([y]),
                    hover,
                    1.0,
                    HOVER_DASH,
                )
        font = self._font_readout
        centre = 0.5 * (HEADER_HEIGHT + atlas.ascent(font) - atlas.descent(font))
        for index, pane in enumerate(panes):
            text = self.readouts[index]
            if text and pane.drawable:
                strip_top = pane.top - HEADER_HEIGHT
                frame.add_text(
                    "readout",
                    -1,
                    atlas,
                    font,
                    text,
                    pane.right - 8.0,
                    strip_top + centre,
                    text_rgba,
                    align=1.0,
                )

    # -- crosshair -------------------------------------------------------------------
    def bar_at(self, x: float) -> int:
        """Index of the bar whose timestamp is nearest to ``x`` (epoch seconds)."""
        n = self._x.shape[0]
        if n == 0:
            return -1
        i = int(np.searchsorted(self._x, x))
        if i <= 0:
            return 0
        if i >= n:
            return int(n - 1)
        nearer_right = bool((self._x[i] - x) < (x - self._x[i - 1]))
        return i if nearer_right else i - 1

    def _pane_at(self, scene_pos: QtCore.QPointF) -> int:
        for index, plot in enumerate(self.plots):
            if plot.getViewBox().sceneBoundingRect().contains(scene_pos):
                return index
        return -1

    def _on_mouse_moved(self, scene_pos: QtCore.QPointF) -> None:
        pane = self._pane_at(scene_pos)
        if pane < 0 or self._x.shape[0] == 0:
            self.set_hover_bar(-1)
            return
        point = self.plots[pane].getViewBox().mapSceneToView(scene_pos)
        level = float(point.y()) if pane == 0 else None
        if level != self._hover_y:
            self._hover_y = level
            self._repaint()
        self.set_hover_bar(self.bar_at(point.x()))

    def _on_mouse_clicked(self, event: object) -> None:
        scene_pos = event.scenePos()  # type: ignore[attr-defined]
        if event.button() != QtCore.Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
            return
        pane = self._pane_at(scene_pos)
        if pane < 0 or self._x.shape[0] == 0:
            return
        if event.double():  # type: ignore[attr-defined]
            self.reset_view()
            return
        point = self.plots[pane].getViewBox().mapSceneToView(scene_pos)
        bar = self.bar_at(point.x())
        self.set_selected_bar(bar)
        self.barSelected.emit(bar)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.set_hover_bar(-1)
        super().leaveEvent(event)

    def set_hover_bar(self, bar: int) -> None:
        """Move the hover crosshair to ``bar`` (``-1`` hides it)."""
        if bar == self._hover_bar:
            return
        self._hover_bar = bar
        if bar < 0:
            self._hover_y = None
        self._refresh_readout()
        self._repaint()
        self.barHovered.emit(bar)

    def set_selected_bar(self, bar: int) -> None:
        """Pin the selection marker on ``bar`` (driven by the ledger)."""
        self._selected_bar = bar if 0 <= bar < self._x.shape[0] else -1
        self._refresh_readout()
        self._repaint()

    @property
    def selected_bar(self) -> int:
        return self._selected_bar

    @property
    def hover_bar(self) -> int:
        return self._hover_bar

    @property
    def hover_time(self) -> float | None:
        """Timestamp under the hover crosshair, or ``None`` while it is hidden."""
        return float(self._x[self._hover_bar]) if self._hover_bar >= 0 else None

    @property
    def selected_time(self) -> float | None:
        """Timestamp of the selection line, or ``None`` while nothing is selected."""
        return float(self._x[self._selected_bar]) if self._selected_bar >= 0 else None

    def _apply_selection(self) -> None:
        if self._selected_bar >= self._x.shape[0]:
            self._selected_bar = -1

    def _refresh_readout(self) -> None:
        run = self._run
        bar = self._hover_bar if self._hover_bar >= 0 else self._selected_bar
        if run is None or bar < 0 or bar >= run.result.n_bars:
            texts = ["", "", ""]
        else:
            row = run.result.telemetry[bar]
            stamp = QtCore.QDateTime.fromMSecsSinceEpoch(
                int(row["ts_utc_ms"]), QtCore.QTimeZone.utc()
            ).toString("yyyy-MM-dd HH:mm")
            fill = row["p_fill_prob"]
            attempt = f"   P_fill {fill:.3f} vs U {row['u_rand_draw']:.3f}" if fill == fill else ""
            texts = [
                f"{stamp} UTC   C {_fmt_usd(row['close'])}   P_floor {_fmt_usd(row['p_floor'])}"
                f"   V/N₀ {_fmt_usd(row['net_equity_usd'] / self._n0)}",
                f"n {row['btc_held']:.4f}   ceil {row['n_ceil']:.4f}   floor "
                f"{row['n_rebuy']:.4f}   max {row['n_max']:.4f}   d {row['d_norm']:.3f}",
                f"S_down {row['s_down']:.3f}   S_up {row['s_up']:.3f}{attempt}",
            ]
        if texts == self.readouts:
            return
        self.readouts = texts
        atlas = self._ensure_atlas()
        for header, text in zip(self.headers, texts, strict=True):
            header.set_readout_width(atlas.text_width(self._font_readout, text) if text else 0.0)
