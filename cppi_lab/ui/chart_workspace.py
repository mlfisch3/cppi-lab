"""Three-pane synchronised chart canvas (PRD FR-4, TSD section 4.1).

One ``GraphicsLayoutWidget`` holds three plots with stretch ratios
``55 : 25 : 20`` and a shared, locked time axis:

* pane 1 - price, floor levels, net equity and execution markers;
* pane 2 - BTC holdings inside the hysteresis deadband;
* pane 3 - decline and growth signals and the maker fill diagnostics.

Lines, steps and filled bands are :class:`~cppi_lab.ui.chart_gl.GLLayerItem`
layers; markers are pyqtgraph scatter items.  A thin header strip floats over
the top of each pane and carries the title, the ``?`` badge, a colour legend
and the crosshair read-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

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
from cppi_lab.ui.chart_gl import GLLayerItem
from cppi_lab.ui.popover_manager import HelpBadge

FloatArray = NDArray[np.float64]

PANE_STRETCH: tuple[int, int, int] = (55, 25, 20)
HEADER_HEIGHT = 20
AXIS_WIDTH = 66
#: Markers shrink once more than this many are inside the visible range.
MARKER_DENSITY_SOFT = 250
_STEM_PX = 16.0


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


class _UtcAxis(pg.DateAxisItem):
    """Date axis in UTC whose labels always carry enough calendar context."""

    def __init__(self) -> None:
        super().__init__(orientation="bottom", utcOffset=0)

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        labels = []
        for value in values:
            try:
                moment = datetime.fromtimestamp(value, UTC)
            except (OverflowError, OSError, ValueError):
                labels.append("")
                continue
            if spacing >= 28 * 86400:
                labels.append(moment.strftime("%b %Y"))
            elif spacing >= 86400 or (moment.hour == 0 and moment.minute == 0):
                labels.append(moment.strftime("%b %d").replace(" 0", " "))
            elif spacing >= 60:
                labels.append(moment.strftime("%H:%M"))
            else:
                labels.append(moment.strftime("%H:%M:%S"))
        return labels


class PaneHeader(QtWidgets.QWidget):
    """Title, ``?`` badge, legend and read-out strip floating over a pane."""

    def __init__(self, index: int, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.title = QtWidgets.QLabel(PANE_TITLES[index], self)
        self.title.setObjectName("paneTitle")
        self.badge = HelpBadge(PANE_POPOVERS[index], self)
        self.legend = QtWidgets.QLabel(_legend_html(PANE_LEGENDS[index]), self)
        self.legend.setObjectName("paneLegend")
        self.legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout = QtWidgets.QLabel("", self)
        self.readout.setObjectName("paneReadout")
        self.readout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(AXIS_WIDTH + 4, 0, 8, 0)
        layout.setSpacing(6)
        layout.addWidget(self.title)
        layout.addWidget(self.badge)
        layout.addSpacing(6)
        layout.addWidget(self.legend)
        layout.addStretch(1)
        layout.addWidget(self.readout)


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
        self._selected_bar = -1
        self._marker_scale = 1.0
        self._pane_heights: list[int] = []
        self._liquidation_lines: list[pg.InfiniteLine] = []
        self._liquidation_badges: list[pg.TextItem] = []

        self.canvas = pg.GraphicsLayoutWidget(self)
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

        self._build_layers()
        self._build_crosshair()

        self.headers = [PaneHeader(i, self.canvas) for i in range(3)]
        self.banner = EmptyStateBanner(self.canvas)
        self.banner.normalizeRequested.connect(self.normalizeRequested)
        self.banner.scenarioRequested.connect(self.scenarioRequested)

        self.canvas.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.canvas.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.plots[0].getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)
        for plot in self.plots:
            plot.getViewBox().sigResized.connect(self._schedule_layout)
        self.plots[0].getViewBox().sigYRangeChanged.connect(self._position_badges)
        self._layout_timer = QtCore.QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(0)
        self._layout_timer.timeout.connect(self._layout_overlays)
        self._schedule_layout()

    # -- construction --------------------------------------------------------------
    def _make_plot(self, row: int) -> pg.PlotItem:
        plot: pg.PlotItem = self.canvas.addPlot(row=row, col=0, axisItems={"bottom": _UtcAxis()})
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setTitle(None)
        plot.layout.setRowFixedHeight(0, HEADER_HEIGHT)
        plot.layout.setRowMinimumHeight(0, HEADER_HEIGHT)
        plot.showGrid(x=True, y=True, alpha=1.0)
        view = plot.getViewBox()
        view.setMouseEnabled(x=True, y=False)
        view.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        view.setAutoVisible(y=True)
        view.setDefaultPadding(0.04)
        view.setBackgroundColor(theme.CANVAS)
        tick_font = QtGui.QFont()
        tick_font.setPixelSize(10)
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(theme.GRID))
            axis.setTickPen(pg.mkPen(theme.GRID))
            axis.setTextPen(pg.mkPen(theme.TEXT_MUTED))
            axis.setStyle(tickFont=tick_font, tickLength=-4, tickTextOffset=4)
        plot.getAxis("left").setWidth(AXIS_WIDTH)
        if row < 2:
            plot.getAxis("bottom").setStyle(showValues=False)
            plot.getAxis("bottom").setHeight(4)
        else:
            plot.getAxis("bottom").setHeight(22)
        return plot

    def _layer(self, pane: int, *args: object, z: float = 0.0, **kwargs: object) -> GLLayerItem:
        item = GLLayerItem(*args, **kwargs)  # type: ignore[arg-type]
        item.setZValue(z)
        self.plots[pane].addItem(item, ignoreBounds=False)
        return item

    def _scatter(
        self,
        pane: int,
        symbol: str,
        size: float,
        colour: str,
        hollow: bool = False,
        z: float = 30.0,
    ) -> pg.ScatterPlotItem:
        if hollow:
            item = pg.ScatterPlotItem(
                symbol=symbol, size=size, pen=pg.mkPen(colour, width=2.0), brush=None, pxMode=True
            )
        else:
            item = pg.ScatterPlotItem(
                symbol=symbol,
                size=size,
                pen=pg.mkPen(theme.CANVAS, width=1.0),
                brush=pg.mkBrush(colour),
                pxMode=True,
            )
        item.setZValue(z)
        item.base_size = size
        self.plots[pane].addItem(item, ignoreBounds=True)
        return item

    def _build_layers(self) -> None:
        # Pane 1 - TSD visual-encoding matrix, back to front.
        self.mc_fan = self._layer(0, "band", theme.MC_FAN, z=1)
        self.price_band = self._layer(0, "band", theme.PRICE_BAND, z=2)
        self.floor_zone = self._layer(0, "band", theme.P_FLOOR_ZONE, z=3)
        self.p_liq = self._layer(0, "line", theme.P_LIQ, 1.5, (6.0, 4.0), z=4)
        self.p_floor = self._layer(0, "step", theme.P_FLOOR, 2.2, z=5)
        self.price = self._layer(0, "line", theme.PRICE, 1.6, z=6)
        self.equity_glow = self._layer(0, "line", theme.EQUITY_GLOW, 7.0, z=7)
        self.equity = self._layer(0, "line", theme.EQUITY, 2.4, z=8)
        self.taker_sell_stems = self._layer(
            0, "stems", theme.TAKER_SELL, 1.4, stem_px=_STEM_PX, z=20
        )
        self.markers = {
            EVENT_MISSED_MAKER_SELL: self._scatter(0, "o", 13, theme.MISSED, hollow=True, z=21),
            EVENT_MAKER_SELL: self._scatter(0, "t", 11, theme.MAKER_SELL, z=22),
            EVENT_MAKER_BUY: self._scatter(0, "t1", 11, theme.MAKER_BUY, z=23),
            EVENT_TAKER_BUY: self._scatter(0, "d", 12, theme.TAKER_BUY, z=24),
            EVENT_TAKER_SELL: self._scatter(0, "d", 12, theme.TAKER_SELL, z=25),
        }
        # Pane 2
        self.deadband = self._layer(1, "band", theme.DEADBAND, reduce="mean", z=1)
        self.n_max = self._layer(1, "line", theme.N_MAX, 1.2, reduce="mean", z=2)
        self.n_rebuy = self._layer(1, "line", theme.N_REBUY, 1.5, (6.0, 4.0), reduce="mean", z=3)
        self.n_ceil = self._layer(1, "line", theme.N_CEIL, 1.5, (6.0, 4.0), reduce="mean", z=4)
        self.n_held = self._layer(1, "step", theme.N_HELD, 2.5, z=5)
        # Pane 3
        self.s_down_fill = self._layer(2, "band", theme.S_DOWN_FILL, reduce="mean", z=1)
        self.s_up_fill = self._layer(2, "band", theme.S_UP_FILL, reduce="mean", z=2)
        self.s_down = self._layer(2, "line", theme.S_DOWN, 1.5, reduce="mean", z=3)
        self.s_up = self._layer(2, "line", theme.S_UP, 1.5, reduce="mean", z=4)
        self.fill_stems = self._layer(2, "segments", theme.FILL_PROB, 1.6, z=5)
        self.draw_hit = self._scatter(2, "o", 6, theme.FILL_PROB, z=6)
        self.draw_miss = self._scatter(2, "o", 7, theme.MISSED, hollow=True, z=7)
        self.plots[2].getViewBox().setYRange(-0.03, 1.05, padding=0)
        self.plots[2].getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)

    def _build_crosshair(self) -> None:
        hover_pen = pg.mkPen(theme.CROSSHAIR, width=1.0, style=QtCore.Qt.PenStyle.DotLine)
        select_pen = pg.mkPen(theme.SELECTION, width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        self.hover_lines: list[pg.InfiniteLine] = []
        self.select_lines: list[pg.InfiniteLine] = []
        for plot in self.plots:
            hover = pg.InfiniteLine(angle=90, movable=False, pen=hover_pen)
            select = pg.InfiniteLine(angle=90, movable=False, pen=select_pen)
            for line in (hover, select):
                line.setZValue(50)
                line.hide()
                plot.addItem(line, ignoreBounds=True)
            self.hover_lines.append(hover)
            self.select_lines.append(select)
        self.hover_h = pg.InfiniteLine(angle=0, movable=False, pen=hover_pen)
        self.hover_h.setZValue(50)
        self.hover_h.hide()
        self.plots[0].addItem(self.hover_h, ignoreBounds=True)

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

        close = np.ascontiguousarray(tel["close"])
        p_floor = np.ascontiguousarray(tel["p_floor"])
        equity = tel["net_equity_usd"] / params.n0
        liq = np.full(x.shape[0], params.p_liq)
        self.price_band.set_band(x, tel["low"], tel["high"])
        self.floor_zone.set_band(x, liq, p_floor)
        self.p_liq.set_polyline(x, liq)
        self.p_floor.set_polyline(x, p_floor)
        self.price.set_polyline(x, close)
        self.equity_glow.set_polyline(x, equity)
        self.equity.set_polyline(x, equity)

        self.deadband.set_band(x, tel["n_rebuy"], tel["n_ceil"])
        self.n_max.set_polyline(x, tel["n_max"])
        self.n_rebuy.set_polyline(x, tel["n_rebuy"])
        self.n_ceil.set_polyline(x, tel["n_ceil"])
        self.n_held.set_polyline(x, tel["btc_held"])

        zeros = np.zeros(x.shape[0])
        self.s_down_fill.set_band(x, zeros, tel["s_down"])
        self.s_up_fill.set_band(x, zeros, tel["s_up"])
        self.s_down.set_polyline(x, tel["s_down"])
        self.s_up.set_polyline(x, tel["s_up"])

        self._set_markers(run)
        self._set_liquidations(run)
        if not same_axis:
            self.mc_fan.clear()
            self.reset_view()
        self._update_banner(run)
        self._update_marker_density()
        self._refresh_readout()
        self._apply_selection()

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
        for code, item in self.markers.items():
            if code == EVENT_MISSED_MAKER_SELL:
                mask = (flags & (event_bit(code) | event_bit(EVENT_MISSED_MAKER_BUY))) != 0
                y = prev_close
            else:
                mask = (flags & event_bit(code)) != 0
                y = levels[code]
            item.setData(x=x[mask], y=y[mask])
        sell_mask = (flags & event_bit(EVENT_TAKER_SELL)) != 0
        self.taker_sell_stems.set_stems(x[sell_mask], levels[EVENT_TAKER_SELL][sell_mask])

        p_fill = tel["p_fill_prob"]
        attempt = np.isfinite(p_fill)
        ax, ap, au = x[attempt], p_fill[attempt], tel["u_rand_draw"][attempt]
        self.fill_stems.set_segments(ax, np.zeros(ax.shape[0]), ax, ap)
        hit = (au <= ap) & (ap > 0.0)
        self.draw_hit.setData(x=ax[hit], y=au[hit])
        self.draw_miss.setData(x=ax[~hit], y=au[~hit])

    def _set_liquidations(self, run: RunOutput) -> None:
        for plot, line in self._liquidation_lines_with_plots():
            plot.removeItem(line)
        for badge in self._liquidation_badges:
            self.plots[0].removeItem(badge)
        self._liquidation_lines = []
        self._liquidation_badges = []
        self._liquidation_owner: list[pg.PlotItem] = []
        bars = np.nonzero(run.result.flags & event_bit(EVENT_HARD_BUFFER_LIQUIDATION))[0]
        pen = pg.mkPen(theme.LIQUIDATION, width=1.6)
        for bar in bars[:200]:
            x = float(self._x[bar])
            for plot in self.plots:
                line = pg.InfiniteLine(pos=x, angle=90, movable=False, pen=pen)
                line.setZValue(40)
                plot.addItem(line, ignoreBounds=True)
                self._liquidation_lines.append(line)
                self._liquidation_owner.append(plot)
            gap = run.result.stats.gap_bar == int(bar)
            badge = pg.TextItem(
                text="GAP LIQUIDATION" if gap else "LIQUIDATION",
                color=theme.CANVAS,
                fill=pg.mkBrush(theme.LIQUIDATION),
                anchor=(0.0, 0.0),
            )
            font = QtGui.QFont()
            font.setPixelSize(9)
            font.setBold(True)
            badge.setFont(font)
            badge.setZValue(41)
            self.plots[0].addItem(badge, ignoreBounds=True)
            self._liquidation_badges.append(badge)
            badge.bar_x = x
        self._position_badges()

    def _liquidation_lines_with_plots(self) -> list[tuple[pg.PlotItem, pg.InfiniteLine]]:
        owners = getattr(self, "_liquidation_owner", [])
        return list(zip(owners, self._liquidation_lines, strict=False))

    def _position_badges(self) -> None:
        if not self._liquidation_badges:
            return
        (_, _), (_, y_max) = self.plots[0].getViewBox().viewRange()
        for badge in self._liquidation_badges:
            badge.setPos(badge.bar_x, y_max)

    def set_ensemble(self, ensemble: EnsembleResult | None) -> None:
        """Show or clear the Monte Carlo 5-95 % equity fan (FR-3.3)."""
        if ensemble is None or self._x.shape[0] == 0:
            self.mc_fan.clear()
            return
        idx = ensemble.sample_idx
        idx = idx[idx < self._x.shape[0]]
        x = self._x[idx]
        count = idx.shape[0]
        self.mc_fan.set_band(x, ensemble.p05[:count] / self._n0, ensemble.p95[:count] / self._n0)

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
        for plot in self.plots[:2]:
            plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

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
        self._update_marker_density()
        self._position_badges()

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
        for item in (*self.markers.values(), self.draw_hit, self.draw_miss):
            item.setSize(item.base_size * scale)
        self.taker_sell_stems.stem_px = _STEM_PX * scale
        self.taker_sell_stems.invalidate()
        self.fill_stems.width = 1.6 * max(0.6, scale)
        self.fill_stems.color.setAlphaF(0.55 + 0.45 * (scale - 0.45) / 0.55)
        self.fill_stems.invalidate()

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
        self._position_badges()

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
        self.set_hover_bar(self.bar_at(point.x()))
        if pane == 0:
            self.hover_h.setPos(point.y())
            self.hover_h.show()
        else:
            self.hover_h.hide()

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
            for line in self.hover_lines:
                line.hide()
            self.hover_h.hide()
        else:
            x = float(self._x[bar])
            for line in self.hover_lines:
                line.setPos(x)
                line.show()
        self._refresh_readout()
        self.barHovered.emit(bar)

    def set_selected_bar(self, bar: int) -> None:
        """Pin the selection marker on ``bar`` (driven by the ledger)."""
        self._selected_bar = bar if 0 <= bar < self._x.shape[0] else -1
        self._apply_selection()
        self._refresh_readout()

    @property
    def selected_bar(self) -> int:
        return self._selected_bar

    @property
    def hover_bar(self) -> int:
        return self._hover_bar

    def _apply_selection(self) -> None:
        if self._selected_bar >= self._x.shape[0]:
            self._selected_bar = -1
        for line in self.select_lines:
            if self._selected_bar < 0:
                line.hide()
            else:
                line.setPos(float(self._x[self._selected_bar]))
                line.show()

    def _refresh_readout(self) -> None:
        run = self._run
        bar = self._hover_bar if self._hover_bar >= 0 else self._selected_bar
        if run is None or bar < 0 or bar >= run.result.n_bars:
            for header in self.headers:
                header.readout.setText("")
            return
        row = run.result.telemetry[bar]
        stamp = QtCore.QDateTime.fromMSecsSinceEpoch(
            int(row["ts_utc_ms"]), QtCore.QTimeZone.utc()
        ).toString("yyyy-MM-dd HH:mm")
        self.headers[0].readout.setText(
            f"{stamp} UTC   C {_fmt_usd(row['close'])}   P_floor {_fmt_usd(row['p_floor'])}   "
            f"V/N₀ {_fmt_usd(row['net_equity_usd'] / self._n0)}"
        )
        self.headers[1].readout.setText(
            f"n {row['btc_held']:.4f}   ceil {row['n_ceil']:.4f}   floor {row['n_rebuy']:.4f}   "
            f"max {row['n_max']:.4f}   d {row['d_norm']:.3f}"
        )
        fill = row["p_fill_prob"]
        attempt = f"   P_fill {fill:.3f} vs U {row['u_rand_draw']:.3f}" if fill == fill else ""
        self.headers[2].readout.setText(
            f"S_down {row['s_down']:.3f}   S_up {row['s_up']:.3f}{attempt}"
        )
