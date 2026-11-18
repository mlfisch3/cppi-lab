"""Top KPI ribbon: eight summary telemetry cards (TSD sections 4.1 and 6).

:func:`compute_kpis` is a pure function from a back-test (and optionally a
Monte Carlo ensemble) to eight :class:`KpiValue` records, so the numbers on
screen are unit-testable without building a widget.

The ribbon is 64 px high at every size and adapts what a card shows to the
width it gets:

* **full** - caption, value and a sub-line (the layout at 1920x1080, where
  nothing is shortened);
* **compact** - a short caption, a smaller value and a short sub-line;
* **grid** - when even that would not fit, the eight cards form two rows of
  four, each one line with caption and value.

Whatever does not fit after that is elided with an ellipsis, and the card's
tool tip always carries the complete caption, value and sub-line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6 import QtCore, QtGui, QtWidgets

from cppi_lab.engine.controller import RunOutput
from cppi_lab.engine.monte_carlo import EnsembleResult
from cppi_lab.ui.popover_manager import HelpBadge
from cppi_lab.ui.widgets import ElidedLabel

RIBBON_HEIGHT = 64
#: A card at least this wide shows its full caption and sub-line.
FULL_FROM_PX = 168
#: A card narrower than this no longer fits in one row of eight.
COMPACT_FROM_PX = 104
#: Longest sub-line of the full layout, in characters (fits a 1920x1080 window).
SUBTITLE_CHARS = 26
#: Longest sub-line of the compact layout.
SHORT_CHARS = 16
Tone = Literal["", "good", "warn", "bad"]
Mode = Literal["full", "compact", "grid"]


@dataclass(frozen=True)
class KpiSpec:
    key: str
    title: str
    popover: str
    #: Caption used when the card is too narrow for the full one.
    short: str


KPI_SPECS: tuple[KpiSpec, ...] = (
    KpiSpec("equity", "NET EQUITY  V_T", "kpi-equity", "EQUITY"),
    KpiSpec("cushion", "CUSHION  V_T − F", "kpi-cushion", "CUSHION"),
    KpiSpec("floor", "FLOOR INTEGRITY", "kpi-floor", "FLOOR"),
    KpiSpec("holdings", "BTC HELD  n_T", "kpi-holdings", "BTC HELD"),
    KpiSpec("trigger", "TRIGGER  P_floor", "kpi-trigger", "TRIGGER"),
    KpiSpec("fills", "FILLS  MAKER / TAKER", "kpi-fills", "FILLS M/T"),
    KpiSpec("missed", "MISSED MAKERS", "kpi-missed", "MISSED"),
    KpiSpec("costs", "FEES + SLIPPAGE", "kpi-costs", "COSTS"),
)


@dataclass(frozen=True)
class KpiValue:
    value: str
    #: Sub-line of the full layout, at most :data:`SUBTITLE_CHARS` characters.
    subtitle: str
    tone: Tone = ""
    #: Sub-line of the compact layout, at most :data:`SHORT_CHARS` characters.
    short: str = ""


def _usd(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _usd_tight(amount: float) -> str:
    """A dollar amount for a sub-line: cents while they matter, then K-less round units."""
    sign = "-" if amount < 0 else ""
    size = abs(amount)
    if size < 100_000.0:
        return f"{sign}${size:,.2f}"
    if size < 10_000_000.0:
        return f"{sign}${size:,.0f}"
    if size < 1e9:
        return f"{sign}${size / 1e6:,.2f}M"
    return f"{sign}${size / 1e9:,.2f}B"


def _btc_tight(quantity: float) -> str:
    """A BTC quantity for a sub-line."""
    if quantity < 100.0:
        return f"{quantity:.2f}"
    if quantity < 10_000.0:
        return f"{quantity:,.1f}"
    return f"{quantity / 1_000.0:,.1f}k"


def _first_fitting(limit: int, *candidates: str) -> str:
    """The first candidate of at most ``limit`` characters (the last one if none fits)."""
    return next((text for text in candidates if len(text) <= limit), candidates[-1])


def _floor_status(run: RunOutput, ensemble: EnsembleResult | None) -> KpiValue:
    """Floor Integrity: whether the floor held, how often the exit fired, and where we are now.

    The value answers the card's own question - INTACT, GAP BREACH or
    VIOLATED.  An emergency exit keeps the floor, so it does not change the
    value; the sub-line counts the exits and says whether the strategy has
    bought back in since, which is the state the account is in at the end.
    """
    stats = run.result.stats
    last = run.result.telemetry[-1]
    margin = float(last["net_equity_usd"]) - run.result.floor_value
    exits = stats.liquidations
    if stats.floor_breaches > 0:
        count = stats.floor_breaches
        return KpiValue(
            "VIOLATED",
            _first_fitting(
                SUBTITLE_CHARS, f"{count:,} non-gap bars below F", f"{count:,} bars below F"
            ),
            "bad",
            _first_fitting(SHORT_CHARS, f"{count:,} bars below F", f"{count:,} bars < F"),
        )
    if stats.gapped:
        gap = _usd_tight(margin)
        return KpiValue(
            "GAP BREACH",
            f"gapped: {gap} vs F",
            "bad",
            _first_fitting(SHORT_CHARS, f"{gap} vs F", gap),
        )
    if ensemble is not None:
        breaches, stopped = ensemble.floor_breaches, ensemble.liquidated_members
        return KpiValue(
            "INTACT" if not breaches else "MC BREACH",
            _first_fitting(
                SUBTITLE_CHARS,
                f"MC {breaches} breaches · {stopped} liq.",
                f"MC {breaches} br. · {stopped} liq.",
            ),
            "bad" if breaches else ("warn" if exits else "good"),
            _first_fitting(
                SHORT_CHARS, f"MC {breaches} br · {stopped} liq", f"MC {breaches}/{stopped}"
            ),
        )
    if exits > 0:
        holding = float(last["btc_held"]) > 0.0
        noun = "liquidation" if exits == 1 else "liquidations"
        now = "holds BTC" if holding else "in cash"
        return KpiValue(
            "INTACT",
            _first_fitting(SUBTITLE_CHARS, f"{exits:,} {noun}, {now}", f"{exits:,} liq., {now}"),
            "warn",
            _first_fitting(
                SHORT_CHARS,
                f"{exits:,} liq. · {'in BTC' if holding else 'in cash'}",
                f"{exits:,} liq.",
            ),
        )
    floor = _usd_tight(run.result.floor_value)
    return KpiValue("INTACT", f"no breach · F {floor}", "good", "no breach")


def compute_kpis(run: RunOutput, ensemble: EnsembleResult | None = None) -> dict[str, KpiValue]:
    """Summarise a back-test into the eight ribbon cards."""
    tel = run.result.telemetry
    stats = run.result.stats
    params = run.params
    floor = run.result.floor_value
    first, last = tel[0], tel[-1]
    v_start = float(first["net_equity_usd"])
    v_end = float(last["net_equity_usd"])
    change = (v_end / v_start - 1.0) * 100.0 if v_start > 0 else 0.0
    margin = v_end - floor

    attempts = stats.maker_attempts
    volume = stats.maker_volume_btc + stats.taker_volume_btc
    maker_share = stats.maker_volume_btc / volume * 100.0 if volume > 0 else 0.0
    miss_rate = stats.missed / attempts * 100.0 if attempts else 0.0
    cost_share = stats.total_costs_usd / v_start * 100.0 if v_start > 0 else 0.0
    buffer = float(last["safety_buf_usd"])
    n_max = float(last["n_max"])

    return {
        "equity": KpiValue(
            _usd(v_end),
            f"{change:+.2f}% vs start",
            "good" if change >= 0 else "",
            _first_fitting(SHORT_CHARS, f"{change:+.2f}% vs start", f"{change:+.2f}%"),
        ),
        "cushion": KpiValue(
            _usd(max(0.0, margin)),
            f"min {_usd_tight(stats.min_margin_usd)}",
            "warn" if margin < 0.01 * floor else "",
            f"min {_usd_tight(stats.min_margin_usd)}",
        ),
        "floor": _floor_status(run, ensemble),
        "holdings": KpiValue(
            f"{float(last['btc_held']):.4f}",
            f"max recoverable {n_max:.4f}",
            "",
            _first_fitting(SHORT_CHARS, f"n_max {n_max:.4f}", f"max {n_max:.2f}"),
        ),
        "trigger": KpiValue(
            _usd(float(last["p_floor"])),
            f"+{_usd_tight(buffer)} over P_liq",
            "warn" if buffer > 2.0 * params.p_liq * params.gamma / (1.0 - params.gamma) else "",
            _first_fitting(SHORT_CHARS, f"+{_usd_tight(buffer)} buffer", f"+{_usd_tight(buffer)}"),
        ),
        "fills": KpiValue(
            f"{stats.maker_fills:,} / {stats.taker_fills:,}",
            f"maker {maker_share:.1f}% of {_btc_tight(volume)} BTC",
            "",
            f"maker {maker_share:.1f}%",
        ),
        "missed": KpiValue(
            f"{stats.missed:,}",
            f"miss rate {miss_rate:.1f}% of {attempts:,}" if attempts else "no maker attempts",
            "warn" if attempts and miss_rate > 60.0 else "",
            _first_fitting(SHORT_CHARS, f"{miss_rate:.1f}% of {attempts:,}", f"{miss_rate:.1f}%")
            if attempts
            else "no attempts",
        ),
        "costs": KpiValue(
            _usd(stats.total_costs_usd),
            f"{cost_share:.2f}% of start equity",
            "warn" if cost_share > 2.0 else "",
            _first_fitting(SHORT_CHARS, f"{cost_share:.2f}% of equity", f"{cost_share:.2f}%"),
        ),
    }


class KpiCard(QtWidgets.QFrame):
    """One ribbon card: caption with ``?`` badge, value, and (room permitting) a sub-line."""

    def __init__(self, spec: KpiSpec, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.mode: Mode = "full"
        self._kpi = KpiValue("-", "")
        self.setObjectName("kpiCard")
        self.title = ElidedLabel(spec.title, self)
        self.title.setObjectName("kpiTitle")
        self.badge = HelpBadge(spec.popover, self)
        self.value = ElidedLabel("-", self)
        self.value.setObjectName("kpiValue")
        self.subtitle = ElidedLabel("", self)
        self.subtitle.setObjectName("kpiSub")
        for label in (self.title, self.value, self.subtitle):
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
            )
        self._layout = QtWidgets.QGridLayout(self)
        self._layout.setHorizontalSpacing(4)
        self._layout.setVerticalSpacing(0)
        self._arrange()

    def _arrange(self) -> None:
        """Place the four parts for the current mode."""
        layout = self._layout
        for widget in (self.title, self.badge, self.value, self.subtitle):
            layout.removeWidget(widget)
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        left = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        policy = QtWidgets.QSizePolicy.Policy
        if self.mode == "grid":
            # One line: the caption at its own width, the badge, the value in the rest.
            layout.setContentsMargins(7, 2, 6, 2)
            layout.addWidget(self.title, 0, 0)
            layout.addWidget(self.badge, 0, 1)
            layout.addWidget(self.value, 0, 2)
            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 0)
            layout.setColumnStretch(2, 1)
            self.title.setSizePolicy(policy.Maximum, policy.Fixed)
            self.value.setAlignment(right)
            self.subtitle.hide()
        else:
            layout.setContentsMargins(9, 5, 7, 5)
            layout.addWidget(self.title, 0, 0)
            layout.addWidget(self.badge, 0, 1)
            layout.addWidget(self.value, 1, 0, 1, 2)
            layout.addWidget(self.subtitle, 2, 0, 1, 2)
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 0)
            layout.setColumnStretch(2, 0)
            self.title.setSizePolicy(policy.Ignored, policy.Fixed)
            self.value.setAlignment(left)
            self.subtitle.show()
        for label in (self.title, self.value, self.subtitle):
            if label.property("layout") != self.mode:
                label.setProperty("layout", self.mode)
                label.style().unpolish(label)
                label.style().polish(label)
        self._show()

    def set_mode(self, mode: Mode) -> None:
        if mode != self.mode:
            self.mode = mode
            self._arrange()

    def set_value(self, kpi: KpiValue) -> None:
        self._kpi = kpi
        if self.value.property("tone") != kpi.tone:
            self.value.setProperty("tone", kpi.tone)
            self.value.style().unpolish(self.value)
            self.value.style().polish(self.value)
        self._show()

    def _show(self) -> None:
        kpi = self._kpi
        full = self.mode == "full"
        self.title.setText(self.spec.title if full else self.spec.short)
        self.value.setText(kpi.value)
        self.subtitle.setText(kpi.subtitle if full else (kpi.short or kpi.subtitle))
        # The tool tip always has everything, whatever the card had room for.
        tip = f"{self.spec.title}\n{kpi.value}" + (f"\n{kpi.subtitle}" if kpi.subtitle else "")
        for widget in (self, self.title, self.value, self.subtitle):
            widget.setToolTip(tip)

    def elided(self) -> list[str]:
        """Names of the visible parts that do not fit and are shown with an ellipsis."""
        parts = {"title": self.title, "value": self.value, "subtitle": self.subtitle}
        return [name for name, label in parts.items() if label.isVisible() and label.is_elided()]


class KpiRibbon(QtWidgets.QWidget):
    """Fixed-height strip of the eight KPI cards: one row, or two rows of four."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("kpiRibbon")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(RIBBON_HEIGHT)
        self.mode: Mode = "full"
        self.cards: dict[str, KpiCard] = {}
        self._grid = QtWidgets.QGridLayout(self)
        for spec in KPI_SPECS:
            self.cards[spec.key] = KpiCard(spec, self)
        self._arrange()

    @property
    def badges(self) -> list[HelpBadge]:
        return [card.badge for card in self.cards.values()]

    def card_width(self) -> float:
        """Width each card gets when all eight share one row."""
        margins = self._grid.contentsMargins()
        spacing = self._grid.horizontalSpacing()
        count = len(self.cards)
        free = self.width() - margins.left() - margins.right() - (count - 1) * spacing
        return free / count

    def _arrange(self) -> None:
        grid = self.mode == "grid"
        for card in self.cards.values():
            self._grid.removeWidget(card)
        self._grid.setContentsMargins(6, 4 if grid else 5, 6, 3)
        self._grid.setHorizontalSpacing(5)
        self._grid.setVerticalSpacing(3)
        columns = 4 if grid else len(self.cards)
        for index, card in enumerate(self.cards.values()):
            card.set_mode(self.mode)
            self._grid.addWidget(card, index // columns, index % columns)
        for column in range(len(self.cards)):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width = self.card_width()
        mode: Mode = (
            "full" if width >= FULL_FROM_PX else "compact" if width >= COMPACT_FROM_PX else "grid"
        )
        if mode != self.mode:
            self.mode = mode
            self._arrange()

    def set_kpis(self, kpis: dict[str, KpiValue]) -> None:
        for key, kpi in kpis.items():
            self.cards[key].set_value(kpi)
