"""Top KPI ribbon: eight summary telemetry cards (TSD sections 4.1 and 6).

:func:`compute_kpis` is a pure function from a back-test (and optionally a
Monte Carlo ensemble) to eight ``(value, subtitle, tone)`` triples, so the
numbers on screen are unit-testable without building a widget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6 import QtCore, QtGui, QtWidgets

from cppi_lab.engine.controller import RunOutput
from cppi_lab.engine.monte_carlo import EnsembleResult
from cppi_lab.ui.popover_manager import HelpBadge

RIBBON_HEIGHT = 64
#: Cards narrower than this switch to short captions and a smaller value font.
COMPACT_BELOW_PX = 168
Tone = Literal["", "good", "warn", "bad"]


@dataclass(frozen=True)
class KpiSpec:
    key: str
    title: str
    popover: str
    #: Caption used when the card is too narrow for the full one.
    short: str = ""


KPI_SPECS: tuple[KpiSpec, ...] = (
    KpiSpec("equity", "NET EQUITY&nbsp; V<sub>T</sub>", "kpi-equity", "EQUITY"),
    KpiSpec("cushion", "CUSHION&nbsp; V<sub>T</sub> − F", "kpi-cushion", "CUSHION"),
    KpiSpec("floor", "FLOOR INTEGRITY", "kpi-floor", "FLOOR"),
    KpiSpec("holdings", "BTC HELD&nbsp; n<sub>T</sub>", "kpi-holdings", "BTC HELD"),
    KpiSpec("trigger", "TRIGGER&nbsp; P<sub>floor</sub>", "kpi-trigger", "TRIGGER"),
    KpiSpec("fills", "FILLS&nbsp; MAKER / TAKER", "kpi-fills", "FILLS M/T"),
    KpiSpec("missed", "MISSED MAKERS", "kpi-missed", "MISSED"),
    KpiSpec("costs", "FEES + SLIPPAGE", "kpi-costs", "COSTS"),
)


@dataclass(frozen=True)
class KpiValue:
    value: str
    subtitle: str
    tone: Tone = ""


def _usd(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


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
    if stats.floor_breaches > 0:
        status = KpiValue("VIOLATED", f"{stats.floor_breaches} bars below F on non-gap bars", "bad")
    elif stats.gapped:
        status = KpiValue("GAP BREACH", f"gap past trigger · {_usd(margin)} vs F", "bad")
    elif stats.liquidations > 0:
        status = KpiValue("LIQUIDATED", f"exit at trigger, V ≥ F · ×{stats.liquidations}", "warn")
    else:
        status = KpiValue("INTACT", f"no breach · F {_usd(floor)}", "good")
    if ensemble is not None and stats.floor_breaches == 0:
        tone: Tone = "bad" if ensemble.floor_breaches else status.tone
        status = KpiValue(
            status.value,
            f"MC {ensemble.floor_breaches} breaches · {ensemble.liquidated_members} liq.",
            tone,
        )

    attempts = stats.maker_attempts
    volume = stats.maker_volume_btc + stats.taker_volume_btc
    maker_share = stats.maker_volume_btc / volume * 100.0 if volume > 0 else 0.0
    miss_rate = stats.missed / attempts * 100.0 if attempts else 0.0
    cost_share = stats.total_costs_usd / v_start * 100.0 if v_start > 0 else 0.0
    buffer = float(last["safety_buf_usd"])

    return {
        "equity": KpiValue(_usd(v_end), f"{change:+.2f}% vs start", "good" if change >= 0 else ""),
        "cushion": KpiValue(
            _usd(max(0.0, margin)),
            f"min {_usd(stats.min_margin_usd)}",
            "warn" if margin < 0.01 * floor else "",
        ),
        "floor": status,
        "holdings": KpiValue(
            f"{float(last['btc_held']):.4f}", f"recoverable n_max {float(last['n_max']):.4f}"
        ),
        "trigger": KpiValue(
            _usd(float(last["p_floor"])),
            f"buffer +{_usd(buffer)} over P_liq",
            "warn" if buffer > 2.0 * params.p_liq * params.gamma / (1.0 - params.gamma) else "",
        ),
        "fills": KpiValue(
            f"{stats.maker_fills} / {stats.taker_fills}",
            f"maker {maker_share:.1f}% of {volume:.2f} BTC",
        ),
        "missed": KpiValue(
            f"{stats.missed}",
            f"miss rate {miss_rate:.1f}% of {attempts}" if attempts else "no maker attempts",
            "warn" if attempts and miss_rate > 60.0 else "",
        ),
        "costs": KpiValue(
            _usd(stats.total_costs_usd),
            f"{cost_share:.2f}% of start equity",
            "warn" if cost_share > 2.0 else "",
        ),
    }


class KpiCard(QtWidgets.QFrame):
    """One ribbon card: caption with ``?`` badge, value, and subtitle."""

    def __init__(self, spec: KpiSpec, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("kpiCard")
        self.title = QtWidgets.QLabel(spec.title, self)
        self.title.setObjectName("kpiTitle")
        self.title.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.badge = HelpBadge(spec.popover, self)
        self.value = QtWidgets.QLabel("-", self)
        self.value.setObjectName("kpiValue")
        self.subtitle = QtWidgets.QLabel("", self)
        self.subtitle.setObjectName("kpiSub")
        for label in (self.title, self.value, self.subtitle):
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
            label.setMinimumWidth(0)
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
            )
        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        head.addWidget(self.title, 1)
        head.addWidget(self.badge)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(9, 5, 7, 5)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.value)
        layout.addWidget(self.subtitle)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        compact = "true" if self.width() < COMPACT_BELOW_PX else "false"
        if self.property("compact") != compact:
            self.setProperty("compact", compact)
            self.title.setText(self.spec.short if compact == "true" else self.spec.title)
            for label in (self.value, self.subtitle):
                label.setProperty("compact", compact)
                label.style().unpolish(label)
                label.style().polish(label)

    def set_value(self, kpi: KpiValue) -> None:
        self.value.setText(kpi.value)
        self.subtitle.setText(kpi.subtitle)
        self.subtitle.setToolTip(kpi.subtitle)
        if self.value.property("tone") != kpi.tone:
            self.value.setProperty("tone", kpi.tone)
            self.value.style().unpolish(self.value)
            self.value.style().polish(self.value)


class KpiRibbon(QtWidgets.QWidget):
    """Fixed-height row of the eight KPI cards."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("kpiRibbon")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(RIBBON_HEIGHT)
        self.cards: dict[str, KpiCard] = {}
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 5, 6, 3)
        layout.setSpacing(5)
        for spec in KPI_SPECS:
            card = KpiCard(spec, self)
            self.cards[spec.key] = card
            layout.addWidget(card, 1)

    @property
    def badges(self) -> list[HelpBadge]:
        return [card.badge for card in self.cards.values()]

    def set_kpis(self, kpis: dict[str, KpiValue]) -> None:
        for key, kpi in kpis.items():
            self.cards[key].set_value(kpi)
