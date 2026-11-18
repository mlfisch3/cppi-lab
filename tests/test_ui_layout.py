"""Nothing is clipped: the layout at 1920x1080 and its deliberate degradation below.

PRD FR-4.1 asks for everything on one 1080p screen.  These tests pin the
stronger property the product is held to: at 1920x1080 no caption, value or
sub-line is shortened at all, and at smaller sizes whatever no longer fits is
replaced by a shorter wording or an ellipsis with a tool tip - never cut off
in the middle or covered by another control.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from itertools import pairwise

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from cppi_lab.config.defaults import default_params
from cppi_lab.data.loader import MarketData
from cppi_lab.engine.controller import SimulationController
from cppi_lab.ui.chart_workspace import ChartWorkspace
from cppi_lab.ui.control_sidebar import SCROLL_GUTTER, SIDEBAR_WIDTH
from cppi_lab.ui.kpi_ribbon import (
    COMPACT_FROM_PX,
    FULL_FROM_PX,
    KPI_SPECS,
    RIBBON_HEIGHT,
    KpiRibbon,
    KpiValue,
    compute_kpis,
)
from cppi_lab.ui.main_window import DATASET_MIN_WIDTH, SCENARIOS, MainWindow
from cppi_lab.ui.widgets import AdaptiveLabel, ElasticComboBox, ElidedLabel
from tests.conftest import ms

pytestmark = pytest.mark.ui


@pytest.fixture
def window(qtbot: QtBot, bundled: MarketData) -> Iterator[MainWindow]:
    win = MainWindow(bundled, use_web=False, async_ensemble=False)
    qtbot.addWidget(win)
    win.resize(1920, 1040)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def assert_rows_clear_of_the_scrollbar(window: MainWindow) -> None:
    """Every parameter row lies wholly inside the scroll viewport, beside the gutter."""
    sidebar = window.sidebar
    viewport = sidebar.scroll_area.viewport()
    assert viewport.width() <= SIDEBAR_WIDTH - SCROLL_GUTTER  # the gutter is always kept free
    assert sidebar.body.width() == viewport.width()  # no row is wider than the viewport
    assert sidebar.body.minimumSizeHint().width() <= viewport.width()
    for key, row in sidebar.rows.items():
        button = row.reset_button
        right = button.mapTo(sidebar.body, QtCore.QPoint(button.width(), 0)).x()
        assert right <= viewport.width(), key
        assert row.width() == viewport.width(), key
        holder = row.label.parentWidget()
        assert holder is not None
        widest = row.label.sizeHint().width() + row.symbol.sizeHint().width() + 5
        assert widest <= holder.width(), key  # caption and symbol fit


def layout_of(ribbon: KpiRibbon) -> str:
    """The ribbon's current layout (as a plain string, so comparisons do not narrow it)."""
    return str(ribbon.mode)


def top_bar_is_sound(window: MainWindow) -> None:
    chip = window.status_chip
    assert chip.text() in chip.wordings()
    assert not chip.is_elided()  # a whole wording, never a fragment
    assert window.dataset_combo.width() >= min(
        window.dataset_combo.sizeHint().width(), DATASET_MIN_WIDTH
    )


@pytest.mark.parametrize("scenario", [0, 1, 3, 5, 6])
def test_nothing_is_shortened_at_1920x1080(qtbot: QtBot, window: MainWindow, scenario: int) -> None:
    window.apply_scenario(SCENARIOS[scenario])
    qtbot.wait(40)
    assert window.kpis.mode == "full"
    assert window.kpis.card_width() >= FULL_FROM_PX
    for key, card in window.kpis.cards.items():
        assert card.title.text() == card.spec.title
        assert card.elided() == [], (key, card.subtitle.text())
    chip = window.status_chip
    assert chip.text() == chip.wordings()[0]  # every timing
    assert not chip.is_elided()
    assert window.dataset_combo.width() == window.dataset_combo.sizeHint().width()
    assert not window.sidebar.scrolls()
    assert_rows_clear_of_the_scrollbar(window)


def test_monte_carlo_subline_fits_at_1920x1080(qtbot: QtBot, window: MainWindow) -> None:
    window.apply_scenario(SCENARIOS[5])
    toggle = window.sidebar.rows["mc_fan"].toggle
    assert toggle is not None
    toggle.click()
    qtbot.waitUntil(lambda: window.ensemble is not None, timeout=20_000)
    card = window.kpis.cards["floor"]
    assert card.subtitle.text().startswith("MC ")
    assert card.elided() == []


@pytest.mark.parametrize(("width", "height"), [(1536, 824), (1366, 728), (1280, 680)])
def test_laptop_sizes_use_short_wordings_instead_of_clipping(
    qtbot: QtBot, window: MainWindow, width: int, height: int
) -> None:
    window.apply_scenario(SCENARIOS[3])  # five liquidations, long numbers
    window.resize(width, height)
    qtbot.wait(80)
    ribbon = window.kpis
    assert ribbon.height() == RIBBON_HEIGHT
    assert ribbon.mode == ("compact" if ribbon.card_width() >= COMPACT_FROM_PX else "grid")
    for key, card in ribbon.cards.items():
        assert card.title.text() == card.spec.short
        assert card.elided() == [], (key, ribbon.mode)
        assert card.spec.title in card.toolTip()  # the full caption is one hover away
        assert card.toolTip().endswith(compute_kpis(window.run)[key].subtitle)  # type: ignore[arg-type]
    top_bar_is_sound(window)
    assert window.dataset_combo.width() == window.dataset_combo.sizeHint().width()
    assert window.sidebar.scrolls()
    assert window.sidebar.scroll_area.verticalScrollBar().isEnabled()
    assert window.sidebar.reset_all_button.isVisible()
    assert_rows_clear_of_the_scrollbar(window)


def test_minimum_window_keeps_whole_words_in_the_top_bar(qtbot: QtBot, window: MainWindow) -> None:
    window.resize(window.minimumSize())
    qtbot.wait(80)
    assert window.width() == window.minimumWidth()
    top_bar_is_sound(window)
    assert window.status_chip.text() == window.status_chip.wordings()[-1]  # the bar count
    assert layout_of(window.kpis) == "grid"
    window.resize(1920, 1040)
    qtbot.wait(80)
    assert window.status_chip.text() == window.status_chip.wordings()[0]
    assert layout_of(window.kpis) == "full"


def test_status_wording_steps_down_with_the_window(qtbot: QtBot, window: MainWindow) -> None:
    chip = window.status_chip
    seen = []
    for width in (1920, 1700, 1536, 1440, 1366, 1280, 1180, 1100):
        window.resize(width, 900)
        qtbot.wait(60)
        top_bar_is_sound(window)
        seen.append(chip.wordings().index(chip.text()))
    assert seen == sorted(seen)  # only ever shorter as the window narrows
    assert seen[0] == 0
    assert seen[-1] == len(chip.wordings()) - 1
    assert len(set(seen)) >= 3  # it really goes through intermediate wordings


def test_rescale_chip_takes_its_room_from_the_readout(qtbot: QtBot, window: MainWindow) -> None:
    window.resize(1440, 900)
    qtbot.wait(60)

    def shared() -> int:
        return (
            window.dataset_combo.width()
            + window.scenario_combo.width()
            + window.status_chip.width()
        )

    before = shared()
    window.apply_scenario(SCENARIOS[5])  # rescaled: the amber chip appears
    qtbot.wait(60)
    assert window.norm_chip.isVisible()
    assert "RESCALED" in window.norm_chip.text()
    # The chip is shown whole; the selectors and the read-out make room for it.
    assert window.norm_chip.width() >= window.norm_chip.sizeHint().width()
    assert shared() <= before - window.norm_chip.sizeHint().width()
    top_bar_is_sound(window)
    assert window.dataset_combo.width() == window.dataset_combo.sizeHint().width()


# -- the ribbon on its own -----------------------------------------------------------------


def sample_kpis(bundled: MarketData) -> dict[str, KpiValue]:
    run = SimulationController(
        bundled, replace(default_params(), start_ms=ms(2026, 2, 3), duration_days=6)
    ).run()
    return compute_kpis(run)


def test_ribbon_goes_from_one_row_to_two_rows_of_four(qtbot: QtBot, bundled: MarketData) -> None:
    ribbon = KpiRibbon()
    qtbot.addWidget(ribbon)
    ribbon.set_kpis(sample_kpis(bundled))
    ribbon.resize(1452, RIBBON_HEIGHT)
    ribbon.show()
    qtbot.wait(20)
    cards = list(ribbon.cards.values())
    assert layout_of(ribbon) == "full"
    assert len({card.y() for card in cards}) == 1  # one row
    assert all(card.subtitle.isVisible() for card in cards)
    assert [card.title.text() for card in cards] == [spec.title for spec in KPI_SPECS]

    ribbon.resize(900, RIBBON_HEIGHT)
    qtbot.wait(20)
    assert layout_of(ribbon) == "compact"
    assert len({card.y() for card in cards}) == 1
    assert [card.title.text() for card in cards] == [spec.short for spec in KPI_SPECS]
    assert all(card.subtitle.isVisible() for card in cards)
    assert all(card.elided() == [] for card in cards)

    ribbon.resize(640, RIBBON_HEIGHT)
    qtbot.wait(20)
    assert layout_of(ribbon) == "grid"
    assert ribbon.height() == RIBBON_HEIGHT  # still 64 px
    rows = sorted({card.y() for card in cards})
    assert len(rows) == 2
    assert [card.y() for card in cards] == [rows[0]] * 4 + [rows[1]] * 4
    assert [card.x() for card in cards[:4]] == [card.x() for card in cards[4:]]
    for card in cards:
        assert not card.subtitle.isVisible()  # one line: caption and value
        assert card.badge.isVisible()
        assert card.geometry().bottom() < RIBBON_HEIGHT
        assert card.value.geometry().right() <= card.width()
        assert card.elided() == []
        assert card.toolTip().count("\n") == 2  # caption, value and sub-line

    ribbon.resize(330, RIBBON_HEIGHT)  # far too narrow: an ellipsis, and the tool tip
    qtbot.wait(20)
    squeezed = [card for card in cards if card.elided()]
    assert squeezed
    assert all(card.value.text() in card.toolTip() for card in squeezed)
    ribbon.resize(1452, RIBBON_HEIGHT)
    qtbot.wait(20)
    assert layout_of(ribbon) == "full"
    assert all(card.subtitle.isVisible() and card.elided() == [] for card in cards)


def test_value_colour_follows_the_tone(qtbot: QtBot, bundled: MarketData) -> None:
    ribbon = KpiRibbon()
    qtbot.addWidget(ribbon)
    ribbon.resize(1452, RIBBON_HEIGHT)
    ribbon.show()
    ribbon.set_kpis(sample_kpis(bundled))
    assert ribbon.cards["floor"].value.property("tone") == "warn"
    assert ribbon.cards["floor"].value.text() == "INTACT"
    ribbon.set_kpis({"floor": KpiValue("INTACT", "no breach · F $60,000.00", "good", "no breach")})
    assert ribbon.cards["floor"].value.property("tone") == "good"


# -- small widgets ----------------------------------------------------------------------------


def test_adaptive_label_shows_the_longest_wording_that_fits(qtbot: QtBot) -> None:
    label = AdaptiveLabel()
    qtbot.addWidget(label)
    wordings = ("one two three four five", "one two three", "one")
    label.set_wordings(*wordings)
    label.show()
    widths = [label.fontMetrics().horizontalAdvance(text) for text in wordings]
    label.resize(widths[0] + 4, 20)
    assert label.text() == wordings[0]
    assert label.toolTip() == wordings[0]
    label.resize(widths[0] - 1, 20)
    assert label.text() == wordings[1]
    label.resize(widths[1] - 1, 20)
    assert label.text() == wordings[2]
    assert not label.is_elided()
    label.resize(max(1, widths[2] - 4), 20)
    assert label.text() == wordings[2]  # nothing shorter is left: it is elided
    assert label.is_elided()
    label.resize(widths[0] + 4, 20)
    assert label.text() == wordings[0]


def test_elided_label_reports_when_it_is_shortened(qtbot: QtBot) -> None:
    label = ElidedLabel("a fairly long caption")
    qtbot.addWidget(label)
    label.show()
    label.resize(label.fontMetrics().horizontalAdvance(label.text()) + 2, 20)
    assert not label.is_elided()
    label.resize(40, 20)
    assert label.is_elided()
    assert label.text() == "a fairly long caption"  # the text itself is kept
    assert label.minimumSizeHint().width() == 0
    assert not label.grab().isNull()  # paints


def test_combo_box_paints_when_narrower_than_its_text(qtbot: QtBot) -> None:
    combo = ElasticComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Bundled BTC-USD 1m (Binance spot BTCUSDT)")
    combo.show()
    natural = combo.sizeHint().width()
    assert natural <= 340
    assert combo.minimumSizeHint().width() == 120
    combo.resize(140, combo.sizeHint().height())
    image = combo.grab().toImage()
    assert not image.isNull()
    combo.resize(natural, combo.sizeHint().height())
    assert not combo.grab().isNull()


# -- chart axis labels on a short pane ----------------------------------------------------------


def test_axis_labels_never_overlap_on_a_short_pane(qtbot: QtBot, bundled: MarketData) -> None:
    chart = ChartWorkspace()
    qtbot.addWidget(chart)
    run = SimulationController(
        bundled, replace(default_params(), start_ms=ms(2026, 6, 1), duration_days=30)
    ).run()
    for height in (900, 560, 420, 330):
        chart.resize(1000, height)
        chart.show()
        chart.set_run(run)
        qtbot.wait(40)
        scene, _ = chart.frames()
        runs = [text for op in scene.find("tick-y") for text in op.texts]  # type: ignore[union-attr]
        assert runs
        for index in range(3):
            pane = chart.pane(index)
            baselines = sorted(
                baseline
                for _, baseline, _, _ in runs
                if pane.top - 8 <= baseline <= pane.bottom + 8
            )
            assert baselines, (height, index)  # every pane keeps at least one label
            gaps = [b - a for a, b in pairwise(baselines)]
            assert all(gap >= 12.0 for gap in gaps), (height, index, gaps)
    # The labels that remain on the signal pane are round values counted from zero.
    pane = chart.pane(2)
    kept = [
        float(text) for _, baseline, text, _ in runs if pane.top - 8 <= baseline <= pane.bottom + 8
    ]
    assert 0.0 in kept
    assert len(kept) < 6  # the six majors 0, 0.2 ... 1.0 no longer all fit


def test_widget_types_used_for_the_cards(qtbot: QtBot) -> None:
    ribbon = KpiRibbon()
    qtbot.addWidget(ribbon)
    for card in ribbon.cards.values():
        assert isinstance(card.title, ElidedLabel)
        assert isinstance(card.value, ElidedLabel)
        assert isinstance(card.subtitle, ElidedLabel)
        assert isinstance(card, QtWidgets.QFrame)
