"""Ledger table model: virtualisation, formatting, filters, sorting and export (FR-5.2)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PySide6 import QtCore

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketData
from cppi_lab.engine.kernel_numba import (
    FLAG_EXECUTED,
    FLAG_LIQUIDATION,
    FLAG_MISSED,
    TELEMETRY_FIELDS,
    SimResult,
)
from cppi_lab.ui.ledger_window import (
    COLUMNS,
    FILTER_ALL,
    FILTER_EXECUTED,
    FILTER_LIQUIDATIONS,
    FILTER_MISSED,
    FILTERS,
    TelemetryTableModel,
    event_label,
    format_cell,
)
from tests.conftest import backtest, ms, real_window

Role = QtCore.Qt.ItemDataRole
COLUMN = {name: i for i, (name, _, _) in enumerate(COLUMNS)}


@pytest.fixture(scope="module")
def result(bundled: MarketData, params: SimParams) -> SimResult:
    p = replace(params, start_ms=ms(2026, 2, 3))
    return backtest(real_window(bundled, p.start_ms, 6, p), p)[1]


@pytest.fixture
def model(result: SimResult) -> TelemetryTableModel:
    table = TelemetryTableModel()
    table.set_result(result)
    return table


def test_shape_and_headers_cover_every_telemetry_field(
    model: TelemetryTableModel, result: SimResult
) -> None:
    assert model.columnCount() == len(TELEMETRY_FIELDS) == 26
    assert model.rowCount() == result.n_bars
    horizontal = QtCore.Qt.Orientation.Horizontal
    headers = [model.headerData(i, horizontal) for i in range(26)]
    assert headers[0] == "t"
    assert headers[COLUMN["net_equity_usd"]] == "Net Equity $"
    assert len(set(headers)) == 26
    assert "ts_utc_ms" in model.headerData(1, horizontal, Role.ToolTipRole)
    assert model.rowCount(model.index(0, 0)) == 0  # flat table: no children
    assert not model.data(QtCore.QModelIndex())


def test_model_is_a_zero_copy_view(model: TelemetryTableModel, result: SimResult) -> None:
    assert np.shares_memory(model._columns[COLUMN["close"]], result.telemetry)
    assert (
        model.data(model.index(5, COLUMN["close"]), Role.UserRole) == result.telemetry["close"][5]
    )


def test_cell_formatting(model: TelemetryTableModel, result: SimResult) -> None:
    tel = result.telemetry
    assert model.data(model.index(0, COLUMN["ts_utc_ms"])) == "2026-02-03 00:00"
    assert model.data(model.index(0, COLUMN["close"])) == f"{tel['close'][0]:,.2f}"
    assert model.data(model.index(0, COLUMN["btc_held"])) == f"{tel['btc_held'][0]:.6f}"
    quiet = int(np.nonzero(result.flags == 0)[0][0])
    assert model.data(model.index(quiet, COLUMN["event_code"])) == ""
    assert model.data(model.index(quiet, COLUMN["exec_px_usd"])) == ""  # NaN -> blank
    assert model.data(model.index(quiet, COLUMN["p_fill_prob"])) == ""
    busy = int(np.nonzero(tel["event_code"] == 2)[0][0])
    assert model.data(model.index(busy, COLUMN["event_code"])).startswith("TAKER_SELL")
    assert model.data(model.index(busy, COLUMN["event_code"]), Role.ForegroundRole) is not None
    assert model.data(model.index(busy, 0), Role.BackgroundRole) is not None
    assert model.data(model.index(quiet, 0), Role.BackgroundRole) is None
    align = model.data(model.index(0, COLUMN["close"]), Role.TextAlignmentRole)
    assert align & int(QtCore.Qt.AlignmentFlag.AlignRight)


def test_format_cell_kinds() -> None:
    assert format_cell("usd", 1234567.891) == "1,234,567.89"
    assert format_cell("usd", float("nan")) == ""
    assert format_cell("int", 43199) == "43,199"
    assert format_cell("unit", 0.25) == "0.2500"
    assert format_cell("qty", 0.0) == ""
    assert format_cell("fee", 12.5) == "12.5000"
    assert format_cell("vol", 3.14159265) == "3.14159"
    assert format_cell("time", 1767225600000) == "2026-01-01 00:00"


def test_event_label_lists_every_event_of_the_bar() -> None:
    assert event_label(0, 0) == ""
    assert event_label(1, 1) == "MAKER_SELL"
    assert event_label(2, 2 | 16) == "TAKER_SELL +MISS_SELL"
    assert event_label(7, 64 | 4) == "HARD_BUFFER_LIQUIDATION +MAKER_BUY"


@pytest.mark.parametrize(
    ("name", "mask"),
    [
        (FILTER_EXECUTED, FLAG_EXECUTED),
        (FILTER_MISSED, FLAG_MISSED),
        (FILTER_LIQUIDATIONS, FLAG_LIQUIDATION),
    ],
)
def test_filters_select_exactly_the_flagged_bars(
    model: TelemetryTableModel, result: SimResult, name: str, mask: int
) -> None:
    model.set_filter(name)
    expected = np.nonzero(result.flags & mask)[0]
    assert expected.size > 0
    np.testing.assert_array_equal(model.view_rows, expected)
    assert model.rowCount() == expected.size
    assert model.filter_name == name
    model.set_filter(FILTER_ALL)
    assert model.rowCount() == result.n_bars


def test_filter_names_match_the_prd() -> None:
    assert FILTERS == (
        "All Timesteps",
        "Executed Trades",
        "Missed Maker Orders",
        "Floor Liquidations",
    )
    with pytest.raises(ValueError):
        TelemetryTableModel().set_filter("Everything")


def test_missed_filter_includes_misses_hidden_behind_a_taker_fill(
    model: TelemetryTableModel, result: SimResult
) -> None:
    """A bar whose primary event is a taker fill can still hold a missed maker order."""
    hidden = (result.flags & FLAG_MISSED != 0) & np.isin(result.telemetry["event_code"], (2, 4))
    assert hidden.any()
    model.set_filter(FILTER_MISSED)
    bar = int(np.nonzero(hidden)[0][0])
    row = model.row_of_bar(bar, nearest=False)
    assert row >= 0
    assert "+MISS" in model.data(model.index(row, COLUMN["event_code"]))


def test_sorting_ascending_descending_and_stable(
    model: TelemetryTableModel, result: SimResult
) -> None:
    close = result.telemetry["close"]
    model.sort(COLUMN["close"], QtCore.Qt.SortOrder.AscendingOrder)
    assert np.all(np.diff(close[model.view_rows]) >= 0)
    model.sort(COLUMN["close"], QtCore.Qt.SortOrder.DescendingOrder)
    assert np.all(np.diff(close[model.view_rows]) <= 0)
    assert model.rowCount() == result.n_bars
    # Sorting composes with a filter and keeps NaN ("not applicable") rows last.
    model.set_filter(FILTER_EXECUTED)
    for order in (QtCore.Qt.SortOrder.AscendingOrder, QtCore.Qt.SortOrder.DescendingOrder):
        model.sort(COLUMN["p_fill_prob"], order)
        values = result.telemetry["p_fill_prob"][model.view_rows]
        finite = np.isfinite(values)
        assert finite.any()
        assert not finite.all()
        assert not finite[int(np.argmin(finite)) :].any()
    model.sort(-1)
    np.testing.assert_array_equal(model.view_rows, np.nonzero(result.flags & FLAG_EXECUTED)[0])


def test_row_and_bar_mapping(model: TelemetryTableModel, result: SimResult) -> None:
    assert model.bar_of_row(17) == 17
    assert model.row_of_bar(17) == 17
    model.set_filter(FILTER_LIQUIDATIONS)
    bars = np.nonzero(result.flags & FLAG_LIQUIDATION)[0]
    assert model.bar_of_row(0) == bars[0]
    assert model.row_of_bar(int(bars[0])) == 0
    assert model.row_of_bar(int(bars[0]) + 1, nearest=False) == -1
    assert model.row_of_bar(int(bars[0]) + 1) == 0  # nearest shown bar
    assert TelemetryTableModel().row_of_bar(3) == -1


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_export_writes_the_rows_on_screen(
    model: TelemetryTableModel, result: SimResult, tmp_path: Path, suffix: str
) -> None:
    model.set_filter(FILTER_EXECUTED)
    model.sort(COLUMN["exec_qty_btc"], QtCore.Qt.SortOrder.DescendingOrder)
    target = tmp_path / f"ledger{suffix}"
    written = model.export(target)
    frame = pl.read_csv(target) if suffix == ".csv" else pl.read_parquet(target)
    assert written == frame.height == model.rowCount()
    assert frame.columns == [*TELEMETRY_FIELDS, "event_flags"]
    np.testing.assert_array_equal(frame["bar_idx"].to_numpy(), model.view_rows)
    np.testing.assert_allclose(
        frame["net_equity_usd"].to_numpy(), result.telemetry["net_equity_usd"][model.view_rows]
    )
    assert (frame["event_flags"].to_numpy() & FLAG_EXECUTED != 0).all()
    if suffix == ".parquet":
        assert frame.schema["bar_idx"] == pl.Int32
        assert frame.schema["event_code"] == pl.UInt8


def test_export_rejects_other_extensions(model: TelemetryTableModel, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        model.export(tmp_path / "ledger.xlsx")


def test_new_result_keeps_filter_and_sort(
    model: TelemetryTableModel, bundled: MarketData, params: SimParams
) -> None:
    model.set_filter(FILTER_MISSED)
    model.sort(COLUMN["close"], QtCore.Qt.SortOrder.DescendingOrder)
    p = replace(params, start_ms=ms(2026, 6, 1), rho=0.0)
    other = backtest(real_window(bundled, p.start_ms, 3, p), p)[1]
    model.set_result(other)
    assert model.filter_name == FILTER_MISSED
    assert model.rowCount() == int(np.count_nonzero(other.flags & FLAG_MISSED))
    assert np.all(np.diff(other.telemetry["close"][model.view_rows]) <= 0)
