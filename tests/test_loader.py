"""Market-data engine: bundled dataset, importers and the Coinbase fetcher (PRD FR-1)."""

from __future__ import annotations

import json
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cppi_lab.data.loader import (
    BUNDLED_PATH,
    CANONICAL_COLUMNS,
    COINBASE_MAX_CANDLES,
    MINUTE_MS,
    DataFormatError,
    FetchError,
    MarketData,
    SideConvention,
    SourceFormat,
    clv_split,
    detect_format,
    fetch_coinbase_candles,
    from_frame,
    load_file,
    save_parquet,
)
from tests.conftest import ms

T0 = ms(2026, 1, 1)


# -- bundled dataset ---------------------------------------------------------------


def test_bundled_file_has_the_prd_schema() -> None:
    schema = pl.read_parquet_schema(BUNDLED_PATH)
    assert tuple(schema) == CANONICAL_COLUMNS
    assert schema["timestamp_utc"] == pl.Datetime("ms", "UTC")
    assert all(schema[c] == pl.Float64 for c in CANONICAL_COLUMNS[1:])


def test_bundled_data_is_real_multi_year_one_minute_history(bundled: MarketData) -> None:
    assert bundled.n > 2_000_000
    assert bundled.start_ms <= ms(2023, 1, 1)
    assert bundled.end_ms >= ms(2026, 10, 31, 23, 59)
    steps = np.diff(bundled.ts_ms)
    assert steps.min() == MINUTE_MS  # strictly increasing 1-minute grid
    assert bundled.missing_minutes == int((steps // MINUTE_MS - 1).sum())
    assert bundled.missing_minutes < 0.001 * bundled.n
    assert not bundled.synthesized_flow
    # Covers the PRD default window, complete.
    window = bundled.window(ms(2026, 1, 1), 30)
    assert window.n == 43_200
    assert window.ts_ms[0] == ms(2026, 1, 1)
    # Known prints of the Binance BTCUSDT tape.
    first = bundled.index_at(ms(2026, 1, 1))
    assert bundled.open[first] == 87_648.21
    assert bundled.close.max() == 126_114.50
    # Real, two-sided taker flow rather than a synthetic split.
    total = bundled.buy_vol + bundled.sell_vol
    assert (bundled.buy_vol >= 0).all()
    assert (bundled.sell_vol >= 0).all()
    share = bundled.buy_vol[total > 0] / total[total > 0]
    assert 0.45 < share.mean() < 0.55
    assert share.std() > 0.1
    assert (bundled.high >= np.maximum(bundled.open, bundled.close)).all()
    assert (bundled.low <= np.minimum(bundled.open, bundled.close)).all()


def test_window_slicing_is_zero_copy_and_half_open(bundled: MarketData) -> None:
    window = bundled.window(ms(2026, 3, 1), 1)
    assert window.n == 1_440
    assert window.ts_ms[-1] == ms(2026, 3, 1, 23, 59)
    assert np.shares_memory(window.close, bundled.close)
    with pytest.raises(ValueError):
        bundled.window(bundled.end_ms + 10 * MINUTE_MS, 1)
    with pytest.raises(ValueError):
        bundled.window(bundled.start_ms, 0)


def test_rescaling_preserves_returns_and_volumes(bundled: MarketData) -> None:
    window = bundled.window(ms(2025, 10, 9), 3)
    factor = 68_000.0 / float(window.open[0])
    scaled = window.rescaled(factor)
    assert scaled.open[0] == pytest.approx(68_000.0)
    assert scaled.price_scale == pytest.approx(factor)
    np.testing.assert_allclose(
        scaled.close[1:] / scaled.close[:-1], window.close[1:] / window.close[:-1], rtol=1e-13
    )
    np.testing.assert_allclose(scaled.high / scaled.low, window.high / window.low, rtol=1e-13)
    assert scaled.buy_vol is window.buy_vol
    assert scaled.sell_vol is window.sell_vol
    np.testing.assert_array_equal(window.close, bundled.window(ms(2025, 10, 9), 3).close)
    with pytest.raises(ValueError):
        window.rescaled(0.0)


# -- helpers -------------------------------------------------------------------------


def canonical_frame(n: int = 5) -> pl.DataFrame:
    base = 70_000.0 + np.arange(n) * 10.0
    return pl.DataFrame(
        {
            "timestamp_utc": pl.Series(T0 + np.arange(n) * MINUTE_MS).cast(
                pl.Datetime("ms", "UTC")
            ),
            "open": base,
            "high": base + 30.0,
            "low": base - 20.0,
            "close": base + 5.0,
            "buy_taker_vol": np.linspace(1.0, 2.0, n),
            "sell_taker_vol": np.linspace(2.0, 1.0, n),
        }
    )


def kline_rows(n: int = 4, micro: bool = False) -> list[list[object]]:
    unit = 1000 if micro else 1
    rows: list[list[object]] = []
    for i in range(n):
        start = (T0 + i * MINUTE_MS) * unit
        o = 87_000.0 + i
        rows.append(
            [
                start,
                o,
                o + 9.5,
                o - 7.25,
                o + 1.0,
                10.0 + i,
                start + 59_999 * unit,
                870_000.0,
                100 + i,
                4.0 + i,
                348_000.0,
                0,
            ]
        )
    return rows


def write_rows(path: Path, rows: list[list[object]], header: list[str] | None = None) -> Path:
    lines = [",".join(header)] if header else []
    lines += [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# -- importers -----------------------------------------------------------------------


def test_canonical_parquet_round_trip(tmp_path: Path) -> None:
    data = from_frame(canonical_frame(), name="x", source="test")
    save_parquet(data, tmp_path / "x.parquet")
    loaded = load_file(tmp_path / "x.parquet")
    np.testing.assert_array_equal(loaded.ts_ms, data.ts_ms)
    for field in ("open", "high", "low", "close", "buy_vol", "sell_vol"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(data, field))
    assert not loaded.synthesized_flow
    assert pl.read_parquet_schema(tmp_path / "x.parquet")["timestamp_utc"] == pl.Datetime(
        "ms", "UTC"
    )


@pytest.mark.parametrize("micro", [False, True], ids=["milliseconds", "microseconds"])
def test_binance_klines_headerless(tmp_path: Path, micro: bool) -> None:
    data = load_file(write_rows(tmp_path / "BTCUSDT-1m.csv", kline_rows(micro=micro)))
    assert data.n == 4
    assert data.ts_ms[0] == T0
    assert data.ts_ms[1] - data.ts_ms[0] == MINUTE_MS
    np.testing.assert_allclose(data.buy_vol, [4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(data.sell_vol, [6.0, 6.0, 6.0, 6.0])  # volume - taker buy
    assert data.high[0] == 87_009.5
    assert data.low[0] == 86_992.75
    assert not data.synthesized_flow


def test_binance_klines_with_header(tmp_path: Path) -> None:
    header = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    data = load_file(write_rows(tmp_path / "k.csv", kline_rows(), header))
    np.testing.assert_allclose(data.sell_vol, 6.0)
    assert detect_format(header) is SourceFormat.BINANCE_KLINES


def test_binance_aggtrades_are_aggregated_by_taker_side(tmp_path: Path) -> None:
    rows = [
        # id, price, qty, first, last, time, is_buyer_maker, best
        [1, 100.0, 1.0, 1, 1, T0 + 1_000, "True", "True"],  # buyer is maker -> taker SELL
        [2, 103.0, 2.0, 2, 2, T0 + 20_000, "False", "True"],  # taker BUY
        [3, 99.0, 0.5, 3, 3, T0 + 59_000, "True", "True"],
        [4, 101.0, 4.0, 4, 4, T0 + 61_000, "False", "True"],
    ]
    data = load_file(write_rows(tmp_path / "BTCUSDT-aggTrades.csv", rows))
    assert data.n == 2
    assert (data.open[0], data.high[0], data.low[0], data.close[0]) == (100.0, 103.0, 99.0, 99.0)
    assert data.buy_vol.tolist() == [2.0, 4.0]
    assert data.sell_vol.tolist() == [1.5, 0.0]
    assert data.ts_ms.tolist() == [T0, T0 + MINUTE_MS]


def test_coinbase_trades_respect_the_side_convention(tmp_path: Path) -> None:
    path = tmp_path / "market_trades.csv"
    path.write_text(
        "trade_id,product_id,price,size,time,side\n"
        "1,BTC-USD,100.0,1.0,2026-01-01T00:00:05Z,BUY\n"
        "2,BTC-USD,102.0,3.0,2026-01-01T00:00:40.250Z,SELL\n"
        "3,BTC-USD,101.0,2.0,2026-01-01T00:01:10Z,SELL\n"
    )
    maker = load_file(path)  # Coinbase reports the maker side: BUY = resting bid was hit
    assert maker.sell_vol.tolist() == [1.0, 0.0]
    assert maker.buy_vol.tolist() == [3.0, 2.0]
    taker = load_file(path, side_convention=SideConvention.TAKER)
    assert taker.buy_vol.tolist() == [1.0, 0.0]
    assert taker.sell_vol.tolist() == [3.0, 2.0]
    assert maker.close.tolist() == [102.0, 101.0]
    assert not maker.synthesized_flow


def test_coinbase_candles_use_the_clv_fallback(tmp_path: Path) -> None:
    path = tmp_path / "candles.csv"
    path.write_text(
        "start,low,high,open,close,volume\n"
        "1767225600,90.0,110.0,100.0,105.0,8.0\n"
        "1767225660,100.0,100.0,100.0,100.0,2.0\n"
    )
    data = load_file(path)
    assert data.synthesized_flow
    assert data.ts_ms[0] == T0
    # V_sell = V (H - C) / (H - L) = 8 * 5 / 20 = 2 ; V_buy = 6
    assert data.sell_vol[0] == pytest.approx(2.0)
    assert data.buy_vol[0] == pytest.approx(6.0)
    # Degenerate range: H = L, so the whole volume counts as buy-side by the formula.
    assert data.sell_vol[1] == 0.0
    assert data.buy_vol[1] == 2.0


def test_headerless_coinbase_exchange_candles(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "c.csv", [[1767225600, 90.0, 110.0, 100.0, 105.0, 8.0]])
    data = load_file(path)
    assert (data.low[0], data.high[0], data.open[0], data.close[0]) == (90.0, 110.0, 100.0, 105.0)


def test_generic_ohlcv_with_iso_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "ohlcv.csv"
    path.write_text(
        "Timestamp,Open,High,Low,Close,Volume\n"
        "2026-01-01 00:01:00,10,12,9,11,5\n"
        "2026-01-01 00:00:00,10,11,9,10,4\n"
    )
    data = load_file(path)
    assert data.ts_ms.tolist() == [T0, T0 + MINUTE_MS]  # sorted
    assert data.synthesized_flow
    np.testing.assert_allclose(data.buy_vol + data.sell_vol, [4.0, 5.0])


def test_clv_split_formula() -> None:
    high, low = np.array([110.0, 10.0]), np.array([90.0, 10.0])
    close, volume = np.array([95.0, 10.0]), np.array([4.0, 3.0])
    buy, sell = clv_split(high, low, close, volume)
    np.testing.assert_allclose(sell, [3.0, 0.0])
    np.testing.assert_allclose(buy, [1.0, 3.0])


def test_duplicates_are_collapsed_and_rows_sorted() -> None:
    frame = pl.concat([canonical_frame(3), canonical_frame(3).reverse()])
    data = from_frame(frame, name="x", source="test")
    assert data.n == 3
    assert np.all(np.diff(data.ts_ms) > 0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda f: f.drop("close"), "missing canonical columns"),
        (lambda f: f.head(0), "empty"),
        (lambda f: f.with_columns(low=pl.col("high") + 1.0), "bracket"),
        (lambda f: f.with_columns(open=pl.lit(-1.0)), "positive"),
        (lambda f: f.with_columns(buy_taker_vol=pl.lit(-0.5)), "non-negative"),
        (lambda f: f.with_columns(close=pl.lit(float("nan"))), "non-finite"),
        (
            lambda f: f.with_columns(
                timestamp_utc=pl.col("timestamp_utc") + pl.duration(seconds=7)
            ),
            "whole UTC minutes",
        ),
    ],
)
def test_invalid_data_is_rejected(mutation: object, message: str) -> None:
    with pytest.raises(DataFormatError, match=message):
        from_frame(mutation(canonical_frame()), name="x", source="test")  # type: ignore[operator]


def test_unreadable_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataFormatError, match="not found"):
        load_file(tmp_path / "nope.csv")
    weird = tmp_path / "a.xlsx"
    weird.write_text("x")
    with pytest.raises(DataFormatError, match="unsupported file type"):
        load_file(weird)
    unknown = tmp_path / "u.csv"
    unknown.write_text("alpha,beta\n1,2\n")
    with pytest.raises(DataFormatError, match="unrecognised file layout"):
        load_file(unknown)
    empty = tmp_path / "e.csv"
    empty.write_text("")
    with pytest.raises(DataFormatError, match="empty"):
        load_file(empty)


# -- Coinbase REST fetcher (no network: the transport is injected) --------------------


class FakeCoinbase:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start, end = int(query["start"][0]), int(query["end"][0])
        candles = [
            {
                "start": str(s),
                "low": "99",
                "high": "103",
                "open": "100",
                "close": "102",
                "volume": "4",
            }
            for s in range(end, start - 1, -60)  # newest first, like the API
        ]
        return json.dumps({"candles": candles}).encode()


def test_fetcher_pages_through_the_range_without_credentials() -> None:
    fake = FakeCoinbase()
    seen: list[tuple[int, int]] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    data = fetch_coinbase_candles(
        start, end, transport=fake, progress=lambda a, b: seen.append((a, b)), pause_s=0.0
    )
    assert data.n == 720
    assert data.ts_ms[0] == T0
    assert data.ts_ms[-1] == T0 + 719 * MINUTE_MS
    assert data.synthesized_flow
    assert len(fake.urls) == 3  # ceil(720 / 350)
    assert seen == [(1, 3), (2, 3), (3, 3)]
    for url in fake.urls:
        parsed = urllib.parse.urlparse(url)
        assert parsed.netloc == "api.coinbase.com"
        assert parsed.path.endswith("/market/products/BTC-USD/candles")
        query = urllib.parse.parse_qs(parsed.query)
        assert set(query) == {"start", "end", "granularity"}  # nothing else is sent
        assert query["granularity"] == ["ONE_MINUTE"]
        assert (int(query["end"][0]) - int(query["start"][0])) // 60 + 1 <= COINBASE_MAX_CANDLES
    np.testing.assert_allclose(data.sell_vol, 1.0)  # 4 * (103 - 102) / (103 - 99)


def test_fetcher_reports_failures() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 1, tzinfo=UTC)
    with pytest.raises(FetchError, match="malformed response"):
        fetch_coinbase_candles(start, end, transport=lambda url: b"<html>", pause_s=0.0)
    with pytest.raises(FetchError, match="unexpected response"):
        fetch_coinbase_candles(start, end, transport=lambda url: b'{"error": "x"}', pause_s=0.0)
    with pytest.raises(FetchError, match="no candles"):
        fetch_coinbase_candles(start, end, transport=lambda url: b'{"candles": []}', pause_s=0.0)
    with pytest.raises(FetchError, match="cancelled"):
        fetch_coinbase_candles(start, end, transport=FakeCoinbase(), cancelled=lambda: True)
    with pytest.raises(ValueError):
        fetch_coinbase_candles(end, start, transport=FakeCoinbase())
