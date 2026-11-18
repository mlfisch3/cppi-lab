"""Historical market-data engine (PRD FR-1, TSD section 2.1).

Everything the simulator consumes is a :class:`MarketData` instance: seven
C-contiguous NumPy arrays on a strictly increasing 1-minute UTC timeline.
This module produces that object from

* the bundled Zstd Parquet file (``btc_usd_1m.parquet``),
* user files in the layouts named by FR-1.2 (Binance ``klines`` and
  ``aggTrades``, Coinbase candles and ``market_trades``) plus plain OHLCV, and
* the public Coinbase Advanced Trade candles endpoint.

Sources without a taker-side split get one synthesised from the Close
Location Value, exactly as TSD section 2.1 prescribes, and are flagged with
``synthesized_flow=True`` so the UI can say so.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Column names of the canonical on-disk schema (PRD FR-1.1).
CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "buy_taker_vol",
    "sell_taker_vol",
)

BUNDLED_PATH = Path(__file__).with_name("btc_usd_1m.parquet")
BUNDLED_NAME = "Bundled BTC-USD 1m (Binance spot BTCUSDT)"
BUNDLED_SOURCE = (
    "Binance spot BTCUSDT 1-minute klines from data.binance.vision; "
    "taker buy volume is the exchange-reported field, taker sell volume is "
    "total volume minus taker buy volume"
)

MINUTE_MS = 60_000
DAY_MS = 86_400_000

#: Column order of a Binance kline CSV (spot archives ship without a header).
BINANCE_KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

#: Column order of a Binance aggTrades CSV (spot archives ship without a header).
BINANCE_AGGTRADE_COLUMNS: tuple[str, ...] = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
)

COINBASE_CANDLES_URL = "https://api.coinbase.com/api/v3/brokerage/market/products/{product}/candles"
#: The public candles endpoint returns at most 350 candles per request.
COINBASE_MAX_CANDLES = 350
#: A generic client name; nothing about the machine or the user is sent.
HTTP_USER_AGENT = "cppi-lab"


class DataFormatError(ValueError):
    """Raised when a file or frame cannot be mapped onto the canonical schema."""


class SourceFormat(Enum):
    """File layouts understood by :func:`load_file`."""

    CANONICAL = "canonical"
    BINANCE_KLINES = "binance_klines"
    BINANCE_AGGTRADES = "binance_aggtrades"
    COINBASE_CANDLES = "coinbase_candles"
    COINBASE_TRADES = "coinbase_trades"
    GENERIC_OHLCV = "generic_ohlcv"


class SideConvention(Enum):
    """Meaning of the ``side`` column in a Coinbase trade file.

    Coinbase reports the *maker* order side on its trade feeds: a ``BUY``
    print means a resting bid was hit, so the aggressor was a seller.  Files
    produced by other tools sometimes store the taker side instead, so the
    importer lets the caller say which one a file uses.
    """

    MAKER = "maker"
    TAKER = "taker"


@dataclass(frozen=True)
class MarketWindow:
    """A contiguous slice of a :class:`MarketData` timeline (zero-copy views)."""

    ts_ms: IntArray
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    buy_vol: FloatArray
    sell_vol: FloatArray
    #: Multiplicative price factor applied by FR-1.3 normalisation (1.0 = raw).
    price_scale: float = 1.0

    @property
    def n(self) -> int:
        return int(self.ts_ms.shape[0])

    def rescaled(self, factor: float) -> MarketWindow:
        """Return a copy whose OHLC prices are multiplied by ``factor``.

        Volumes are in BTC and are left untouched, so percentage returns,
        volatility clustering and the taker-flow imbalance are preserved
        exactly (PRD FR-1.3).
        """
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"price scale must be positive and finite, got {factor!r}")
        return MarketWindow(
            ts_ms=self.ts_ms,
            open=np.ascontiguousarray(self.open * factor),
            high=np.ascontiguousarray(self.high * factor),
            low=np.ascontiguousarray(self.low * factor),
            close=np.ascontiguousarray(self.close * factor),
            buy_vol=self.buy_vol,
            sell_vol=self.sell_vol,
            price_scale=self.price_scale * factor,
        )


@dataclass(frozen=True)
class MarketData:
    """A full 1-minute dataset held as NumPy arrays."""

    name: str
    source: str
    ts_ms: IntArray
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    buy_vol: FloatArray
    sell_vol: FloatArray
    #: True when the taker split was synthesised from the Close Location Value.
    synthesized_flow: bool = False

    @property
    def n(self) -> int:
        return int(self.ts_ms.shape[0])

    @property
    def start_ms(self) -> int:
        return int(self.ts_ms[0])

    @property
    def end_ms(self) -> int:
        return int(self.ts_ms[-1])

    @property
    def missing_minutes(self) -> int:
        """Number of 1-minute slots absent between the first and last bar."""
        expected = (self.end_ms - self.start_ms) // MINUTE_MS + 1
        return int(expected - self.n)

    def index_at(self, ts_ms: int) -> int:
        """Index of the first bar whose timestamp is ``>= ts_ms``."""
        return int(np.searchsorted(self.ts_ms, ts_ms, side="left"))

    def window(self, start_ms: int, days: int) -> MarketWindow:
        """Bars with ``start_ms <= timestamp < start_ms + days`` (zero-copy)."""
        if days < 1:
            raise ValueError("window length must be at least one day")
        lo = self.index_at(start_ms)
        hi = self.index_at(start_ms + days * DAY_MS)
        if hi <= lo:
            raise ValueError("the requested window contains no bars")
        return MarketWindow(
            ts_ms=self.ts_ms[lo:hi],
            open=self.open[lo:hi],
            high=self.high[lo:hi],
            low=self.low[lo:hi],
            close=self.close[lo:hi],
            buy_vol=self.buy_vol[lo:hi],
            sell_vol=self.sell_vol[lo:hi],
        )

    def to_frame(self) -> pl.DataFrame:
        """The dataset as a canonical-schema Polars frame."""
        return pl.DataFrame(
            {
                "timestamp_utc": pl.Series(self.ts_ms).cast(pl.Datetime("ms", "UTC")),
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "buy_taker_vol": self.buy_vol,
                "sell_taker_vol": self.sell_vol,
            }
        )


# --------------------------------------------------------------------------
# Canonicalisation helpers
# --------------------------------------------------------------------------


def clv_split(
    high: FloatArray, low: FloatArray, close: FloatArray, volume: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Synthesise ``(buy, sell)`` taker volume from the Close Location Value.

    TSD section 2.1: ``V_sell = V_tot * (H - C) / max(H - L, 1e-8)`` and
    ``V_buy = V_tot - V_sell``.  The fraction is clipped to ``[0, 1]`` so a
    malformed bar whose close lies outside its own range cannot produce a
    negative volume.
    """
    frac = np.clip((high - close) / np.maximum(high - low, 1e-8), 0.0, 1.0)
    sell = volume * frac
    return volume - sell, sell


def _epoch_to_ms(values: pl.Series) -> pl.Series:
    """Convert integer or float epoch stamps of any common unit to milliseconds."""
    if values.len() == 0:
        return values.cast(pl.Int64)
    numeric = values.cast(pl.Float64)
    magnitude_raw = numeric.abs().max()
    magnitude = float(magnitude_raw) if isinstance(magnitude_raw, (int, float)) else 0.0
    if magnitude < 1e11:  # seconds (valid until the year 5138)
        return (numeric * 1000.0).round().cast(pl.Int64)
    if magnitude < 1e14:  # milliseconds
        return numeric.round().cast(pl.Int64)
    if magnitude < 1e17:  # microseconds
        return values.cast(pl.Int64) // 1_000
    return values.cast(pl.Int64) // 1_000_000  # nanoseconds


def _timestamps_to_ms(values: pl.Series) -> pl.Series:
    """Convert a timestamp column of any supported dtype to epoch milliseconds."""
    dtype = values.dtype
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            values = values.dt.replace_time_zone("UTC")
        return values.dt.epoch("ms")
    if dtype == pl.Date:
        return values.cast(pl.Datetime("ms")).dt.replace_time_zone("UTC").dt.epoch("ms")
    if dtype == pl.String:
        stripped = values.str.strip_chars()
        as_number = stripped.cast(pl.Float64, strict=False)
        if as_number.null_count() == 0:
            return _epoch_to_ms(as_number)
        try:
            parsed = stripped.str.to_datetime(time_zone="UTC", time_unit="ms")
        except pl.exceptions.PolarsError as exc:
            raise DataFormatError(f"cannot parse timestamps: {exc}") from exc
        return parsed.dt.epoch("ms")
    if dtype.is_numeric():
        return _epoch_to_ms(values)
    raise DataFormatError(f"unsupported timestamp dtype {dtype}")


def _float_column(frame: pl.DataFrame, name: str) -> pl.Series:
    try:
        return frame[name].cast(pl.Float64, strict=True)
    except pl.exceptions.PolarsError as exc:
        raise DataFormatError(f"column {name!r} is not numeric: {exc}") from exc


def from_frame(
    frame: pl.DataFrame,
    *,
    name: str,
    source: str,
    synthesized_flow: bool = False,
) -> MarketData:
    """Validate a canonical-schema frame and convert it to :class:`MarketData`.

    Rows are sorted by time and exact duplicate timestamps are collapsed to
    the last occurrence.  Rows with non-finite or non-positive prices, an
    inconsistent high/low envelope, or negative volumes are rejected with a
    :class:`DataFormatError` rather than silently repaired.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in frame.columns]
    if missing:
        raise DataFormatError(f"missing canonical columns: {', '.join(missing)}")
    if frame.height == 0:
        raise DataFormatError("the dataset is empty")

    canonical = pl.DataFrame(
        {
            "ts_ms": _timestamps_to_ms(frame["timestamp_utc"]),
            **{c: _float_column(frame, c) for c in CANONICAL_COLUMNS[1:]},
        }
    )
    if canonical.null_count().row(0) != (0,) * canonical.width:
        raise DataFormatError("the dataset contains null values")
    canonical = canonical.unique(subset="ts_ms", keep="last").sort("ts_ms")

    ts = np.ascontiguousarray(canonical["ts_ms"].to_numpy(), dtype=np.int64)
    arrays = {
        c: np.ascontiguousarray(canonical[c].to_numpy(), dtype=np.float64)
        for c in CANONICAL_COLUMNS[1:]
    }
    o, h, lo, c = arrays["open"], arrays["high"], arrays["low"], arrays["close"]
    bv, sv = arrays["buy_taker_vol"], arrays["sell_taker_vol"]

    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise DataFormatError("the dataset contains non-finite values")
    if (ts % MINUTE_MS != 0).any():
        raise DataFormatError("timestamps are not aligned to whole UTC minutes")
    if min(o.min(), h.min(), lo.min(), c.min()) <= 0.0:
        raise DataFormatError("prices must be strictly positive")
    if (h < np.maximum(o, c)).any() or (lo > np.minimum(o, c)).any():
        raise DataFormatError("high/low do not bracket open/close on every bar")
    if bv.min() < 0.0 or sv.min() < 0.0:
        raise DataFormatError("taker volumes must be non-negative")

    return MarketData(
        name=name,
        source=source,
        ts_ms=ts,
        open=o,
        high=h,
        low=lo,
        close=c,
        buy_vol=bv,
        sell_vol=sv,
        synthesized_flow=synthesized_flow,
    )


# --------------------------------------------------------------------------
# Format-specific parsers (each returns a canonical-schema frame)
# --------------------------------------------------------------------------


def _lower_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({c: c.strip().lower() for c in frame.columns})


def _canonical_frame(
    ts_ms: pl.Series,
    open_: pl.Series,
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    buy: pl.Series,
    sell: pl.Series,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp_utc": ts_ms.cast(pl.Int64).cast(pl.Datetime("ms", "UTC")),
            "open": open_.cast(pl.Float64),
            "high": high.cast(pl.Float64),
            "low": low.cast(pl.Float64),
            "close": close.cast(pl.Float64),
            "buy_taker_vol": buy.cast(pl.Float64),
            "sell_taker_vol": sell.cast(pl.Float64),
        }
    )


def parse_binance_klines(frame: pl.DataFrame) -> pl.DataFrame:
    """Map Binance klines onto the canonical schema using the real taker split."""
    frame = _lower_columns(frame)
    aliases = {
        "taker_buy_volume": "taker_buy_base_asset_volume",
        "taker_buy_base": "taker_buy_base_asset_volume",
        "taker_buy_base_volume": "taker_buy_base_asset_volume",
    }
    frame = frame.rename({k: v for k, v in aliases.items() if k in frame.columns})
    needed = ("open_time", "open", "high", "low", "close", "volume", "taker_buy_base_asset_volume")
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataFormatError(f"Binance klines file is missing: {', '.join(missing)}")
    volume = _float_column(frame, "volume")
    buy = _float_column(frame, "taker_buy_base_asset_volume")
    # volume - taker_buy carries float noise at the 1e-13 level; Binance
    # quotes BTC quantities to 8 decimals, so rounding there restores the
    # exact published value and can never produce a tiny negative number.
    sell = (volume - buy).round(8).clip(lower_bound=0.0)
    return _canonical_frame(
        _timestamps_to_ms(frame["open_time"]),
        frame["open"],
        frame["high"],
        frame["low"],
        frame["close"],
        buy,
        sell,
    )


def _aggregate_trades(
    ts_ms: pl.Series, price: pl.Series, size: pl.Series, taker_is_buyer: pl.Series
) -> pl.DataFrame:
    """Aggregate individual trades into canonical 1-minute bars."""
    trades = pl.DataFrame(
        {
            "ts_ms": ts_ms.cast(pl.Int64),
            "price": price.cast(pl.Float64),
            "size": size.cast(pl.Float64),
            "taker_is_buyer": taker_is_buyer.cast(pl.Boolean),
        }
    )
    if trades.null_count().row(0) != (0, 0, 0, 0):
        raise DataFormatError("trade file contains null values")
    bars = (
        trades.with_columns(minute=(pl.col("ts_ms") // MINUTE_MS) * MINUTE_MS)
        .sort("ts_ms", maintain_order=True)
        .group_by("minute", maintain_order=True)
        .agg(
            open=pl.col("price").first(),
            high=pl.col("price").max(),
            low=pl.col("price").min(),
            close=pl.col("price").last(),
            buy=pl.col("size").filter(pl.col("taker_is_buyer")).sum(),
            sell=pl.col("size").filter(~pl.col("taker_is_buyer")).sum(),
        )
        .sort("minute")
    )
    return _canonical_frame(
        bars["minute"],
        bars["open"],
        bars["high"],
        bars["low"],
        bars["close"],
        bars["buy"],
        bars["sell"],
    )


def _as_bool(values: pl.Series) -> pl.Series:
    if values.dtype == pl.Boolean:
        return values
    if values.dtype == pl.String:
        lowered = values.str.strip_chars().str.to_lowercase()
        known = lowered.is_in(["true", "false", "1", "0"])
        if not bool(known.all()):
            raise DataFormatError("cannot interpret a boolean column")
        return lowered.is_in(["true", "1"])
    return values.cast(pl.Int64) != 0


def parse_binance_aggtrades(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate Binance aggTrades into 1-minute bars.

    ``is_buyer_maker`` is true when the buyer rested on the book, i.e. the
    aggressor was a *seller*, so those quantities count as taker sell volume.
    """
    frame = _lower_columns(frame)
    aliases = {"timestamp": "transact_time", "time": "transact_time", "qty": "quantity"}
    frame = frame.rename(
        {k: v for k, v in aliases.items() if k in frame.columns and v not in frame.columns}
    )
    needed = ("price", "quantity", "transact_time", "is_buyer_maker")
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataFormatError(f"Binance aggTrades file is missing: {', '.join(missing)}")
    return _aggregate_trades(
        _timestamps_to_ms(frame["transact_time"]),
        frame["price"],
        frame["quantity"],
        ~_as_bool(frame["is_buyer_maker"]),
    )


def parse_coinbase_trades(
    frame: pl.DataFrame, side_convention: SideConvention = SideConvention.MAKER
) -> pl.DataFrame:
    """Aggregate Coinbase ``market_trades`` rows into 1-minute bars."""
    frame = _lower_columns(frame)
    needed = ("price", "size", "time", "side")
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataFormatError(f"Coinbase trades file is missing: {', '.join(missing)}")
    side = frame["side"].cast(pl.String).str.strip_chars().str.to_uppercase()
    if not bool(side.is_in(["BUY", "SELL"]).all()):
        raise DataFormatError("Coinbase trade side must be BUY or SELL")
    side_is_buy = side == "BUY"
    taker_is_buyer = ~side_is_buy if side_convention is SideConvention.MAKER else side_is_buy
    return _aggregate_trades(
        _timestamps_to_ms(frame["time"]), frame["price"], frame["size"], taker_is_buyer
    )


def _ohlcv_with_clv(frame: pl.DataFrame, time_column: str) -> pl.DataFrame:
    high = _float_column(frame, "high").to_numpy()
    low = _float_column(frame, "low").to_numpy()
    close = _float_column(frame, "close").to_numpy()
    volume = _float_column(frame, "volume").to_numpy()
    buy, sell = clv_split(high, low, close, volume)
    return _canonical_frame(
        _timestamps_to_ms(frame[time_column]),
        frame["open"],
        frame["high"],
        frame["low"],
        frame["close"],
        pl.Series(buy),
        pl.Series(sell),
    )


def parse_coinbase_candles(frame: pl.DataFrame) -> pl.DataFrame:
    """Map Coinbase candles (``start``/``time``, low, high, open, close, volume)."""
    frame = _lower_columns(frame)
    time_column = "start" if "start" in frame.columns else "time"
    needed = (time_column, "open", "high", "low", "close", "volume")
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataFormatError(f"Coinbase candles file is missing: {', '.join(missing)}")
    return _ohlcv_with_clv(frame, time_column)


_GENERIC_TIME_COLUMNS = (
    "timestamp_utc",
    "timestamp",
    "ts",
    "datetime",
    "date",
    "time",
    "open_time",
    "start",
)


def parse_generic_ohlcv(frame: pl.DataFrame) -> pl.DataFrame:
    """Map a plain OHLCV table, synthesising the taker split from the CLV."""
    frame = _lower_columns(frame)
    time_column = next((c for c in _GENERIC_TIME_COLUMNS if c in frame.columns), None)
    if time_column is None:
        raise DataFormatError("no timestamp column found in the OHLCV file")
    if "volume" not in frame.columns and "vol" in frame.columns:
        frame = frame.rename({"vol": "volume"})
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in frame.columns]
    if missing:
        raise DataFormatError(f"OHLCV file is missing: {', '.join(missing)}")
    return _ohlcv_with_clv(frame, time_column)


def detect_format(columns: list[str]) -> SourceFormat:
    """Infer the layout of a headed table from its column names."""
    names = {c.strip().lower() for c in columns}
    if set(CANONICAL_COLUMNS) <= names:
        return SourceFormat.CANONICAL
    if "is_buyer_maker" in names:
        return SourceFormat.BINANCE_AGGTRADES
    if names & {"taker_buy_base_asset_volume", "taker_buy_volume", "taker_buy_base"}:
        return SourceFormat.BINANCE_KLINES
    if {"price", "size", "side", "time"} <= names:
        return SourceFormat.COINBASE_TRADES
    if {"open", "high", "low", "close", "volume"} <= names:
        if "start" in names or ("time" in names and "timestamp" not in names):
            return SourceFormat.COINBASE_CANDLES
        return SourceFormat.GENERIC_OHLCV
    raise DataFormatError(
        "unrecognised file layout; expected the canonical CPPI-Lab schema, Binance "
        "klines/aggTrades, Coinbase candles/market_trades, or a plain OHLCV table"
    )


def _read_csv(path: Path) -> pl.DataFrame:
    """Read a CSV, assigning Binance column names when the file has no header."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline()
    if not first.strip():
        raise DataFormatError("the file is empty")
    separator = "\t" if first.count("\t") > first.count(",") else ","
    cells = [c.strip().strip('"') for c in first.rstrip("\r\n").split(separator)]
    headerless = all(_looks_like_value(c) for c in cells)
    if not headerless:
        return pl.read_csv(path, separator=separator, infer_schema_length=10_000)
    if len(cells) == len(BINANCE_KLINE_COLUMNS):
        names: tuple[str, ...] = BINANCE_KLINE_COLUMNS
    elif len(cells) in (len(BINANCE_AGGTRADE_COLUMNS), len(BINANCE_AGGTRADE_COLUMNS) - 1):
        names = BINANCE_AGGTRADE_COLUMNS[: len(cells)]
    elif len(cells) == 6:
        # Coinbase Exchange candle rows: time, low, high, open, close, volume.
        names = ("time", "low", "high", "open", "close", "volume")
    else:
        raise DataFormatError(
            f"headerless CSV with {len(cells)} columns is not a known Binance or Coinbase layout"
        )
    return pl.read_csv(
        path,
        separator=separator,
        has_header=False,
        new_columns=list(names),
        infer_schema_length=10_000,
    )


def _looks_like_value(cell: str) -> bool:
    if cell.lower() in ("true", "false", ""):
        return True
    try:
        float(cell)
    except ValueError:
        return False
    return True


def read_table(path: Path) -> pl.DataFrame:
    """Read a CSV, TSV or Parquet file into a Polars frame."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".parquet", ".pq"):
            return pl.read_parquet(path)
        if suffix in (".csv", ".tsv", ".txt"):
            return _read_csv(path)
    except pl.exceptions.PolarsError as exc:
        raise DataFormatError(f"cannot read {path.name}: {exc}") from exc
    raise DataFormatError(f"unsupported file type {suffix!r}; use .csv or .parquet")


def canonicalize(
    frame: pl.DataFrame,
    fmt: SourceFormat | None = None,
    side_convention: SideConvention = SideConvention.MAKER,
) -> tuple[pl.DataFrame, SourceFormat]:
    """Convert a raw table in any supported layout to the canonical schema."""
    resolved = fmt if fmt is not None else detect_format(frame.columns)
    if resolved is SourceFormat.CANONICAL:
        return _lower_columns(frame).select(CANONICAL_COLUMNS), resolved
    if resolved is SourceFormat.BINANCE_KLINES:
        return parse_binance_klines(frame), resolved
    if resolved is SourceFormat.BINANCE_AGGTRADES:
        return parse_binance_aggtrades(frame), resolved
    if resolved is SourceFormat.COINBASE_TRADES:
        return parse_coinbase_trades(frame, side_convention), resolved
    if resolved is SourceFormat.COINBASE_CANDLES:
        return parse_coinbase_candles(frame), resolved
    return parse_generic_ohlcv(frame), resolved


_SYNTHESIZED_FORMATS = frozenset({SourceFormat.COINBASE_CANDLES, SourceFormat.GENERIC_OHLCV})

_FORMAT_LABELS = {
    SourceFormat.CANONICAL: "CPPI-Lab canonical schema",
    SourceFormat.BINANCE_KLINES: "Binance klines (exchange-reported taker split)",
    SourceFormat.BINANCE_AGGTRADES: "Binance aggTrades aggregated to 1-minute bars",
    SourceFormat.COINBASE_TRADES: "Coinbase market_trades aggregated to 1-minute bars",
    SourceFormat.COINBASE_CANDLES: "Coinbase candles (taker split synthesised from CLV)",
    SourceFormat.GENERIC_OHLCV: "OHLCV candles (taker split synthesised from CLV)",
}


def load_file(
    path: str | Path,
    fmt: SourceFormat | None = None,
    side_convention: SideConvention = SideConvention.MAKER,
) -> MarketData:
    """Import a user CSV/Parquet file in any layout named by PRD FR-1.2."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DataFormatError(f"file not found: {file_path}")
    frame, resolved = canonicalize(read_table(file_path), fmt, side_convention)
    return from_frame(
        frame,
        name=file_path.name,
        source=f"{_FORMAT_LABELS[resolved]} imported from {file_path.name}",
        synthesized_flow=resolved in _SYNTHESIZED_FORMATS,
    )


def load_bundled(path: Path = BUNDLED_PATH) -> MarketData:
    """Load the dataset shipped with the application."""
    if not path.is_file():
        raise DataFormatError(
            f"bundled dataset not found at {path}; regenerate it with "
            "`uv run python scripts/build_dataset.py`"
        )
    return from_frame(pl.read_parquet(path), name=BUNDLED_NAME, source=BUNDLED_SOURCE)


def save_parquet(data: MarketData, path: str | Path) -> None:
    """Write a dataset as Zstd-compressed Parquet in the canonical schema."""
    data.to_frame().write_parquet(Path(path), compression="zstd", compression_level=19)


# --------------------------------------------------------------------------
# Coinbase Advanced Trade REST fetcher
# --------------------------------------------------------------------------

Transport = Callable[[str], bytes]
ProgressCallback = Callable[[int, int], None]


class FetchError(RuntimeError):
    """Raised when the Coinbase candles endpoint cannot be read."""


def http_get(url: str, timeout: float = 30.0) -> bytes:
    """Default transport: a plain credential-free HTTPS GET."""
    request = urllib.request.Request(
        url, headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: bytes = response.read()
            return payload
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"request failed: {exc}") from exc


def fetch_coinbase_candles(
    start: datetime,
    end: datetime,
    product: str = "BTC-USD",
    *,
    transport: Transport = http_get,
    progress: ProgressCallback | None = None,
    pause_s: float = 0.12,
    cancelled: Callable[[], bool] | None = None,
) -> MarketData:
    """Fetch 1-minute candles for ``[start, end)`` from the public Coinbase API.

    The Advanced Trade *market* endpoint needs no credentials.  It returns at
    most 350 candles per call, so the range is walked in 350-minute pages.
    Coinbase candles carry total volume only; the taker split is synthesised
    from the Close Location Value and the result is flagged accordingly.
    """
    start_s = int(_as_utc(start).timestamp()) // 60 * 60
    end_s = int(_as_utc(end).timestamp()) // 60 * 60
    if end_s <= start_s:
        raise ValueError("the fetch range is empty")

    page = COINBASE_MAX_CANDLES * 60
    page_starts = list(range(start_s, end_s, page))
    rows: dict[int, tuple[float, float, float, float, float]] = {}
    for done, page_start in enumerate(page_starts, start=1):
        if cancelled is not None and cancelled():
            raise FetchError("cancelled")
        page_end = min(page_start + page, end_s) - 60
        query = urllib.parse.urlencode(
            {"start": page_start, "end": page_end, "granularity": "ONE_MINUTE"}
        )
        url = COINBASE_CANDLES_URL.format(product=urllib.parse.quote(product)) + "?" + query
        try:
            payload = json.loads(transport(url))
        except json.JSONDecodeError as exc:
            raise FetchError(f"malformed response: {exc}") from exc
        candles = payload.get("candles") if isinstance(payload, dict) else None
        if not isinstance(candles, list):
            raise FetchError(f"unexpected response: {str(payload)[:200]}")
        for candle in candles:
            try:
                stamp = int(candle["start"])
                rows[stamp] = (
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                    float(candle["volume"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise FetchError(f"malformed candle: {candle!r}") from exc
        if progress is not None:
            progress(done, len(page_starts))
        if pause_s > 0.0 and done < len(page_starts):
            time.sleep(pause_s)

    stamps = sorted(s for s in rows if start_s <= s < end_s)
    if not stamps:
        raise FetchError("Coinbase returned no candles for the requested range")
    table = np.array([rows[s] for s in stamps], dtype=np.float64)
    buy, sell = clv_split(table[:, 1], table[:, 2], table[:, 3], table[:, 4])
    frame = _canonical_frame(
        pl.Series(np.array(stamps, dtype=np.int64) * 1000),
        pl.Series(table[:, 0]),
        pl.Series(table[:, 1]),
        pl.Series(table[:, 2]),
        pl.Series(table[:, 3]),
        pl.Series(buy),
        pl.Series(sell),
    )
    return from_frame(
        frame,
        name=f"Coinbase {product} 1m",
        source=(
            f"Coinbase Advanced Trade public candles for {product} "
            "(taker split synthesised from CLV)"
        ),
        synthesized_flow=True,
    )


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
