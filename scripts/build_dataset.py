"""Regenerate the bundled dataset ``cppi_lab/data/btc_usd_1m.parquet``.

Usage::

    uv run python scripts/build_dataset.py                       # default range
    uv run python scripts/build_dataset.py --start 2023-01 --end 2026-10

Source: the public, credential-free Binance market-data archive
(https://data.binance.vision), spot ``BTCUSDT`` 1-minute klines, one zip per
calendar month.  Every archive is verified against the SHA-256 checksum
published next to it.  The request carries nothing but the file path and a
generic client name.

The kline rows are converted with the same parser the application uses for
user imports (:func:`cppi_lab.data.loader.parse_binance_klines`), so the
taker buy volume is the exchange-reported field and the taker sell volume is
``volume - taker_buy_volume``.  No bar is invented: minutes the exchange did
not publish (maintenance windows) stay absent.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

from cppi_lab.data.loader import (
    BINANCE_KLINE_COLUMNS,
    BUNDLED_NAME,
    BUNDLED_PATH,
    BUNDLED_SOURCE,
    HTTP_USER_AGENT,
    from_frame,
    parse_binance_klines,
)

BASE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"
)
DEFAULT_START = "2023-01"
DEFAULT_END = "2026-10"
DEFAULT_SYMBOL = "BTCUSDT"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of ``YYYY-MM`` strings from ``start`` to ``end``."""
    sy, sm = (int(part) for part in start.split("-"))
    ey, em = (int(part) for part in end.split("-"))
    if (sy, sm) > (ey, em):
        raise ValueError("--start must not be after --end")
    months = []
    year, month = sy, sm
    while (year, month) <= (ey, em):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _get(url: str, attempts: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload: bytes = response.read()
                return payload
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def fetch_month(symbol: str, month: str, cache_dir: Path) -> Path:
    """Download one monthly archive (cached) and verify its SHA-256 checksum."""
    url = BASE_URL.format(symbol=symbol, month=month)
    target = cache_dir / f"{symbol}-1m-{month}.zip"
    expected = _get(url + ".CHECKSUM").decode("ascii").split()[0].lower()
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return target
    payload = _get(url)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {month}: expected {expected}, got {actual}")
    target.write_bytes(payload)
    return target


def read_month(archive: Path) -> pl.DataFrame:
    """Parse one monthly archive into a canonical-schema frame."""
    with zipfile.ZipFile(archive) as bundle:
        (member,) = [n for n in bundle.namelist() if n.endswith(".csv")]
        raw = bundle.read(member)
    has_header = not raw[:1].isdigit()
    frame = pl.read_csv(
        io.BytesIO(raw),
        has_header=has_header,
        new_columns=None if has_header else list(BINANCE_KLINE_COLUMNS),
        infer_schema_length=10_000,
    )
    return parse_binance_klines(frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default=DEFAULT_START, help="first month, YYYY-MM")
    parser.add_argument("--end", default=DEFAULT_END, help="last month, YYYY-MM (inclusive)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--out", type=Path, default=BUNDLED_PATH)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "cppi-lab" / "binance",
        help="where downloaded archives are kept between runs",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    months = month_range(args.start, args.end)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetching {len(months)} monthly archives of {args.symbol} into {args.cache_dir}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        archives = list(pool.map(lambda m: fetch_month(args.symbol, m, args.cache_dir), months))

    frames = [read_month(archive) for archive in archives]
    data = from_frame(pl.concat(frames), name=BUNDLED_NAME, source=BUNDLED_SOURCE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_frame().write_parquet(args.out, compression="zstd", compression_level=19)

    first = data.to_frame()["timestamp_utc"][0]
    last = data.to_frame()["timestamp_utc"][-1]
    size_mb = args.out.stat().st_size / 1e6
    print(f"rows            : {data.n:,}")
    print(f"first bar (UTC) : {first}")
    print(f"last bar (UTC)  : {last}")
    print(f"missing minutes : {data.missing_minutes:,}")
    print(f"close range     : {data.close.min():,.2f} .. {data.close.max():,.2f}")
    print(f"written         : {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
