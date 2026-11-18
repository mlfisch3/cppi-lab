# CPPI-Lab

CPPI-Lab is a desktop workbench for back-testing and teaching a signal-modulated Bitcoin CPPI strategy with a hysteresis deadband.
It replays real 1-minute BTC history through a strategy that promises never to let the account fall below a wealth floor, and shows on one screen what that promise costs.

The product is specified by two documents in this repository: [`docs/spec/PRD.md`](docs/spec/PRD.md) and [`docs/spec/TSD.md`](docs/spec/TSD.md).
How every requirement was implemented and verified is tabulated in [`docs/REQUIREMENTS_TRACE.md`](docs/REQUIREMENTS_TRACE.md).
Every place where the two documents were ambiguous or contradicted each other, and how it was resolved, is recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Quick start

The project uses [uv](https://docs.astral.sh/uv/) and an isolated environment with a committed lockfile.
Nothing is installed into a system Python.

```bash
uv sync            # create .venv from uv.lock (Python 3.12)
uv run cppi-lab    # start the application
```

`uv run cppi-lab` is the one command that starts the app.
The first launch compiles the simulation kernels, which takes about ten seconds; the result is cached and later launches open in about three seconds.

Options:

| Option | Meaning |
| :--- | :--- |
| `--dataset FILE` | Open a CSV or Parquet file instead of the bundled dataset. |
| `--size 1920x1080` | Initial window size. By default the window is maximised on screens up to 1080p and 1920x1040 on larger ones. |
| `--no-opengl` | Draw the charts with the software rasteriser. This is slow and only meant for machines without OpenGL. |

## What you see

- **Left:** every parameter of PRD section 5, each with a slider, a numeric field, a `?` help badge and a reset button, plus *Reset All Parameters to Defaults*.
- **Top:** eight KPI cards (net equity, cushion, floor integrity, BTC held, liquidation trigger, fills, missed maker orders, fees and slippage).
- **Centre:** three time-locked panes: price with the floor levels, equity and execution markers; BTC holdings inside the hysteresis deadband; the two signals with the fill-probability diagnostics.
- **Ledger** (toolbar button or `Ctrl+L`): the full per-bar telemetry in a floating, always-on-top window with filters, sorting and CSV/Parquet export.
  Selecting a row moves the chart's selection line and clicking the chart selects the row.
- **Documentation** (toolbar button or `F1`): two cross-linked volumes, strategy theory and technical implementation, side by side with a live counterpart preview.
- **`?` badges:** hover for 150 ms to open a formatted explanation with rendered equations and links into both documentation volumes; the card can be pinned and moved.

Both secondary windows open floating.
Drag their title bar over the main window to dock them left, right or bottom, or use the small menu next to each toolbar button, which does the same without a drag.

With the PRD's default floor (60,000 USD) and reference (72,000 USD) the default window, January 2026, contains no orders: Bitcoin traded between about 75,700 and 97,900 USD that month, always above the reference, so the strategy correctly holds its whole allocation.
The chart says so and offers two one-click ways to see the machinery work: rescale the window to start near the floor, or load one of the scenarios in the top bar (several are 2026 periods that test the default floor at real prices).

Chart navigation: drag to pan, scroll to zoom, double-click to zoom out, click to select a bar, `Ctrl+0` to reset the view.

## Data

### Bundled dataset

`cppi_lab/data/btc_usd_1m.parquet` holds 2,015,920 real one-minute bars of the Binance spot pair **BTCUSDT** from 2023-01-01 to 2026-10-31 (44.5 MB, Zstd Parquet).
Columns: `timestamp_utc`, `open`, `high`, `low`, `close`, `buy_taker_vol`, `sell_taker_vol`.
The taker buy volume is the exchange-reported field and the taker sell volume is total volume minus taker buy volume; no value is synthetic.
Eighty minutes during which the exchange was down are absent rather than filled in.

The quote currency is the dollar stablecoin USDT, used here as the BTC-USD proxy.
The reasons for this choice, and for the date range, are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

### Regenerating it

```bash
uv run python scripts/build_dataset.py                           # the committed range
uv run python scripts/build_dataset.py --start 2020-01 --end 2026-10 --out /tmp/btc_long.parquet
```

The script downloads the monthly kline archives from the public, credential-free Binance market-data archive (`https://data.binance.vision`), verifies each against its published SHA-256 checksum, and writes the Parquet file.
The requests carry only the file path and the client name `cppi-lab`.
A longer file can be opened with `uv run cppi-lab --dataset /tmp/btc_long.parquet`.

### Importing your own data

*Import CSV / Parquet…* accepts the CPPI-Lab schema above, Binance `klines` and `aggTrades`, Coinbase `market_trades` and candles, and plain OHLCV tables.
The layout is detected from the column names (or from the column count for headerless Binance files).
Sources without a taker-side split get one synthesised from the close location value, as TSD section 2.1 prescribes, and the dataset selector's tooltip says so.

*Coinbase Sync…* downloads BTC-USD 1-minute candles for a date range from the public Coinbase Advanced Trade market endpoint.
No account or key is needed.

## Tests, lint and type checks

```bash
uv run pytest                     # 397 tests, about one minute
uv run ruff check .               # lint
uv run ruff format --check .      # formatting
uv run mypy                       # strict type check of the package, scripts and tests
```

The suite runs without a display: Qt is forced onto its `offscreen` platform.
It covers the signal math, the simulation kernel against an independent plain-Python reference (all 26 telemetry fields bit for bit), the floor invariant as a property over random markets, seeds and parameters, the routing law, the fill model, PRNG determinism, the Monte Carlo ensemble, the loader and importers, the table model, and the user interface including the QtWebEngine documentation viewer.

## Benchmarks

```bash
uv run python benchmarks/bench_latency.py      # headless
uv run python benchmarks/bench_fps.py          # needs a display and OpenGL
```

Measured on the development machine (Intel i9-12900H, 16 logical CPUs, NVIDIA RTX 3080 Ti Laptop, Linux under WSL2 with WSLg) for a 43,200-bar window:

| PRD budget | Requirement | Measured | Result |
| :--- | :--- | :--- | :--- |
| Single-trajectory kernel | ≤ 15 ms | 1.0 to 1.6 ms median, 3.7 ms worst | met |
| 100-seed Monte Carlo ensemble | ≤ 120 ms | 12 to 24 ms kernel only; 51 to 71 ms median including the 4.32 million draws and the percentiles (105 ms worst) | met |
| Parameter change to refreshed plots, cards and table | < 50 ms | 10 to 18 ms median, 23 ms worst | met |
| Chart pan, zoom and crosshair | 60 FPS | 40 to 64 FPS at the median frame, 32 to 55 FPS at the 95th percentile | **not met** |

The frame-rate requirement is missed on this machine.
A frame costs 4 to 16 ms to paint and 16 to 25 ms end to end at the median (18 to 31 ms at the 95th percentile), where 16.7 ms would be needed.
The cost no longer depends on the length of the back-test (a 365-day window is as fast as a 30-day one), but the per-frame work of the axis, marker and layer items plus the WSLg presentation path leaves it above one refresh interval.
See [`docs/DECISIONS.md`](docs/DECISIONS.md), decision 27, for what was measured and done.

## Platform support

The application was built and verified on **Linux under WSL2 with WSLg** only.
The code uses no platform-specific API and Windows 11 and macOS are named targets of the PRD, but **neither has been run or tested**, so treat them as unverified.

Notes for Linux:

- **Docking by drag and drop needs X11.**
  Qt Advanced Docking System relies on global cursor positions and window placement, which a Wayland session does not provide.
  The app selects X11 automatically when it is usable, and for that Qt needs the system package `libxcb-cursor0` (`sudo apt install libxcb-cursor0`).
  Without it the app runs on Wayland: everything works, floating windows can be moved and resized, and docking is available from the toolbar menus, but the windows cannot be dropped onto the main window with the mouse and are placed by the compositor.
- **Under WSL the documentation is rendered in software.**
  WSLg's graphics stack lacks an extension QtWebEngine's GPU compositor needs, so the app passes `--disable-gpu` to QtWebEngine there.
  The charts still use OpenGL.
  Set `QTWEBENGINE_CHROMIUM_FLAGS` yourself to override this.
- `QT_QPA_PLATFORM` overrides the platform choice, and `CPPI_LAB_OPENGL=0` is equivalent to `--no-opengl`.

## Repository layout

```
cppi_lab/
  main.py                    entry point (uv run cppi-lab)
  config/defaults.py         every parameter: default, range, step, metadata
  config/doc_registry.json   documentation spine and the 37 "?" popover texts
  data/loader.py             dataset loader, importers, Coinbase fetcher
  data/btc_usd_1m.parquet    bundled dataset
  engine/signals.py          decline and growth signals
  engine/kernel_numba.py     CPPI state machine, rho router, fill model
  engine/monte_carlo.py      parallel 100-seed ensemble
  engine/controller.py       cached recalculation pipeline (no Qt)
  ui/                        main window, sidebar, KPI ribbon, chart, ledger, popovers, docs viewer
  docs/                      theory.html, tech.html and the bundled KaTeX
tests/                       pytest suite and the plain-Python reference machine
benchmarks/                  latency and frame-rate benchmarks
scripts/build_dataset.py     regenerates the bundled dataset
scripts/ui_tour.py           drives the real app through its states for screenshots
docs/spec/                   PRD and TSD
docs/DECISIONS.md            resolutions of ambiguities and substitutions
docs/REQUIREMENTS_TRACE.md   requirement-by-requirement traceability
```

## Scope

CPPI-Lab is a back-testing and teaching tool.
It places no orders, connects to no exchange account and needs no credential.
