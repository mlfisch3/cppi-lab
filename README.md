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

Both secondary windows open floating, above the main window.
Drag their title bar to move them and their edges or corners to resize them.
Drag the title bar over the main window and a docking compass appears: drop on its left, right or bottom indicator to dock the window there.
The small menu next to each toolbar button does the same without a drag.
On Linux the drag needs the X11 platform; see [Platform support](#platform-support).

With the PRD's default floor (60,000 USD) and reference (72,000 USD) the default window, the 30 days from 2026-01-01, contains no orders: Bitcoin traded between 81,118 and 97,924 USD in it, always above the reference, so the strategy correctly holds its whole allocation.
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
uv run pytest                     # 456 tests, about one and a half minutes
uv run ruff check .               # lint
uv run ruff format --check .      # formatting
uv run mypy                       # strict type check of the package, scripts and tests
```

The suite runs without a display: Qt is forced onto its `offscreen` platform, with the application's own theme applied so that layout tests measure what is on screen.
It covers the signal math, the simulation kernel against an independent plain-Python reference (all 26 telemetry fields bit for bit), the floor invariant as a property over random markets, seeds and parameters, the routing law, the fill model, PRNG determinism, the Monte Carlo ensemble, the loader and importers, the table model, and the user interface including the chart's display lists, the layout at 1920x1080 and at laptop sizes, the floating windows and the QtWebEngine documentation viewer.

Two scripts exercise the real application on a display and save screenshots.
They are not part of `pytest` because they need an X11 display, `xdotool` and ImageMagick's `import`:

```bash
uv run python scripts/ui_tour.py OUT_DIR          # every main state of the application
uv run python scripts/drag_dock_tour.py OUT_DIR   # real mouse drags: move, resize, dock
```

`drag_dock_tour.py` injects genuine pointer events through the X server.
For the ledger and for the documentation window it moves the floating window by its title bar, resizes it by a corner and an edge, drops it on the left, right and bottom indicator of the docking compass, and checks every step (66 checks); its exit status is 0 only if all of them hold.
Keep your hands off the mouse while it runs: the X pointer is shared with the real one.

## Benchmarks

```bash
uv run python benchmarks/bench_latency.py      # headless
uv run python benchmarks/bench_fps.py          # needs a display and OpenGL
```

Measured on the development machine (Intel i9-12900H, 16 logical CPUs, NVIDIA RTX 3080 Ti Laptop, Linux under WSL2 with WSLg) for a 43,200-bar window.
The ranges cover the scenarios of each benchmark and the runs made during development, during which the machine was not idle: other long-running processes used about one CPU core and part of the GPU.

| PRD budget | Requirement | Measured | Result |
| :--- | :--- | :--- | :--- |
| Single-trajectory kernel | ≤ 15 ms | 1.0 to 1.7 ms median; single runs up to 5 ms | met |
| 100-seed Monte Carlo ensemble | ≤ 120 ms | 10 to 24 ms kernel only; 47 to 73 ms median including the 4.32 million draws and the percentiles; single runs up to 95 ms | met |
| Parameter change to refreshed plots, cards and table | < 50 ms | 7 to 13 ms median; single runs up to 18 ms | met |
| Chart pan, zoom and crosshair | 60 FPS (16.7 ms) at the 95th percentile | worst 95th-percentile frame about 13 to 14 ms on Wayland and 14 to 16.5 ms on X11 | met, narrowly on X11 |

The frame-rate benchmark replays panning, zooming and a crosshair sweep on the canvas at the size it has in a 1920x1080 window, for a 30-day and a 365-day back-test, with vertical sync off.
It judges each case by the 95th percentile of the time from requesting a change to the end of the event-loop pass that presented it.
It also reports two figures for context, which are not part of its verdict: the *presentation floor*, an empty OpenGL widget that only clears itself, and the same interactions inside the complete 1920x1040 window.
The floor is 5.3 to 7.6 ms at the median and 7.3 to 10.1 ms at the 95th percentile on this machine, because WSLg copies every frame back from the GPU; the chart itself adds 1.2 to 3.6 ms of painting.
On Wayland the benchmark is met with about 3 ms to spare.
On X11, where every frame also passes through XWayland, it was met in every run, but the slowest case varied between 14.0 and 16.5 ms from run to run, so the margin can be as small as 0.2 ms.
In the complete window the worst 95th-percentile frame was 14.3 to 15.1 ms on Wayland.
On X11, where an empty window of that size alone needs 9.5 to 10.6 ms at the 95th percentile, it was 15.1 to 17.2 ms over six runs: five within the budget and one above it.
On X11 this machine is therefore at the limit, not comfortably inside it.
The figures may differ on a native Linux, Windows or macOS session; they were not measured there.
See [`docs/DECISIONS.md`](docs/DECISIONS.md), decisions 25 to 27, for how the chart is drawn and what was measured.

## Platform support

The application was built and verified on **Linux under WSL2 with WSLg** only.
Windows 11 and macOS are named targets of the PRD, but **neither has been run or tested**, so treat them as unverified.
The only platform-specific code is for Linux: the choice of the X11 platform in `cppi_lab/main.py` and the window-manager query in `cppi_lab/ui/x11.py`, both of which do nothing elsewhere.

Notes for Linux:

- **Docking by drag and drop needs X11.**
  Qt Advanced Docking System relies on global cursor positions and window placement, which a Wayland session does not provide.
  The app selects the X11 platform (`xcb`) automatically when it is usable, and for that Qt needs the library `libxcb-cursor.so.0`.
  With administrator rights: `sudo apt install libxcb-cursor0`.
  Without: put a copy of the library in `~/.local/lib/` (for example `apt-get download libxcb-cursor0 && dpkg-deb -x libxcb-cursor0_*.deb x && cp x/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0 ~/.local/lib/`), or name it in the environment variable `CPPI_LAB_XCB_CURSOR`.
  The library has to match the system's C library, so take the distribution's own package.
  Without it the app runs on Wayland: floating windows are moved and resized with the compositor's own title bar and frame, and docking is available from the toolbar menus, but a window cannot be dropped onto the main window with the mouse, and the compositor decides where it appears.
- **On X11 the floating windows have the application's own title bar and frame.**
  Moving, resizing and dropping onto the docking compass are then handled inside the application and do not depend on the window manager.
  Where the window manager does not know utility windows (WSLg's does not, and never shows one), a floating panel is an ordinary window that belongs to the main window instead of a `Qt.Tool` window.
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
  ui/                        main window, sidebar, KPI ribbon, ledger, popovers, docs viewer
  ui/chart_workspace.py      the three panes: data, ranges, interaction, display lists
  ui/chart_geometry.py       level-of-detail pyramids, tessellation, tick arithmetic
  ui/chart_gl.py             sprite atlas, frames, OpenGL and QPainter back ends
  ui/x11.py                  what the X11 window manager supports
  docs/                      theory.html, tech.html and the bundled KaTeX
tests/                       pytest suite and the plain-Python reference machine
benchmarks/                  latency and frame-rate benchmarks
scripts/build_dataset.py     regenerates the bundled dataset
scripts/ui_tour.py           drives the real app through its states for screenshots
scripts/drag_dock_tour.py    real mouse drags: move, resize and dock the floating windows
scripts/x11_capture.py       screenshot helper of the two tours
docs/spec/                   PRD and TSD
docs/DECISIONS.md            resolutions of ambiguities and substitutions
docs/REQUIREMENTS_TRACE.md   requirement-by-requirement traceability
```

## Scope

CPPI-Lab is a back-testing and teaching tool.
It places no orders, connects to no exchange account and needs no credential.
