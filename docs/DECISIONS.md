# CPPI-Lab: decisions log

This file records every place where the PRD (`docs/spec/PRD.md`) and the TSD (`docs/spec/TSD.md`) were ambiguous, contradicted each other, or could not be followed literally on the development machine, and what was done instead.
Each entry states the issue, the resolution and the evidence.
The rule applied throughout: follow the documents' evident intent, and where two statements conflict prefer the more precise one (an explicit dtype, an explicit formula).

## A. Contradictions inside the specification

### 1. The telemetry schema is called "24 columns" but lists 26 fields

TSD sections 1.2, 3.1 and 3.2 say "24 columns"; the `TELEMETRY_DTYPE` listing in section 3.2 has 26 fields, numbered 0 to 25.
**Resolution:** the explicit dtype is authoritative.
`cppi_lab/engine/kernel_numba.py` defines exactly those 26 fields with those names, types and order, and `tests/test_accounting.py::test_telemetry_dtype_is_the_tsd_schema` pins it.

### 2. The documentation spine is called "8-section" but the table lists 6 concept ids

TSD section 5.1 says both volumes share "an identical 8-section structural spine" and then tabulates six `data-concept-id` values.
**Resolution:** both statements are honoured.
The six tabulated ids are used verbatim, in order, with the section numbers and titles the table gives.
Two more sections complete the spine of eight, covering material PRD FR-6.2 requires and the six ids do not: `sec-5-state-machine` (theory: discrete-time execution, event ordering and gap risk; technical: the Numba loop, telemetry schema and floating-point guards) and `sec-6-workbench` (theory: data, rescaling, KPIs and the Monte Carlo fan; technical: data engine, recalculation pipeline and UI event bindings).

### 3. "Dataclass of all 25 default parameters" versus 27 values

PRD section 5 has 25 rows, but the two "Fast / Slow EMA Windows" rows each hold two values, so there are 27 tunable values.
**Resolution:** `SimParams` has 27 fields for the table, and the sidebar shows the paired rows as two rows each ("Fast EMA Window", "Slow EMA Window") so that every value has its own slider, numeric field and reset button, as FR-2.2 requires.
Both rows of a pair share one `?` popover titled as in the PRD.

### 4. The popover engine is named KaTeX in one place and MathJax in another

TSD section 1.1 specifies KaTeX; the comment on `popover_manager.py` in section 6 says "MathJax".
**Resolution:** KaTeX, the more specific statement (it comes with a version-independent performance rationale).
KaTeX 0.19.1 (MIT licence) is vendored under `cppi_lab/docs/assets/katex/`; nothing is fetched at run time.

### 5. `cushion_usd` is described two ways

The dtype comment says "Active risk cushion V_t - F"; TSD section 2.3.1 defines the cushion as `max(0, V_t - F)`.
**Resolution:** the explicit formula.
The signed quantity is still available as `net_equity_usd - F`, and the kernel reports its minimum (`min_margin_usd`).

### 6. `fastmath=True` versus IEEE-754 accounting

TSD section 1.1 asks for `@njit(nogil=True, fastmath=True, parallel=True)`.
PRD section 6.2 requires IEEE-754 `float64` accounting with strict conservation assertions, and the brief requires the kernel to equal a readable reference implementation.
`fastmath` allows the compiler to assume no NaN or infinity exists and to reorder arithmetic, so results can differ between CPUs and NaN sentinels can be optimised away.
**Resolution:** the kernels are compiled with `nogil=True` and without `fastmath`; `parallel=True` is used where it applies (the ensemble and the percentile reduction).
**Evidence:** without `fastmath` the single-trajectory kernel takes 1.0 to 1.6 ms for 43,200 bars against a 15 ms budget, and it reproduces the plain-Python reference bit for bit in all 26 fields (`tests/test_kernel_reference.py`).

## B. Points the specification leaves open

### 7. Friction of ordinary taker orders

TSD section 3.1 says taker orders execute "at C_t(1 -/+ delta_slip) paying f_taker", which could be multiplicative, while the emergency exit is given explicitly as `c_t = c_{t-1} + n P_trig (1 - gamma)`, which is additive.
**Resolution:** additive everywhere, matching the explicit formula.
A taker sale nets `q C (1 - gamma)` and a taker purchase costs `q C (1 + gamma)`; the reported execution price is `C (1 -/+ delta_slip)` and the reported cost is `q C gamma`.
A useful consequence is that a taker sale leaves `V_t` exactly unchanged.

### 8. Which trigger enters `d_t`

**Resolution:** the step-1 value of the bar, computed from the previous bar's end state, as TSD section 3.1 defines `P_floor,t`.
The telemetry columns `p_floor`, `safety_buf_usd` and `d_norm` are therefore mutually consistent.
If a maker order filled earlier in the same bar, `d_t` is one bar stale and corrects at the next close; the safety guards below always use a freshly computed trigger.

### 9. Three guards required by the primary invariant

The literal TSD state machine does not maintain `V_t >= F`, which the PRD calls the invariant primary constraint.
A counter-example: after an exit exactly at the trigger the account holds cash `F` and no coins; if the price then rises above `P_ref` with `S_up = 1`, the control law buys with all the cash, pays round-trip friction, and the account is immediately worth about `F (1 - 2 gamma)`.
Three guards were added.
With the defaults none of them changes behaviour except in the situations described.

- **Rebuy cap.**
  A buy is trimmed to the largest quantity for which the resulting holding is still at or below the sell ceiling evaluated on the post-trade state.
  This keeps a purchase funded by the cushion, guarantees the trigger stays below the purchase price, and prevents a buy from landing above its own ceiling and being sold again in the next bar.
  It is the envelope's own consistency condition, so it introduces no new parameter.
  A capped quantity below the minimum quantum is not traded.
- **Post-fill trigger re-check (step 3b).**
  A resting bid fills when the market trades down to it, so the bar's low comes at or after the fill.
  After a maker buy fill the trigger is recomputed, and if the same bar's low is at or below it the emergency exit is taken in that bar at the new trigger.
- **Maker dominance.**
  If `f_maker >= f_taker + delta_slip` a passive order is dearer than an immediate one and uncertain as well; every order is then routed as a taker, for any `rho`.
  Without this, a maker sale at a net price below the taker price could raise the trigger.

**Evidence:** `tests/test_floor_invariant.py` checks the invariant on 120 Hypothesis-generated continuous markets with parameters drawn from the full PRD ranges, on 480 real-window runs across seeds, anchors and `rho`, on 30 random real windows with random parameters, and on hand-built paths (exit exactly at the trigger, gap through the buffer, start below the trigger, cash lock, buy followed by a drop).
The theory volume derives the guards in sections 1.2, 2.1, 2.2 and 5.1.

### 10. Degenerate envelopes

With `k_sell > 1` the bracket `1 - k_sell (1 - d) S_down` can be negative, and with a large gain and `tau_veto` near 1 the rebuy floor can exceed the sell ceiling.
**Resolution:** the bracket is floored at zero and `n_floor` is capped at `n_ceil`, so the deadband can close but never invert.

### 11. Lifetime of the miss counter

TSD section 2.4 defines `m_miss` as the count of consecutive missed maker orders but not when it resets.
**Resolution:** it belongs to the current rebalance.
It increases on a miss; it resets on a fill, when a bar's close needs no order, when the order side flips, and on a liquidation.
A full taker escalation leaves it unchanged.

### 12. One `event_code` per bar, but a bar can hold two events

A maker order resolved in step 3 and a taker child executed in step 4 can share a bar; with interior `rho` this is the normal shape of an escalation.
**Resolution:** the row records the highest-priority event (liquidation > taker fill > maker fill > miss) and nothing is lost: the kernel also returns a `uint8` flags array with one bit per event code for everything that happened in the bar, and exact counters.
Chart markers, ledger filters, the ledger's event label (for example `TAKER_SELL +MISS_SELL`) and the KPI counts use the flags and counters.
`fee_paid_usd` sums the cost of all executions of the bar, so the column total equals the run total.

### 13. Sentinels

`p_fill_prob` is NaN on bars without a maker attempt, so "no order" is distinguishable from "zero probability"; `exec_px_usd` is NaN on bars without an event; `u_rand_draw` holds `U_t` on every bar.

### 14. A zero uniform draw

`Generator.random()` draws from `[0, 1)`, so `U_t = 0` is possible, and `U_t <= P_fill` with `P_fill = 0` would then fill an order the bar never traded through.
**Resolution:** a fill requires `P_fill > 0` as well.

### 15. Signal conventions

An EMA is seeded with its first sample.
Rolling sums, maxima and minima use the bars available while the window is still filling.
Signals are computed on the selected window only, with no look-back before bar 0, consistent with `TR_0 = H_0 - L_0`.
The TSD describes the pipeline as "Polars/NumPy"; it is implemented as Numba array passes so that it is exact and fast, Polars is used for all data loading, import and export, and the building blocks are cross-checked against Polars in `tests/test_signals.py`.

### 16. Monte Carlo seeding and the fan's resolution

"Counter-seeded" streams are implemented as member `m` using `PCG64(s_rng + m)`, so member 0 is exactly the trajectory on screen.
The fan is sampled on at most 43,200 bars per member (every bar for windows up to 30 days, every k-th bar beyond), which bounds the equity matrix at about 35 MB for any window length.
Uniform draws for the ensemble are generated in batches of at most 96 MB.

### 17. The rescaling toggle (FR-1.3) is not in the section 5 table

**Resolution:** it is an extra control in the *Account & Horizon* group: a switch plus an anchor price defaulting to 68,000 USD, the PRD's own example.
It defaults to off, because "optional toggle" reads that way and because showing rescaled prices by default would misrepresent history.
While it is on, the top bar shows the factor in an amber chip.

### 18. The eight KPI cards

TSD section 6 says "Top 8 real-time telemetry summary cards" without naming them.
**Resolution:** net equity, cushion, floor integrity, BTC held, liquidation trigger, maker/taker fills, missed maker orders, fees plus slippage.
They are the quantities the PRD's personas are told to inspect (missed maker fills against fee bleed, zero floor breaches across seeds).

The Floor Integrity card answers its own question and nothing else: `INTACT`, `GAP BREACH` or `VIOLATED`.
An emergency exit at the trigger keeps the floor, so it does not change the verdict.
The sub-line counts the liquidations and says whether the strategy holds BTC again or is in cash ("1 liquidation, holds BTC"), and the value turns amber.
An earlier version showed `LIQUIDATED` as the value, which read as the current state even after the strategy had bought back in.

### 19. `Start Date & Time` range and coupling of `P_ref`

The start is clamped to the first and last bar of the active dataset; a window that runs past the end is simply shorter.
`P_ref` is kept at least 1,000 USD above `P_liq`, as the section 5 range states, by raising `P_ref` when `P_liq` is raised.

### 20. The default window contains no orders

With the section 5 defaults (floor 60,000, reference 72,000, start 2026-01-01, 30 days) and real data the strategy never trades.
**Evidence:** BTCUSDT traded between 81,118.00 and 97,924.49 in those 30 days (75,719.90 was the low of the calendar month, on its last day, outside the window), always above `P_ref`, where `d_t = 1` and the ceiling equals `n_max`.
**Resolution:** the defaults are kept exactly as specified (FR-2.1).
The chart shows a banner explaining why nothing happens, with one-click buttons to rescale the window or load a crash scenario, and the top bar offers ten curated scenarios, five of them 2026 periods that exercise the default floor at real prices.

## C. Data

### 21. Source of the bundled dataset

The brief asks for real 1-minute BTC-USD data with real taker buy/sell volume from a public, credential-free source.
**Resolution:** Binance spot **BTCUSDT** 1-minute klines from the public archive `data.binance.vision`.
**Evidence and reasoning:**

- Coinbase's public candle endpoints (Exchange and Advanced Trade) return OHLCV without a taker split, and reconstructing it from individual public trades for several years is not feasible.
- Binance's own BTCUSD pair exists in the archive only for recent months and is illiquid: in January 2026 it had 25,437 of 44,640 minutes with zero volume and 401 BTC traded in the month, against 491,753 BTC for BTCUSDT.
  Binance.US BTCUSD is similarly thin (zero-volume minutes at the default start time).
- `api.binance.com` refuses requests from this location; the archive host does not, and it publishes a SHA-256 checksum next to every file, which the build script verifies.

USDT is therefore used as the USD proxy; the two normally trade within a few basis points.
The dataset selector and both documentation volumes state this.

### 22. Date range and storage

The full history from 2020-01 is 80.5 MB as Zstd Parquet.
**Resolution:** the committed file covers 2023-01-01 to 2026-10-31 (2,015,920 bars, 44.5 MB), which keeps it a single file at the path the TSD names and under 50 MB.
It spans sideways consolidation (2023), the 2024 rally and its August crash and recovery, the 2025 highs and the October 2025 liquidation cascade, and the 2026 decline through the default floor.
`scripts/build_dataset.py --start 2020-01` regenerates a longer file, which the app opens with `--dataset`.

### 23. Missing minutes

Eighty consecutive minutes on 2023-03-24 were never published by the exchange.
**Resolution:** they stay absent; filling them would mean inventing bars.
The loader reports the count and the theory volume explains that such a gap behaves like any other price gap.

### 24. Importers

Binance spot archives switched from millisecond to microsecond timestamps in 2025, so the unit is detected from the magnitude.
Coinbase `market_trades` reports the maker side of each trade (confirmed against Coinbase's Advanced Trade API reference), so `BUY` is counted as taker sell volume by default; the loader takes a `side_convention` argument for files that store the taker side.
The Coinbase REST fetcher uses the public market endpoint (`/api/v3/brokerage/market/products/BTC-USD/candles`), which needs no credential and returns at most 350 candles per request.

## D. User interface and platform

### 25. Chart rendering path

TSD section 1.1 asks for pyqtgraph with `useOpenGL=True` and `enableExperimental=True`.
Both options are set, and pyqtgraph provides the three-pane layout, the linked view ranges and the mouse interaction (drag to pan, wheel to zoom).
Nothing is painted by pyqtgraph items, though.
**Evidence:** pyqtgraph's OpenGL curve path draws `GL_LINE_STRIP` primitives, and the driver on this machine reports `GL_ALIASED_LINE_WIDTH_RANGE = [1.0, 1.0]`, so the 1.5 to 2.5 px line weights and the dashes of the TSD visual-encoding matrix cannot be drawn that way.
Its axis, text and scatter items go through `QPainter`, and every item that draws with OpenGL itself has to open its own native-painting block; together that was most of a frame (profiled; see decision 27).
**Resolution:** the whole canvas is drawn in one pass by `cppi_lab/ui/chart_gl.py`.
The workspace assembles two display lists: the *scene* (grid, layers, markers, liquidation flashes, axes, tick labels) and the *overlay* (crosshair, liquidation badges, read-outs).
Lines, steps and bands are tessellated into triangles in screen space; a small shader anti-aliases the ribbon edges and cuts dashes in window space, so weight, dashing and smoothing are identical on every driver.
Text and markers are rasterised once by Qt into a sprite atlas (pyqtgraph's own symbol shapes are used for the markers) and drawn as textured quads.
Both lists are replayed inside a single native-painting block of the viewport.
Without an OpenGL viewport (`--no-opengl`, the test-suite) the same display lists are drawn with `QPainter`.

### 26. Level of detail

Each layer is reduced to at most two points per pixel for the range in view.
Prices, the trigger, equity and holdings use a min/max reduction that preserves every extreme.
The two signals and the envelope edges oscillate every minute, and their min/max envelope would paint the pane solid, so they are drawn as bucket means when several bars share a pixel and exactly when zoomed in.
Every series carries a pyramid of block minima, maxima and sums at block sizes 4, 16, 64 and so on (`cppi_lab/ui/chart_geometry.py`).
A frame therefore scans at most about eight samples per pixel at any zoom level and any back-test length, and the value axes are auto-ranged from the same pyramids in microseconds.
Points on a straight line between their neighbours are dropped before tessellation, which reduces flat levels and long steps to their corners without changing a pixel.
Marker sizes shrink when more than 250 are in view.

### 27. Chart frame rate: 60 FPS at the 95th percentile

**Requirement:** PRD section 6.1, "OpenGL chart pan/zoom and crosshair inspection must maintain 60 FPS", read as a 95th-percentile frame time of at most 16.7 ms.
**Measured** (`benchmarks/bench_fps.py`, 1450x940 canvas, WSLg, NVIDIA RTX 3080 Ti through the D3D12 driver, vertical sync off): the worst 95th-percentile frame over all pan, zoom and crosshair cases of the 30-day and the 365-day back-test was 12.9 to 13.6 ms on Wayland and 14.0 to 15.5 ms on X11 across the runs made with the final renderer.
Every run met the budget.
The first version measured 18 to 31 ms.
**The floor.**
An empty OpenGL widget of the same size that only clears itself takes 5.3 to 7.6 ms at the median and 7.3 to 10.1 ms at the 95th percentile on this machine, because WSLg has no zero-copy path and reads every frame back from the GPU.
The benchmark reports this floor next to its results.
Whatever the application draws, a frame cannot be faster than that here; the chart adds 2.2 to 3.6 ms of painting to a pan or zoom frame and 1.2 to 1.7 ms to a crosshair frame.
A profile of a pan with the final renderer shows about 6 ms of a 9.9 ms frame inside Qt's presentation, 2.1 ms building the display lists, 0.8 ms in pyqtgraph's range handling, 0.7 ms in the graphics view and 0.4 ms issuing the OpenGL calls.
**What was done.**
A profile of a pan in the first version (19.4 ms per frame at the median, 16.7 ms of it inside the paint event) attributed about 6 ms to the 24 layer items, each of which opened its own native-painting block (2.9 ms of that was decimation and tessellation), 4 ms to pyqtgraph's axis items (tick generation and text through `QPainter`) and 3 ms to the signal chain of a range change, including auto-ranging.
Three empty plots alone took 6.5 to 7.5 ms per repaint.
The single-pass renderer of decision 25 removed the per-item overhead, the pyramids of decision 26 bounded decimation and auto-ranging, and the plots' own axis items no longer track the view.
Crosshair and read-out updates are decoupled from the scene: moving the crosshair rebuilds only the overlay list (a few hundred bytes) and replays buffers that are already on the GPU, with no decimation, tessellation or widget layout.
**In the full window.**
The benchmark also reports the same interactions inside the complete 1920x1040 application window, as context and not as part of its verdict, which is taken on the canvas as in the first version.
There the worst 95th-percentile frame was 14.3, 14.3 and 14.6 ms on Wayland (three runs) and 15.2, 15.9, 16.1, 16.4 and 17.2 ms on X11 (five runs): four within the budget and one above it.
The larger window raises the floor: an empty OpenGL widget of that size needs 8.8 to 8.9 ms at the 95th percentile on Wayland and 9.5 to 10.6 ms on X11, where every frame passes through XWayland as well.
The full window on X11 is therefore at the limit on this machine rather than inside it, and the remaining 4 ms of application work per frame is spread thinly, as the profile above shows.
X11 is nevertheless the platform the application prefers on Linux, because drag-and-drop docking does not work on Wayland (decision 31).
The figures may differ on a native Linux, Windows or macOS session; they were not measured there.

### 28. Popovers are cards inside the main window

The TSD describes `RichPopoverWidget(QFrame)` anchored 8 px right of the `?` icon and pinnable "as a movable floating card".
**Resolution:** the popover is a `QFrame` overlay of the main window with a KaTeX page inside.
It is anchored 8 px to the right of the badge (to the left when it would not fit), and a pinned card can be dragged anywhere inside the main window.
An in-window card behaves identically on every platform, including Wayland, where an application cannot position its own top-level windows.
When QtWebEngine is switched off the same text and links are shown in a `QTextBrowser` without rendered equations.

### 29. Documentation URLs

Links use `cppidoc://<volume>/#<concept-id>`; the TSD's shorter form `cppidoc://tech#sec-2-4-rho` is parsed identically.
Assets are served relative to each page so every request is same-origin.

### 30. Floating window flags

The TSD's flag set is `Tool | WindowStaysOnTopHint | WindowMinimizeButtonHint | WindowMaximizeButtonHint`.
**Resolution:** the docking system's own hints are kept and only the window type and these hints are set.
`Qt.Tool` becomes a utility window on X11.
**Evidence:** WSLg's window manager advertises `_NET_WM_MOVERESIZE`, `_NET_WM_STATE` (fullscreen and maximised only), `_NET_ACTIVE_WINDOW` and `_NET_FRAME_EXTENTS` in `_NET_SUPPORTED`, and no window type at all.
It never maps a utility window: a floating ledger flagged `Qt.Tool` sat about 32,700 pixels off screen in both axes and did not exist on the desktop.
Ordinary windows, dialogs and frameless windows are shown, moved and resized as requested.
Where the window manager answers and does not list `_NET_WM_WINDOW_TYPE_UTILITY`, the floating panel is therefore an ordinary window (`cppi_lab/ui/x11.py` reads the list through Xlib).
It still belongs to the main window (it is transient for it), which keeps it above; the on-top hint is set as well.
Everywhere else the TSD flags are used unchanged.

### 31. Qt platform on Linux, and floating windows on X11

Qt Advanced Docking System needs global cursor positions and window placement for drag-and-drop docking, which a Wayland session does not provide.
**Resolution:** on Linux the app selects the X11 platform plugin when a display is present and `libxcb-cursor.so.0` (required by Qt's X11 plugin since 6.5) can be loaded, and otherwise leaves Qt's choice in place.
The library is looked for in the system, then in the file named by `CPPI_LAB_XCB_CURSOR`, then in `~/.local/lib` and the nix profile; a copy loaded this way satisfies the plugin's dependency without `LD_LIBRARY_PATH`.
Docking is also available from a menu next to each toolbar button, which works on every platform.
**On this machine** the system package is not installed and no system package was installed for this work.
The distribution's own `libxcb-cursor0` package was unpacked into a user directory (`apt-get download`, `dpkg-deb -x`); the nix build of the library needs a newer C library than the system has and does not load.

On X11 the floating windows carry the docking system's own title bar (`FloatingContainerForceQWidgetTitleBar`), so a drag onto the compass is handled inside the application instead of depending on how a window manager reports a move it performs itself.
Three things had to be added for that to work properly; all were found by driving the real application with injected mouse events (decision 34).

- **Title-bar drag.**
  The docking system moves the window to `cursor - press position` and corrects only for a frame on the sides.
  WSLg reports a frame of 59 px above an undecorated window (an invisible shadow margin), and the window jumped down by that amount when a drag started.
  The press is reported that much lower (`TitleBarPressFix`), which cancels the error exactly and changes nothing where no frame is reported.
- **Resizing.**
  A frameless window has no border to pull, so eight grips along the edges and corners resize it (`ResizeGrips`); they also draw the window's one-pixel outline.
- **Native child windows.**
  The docking system made sibling widgets native X windows the first time a panel floated, and a late geometry event from the server then put child widgets back where they had been.
  `AA_DontCreateNativeWidgetSiblings` prevents that.

### 32. QtWebEngine under WSL

**Evidence:** with GPU compositing enabled every documentation page and popover rendered blank under WSLg, with the log line "Compositor returned null texture"; WSLg's EGL reports that `EGL_EXT_image_dma_buf_import` is not supported.
**Resolution:** under WSL the app sets `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu` unless the variable is already set.
The pages are static text and mathematics, and the charts are unaffected.

### 33. Responsive behaviour: nothing is clipped

FR-4.1 is met at 1920x1080: every control, card and pane is visible without scrolling, and no caption, value or sub-line is shortened at all.
Below that, whatever no longer fits is replaced by a shorter wording or an ellipsis with a tool tip; nothing is cut off in the middle or covered by another control.

- **KPI ribbon** (64 px at every size).
  Sub-lines are at most 26 characters, which fits a card of a 1920-pixel window.
  Cards narrower than 168 px use short captions and sub-lines of at most 16 characters; when eight cards no longer fit in a row (below 104 px each, for example with a panel docked beside the workbench) they form two rows of four, one line each.
  The tool tip of a card always has its full caption, value and sub-line.
- **Parameter column.**
  Only this column scrolls on a short screen, and its reset button stays pinned.
  A row is exactly as wide as the scroll viewport, and the gutter of the scrollbar is reserved whether or not there is anything to scroll, so the scrollbar never covers or shifts the per-row reset buttons.
- **Top bar.**
  The status read-out has four wordings, from every timing down to the bar count.
  It gets the longest one that fits once the scenario selector has given up what it can spare; the dataset name keeps its full width unless the window is too narrow even for the bar count.
  A selector that is narrower than its text ends it with an ellipsis.
- **Chart.**
  A pane's legend gives way to the crosshair read-out when both do not fit; value-axis labels are thinned on a short pane instead of overlapping.
- **Liquidation badges.**
  Liquidations whose flash lines would fall inside one label share it ("LIQUIDATION ×4").
  A label that would leave the pane is placed to the left of its line, and the opaque labels are drawn after every flash and crosshair line, so no line strikes through one.
- **Secondary windows.**
  A documentation panel docked into a narrow side panel stacks its two viewers and shows the short name of the volume; the ledger's toolbar shortens its export caption; a wide documentation table scrolls inside its own box.

`tests/test_ui_layout.py` pins these properties at 1920x1040, 1536x824, 1366x728, 1280x680 and at the minimum window size.

### 34. The chart went black after a panel was docked and floated again

**Evidence:** on Wayland and on X11, in the first version as well, docking a floating panel and then floating it again left the OpenGL chart black until the window was next resized; repaints did not help.
When the docking system re-parents the workbench, Qt tells every OpenGL widget below it that its window is about to change, and the widget drops the texture through which its framebuffer is composited into the window.
A stand-alone `QOpenGLWidget` recreates it in its own paint event; the viewport of a graphics view never runs that paint event, and a scroll area asks its viewport to rebuild the framebuffer only when its size really changes.
**Resolution:** the canvas notices the two window-change events and has the viewport rebuild its surface by changing its height by one pixel and back, before anything is painted (`ChartCanvas.recreate_surface`).

## E. Verification method

### 35. How the real application was exercised

`scripts/ui_tour.py` builds the real main window on the display with OpenGL and QtWebEngine enabled and performs the user actions: parameter changes, scenario loading, the Monte Carlo fan, zoom and crosshair, hovering `?` badges, pinning a popover, opening, floating and docking the ledger and the documentation, following a popover link, searching, and resizing to laptop sizes.
`QWidget.grab()` cannot see an OpenGL viewport, and the root window of WSLg's rootless X server cannot be captured as a whole.
On the X11 platform each screenshot is therefore assembled from the application's own top-level windows, read back from the X server one by one (ImageMagick `import -window`) and painted at their screen positions; translucent overlays such as the docking compass are rendered by Qt, which keeps their transparency.
The images show the client areas as the X server holds them; the window manager's decoration of the main window and the shadow around the floating windows are not part of them.
Nothing but the application's own windows is ever read.

### 36. Drag-and-drop docking, title-bar move and edge resize with a real mouse

`scripts/drag_dock_tour.py` drives the real application from outside with `xdotool`, which injects genuine pointer events through the X server (XTEST).
For the ledger and for the documentation window it moves the floating window by its title bar, resizes it by its bottom-right corner and by its right edge, drags it onto the left, right and bottom indicator of the docking compass, checks before the release that the compass reports that drop area, releases, and checks that the panel sits on that side and that the chart is still drawn; then it floats the panel again.
It makes 66 checks and saves a screenshot in the middle of each docking drag and after each drop.
`xdotool` was obtained through nix and is not installed in the system.
The tour found the defects described in decisions 30, 31 and 34 and a clipped toolbar in the docked ledger.

One property of the environment had to be handled.
The X pointer is shared with the real mouse: when a window moves or appears under the resting pointer, the X server's pointer jumps there until the next injected motion, and the docking system, which asks the server for the pointer position while it moves a window, then puts the dragged window in the wrong place.
The tour checks the server's pointer at every step of a drag, repeats a gesture that was disturbed, and keeps its windows away from where the real pointer rests.
The injected events cannot reach the window manager's own decoration, so moving a window by a native title bar (the main window's, or a floating window's on Wayland) was not exercised this way.

### 37. The test session applies the application theme

Fonts, paddings and borders come from the theme.
Without it every widget has a different size than on screen, and a layout test proves nothing; the first version's suite ran unstyled.
`tests/conftest.py` now styles the test application exactly as `cppi-lab` styles the real one.
