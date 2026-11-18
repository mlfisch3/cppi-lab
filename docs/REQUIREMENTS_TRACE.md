# CPPI-Lab: requirements traceability

This table was produced by re-reading `docs/spec/PRD.md` and `docs/spec/TSD.md` against the finished product, requirement by requirement.
For each requirement it states where it is implemented, how it was verified, and any gap.
"D-n" refers to decision n in [`DECISIONS.md`](DECISIONS.md).
Paths are relative to the repository root; test names are in `tests/`.

Verification methods:

- **test**: an automated pytest test (run headless with `uv run pytest`).
- **bench**: a measurement by `benchmarks/bench_latency.py` or `benchmarks/bench_fps.py`.
- **display**: exercised in the real application on the display by `scripts/ui_tour.py`, with the resulting screenshot inspected.
- **mouse**: exercised in the real application with injected pointer events by `scripts/drag_dock_tour.py` (X11, `xdotool`), which checks every step.
- **manual**: a one-off check described in the row.

Status values: **met**, **met with note** (met, with a clarification or resolved ambiguity), **partial**, **not met**.

## Summary of gaps

| Gap | Where |
| :--- | :--- |
| The bundled data is Binance BTCUSDT, used as the BTC-USD proxy, not a USD-quoted pair. | FR-1.1, D-21 |
| Moving a window by a title bar the window manager draws (the main window's, or a floating window's on Wayland) cannot be reached by injected pointer events and was not exercised; on X11 the floating windows have the application's own title bar, and moving, resizing and docking them by mouse were. | FR-5.1, FR-6.3, D-36 |
| Under a Wayland session drag-and-drop docking is not available and floating windows are positioned by the compositor; the toolbar menus dock instead. The X11 platform, on which it works, needs `libxcb-cursor.so.0`, which is not installed in this machine's system (a user-level copy is enough). | FR-5.1, D-31 |
| Windows 11 and macOS, named as target platforms, were not run or tested. | PRD header |
| The kernels are compiled without `fastmath`, contrary to the decorator shown in the TSD. | TSD 1.1, D-6 |
| With the specified defaults the default window contains no orders; this is correct behaviour and is explained in the UI. | FR-2.1, D-20 |
| The frame-rate benchmark is met on both display back ends, on X11 narrowly: its slowest case varied between 14.0 and 16.5 ms from run to run against a budget of 16.7 ms. Inside the complete 1920x1040 window, which the benchmark reports as context, the worst 95th-percentile frame on X11 was 15.1 to 17.2 ms over six runs, one of them above the budget; an empty window of that size alone needs 9.5 to 10.6 ms there. On Wayland the figures are 12.9 to 13.6 ms and 14.3 to 15.1 ms. | PRD 6.1, D-27 |

## PRD section 3: strategy and invariants

| Requirement | Implementation | Verification | Status |
| :--- | :--- | :--- | :--- |
| 3.1 (1) Hard floor `V_t >= F = N0 P_liq + c0` at all times | `engine/kernel_numba.py` `simulate_core` (steps 1 to 5 and the guards of D-9); counted in the loop as `STAT_FLOOR_BREACHES` | test: `test_floor_invariant.py` (Hypothesis over 120 random continuous markets and parameter sets; 480 real-window runs; 30 random real windows; five hand-built paths) | met with note: holds on every bar that does not gap; a bar that opens below the trigger is reported as GAP BREACH (theory volume 5.1) |
| 3.1 (2) `P_floor,t = max(P_liq/(1-gamma), max(0, F-c_t)/(n_t(1-gamma)))` | `trigger_price()` | test: `test_exit_at_the_trigger_lands_exactly_on_the_floor`, `test_row_identities`, `test_kernel_reference.py` | met |
| 3.1 Baseline buffer `P_liq gamma/(1-gamma)` | follows from `trigger_price()`; column `safety_buf_usd` | test: `test_row_identities` (never below the baseline), `test_idle_window_reads_intact` (60,240.96) | met |
| 3.1 Dynamic whipsaw buffer expansion | second term of `trigger_price()`; drawn as the stepped `P_floor` line and shaded zone | test: `test_kernel_reference.py`; display: screenshots 02, 03 | met |
| 3.2 `d_t`, complete sale as `P -> P_floor` | step 4 of `simulate_core` | test: `test_kernel_reference.py`, `test_floor_invariant.py` | met |
| 3.2 Sell ceiling `n_ceil` | `sell_ceiling()` | test: `test_kernel_reference.py`, `test_row_identities` | met with note: bracket floored at zero (D-10) |
| 3.2 Rebuy floor `n_floor` | step 4 of `simulate_core`, column `n_rebuy` | test: `test_kernel_reference.py`, `test_row_identities` | met with note: capped at `n_ceil` (D-10); purchases limited by the rebuy cap (D-9) |

## PRD section 4: functional requirements

| Requirement | Implementation | Verification | Status |
| :--- | :--- | :--- | :--- |
| FR-1.1 Bundled real 1-minute dataset with `timestamp_utc, open, high, low, close, buy_taker_vol, sell_taker_vol`, multi-year, drawdowns, consolidations, V-recoveries | `cppi_lab/data/btc_usd_1m.parquet` (2,015,920 bars, 2023-01-01 to 2026-10-31); `data/loader.py` `load_bundled`; `scripts/build_dataset.py` | test: `test_bundled_file_has_the_prd_schema`, `test_bundled_data_is_real_multi_year_one_minute_history`; manual: SHA-256 of every archive verified by the build script | met with note: Binance BTCUSDT as the BTC-USD proxy (D-21); range chosen for file size (D-22) |
| FR-1.2 Import CSV/Parquet: Coinbase `market_trades`/candles, Binance `aggTrades`/`klines` | `data/loader.py` `load_file`, `detect_format`, `parse_*`; `ui/main_window.py` `import_file` | test: `test_loader.py` (one test per layout, headed and headerless, millisecond and microsecond stamps, both side conventions, invalid files); `test_import_file_becomes_the_active_dataset`, `test_bad_import_is_reported_not_raised` | met |
| FR-1.2 Fetch candles via the Coinbase Advanced Trade REST API | `fetch_coinbase_candles`; `CoinbaseSyncDialog` | test: `test_fetcher_pages_through_the_range_without_credentials`, `test_fetcher_reports_failures` (injected transport); manual: one live fetch of 480 candles for 2026-11-10 succeeded | met with note: the dialog itself was not driven in an automated test |
| FR-1.3 `Normalize Start Price to Reference Anchor` | `MarketWindow.rescaled`; controller; sidebar row with anchor price; amber chip in the top bar | test: `test_rescaling_preserves_returns_and_volumes`, `test_normalisation_anchors_the_first_open`, `test_anchor_price_edit`; display: screenshot 03 | met with note: default off (D-17) |
| FR-2.1 Default initialisation | `config/defaults.py` | test: `test_defaults.py` (every row of section 5), `test_launches_with_prd_defaults_in_every_control` | met |
| FR-2.2 Slider and spin box in sync, per-row reset, global reset | `ui/control_sidebar.py` `ParamRow`, `ControlSidebar` | test: `test_every_numeric_row_has_slider_spin_reset_and_badge`, `test_spin_edit_recalculates_and_moves_the_slider`, `test_slider_edit_recalculates_and_moves_the_spin_box`, `test_reset_one_row`, `test_reset_all`, `test_start_time_edit_and_reset`, `test_values_are_clamped_to_the_prd_range`, `test_p_ref_follows_p_liq` | met with note: the two toggles (Monte Carlo fan, normalisation) are switches, not sliders |
| FR-2.3 Re-run 43,200 bars and refresh plots and tables in under 50 ms | `engine/controller.py` (staged caches); `MainWindow.recalculate` | bench: 7 to 13 ms median, single runs up to 18 ms; test: `test_recalculation_refreshes_kpis_chart_and_ledger`, `test_burst_of_edits_is_coalesced_into_one_run`, `test_stage_caches_are_reused_only_when_valid` | met with note: the figure covers the back-test and the update of every view's data; painting the next frame costs a further 2 to 4 ms |
| FR-3.1 Maker fee, taker fee, slippage buffer | parameters `f_maker`, `f_taker`, `delta_slip` | test: `test_defaults.py`, `test_uniform_draw_decides_fill_or_miss`, `test_row_identities` | met |
| FR-3.2 Stochastic maker fill model, seeded uniform draw, fill at limit with maker fee, missed orders logged | `fill_probability`, step 3 of `simulate_core`, `uniform_draws` | test: `test_fill_model.py`, `test_prng.py`, `test_kernel_reference.py` | met |
| FR-3.3 Monte Carlo fan, 100 seeds in parallel, 5th to 95th percentile band behind the active trajectory | `engine/monte_carlo.py`; `MainWindow._request_ensemble`; `ChartWorkspace.set_ensemble` | test: `test_monte_carlo.py`, `test_monte_carlo_fan_toggle`, `test_monte_carlo_fan_runs_on_a_worker_thread`, `test_monte_carlo_fan_layer`; display: screenshot 03 | met |
| FR-3.4 Risk aversion `rho`: 0 = maker only plus the emergency exit, 1 = taker only, interior = split or escalate | `route_order_by_rho`, `taker_fraction`, `execution_urgency` | test: `test_routing.py` | met with note: maker dominance override when `f_maker >= f_taker + delta_slip` (D-9) |
| FR-4.1 Everything on one screen without vertical scrolling or tabs | `ui/main_window.py` layout; sidebar rows of 25 px; `ui/kpi_ribbon.py` | test: `test_everything_fits_one_1080p_screen_without_scrolling`, `test_small_screens_scroll_only_the_sidebar`, `test_ui_layout.py` (`test_nothing_is_shortened_at_1920x1080` for five scenarios, laptop sizes, minimum size); display: screenshots 01 to 03 at 1920x1040, 21 at 1536x864 and 1366x768 | met at 1920x1080 with nothing shortened; below that only the parameter column scrolls and texts use deliberate short wordings (D-33) |
| FR-4.2 Pane 1: close with low-high shading, `P_liq`, `P_floor`, `V_t`, six marker kinds | `ui/chart_workspace.py`, `ui/chart_gl.py` | test: `test_layers_carry_the_telemetry`, `test_markers_are_lossless`, `test_visual_encoding_matches_the_tsd_matrix`; display: screenshots 02 to 04 | met |
| FR-4.2 Pane 2: holdings, `n_max`, ceiling, rebuy floor, shaded deadband | same | same | met |
| FR-4.2 Pane 3: both signals and a stem per maker attempt showing `P_fill` against `U_t` | same | same | met |
| FR-4.2 Vertically synchronised panes | linked x axes; shared crosshair | test: `test_three_linked_panes_with_headers`, `test_zoom_is_synchronised_across_panes`, `test_crosshair_and_readout` | met |
| FR-4.3 Automated visual hierarchy and event callouts | draw order, line weights, level of detail, marker density scaling, liquidation flash lines and clustered badges | test: `test_visual_encoding_matches_the_tsd_matrix`, `test_the_frame_draws_every_layer_marker_and_flash`, `test_liquidation_badges_cluster_and_never_overlap`; display: screenshots 02 to 04, 21 | met |
| FR-5.1 (1) Ledger window movable and resizable | `CDockWidget` in a `CFloatingDockContainer`; on X11 its own title bar, `TitleBarPressFix` and `ResizeGrips` (`ui/widgets.py`) | mouse: title-bar drag follows the cursor to the pixel, corner and edge drags resize; test: `test_ui_floating.py` (grips from every edge and corner, minimum size, title-bar correction), `test_ledger_opens_floating_on_top_and_docks`; display: screenshot 08 | met; with a window-manager title bar (Wayland) the move was not exercised by mouse (D-36) |
| FR-5.1 (2) Stays on top of the main window | `MainWindow._apply_floating_flags`, `floating_window_flags` (`WindowStaysOnTopHint`; `Tool`, or an owned ordinary window where utility windows are not shown) | test: `test_ledger_opens_floating_on_top_and_docks`, `test_tool_windows_are_used_only_where_they_are_shown`, `test_floating_window_on_x11_has_its_own_title_bar_and_resize_grips`; mouse: the window is on screen over the main window | met with note: window type chosen from the window manager's capabilities (D-30) |
| FR-5.1 (3) Dockable left, right, bottom by drag and drop | Qt Advanced Docking System; `MainWindow.dock_to`; toolbar menus | mouse: the floating window is dragged onto the left, right and bottom compass indicator, the compass reports the drop area, and the panel is docked on that side with the chart still drawn; test: `test_ledger_opens_floating_on_top_and_docks`, `test_docking_menu_lists_every_position`, `test_docked_panel_can_collapse_into_an_auto_hide_side_bar`; display: screenshots 10 to 12, 17 | met on X11; on Wayland docking is by the toolbar menu (D-31) |
| FR-5.1 (4) Vertical and horizontal scrolling in the table | `QTableView` | test: `test_table_scrolls_both_ways_and_is_read_only` | met |
| FR-5.2 All quantities at every timestep | `TelemetryTableModel`, 26 columns | test: `test_shape_and_headers_cover_every_telemetry_field`, `test_model_is_a_zero_copy_view`, `test_cell_formatting` | met |
| FR-5.2 Filters `All Timesteps`, `Executed Trades`, `Missed Maker Orders`, `Floor Liquidations` | `TelemetryTableModel.set_filter` | test: `test_filter_names_match_the_prd`, `test_filters_select_exactly_the_flagged_bars`, `test_missed_filter_includes_misses_hidden_behind_a_taker_fill`, `test_filter_combo_offers_the_four_prd_filters`; display: screenshots 08 to 10 | met |
| FR-5.2 Column sorting | `TelemetryTableModel.sort` | test: `test_sorting_ascending_descending_and_stable`, `test_header_click_sorts_and_toggles_direction` | met |
| FR-5.2 Bidirectional row-to-chart crosshair synchronisation | `LedgerWidget.barActivated`, `ChartWorkspace.barSelected` | test: `test_ledger_row_drives_the_chart_and_back`, `test_row_selection_emits_the_bar_and_select_bar_is_silent`, `test_changing_filter_resyncs_the_chart` | met |
| FR-5.2 CSV and Parquet export | `TelemetryTableModel.export` | test: `test_export_writes_the_rows_on_screen`, `test_export_from_the_widget` | met |
| FR-6.1 `?` badge next to every parameter, KPI card and pane header | `HelpBadge` in `ParamRow`, `KpiCard`, `PaneHeader` | test: `test_every_explained_item_has_a_question_mark_badge` (39 badges), `test_every_badge_has_a_popover_entry` | met |
| FR-6.1 Rich popover with rendered LaTeX explaining definition and implementation | `RichPopoverWidget`; `cppidoc://popover/<id>`; 37 entries in `config/doc_registry.json` | test: `test_popover_entry_is_complete`, `test_rho_popover_matches_the_tsd_example`, `test_web_popover_renders_math_and_its_links_open_the_docs`; display: screenshots 05 to 07, 19 | met |
| FR-6.1 300 ms hover-intent grace, links, pin as a movable card | `RichPopoverManager` | test: `test_ui_popover.py` (13 tests) | met with note: pinned cards move inside the main window (D-28) |
| FR-6.2 Theoretical Strategy Documentation with complete derivations | `cppi_lab/docs/theory.html` (8 sections, 24 display equations) | test: `test_sections_are_substantial_prose_with_math`, `test_volumes_cover_the_topics_required_by_the_prd`, `test_theory_volume_renders_with_math_and_the_full_spine`; display: screenshots 13, 17 | met |
| FR-6.2 Technical Implementation Documentation | `cppi_lab/docs/tech.html` (8 sections) | test: same, plus `test_technical_volume_renders_without_errors`; display: screenshots 14, 20 | met |
| FR-6.3 Every section linked to its exact counterpart, both ways | `a.xref` links; `doc_registry.json` | test: `test_every_section_links_to_its_exact_counterpart`, `test_every_link_resolves`, `test_clicking_a_counterpart_link_navigates_the_primary_view` | met |
| FR-6.3 Viewer movable, resizable, on top when floating, dockable | `docs_dock` `CDockWidget` | mouse: the same move, resize and three drops as for the ledger; test: `test_popover_link_opens_the_workbench_at_the_exact_section`, `test_docking_menu_lists_every_position`; display: screenshots 13 to 18, 20 | met with the same notes as FR-5.1 |
| FR-6.3 Live counterpart preview while reading or hovering links | `DocWorkbench`, `DocSyncBridge`, `docs/assets/doc_sync.js` | test: `test_scroll_spy_moves_the_counterpart_preview`, `test_hovering_a_link_peeks_and_leaving_returns`, `test_collapsed_preview_falls_back_to_a_snippet_card`; display: screenshots 13 to 15 | met |

## PRD section 5: parameters

Every row is defined once in `cppi_lab/config/defaults.py` (`PARAM_SPECS`) and verified by `test_defaults.py::test_numeric_row_matches_prd` (default, minimum, maximum and step) and by `test_launches_with_prd_defaults_in_every_control`.

| Rows | Implementation | Status |
| :--- | :--- | :--- |
| Account & Horizon: `t_0`, `Delta T`, `N_0`, `c_0` | fields `start_ms`, `duration_days`, `n0`, `c0` | met with note: `t_0` is bounded by the active dataset (D-19) |
| CPPI Floor & Band: `P_liq`, `P_ref`, `Delta n_min` | `p_liq`, `p_ref`, `dn_min_pct` | met with note: `P_ref >= P_liq + 1,000` enforced by raising `P_ref` (D-19) |
| Fees & Execution: `f_maker`, `f_taker`, `delta_slip`, `rho` | `f_maker`, `f_taker`, `delta_slip`, `rho` | met |
| Liquidity Model: `lambda_liq`, `beta_vel`, `alpha_size`, `s_rng`, `MC_fan` | `lambda_liq`, `beta_vel`, `alpha_size`, `seed`, `mc_fan` | met |
| Decline Signal: `W_f,d / W_s,d`, `W_atr,d`, `W_imb,d`, `w_m,d`, `k_sell` | `w_fast_d`, `w_slow_d`, `w_atr_d`, `w_imb_d`, `w_mom_d`, `k_sell` | met with note: the paired row is shown as two rows (D-3) |
| Growth Signal: `W_f,u / W_s,u`, `W_atr,u`, `W_brk,u`, `tau_veto` | `w_fast_u`, `w_slow_u`, `w_atr_u`, `w_brk_u`, `tau_veto` | met with the same note |

## PRD section 6: quality requirements

| Requirement | Implementation | Verification | Status |
| :--- | :--- | :--- | :--- |
| 6.1 Single trajectory, 43,200 bars, <= 15 ms on one core | `simulate_core` | bench: 1.0 to 1.7 ms median, single runs up to 5 ms | met |
| 6.1 100-seed ensemble <= 120 ms with `@njit(parallel=True)` | `ensemble_core`, `percentile_band` | bench: 10 to 24 ms kernel only; 47 to 73 ms median including the draws and percentiles; single runs up to 95 ms | met |
| 6.1 Pan, zoom and crosshair at 60 FPS | `ui/chart_gl.py`, `ui/chart_geometry.py`, `ui/chart_workspace.py` | bench: worst 95th-percentile frame 12.9 to 13.6 ms (Wayland) and 14.0 to 16.5 ms (X11) over pan, zoom and crosshair for a 30-day and a 365-day back-test, met in every run; paint 1.2 to 3.6 ms; empty-scene floor 7.3 to 10.1 ms at the 95th percentile | met with note: narrowly on X11, and inside the complete window on X11 the figure is at the limit, 15.1 to 17.2 ms over six runs (D-27) |
| 6.2 IEEE-754 `float64` accounting with conservation assertions `V_t >= F - 1e-6` | kernel compiled without `fastmath`; breach counter in the loop; Floor Integrity card | test: `test_kernel_reference.py` (bit-identical to the reference), `test_floor_invariant.py`, `test_accounting.py` | met |
| 6.3 Dark theme `#0B0F19`, `#111827`, `#1E293B` | `ui/theme.py` | test: `test_visual_encoding_matches_the_tsd_matrix`; display: all screenshots | met |

## TSD

| Item | Implementation | Verification | Status |
| :--- | :--- | :--- | :--- |
| 1.1 Python 3.11+, PySide6 (Qt 6.7+), PySide6-QtAds | `pyproject.toml`, `uv.lock` (Python 3.12, PySide6 6.12, QtAds 5.1) | `uv sync`; the test-suite | met |
| 1.1 pyqtgraph with `useOpenGL=True`, `enableExperimental=True` | `main.py` sets both; pyqtgraph provides the layout, ranges and interaction; the canvas is drawn in one pass by `ui/chart_gl.py` on the same OpenGL viewport | display: all screenshots; bench | met with note: single-pass renderer instead of pyqtgraph's items (D-25) |
| 1.1 Numba `@njit(nogil=True, fastmath=True, parallel=True)`, counter-seeded PCG64 | `engine/kernel_numba.py`, `engine/monte_carlo.py` | test: `test_prng.py`, `test_monte_carlo.py` | met with note: no `fastmath` (D-6); seeds `s_rng + m` (D-16) |
| 1.1 Polars + PyArrow, Zstd Parquet, zero-copy views into a virtualised `QAbstractTableModel` | `data/loader.py` (PyArrow reader for imported Parquet); `TelemetryTableModel` | test: `test_loader.py`, `test_model_is_a_zero_copy_view`, `test_window_slicing_is_zero_copy_and_half_open` | met |
| 1.1 `QWebEngineView` + `cppidoc://` scheme handler + local KaTeX + `QWebChannel` | `ui/doc_scheme.py`, `ui/doc_workbench.py`, `docs/assets/` | test: `test_scheme_handler_serves_pages_assets_and_popovers`, `test_katex_is_bundled_locally_with_its_fonts`, `test_ui_docs_web.py` | met |
| 1.2 Component topology | `ControlSidebar`, `SimulationController`, `ChartWorkspace`, `LedgerWidget`, `DocWorkbench` | the test-suite | met with note: the widgets inside the dock containers are named `LedgerWidget` and `DocWorkbench` |
| 2.1 Bar representation, CLV fallback | `MarketData`, `clv_split` | test: `test_clv_split_formula`, `test_coinbase_candles_use_the_clv_fallback` | met |
| 2.2 Signals | `engine/signals.py` | test: `test_signals.py` (exact against a naive reference, cross-checked against Polars) | met with note: conventions of D-15 |
| 2.3 Floor, trigger, envelope, sell and buy criteria | `simulate_core` | test: `test_kernel_reference.py`, `test_floor_invariant.py` | met with notes: D-8, D-9, D-10 |
| 2.4 Urgency and routing law | `execution_urgency`, `taker_fraction`, `route_order_by_rho` | test: `test_routing.py` | met with note: miss counter lifetime (D-11) |
| 2.5 Fill model and uniform resolution | `fill_probability`, step 3 | test: `test_fill_model.py` | met with note: a fill needs `P_fill > 0` (D-14) |
| 3.1 Intra-bar order: floor, trigger, maker resolution, control law, telemetry | `simulate_core` | test: `test_kernel_reference.py` (reference written step by step from this section) | met with note: step 3b added (D-9) |
| 3.2 Telemetry schema | `TELEMETRY_DTYPE` | test: `test_telemetry_dtype_is_the_tsd_schema`, `test_row_identities` | met with note: 26 fields (D-1); event priority and flags (D-12); sentinels (D-13) |
| 4.1 KPI ribbon of 64 px with 8 cards; 3 panes at 55:25:20; shared date axis | `ui/kpi_ribbon.py`, `ChartWorkspace._apply_pane_heights`, `chart_geometry.time_ticks` | test: `test_everything_fits_one_1080p_screen_without_scrolling`, `test_there_are_eight_cards_with_popovers`, `test_ribbon_goes_from_one_row_to_two_rows_of_four`, `test_time_and_value_ticks` | met with note: on a narrow canvas the eight cards form two rows inside the same 64 px (D-33) |
| 4.1 Visual-encoding matrix (colours, weights, dashes, symbols, sizes) | `ui/theme.py`, `ChartWorkspace._build_layers` | test: `test_visual_encoding_matches_the_tsd_matrix`; display | met with note: a missed draw is shown as an amber ring and a filled one as a purple dot, so the outcome of each coin flip is visible |
| 4.2 Floating flags; docked modes left, right, bottom, auto-hide | `MainWindow._apply_floating_flags`, `floating_window_flags`, `dock_to`, auto-hide configuration | test: `test_ledger_opens_floating_on_top_and_docks`, `test_tool_windows_are_used_only_where_they_are_shown`, `test_docked_panel_can_collapse_into_an_auto_hide_side_bar`; mouse | met with note: `Tool` only where the window manager shows utility windows (D-30) |
| 4.3 Popover flow: 150 ms debounce, 8 px anchor, 300 ms grace, fade, pin; example content for `rho` | `ui/popover_manager.py`, `doc_registry.json` | test: `test_ui_popover.py`, `test_rho_popover_matches_the_tsd_example` | met |
| 5.1 `doc_registry.json`, shared spine, `data-concept-id` | `config/doc_registry.json`, both HTML volumes | test: `test_registry_spine_has_eight_sections_including_the_six_tsd_concepts`, `test_both_volumes_share_the_spine_in_order` | met with note: 6 tabulated ids plus 2 (D-2) |
| 5.2 Primary 65 %, preview 35 %; toolbar Back, Forward, Switch, Search, Toggle | `DocWorkbench` | test: `test_splitter_gives_primary_65_and_preview_35`, `test_toolbar_has_the_five_specified_controls`, `test_switching_volume_keeps_the_concept`, `test_search_reports_matches` | met |
| 5.2 Scroll-spy with `IntersectionObserver` and `notifySectionFocused`; left-border highlight; hover peek; snippet card when collapsed; click navigates | `docs/assets/doc_sync.js`, `DocSyncBridge` | test: `test_ui_docs_web.py` | met |
| 6 Module layout | as listed, plus `engine/controller.py`, `ui/chart_gl.py`, `ui/chart_geometry.py`, `ui/doc_scheme.py`, `ui/theme.py`, `ui/widgets.py`, `ui/x11.py` | - | met with note: seven helper modules were added |

## Engineering quality

| Check | Command | Result |
| :--- | :--- | :--- |
| Tests | `uv run pytest` | 456 passed; run three times in a row on the final commit with no intermittent failure |
| Lint and format | `uv run ruff check . && uv run ruff format --check .` | clean |
| Types | `uv run mypy` (strict) | clean, 52 source files |
| Real application, every state | `uv run python scripts/ui_tour.py OUT` | 22 screenshots, inspected |
| Real mouse: move, resize, dock | `uv run python scripts/drag_dock_tour.py OUT` | 66 checks passed, three runs in a row |
| Latency budgets | `uv run python benchmarks/bench_latency.py` | all met |
| Frame rate | `uv run python benchmarks/bench_fps.py` | met on Wayland and on X11 in every run, narrowly on X11; full-window context figure at the limit on X11 (D-27) |
