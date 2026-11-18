"""Measure the PRD section 6 latency budgets on this machine.

Usage::

    uv run python benchmarks/bench_latency.py [--out DIR] [--reps N]

Budgets (PRD FR-2.3 and section 6.1):

* single-trajectory kernel over 43,200 bars ............ <= 15 ms (one core)
* 100-seed Monte Carlo ensemble (4.32e6 bar steps) ..... <= 120 ms
* parameter change -> re-run + refresh plots and tables  <  50 ms

Every figure is a wall-clock measurement taken after the Numba kernels have
been compiled; the median, minimum and maximum over the repetitions are
reported, and a budget counts as met only if the *median* is inside it.  The
worst case is printed as well so a missed budget cannot hide.

The script needs no display: the refresh measurement builds the real main
window on Qt's ``offscreen`` platform, which exercises exactly the code that
pushes data into the chart layers, KPI cards and table model.  Frame painting
on a real display is measured separately by ``bench_fps.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numba
import numpy as np

from cppi_lab.config.defaults import MC_SEEDS, SimParams, default_params
from cppi_lab.data.loader import MarketData, load_bundled
from cppi_lab.engine.controller import SimulationController, warmup
from cppi_lab.engine.kernel_numba import N_STATS, TELEMETRY_DTYPE, kernel_args, run_simulation
from cppi_lab.engine.monte_carlo import draw_block, ensemble_core, fan_stride, run_ensemble
from cppi_lab.engine.signals import compute_signals

BUDGET_KERNEL_MS = 15.0
BUDGET_ENSEMBLE_MS = 120.0
BUDGET_RECALC_MS = 50.0
BARS_30D = 43_200


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


def scenarios() -> dict[str, SimParams]:
    """Thirty-day windows from idle to very busy, all on real data."""
    base = default_params()
    return {
        "default window (no orders)": base,
        "Jan 2026 rescaled to $68k": replace(base, normalize=True),
        "Feb 2026 crash": replace(base, start_ms=_ms(2026, 1, 25)),
        "Jun 2026 breakdown, rho=0": replace(base, start_ms=_ms(2026, 6, 1), rho=0.0),
        "Aug 2026 recovery, rho=1": replace(base, start_ms=_ms(2026, 8, 1), rho=1.0),
        "Apr 2026 chop, dn_min=0.1%": replace(base, start_ms=_ms(2026, 4, 1), dn_min_pct=0.001),
    }


def timed(fn: Callable[[], Any], reps: int) -> dict[str, float]:
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "reps": reps,
    }


def bench_kernel(data: MarketData, reps: int) -> dict[str, Any]:
    rows = {}
    for name, params in scenarios().items():
        controller = SimulationController(data, params)
        run = controller.run()
        window, signals = run.window, run.signals
        draws = run.result.telemetry["u_rand_draw"].copy()
        stats = timed(
            lambda w=window, s=signals, p=params, u=draws: run_simulation(w, s, p, u), reps
        )
        events = run.result.stats
        stats["bars"] = window.n
        stats["orders"] = events.maker_attempts + events.taker_fills + events.liquidations
        rows[name] = stats
    worst = max(row["median_ms"] for row in rows.values())
    return {
        "budget_ms": BUDGET_KERNEL_MS,
        "scenarios": rows,
        "worst_median_ms": worst,
        "met": worst <= BUDGET_KERNEL_MS,
    }


def bench_ensemble(data: MarketData, reps: int) -> dict[str, Any]:
    rows = {}
    for name, params in scenarios().items():
        controller = SimulationController(data, params)
        run = controller.run()
        window, signals = run.window, run.signals
        seeds = [int(params.seed) + m for m in range(MC_SEEDS)]
        block = draw_block(seeds, window.n)
        stride = fan_stride(window.n)
        samples = (window.n + stride - 1) // stride
        equity = np.empty((MC_SEEDS, samples))
        stats = np.empty((MC_SEEDS, N_STATS))
        scratch_rows = np.zeros(1, dtype=TELEMETRY_DTYPE)
        scratch_flags = np.zeros(1, dtype=np.uint8)
        args = kernel_args(params)

        def kernel_only(
            w: Any = window,
            s: Any = signals,
            b: Any = block,
            a: Any = args,
            rows_: Any = scratch_rows,
            flags_: Any = scratch_flags,
            equity_: Any = equity,
            stride_: int = stride,
            stats_: Any = stats,
        ) -> None:
            ensemble_core(
                w.ts_ms,
                w.open,
                w.high,
                w.low,
                w.close,
                w.buy_vol,
                w.sell_vol,
                s.s_down,
                s.s_up,
                s.atr,
                b,
                *a,
                rows_,
                flags_,
                equity_,
                stride_,
                stats_,
            )

        rows[name] = {
            "bar_steps": window.n * MC_SEEDS,
            "kernel_only": timed(kernel_only, reps),
            "with_percentiles_cached_draws": timed(
                lambda w=window, s=signals, p=params, b=block: run_ensemble(w, s, p, MC_SEEDS, b),
                reps,
            ),
            "end_to_end_fresh_draws": timed(
                lambda w=window, s=signals, p=params: run_ensemble(w, s, p, MC_SEEDS), reps
            ),
        }
    worst_kernel = max(row["kernel_only"]["median_ms"] for row in rows.values())
    worst_total = max(row["end_to_end_fresh_draws"]["median_ms"] for row in rows.values())
    return {
        "budget_ms": BUDGET_ENSEMBLE_MS,
        "threads": numba.get_num_threads(),
        "threading_layer": numba.threading_layer(),
        "scenarios": rows,
        "worst_kernel_median_ms": worst_kernel,
        "worst_end_to_end_median_ms": worst_total,
        "met": worst_kernel <= BUDGET_ENSEMBLE_MS,
        "met_end_to_end": worst_total <= BUDGET_ENSEMBLE_MS,
    }


def bench_recalc(data: MarketData, reps: int) -> dict[str, Any]:
    """Parameter change -> back-test -> refreshed chart layers, KPIs and table."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from cppi_lab.ui.main_window import MainWindow

    pg.setConfigOptions(useOpenGL=False, antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = MainWindow(data, use_web=False, async_ensemble=False)
    window.resize(1920, 1040)
    window.show()
    app.processEvents()

    cases: dict[str, list[tuple[str, Any]]] = {
        "routing parameter (rho)": [("rho", 0.30), ("rho", 0.25)],
        "fee parameter (f_taker)": [("f_taker", 0.0035), ("f_taker", 0.0030)],
        "signal parameter (W_f,d)": [("w_fast_d", 6), ("w_fast_d", 5)],
        "liquidity parameter (lambda)": [("lambda_liq", 2.6), ("lambda_liq", 2.5)],
        "seed": [("seed", 43), ("seed", 42)],
        "start time (new window)": [("start_ms", _ms(2026, 1, 2)), ("start_ms", _ms(2026, 1, 1))],
    }
    out: dict[str, Any] = {}
    for label, base in (
        ("default window", default_params()),
        ("Jan 2026 rescaled to $68k", replace(default_params(), normalize=True)),
    ):
        window.controller.set_params(base)
        window.recalculate()
        rows = {}
        for name, toggles in cases.items():
            state = {"i": 0}

            def step(
                toggles: list[tuple[str, Any]] = toggles, state: dict[str, int] = state
            ) -> None:
                key, value = toggles[state["i"] % len(toggles)]
                state["i"] += 1
                window.controller.update(key, value)
                window.recalculate()

            rows[name] = timed(step, reps)
        out[label] = rows
    worst = max(row["median_ms"] for rows in out.values() for row in rows.values())
    window.close()
    app.processEvents()
    return {
        "budget_ms": BUDGET_RECALC_MS,
        "windows": out,
        "worst_median_ms": worst,
        "met": worst < BUDGET_RECALC_MS,
    }


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def render(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        "CPPI-Lab latency benchmark",
        "==========================",
        f"date        : {env['date_utc']}",
        f"cpu         : {env['cpu']} ({env['logical_cpus']} logical CPUs)",
        f"platform    : {env['platform']}",
        f"python      : {env['python']}  numba {env['numba']}  numpy {env['numpy']}",
        "",
        f"1. Single-trajectory kernel, {BARS_30D:,} bars - budget <= {BUDGET_KERNEL_MS:.0f} ms",
    ]
    kernel = report["kernel"]
    for name, row in kernel["scenarios"].items():
        lines.append(
            f"   {name:<32} median {row['median_ms']:7.3f} ms   min {row['min_ms']:7.3f}   "
            f"p95 {row['p95_ms']:7.3f}   max {row['max_ms']:7.3f}   ({row['orders']:,} orders)"
        )
    lines.append(
        f"   -> worst median {kernel['worst_median_ms']:.3f} ms: "
        f"{'MET' if kernel['met'] else 'MISSED'}"
    )
    ens = report["ensemble"]
    lines += [
        "",
        f"2. Monte Carlo ensemble, {MC_SEEDS} seeds x {BARS_30D:,} bars - budget <= "
        f"{BUDGET_ENSEMBLE_MS:.0f} ms  ({ens['threads']} threads, {ens['threading_layer']})",
    ]
    for name, row in ens["scenarios"].items():
        lines.append(
            f"   {name:<32} kernel {row['kernel_only']['median_ms']:7.2f} ms   "
            f"+percentiles {row['with_percentiles_cached_draws']['median_ms']:7.2f}   "
            f"end-to-end incl. draws {row['end_to_end_fresh_draws']['median_ms']:7.2f}   "
            f"(max {row['end_to_end_fresh_draws']['max_ms']:.2f})"
        )
    lines.append(
        f"   -> worst kernel median {ens['worst_kernel_median_ms']:.2f} ms: "
        f"{'MET' if ens['met'] else 'MISSED'};  worst end-to-end median "
        f"{ens['worst_end_to_end_median_ms']:.2f} ms: "
        f"{'MET' if ens['met_end_to_end'] else 'MISSED'}"
    )
    recalc = report["recalc"]
    lines += [
        "",
        f"3. Parameter change -> back-test + refresh of plots, KPIs and table - "
        f"budget < {BUDGET_RECALC_MS:.0f} ms",
    ]
    for label, rows in recalc["windows"].items():
        lines.append(f"   [{label}]")
        for name, row in rows.items():
            lines.append(
                f"   {name:<32} median {row['median_ms']:7.2f} ms   p95 {row['p95_ms']:7.2f}   "
                f"max {row['max_ms']:7.2f}"
            )
    lines.append(
        f"   -> worst median {recalc['worst_median_ms']:.2f} ms: "
        f"{'MET' if recalc['met'] else 'MISSED'}"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="directory for the report files")
    parser.add_argument("--reps", type=int, default=60)
    args = parser.parse_args(argv)

    data = load_bundled()
    warmup()
    window = data.window(default_params().start_ms, 30)
    compute_signals(window, default_params())

    report = {
        "environment": {
            "date_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numba": numba.__version__,
            "numpy": np.__version__,
        },
        "kernel": bench_kernel(data, args.reps * 3),
        "ensemble": bench_ensemble(data, args.reps),
        "recalc": bench_recalc(data, args.reps),
    }
    text = render(report)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "bench_latency.txt").write_text(text)
        (args.out / "bench_latency.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
