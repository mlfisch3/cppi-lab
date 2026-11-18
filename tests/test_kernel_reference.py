"""The Numba kernel must reproduce the plain-Python reference exactly."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketData, MarketWindow
from cppi_lab.engine.kernel_numba import EVENT_NAMES, TELEMETRY_FIELDS, SimResult
from cppi_lab.engine.signals import Signals
from tests.conftest import anchored, backtest, ms, random_params, real_window, synthetic_window
from tests.reference import ref_simulate


def assert_identical(
    window: MarketWindow, signals: Signals, result: SimResult, p: SimParams
) -> None:
    """Every telemetry field and the event flags are bit-for-bit equal."""
    rows, machine = ref_simulate(
        window, signals.s_down, signals.s_up, signals.atr, result.telemetry["u_rand_draw"], p
    )
    for name in TELEMETRY_FIELDS:
        fast = result.telemetry[name]
        slow = np.array([getattr(row, name) for row in rows], dtype=fast.dtype)
        equal_nan = fast.dtype.kind == "f"
        if not np.array_equal(fast, slow, equal_nan=equal_nan):
            bad = np.nonzero(~((fast == slow) | (equal_nan & (fast != fast) & (slow != slow))))[0]
            first = int(bad[0])
            raise AssertionError(
                f"{name} differs at bar {first}: kernel {fast[first]!r} vs reference "
                f"{slow[first]!r} ({bad.size} bars)"
            )
    np.testing.assert_array_equal(result.flags, np.array([r.flags for r in rows], dtype=np.uint8))
    stats = result.stats
    counts = machine.counts
    assert (stats.maker_sells, stats.taker_sells, stats.maker_buys, stats.taker_buys) == (
        counts[1],
        counts[2],
        counts[3],
        counts[4],
    )
    assert (stats.missed_sells, stats.missed_buys, stats.liquidations) == (
        counts[5],
        counts[6],
        counts[7],
    )
    assert stats.gap_bar == machine.gap_bar


REAL_CASES = [
    ("default", {}, ms(2026, 1, 1), 2),
    ("rescaled", {"normalize": True}, ms(2026, 1, 1), 6),
    ("feb crash rho 0", {"rho": 0.0}, ms(2026, 2, 3), 4),
    ("feb crash rho .25", {"rho": 0.25}, ms(2026, 2, 3), 4),
    ("feb crash rho .6", {"rho": 0.6}, ms(2026, 2, 3), 4),
    ("feb crash rho 1", {"rho": 1.0}, ms(2026, 2, 3), 4),
    ("june with cash", {"c0": 5_000.0, "n0": 2.5}, ms(2026, 6, 1), 6),
    (
        "maker dominated",
        {"f_maker": 0.01, "f_taker": 0.0, "delta_slip": 0.0, "rho": 0.0},
        ms(2026, 3, 1),
        4,
    ),
    (
        "gain 2",
        {"k_sell": 2.0, "tau_veto": 1.0, "dn_min_pct": 0.001, "rho": 0.4},
        ms(2026, 3, 1),
        4,
    ),
    ("zero friction", {"f_maker": 0.0, "f_taker": 0.0, "delta_slip": 0.0}, ms(2026, 6, 10), 4),
    ("starts below trigger", {"p_liq": 90_000.0, "p_ref": 99_000.0}, ms(2026, 1, 1), 1),
]


@pytest.mark.parametrize(("label", "changes", "start", "days"), REAL_CASES, ids=lambda c: str(c))
def test_kernel_equals_reference_on_real_windows(
    bundled: MarketData,
    params: SimParams,
    label: str,
    changes: dict[str, float],
    start: int,
    days: int,
) -> None:
    p = replace(params, **changes)  # type: ignore[arg-type]
    window = real_window(bundled, start, days, p)
    signals, result = backtest(window, p)
    assert_identical(window, signals, result, p)


@pytest.mark.parametrize("seed", range(24))
def test_kernel_equals_reference_on_random_markets(seed: int) -> None:
    rng = np.random.default_rng(9_000 + seed)
    window = synthetic_window(
        2_500,
        seed,
        start_price=float(rng.uniform(2_000, 240_000)),
        volatility=float(rng.uniform(0.0003, 0.004)),
        drift=float(rng.uniform(-0.0003, 0.0002)),
        gap_free=bool(seed % 3),
    )
    p = anchored(random_params(rng), window, rng)
    signals, result = backtest(window, p)
    assert_identical(window, signals, result, p)


def test_reference_run_covers_every_event_code(bundled: MarketData, params: SimParams) -> None:
    """The comparison is only meaningful if the scenarios exercise every branch."""
    seen: set[int] = set()
    for rho, start in ((0.25, ms(2026, 6, 1)), (0.0, ms(2026, 2, 3)), (1.0, ms(2026, 2, 3))):
        p = replace(params, rho=rho)
        _, result = backtest(real_window(bundled, start, 6, p), p)
        for code in range(1, 8):
            if (result.flags & (1 << (code - 1))).any():
                seen.add(code)
    assert seen == set(range(1, 8)), [EVENT_NAMES[c] for c in set(range(1, 8)) - seen]
