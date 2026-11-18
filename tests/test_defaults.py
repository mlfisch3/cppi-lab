"""PRD section 5: every default, range and step, exactly as tabulated."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cppi_lab.config.defaults import (
    DEFAULT_START_MS,
    GROUPS,
    PARAM_SPECS,
    SPEC_BY_KEY,
    constrain,
    default_params,
)
from tests.conftest import ms

# key: (default, minimum, maximum, step) in the stored unit.
PRD_SECTION_5 = {
    "duration_days": (30, 1, 365, 1),
    "n0": (1.0, 0.001, 1000.0, 0.05),
    "c0": (0.0, 0.0, 10_000_000.0, 100.0),
    "p_liq": (60_000.0, 1_000.0, 250_000.0, 250.0),
    "p_ref": (72_000.0, None, 350_000.0, 500.0),
    "dn_min_pct": (0.03, 0.001, 0.25, 0.0025),
    "f_maker": (0.0015, 0.0, 0.01, 0.0001),
    "f_taker": (0.0030, 0.0, 0.02, 0.0001),
    "delta_slip": (0.0010, 0.0, 0.02, 0.0001),
    "rho": (0.25, 0.0, 1.0, 0.05),
    "lambda_liq": (2.50, 0.10, 10.0, 0.10),
    "beta_vel": (1.80, 0.0, 10.0, 0.10),
    "alpha_size": (1.20, 0.10, 5.0, 0.10),
    "seed": (42, 1, 9_999_999, 1),
    "w_fast_d": (5, 1, 60, 1),
    "w_slow_d": (30, 10, 240, 1),
    "w_atr_d": (15, 5, 120, 1),
    "w_imb_d": (10, 2, 60, 1),
    "w_mom_d": (0.60, 0.0, 1.0, 0.05),
    "k_sell": (1.00, 0.10, 2.0, 0.05),
    "w_fast_u": (15, 5, 120, 5),
    "w_slow_u": (60, 30, 480, 5),
    "w_atr_u": (60, 15, 240, 5),
    "w_brk_u": (240, 30, 1440, 15),
    "tau_veto": (0.50, 0.10, 1.0, 0.05),
}


@pytest.mark.parametrize("key", sorted(PRD_SECTION_5))
def test_numeric_row_matches_prd(key: str) -> None:
    default, minimum, maximum, step = PRD_SECTION_5[key]
    spec = SPEC_BY_KEY[key]
    assert spec.default == pytest.approx(default)
    assert getattr(default_params(), key) == pytest.approx(default)
    if minimum is not None:
        assert spec.minimum == pytest.approx(minimum)
    assert spec.maximum == pytest.approx(maximum)
    assert spec.step == pytest.approx(step)


def test_start_time_and_toggle_defaults() -> None:
    p = default_params()
    assert p.start_ms == DEFAULT_START_MS == ms(2026, 1, 1)
    assert SPEC_BY_KEY["start_ms"].step == 60_000
    assert p.mc_fan is False
    assert p.normalize is False


def test_all_twenty_five_prd_rows_are_present() -> None:
    table_keys = [s.key for s in PARAM_SPECS if s.key not in ("normalize", "anchor_price")]
    # 25 PRD rows; the two "Fast / Slow EMA Windows" rows hold two values each.
    assert len(table_keys) == 27
    rows = {s.key for s in PARAM_SPECS if not s.paired_with} - {"normalize"}
    assert len(rows) == 25
    assert {s.group for s in PARAM_SPECS} == set(GROUPS)
    assert all(s.popover for s in PARAM_SPECS)


def test_derived_quantities() -> None:
    p = default_params()
    assert p.gamma == pytest.approx(0.0040)
    assert p.floor_value == pytest.approx(60_000.0)
    assert p.dn_min == pytest.approx(0.03)
    assert replace(p, n0=2.0, c0=500.0).floor_value == pytest.approx(120_500.0)


def test_p_ref_stays_a_thousand_above_p_liq() -> None:
    p = default_params().with_value("p_liq", 71_500.0)
    assert p.p_ref == pytest.approx(72_500.0)
    assert default_params().with_value("p_ref", 10_000.0).p_ref == pytest.approx(61_000.0)


def test_constrain_clamps_every_field() -> None:
    wild = replace(default_params(), rho=4.0, f_maker=-1.0, w_brk_u=10_000, n0=0.0, seed=0)
    fixed = constrain(wild)
    assert fixed.rho == 1.0
    assert fixed.f_maker == 0.0
    assert fixed.w_brk_u == 1440
    assert fixed.n0 == 0.001
    assert fixed.seed == 1


def test_display_conversion_round_trips() -> None:
    spec = SPEC_BY_KEY["f_taker"]
    assert spec.to_display(0.003) == pytest.approx(0.30)
    assert spec.from_display(0.30) == pytest.approx(0.003)
    assert SPEC_BY_KEY["w_fast_d"].from_display(7.0) == 7


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(KeyError):
        default_params().with_value("nope", 1.0)
