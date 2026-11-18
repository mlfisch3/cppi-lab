"""Default parameters, ranges and metadata (PRD section 5, TSD section 6).

:class:`SimParams` is the immutable parameter struct handed to the
simulation controller.  :data:`PARAM_SPECS` describes every user-facing
control - label, symbol, default, range, step and display format - and is
the single source the sidebar, the popovers and the tests all read, so a
default can never drift between the UI and the engine.

Values are stored in natural units: rates and percentages as fractions
(``0.0015`` for 0.15 %), prices in USD, quantities in BTC, windows in bars.
``ParamSpec.scale`` converts to the unit shown in the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from typing import Any, Literal

ParamKind = Literal["float", "int", "bool", "datetime"]
ParamValue = float | int | bool

#: PRD section 5 default start: 2026-01-01 00:00 UTC, as epoch milliseconds.
DEFAULT_START_MS = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)

#: P_ref must stay at least this far above P_liq (PRD section 5 range).
P_REF_MIN_GAP_USD = 1_000.0

#: Monte Carlo ensemble size (PRD FR-3.3).
MC_SEEDS = 100


@dataclass(frozen=True)
class SimParams:
    """Every tunable input of one back-test (PRD section 5 plus FR-1.3)."""

    # Account & horizon
    start_ms: int = DEFAULT_START_MS
    duration_days: int = 30
    n0: float = 1.0
    c0: float = 0.0
    # CPPI floor & band
    p_liq: float = 60_000.0
    p_ref: float = 72_000.0
    dn_min_pct: float = 0.03
    # Fees & execution
    f_maker: float = 0.0015
    f_taker: float = 0.0030
    delta_slip: float = 0.0010
    rho: float = 0.25
    # Liquidity model
    lambda_liq: float = 2.50
    beta_vel: float = 1.80
    alpha_size: float = 1.20
    seed: int = 42
    mc_fan: bool = False
    # Decline signal
    w_fast_d: int = 5
    w_slow_d: int = 30
    w_atr_d: int = 15
    w_imb_d: int = 10
    w_mom_d: float = 0.60
    k_sell: float = 1.00
    # Growth signal
    w_fast_u: int = 15
    w_slow_u: int = 60
    w_atr_u: int = 60
    w_brk_u: int = 240
    tau_veto: float = 0.50
    # FR-1.3 historical price rescaling tool (not part of the section 5 table)
    normalize: bool = False
    anchor_price: float = 68_000.0

    @property
    def gamma(self) -> float:
        """Total emergency taker exit friction, ``f_taker + delta_slip``."""
        return self.f_taker + self.delta_slip

    @property
    def floor_value(self) -> float:
        """Hard net-value floor ``F = N0 * P_liq + c0``."""
        return self.n0 * self.p_liq + self.c0

    @property
    def dn_min(self) -> float:
        """Minimum rebalance quantum in BTC (a percentage of ``N0``)."""
        return self.dn_min_pct * self.n0

    def with_value(self, key: str, value: ParamValue) -> SimParams:
        """Return a copy with one field replaced and cross-field ranges enforced."""
        if key not in _FIELD_NAMES:
            raise KeyError(key)
        return constrain(_updated(self, {key: value}))


_FIELD_NAMES = frozenset(f.name for f in fields(SimParams))


def _updated(params: SimParams, updates: dict[str, ParamValue]) -> SimParams:
    """Copy ``params`` with some fields replaced (keys are validated by the caller)."""
    merged: dict[str, Any] = {**asdict(params), **updates}
    return SimParams(**merged)


@dataclass(frozen=True)
class ParamSpec:
    """UI and validation metadata for one :class:`SimParams` field."""

    key: str
    group: str
    label: str
    symbol: str
    kind: ParamKind
    default: ParamValue
    minimum: float
    maximum: float
    step: float
    decimals: int = 0
    scale: float = 1.0
    prefix: str = ""
    suffix: str = ""
    #: Identifier of the ``?`` popover entry in ``doc_registry.json``.
    popover: str = ""
    #: True for the second control of a paired row (e.g. the slow EMA window).
    paired_with: str = ""

    def to_display(self, value: ParamValue) -> float:
        """Convert a stored value to the unit shown in the UI."""
        return float(value) * self.scale

    def from_display(self, shown: float) -> ParamValue:
        """Convert a UI value back to the stored unit and type."""
        raw = shown / self.scale
        if self.kind == "int" or self.kind == "datetime":
            return round(raw)
        if self.kind == "bool":
            return bool(raw)
        return float(raw)

    def clamp(self, value: ParamValue) -> ParamValue:
        """Clamp a stored value into ``[minimum, maximum]``."""
        if self.kind == "bool":
            return bool(value)
        bounded = min(max(float(value), self.minimum), self.maximum)
        if self.kind == "int" or self.kind == "datetime":
            return round(bounded)
        return bounded

    @property
    def steps(self) -> int:
        """Number of slider positions minus one."""
        return max(1, round((self.maximum - self.minimum) / self.step))


GROUP_ACCOUNT = "Account & Horizon"
GROUP_FLOOR = "CPPI Floor & Band"
GROUP_FEES = "Fees & Execution"
GROUP_LIQUIDITY = "Liquidity Model"
GROUP_DECLINE = "Decline Signal"
GROUP_GROWTH = "Growth Signal"

GROUPS: tuple[str, ...] = (
    GROUP_ACCOUNT,
    GROUP_FLOOR,
    GROUP_FEES,
    GROUP_LIQUIDITY,
    GROUP_DECLINE,
    GROUP_GROWTH,
)

#: LaTeX shown next to the group title in the sidebar and documentation.
GROUP_SYMBOLS: dict[str, str] = {
    GROUP_DECLINE: r"S_{\text{down}}",
    GROUP_GROWTH: r"S_{\text{up}}",
}

_D = SimParams()

#: One entry per user control, in sidebar order.  The 25 rows of PRD section 5
#: map onto 27 entries because each "Fast / Slow EMA Windows" row holds two
#: values; the last two entries are the FR-1.3 rescaling tool.
PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "start_ms",
        GROUP_ACCOUNT,
        "Start Date & Time (UTC)",
        "t_0",
        "datetime",
        _D.start_ms,
        0.0,
        0.0,
        60_000.0,
        popover="param-start",
    ),
    ParamSpec(
        "duration_days",
        GROUP_ACCOUNT,
        "Simulation Duration (Days)",
        r"\Delta T",
        "int",
        _D.duration_days,
        1,
        365,
        1,
        suffix=" d",
        popover="param-duration",
    ),
    ParamSpec(
        "n0",
        GROUP_ACCOUNT,
        "Initial BTC Holdings",
        "N_0",
        "float",
        _D.n0,
        0.001,
        1000.0,
        0.05,
        decimals=4,
        suffix=" BTC",
        popover="param-n0",
    ),
    ParamSpec(
        "c0",
        GROUP_ACCOUNT,
        "Initial USD Cash",
        "c_0",
        "float",
        _D.c0,
        0.0,
        10_000_000.0,
        100.0,
        decimals=2,
        prefix="$",
        popover="param-c0",
    ),
    ParamSpec(
        "p_liq",
        GROUP_FLOOR,
        "Hard Liquidation Floor Price",
        r"P_{\text{liq}}",
        "float",
        _D.p_liq,
        1_000.0,
        250_000.0,
        250.0,
        decimals=2,
        prefix="$",
        popover="param-p-liq",
    ),
    ParamSpec(
        "p_ref",
        GROUP_FLOOR,
        "Upper Full-Allocation Reference",
        r"P_{\text{ref}}",
        "float",
        _D.p_ref,
        2_000.0,
        350_000.0,
        500.0,
        decimals=2,
        prefix="$",
        popover="param-p-ref",
    ),
    ParamSpec(
        "dn_min_pct",
        GROUP_FLOOR,
        "Minimum Rebalance Threshold",
        r"\Delta n_{\min}",
        "float",
        _D.dn_min_pct,
        0.001,
        0.25,
        0.0025,
        decimals=2,
        scale=100.0,
        suffix=" %",
        popover="param-dn-min",
    ),
    ParamSpec(
        "f_maker",
        GROUP_FEES,
        "Maker Fee Rate",
        r"f_{\text{maker}}",
        "float",
        _D.f_maker,
        0.0,
        0.01,
        0.0001,
        decimals=2,
        scale=100.0,
        suffix=" %",
        popover="param-f-maker",
    ),
    ParamSpec(
        "f_taker",
        GROUP_FEES,
        "Taker Fee Rate",
        r"f_{\text{taker}}",
        "float",
        _D.f_taker,
        0.0,
        0.02,
        0.0001,
        decimals=2,
        scale=100.0,
        suffix=" %",
        popover="param-f-taker",
    ),
    ParamSpec(
        "delta_slip",
        GROUP_FEES,
        "Safety Buffer Slippage Allowance",
        r"\delta_{\text{slip}}",
        "float",
        _D.delta_slip,
        0.0,
        0.02,
        0.0001,
        decimals=2,
        scale=100.0,
        suffix=" %",
        popover="param-delta-slip",
    ),
    ParamSpec(
        "rho",
        GROUP_FEES,
        "Risk-Aversion Parameter",
        r"\rho",
        "float",
        _D.rho,
        0.0,
        1.0,
        0.05,
        decimals=2,
        popover="param-rho",
    ),
    ParamSpec(
        "lambda_liq",
        GROUP_LIQUIDITY,
        "Base Liquidity Depth Factor",
        r"\lambda_{\text{liq}}",
        "float",
        _D.lambda_liq,
        0.10,
        10.0,
        0.10,
        decimals=2,
        popover="param-lambda-liq",
    ),
    ParamSpec(
        "beta_vel",
        GROUP_LIQUIDITY,
        "Adverse Price Velocity Penalty",
        r"\beta_{\text{vel}}",
        "float",
        _D.beta_vel,
        0.0,
        10.0,
        0.10,
        decimals=2,
        popover="param-beta-vel",
    ),
    ParamSpec(
        "alpha_size",
        GROUP_LIQUIDITY,
        "Order Size Impact Exponent",
        r"\alpha_{\text{size}}",
        "float",
        _D.alpha_size,
        0.10,
        5.0,
        0.10,
        decimals=2,
        popover="param-alpha-size",
    ),
    ParamSpec(
        "seed",
        GROUP_LIQUIDITY,
        "Uniform PRNG Seed",
        r"s_{\text{rng}}",
        "int",
        _D.seed,
        1,
        9_999_999,
        1,
        popover="param-seed",
    ),
    ParamSpec(
        "mc_fan",
        GROUP_LIQUIDITY,
        "Monte Carlo Fan Overlay (M = 100)",
        r"\text{MC}_{\text{fan}}",
        "bool",
        _D.mc_fan,
        0,
        1,
        1,
        popover="param-mc-fan",
    ),
    ParamSpec(
        "w_fast_d",
        GROUP_DECLINE,
        "Fast / Slow EMA Windows",
        "W_{f,d}",
        "int",
        _D.w_fast_d,
        1,
        60,
        1,
        suffix=" m",
        popover="param-ema-d",
    ),
    ParamSpec(
        "w_slow_d",
        GROUP_DECLINE,
        "Fast / Slow EMA Windows",
        "W_{s,d}",
        "int",
        _D.w_slow_d,
        10,
        240,
        1,
        suffix=" m",
        popover="param-ema-d",
        paired_with="w_fast_d",
    ),
    ParamSpec(
        "w_atr_d",
        GROUP_DECLINE,
        "Volatility ATR Window",
        r"W_{\text{atr},d}",
        "int",
        _D.w_atr_d,
        5,
        120,
        1,
        suffix=" m",
        popover="param-atr-d",
    ),
    ParamSpec(
        "w_imb_d",
        GROUP_DECLINE,
        "Trade Flow Imbalance Window",
        r"W_{\text{imb},d}",
        "int",
        _D.w_imb_d,
        2,
        60,
        1,
        suffix=" m",
        popover="param-imb-d",
    ),
    ParamSpec(
        "w_mom_d",
        GROUP_DECLINE,
        "Momentum vs. Imbalance Weight",
        "w_{m,d}",
        "float",
        _D.w_mom_d,
        0.0,
        1.0,
        0.05,
        decimals=2,
        popover="param-w-mom-d",
    ),
    ParamSpec(
        "k_sell",
        GROUP_DECLINE,
        "Decline Signal Gain",
        r"k_{\text{sell}}",
        "float",
        _D.k_sell,
        0.10,
        2.0,
        0.05,
        decimals=2,
        popover="param-k-sell",
    ),
    ParamSpec(
        "w_fast_u",
        GROUP_GROWTH,
        "Fast / Slow EMA Windows",
        "W_{f,u}",
        "int",
        _D.w_fast_u,
        5,
        120,
        5,
        suffix=" m",
        popover="param-ema-u",
    ),
    ParamSpec(
        "w_slow_u",
        GROUP_GROWTH,
        "Fast / Slow EMA Windows",
        "W_{s,u}",
        "int",
        _D.w_slow_u,
        30,
        480,
        5,
        suffix=" m",
        popover="param-ema-u",
        paired_with="w_fast_u",
    ),
    ParamSpec(
        "w_atr_u",
        GROUP_GROWTH,
        "Volatility ATR Window",
        r"W_{\text{atr},u}",
        "int",
        _D.w_atr_u,
        15,
        240,
        5,
        suffix=" m",
        popover="param-atr-u",
    ),
    ParamSpec(
        "w_brk_u",
        GROUP_GROWTH,
        "Range Breakout Window",
        r"W_{\text{brk},u}",
        "int",
        _D.w_brk_u,
        30,
        1440,
        15,
        suffix=" m",
        popover="param-brk-u",
    ),
    ParamSpec(
        "tau_veto",
        GROUP_GROWTH,
        "Decline Veto Gate Threshold",
        r"\tau_{\text{veto}}",
        "float",
        _D.tau_veto,
        0.10,
        1.0,
        0.05,
        decimals=2,
        popover="param-tau-veto",
    ),
    ParamSpec(
        "normalize",
        GROUP_ACCOUNT,
        "Normalize Start Price to Reference Anchor",
        "",
        "bool",
        _D.normalize,
        0,
        1,
        1,
        popover="param-normalize",
    ),
    ParamSpec(
        "anchor_price",
        GROUP_ACCOUNT,
        "Reference Anchor Price",
        r"P_{\text{anchor}}",
        "float",
        _D.anchor_price,
        1_000.0,
        350_000.0,
        250.0,
        decimals=2,
        prefix="$",
        popover="param-normalize",
        paired_with="normalize",
    ),
)

SPEC_BY_KEY: dict[str, ParamSpec] = {spec.key: spec for spec in PARAM_SPECS}


def default_params() -> SimParams:
    """A fresh parameter struct holding the PRD section 5 defaults (FR-2.1)."""
    return SimParams()


def p_ref_minimum(p_liq: float) -> float:
    """Lower bound of ``P_ref`` given ``P_liq`` (PRD section 5: ``P_liq + $1,000``)."""
    return p_liq + P_REF_MIN_GAP_USD


def constrain(params: SimParams) -> SimParams:
    """Clamp every field into its range and apply the ``P_ref`` coupling."""
    updates: dict[str, ParamValue] = {}
    for spec in PARAM_SPECS:
        value: ParamValue = getattr(params, spec.key)
        if spec.kind == "datetime":
            continue  # bounded by the active dataset, enforced by the controller
        clamped = spec.clamp(value)
        if clamped != value:
            updates[spec.key] = clamped
    p_liq = float(updates.get("p_liq", params.p_liq))
    p_ref = float(updates.get("p_ref", params.p_ref))
    if p_ref < p_ref_minimum(p_liq):
        updates["p_ref"] = p_ref_minimum(p_liq)
    return _updated(params, updates) if updates else params
