"""CPPI state machine, rho router and stochastic fill model (TSD sections 2.3-3.2).

One Numba-compiled loop, :func:`simulate_core`, walks the bars in the strict
intra-bar order of TSD section 3.1:

1. recompute the buffered floor ``P_floor,t`` from the previous end state;
2. fire the emergency taker liquidation if the bar's low reaches it;
3. resolve the passive maker order pegged at the previous close (and, after a
   maker *buy* fill, re-check the trigger against the post-fill state - 3b);
4. evaluate the dual-envelope control law at the close and route any
   rebalance between an immediate taker child and a pegged maker child;
5. commit the telemetry row.

The same loop serves the single trajectory (full telemetry) and the Monte
Carlo ensemble (equity only), so the two can never disagree.

Three guards go beyond the literal TSD text; each is required for the PRD's
primary invariant ``V_t >= F`` and is derived in the bundled documentation
and recorded in ``docs/DECISIONS.md``:

* **Rebuy cap** - a buy is trimmed so that the position it creates is still
  at or below the sell ceiling evaluated on the *post-trade* state.  A buy
  funded by more than the available cushion would otherwise lift the trigger
  above the market.
* **Post-fill trigger re-check (step 3b)** - a maker buy fills on the way
  down; if the same bar's low reaches the trigger of the position just
  acquired, the emergency exit fires in that bar.
* **Maker dominance** - when ``f_maker >= f_taker + delta_slip`` a passive
  order is both dearer and uncertain, so every order is routed as a taker.

The kernel is compiled *without* ``fastmath``: the PRD demands IEEE-754
``float64`` accounting, the telemetry uses NaN as a sentinel, and the result
must be bit-identical to the readable reference implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketWindow
from cppi_lab.engine.signals import Signals

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Telemetry row layout, exactly as listed in TSD section 3.2 (26 fields).
TELEMETRY_DTYPE = np.dtype(
    [
        ("bar_idx", np.int32),  # 0: integer index t
        ("ts_utc_ms", np.int64),  # 1: epoch timestamp (ms)
        ("open", np.float64),  # 2: bar open ($)
        ("high", np.float64),  # 3: bar high ($)
        ("low", np.float64),  # 4: bar low ($)
        ("close", np.float64),  # 5: bar close ($)
        ("buy_vol", np.float64),  # 6: buyer-taker volume (BTC)
        ("sell_vol", np.float64),  # 7: seller-taker volume (BTC)
        ("s_down", np.float64),  # 8: decline signal in [0, 1]
        ("s_up", np.float64),  # 9: growth signal in [0, 1]
        ("p_floor", np.float64),  # 10: liquidation level + safety buffer ($)
        ("safety_buf_usd", np.float64),  # 11: P_floor - P_liq ($)
        ("d_norm", np.float64),  # 12: normalised distance d_t in [0, 1]
        ("n_max", np.float64),  # 13: max recoverable BTC
        ("n_ceil", np.float64),  # 14: upper sell ceiling (BTC)
        ("n_rebuy", np.float64),  # 15: lower rebuy floor (BTC)
        ("btc_held", np.float64),  # 16: BTC held n_t
        ("cash_usd", np.float64),  # 17: USD cash c_t
        ("net_equity_usd", np.float64),  # 18: net liquidation value V_t ($)
        ("cushion_usd", np.float64),  # 19: active risk cushion max(0, V_t - F) ($)
        ("event_code", np.uint8),  # 20: 0..7 execution/miss event enum
        ("exec_qty_btc", np.float64),  # 21: filled or missed order size (BTC)
        ("exec_px_usd", np.float64),  # 22: realised fill price or missed limit ($)
        ("fee_paid_usd", np.float64),  # 23: transaction fee + slippage paid ($)
        ("p_fill_prob", np.float64),  # 24: maker fill probability P_fill,t
        ("u_rand_draw", np.float64),  # 25: uniform random draw U_t
    ]
)

# Event codes (TSD section 3.1).
EVENT_NONE = 0
EVENT_MAKER_SELL = 1
EVENT_TAKER_SELL = 2
EVENT_MAKER_BUY = 3
EVENT_TAKER_BUY = 4
EVENT_MISSED_MAKER_SELL = 5
EVENT_MISSED_MAKER_BUY = 6
EVENT_HARD_BUFFER_LIQUIDATION = 7

EVENT_NAMES: tuple[str, ...] = (
    "NONE",
    "MAKER_SELL",
    "TAKER_SELL",
    "MAKER_BUY",
    "TAKER_BUY",
    "MISSED_MAKER_SELL",
    "MISSED_MAKER_BUY",
    "HARD_BUFFER_LIQUIDATION",
)


def event_bit(code: int) -> int:
    """Bit of event ``code`` (1..7) in the per-bar flags mask."""
    return 1 << (code - 1)


FLAG_EXECUTED = (
    event_bit(EVENT_MAKER_SELL)
    | event_bit(EVENT_TAKER_SELL)
    | event_bit(EVENT_MAKER_BUY)
    | event_bit(EVENT_TAKER_BUY)
    | event_bit(EVENT_HARD_BUFFER_LIQUIDATION)
)
FLAG_MISSED = event_bit(EVENT_MISSED_MAKER_SELL) | event_bit(EVENT_MISSED_MAKER_BUY)
FLAG_LIQUIDATION = event_bit(EVENT_HARD_BUFFER_LIQUIDATION)

#: Holdings at or below this are treated as flat (TSD section 2.3.1).
QTY_EPS = 1e-8
#: Sells ignore the minimum quantum inside this band above the trigger (TSD 2.3.2).
NEAR_FLOOR_ZONE_USD = 250.0
#: Buys are vetoed while the decline signal is at or above this (TSD 2.3.2).
BUY_DECLINE_VETO = 0.25
#: Tolerance of the conservation assertion ``V_t >= F - tol`` (PRD section 6).
FLOOR_TOLERANCE_USD = 1e-6
#: Bisection steps of the rebuy cap: the bracket shrinks to 2**-40 of the order.
CAP_ITERATIONS = 40

# Indices into the statistics vector returned by the kernel.
STAT_MAKER_SELLS = 0
STAT_TAKER_SELLS = 1
STAT_MAKER_BUYS = 2
STAT_TAKER_BUYS = 3
STAT_MISSED_SELLS = 4
STAT_MISSED_BUYS = 5
STAT_LIQUIDATIONS = 6
STAT_FIRST_LIQUIDATION_BAR = 7
STAT_GAP_BAR = 8
STAT_FLOOR_BREACHES = 9
STAT_MIN_MARGIN_USD = 10
STAT_MAKER_FEES_USD = 11
STAT_TAKER_COSTS_USD = 12
STAT_MAKER_VOLUME_BTC = 13
STAT_TAKER_VOLUME_BTC = 14
STAT_FILL_PROB_SUM = 15
STAT_FINAL_EQUITY_USD = 16
N_STATS = 17


@njit(cache=True, nogil=True)
def trigger_price(n: float, c: float, floor_value: float, base_floor: float, keep: float) -> float:
    """Buffered liquidation trigger ``P_floor`` (TSD section 2.3.1).

    ``keep`` is ``1 - gamma``; ``base_floor`` is ``P_liq / (1 - gamma)``.
    """
    if n > QTY_EPS:
        need = floor_value - c
        if need < 0.0:
            need = 0.0
        required = need / (n * keep)
        if required > base_floor:
            return required
    return base_floor


@njit(cache=True, nogil=True)
def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(cache=True, nogil=True)
def sell_ceiling(n_max: float, d: float, k_sell: float, s_down: float) -> float:
    """Upper sell ceiling ``n_ceil`` with its bracket floored at zero."""
    bracket = 1.0 - k_sell * (1.0 - d) * s_down
    if bracket < 0.0:
        bracket = 0.0
    return n_max * d * bracket


@njit(cache=True, nogil=True)
def _post_buy_slack(
    q: float,
    n: float,
    c: float,
    price: float,
    unit_cost: float,
    floor_value: float,
    base_floor: float,
    keep: float,
    p_ref: float,
    k_sell: float,
    s_down: float,
    f_eff: float,
) -> float:
    """``n_ceil' - n'`` on the state reached by buying ``q`` at ``unit_cost``.

    Non-negative means the purchase leaves the position inside its own
    envelope; negative cash is reported as infeasible.
    """
    c_post = c - q * unit_cost
    if c_post < 0.0:
        return -1.0
    n_post = n + q
    floor_post = trigger_price(n_post, c_post, floor_value, base_floor, keep)
    span = p_ref - floor_post
    if span < 1.0:
        span = 1.0
    d_post = _clip01((price - floor_post) / span)
    n_max_post = n_post + c_post / (price * (1.0 + f_eff))
    slack: float = sell_ceiling(n_max_post, d_post, k_sell, s_down) - n_post
    return slack


@njit(cache=True, nogil=True)
def rebuy_cap(
    target: float,
    minimum: float,
    n: float,
    c: float,
    price: float,
    unit_cost: float,
    floor_value: float,
    base_floor: float,
    keep: float,
    p_ref: float,
    k_sell: float,
    s_down: float,
    f_eff: float,
) -> float:
    """Largest buy ``q <= target`` that stays inside the post-trade envelope.

    Returns ``0.0`` when not even ``minimum`` (the smallest order the control
    law would place) is feasible, which is the common case for a cushion-
    starved account and costs two evaluations.  Otherwise the feasible set is
    an interval, and a bisection between a feasible lower end and an
    infeasible upper end converges onto its edge; the returned quantity is
    always feasible.
    """
    hi = target
    affordable = c / unit_cost
    if affordable < hi:
        hi = affordable
    if hi < minimum or hi <= 0.0:
        return 0.0
    if (
        _post_buy_slack(
            hi, n, c, price, unit_cost, floor_value, base_floor, keep, p_ref, k_sell, s_down, f_eff
        )
        >= 0.0
    ):
        return hi
    lo = minimum
    if (
        _post_buy_slack(
            lo, n, c, price, unit_cost, floor_value, base_floor, keep, p_ref, k_sell, s_down, f_eff
        )
        < 0.0
    ):
        return 0.0
    for _ in range(CAP_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if (
            _post_buy_slack(
                mid,
                n,
                c,
                price,
                unit_cost,
                floor_value,
                base_floor,
                keep,
                p_ref,
                k_sell,
                s_down,
                f_eff,
            )
            >= 0.0
        ):
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True)
def fill_probability(
    is_sell: bool,
    limit_px: float,
    qty: float,
    open_: float,
    high: float,
    low: float,
    close: float,
    opposing_vol: float,
    atr: float,
    lambda_liq: float,
    beta_vel: float,
    alpha_size: float,
) -> float:
    """Three-factor maker fill probability ``P_fill,t`` (TSD section 2.5).

    Returns exactly ``0.0`` when the bar never traded through the limit.
    """
    if is_sell:
        penetration = high - limit_px
        adverse = open_ - close
    else:
        penetration = limit_px - low
        adverse = close - open_
    if penetration <= 0.0:
        return 0.0
    if adverse < 0.0:
        adverse = 0.0
    phi_pen = 1.0 - math.exp(-lambda_liq * (penetration / atr))
    phi_vel = math.exp(-beta_vel * (adverse / atr))
    absorb = 0.01 * opposing_vol
    if absorb < 1e-8:
        absorb = 1e-8
    phi_size = 1.0 / (1.0 + (qty / absorb) ** alpha_size)
    probability: float = _clip01(phi_pen * phi_vel * phi_size)
    return probability


@njit(cache=True, nogil=True)
def execution_urgency(is_sell: bool, d: float, s_down: float, s_up: float, m_miss: int) -> float:
    """Normalised execution urgency ``u_t`` (TSD section 2.4)."""
    if is_sell:
        return 1.0 - d * (1.0 - s_down) * 0.5**m_miss
    return s_up * (1.0 - 0.5 ** (m_miss + 1))


@njit(cache=True, nogil=True)
def taker_fraction(rho: float, urgency: float) -> float:
    """Taker routing fraction ``theta_taker(rho)`` (TSD section 2.4)."""
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return 1.0
    if urgency >= 1.0 - rho:
        return 1.0
    return rho


@njit(cache=True, nogil=True)
def route_order_by_rho(
    is_sell: bool,
    dn: float,
    d: float,
    s_down: float,
    s_up: float,
    m_miss: int,
    rho: float,
    maker_dominated: bool,
) -> tuple[float, float]:
    """Split a rebalance of ``dn`` BTC into ``(taker, maker)`` child quantities.

    The taker child is ``theta * dn`` and the maker child is the remainder, so
    the two always add up to the order exactly.
    """
    theta = taker_fraction(rho, execution_urgency(is_sell, d, s_down, s_up, m_miss))
    if maker_dominated:
        theta = 1.0
    q_taker = theta * dn
    return q_taker, dn - q_taker


@njit(cache=True, nogil=True)
def simulate_core(
    ts_ms: IntArray,
    open_: FloatArray,
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    buy_vol: FloatArray,
    sell_vol: FloatArray,
    s_down: FloatArray,
    s_up: FloatArray,
    atr: FloatArray,
    u: FloatArray,
    n0: float,
    c0: float,
    p_liq: float,
    p_ref: float,
    dn_min: float,
    f_maker: float,
    f_taker: float,
    delta_slip: float,
    rho: float,
    lambda_liq: float,
    beta_vel: float,
    alpha_size: float,
    k_sell: float,
    telemetry: np.ndarray,
    flags: NDArray[np.uint8],
    write_rows: bool,
    equity: FloatArray,
    stride: int,
    stats: FloatArray,
) -> None:
    """Run the state machine over one window.

    ``telemetry`` and ``flags`` are filled when ``write_rows`` is true.
    ``equity[t // stride]`` receives ``V_t`` on every ``stride``-th bar.
    ``stats`` (length :data:`N_STATS`) receives the run aggregates.
    """
    n_bars = close.shape[0]
    gamma = f_taker + delta_slip
    keep = 1.0 - gamma
    floor_value = n0 * p_liq + c0
    base_floor = p_liq / keep
    f_eff = (1.0 - rho) * f_maker + rho * f_taker
    maker_dominated = f_maker >= gamma
    worst_buy_rate = gamma
    if f_maker > worst_buy_rate:
        worst_buy_rate = f_maker

    n = n0
    c = c0
    pend_side = 0  # +1 resting maker sell, -1 resting maker buy, 0 none
    pend_qty = 0.0
    pend_px = 0.0
    m_miss = 0
    chain_side = 0  # side of the rebalance the miss counter belongs to

    for k in range(N_STATS):
        stats[k] = 0.0
    stats[STAT_FIRST_LIQUIDATION_BAR] = -1.0
    stats[STAT_GAP_BAR] = -1.0
    stats[STAT_MIN_MARGIN_USD] = np.inf
    gapped = False

    for t in range(n_bars):
        o_t = open_[t]
        h_t = high[t]
        l_t = low[t]
        c_t = close[t]
        sd = s_down[t]
        su = s_up[t]

        event = 0
        event_qty = 0.0
        event_px = np.nan
        fees = 0.0
        p_fill = np.nan
        bits = 0
        liquidated = False

        # Step 1 - active buffered floor from the previous end state.
        p_floor = trigger_price(n, c, floor_value, base_floor, keep)

        # Step 2 - intra-bar hard buffered floor trigger.
        if n > QTY_EPS and l_t <= p_floor:
            p_trig = p_floor
            if o_t < p_floor:
                p_trig = o_t
                if not gapped:
                    gapped = True
                    stats[STAT_GAP_BAR] = t
            fees += n * p_trig * gamma
            stats[STAT_TAKER_COSTS_USD] += n * p_trig * gamma
            stats[STAT_TAKER_VOLUME_BTC] += n
            event = 7
            event_qty = n
            event_px = p_trig * (1.0 - delta_slip)
            c = c + n * p_trig * keep
            n = 0.0
            pend_side = 0
            pend_qty = 0.0
            m_miss = 0
            chain_side = 0
            liquidated = True
            bits |= 64
            stats[STAT_LIQUIDATIONS] += 1.0
            if stats[STAT_FIRST_LIQUIDATION_BAR] < 0.0:
                stats[STAT_FIRST_LIQUIDATION_BAR] = t

        # Step 3 - resolve the passive maker order pegged at the previous close.
        if (not liquidated) and pend_side != 0:
            is_sell = pend_side == 1
            opposing = buy_vol[t] if is_sell else sell_vol[t]
            p_fill = fill_probability(
                is_sell,
                pend_px,
                pend_qty,
                o_t,
                h_t,
                l_t,
                c_t,
                opposing,
                atr[t],
                lambda_liq,
                beta_vel,
                alpha_size,
            )
            stats[STAT_FILL_PROB_SUM] += p_fill
            if p_fill > 0.0 and u[t] <= p_fill:
                if is_sell:
                    q = pend_qty
                    if q > n:
                        q = n
                    c = c + q * pend_px * (1.0 - f_maker)
                    n = n - q
                    event = 1
                    bits |= 1
                    stats[STAT_MAKER_SELLS] += 1.0
                else:
                    q = pend_qty
                    affordable = c / (pend_px * (1.0 + f_maker))
                    if q > affordable:
                        q = affordable
                    c = c - q * pend_px * (1.0 + f_maker)
                    n = n + q
                    event = 3
                    bits |= 4
                    stats[STAT_MAKER_BUYS] += 1.0
                fees += q * pend_px * f_maker
                stats[STAT_MAKER_FEES_USD] += q * pend_px * f_maker
                stats[STAT_MAKER_VOLUME_BTC] += q
                event_qty = q
                event_px = pend_px
                m_miss = 0
                if not is_sell:
                    # Step 3b - the bid filled on the way down; the bar's low
                    # comes at or after the fill, so re-check the trigger of
                    # the position just acquired.
                    post_floor = trigger_price(n, c, floor_value, base_floor, keep)
                    if n > QTY_EPS and l_t <= post_floor:
                        p_trig = post_floor
                        if pend_px < post_floor:
                            p_trig = pend_px
                        fees += n * p_trig * gamma
                        stats[STAT_TAKER_COSTS_USD] += n * p_trig * gamma
                        stats[STAT_TAKER_VOLUME_BTC] += n
                        event = 7
                        event_qty = n
                        event_px = p_trig * (1.0 - delta_slip)
                        c = c + n * p_trig * keep
                        n = 0.0
                        chain_side = 0
                        liquidated = True
                        bits |= 64
                        stats[STAT_LIQUIDATIONS] += 1.0
                        if stats[STAT_FIRST_LIQUIDATION_BAR] < 0.0:
                            stats[STAT_FIRST_LIQUIDATION_BAR] = t
            else:
                if is_sell:
                    event = 5
                    bits |= 16
                    stats[STAT_MISSED_SELLS] += 1.0
                else:
                    event = 6
                    bits |= 32
                    stats[STAT_MISSED_BUYS] += 1.0
                event_qty = pend_qty
                event_px = pend_px
                m_miss += 1
            pend_side = 0
            pend_qty = 0.0

        # Step 4 - dual-envelope control law at the bar close.
        n_max = n + c / (c_t * (1.0 + f_eff))
        span = p_ref - p_floor
        if span < 1.0:
            span = 1.0
        d = _clip01((c_t - p_floor) / span)
        n_ceil = sell_ceiling(n_max, d, k_sell, sd)
        n_rebuy = n_max * d * su
        if n_rebuy > n_ceil:
            n_rebuy = n_ceil

        side = 0
        dn = 0.0
        dn_eff = dn_min
        if c_t <= p_floor + NEAR_FLOOR_ZONE_USD:
            dn_eff = 0.0
        if n > n_ceil and (n - n_ceil) >= dn_eff:
            side = 1
            dn = n - n_ceil
        elif n < n_rebuy and sd < BUY_DECLINE_VETO and (n_rebuy - n) >= dn_min:
            dn = rebuy_cap(
                n_rebuy - n,
                dn_min,
                n,
                c,
                c_t,
                c_t * (1.0 + worst_buy_rate),
                floor_value,
                base_floor,
                keep,
                p_ref,
                k_sell,
                sd,
                f_eff,
            )
            if dn >= dn_min:
                side = -1

        if side != 0 and dn > QTY_EPS:
            if side != chain_side:
                m_miss = 0
                chain_side = side
            q_taker, q_maker = route_order_by_rho(
                side == 1, dn, d, sd, su, m_miss, rho, maker_dominated
            )
            if q_taker > QTY_EPS:
                if side == 1:
                    if q_taker > n:
                        q_taker = n
                    c = c + q_taker * c_t * keep
                    n = n - q_taker
                    code = 2
                    bits |= 2
                    stats[STAT_TAKER_SELLS] += 1.0
                    px = c_t * (1.0 - delta_slip)
                else:
                    c = c - q_taker * c_t * (1.0 + gamma)
                    n = n + q_taker
                    code = 4
                    bits |= 8
                    stats[STAT_TAKER_BUYS] += 1.0
                    px = c_t * (1.0 + delta_slip)
                fees += q_taker * c_t * gamma
                stats[STAT_TAKER_COSTS_USD] += q_taker * c_t * gamma
                stats[STAT_TAKER_VOLUME_BTC] += q_taker
                if event != 7:
                    event = code
                    event_qty = q_taker
                    event_px = px
            if q_maker > QTY_EPS:
                pend_side = side
                pend_qty = q_maker
                pend_px = c_t
        else:
            m_miss = 0
            chain_side = 0

        # Step 5 - commit the end-of-bar state.
        v_t = c + n * c_t * keep
        margin = v_t - floor_value
        if not gapped:
            if margin < stats[STAT_MIN_MARGIN_USD]:
                stats[STAT_MIN_MARGIN_USD] = margin
            if margin < -FLOOR_TOLERANCE_USD:
                stats[STAT_FLOOR_BREACHES] += 1.0
        if t % stride == 0:
            equity[t // stride] = v_t
        if write_rows:
            row = telemetry[t]
            row.bar_idx = t
            row.ts_utc_ms = ts_ms[t]
            row.open = o_t
            row.high = h_t
            row.low = l_t
            row.close = c_t
            row.buy_vol = buy_vol[t]
            row.sell_vol = sell_vol[t]
            row.s_down = sd
            row.s_up = su
            row.p_floor = p_floor
            row.safety_buf_usd = p_floor - p_liq
            row.d_norm = d
            row.n_max = n_max
            row.n_ceil = n_ceil
            row.n_rebuy = n_rebuy
            row.btc_held = n
            row.cash_usd = c
            row.net_equity_usd = v_t
            row.cushion_usd = margin if margin > 0.0 else 0.0
            row.event_code = event
            row.exec_qty_btc = event_qty
            row.exec_px_usd = event_px
            row.fee_paid_usd = fees
            row.p_fill_prob = p_fill
            row.u_rand_draw = u[t]
            flags[t] = bits

    if not np.isfinite(stats[STAT_MIN_MARGIN_USD]):
        stats[STAT_MIN_MARGIN_USD] = 0.0
    if n_bars > 0:
        stats[STAT_FINAL_EQUITY_USD] = c + n * close[n_bars - 1] * keep


def uniform_draws(seed: int, n_bars: int) -> FloatArray:
    """The per-bar uniform variates ``U_t`` for one seed (TSD section 2.5).

    ``np.random.Generator(np.random.PCG64(seed))`` is platform independent, so
    a seed reproduces the same trajectory on every machine.
    """
    return np.random.Generator(np.random.PCG64(int(seed))).random(n_bars)


@dataclass(frozen=True)
class SimStats:
    """Aggregates of one trajectory, counted losslessly inside the kernel."""

    maker_sells: int
    taker_sells: int
    maker_buys: int
    taker_buys: int
    missed_sells: int
    missed_buys: int
    liquidations: int
    #: Bar of the first hard-buffer liquidation, or -1.
    first_liquidation_bar: int
    #: Bar where the open gapped below the trigger, or -1.
    gap_bar: int
    #: Bars with ``V_t < F - 1e-6`` before any gap; the conservation assertion.
    floor_breaches: int
    #: Smallest ``V_t - F`` seen before any gap.
    min_margin_usd: float
    maker_fees_usd: float
    #: Taker fees plus slippage, including emergency liquidations.
    taker_costs_usd: float
    maker_volume_btc: float
    taker_volume_btc: float
    mean_fill_prob: float
    final_equity_usd: float

    @property
    def maker_fills(self) -> int:
        return self.maker_sells + self.maker_buys

    @property
    def taker_fills(self) -> int:
        return self.taker_sells + self.taker_buys

    @property
    def missed(self) -> int:
        return self.missed_sells + self.missed_buys

    @property
    def maker_attempts(self) -> int:
        return self.maker_fills + self.missed

    @property
    def total_costs_usd(self) -> float:
        return self.maker_fees_usd + self.taker_costs_usd

    @property
    def gapped(self) -> bool:
        return self.gap_bar >= 0


def stats_from_vector(vector: FloatArray) -> SimStats:
    """Wrap the raw statistics vector written by :func:`simulate_core`."""
    attempts = (
        vector[STAT_MAKER_SELLS]
        + vector[STAT_MAKER_BUYS]
        + vector[STAT_MISSED_SELLS]
        + vector[STAT_MISSED_BUYS]
    )
    return SimStats(
        maker_sells=int(vector[STAT_MAKER_SELLS]),
        taker_sells=int(vector[STAT_TAKER_SELLS]),
        maker_buys=int(vector[STAT_MAKER_BUYS]),
        taker_buys=int(vector[STAT_TAKER_BUYS]),
        missed_sells=int(vector[STAT_MISSED_SELLS]),
        missed_buys=int(vector[STAT_MISSED_BUYS]),
        liquidations=int(vector[STAT_LIQUIDATIONS]),
        first_liquidation_bar=int(vector[STAT_FIRST_LIQUIDATION_BAR]),
        gap_bar=int(vector[STAT_GAP_BAR]),
        floor_breaches=int(vector[STAT_FLOOR_BREACHES]),
        min_margin_usd=float(vector[STAT_MIN_MARGIN_USD]),
        maker_fees_usd=float(vector[STAT_MAKER_FEES_USD]),
        taker_costs_usd=float(vector[STAT_TAKER_COSTS_USD]),
        maker_volume_btc=float(vector[STAT_MAKER_VOLUME_BTC]),
        taker_volume_btc=float(vector[STAT_TAKER_VOLUME_BTC]),
        mean_fill_prob=float(vector[STAT_FILL_PROB_SUM] / attempts) if attempts > 0 else 0.0,
        final_equity_usd=float(vector[STAT_FINAL_EQUITY_USD]),
    )


@dataclass(frozen=True)
class SimResult:
    """Output of one single-trajectory back-test."""

    telemetry: np.ndarray
    #: Per-bar bit mask of every event that occurred (bit ``code - 1``).
    flags: NDArray[np.uint8]
    stats: SimStats
    floor_value: float
    params: SimParams

    @property
    def n_bars(self) -> int:
        return int(self.telemetry.shape[0])


def kernel_args(params: SimParams) -> tuple[float, ...]:
    """The scalar strategy arguments of :func:`simulate_core`, in call order."""
    return (
        float(params.n0),
        float(params.c0),
        float(params.p_liq),
        float(params.p_ref),
        float(params.dn_min),
        float(params.f_maker),
        float(params.f_taker),
        float(params.delta_slip),
        float(params.rho),
        float(params.lambda_liq),
        float(params.beta_vel),
        float(params.alpha_size),
        float(params.k_sell),
    )


def run_simulation(
    window: MarketWindow,
    signals: Signals,
    params: SimParams,
    u: FloatArray | None = None,
) -> SimResult:
    """Back-test one trajectory and return its full telemetry."""
    n_bars = window.n
    draws = uniform_draws(params.seed, n_bars) if u is None else u
    if draws.shape[0] != n_bars:
        raise ValueError("the uniform-variate vector must have one entry per bar")
    telemetry = np.zeros(n_bars, dtype=TELEMETRY_DTYPE)
    flags = np.zeros(n_bars, dtype=np.uint8)
    equity = np.empty(n_bars, dtype=np.float64)
    stats = np.empty(N_STATS, dtype=np.float64)
    simulate_core(
        window.ts_ms,
        window.open,
        window.high,
        window.low,
        window.close,
        window.buy_vol,
        window.sell_vol,
        signals.s_down,
        signals.s_up,
        signals.atr,
        draws,
        *kernel_args(params),
        telemetry,
        flags,
        True,
        equity,
        1,
        stats,
    )
    return SimResult(
        telemetry=telemetry,
        flags=flags,
        stats=stats_from_vector(stats),
        floor_value=params.floor_value,
        params=params,
    )
