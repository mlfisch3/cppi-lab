"""Readable plain-Python reference of the TSD state machine and signals.

Nothing here is compiled or vectorised.  It is written directly from TSD
sections 2.2-3.1 (plus the guards recorded in ``docs/DECISIONS.md``) as a
small object with one method per step, so a reader can check it against the
specification line by line.  The test-suite then requires the Numba kernel to
reproduce it exactly, field for field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketWindow

NAN = float("nan")


# --------------------------------------------------------------------------
# Signals (TSD section 2.2)
# --------------------------------------------------------------------------


def clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def ref_true_range(high: list[float], low: list[float], close: list[float]) -> list[float]:
    out = [high[0] - low[0]]
    for t in range(1, len(high)):
        out.append(max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1])))
    return out


def ref_ema(x: list[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    out = [x[0]]
    for t in range(1, len(x)):
        out.append(alpha * x[t] + (1.0 - alpha) * out[t - 1])
    return out


def ref_rolling_sum(x: list[float], window: int) -> list[float]:
    out = []
    for t in range(len(x)):
        total = 0.0
        for k in range(window):
            if t - k < 0:
                break
            total += x[t - k]
        out.append(total)
    return out


def ref_rolling_max(x: list[float], window: int) -> list[float]:
    return [max(x[max(0, t - window + 1) : t + 1]) for t in range(len(x))]


def ref_rolling_min(x: list[float], window: int) -> list[float]:
    return [min(x[max(0, t - window + 1) : t + 1]) for t in range(len(x))]


@dataclass
class RefSignals:
    s_down: list[float]
    s_up: list[float]
    atr: list[float]
    m_down: list[float]
    i_down: list[float]
    m_up: list[float]
    b_up: list[float]
    g_veto: list[float]


def ref_signals(window: MarketWindow, p: SimParams) -> RefSignals:
    high, low, close = window.high.tolist(), window.low.tolist(), window.close.tolist()
    buy, sell = window.buy_vol.tolist(), window.sell_vol.tolist()
    tr = ref_true_range(high, low, close)
    count = len(close)

    fast_d, slow_d = ref_ema(close, p.w_fast_d), ref_ema(close, p.w_slow_d)
    atr_d = [max(a, 1.0) for a in ref_ema(tr, p.w_atr_d)]
    sum_sell, sum_buy = ref_rolling_sum(sell, p.w_imb_d), ref_rolling_sum(buy, p.w_imb_d)
    m_down = [clip01((slow_d[t] - fast_d[t]) / (1.5 * atr_d[t])) for t in range(count)]
    i_down = [
        clip01((sum_sell[t] - sum_buy[t]) / (0.40 * max(sum_sell[t] + sum_buy[t], 1e-8)))
        for t in range(count)
    ]
    s_down = [clip01(p.w_mom_d * m_down[t] + (1.0 - p.w_mom_d) * i_down[t]) for t in range(count)]

    fast_u, slow_u = ref_ema(close, p.w_fast_u), ref_ema(close, p.w_slow_u)
    atr_u = ref_ema(tr, p.w_atr_u)
    top, bottom = ref_rolling_max(high, p.w_brk_u), ref_rolling_min(low, p.w_brk_u)
    m_up = [clip01((fast_u[t] - slow_u[t]) / (2.0 * max(atr_u[t], 1.0))) for t in range(count)]
    b_up = []
    for t in range(count):
        mid = 0.5 * (top[t] + bottom[t])
        half = max(0.5 * (top[t] - bottom[t]), 1.0)
        b_up.append(clip01((close[t] - mid) / half))
    g_veto = [max(0.0, 1.0 - s_down[t] / p.tau_veto) for t in range(count)]
    s_up = [clip01((0.50 * m_up[t] + 0.50 * b_up[t]) * g_veto[t]) for t in range(count)]
    return RefSignals(s_down, s_up, atr_d, m_down, i_down, m_up, b_up, g_veto)


# --------------------------------------------------------------------------
# State machine (TSD sections 2.3-3.1)
# --------------------------------------------------------------------------

SELL, BUY = 1, -1


@dataclass
class Pending:
    side: int
    qty: float
    limit: float


@dataclass
class RefRow:
    """One telemetry row; field names follow ``TELEMETRY_DTYPE``."""

    bar_idx: int
    ts_utc_ms: int
    open: float
    high: float
    low: float
    close: float
    buy_vol: float
    sell_vol: float
    s_down: float
    s_up: float
    p_floor: float
    safety_buf_usd: float
    d_norm: float
    n_max: float
    n_ceil: float
    n_rebuy: float
    btc_held: float
    cash_usd: float
    net_equity_usd: float
    cushion_usd: float
    event_code: int
    exec_qty_btc: float
    exec_px_usd: float
    fee_paid_usd: float
    p_fill_prob: float
    u_rand_draw: float
    flags: int


@dataclass
class ReferenceMachine:
    """The CPPI strategy as a step-by-step interpreter."""

    p: SimParams
    n: float = 0.0
    c: float = 0.0
    pending: Pending | None = None
    m_miss: int = 0
    chain_side: int = 0
    gap_bar: int = -1
    counts: dict[int, int] = field(default_factory=lambda: dict.fromkeys(range(1, 8), 0))

    def __post_init__(self) -> None:
        p = self.p
        self.n, self.c = p.n0, p.c0
        self.gamma = p.f_taker + p.delta_slip
        self.keep = 1.0 - self.gamma
        self.floor_value = p.n0 * p.p_liq + p.c0
        self.base_floor = p.p_liq / self.keep
        self.f_eff = (1.0 - p.rho) * p.f_maker + p.rho * p.f_taker
        self.maker_dominated = p.f_maker >= self.gamma
        self.dn_min = p.dn_min_pct * p.n0

    # -- section 2.3.1 ------------------------------------------------------
    def trigger(self, n: float, c: float) -> float:
        if n > 1e-8:
            required = max(0.0, self.floor_value - c) / (n * self.keep)
            return max(self.base_floor, required)
        return self.base_floor

    # -- section 2.3.2 ------------------------------------------------------
    def distance(self, price: float, floor: float) -> float:
        return clip01((price - floor) / max(self.p.p_ref - floor, 1.0))

    def ceiling(self, n_max: float, d: float, s_down: float) -> float:
        bracket = max(0.0, 1.0 - self.p.k_sell * (1.0 - d) * s_down)
        return n_max * d * bracket

    # -- guard: rebuy cap ---------------------------------------------------
    def post_buy_slack(self, q: float, price: float, unit_cost: float, s_down: float) -> float:
        c_post = self.c - q * unit_cost
        if c_post < 0.0:
            return -1.0
        n_post = self.n + q
        d_post = self.distance(price, self.trigger(n_post, c_post))
        n_max_post = n_post + c_post / (price * (1.0 + self.f_eff))
        return self.ceiling(n_max_post, d_post, s_down) - n_post

    def rebuy_cap(self, target: float, price: float, unit_cost: float, s_down: float) -> float:
        hi = min(target, self.c / unit_cost)
        if hi < self.dn_min or hi <= 0.0:
            return 0.0
        if self.post_buy_slack(hi, price, unit_cost, s_down) >= 0.0:
            return hi
        lo = self.dn_min
        if self.post_buy_slack(lo, price, unit_cost, s_down) < 0.0:
            return 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.post_buy_slack(mid, price, unit_cost, s_down) >= 0.0:
                lo = mid
            else:
                hi = mid
        return lo

    # -- section 2.5 --------------------------------------------------------
    def fill_probability(
        self, order: Pending, o: float, h: float, low: float, c: float, opposing: float, atr: float
    ) -> float:
        p = self.p
        if order.side == SELL:
            penetration, adverse = h - order.limit, o - c
        else:
            penetration, adverse = order.limit - low, c - o
        if penetration <= 0.0:
            return 0.0
        adverse = max(0.0, adverse)
        phi_pen = 1.0 - math.exp(-p.lambda_liq * (penetration / atr))
        phi_vel = math.exp(-p.beta_vel * (adverse / atr))
        phi_size = 1.0 / (1.0 + (order.qty / max(0.01 * opposing, 1e-8)) ** p.alpha_size)
        return clip01(phi_pen * phi_vel * phi_size)

    # -- section 2.4 --------------------------------------------------------
    def taker_fraction(self, side: int, d: float, s_down: float, s_up: float) -> float:
        rho = self.p.rho
        if side == SELL:
            urgency = 1.0 - d * (1.0 - s_down) * 0.5**self.m_miss
        else:
            urgency = s_up * (1.0 - 0.5 ** (self.m_miss + 1))
        if self.maker_dominated:
            return 1.0
        if rho <= 0.0:
            return 0.0
        if rho >= 1.0 or urgency >= 1.0 - rho:
            return 1.0
        return rho

    # -- section 3.1 --------------------------------------------------------
    def liquidate(self, t: int, p_trig: float) -> tuple[float, float, float]:
        qty = self.n
        cost = qty * p_trig * self.gamma
        self.c = self.c + qty * p_trig * self.keep
        self.n = 0.0
        self.chain_side = 0
        self.counts[7] += 1
        return qty, p_trig * (1.0 - self.p.delta_slip), cost

    def step(
        self,
        t: int,
        ts: int,
        o: float,
        h: float,
        low: float,
        c: float,
        buy_vol: float,
        sell_vol: float,
        s_down: float,
        s_up: float,
        atr: float,
        u: float,
    ) -> RefRow:
        p = self.p
        event, qty, px, fees, p_fill, flags = 0, 0.0, NAN, 0.0, NAN, 0
        liquidated = False

        # Step 1 - buffered floor from the previous end state.
        p_floor = self.trigger(self.n, self.c)

        # Step 2 - intra-bar hard buffered floor trigger.
        if self.n > 1e-8 and low <= p_floor:
            p_trig = p_floor
            if o < p_floor:
                p_trig = o
                if self.gap_bar < 0:
                    self.gap_bar = t
            qty, px, cost = self.liquidate(t, p_trig)
            fees += cost
            event, liquidated = 7, True
            flags |= 64
            self.pending = None
            self.m_miss = 0

        # Step 3 - resolve the maker order pegged at the previous close.
        if not liquidated and self.pending is not None:
            order = self.pending
            opposing = buy_vol if order.side == SELL else sell_vol
            p_fill = self.fill_probability(order, o, h, low, c, opposing, atr)
            if p_fill > 0.0 and u <= p_fill:
                if order.side == SELL:
                    q = min(order.qty, self.n)
                    self.c = self.c + q * order.limit * (1.0 - p.f_maker)
                    self.n = self.n - q
                    event = 1
                    flags |= 1
                else:
                    q = min(order.qty, self.c / (order.limit * (1.0 + p.f_maker)))
                    self.c = self.c - q * order.limit * (1.0 + p.f_maker)
                    self.n = self.n + q
                    event = 3
                    flags |= 4
                self.counts[event] += 1
                fees += q * order.limit * p.f_maker
                qty, px = q, order.limit
                self.m_miss = 0
                if order.side == BUY:
                    # Step 3b - the low follows the fill of a resting bid.
                    post_floor = self.trigger(self.n, self.c)
                    if self.n > 1e-8 and low <= post_floor:
                        qty, px, cost = self.liquidate(t, min(order.limit, post_floor))
                        fees += cost
                        event, liquidated = 7, True
                        flags |= 64
            else:
                event = 5 if order.side == SELL else 6
                flags |= 16 if order.side == SELL else 32
                self.counts[event] += 1
                qty, px = order.qty, order.limit
                self.m_miss += 1
            self.pending = None

        # Step 4 - dual-envelope control law at the close.
        n_max = self.n + self.c / (c * (1.0 + self.f_eff))
        d = self.distance(c, p_floor)
        n_ceil = self.ceiling(n_max, d, s_down)
        n_rebuy = min(n_max * d * s_up, n_ceil)

        side, dn = 0, 0.0
        dn_eff = 0.0 if c <= p_floor + 250.0 else self.dn_min
        if self.n > n_ceil and (self.n - n_ceil) >= dn_eff:
            side, dn = SELL, self.n - n_ceil
        elif self.n < n_rebuy and s_down < 0.25 and (n_rebuy - self.n) >= self.dn_min:
            worst_rate = max(self.gamma, p.f_maker)
            dn = self.rebuy_cap(n_rebuy - self.n, c, c * (1.0 + worst_rate), s_down)
            if dn >= self.dn_min:
                side = BUY

        if side != 0 and dn > 1e-8:
            if side != self.chain_side:
                self.m_miss = 0
                self.chain_side = side
            theta = self.taker_fraction(side, d, s_down, s_up)
            q_taker = theta * dn
            q_maker = dn - q_taker
            if q_taker > 1e-8:
                if side == SELL:
                    q_taker = min(q_taker, self.n)
                    self.c = self.c + q_taker * c * self.keep
                    self.n = self.n - q_taker
                    code, taker_px = 2, c * (1.0 - p.delta_slip)
                    flags |= 2
                else:
                    self.c = self.c - q_taker * c * (1.0 + self.gamma)
                    self.n = self.n + q_taker
                    code, taker_px = 4, c * (1.0 + p.delta_slip)
                    flags |= 8
                self.counts[code] += 1
                fees += q_taker * c * self.gamma
                if event != 7:
                    event, qty, px = code, q_taker, taker_px
            if q_maker > 1e-8:
                self.pending = Pending(side, q_maker, c)
        else:
            self.m_miss = 0
            self.chain_side = 0

        # Step 5 - end-of-bar telemetry.
        v = self.c + self.n * c * self.keep
        margin = v - self.floor_value
        return RefRow(
            bar_idx=t,
            ts_utc_ms=ts,
            open=o,
            high=h,
            low=low,
            close=c,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            s_down=s_down,
            s_up=s_up,
            p_floor=p_floor,
            safety_buf_usd=p_floor - p.p_liq,
            d_norm=d,
            n_max=n_max,
            n_ceil=n_ceil,
            n_rebuy=n_rebuy,
            btc_held=self.n,
            cash_usd=self.c,
            net_equity_usd=v,
            cushion_usd=margin if margin > 0.0 else 0.0,
            event_code=event,
            exec_qty_btc=qty,
            exec_px_usd=px,
            fee_paid_usd=fees,
            p_fill_prob=p_fill,
            u_rand_draw=u,
            flags=flags,
        )


def ref_simulate(
    window: MarketWindow,
    s_down: np.ndarray,
    s_up: np.ndarray,
    atr: np.ndarray,
    u: np.ndarray,
    params: SimParams,
) -> tuple[list[RefRow], ReferenceMachine]:
    """Run the reference machine over a window, one Python step per bar."""
    machine = ReferenceMachine(params)
    rows = []
    for t in range(window.n):
        rows.append(
            machine.step(
                t,
                int(window.ts_ms[t]),
                float(window.open[t]),
                float(window.high[t]),
                float(window.low[t]),
                float(window.close[t]),
                float(window.buy_vol[t]),
                float(window.sell_vol[t]),
                float(s_down[t]),
                float(s_up[t]),
                float(atr[t]),
                float(u[t]),
            )
        )
    return rows, machine
