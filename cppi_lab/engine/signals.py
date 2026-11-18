"""Decline and growth signal pre-computation (TSD section 2.2).

Both signals are pure functions of the selected market window and are
computed once, in ``O(T)``, before the sequential state machine runs.  The
kernels are Numba-compiled array passes written in strict IEEE-754 order so
they agree to the last bit with the plain-Python reference used by the tests.

Conventions the TSD leaves open (recorded in ``docs/DECISIONS.md``):

* an EMA starts at its first sample, ``E_0 = X_0``;
* rolling sums, maxima and minima use the bars available so far while the
  window is still filling (``t < W - 1``);
* signals are computed on the selected window only, with no look-back into
  bars before ``t = 0``, matching ``TR_0 = H_0 - L_0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketWindow

FloatArray = NDArray[np.float64]

#: Normalisation constants of TSD section 2.2.
MOMENTUM_DOWN_ATR_MULT = 1.5
MOMENTUM_UP_ATR_MULT = 2.0
IMBALANCE_SATURATION = 0.40
ATR_FLOOR_USD = 1.0
RANGE_FLOOR_USD = 1.0
VOLUME_FLOOR = 1e-8
GROWTH_TREND_WEIGHT = 0.50
GROWTH_BREAKOUT_WEIGHT = 0.50


@njit(cache=True, nogil=True)
def true_range(high: FloatArray, low: FloatArray, close: FloatArray) -> FloatArray:
    """``TR_t = max(H-L, |H-C_prev|, |L-C_prev|)`` with ``TR_0 = H_0 - L_0``."""
    n = high.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = high[0] - low[0]
    for t in range(1, n):
        prev = close[t - 1]
        a = high[t] - low[t]
        b = abs(high[t] - prev)
        c = abs(low[t] - prev)
        out[t] = max(a, max(b, c))
    return out


@njit(cache=True, nogil=True)
def ema(x: FloatArray, span: int) -> FloatArray:
    """Recursive EMA with ``alpha = 2 / (span + 1)`` seeded at ``x[0]``."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = x[0]
    for t in range(1, n):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


@njit(cache=True, nogil=True)
def rolling_sum(x: FloatArray, window: int) -> FloatArray:
    """Sum of the last ``window`` samples, newest first, partial at the start.

    Each window is summed afresh so the result carries no accumulated drift.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        acc = 0.0
        stop = t - window
        k = t
        while k > stop and k >= 0:
            acc += x[k]
            k -= 1
        out[t] = acc
    return out


@njit(cache=True, nogil=True)
def rolling_max(x: FloatArray, window: int) -> FloatArray:
    """Maximum of the last ``window`` samples via a monotonic deque, ``O(T)``."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    queue = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for t in range(n):
        while tail > head and x[queue[tail - 1]] <= x[t]:
            tail -= 1
        queue[tail] = t
        tail += 1
        if queue[head] <= t - window:
            head += 1
        out[t] = x[queue[head]]
    return out


@njit(cache=True, nogil=True)
def rolling_min(x: FloatArray, window: int) -> FloatArray:
    """Minimum of the last ``window`` samples via a monotonic deque, ``O(T)``."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    queue = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for t in range(n):
        while tail > head and x[queue[tail - 1]] >= x[t]:
            tail -= 1
        queue[tail] = t
        tail += 1
        if queue[head] <= t - window:
            head += 1
        out[t] = x[queue[head]]
    return out


@njit(cache=True, nogil=True)
def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(cache=True, nogil=True)
def decline_components(
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    buy_vol: FloatArray,
    sell_vol: FloatArray,
    w_fast: int,
    w_slow: int,
    w_atr: int,
    w_imb: int,
    w_mom: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(S_down, M_down, I_down, ATR_d)`` per TSD section 2.2.2.

    ``ATR_d`` is already floored at 1 USD; the fill model reuses it.
    """
    n = close.shape[0]
    fast = ema(close, w_fast)
    slow = ema(close, w_slow)
    atr = ema(true_range(high, low, close), w_atr)
    sum_sell = rolling_sum(sell_vol, w_imb)
    sum_buy = rolling_sum(buy_vol, w_imb)
    s_down = np.empty(n, dtype=np.float64)
    m_down = np.empty(n, dtype=np.float64)
    i_down = np.empty(n, dtype=np.float64)
    for t in range(n):
        atr[t] = max(atr[t], ATR_FLOOR_USD)
        m_down[t] = _clip01((slow[t] - fast[t]) / (MOMENTUM_DOWN_ATR_MULT * atr[t]))
        i_down[t] = _clip01(
            (sum_sell[t] - sum_buy[t])
            / (IMBALANCE_SATURATION * max(sum_sell[t] + sum_buy[t], VOLUME_FLOOR))
        )
        s_down[t] = _clip01(w_mom * m_down[t] + (1.0 - w_mom) * i_down[t])
    return s_down, m_down, i_down, atr


@njit(cache=True, nogil=True)
def growth_components(
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    s_down: FloatArray,
    w_fast: int,
    w_slow: int,
    w_atr: int,
    w_brk: int,
    tau_veto: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(S_up, M_up, B_up, G_veto)`` per TSD section 2.2.3."""
    n = close.shape[0]
    fast = ema(close, w_fast)
    slow = ema(close, w_slow)
    atr = ema(true_range(high, low, close), w_atr)
    range_high = rolling_max(high, w_brk)
    range_low = rolling_min(low, w_brk)
    s_up = np.empty(n, dtype=np.float64)
    m_up = np.empty(n, dtype=np.float64)
    b_up = np.empty(n, dtype=np.float64)
    g_veto = np.empty(n, dtype=np.float64)
    for t in range(n):
        m_up[t] = _clip01((fast[t] - slow[t]) / (MOMENTUM_UP_ATR_MULT * max(atr[t], ATR_FLOOR_USD)))
        mid = 0.5 * (range_high[t] + range_low[t])
        half = max(0.5 * (range_high[t] - range_low[t]), RANGE_FLOOR_USD)
        b_up[t] = _clip01((close[t] - mid) / half)
        g_veto[t] = max(0.0, 1.0 - s_down[t] / tau_veto)
        s_up[t] = _clip01(
            (GROWTH_TREND_WEIGHT * m_up[t] + GROWTH_BREAKOUT_WEIGHT * b_up[t]) * g_veto[t]
        )
    return s_up, m_up, b_up, g_veto


@dataclass(frozen=True)
class Signals:
    """Signal arrays for one market window, aligned bar for bar."""

    s_down: FloatArray
    s_up: FloatArray
    #: ``max(EMA_{W_atr,d}(TR), 1.0)``, the volatility scale of the fill model.
    atr: FloatArray
    m_down: FloatArray
    i_down: FloatArray
    m_up: FloatArray
    b_up: FloatArray
    g_veto: FloatArray


#: The :class:`SimParams` fields the signals depend on (used for caching).
SIGNAL_PARAM_KEYS: tuple[str, ...] = (
    "w_fast_d",
    "w_slow_d",
    "w_atr_d",
    "w_imb_d",
    "w_mom_d",
    "w_fast_u",
    "w_slow_u",
    "w_atr_u",
    "w_brk_u",
    "tau_veto",
)


def compute_signals(window: MarketWindow, params: SimParams) -> Signals:
    """Compute ``S_down`` and ``S_up`` (and their components) for a window."""
    s_down, m_down, i_down, atr = decline_components(
        window.high,
        window.low,
        window.close,
        window.buy_vol,
        window.sell_vol,
        int(params.w_fast_d),
        int(params.w_slow_d),
        int(params.w_atr_d),
        int(params.w_imb_d),
        float(params.w_mom_d),
    )
    s_up, m_up, b_up, g_veto = growth_components(
        window.high,
        window.low,
        window.close,
        s_down,
        int(params.w_fast_u),
        int(params.w_slow_u),
        int(params.w_atr_u),
        int(params.w_brk_u),
        float(params.tau_veto),
    )
    return Signals(
        s_down=s_down,
        s_up=s_up,
        atr=atr,
        m_down=m_down,
        i_down=i_down,
        m_up=m_up,
        b_up=b_up,
        g_veto=g_veto,
    )
