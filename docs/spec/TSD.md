# Technical Specification Document (TSD): CPPI-Lab
**Architecture, Mathematical Formulations, Simulation Kernel & UI/Documentation Engineering**

* **Document Version:** 2.0.0
* **Date:** November 16, 2026
* **Companion Document:** `cppi_lab_prd.md`

---

## 1. System Architecture & Technology Stack

### 1.1 Runtime Stack
* **Language:** Python 3.11+
* **GUI & Windowing Engine:** `PySide6` (Qt 6.7+) + **`PySide6-QtAds` (Qt Advanced Docking System)** for tear-off movable/resizable always-on-top windows and multi-zone split-panel docking.
* **GPU/OpenGL Visualization Canvas:** `pyqtgraph` (`0.13+`) configured with `useOpenGL=True` and `enableExperimental=True` for hardware-accelerated time-series rendering.
* **JIT Simulation Kernel:** `Numba` (`0.59+`) using `@njit(nogil=True, fastmath=True, parallel=True)` with deterministic counter-seeded `numpy.random.PCG64` streams.
* **Data Storage & Table Model:** `Polars` + `PyArrow` backed by local Zstd-compressed Apache Parquet files (`data/btc_usd_1m.parquet`), bound via zero-copy NumPy views to a virtualized `QAbstractTableModel`.
* **Documentation & Math Rendering Engine:** Embedded `QWebEngineView` instances served via a custom `QWebEngineUrlSchemeHandler` (`cppidoc://`) with local **KaTeX** (for sub-10ms math layout) and bidirectional `QWebChannel` JavaScript-to-Python RPC bridges.

### 1.2 Component Topology & Data Flow
```
+-----------------------------------------------------------------------------------+
|                              CPPI-Lab Application Process                         |
|                                                                                   |
|  +------------------------+    ParameterStruct     +---------------------------+  |
|  | ParameterControlPanel  | ---------------------> | SimulationController      |  |
|  | (Sidebar + ? Popovers) |                        | - Signal Precomputer      |  |
|  +------------------------+                        | - Numba JIT CPPI Kernel   |  |
|              |                                     | - Monte Carlo Pool        |  |
|              | cppidoc://anchor                    +-------------+-------------+  |
|              v                                                   |                |
|  +------------------------+                        Zero-Copy     | RecArray       |
|  | DocWorkbenchDockWidget |                        Telemetry     | (24 Columns)   |
|  | - Primary WebView      | <------------------------------------+-------------+  |
|  | - Counterpart Preview  |                                      |             |  |
|  +------------------------+                                      v             v  |
|                                                    +---------------+  +--------+--+
|                                                    | 3-Tier OpenGL |  | Floating/ |
|                                                    | Chart Canvas  |  | Dockable  |
|                                                    | (PyQtGraph)   |  | TableView |
|                                                    +---------------+  +-----------+
+-----------------------------------------------------------------------------------+
```

---

## 2. Complete Mathematical & Algorithmic Specification

### 2.1 Market Data Representation & Pre-Computation
Let the selected historical window consist of $T$ 1-minute bars indexed by $t \in \{0, 1, \dots, T-1\}$. Each bar $t$ is defined by the vector:
$$\mathbf{B}_t = \big(\tau_t,\; O_t,\; H_t,\; L_t,\; C_t,\; V_{\text{buy}, t},\; V_{\text{sell}, t}\big)$$
where:
* $\tau_t$: UTC epoch timestamp in milliseconds (`int64`).
* $O_t, H_t, L_t, C_t$: Open, High, Low, and Close prices in USD (`float64`).
* $V_{\text{buy}, t}$: Volume initiated by aggressive buyer-takers lifting the ask during minute $t$ (`float64`).
* $V_{\text{sell}, t}$: Volume initiated by aggressive seller-takers hitting the bid during minute $t$ (`float64`).
* *Fallback for Standard OHLCV Data:* When importing historical candles lacking explicit taker side splits, signed volumes are synthesized via the Close Location Value (CLV):
  $$V_{\text{sell}, t} = V_{\text{tot}, t} \cdot \frac{H_t - C_t}{\max(H_t - L_t,\; 10^{-8})}, \quad V_{\text{buy}, t} = V_{\text{tot}, t} - V_{\text{sell}, t}$$

---

### 2.2 Vectorized Signal Definitions ($S_{\text{down}, t}$ and $S_{\text{up}, t}$)
The two signals are computed independently prior to invoking the sequential state-machine loop.

#### 2.2.1 True Range & Volatility Normalization
For any bar $t \ge 1$, the 1-minute True Range is:
$$\text{TR}_t = \max\Big(H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|\Big), \quad \text{with } \text{TR}_0 = H_0 - L_0$$
For any span $W$, the exponential moving average operator $\mathcal{E}_W(X)_t$ uses smoothing factor $\alpha_W = \frac{2}{W + 1}$:
$$\mathcal{E}_W(X)_t = \alpha_W X_t + (1 - \alpha_W) \mathcal{E}_W(X)_{t-1}$$

#### 2.2.2 Decline Signal $S_{\text{down}, t} \in [0, 1]$
1. **Volatility-Normalized Downward Momentum ($M_{\text{down}, t}$):**
   $$M_{\text{down}, t} = \text{clip}\left(\frac{\mathcal{E}_{W_{s,d}}(C)_t - \mathcal{E}_{W_{f,d}}(C)_t}{1.5 \cdot \max\big(\mathcal{E}_{W_{\text{atr},d}}(\text{TR})_t,\; 1.0\big)},\; 0.0,\; 1.0\right)$$
2. **Rolling Taker Sell Flow Imbalance ($I_{\text{down}, t}$):**
   Let $\Sigma_{\text{sell}, t} = \sum_{k=0}^{W_{\text{imb},d}-1} V_{\text{sell}, t-k}$ and $\Sigma_{\text{buy}, t} = \sum_{k=0}^{W_{\text{imb},d}-1} V_{\text{buy}, t-k}$. Then:
   $$I_{\text{down}, t} = \text{clip}\left(\frac{\Sigma_{\text{sell}, t} - \Sigma_{\text{buy}, t}}{0.40 \cdot \max\big(\Sigma_{\text{sell}, t} + \Sigma_{\text{buy}, t},\; 10^{-8}\big)},\; 0.0,\; 1.0\right)$$
3. **Composite Decline Signal:**
   $$S_{\text{down}, t} = \text{clip}\Big(w_{m,d} \cdot M_{\text{down}, t} + (1 - w_{m,d}) \cdot I_{\text{down}, t},\; 0.0,\; 1.0\Big)$$

#### 2.2.3 Growth Signal $S_{\text{up}, t} \in [0, 1]$
1. **Multi-Horizon Upward Trend Structure ($M_{\text{up}, t}$):**
   $$M_{\text{up}, t} = \text{clip}\left(\frac{\mathcal{E}_{W_{f,u}}(C)_t - \mathcal{E}_{W_{s,u}}(C)_t}{2.0 \cdot \max\big(\mathcal{E}_{W_{\text{atr},u}}(\text{TR})_t,\; 1.0\big)},\; 0.0,\; 1.0\right)$$
2. **Rolling Range Breakout Confirmation ($B_{\text{up}, t}$):**
   Let $H^{(W_{\text{brk},u})}_t = \max_{0 \le k \lt W_{\text{brk},u}} H_{t-k}$ and $L^{(W_{\text{brk},u})}_t = \min_{0 \le k \lt W_{\text{brk},u}} L_{t-k}$. Define the range midpoint $M^{(W_{\text{brk},u})}_t = \frac{1}{2}\big(H^{(W_{\text{brk},u})}_t + L^{(W_{\text{brk},u})}_t\big)$ and half-span $R^{(W_{\text{brk},u})}_t = \max\big(\frac{1}{2}(H^{(W_{\text{brk},u})}_t - L^{(W_{\text{brk},u})}_t),\; 1.0\big)$:
   $$B_{\text{up}, t} = \text{clip}\left(\frac{C_t - M^{(W_{\text{brk},u})}_t}{R^{(W_{\text{brk},u})}_t},\; 0.0,\; 1.0\right)$$
3. **Crash-Veto Gate & Composite Growth Signal:**
   To prevent rebuying during active liquidation cascades, a linear crash-veto gate suppresses $S_{\text{up}, t}$ to zero whenever $S_{\text{down}, t} \ge \tau_{\text{veto}}$ (default $\tau_{\text{veto}} = 0.50$):
   $$G_{\text{veto}, t} = \max\left(0.0,\; 1.0 - \frac{S_{\text{down}, t}}{\tau_{\text{veto}}}\right)$$
   $$S_{\text{up}, t} = \text{clip}\Big(\big(0.50 \cdot M_{\text{up}, t} + 0.50 \cdot B_{\text{up}, t}\big) \cdot G_{\text{veto}, t},\; 0.0,\; 1.0\Big)$$

---

### 2.3 Hard Wealth Floor, Safety Buffer & Dual-Envelope Control Law

#### 2.3.1 Hard Wealth Floor ($F$) and Buffered Liquidation Trigger ($P_{\text{floor}, t}$)
Given initial BTC holdings $N_0$, initial USD cash $c_0$, and hard liquidation price $P_{\text{liq}}$ (default $\$60,000$), the invariant minimum portfolio value is:
$$F = N_0 \cdot P_{\text{liq}} + c_0$$
Let $\gamma = f_{\text{taker}} + \delta_{\text{slip}}$ denote the total emergency taker exit friction (with default $f_{\text{taker}} = 0.0030$ and safety slippage allowance $\delta_{\text{slip}} = 0.0010$, giving $\gamma = 0.0040$).

At any bar $t$ with current holdings $n_t \ge 0$ and cash $c_t \ge 0$:
1. **Liquidation-Adjusted Net Portfolio Value ($V_t$):**
   $$V_t(P_t) = c_t + n_t \cdot P_t \cdot (1 - \gamma)$$
2. **Active Risk Cushion ($C_t$):**
   $$C_t(P_t) = \max\big(0.0,\; V_t(P_t) - F\big)$$
3. **Liquidation Level + Safety Buffer ($P_{\text{floor}, t}$):**
   To guarantee both Condition 1 (complete sale before price falls below $P_{\text{liq}}$) and the Net Value Invariant ($V_t \ge F$ at all times), the emergency taker liquidation trigger price $P_{\text{floor}, t}$ is strictly defined across all values of $\rho \in [0, 1]$ as:
   $$P_{\text{floor}, t} = \begin{cases} \max\left(\dfrac{P_{\text{liq}}}{1 - \gamma},\; \dfrac{\max(0.0,\; F - c_t)}{n_t \cdot (1 - \gamma)}\right) & \text{if } n_t \gt 10^{-8} \\ \dfrac{P_{\text{liq}}}{1 - \gamma} & \text{if } n_t \le 10^{-8} \end{cases}$$
   The explicit **Safety Buffer** above the liquidation level at timestep $t$ is therefore:
   $$\Delta P_{\text{buf}, t} = P_{\text{floor}, t} - P_{\text{liq}} \ge P_{\text{liq}} \cdot \frac{\gamma}{1 - \gamma}$$
   When a taker liquidation order is triggered at $P_{\text{floor}, t}$, the realized post-slippage and post-fee fill price lands inside the safety buffer $[P_{\text{liq}}, P_{\text{floor}, t}]$, ensuring net USD proceeds $\ge F$.

#### 2.3.2 Dual-Envelope Target Allocation ($n_{\text{ceil}, t}$ and $n_{\text{floor}, t}$)
At bar close $P_t = C_t$, the maximum recoverable BTC inventory $n_{\text{max}, t}$ (accounting for the blended fee rate $f_{\text{eff}}(\rho) = (1 - \rho) f_{\text{maker}} + \rho f_{\text{taker}}$ required to convert current cash $c_t$ back into BTC) is:
$$n_{\text{max}, t} = n_t + \frac{c_t}{P_t \cdot \big(1 + f_{\text{eff}}(\rho)\big)}$$
The normalized distance above the buffered floor $d_t \in [0, 1]$ is:
$$d_t = \text{clip}\left(\frac{P_t - P_{\text{floor}, t}}{\max(P_{\text{ref}} - P_{\text{floor}, t},\; 1.0)},\; 0.0,\; 1.0\right)$$
The **Upper Sell Ceiling** $n_{\text{ceil}, t}$ and **Lower Rebuy Floor** $n_{\text{floor}, t}$ are:
$$n_{\text{ceil}, t} = n_{\text{max}, t} \cdot d_t \cdot \Big(1.0 - k_{\text{sell}} \cdot (1.0 - d_t) \cdot S_{\text{down}, t}\Big)$$
$$n_{\text{floor}, t} = n_{\text{max}, t} \cdot d_t \cdot S_{\text{up}, t}$$

* **Sell Criterion:** Triggered at bar close $t$ if $n_t \gt n_{\text{ceil}, t}$ and $n_t - n_{\text{ceil}, t} \ge \Delta n_{\text{eff}, t}$, where the effective minimum trade size $\Delta n_{\text{eff}, t}$ drops to $0$ inside the final $\$250$ buffer zone above $P_{\text{floor}, t}$:
  $$\Delta n_{\text{eff}, t} = \begin{cases} 0.0 & \text{if } P_t \le P_{\text{floor}, t} + 250.0 \\ \Delta n_{\text{min}} & \text{otherwise} \end{cases}$$
  Required sell quantity: $\Delta n_{\text{sell}, t} = n_t - n_{\text{ceil}, t}$.
* **Buy Criterion:** Triggered at bar close $t$ if $n_t \lt n_{\text{floor}, t}$, $S_{\text{down}, t} \lt 0.25$, and $n_{\text{floor}, t} - n_t \ge \Delta n_{\text{min}}$.
  Required buy quantity: $\Delta n_{\text{buy}, t} = n_{\text{floor}, t} - n_t$.

---

### 2.4 Risk-Aversion Parameter ($\rho \in [0, 1]$) & Order Routing Law
Let $\rho \in [0, 1]$ be the user-tunable Risk-Aversion parameter, and let $m_{\text{miss}, t} \in \{0, 1, 2, \dots\}$ be the count of consecutive bars prior to $t$ on which a passive maker order failed to fill ($U \gt P_{\text{fill}}$).

When a rebalance order of size $\Delta n_t \gt 0$ (sell or buy) is triggered above $P_{\text{floor}, t}$, define the normalized **Execution Urgency** $u_t \in [0, 1)$:
$$u_t = \begin{cases} 1.0 - d_t \cdot (1.0 - S_{\text{down}, t}) \cdot 0.5^{m_{\text{miss}, t}} & \text{if Selling } (\Delta n_{\text{sell}, t} \gt 0) \\ S_{\text{up}, t} \cdot \big(1.0 - 0.5^{m_{\text{miss}, t} + 1}\big) & \text{if Buying } (\Delta n_{\text{buy}, t} \gt 0) \end{cases}$$

The **Taker Routing Fraction** $\theta_{\text{taker}, t}(\rho) \in [0, 1]$ is governed by:
$$\theta_{\text{taker}, t}(\rho) = \begin{cases} 0.0 & \text{if } \rho = 0.0 \\ 1.0 & \text{if } \rho = 1.0 \text{ or } \big(\rho \gt 0.0 \text{ and } u_t \ge 1.0 - \rho\big) \\ \rho & \text{if } 0.0 \lt \rho \lt 1.0 \text{ and } u_t \lt 1.0 - \rho \end{cases}$$

#### Exact Boundary & Interior Verification:
1. **When $\rho = 0.0$:** $\theta_{\text{taker}, t}(0) = 0.0$ identically for all rebalancing orders above $P_{\text{floor}, t}$. The strategy places **only** passive `post_only` maker orders during normal trading and **only places a taker sell order when the liquidation level + safety buffer ($P_{\text{floor}, t}$) is reached**.
2. **When $\rho = 1.0$:** $\theta_{\text{taker}, t}(1) = 1.0$ identically. Every buy and sell rebalance is executed 100% as an immediate taker order paying $f_{\text{taker}} = 0.30\%$.
3. **When $0.0 \lt \rho \lt 1.0$:** Low-urgency rebalances split $\rho \cdot \Delta n_t$ into an immediate taker child order and $(1 - \rho) \cdot \Delta n_t$ into a passive maker child order; high-urgency rebalances ($u_t \ge 1 - \rho$, such as fast drops near the floor or repeated missed maker orders) escalate to 100% taker execution.

---

### 2.5 Stochastic Liquidity Model ($P_{\text{fill}, t}$ and Uniform Variate Draw)
When a passive `post_only` maker order of size $q_{\text{maker}, t-1} = (1 - \theta_{\text{taker}, t-1}) \Delta n_{t-1}$ is pegged at limit price $P_{\text{lim}} = C_{t-1}$ at the close of bar $t-1$, its execution outcome during bar $t$ is resolved as follows:

1. **Favorable Excursion Check ($\Delta P_{\text{pen}, t}$):**
   $$\Delta P_{\text{pen}, t} = \begin{cases} H_t - P_{\text{lim}} & \text{for a Maker Sell order} \\ P_{\text{lim}} - L_t & \text{for a Maker Buy order} \end{cases}$$
   If $\Delta P_{\text{pen}, t} \le 0$, the market price never reached or crossed the limit price during bar $t$, so $P_{\text{fill}, t} = 0.0$.

2. **Three-Factor Microstructure Fill Probability ($\Delta P_{\text{pen}, t} \gt 0$):**
   Let $\text{ATR}_t = \max\big(\mathcal{E}_{W_{\text{atr}, d}}(\text{TR})_t,\; 1.0\big)$.
   * **Queue Penetration Factor ($\Phi_{\text{pen}, t}$):**
     $$\Phi_{\text{pen}, t} = 1.0 - \exp\left(-\lambda_{\text{liq}} \cdot \frac{\Delta P_{\text{pen}, t}}{\text{ATR}_t}\right)$$
   * **Adverse Price Velocity Factor ($\Phi_{\text{vel}, t}$):**
     Measures how rapidly the bar swept *away* from the resting limit order:
     $$v_{\text{adv}, t} = \begin{cases} \dfrac{\max(0.0,\; O_t - C_t)}{\text{ATR}_t} & \text{for a Maker Sell order} \\ \dfrac{\max(0.0,\; C_t - O_t)}{\text{ATR}_t} & \text{for a Maker Buy order} \end{cases}$$
     $$\Phi_{\text{vel}, t} = \exp\big(-\beta_{\text{vel}} \cdot v_{\text{adv}, t}\big)$$
   * **Counter-Party Volume Absorption Factor ($\Phi_{\text{size}, t}$):**
     Let $V_{\text{opp}, t} = V_{\text{buy}, t}$ for a Maker Sell (requires buyer-takers) and $V_{\text{opp}, t} = V_{\text{sell}, t}$ for a Maker Buy (requires seller-takers):
     $$\Phi_{\text{size}, t} = \frac{1.0}{1.0 + \left(\dfrac{q_{\text{maker}, t-1}}{\max(0.01 \cdot V_{\text{opp}, t},\; 10^{-8})}\right)^{\alpha_{\text{size}}}}$$
   * **Combined Fill Probability ($P_{\text{fill}, t}$):**
     $$P_{\text{fill}, t} = \text{clip}\Big(\Phi_{\text{pen}, t} \cdot \Phi_{\text{vel}, t} \cdot \Phi_{\text{size}, t},\; 0.0,\; 1.0\Big)$$

3. **Uniform Variate Resolution ($U_t \sim \mathcal{U}(0, 1)$):**
   To ensure that step-by-step PRNG draws remain reproducible across parameter sensitivity tests, the engine pre-generates a vector of uniform variates $\mathbf{U} = (U_0, U_1, \dots, U_{T-1}) \in [0, 1]^T$ indexed by bar $t$ from `np.random.Generator(np.random.PCG64(s_rng))`.
   * **Fill Outcome ($U_t \le P_{\text{fill}, t}$):** The maker order executes at $P_{\text{lim}}$, paying maker fee $f_{\text{maker}} = 0.15\%$, and resets $m_{\text{miss}, t} = 0$.
   * **Miss / Pass-Through Outcome ($U_t \gt P_{\text{fill}, t}$):** The order does not fill, logs a `MISSED_MAKER_SELL` or `MISSED_MAKER_BUY` event at $(t, P_{\text{lim}}, P_{\text{fill}, t}, U_t)$, and increments $m_{\text{miss}, t} = m_{\text{miss}, t-1} + 1$.

---

## 3. Core Simulation State-Machine Specification (Numba JIT)

### 3.1 Intra-Bar Execution Priority Order
Within each 1-minute bar $t \in \{0, \dots, T-1\}$, the Numba state machine executes operations in strict chronological microstructure order:

1. **Step 1 — Recompute Active Buffered Floor ($P_{\text{floor}, t}$):**
   Compute $P_{\text{floor}, t}$ from $(n_{t-1}, c_{t-1})$ using Section 2.3.1.
2. **Step 2 — Check Intra-Bar Hard Buffered Floor Trigger ($L_t \le P_{\text{floor}, t}$):**
   If $n_{t-1} \gt 10^{-8}$ and $L_t \le P_{\text{floor}, t}$:
   * Cancel any resting maker order from bar $t-1$.
   * Trigger an immediate **Emergency Taker Sell** of 100% of remaining BTC ($q = n_{t-1}$) at trigger price $P_{\text{trig}} = \min(O_t, P_{\text{floor}, t})$.
   * Realized execution price after slippage $\delta_{\text{slip}}$ is $P_{\text{exec}} = P_{\text{trig}} \cdot (1 - \delta_{\text{slip}})$, and taker fee $f_{\text{taker}}$ is deducted:
     $$c_t = c_{t-1} + n_{t-1} \cdot P_{\text{trig}} \cdot (1 - \gamma), \quad n_t = 0.0$$
   * Note that while $P_{\text{exec}} \lt P_{\text{floor}, t}$ (the sell price ends up below the buffered liquidation trigger), the safety buffer $\Delta P_{\text{buf}, t}$ ensures $P_{\text{exec}} \ge P_{\text{liq}}$ and $c_t \ge F$ (unless an instantaneous market gap at bar open $O_t$ jumped straight past the entire safety buffer).
   * Record `EventCode = 7 (HARD_BUFFER_LIQUIDATION)`.
3. **Step 3 — Resolve Pending Passive Maker Order from Bar $t-1$:**
   If no floor liquidation occurred and $q_{\text{maker}, t-1} \neq 0$:
   * Evaluate $P_{\text{fill}, t}$ using Section 2.5 and compare against $U_t$.
   * If $U_t \le P_{\text{fill}, t}$: update $(n_t, c_t)$ at $P_{\text{lim}}$ with fee $f_{\text{maker}}$, set $m_{\text{miss}, t} = 0$, and record `EventCode = 1 (MAKER_SELL)` or `3 (MAKER_BUY)`.
   * If $U_t \gt P_{\text{fill}, t}$: leave $(n_t, c_t)$ unchanged, increment $m_{\text{miss}, t}$, and record `EventCode = 5 (MISSED_MAKER_SELL)` or `6 (MISSED_MAKER_BUY)`.
4. **Step 4 — Evaluate CPPI Dual-Envelope Control Law at Bar Close ($C_t$):**
   If $n_t \gt 0$ (or cash is available to rebuy when $C_t \gt P_{\text{floor}, t}$ and $S_{\text{up}, t} \gt 0$):
   * Compute $d_t$, $n_{\text{ceil}, t}$, and $n_{\text{floor}, t}$.
   * If Sell or Buy criterion is met for $\Delta n_t$:
     * Compute urgency $u_t$ and taker fraction $\theta_{\text{taker}, t}(\rho)$.
     * **Immediate Taker Portion ($q_{\text{taker}, t} = \theta_{\text{taker}, t}(\rho) \cdot \Delta n_t$):** If $q_{\text{taker}, t} \gt 10^{-8}$, execute immediately at $C_t \cdot (1 \mp \delta_{\text{slip}})$ paying $f_{\text{taker}}$, and record `EventCode = 2 (TAKER_SELL)` or `4 (TAKER_BUY)`.
     * **Passive Maker Portion ($q_{\text{maker}, t} = (1 - \theta_{\text{taker}, t}(\rho)) \cdot \Delta n_t$):** Peg at $P_{\text{lim}} = C_t$ for evaluation in Bar $t+1$.
5. **Step 5 — Commit End-of-Bar Telemetry Row:**
   Write all 24 state variables into row $t$ of the pre-allocated telemetry array.

### 3.2 Complete 24-Column Telemetry Buffer Schema
The simulation returns a contiguous NumPy structured array `TELEMETRY_DTYPE` used simultaneously by the OpenGL plotter and the Floating Tabular Window:

```python
TELEMETRY_DTYPE = np.dtype([
    ("bar_idx",        np.int32),          # 0: Integer index t
    ("ts_utc_ms",      np.int64),          # 1: Epoch timestamp (ms)
    ("open",           np.float64),        # 2: Bar Open ($)
    ("high",           np.float64),        # 3: Bar High ($)
    ("low",            np.float64),        # 4: Bar Low ($)
    ("close",          np.float64),        # 5: Bar Close ($)
    ("buy_vol",        np.float64),        # 6: Buyer-taker volume (BTC)
    ("sell_vol",       np.float64),        # 7: Seller-taker volume (BTC)
    ("s_down",         np.float64),        # 8: Decline Signal S_down(t) in [0,1]
    ("s_up",           np.float64),        # 9: Growth Signal S_up(t) in [0,1]
    ("p_floor",        np.float64),        # 10: Liquidation Level + Safety Buffer ($)
    ("safety_buf_usd", np.float64),        # 11: Active Safety Buffer P_floor - P_liq ($)
    ("d_norm",         np.float64),        # 12: Normalized distance d_t in [0,1]
    ("n_max",          np.float64),        # 13: Max recoverable BTC n_max,t
    ("n_ceil",         np.float64),        # 14: Upper Sell Ceiling n_ceil,t (BTC)
    ("n_rebuy",        np.float64),        # 15: Lower Rebuy Floor n_floor,t (BTC)
    ("btc_held",       np.float64),        # 16: Actual BTC held n_t
    ("cash_usd",       np.float64),        # 17: Actual USD cash held c_t
    ("net_equity_usd", np.float64),        # 18: Net liquidation value V_t ($)
    ("cushion_usd",    np.float64),        # 19: Active risk cushion V_t - F ($)
    ("event_code",     np.uint8),          # 20: 0..7 execution/miss event enum
    ("exec_qty_btc",   np.float64),        # 21: Filled or missed order size (BTC)
    ("exec_px_usd",    np.float64),        # 22: Realized fill price or missed limit ($)
    ("fee_paid_usd",   np.float64),        # 23: Transaction fee + slippage paid ($)
    ("p_fill_prob",    np.float64),        # 24: Maker fill probability P_fill,t
    ("u_rand_draw",    np.float64),        # 25: Uniform random draw U_t in [0,1]
])
```

---

## 4. UI Layout, Visual Encoding & Windowing Architecture

### 4.1 Single-Screen Simultaneous Time-Series Canvas (`CentralPlotWidget`)
The central area of `MainWindow` is a zero-scrollbar vertical stack containing the **Top KPI Ribbon** (`64px` fixed height) and a 3-pane `pyqtgraph.GraphicsLayoutWidget` occupying 100% of remaining height with stretch ratios `55 : 25 : 20`. All three panes share a locked `XAxis` (`DateAxisItem`) so zooming or panning any pane synchronizes all three simultaneously.

#### Automated Visual Styling & Event Encoding Matrix
| Visual Layer | Pane | Geometry / Style | Hex Color | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **BTC Close Price ($C_t$)** | Pane 1 | Solid Curve (`1.6px`) + `[L_t, H_t]` Band | `#F8FAFC` / `#33415540` | Primary price action context |
| **Hard Liquidation Level ($P_{\text{liq}}$)** | Pane 1 | Horizontal Dashed Line (`1.5px`) | `#DC2626` | Invariant $\$60,000$ floor boundary |
| **Liquidation + Safety Buffer ($P_{\text{floor}, t}$)** | Pane 1 | Stepped Solid Line (`2.2px`) + Shaded Buffer Zone | `#F97316` | Active taker liquidation trigger level |
| **Net Portfolio Equity ($V_t / N_0$)** | Pane 1 | Glowing Solid Curve (`2.4px`) | `#38BDF8` | Real-time portfolio wealth per initial BTC |
| **Monte Carlo 5%--95% Equity Fan** | Pane 1 | Translucent Polygon Band | `#38BDF825` | Stochastic liquidity variance envelope |
| **Maker Sell Fill (`Code 1`)** | Pane 1 | Downward Triangle ($\blacktriangledown$, `11px`) | `#06B6D4` | Passive `post_only` sell filled ($0.15\%$ fee) |
| **Taker Sell Fill (`Code 2`)** | Pane 1 | Diamond ($\blacklozenge$, `12px`) + Vertical Stem | `#EC4899` | Urgent/conservative taker sell ($0.30\%$ fee) |
| **Maker Buy Fill (`Code 3`)** | Pane 1 | Upward Triangle ($\blacktriangle$, `11px`) | `#10B981` | Passive `post_only` rebuy filled ($0.15\%$ fee) |
| **Taker Buy Fill (`Code 4`)** | Pane 1 | Diamond ($\blacklozenge$, `12px`) | `#84CC16` | Taker rebuy filled ($0.30\%$ fee) |
| **Missed Maker Order (`Code 5, 6`)** | Pane 1 | Hollow Ring ($\circ$, `13px`, `2px` stroke) | `#F59E0B` | Limit crossed or touched but $U_t \gt P_{\text{fill}, t}$ |
| **Hard Buffer Liquidation (`Code 7`)** | Pane 1-3 | Vertical Full-Height Flash Line + Badge | `#EF4444` | 100% exit triggered at $P_{\text{floor}, t}$ |
| **Sell Ceiling ($n_{\text{ceil}, t}$)** | Pane 2 | Dashed Line (`1.5px`) | `#FB7185` | Upper permissible BTC limit |
| **Rebuy Floor ($n_{\text{floor}, t}$)** | Pane 2 | Dashed Line (`1.5px`) | `#34D399` | Lower target BTC rebuy boundary |
| **Hysteresis Deadband** | Pane 2 | Shaded Fill Between $n_{\text{floor}, t}$ & $n_{\text{ceil}, t}$ | `#64748B28` | No-trade zone preventing fee churn |
| **Actual BTC Holdings ($n_t$)** | Pane 2 | Stepped Solid Line (`2.5px`) | `#FACC15` | Instantaneous BTC allocation |
| **Decline Signal ($S_{\text{down}, t}$)** | Pane 3 | Area Curve (`1.5px`) | `#F43F5E` | Bearish momentum + seller taker flow |
| **Growth Signal ($S_{\text{up}, t}$)** | Pane 3 | Area Curve (`1.5px`) | `#10B981` | Bullish trend + 4h breakout (veto-gated) |
| **Fill Probability ($P_{\text{fill}, t}$) vs. $U_t$** | Pane 3 | Vertical Stem ($P_{\text{fill}, t}$) + Dot ($U_t$) | `#A855F7` | Visualizes exact Bernoulli coin flip outcome |

---

### 4.2 Movable, Resizable, Always-On-Top & Dockable Windows (`QtAds`)
All secondary workspaces—specifically the **Tabular Data Ledger (`LedgerDockWidget`)** and the **Documentation Workbench (`DocWorkbenchDockWidget`)**—are instantiated as `PySide6QtAds.CDockWidget` containers managed by the main window's `CDockManager`:

1. **Floating Mode (Default for Popup Inspection):**
   * Calling `dock_widget.setFloating()` wraps the widget in a `CFloatingDockContainer` configured with window flags:
     ```python
     container.setWindowFlags(
         Qt.WindowType.Tool
         | Qt.WindowType.WindowStaysOnTopHint
         | Qt.WindowType.WindowMinimizeButtonHint
         | Qt.WindowType.WindowMaximizeButtonHint
     )
     ```
   * Because `Qt.WindowType.Tool` is parented to `MainWindow`, the floating window **always stays on top of the main application window**, features full 8-edge mouse resizing and title-bar dragging, and automatically hides when the main application is minimized.
2. **Docked Panel Mode (Left / Right / Bottom / Auto-Hide):**
   * Dragging any floating window's title bar over the main window displays the `QtAds` drop compass (`LeftDockWidgetArea`, `RightDockWidgetArea`, `BottomDockWidgetArea`, or `AutoHideSideBar`).
   * Dropping the window snaps it into a resizable splitter panel adjacent to the central 3-tier chart, which dynamically reflows its aspect ratio so all three time-series panes remain 100% visible without scrollbars.

---

### 4.3 Interactive `?` Hover Popovers (`RichPopoverManager`)
To satisfy the requirement that hovering over `?` displays an explanation popup containing clickable links to the Technical and Theoretical Documentation:

```
[User Hovers '?' Icon]
       |
       +--> (150ms Debounce Timer) --> Spawns RichPopoverWidget(QFrame)
                                       Anchored 8px Right of '?' Icon
                                       |
       +-------------------------------+
       |
[User Moves Cursor Off '?' Icon]
       |
       +--> Starts 300ms Grace Timer
              |-- If Cursor Enters RichPopoverWidget within 300ms:
              |     Cancel Grace Timer -> Keep Popover Open for Reading & Link Clicking
              |     [Click Link: cppidoc://tech#risk-aversion-rho]
              |            |
              |            +--> Opens/Focuses DocWorkbenchDockWidget at Exact Section
              |
              +-- If 300ms Expires with Cursor Outside Both '?' and Popover:
                    Fade Out & Close Popover (Unless User Clicked [Pin] Button)
```

#### Example Content Specification for the Risk-Aversion Parameter ($\rho$) `?` Popover:
* **Title:** Risk-Aversion Parameter ($\rho \in [0, 1]$) — Default: `0.25`
* **Summary:** Tunes how conservatively the strategy chooses between passive `post_only` maker orders ($0.15\%$ fee, stochastic fill probability $P_{\text{fill}, t}$) and immediate taker orders ($0.30\%$ fee, guaranteed fill).
* **Boundary Behavior:**
  * $\rho = 0.0$: Places **only** passive maker orders above the buffered floor, and **only places a taker sell order when the liquidation level + safety buffer ($P_{\text{floor}, t}$) is reached** (where the realized sell price after slippage lands below $P_{\text{floor}, t}$ but $\ge P_{\text{liq}}$).
  * $\rho = 1.0$: Places **only** immediate taker orders for every buy and sell rebalance.
  * $0.0 \lt \rho \lt 1.0$: Splits order size ($\rho$ taker, $1-\rho$ maker) and escalates to $100\%$ taker whenever execution urgency $u_t \ge 1 - \rho$.
* **Interactive Links:**
  * `[View Implementation in Technical Docs (§2.4) ↗](cppidoc://tech#sec-2-4-rho)`
  * `[View Derivation in Theoretical Strategy Docs (§2.4) ↗](cppidoc://theory#sec-2-4-rho)`

---

## 5. Dual Interlinked Documentation & Live Counterpart Preview System

### 5.1 Canonical Section Mapping (`doc_registry.json`)
Both the **Theoretical Strategy Documentation (`theory.html`)** and the **Technical Implementation Documentation (`tech.html`)** share an identical 8-section structural spine linked by `data-concept-id`:

| `data-concept-id` | Theoretical Strategy Documentation (`cppidoc://theory#...`) | Technical Implementation Documentation (`cppidoc://tech#...`) |
| :--- | :--- | :--- |
| `sec-1-floor-buffer` | **§1.1 CPPI Invariant Wealth Floor & Safety Buffer Derivation:** Proves why $P_{\text{floor}, t} = \max\big(\frac{P_{\text{liq}}}{1-\gamma}, \frac{F-c_t}{n_t(1-\gamma)}\big)$ guarantees $V_t \ge F$ and complete exit before $P_{\text{liq}}$. | **§1.1 Floor & Safety Buffer State Kernel:** Exact `float64` calculation of `p_floor`, `safety_buf_usd`, and intra-bar `low[t] <= p_floor` execution priority. |
| `sec-1-whipsaw-cost` | **§1.2 Whipsaw Decay as Dynamic Put Replication Cost:** Derives how selling low and rebuying higher reduces $n_{\text{max}, t}$ and raises $\Delta P_{\text{buf}, t}$. | **§1.2 Cushion & Recoverable BTC Accounting:** Implementation of `n_max` and cumulative whipsaw bleed telemetry tracking. |
| `sec-2-dual-envelope` | **§2.1 Dual-Envelope Control Law & Hysteresis Deadband:** Derives $n_{\text{ceil}, t}$ and $n_{\text{floor}, t}$ from normalized distance $d_t$ and proves deadband stability. | **§2.1 Envelope Evaluation & Minimum Rebalance Quantum:** Implementation of $\Delta n_{\text{min}}$ filtering and near-floor buffer override ($P_t \le P_{\text{floor}, t} + 250$). |
| `sec-2-4-rho` | **§2.2 Continuous Risk-Aversion ($\rho$) & Urgency Escalation Theory:** Formulates the trade-off between maker fee savings and missed-fill crash exposure. | **§2.2 Order Routing Engine (`route_order_by_rho`):** Exact branching logic for $\rho = 0$, $0 \lt \rho \lt 1$, and $\rho = 1$, plus miss-counter $m_{\text{miss}, t}$ escalation. |
| `sec-3-liquidity` | **§3.1 Stochastic Order-Book Liquidity & Pass-Through Theory:** Derives the 3-factor Poisson/exponential fill model $P_{\text{fill}, t} = \Phi_{\text{pen}} \cdot \Phi_{\text{vel}} \cdot \Phi_{\text{size}}$. | **§3.1 Stochastic Fill Sampler & PCG64 Seeding:** Implementation of $\lambda_{\text{liq}}, \beta_{\text{vel}}, \alpha_{\text{size}}$, deterministic `PCG64` uniform variate arrays, and Monte Carlo fan threads. |
| `sec-4-signals` | **§4.1 Asymmetric Decline ($S_{\text{down}}$) & Growth ($S_{\text{up}}$) Microstructure Signals:** Explains volatility-normalized EMAs, taker flow imbalance, and the crash-veto gate. | **§4.1 Vectorized Signal Pipeline:** Polars/NumPy implementation of rolling EWM, ATR, CLV volume synthesis, and $O(T)$ pre-computation. |

### 5.2 Synchronized Counterpart Preview Architecture
Inside `DocWorkbenchDockWidget`, a `QSplitter` hosts two `QWebEngineView` instances connected via a shared `DocSyncBridge(QObject)` over `QWebChannel`:

1. **Primary Documentation Viewer (`primary_view`, 65% area):**
   * Renders whichever manual the user opened (`Theoretical Strategy` or `Technical Implementation`).
   * Features a top toolbar with `Back`, `Forward`, `Switch Primary Volume (Theory <-> Technical)`, `Search`, and `Toggle Counterpart Preview Split`.
2. **Counterpart Preview Viewer (`preview_view`, 35% area):**
   * Automatically loads the *opposite* documentation volume (`tech.html` when viewing `theory.html`, and vice versa).
   * **Scroll-Spy Synchronization:** A JavaScript `IntersectionObserver` inside `primary_view` detects which `<section data-concept-id="...">` occupies the top reading viewport and invokes `bridge.notifySectionFocused(concept_id)`. `DocSyncBridge` immediately scrolls `preview_view` to the matching `data-concept-id` in the counterpart manual and applies a subtle left-border highlight.
   * **Hover Link Peek:** Hovering over any inline hyperlink `<a href="cppidoc://tech#sec-3-liquidity">` inside `primary_view` immediately previews that exact target anchor inside `preview_view` (and displays a floating snippet card if `preview_view` is collapsed) before the user clicks. Clicking the link swaps or navigates the primary viewer directly to that entry.

---

## 6. Module Directory Structure & Implementation Entry Points

```
cppi_lab/
├── main.py                         # QApplication bootstrap, High-DPI config, theme loader
├── config/
│   ├── defaults.py                 # Dataclass of all 25 default parameters, ranges & metadata
│   └── doc_registry.json           # Canonical concept_id mapping & ? popover content definitions
├── data/
│   ├── btc_usd_1m.parquet          # Bundled historical 1-minute BTC-USD dataset
│   └── loader.py                   # Polars Parquet loader, CSV importer & Coinbase REST fetcher
├── engine/
│   ├── signals.py                  # Vectorized S_down(t) and S_up(t) pre-computation
│   ├── kernel_numba.py             # @njit CPPI state machine, rho router & stochastic fill model
│   └── monte_carlo.py              # Parallel M=100 seed ensemble runner (@njit(parallel=True))
├── ui/
│   ├── main_window.py              # QMainWindow + QtAds CDockManager orchestration
│   ├── control_sidebar.py          # Parameter sliders, spinboxes, reset buttons & ? badges
│   ├── kpi_ribbon.py               # Top 8 real-time telemetry summary cards
│   ├── chart_workspace.py          # 3-tier synchronized OpenGL PyQtGraph canvas & event markers
│   ├── ledger_window.py            # Floating/Dockable QTableView + virtualized TelemetryTableModel
│   ├── popover_manager.py          # 300ms hover-intent sticky ? popovers with MathJax & deep links
│   └── doc_workbench.py            # Dual QWebEngineView documentation viewer + live counterpart preview
└── docs/
    ├── theory.html                 # Complete Theoretical Strategy Documentation (KaTeX)
    ├── tech.html                   # Complete Technical Implementation Documentation (KaTeX)
    └── assets/                     # Local KaTeX CSS/JS/fonts & QWebChannel synchronization script
```