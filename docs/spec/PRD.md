# Product Requirements Document (PRD): CPPI-Lab
**Automated Bitcoin CPPI Trading Strategy Back-Testing & Educational Workbench**

* **Document Version:** 2.0.0
* **Date:** November 16, 2026
* **Target Platform:** Desktop Application (Windows 11 / Linux WSLg / macOS)
* **Status:** Approved for Engineering Implementation

---

## 1. Executive Summary & Product Vision

**CPPI-Lab** is an institutional-grade quantitative back-testing environment and interactive educational workbench designed to simulate, visualize, and analyze an automated **Signal-Modulated Constant Proportion Portfolio Insurance (CPPI)** strategy with a **Hysteresis Deadband** on real historical Bitcoin (`BTC-USD`) market data.

The strategy is designed around an invariant primary wealth constraint: for a user-specified initial Bitcoin balance $N_0$, initial cash balance $c_0$, and baseline liquidation price $P_{\text{liq}}$ (default $\$60,000$), the strategy **must guarantee complete liquidation before the market price reaches below $P_{\text{liq}}$** and ensure that the net USD equivalent portfolio value $V_t$ at every timestep $t$ remains at least equal to the hard floor $F = N_0 \cdot P_{\text{liq}} + c_0$. Above the buffered liquidation trigger, the strategy dynamically sells in proportion to both proximity to the floor and a Decline Signal $S_{\text{down}}(t)$, and buys back in proportion to the distance above the floor and a Growth Signal $S_{\text{up}}(t)$, treating losses from selling low and buying back higher ("whipsaw decay") as an explicit, bounded insurance cost.

To capture realistic exchange microstructure, CPPI-Lab incorporates:
1. **Asymmetric Maker and Taker Transaction Costs:** Defaulting to $f_{\text{maker}} = 0.15\%$ and $f_{\text{taker}} = 0.30\%$, plus a safety slippage buffer $\delta_{\text{slip}}$ (default $0.10\%$).
2. **Stochastic Order-Book Liquidity Modeling:** Computing a per-bar fill probability $P_{\text{fill}, t} \in [0, 1]$ for passive `post_only` maker orders and resolving execution via a uniform random variate $U_t \sim \mathcal{U}(0, 1)$ ($U_t \le P_{\text{fill}, t}$) to model price sweeping past limit orders without filling due to rapid price velocity or insufficient counter-party volume.
3. **Continuous Risk-Aversion Execution Control ($\rho \in [0, 1]$):** Allowing the user to tune how conservatively the strategy routes orders between passive maker limits and guaranteed-fill taker orders, bounded such that at $\rho = 0$, the strategy only places a taker sell order when the **liquidation level + safety buffer** ($P_{\text{floor}, t}$) is reached (where realized post-fee/post-slippage execution price lands below the trigger $P_{\text{floor}, t}$ but strictly at or above $P_{\text{liq}}$), whereas at $\rho = 1$, all rebalancing orders execute as immediate taker orders.
4. **Single-Screen Simultaneous Visualization & Multi-Window Docking:** Presenting all time-series metrics simultaneously on one screen without vertical scrolling, backed by a movable, resizable, always-on-top (and dockable) tabular ledger window and a cross-linked dual documentation system with live counterpart previewing.

---

## 2. Target User Personas & Use Cases

| Persona | Primary Objective | Key Workflows in CPPI-Lab |
| :--- | :--- | :--- |
| **Quantitative Strategist / Principal Trader** | Optimize CPPI parameters, signal windows, and execution risk-aversion $\rho$ prior to deploying live capital on an exchange. | Loads historical crash/chop windows, adjusts $\rho$ and liquidity parameters ($\lambda_{\text{liq}}, \beta_{\text{vel}}, \alpha_{\text{size}}$), inspects missed maker fills vs. fee bleed, and verifies zero floor breaches across Monte Carlo seeds. |
| **Quantitative Finance Researcher / Student** | Master the mathematical theory and algorithmic mechanics of dynamic option replication, hysteresis control, and microstructure fill modeling. | Hovers over `?` badges to inspect governing equations, navigates bidirectionally between the Theoretical Strategy Documentation and Technical Implementation Documentation, and steps through the floating tabular ledger row-by-row. |

---

## 3. Core Strategy & Mathematical Invariants

### 3.1 Invariant Primary Constraint (Hard Wealth Floor & Buffered Trigger)
1. **Hard Net-Value Floor ($F$):** At all times $t \in [0, T]$, the liquidation-adjusted net USD equivalent value of the account must satisfy:
   $$V_t \ge F = N_0 \cdot P_{\text{liq}} + c_0$$
2. **Liquidation Level + Safety Buffer ($P_{\text{floor}, t}$):** Because executing an emergency liquidation incurs a taker fee $f_{\text{taker}}$ and potential market impact / intra-bar slippage $\delta_{\text{slip}}$ (total emergency exit friction $\gamma = f_{\text{taker}} + \delta_{\text{slip}}$), triggering a taker liquidation only after price hits $P_{\text{liq}}$ would result in net proceeds below $F$. Therefore, the strategy defines the active trigger level as **Liquidation Level + Safety Buffer**:
   $$P_{\text{floor}, t} = P_{\text{liq}} + \Delta P_{\text{buf}, t} = \max\left(\frac{P_{\text{liq}}}{1 - \gamma},\; \frac{\max(0,\; F - c_t)}{n_t (1 - \gamma)}\right)$$
   * **Baseline Safety Buffer:** When no prior whipsaw deficit exists, $\Delta P_{\text{buf}, 0} = P_{\text{liq}} \cdot \frac{\gamma}{1 - \gamma}$.
   * **Dynamic Whipsaw Buffer Expansion:** Whenever the strategy sells BTC and buys back at a higher price, the realized whipsaw cost consumes part of the risk cushion $C_t = V_t - F$ and automatically raises $\Delta P_{\text{buf}, t}$ so that liquidating the remaining BTC $n_t$ at $P_{\text{floor}, t}$ (even if the realized fill lands below $P_{\text{floor}, t}$ by the friction margin $\gamma$) always locks in net USD value $\ge F$ before market price drops below $P_{\text{liq}}$.

### 3.2 Proportional Selling & Buying with Separate Signals
Let $d_t \in [0, 1]$ denote the normalized price distance above $P_{\text{floor}, t}$ relative to an upper full-allocation reference price $P_{\text{ref}}$:
$$d_t = \text{clip}\left(\frac{P_t - P_{\text{floor}, t}}{P_{\text{ref}} - P_{\text{floor}, t}},\; 0.0,\; 1.0\right)$$
1. **Guaranteed Complete Sale by Floor:** As $P_t \to P_{\text{floor}, t}$, $d_t \to 0$, forcing the maximum permissible BTC ceiling $n_{\text{ceil}, t} \to 0$ regardless of signal values.
2. **Proportional Selling Above Floor:** Above $P_{\text{floor}, t}$, the sell ceiling contracts below the linear glidepath in direct proportion to both the proximity to the buffered floor $(1 - d_t)$ and the strength of the Decline Signal $S_{\text{down}}(t) \in [0, 1]$:
   $$n_{\text{ceil}, t} = n_{\text{max}, t} \cdot d_t \cdot \Big(1 - k_{\text{sell}} \cdot (1 - d_t) \cdot S_{\text{down}}(t)\Big)$$
3. **Proportional Buying Above Floor:** On market recoveries where the Decline Signal has subsided, the strategy buys back BTC up to a lower rebuy floor $n_{\text{floor}, t}$ in direct proportion to the distance above the buffered floor $d_t$ and the strength of the Growth Signal $S_{\text{up}}(t) \in [0, 1]$:
   $$n_{\text{floor}, t} = n_{\text{max}, t} \cdot d_t \cdot S_{\text{up}}(t)$$

---

## 4. Functional Requirements (FR)

### FR-1: Historical Market Data Engine
* **FR-1.1 (Bundled Real Historical Data):** The application must ship with a pre-packaged, high-resolution 1-minute historical `BTC-USD` dataset (spanning multi-year market regimes including major drawdowns, sideways consolidations, and V-recoveries) containing `timestamp_utc`, `open`, `high`, `low`, `close`, `buy_taker_vol`, and `sell_taker_vol`.
* **FR-1.2 (External Data Import & Live API Sync):** Users must be able to import custom historical CSV/Parquet files (Coinbase `market_trades`/candles or Binance `aggTrades`/`klines`) or fetch updated historical candles via the Coinbase Advanced Trade REST API.
* **FR-1.3 (Historical Price Rescaling Tool):** Include an optional toggle (`Normalize Start Price to Reference Anchor`) allowing the user to multiplicatively rescale any historical crash window so its starting price aligns with a chosen reference level above $P_{\text{liq}}$ (e.g., starting at $\$68,000$ relative to a $\$60,000$ floor) while preserving exact historical percentage returns, volatility clustering, and volume imbalance.

### FR-2: Parameter Configuration & State Reset
* **FR-2.1 (Default Initialization):** On launch, all strategy, fee, liquidity, and signal parameters must initialize to validated default values (defined in Section 5).
* **FR-2.2 (Interactive Adjustment & Granular Reset):** Every parameter must be editable via synchronized slider and numeric spinbox controls. Each parameter row must feature an individual `Reset to Default` micro-button, and the sidebar footer must provide a global `Reset All Parameters to Defaults` button.
* **FR-2.3 (Instant & Deterministic Recalculation):** Modifying any parameter must re-run the back-test across a 30-day 1-minute window ($43,200$ bars) in under $50\text{ ms}$ and refresh all plots and tables automatically.

### FR-3: Transaction Costs, Risk-Aversion ($\rho$), and Stochastic Liquidity Model
* **FR-3.1 (Configurable Maker & Taker Fees):** Expose Maker Fee $f_{\text{maker}}$ (default $0.15\%$), Taker Fee $f_{\text{taker}}$ (default $0.30\%$), and Safety Slippage Buffer $\delta_{\text{slip}}$ (default $0.10\%$).
* **FR-3.2 (Stochastic Maker Order Fill Model):**
  * Whenever a passive `post_only` maker buy or sell order is submitted at bar $t-1$, the engine evaluates at bar $t$ a fill probability $P_{\text{fill}, t} \in [0, 1]$ governed by user-tunable liquidity parameters: Base Liquidity Depth ($\lambda_{\text{liq}}$), Adverse Price Velocity Penalty ($\beta_{\text{vel}}$), and Order Size Impact Exponent ($\alpha_{\text{size}}$).
  * The engine draws a uniform variate $U_t \sim \mathcal{U}(0, 1)$ using a user-configurable deterministic PRNG seed ($s_{\text{rng}}$, default `42`).
  * If $U_t \le P_{\text{fill}, t}$, the maker order fills at its limit price paying $f_{\text{maker}}$.
  * If $U_t \gt P_{\text{fill}, t}$, the order fails to fill ("liquidity pass-through / missed limit") and is logged as an explicit missed-order event.
* **FR-3.3 (Monte Carlo Ensemble Overlay):** Provide a toggle (`Show Monte Carlo Liquidity Fan, M = 100 Seeds`) that runs $100$ independent uniform-variate trajectories in parallel and renders the $5\text{th}\text{--}95\text{th}$ percentile equity band behind the active seed's trajectory.
* **FR-3.4 (Risk-Aversion Parameter $\rho \in [0, 1]$):**
  * Expose a continuous parameter $\rho \in [0, 1]$ (default $0.25$) tuning execution conservatism between passive maker orders and immediate taker orders:
  * **At $\rho = 0.0$:** The strategy places **100% passive maker orders** for all rebalancing buys and sells above $P_{\text{floor}, t}$, and **only places a taker sell order when the liquidation level + safety buffer ($P_{\text{floor}, t}$) is reached**—where the realized execution price after slippage/intra-bar gap may land below $P_{\text{floor}, t}$, within the safety buffer $[P_{\text{liq}}, P_{\text{floor}, t}]$, preserving $V_t \ge F$.
  * **At $\rho = 1.0$:** The strategy places **100% immediate taker orders** ($P_{\text{fill}} = 1.0$, paying $f_{\text{taker}} = 0.30\%$) whenever any buy or sell criterion is met.
  * **At $0.0 \lt \rho \lt 1.0$:** The strategy dynamically splits or escalates orders from maker to taker whenever trade urgency (proximity to $P_{\text{floor}, t}$, signal intensity, and consecutive missed maker fills) exceeds the conservatism threshold $(1 - \rho)$.

### FR-4: Single-Screen Simultaneous Time-Series Visualization
* **FR-4.1 (No-Scroll Single-Screen Mandate):** The main application window must display all parameter controls, summary KPI telemetry cards, and all time-series plots simultaneously within a single screen without requiring vertical scrolling or tab switching.
* **FR-4.2 (Three-Pane Vertically Synchronized Canvas):**
  * **Pane 1 (Top — Price, Floor & Net Portfolio Equity):** Displays BTC price (`Close` line with `[Low, High]` range shading), the fixed baseline Liquidation Price $P_{\text{liq}}$, the dynamic **Liquidation Level + Safety Buffer** curve $P_{\text{floor}, t}$, the Net USD Portfolio Value $V_t$, and distinct semantic markers for Maker Buys, Maker Sells, Taker Buys, Taker Sells, Missed Maker Orders (Liquidity Pass-Throughs), and Hard Floor Liquidations.
  * **Pane 2 (Middle — BTC Holdings & Hysteresis Deadband):** Displays actual BTC holdings $n_t$, maximum recoverable BTC $n_{\text{max}, t}$, Upper Sell Ceiling $n_{\text{ceil}, t}$, Lower Rebuy Floor $n_{\text{floor}, t}$, and the shaded No-Trade Hysteresis Deadband between $[n_{\text{floor}, t}, n_{\text{ceil}, t}]$.
  * **Pane 3 (Bottom — Separate Signals & Fill Probability Diagnostics):** Displays Decline Signal $S_{\text{down}}(t) \in [0, 1]$, Growth Signal $S_{\text{up}}(t) \in [0, 1]$, and vertical stems at each maker order attempt showing Fill Probability $P_{\text{fill}, t}$ vs. the drawn uniform variate $U_t$.
* **FR-4.3 (Automated Visual Hierarchy & Event Callouts):** Styling, color palettes, line weights, and marker geometries must be automatically configured to maximize contrast for critical events (such as missed maker sells during a fast crash or buffer liquidations) while keeping background context lines legible and unobstructed.

### FR-5: Floating & Dockable Tabular Data Popup Window
* **FR-5.1 (Window Behavior):** The full time-step simulation dataset must be accessible in a separate popup window that:
  1. Can be freely **moved and resized** anywhere on screen using the mouse.
  2. **Remains on top** of the main application window when floating.
  3. Can be **docked** into split panels on the left, right, or bottom of the main application window via drag-and-drop.
  4. Permits vertical and horizontal scrolling within the table viewport.
* **FR-5.2 (Table Features):** Displays all calculated quantities at every timestep $t$, supports instant filtering (`All Timesteps`, `Executed Trades`, `Missed Maker Orders`, `Floor Liquidations`), column sorting, bidirectional row-to-chart crosshair synchronization, and CSV/Parquet export.

### FR-6: Universal `?` Hover Popovers & Dual Interlinked Documentation System
* **FR-6.1 (Interactive `?` Hover Popovers):**
  * Every UI parameter, KPI metric card, and chart pane header must feature an adjacent `?` icon.
  * Hovering the mouse over `?` opens a rich, formatted popup (`RichPopoverWidget`) explaining the mathematical definition and implementation behavior of that item (including rendered LaTeX equations).
  * The popup uses a $300\text{ ms}$ hover-intent grace timer so the user can move the mouse cursor directly into the popup body to click embedded hyperlinks (`Open in Technical Documentation` or `Open in Strategy Theory Documentation`) or pin the popup as a movable floating card.
* **FR-6.2 (Dual Documentation Volumes):**
  1. **Theoretical Strategy Documentation:** Complete mathematical derivations of the CPPI wealth floor, safety buffer sizing, dual-envelope hysteresis control law, asymmetric momentum/order-flow signals, continuous risk-aversion routing ($\rho$), and the stochastic queue-penetration/velocity liquidity model.
  2. **Technical Implementation Documentation:** Complete algorithmic specifications, Numba state-machine loops, data schemas, floating-point precision guards, PRNG seeding architecture, and UI event bindings corresponding to each theoretical concept.
* **FR-6.3 (Bidirectional Hyperlinks & Live Counterpart Preview):**
  * Every section in the Theoretical Documentation is hyperlinked to its exact counterpart in the Technical Documentation, and vice versa.
  * The Documentation Viewer window (which is also movable, resizable, always-on-top when floating, and dockable into main window side/bottom panels) features a **Live Counterpart Preview Pane** that automatically previews the corresponding section of the *other* documentation volume as the user reads or hovers over links in the primary viewer.

---

## 5. Complete UI Parameter Specification & Default Values

| Group | UI Label | Math Symbol | Default Value | Allowed Range | Step Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Account & Horizon** | Start Date & Time (UTC) | $t_0$ | `2026-01-01 00:00` | Dataset Min to Max | $1\text{ min}$ |
| **Account & Horizon** | Simulation Duration (Days) | $\Delta T$ | `30 Days` | $1$ to $365\text{ Days}$ | $1\text{ Day}$ |
| **Account & Horizon** | Initial BTC Holdings | $N_0$ | `1.0000 BTC` | $[0.001, 1000.0]$ | $0.05\text{ BTC}$ |
| **Account & Horizon** | Initial USD Cash | $c_0$ | `$0.00` | $[\$0, \$10,000,000]$ | $\$100.00$ |
| **CPPI Floor & Band** | Hard Liquidation Floor Price | $P_{\text{liq}}$ | `$60,000.00` | $[\$1,000, \$250,000]$ | $\$250.00$ |
| **CPPI Floor & Band** | Upper Full-Allocation Reference | $P_{\text{ref}}$ | `$72,000.00` | $[P_{\text{liq}} + \$1,000, \$350,000]$ | $\$500.00$ |
| **CPPI Floor & Band** | Minimum Rebalance Threshold | $\Delta n_{\text{min}}$ | `3.0%` of $N_0$ | $[0.1\%, 25.0\%]$ | $0.25\%$ |
| **Fees & Execution** | Maker Fee Rate | $f_{\text{maker}}$ | `0.15%` | $[0.00\%, 1.00\%]$ | $0.01\%$ |
| **Fees & Execution** | Taker Fee Rate | $f_{\text{taker}}$ | `0.30%` | $[0.00\%, 2.00\%]$ | $0.01\%$ |
| **Fees & Execution** | Safety Buffer Slippage Allowance | $\delta_{\text{slip}}$ | `0.10%` | $[0.00\%, 2.00\%]$ | $0.01\%$ |
| **Fees & Execution** | **Risk-Aversion Parameter** | $\rho$ | `0.25` | $[0.00, 1.00]$ | $0.05$ |
| **Liquidity Model** | Base Liquidity Depth Factor | $\lambda_{\text{liq}}$ | `2.50` | $[0.10, 10.00]$ | $0.10$ |
| **Liquidity Model** | Adverse Price Velocity Penalty | $\beta_{\text{vel}}$ | `1.80` | $[0.00, 10.00]$ | $0.10$ |
| **Liquidity Model** | Order Size Impact Exponent | $\alpha_{\text{size}}$ | `1.20` | $[0.10, 5.00]$ | $0.10$ |
| **Liquidity Model** | Uniform PRNG Seed | $s_{\text{rng}}$ | `42` | $[1, 9999999]$ | $1$ |
| **Liquidity Model** | Monte Carlo Fan Overlay ($M=100$) | $\text{MC}_{\text{fan}}$ | `Off (False)` | `On / Off` | Toggle |
| **Decline Signal ($S_{\text{down}}$)** | Fast / Slow EMA Windows | $W_{f,d}, W_{s,d}$ | `5m / 30m` | $[1, 60] \text{ / } [10, 240]$ | $1\text{ min}$ |
| **Decline Signal ($S_{\text{down}}$)** | Volatility ATR Window | $W_{\text{atr},d}$ | `15m` | $[5, 120]$ | $1\text{ min}$ |
| **Decline Signal ($S_{\text{down}}$)** | Trade Flow Imbalance Window | $W_{\text{imb},d}$ | `10m` | $[2, 60]$ | $1\text{ min}$ |
| **Decline Signal ($S_{\text{down}}$)** | Momentum vs. Imbalance Weight | $w_{m,d}$ | `0.60` | $[0.00, 1.00]$ | $0.05$ |
| **Decline Signal ($S_{\text{down}}$)** | Decline Signal Gain | $k_{\text{sell}}$ | `1.00` | $[0.10, 2.00]$ | $0.05$ |
| **Growth Signal ($S_{\text{up}}$)** | Fast / Slow EMA Windows | $W_{f,u}, W_{s,u}$ | `15m / 60m` | $[5, 120] \text{ / } [30, 480]$ | $5\text{ min}$ |
| **Growth Signal ($S_{\text{up}}$)** | Volatility ATR Window | $W_{\text{atr},u}$ | `60m` | $[15, 240]$ | $5\text{ min}$ |
| **Growth Signal ($S_{\text{up}}$)** | Range Breakout Window | $W_{\text{brk},u}$ | `240m` | $[30, 1440]$ | $15\text{ min}$ |
| **Growth Signal ($S_{\text{up}}$)** | Decline Veto Gate Threshold | $\tau_{\text{veto}}$ | `0.50` | $[0.10, 1.00]$ | $0.05$ |

---

## 6. Non-Functional & Quality Requirements

1. **Performance & Latency:**
   * Single-trajectory Numba simulation across $43,200$ 1-minute bars (30 days) must complete in $\le 15\text{ ms}$ on a single CPU core.
   * Monte Carlo 100-seed ensemble simulation ($4.32 \times 10^6$ bar evaluations) must complete in $\le 120\text{ ms}$ using `@njit(parallel=True)`.
   * OpenGL chart pan/zoom and crosshair inspection must maintain $60\text{ FPS}$.
2. **Numerical Precision & Accounting Integrity:**
   * All portfolio accounting ($n_t, c_t, V_t, P_{\text{floor}, t}$) must execute in IEEE 754 `float64` precision with strict conservation assertions verifying that $V_t \ge F - 10^{-6}$ across all standard continuous-diffusion bars.
3. **Ergonomics & Visual Polish:**
   * High-contrast dark institutional theme (`#0B0F19` canvas background, `#111827` panel surfaces, `#1E293B` subtle gridlines) engineered for prolonged analytical sessions without eye strain.