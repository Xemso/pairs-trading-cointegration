# Statistical Arbitrage: Cointegration-Based Pairs Trading

A research-grade implementation of a market-neutral pairs trading strategy built on the
Engle-Granger two-step cointegration framework, with a fully vectorized event-driven-style
backtester including transaction costs and institutional performance metrics.

---

## Executive Summary

Pairs trading is a **market-neutral statistical arbitrage** strategy. Instead of forecasting
the direction of the market, it exploits the *relative* mispricing between two economically
linked assets (e.g. two oil majors, two ETFs tracking similar indices).

The pipeline:

1. **Universe screening** — download adjusted close prices for a candidate universe.
2. **Cointegration testing** — identify pairs whose linear combination is stationary
   (Engle-Granger two-step method, ADF test on residuals).
3. **Signal generation** — trade the z-score of the spread: short the spread when it is
   abnormally high, long when abnormally low, exit on mean reversion.
4. **Backtesting** — vectorized P&L simulation with proportional transaction costs,
   Sharpe / Sortino ratios, maximum drawdown, and equity curve plots.

The strategy is **dollar-neutral by construction**: for every long position in asset A we hold
a short position of $\beta$ units of asset B, hedging out market beta.

---

## Mathematical Framework

### 1. Cointegration (Engle-Granger, 1987)

Two non-stationary $I(1)$ price series $P^A_t$ and $P^B_t$ are **cointegrated** if there
exists $\beta$ such that the spread

$$
S_t = \log P^A_t - \beta \log P^B_t - \alpha
$$

is stationary, i.e. $S_t \sim I(0)$. We work in **log prices** so that $\beta$ is a
scale-free elasticity and the spread is symmetric in relative returns.

**Step 1 — Hedge ratio estimation (OLS):**

$$
\log P^A_t = \alpha + \beta \log P^B_t + \varepsilon_t
\quad\Rightarrow\quad
\hat\beta = \frac{\mathrm{Cov}(\log P^A, \log P^B)}{\mathrm{Var}(\log P^B)}
$$

**Step 2 — Stationarity of residuals (Augmented Dickey-Fuller):**

$$
\Delta \hat\varepsilon_t = \gamma\, \hat\varepsilon_{t-1}
 + \sum_{i=1}^{p} \phi_i\, \Delta\hat\varepsilon_{t-i} + u_t
$$

We test $H_0: \gamma = 0$ (unit root, no cointegration) against $H_1: \gamma < 0$.
The pair is retained if the ADF p-value is below a threshold (default $5\%$).
Note: because $\hat\varepsilon_t$ is *estimated*, standard ADF critical values are slightly
too liberal (Phillips-Ouliaris correction) — an acknowledged limitation discussed in
*Next Steps*.

### 2. Mean-reversion dynamics (Ornstein-Uhlenbeck)

A stationary spread is naturally modeled as an OU process:

$$
dS_t = \theta(\mu - S_t)\,dt + \sigma\, dW_t
$$

The **half-life of mean reversion** follows from the discretized AR(1) representation
$S_t = a + b\,S_{t-1} + u_t$ with $b = e^{-\theta \Delta t}$:

$$
t_{1/2} = \frac{\ln 2}{\theta} = -\frac{\ln 2}{\ln b}
$$

The half-life is used both as a **pair-quality filter** (reject pairs that revert too slowly)
and to size the rolling window of the z-score.

### 3. Trading signal (z-score)

Using a rolling window of $w$ observations:

$$
z_t = \frac{S_t - \bar{S}_{t,w}}{\hat\sigma_{t,w}}
$$

| Condition | Action |
|---|---|
| $z_t < -z_{\text{entry}}$ | **Long** spread (long A, short $\beta$ B) |
| $z_t > +z_{\text{entry}}$ | **Short** spread (short A, long $\beta$ B) |
| $\lvert z_t \rvert < z_{\text{exit}}$ | Close position |
| $\lvert z_t \rvert > z_{\text{stop}}$ | Stop-loss (structural break protection) |

Signals are shifted by one bar before execution to avoid **look-ahead bias**.

### 4. Performance metrics

With daily strategy returns $r_t$ net of costs, and $A = 252$ trading days:

$$
\text{Sharpe} = \sqrt{A}\;\frac{\mathbb{E}[r_t] - r_f/A}{\sigma(r_t)}
\qquad
\text{Sortino} = \sqrt{A}\;\frac{\mathbb{E}[r_t] - r_f/A}{\sigma(r_t \mid r_t < 0)}
$$

$$
\text{MaxDD} = \min_t \left( \frac{E_t}{\max_{s \le t} E_s} - 1 \right),
\qquad
\text{Calmar} = \frac{\text{CAGR}}{\lvert\text{MaxDD}\rvert}
$$

Transaction costs are proportional: each unit of turnover is charged
$c$ basis points, $r_t^{\text{net}} = r_t^{\text{gross}} - c \cdot \text{turnover}_t$.

---

## Project Architecture

```
pairs-trading-cointegration/
├── README.md
├── requirements.txt
├── main.py                  # Entry point: full pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data acquisition & cleaning (yfinance)
│   ├── strategy.py          # Cointegration engine + signal generation
│   └── backtester.py        # Vectorized backtest & performance analytics
└── results/                 # Generated plots & metrics (created at runtime)
```

## Installation

```bash
git clone https://github.com/<your-username>/pairs-trading-cointegration.git
cd pairs-trading-cointegration
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Default configuration (editable at the top of `main.py`): a universe of liquid US energy
and financial tickers, 2018-2024 in-sample screening, out-of-sample backtest, 5 bps
transaction costs. Outputs metrics to the console and saves plots to `results/`.

To customize:

```python
from src.data_loader import DataLoader
from src.strategy import PairsTradingStrategy
from src.backtester import Backtester

prices = DataLoader(["XOM", "CVX"], start="2018-01-01").load()
strategy = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, zscore_window=60)
result = strategy.run(prices["XOM"], prices["CVX"])
report = Backtester(cost_bps=5.0).run(result)
print(report.summary())
```

## Disclaimer

This repository is for research and educational purposes only and does not constitute
investment advice. Past (backtested) performance is not indicative of future results.
