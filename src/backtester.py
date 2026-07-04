"""Vectorized backtesting engine and performance analytics.

The backtester converts spread positions into dollar-neutral leg weights,
computes daily portfolio returns net of proportional transaction costs,
and produces institutional performance statistics (Sharpe, Sortino,
maximum drawdown, Calmar, hit rate, turnover).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy import StrategyResult

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR: int = 252


class BacktestError(Exception):
    """Raised when a backtest cannot be computed on the given inputs."""


@dataclass(frozen=True)
class BacktestReport:
    """Container for backtest outputs.

    Attributes:
        equity_curve: Cumulative net equity (base 1.0).
        net_returns: Daily strategy returns net of costs.
        gross_returns: Daily strategy returns before costs.
        metrics: Dictionary of scalar performance statistics.
    """

    equity_curve: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    metrics: dict[str, float]

    def summary(self) -> str:
        """Human-readable, aligned summary of the performance metrics."""
        lines = ["", "=" * 46, "  BACKTEST PERFORMANCE SUMMARY", "=" * 46]
        fmt = {
            "total_return": ("Total return", "{:+.2%}"),
            "cagr": ("CAGR", "{:+.2%}"),
            "annual_vol": ("Annualized volatility", "{:.2%}"),
            "sharpe": ("Sharpe ratio", "{:.2f}"),
            "sortino": ("Sortino ratio", "{:.2f}"),
            "max_drawdown": ("Maximum drawdown", "{:.2%}"),
            "calmar": ("Calmar ratio", "{:.2f}"),
            "hit_rate": ("Hit rate (active days)", "{:.2%}"),
            "avg_turnover": ("Avg daily turnover", "{:.2%}"),
            "n_trades": ("Round-trip trades", "{:.0f}"),
            "time_in_market": ("Time in market", "{:.2%}"),
        }
        for key, (label, pattern) in fmt.items():
            if key in self.metrics:
                lines.append(f"  {label:<26}{pattern.format(self.metrics[key])}")
        lines.append("=" * 46)
        return "\n".join(lines)


class Backtester:
    """Simulate a dollar-neutral pairs strategy from spread positions.

    Position convention: a spread position ``s in {-1, 0, +1}`` maps to
    leg weights ``w_A = s * 1/(1+|beta|)`` and ``w_B = -s * |beta|/(1+|beta|)``
    (sign-adjusted for the hedge ratio), so gross leverage never exceeds 1.

    Attributes:
        cost_bps: Proportional transaction cost per unit of turnover,
            in basis points (covers commissions + half-spread).
        risk_free_rate: Annualized risk-free rate used in Sharpe/Sortino.
    """

    def __init__(self, cost_bps: float = 5.0, risk_free_rate: float = 0.0) -> None:
        if cost_bps < 0:
            raise ValueError("`cost_bps` must be non-negative.")
        self.cost_bps = cost_bps
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, result: StrategyResult) -> BacktestReport:
        """Backtest a :class:`StrategyResult` produced by the strategy.

        Args:
            result: Signals and diagnostics for one traded pair.

        Returns:
            A :class:`BacktestReport` with equity curve and metrics.

        Raises:
            BacktestError: If inputs are empty or misaligned.
        """
        if result.positions.empty:
            raise BacktestError("Empty position series.")

        log_prices = result.log_prices
        if log_prices.shape[1] != 2:
            raise BacktestError("Expected exactly two price columns.")

        beta = result.pair.beta

        # Daily log returns of each leg (log-price differences).
        asset_returns = log_prices.diff().fillna(0.0)

        # Dollar-neutral leg weights, normalized to gross leverage <= 1.
        norm = 1.0 + abs(beta)
        w_a = result.positions * (1.0 / norm)
        w_b = result.positions * (-np.sign(beta) * abs(beta) / norm)
        weights = pd.concat([w_a, w_b], axis=1)
        weights.columns = asset_returns.columns

        # Gross return: weights held over the bar times leg returns.
        gross = (weights * asset_returns).sum(axis=1)

        # Turnover: sum of absolute weight changes across both legs.
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        costs = turnover * (self.cost_bps * 1e-4)

        net = (gross - costs).rename("net_return")
        equity = (1.0 + net).cumprod().rename("equity")

        metrics = self._compute_metrics(net, result.positions, turnover)
        logger.info("Backtest complete: Sharpe=%.2f, MaxDD=%.2f%%",
                    metrics["sharpe"], 100 * metrics["max_drawdown"])

        return BacktestReport(
            equity_curve=equity,
            net_returns=net,
            gross_returns=gross.rename("gross_return"),
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _compute_metrics(
        self,
        net: pd.Series,
        positions: pd.Series,
        turnover: pd.Series,
    ) -> dict[str, float]:
        """Compute the full performance dictionary from net daily returns."""
        n = len(net)
        if n == 0:
            raise BacktestError("No returns to evaluate.")

        equity = (1.0 + net).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        years = n / TRADING_DAYS_PER_YEAR
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan

        daily_rf = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess = net - daily_rf

        vol = float(net.std(ddof=1))
        annual_vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (
            float(excess.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))
            if vol > 0
            else np.nan
        )

        downside = net[net < 0.0]
        downside_vol = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
        sortino = (
            float(excess.mean() / downside_vol * np.sqrt(TRADING_DAYS_PER_YEAR))
            if downside_vol and downside_vol > 0
            else np.nan
        )

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = float(drawdown.min())
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

        active = net[positions != 0.0]
        hit_rate = float((active > 0).mean()) if len(active) else np.nan

        # A round-trip trade is counted each time we leave a position.
        pos_change = positions.diff().fillna(0.0)
        n_trades = float(((pos_change != 0.0) & (positions == 0.0)).sum())

        return {
            "total_return": total_return,
            "cagr": cagr,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate": hit_rate,
            "avg_turnover": float(turnover.mean()),
            "n_trades": n_trades,
            "time_in_market": float((positions != 0.0).mean()),
        }
