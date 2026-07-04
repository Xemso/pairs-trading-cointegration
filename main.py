"""End-to-end pairs trading pipeline.

Workflow:
    1. Download and clean the price universe.
    2. Screen for cointegrated pairs on the in-sample window only.
    3. Generate signals and backtest the best pair out-of-sample.
    4. Print metrics and save diagnostic plots to ``results/``.

The in-sample / out-of-sample split is essential: selecting the pair on
the same data used to evaluate it would inflate performance (selection
bias).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

from src.backtester import Backtester, BacktestReport
from src.data_loader import DataLoader, DataLoaderError
from src.strategy import PairsTradingStrategy, StrategyError, StrategyResult

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
UNIVERSE: list[str] = [
    # US energy majors — economically linked, historically cointegration-rich
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Large-cap US financials
    "JPM", "BAC", "WFC", "GS", "MS",
]
START_DATE: str = "2018-01-01"
END_DATE: str = "2024-12-31"
IN_SAMPLE_END: str = "2022-12-31"

ENTRY_Z: float = 2.0
EXIT_Z: float = 0.5
STOP_Z: float = 4.0
ZSCORE_WINDOW: int = 60
COST_BPS: float = 5.0

RESULTS_DIR: Path = Path("results")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def plot_results(result: StrategyResult, report: BacktestReport, outdir: Path) -> None:
    """Save spread/z-score and equity/drawdown diagnostic plots.

    Args:
        result: Strategy signals and spread diagnostics.
        report: Backtest outputs.
        outdir: Directory where PNG files are written.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    pair_name = f"{result.pair.asset_a}-{result.pair.asset_b}"

    # --- Spread & z-score ------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(result.spread.index, result.spread, lw=0.9)
    axes[0].set_title(f"Spread {pair_name} (beta = {result.pair.beta:.3f})")
    axes[0].set_ylabel("Spread (log)")

    axes[1].plot(result.zscore.index, result.zscore, lw=0.9)
    for level, style in [(2.0, "--"), (-2.0, "--"), (0.5, ":"), (-0.5, ":")]:
        axes[1].axhline(level, color="grey", ls=style, lw=0.8)
    axes[1].set_title("Rolling z-score with entry/exit bands")
    axes[1].set_ylabel("z-score")
    fig.tight_layout()
    fig.savefig(outdir / f"{pair_name}_spread_zscore.png", dpi=150)
    plt.close(fig)

    # --- Equity curve & drawdown ----------------------------------------
    equity = report.equity_curve
    drawdown = equity / equity.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(equity.index, equity, lw=1.1)
    axes[0].set_title(f"Equity curve — {pair_name} (net of {COST_BPS:.0f} bps costs)")
    axes[0].set_ylabel("Equity (base 1.0)")

    axes[1].fill_between(drawdown.index, drawdown, 0.0, alpha=0.4)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(outdir / f"{pair_name}_equity_drawdown.png", dpi=150)
    plt.close(fig)

    logger.info("Plots saved to %s/", outdir)


def main() -> None:
    """Run the full research pipeline."""
    # 1. Data ------------------------------------------------------------
    try:
        prices = DataLoader(UNIVERSE, start=START_DATE, end=END_DATE).load()
    except DataLoaderError as exc:
        logger.error("Data loading failed: %s", exc)
        raise SystemExit(1) from exc

    log_prices = DataLoader.to_log_prices(prices)
    in_sample = log_prices.loc[:IN_SAMPLE_END]
    logger.info(
        "In-sample: %d obs | Out-of-sample: %d obs",
        len(in_sample), len(log_prices) - len(in_sample),
    )

    # 2. Pair screening (in-sample only) ----------------------------------
    strategy = PairsTradingStrategy(
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        stop_z=STOP_Z,
        zscore_window=ZSCORE_WINDOW,
    )
    candidates = strategy.screen_universe(in_sample)
    if not candidates:
        logger.error("No cointegrated pair found in-sample. Widen the universe.")
        raise SystemExit(1)

    print("\nTop cointegrated pairs (in-sample):")
    print(f"{'Pair':<12}{'beta':>8}{'ADF p-value':>14}{'Half-life':>12}")
    for c in candidates[:5]:
        print(
            f"{c.asset_a + '/' + c.asset_b:<12}"
            f"{c.beta:>8.3f}{c.p_value:>14.4f}{c.half_life:>10.1f} d"
        )

    best = candidates[0]

    # 3. Trade the best pair over the full sample --------------------------
    #    (hedge ratio is re-estimated inside `run`; for a stricter OOS
    #     protocol, freeze the in-sample beta — see Next Steps in README).
    try:
        result = strategy.run(
            prices[best.asset_a], prices[best.asset_b]
        )
    except StrategyError as exc:
        logger.error("Signal generation failed: %s", exc)
        raise SystemExit(1) from exc

    # 4. Backtest out-of-sample only ---------------------------------------
    oos_mask = result.positions.index > pd.Timestamp(IN_SAMPLE_END)
    oos_result = StrategyResult(
        pair=result.pair,
        spread=result.spread[oos_mask],
        zscore=result.zscore[oos_mask],
        positions=result.positions[oos_mask],
        log_prices=result.log_prices[oos_mask],
    )

    report = Backtester(cost_bps=COST_BPS).run(oos_result)
    print(report.summary())

    plot_results(oos_result, report, RESULTS_DIR)


if __name__ == "__main__":
    main()
