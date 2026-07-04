"""Cointegration engine and pairs trading signal generation.

Implements the Engle-Granger two-step methodology:

1. Estimate the hedge ratio ``beta`` by OLS on log prices.
2. Test the OLS residuals (the spread) for stationarity with the
   Augmented Dickey-Fuller test.

Stationary spreads are modeled as mean-reverting (Ornstein-Uhlenbeck)
processes; the AR(1)-implied half-life serves as a quality filter.
Trading signals are generated from the rolling z-score of the spread and
shifted by one bar to preclude look-ahead bias.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class StrategyError(Exception):
    """Raised when signal generation fails on invalid inputs."""


# ---------------------------------------------------------------------- #
# Result containers
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class CointegrationResult:
    """Outcome of an Engle-Granger test on one ordered pair.

    Attributes:
        asset_a: Dependent-variable ticker (long leg of the spread).
        asset_b: Regressor ticker (hedge leg).
        alpha: OLS intercept.
        beta: OLS hedge ratio.
        adf_stat: ADF test statistic on the residuals.
        p_value: ADF p-value (H0: unit root, i.e. no cointegration).
        half_life: AR(1)-implied mean-reversion half-life, in bars.
    """

    asset_a: str
    asset_b: str
    alpha: float
    beta: float
    adf_stat: float
    p_value: float
    half_life: float


@dataclass
class StrategyResult:
    """Signals and spread diagnostics for a single traded pair.

    Attributes:
        pair: The cointegration statistics of the traded pair.
        spread: Log-price spread ``log(A) - beta * log(B) - alpha``.
        zscore: Rolling z-score of the spread.
        positions: Executed position in the spread in {-1, 0, +1},
            already shifted by one bar (tradeable, no look-ahead).
        log_prices: Two-column DataFrame of the pair's log prices.
    """

    pair: CointegrationResult
    spread: pd.Series
    zscore: pd.Series
    positions: pd.Series
    log_prices: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------- #
# Strategy
# ---------------------------------------------------------------------- #
class PairsTradingStrategy:
    """Screen for cointegrated pairs and generate z-score trading signals.

    Attributes:
        entry_z: Absolute z-score above which a position is opened.
        exit_z: Absolute z-score below which a position is closed.
        stop_z: Absolute z-score beyond which the position is stopped out
            (protection against structural breaks of the relationship).
        zscore_window: Rolling window (bars) for the z-score.
        adf_pvalue_max: Maximum ADF p-value to accept a pair.
        max_half_life: Maximum acceptable half-life (bars).
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        zscore_window: int = 60,
        adf_pvalue_max: float = 0.05,
        max_half_life: float = 60.0,
    ) -> None:
        if not 0 < exit_z < entry_z < stop_z:
            raise ValueError("Thresholds must satisfy 0 < exit_z < entry_z < stop_z.")
        if zscore_window < 20:
            raise ValueError("`zscore_window` should be at least 20 bars.")

        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.zscore_window = zscore_window
        self.adf_pvalue_max = adf_pvalue_max
        self.max_half_life = max_half_life

    # ------------------------------------------------------------------ #
    # Step 1 & 2: Engle-Granger
    # ------------------------------------------------------------------ #
    @staticmethod
    def hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float]:
        """Estimate the OLS regression ``log_a = alpha + beta * log_b``.

        Args:
            log_a: Log prices of the dependent asset.
            log_b: Log prices of the regressor asset.

        Returns:
            Tuple ``(alpha, beta)``.
        """
        x = np.column_stack([np.ones(len(log_b)), log_b.to_numpy()])
        y = log_a.to_numpy()
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        alpha, beta = float(coeffs[0]), float(coeffs[1])
        return alpha, beta

    @staticmethod
    def half_life(spread: pd.Series) -> float:
        """AR(1)-implied mean-reversion half-life of a spread.

        Fits ``dS_t = a + gamma * S_{t-1} + u_t``; the half-life is
        ``-ln(2) / gamma`` for ``gamma < 0``.

        Args:
            spread: Stationary spread series.

        Returns:
            Half-life in bars (``inf`` if no mean reversion is detected).
        """
        lagged = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        x = np.column_stack([np.ones(len(lagged)), lagged.to_numpy()])
        coeffs, *_ = np.linalg.lstsq(x, delta.to_numpy(), rcond=None)
        gamma = float(coeffs[1])
        if gamma >= 0:
            return float("inf")
        return float(-np.log(2.0) / gamma)

    def test_pair(
        self, log_a: pd.Series, log_b: pd.Series
    ) -> CointegrationResult:
        """Run the Engle-Granger two-step test on one ordered pair.

        Args:
            log_a: Log prices of asset A (dependent variable).
            log_b: Log prices of asset B (regressor).

        Returns:
            A :class:`CointegrationResult` with test statistics.

        Raises:
            StrategyError: If the series are misaligned or too short.
        """
        if len(log_a) != len(log_b):
            raise StrategyError("Series must share the same index length.")
        if len(log_a) < 5 * self.zscore_window:
            raise StrategyError("Insufficient history for a reliable test.")

        alpha, beta = self.hedge_ratio(log_a, log_b)
        spread = log_a - beta * log_b - alpha

        try:
            adf_stat, p_value, *_ = adfuller(spread.to_numpy(), autolag="AIC")
        except Exception as exc:
            raise StrategyError(f"ADF test failed: {exc}") from exc

        return CointegrationResult(
            asset_a=str(log_a.name),
            asset_b=str(log_b.name),
            alpha=alpha,
            beta=beta,
            adf_stat=float(adf_stat),
            p_value=float(p_value),
            half_life=self.half_life(spread),
        )

    def screen_universe(self, log_prices: pd.DataFrame) -> list[CointegrationResult]:
        """Test every unordered pair in the universe and rank candidates.

        Args:
            log_prices: Wide panel of log prices (one column per ticker).

        Returns:
            Accepted pairs (p-value and half-life filters applied),
            sorted by ascending ADF p-value.
        """
        accepted: list[CointegrationResult] = []
        for a, b in itertools.combinations(log_prices.columns, 2):
            try:
                res = self.test_pair(log_prices[a], log_prices[b])
            except StrategyError as exc:
                logger.warning("Skipping pair (%s, %s): %s", a, b, exc)
                continue

            if res.p_value <= self.adf_pvalue_max and res.half_life <= self.max_half_life:
                accepted.append(res)
                logger.info(
                    "Accepted %s/%s: beta=%.3f, p=%.4f, half-life=%.1f bars",
                    a, b, res.beta, res.p_value, res.half_life,
                )

        return sorted(accepted, key=lambda r: r.p_value)

    # ------------------------------------------------------------------ #
    # Step 3: signal generation
    # ------------------------------------------------------------------ #
    def run(self, prices_a: pd.Series, prices_b: pd.Series) -> StrategyResult:
        """Full pipeline on one pair: test, spread, z-score, positions.

        Args:
            prices_a: Price levels of asset A.
            prices_b: Price levels of asset B.

        Returns:
            A :class:`StrategyResult` with tradeable positions.
        """
        log_a, log_b = np.log(prices_a), np.log(prices_b)
        pair = self.test_pair(log_a, log_b)

        spread = log_a - pair.beta * log_b - pair.alpha
        zscore = self._rolling_zscore(spread)
        positions = self._positions_from_zscore(zscore)

        return StrategyResult(
            pair=pair,
            spread=spread,
            zscore=zscore,
            positions=positions,
            log_prices=pd.concat([log_a, log_b], axis=1),
        )

    def _rolling_zscore(self, spread: pd.Series) -> pd.Series:
        """Rolling z-score of the spread over ``zscore_window`` bars."""
        mean = spread.rolling(self.zscore_window).mean()
        std = spread.rolling(self.zscore_window).std(ddof=1)
        return ((spread - mean) / std).rename("zscore")

    def _positions_from_zscore(self, zscore: pd.Series) -> pd.Series:
        """Map z-scores to spread positions with hysteresis and stop-loss.

        The state machine is inherently sequential (the current position
        depends on the previous one), so it is implemented as a single
        O(n) pass over NumPy arrays rather than a fake vectorization that
        would leak state.

        Args:
            zscore: Rolling z-score of the spread.

        Returns:
            Executed positions in {-1, 0, +1}, shifted one bar forward so
            a signal observed at close ``t`` is only traded at ``t+1``.
        """
        z = zscore.to_numpy()
        pos = np.zeros_like(z)
        current = 0.0

        for i in range(len(z)):
            zi = z[i]
            if np.isnan(zi):
                pos[i] = current
                continue

            if current == 0.0:
                if zi <= -self.entry_z:
                    current = 1.0   # long spread: long A, short beta*B
                elif zi >= self.entry_z:
                    current = -1.0  # short spread
            else:
                reverted = abs(zi) <= self.exit_z
                stopped = abs(zi) >= self.stop_z
                if reverted or stopped:
                    current = 0.0

            pos[i] = current

        # Trade next bar: eliminates look-ahead bias.
        return (
            pd.Series(pos, index=zscore.index, name="position")
            .shift(1)
            .fillna(0.0)
        )
