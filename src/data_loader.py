"""Market data acquisition and cleaning.

This module wraps ``yfinance`` behind a small, testable interface. All
downstream modules consume a clean ``pd.DataFrame`` of adjusted close
prices indexed by trading date, which makes the strategy and backtester
fully agnostic to the data source.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Raised when market data cannot be retrieved or is unusable."""


class DataLoader:
    """Download and clean adjusted close prices for a list of tickers.

    Attributes:
        tickers: Ticker symbols to download (Yahoo Finance convention).
        start: Start date, ``YYYY-MM-DD``.
        end: End date, ``YYYY-MM-DD`` (``None`` means today).
        max_missing_ratio: Maximum tolerated fraction of missing values
            per ticker before the ticker is dropped from the universe.
    """

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        max_missing_ratio: float = 0.05,
    ) -> None:
        if not tickers:
            raise ValueError("`tickers` must contain at least one symbol.")
        if not 0.0 <= max_missing_ratio < 1.0:
            raise ValueError("`max_missing_ratio` must lie in [0, 1).")

        self.tickers: list[str] = sorted(set(tickers))
        self.start: str = start
        self.end: str | None = end
        self.max_missing_ratio: float = max_missing_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Download, clean and return adjusted close prices.

        Returns:
            DataFrame of shape ``(n_dates, n_tickers)`` with a
            ``DatetimeIndex``, forward-filled for isolated gaps, with
            leading/trailing all-NaN rows removed.

        Raises:
            DataLoaderError: If the download fails or yields no usable data.
        """
        raw = self._download()
        clean = self._clean(raw)

        if clean.empty or clean.shape[1] < 1:
            raise DataLoaderError(
                "No usable price series after cleaning. Check tickers and dates."
            )

        logger.info(
            "Loaded %d tickers, %d observations (%s -> %s).",
            clean.shape[1],
            clean.shape[0],
            clean.index[0].date(),
            clean.index[-1].date(),
        )
        return clean

    @staticmethod
    def to_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """Convert price levels to natural log prices.

        Args:
            prices: Strictly positive price levels.

        Returns:
            Element-wise natural logarithm of ``prices``.

        Raises:
            ValueError: If any price is non-positive.
        """
        if (prices <= 0).any().any():
            raise ValueError("Prices must be strictly positive to take logs.")
        return np.log(prices)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _download(self) -> pd.DataFrame:
        """Fetch adjusted close prices from Yahoo Finance.

        Returns:
            Wide DataFrame of adjusted close prices, one column per ticker.

        Raises:
            DataLoaderError: On network failure or empty response.
        """
        try:
            import yfinance as yf  # local import: keeps module importable offline

            data = yf.download(
                tickers=self.tickers,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:  # network / API errors
            raise DataLoaderError(f"yfinance download failed: {exc}") from exc

        if data is None or data.empty:
            raise DataLoaderError("yfinance returned an empty dataset.")

        # yfinance returns a MultiIndex for multiple tickers, flat otherwise.
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].copy()
        else:
            close = data[["Close"]].copy()
            close.columns = [self.tickers[0]]

        close.index = pd.to_datetime(close.index)
        return close.sort_index()

    def _clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Drop sparse tickers, forward-fill small gaps, align the panel.

        Args:
            prices: Raw wide price panel.

        Returns:
            Cleaned price panel restricted to dates where every retained
            ticker has a valid observation.
        """
        # 1. Drop tickers with too many missing observations.
        missing_ratio = prices.isna().mean()
        keep = missing_ratio[missing_ratio <= self.max_missing_ratio].index
        dropped = sorted(set(prices.columns) - set(keep))
        if dropped:
            logger.warning("Dropping sparse tickers: %s", dropped)
        prices = prices[keep]

        # 2. Forward-fill isolated gaps (holidays, missing prints), never
        #    backward-fill (that would leak future information).
        prices = prices.ffill()

        # 3. Keep only rows where the full panel is observed.
        prices = prices.dropna(how="any")

        # 4. Guard against zero/negative prints from bad data.
        prices = prices[(prices > 0).all(axis=1)]

        return prices
