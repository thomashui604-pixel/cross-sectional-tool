import pandas as pd
import numpy as np

# Volatility indices must be differenced, not log-returned.
# % returns on VIX are economically meaningless (a move from 12→13 ≠ same as 40→43.3).
_VOL_INDEX_TICKERS = frozenset({
    "^VIX", "VIX", "^VVIX", "VVIX", "MOVE", "^VXN", "^RVX",
    "^VXEEM", "^OVX", "^GVZ", "^VXAPL", "^VXAZN", "^VXGS",
    "^EVZ", "^JNIV",
})

# Bond yield tickers: already in rate units, so difference = Δyield (basis points / 100).
_YIELD_TICKERS = frozenset({
    "^TNX", "^TYX", "^FVX", "^IRX",
})


def _classify_ticker(ticker: str) -> str:
    """Classify a ticker as 'vol_index', 'yield', or 'equity'."""
    t = ticker.upper()
    if t in _VOL_INDEX_TICKERS:
        return "vol_index"
    if t in _YIELD_TICKERS:
        return "yield"
    return "equity"


def _stationary_series(
    prices: pd.Series,
    asset_type: str,
    return_type: str,
    return_interval: int,
) -> pd.Series:
    """
    Transform a price/level series into a stationary return/change series.

    Vol indices and bond yields use first differences (point changes) regardless
    of the user-selected return_type — % returns on these series are spurious.
    All other assets follow the user-selected return_type.
    """
    if asset_type in ("vol_index", "yield"):
        return prices.diff(return_interval)
    if return_type == "pct":
        return prices.pct_change(return_interval) * 100
    if return_type == "price":
        # Raw levels — non-stationary. User's explicit choice; kept for reference.
        return prices
    # Default: log returns
    return np.log(prices / prices.shift(return_interval))


def calculate_rolling_correlation(
    base_prices: pd.Series,
    comp_prices: pd.Series,
    window: int = 60,
    return_interval: int = 1,
    return_type: str = "log",
    base_ticker: str = "",
    comp_ticker: str = "",
) -> pd.Series:
    """
    Calculates rolling Pearson correlation between two securities.

    Applies the statistically correct transformation per asset class before
    computing correlation, preventing spurious results from non-stationary inputs:
      - Equities / ETFs / Futures / FX / Commodities → log returns (default) or % returns
      - Volatility indices (VIX, VVIX, MOVE, …)     → point changes (first differences)
      - Bond yields (^TNX, ^TYX, …)                  → yield changes (first differences)

    Args:
        base_prices:     Adjusted close prices for the base security.
        comp_prices:     Adjusted close prices for the comparison security.
        window:          Rolling window in bars (default 60).
        return_interval: Number of periods for return calculation (default 1).
        return_type:     'log' | 'pct' | 'price' — applies to equity-class assets only.
        base_ticker:     Ticker symbol of the base security (used for auto-classification).
        comp_ticker:     Ticker symbol of the comparison security (used for auto-classification).

    Returns:
        pd.Series of rolling Pearson correlation values in [-1, 1].
    """
    base_type = _classify_ticker(base_ticker) if base_ticker else "equity"
    comp_type = _classify_ticker(comp_ticker) if comp_ticker else "equity"

    # Align on shared timestamps and drop any zero/NaN prices before transforming.
    df = pd.DataFrame({"base": base_prices, "comp": comp_prices}).dropna(how="any")
    df = df.replace(0, np.nan).dropna(how="any")

    df["ret_base"] = _stationary_series(df["base"], base_type, return_type, return_interval)
    df["ret_comp"] = _stationary_series(df["comp"], comp_type, return_type, return_interval)

    # Drop NaNs introduced by differencing/shifting before rolling.
    df = df[["ret_base", "ret_comp"]].dropna()

    corr = df["ret_base"].rolling(window=window, min_periods=window).corr(df["ret_comp"])
    return corr
