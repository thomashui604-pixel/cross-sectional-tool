import pandas as pd
import numpy as np

def calculate_historical_volatility(prices, window=20, interval='1d'):
    """
    Calculates annualised rolling historical volatility as a percentage.

    Args:
        prices   (pd.Series): Adjusted close prices.
        window   (int):       Rolling lookback W in bars (default 20).
        interval (str):       yfinance interval string — '1d', '1wk', '1mo', '1h'.

    Returns:
        pd.Series: Annualised HV in % (e.g. 18.5 means 18.5% p.a.).
    """
    ANNUALISATION = {
        '1d':  252,
        '1wk': 52,
        '1mo': 12,
        '1h':  1638,  # 252 trading days × 6.5 hours
    }
    ann = ANNUALISATION.get(interval, 252)

    prices = prices.replace(0, np.nan).dropna()
    log_ret = np.log(prices / prices.shift(1))
    hv = log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(ann) * 100
    return hv
