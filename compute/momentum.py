import pandas as pd
import numpy as np

def calculate_relative_momentum(base_prices, comp_prices, lookback=20, ema_span=0, 
                                vol_scaled=False, vol_lookback=20):
    """
    Calculates Cross-Sectional Relative Momentum.
    
    Args:
        base_prices (pd.Series): Adjusted prices of the base security.
        comp_prices (pd.Series): Adjusted prices of the comparison security.
        lookback (int): Lookback window L for momentum calculation.
        ema_span (int): EMA span S. If > 0, applies EMA smoothing.
        vol_scaled (bool): Toggle for volatility-scaled mode.
        vol_lookback (int): Volatility lookback window V.
        
    Returns:
        pd.Series: Relative momentum series.
    """
    # Inner join on timestamps — only keep bars where BOTH series have data.
    # This is critical when mixing asset classes (e.g. equity base vs futures comparison):
    # futures trade ~24h while equities only trade market hours, so an outer join would
    # leave NaNs for equity bars during overnight/weekend sessions and silently corrupt
    # the ratio and momentum calculations.
    df = pd.DataFrame({'base': base_prices, 'comp': comp_prices}).dropna(how='any')
    # Guard against zero prices (e.g. data errors)
    df = df.replace(0, np.nan).dropna(how='any')
    
    # Daily relative log return = ln(base[t]/base[t-1]) - ln(comp[t]/comp[t-1])
    # Using cumulative log returns rather than a price-level ratio (base[t]/comp[t]) /
    # (base[t-L]/comp[t-L]) keeps the formula additive across time — each bar contributes
    # independently to the rolling sum. No threshold filtering applied; roll gaps and
    # geopolitical moves are all captured as-is.
    df['relative_daily'] = np.log(df['base'] / df['base'].shift(1)) - np.log(df['comp'] / df['comp'].shift(1))

    # 1. Cumulative relative log return over L bars, expressed as a percentage.
    # Rising = base outperforming comparison.
    df['momentum'] = df['relative_daily'].rolling(window=lookback, min_periods=lookback).sum() * 100
    
    # 3. Volatility scaling
    if vol_scaled:
        # Use the vol of the ratio's own daily returns (= tracking error between base and comp).
        # This is the correct denominator for a cross-asset-class tool:
        #
        #   - Using comp's own vol breaks when base and comp have very different volatility
        #     regimes (e.g. equity base vs bond future comparison). Bond futures have ~0.02%
        #     daily vol vs equities at ~1%, so dividing by comp vol inflates the signal by
        #     ~50x and produces extreme readings like ±98.
        #
        #   - Ratio returns ≈ excess return of base over comp (tracking error). Their vol
        #     captures how variable the *relative* performance actually is — which is exactly
        #     what we want to normalize against. Works correctly for any asset class mix.
        # relative_daily already holds the filtered daily log returns of the ratio —
        # use it directly as the vol estimate so the same roll-gap filtering applies.
        per_bar_vol_pct = df['relative_daily'].rolling(window=vol_lookback, min_periods=vol_lookback).std() * 100

        # Divide by the rolling per-bar vol of the ratio — NO sqrt(L) scaling.
        #
        # We previously used per_bar_vol * sqrt(L) (Sharpe ratio form) which correctly
        # normalises for the lookback horizon but produces bounded values of ~±1-3.
        # TradingView divides by the raw per-bar vol, giving values that naturally scale
        # with lookback strength ("how many daily-vol units is this L-period cumulative
        # return?"). This matches the observed TradingView output range of ~±3-8 typical,
        # ±10 extreme.
        per_bar_vol_pct = per_bar_vol_pct.replace(0, np.nan)
        df['momentum'] = df['momentum'] / per_bar_vol_pct

    # 4. EMA smoothing (applied AFTER momentum)
    if ema_span > 0:
        df['momentum'] = df['momentum'].ewm(span=ema_span, adjust=False, ignore_na=True).mean()
        
    return df['momentum']
