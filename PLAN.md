# Cross-Sectional Momentum & Correlation Tool — Implementation Plan

## Context

Build a TradingView-style interactive charting tool for cross-sectional momentum and correlation analysis. Compare a base security against up to 12 others, viewing relative momentum and rolling correlations on synchronized charts. Light professional theme.

## Tech Stack

**Streamlit + Plotly** — single `make_subplots` figure with `shared_xaxes=True` for perfect chart synchronization. Data from **yfinance** (adjusted prices). Light professional theme.

All 3 panels must live in ONE Plotly figure so zoom/pan stays synchronized.

## File Structure

```
Cross Sectional Tool/
├── app.py                  # Streamlit entry point, sidebar, orchestration
├── data/
│   ├── __init__.py
│   └── fetcher.py          # yfinance fetch with @st.cache_data
├── compute/
│   ├── __init__.py
│   ├── momentum.py         # Relative momentum + vol-scaled
│   └── correlation.py      # Rolling Pearson correlation
├── charts/
│   ├── __init__.py
│   ├── builder.py          # 3-panel Plotly figure construction
│   └── theme.py            # Light professional theme + 12-color palette
└── requirements.txt
```

## Methodology (Verified)

### Cross-Sectional Relative Momentum

Grounded in Jegadeesh & Titman (1993) cross-sectional momentum and Dorsey/Mansfield comparative relative strength:

1. **Price ratio**: `ratio[t] = P_comparison[t] / P_base[t]`
2. **Raw relative momentum**: `momentum[t] = (ratio[t] / ratio[t-L] - 1) * 100`
   - Algebraically equivalent to `((1 + r_comp) / (1 + r_base) - 1) * 100`
   - Positive = comparison is outperforming base over the lookback L
3. **EMA smoothing** (optional, applied AFTER momentum — standard for oscillators, analogous to MACD signal line): `ewm(span=S, adjust=False).mean()`
4. **Volatility-scaled mode**: `scaled[t] = momentum[t] / (rolling_std(returns_comp, V) * 100)`
   - Produces a dimensionless z-score-like quantity: "how many vol units of relative outperformance"
   - Uses the comparison security's own volatility (not relative vol) per user intent
   - Conceptually similar to risk-adjusted relative strength / Information Ratio approach

### Rolling Correlation

Standard Pearson correlation on log returns (preferred for normality properties and time-additivity):

1. `log_ret_base[t] = ln(P_base[t] / P_base[t - N])` where N = return interval period
2. `log_ret_comp[t] = ln(P_comp[t] / P_comp[t - N])`
3. `corr[t] = ret_base.rolling(W).corr(ret_comp)` where W = correlation lookback

The **return interval N** lets the user control the period over which returns are calculated. The unit is determined by the master chart interval — if chart is daily and N=5, returns are 5-day; if weekly and N=2, returns are 2-week. This captures lower-frequency co-movement patterns when N > 1.

Must use returns, NOT prices — correlating price levels produces spurious results due to non-stationarity.

## Edge Case Guards

- **Adjusted prices**: Always use adjusted close from yfinance to handle splits/dividends
- **Division by zero**: Guard against zero prices or zero volatility in vol-scaled mode (return NaN)
- **Data alignment**: Align all series to common dates; use NaN for missing dates rather than forward-fill
- **Lookback > data**: Return NaN gracefully when insufficient data for a calculation
- **EMA NaN handling**: pandas `ewm()` is causal by default (no look-ahead bias) — verify `ignore_na` behavior
- **min_periods**: Set explicitly on rolling calculations for strictness

## Chart Layout

| Row | Height | Content |
|-----|--------|---------|
| 1   | 50%    | Candlestick of base security |
| 2   | 25%    | Momentum lines (1 per comparison ticker, zero line) |
| 3   | 25%    | Correlation lines (1 per comparison ticker, y-axis [-1, 1]) |

### Light Professional Theme
- White/light gray background (`#FFFFFF` paper, `#F8F9FA` plot area)
- Clean grid lines (`#E9ECEF`)
- Professional green/red candles (`#26A69A` / `#EF5350`)
- 12-color palette for comparison securities, consistent across panels 2 & 3
- Weekend gaps hidden via `rangebreaks`
- Streamlit `layout="wide"`

### 12-Color Palette
```
#2962FF, #FF6D00, #00897B, #D50000, #AA00FF,
#FFD600, #00BFA5, #FF3D00, #2E7D32, #C51162,
#6200EA, #0091EA
```

## Sidebar Controls

- **Base security**: text input (default: SPY)
- **Comparison securities**: up to 12 individual text inputs
- **Interval**: selectbox — 1h / Daily / Weekly / Monthly
- **Timeframe**: selectbox — 1mo / 3mo / 6mo / 1y / 2y / 5y
- **Momentum settings**: lookback (5-100, default 20), EMA smoothing (0=off, up to 50), vol-scaled toggle, vol lookback (5-100, default 20)
- **Correlation settings**: correlation lookback window (10-252, default 60), return interval period (1-60, default 1 — number of bars for return calculation)

## Data Layer

- Single `yf.download()` call with `group_by='ticker'`, `auto_adjust=True` for adjusted prices
- `@st.cache_data(ttl=300)` to avoid re-fetching on widget changes
- Hourly data capped at 60 days (yfinance limitation) with user warning
- Normalize multi-level columns into `{ticker: DataFrame}` dict

## Implementation Order

1. `requirements.txt` + install plotly
2. `charts/theme.py` — light professional theme constants, color palette
3. `data/fetcher.py` — cached yfinance fetcher with data normalization
4. `compute/momentum.py` — relative momentum + vol-scaled with edge case guards
5. `compute/correlation.py` — rolling correlation on log returns
6. `charts/builder.py` — 3-panel Plotly figure builder
7. `app.py` — sidebar controls + orchestration + rendering
8. End-to-end testing and polish

## Verification

1. Run `streamlit run app.py`
2. Test with SPY as base, QQQ/IWM/GLD as comparisons
3. Verify all 3 panels render and zoom/pan stays synchronized
4. Toggle between intervals and timeframes
5. Toggle volatility-scaled momentum on/off
6. Test with invalid tickers — should show error gracefully
7. Test hourly interval with >60 day timeframe — should show warning
