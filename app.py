import streamlit as st
from data.fetcher import fetch_data
from compute.momentum import calculate_relative_momentum
from compute.correlation import calculate_rolling_correlation
from compute.volatility import calculate_historical_volatility
from charts.builder import build_three_panel_chart

st.set_page_config(page_title="Cross-Sectional Momentum & Correlation Tool", layout="wide")

st.title("Cross-Sectional Momentum & Correlation Tool")

# Sidebar Controls
st.sidebar.header("Data Parameters")
st.sidebar.caption("Equities: SPY, QQQ  |  Futures: ES=F, NQ=F, CL=F, GC=F")
base_ticker = st.sidebar.text_input("Base Security", value="SPY").upper().strip()

st.sidebar.subheader("Comparison Securities (Up to 12)")
# Create a few default comparison tickers
default_comps = ["QQQ", "IWM", "GLD"]
comp_tickers = []
for i in range(12):
    default_val = default_comps[i] if i < len(default_comps) else ""
    t = st.sidebar.text_input(f"Comparison {i+1}", value=default_val, key=f"comp_{i}").upper().strip()
    if t:
        comp_tickers.append(t)

# Interval and Timeframe mapping for yfinance
intervals = {"1 Hour": "1h", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
timeframes = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}

interval_label = st.sidebar.selectbox("Interval", options=list(intervals.keys()), index=1)
timeframe_label = st.sidebar.selectbox("Timeframe", options=list(timeframes.keys()), index=3)

interval = intervals[interval_label]
timeframe = timeframes[timeframe_label]

st.sidebar.header("Price Chart Settings")
overlay_enabled = st.sidebar.toggle("Overlay Comparisons on Price Chart", value=False)
overlay_scale = "pct"
if overlay_enabled:
    scale_label = st.sidebar.radio(
        "Overlay Scale",
        options=["% Change", "Independent Price"],
        index=0,
        horizontal=True,
        help="% Change: all series indexed to 0% at chart start  |  Independent Price: each ticker on its own hidden price axis",
    )
    overlay_scale = "pct" if scale_label == "% Change" else "price"

y_scale = st.sidebar.slider(
    "Y-Axis Scale",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.05,
    help="< 1.0 zooms in (tighter range), > 1.0 zooms out (wider range). Applies to all panels.",
)

st.sidebar.header("Momentum Settings")
mo_lookback = st.sidebar.slider("Momentum Lookback (L)", min_value=5, max_value=100, value=20)
mo_ema_span = st.sidebar.slider("EMA Smoothing (S, 0=off)", min_value=0, max_value=50, value=0)
vol_scaled = st.sidebar.toggle("Volatility-Scaled Mode", value=False)
vol_lookback = st.sidebar.slider("Relative Vol Lookback (V)", min_value=5, max_value=100, value=20) if vol_scaled else 20

st.sidebar.header("Correlation Settings")
corr_window = st.sidebar.slider("Correlation Window (W)", min_value=10, max_value=252, value=60)
corr_return_interval = st.sidebar.slider("Return Interval Period (N)", min_value=1, max_value=60, value=1)
corr_return_type = st.sidebar.radio(
    "Return Type",
    options=["Log", "Pct", "Price"],
    index=0,
    horizontal=True,
    help="Log = ln(P[t]/P[t-N])  |  Pct = % change  |  Price = raw price levels (non-stationary)"
)

st.sidebar.header("Volatility Settings")
vol_window = st.sidebar.slider("Vol Lookback (W)", min_value=5, max_value=100, value=20)

# Application Logic
# Use a button to avoid re-fetching data on every keystroke across the 12 ticker inputs
if st.button("Generate Chart", type="primary"):
    if interval == "1h" and timeframe in ["2y", "5y"]:
        st.warning("Note: Hourly data in yfinance is capped at roughly 730 days. Data may be truncated.")

    all_tickers = [base_ticker] + comp_tickers

    # Warn when futures tickers are present: yfinance =F contracts are unadjusted front-month.
    # TradingView uses back-adjusted continuous contracts, so historical prices differ —
    # especially for energy futures (CL, NG) where roll gaps of 1-3% accumulate over months.
    futures_in_use = [t for t in all_tickers if t.endswith("=F")]
    if futures_in_use:
        st.info(
            f"⚠️ Futures detected: **{', '.join(futures_in_use)}**. "
            "yfinance provides unadjusted front-month contracts, while TradingView uses "
            "back-adjusted continuous data. Momentum values may diverge from TradingView, "
            "particularly for energy and commodity futures where roll gaps accumulate."
        )

    with st.spinner("Fetching data and calculating metrics..."):
        from datetime import timedelta
        
        # Calculate calendar days for the requested chart timeframe
        tf_days_map = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 730, "5y": 1825}
        chart_days = tf_days_map.get(timeframe, 365)
        
        # Map the requested timeframe to a padded fetch period to accommodate indicators
        tf_padding = {"1mo": "3mo", "3mo": "6mo", "6mo": "1y", "1y": "2y", "2y": "5y", "5y": "10y"}
        fetch_period = tf_padding.get(timeframe, "5y")
        
        if interval == "1h":
            # For 1h, max is 730d (approx 2y)
            fetch_period = "730d"

        data_dict = fetch_data(all_tickers, interval, fetch_period)

        if data_dict is None or base_ticker not in data_dict:
            st.error(f"Failed to fetch data for the Base Security ({base_ticker}). Please check the ticker or timeframe.")
        else:
            base_prices = data_dict[base_ticker]['Close']
            
            # Determine the starting date of the chart from the most recently fetched datapoint.
            # This is robust against stale datasets or weekends.
            latest_ts = base_prices.index.max()
            chart_start_ts = (latest_ts - timedelta(days=chart_days)).tz_localize(None)

            momentum_dict = {}
            corr_dict = {}
            vol_dict = {}
            
            def subset_series(s):
                try:
                    s_idx = s.index.tz_localize(None)
                except TypeError:
                    s_idx = s.index
                return s[s_idx >= chart_start_ts]

            # Volatility for base ticker
            vol_dict[base_ticker] = subset_series(calculate_historical_volatility(
                base_prices, window=vol_window, interval=interval
            ).dropna())

            for comp_ticker in comp_tickers:
                if comp_ticker in data_dict:
                    comp_prices = data_dict[comp_ticker]['Close']

                    # Momentum
                    mom = calculate_relative_momentum(
                        base_prices,
                        comp_prices,
                        lookback=mo_lookback,
                        ema_span=mo_ema_span,
                        vol_scaled=vol_scaled,
                        vol_lookback=vol_lookback
                    )
                    momentum_dict[comp_ticker] = subset_series(mom.dropna())

                    # Correlation
                    corr = calculate_rolling_correlation(
                        base_prices,
                        comp_prices,
                        window=corr_window,
                        return_interval=corr_return_interval,
                        return_type=corr_return_type.lower(),
                        base_ticker=base_ticker,
                        comp_ticker=comp_ticker,
                    )
                    corr_dict[comp_ticker] = subset_series(corr.dropna())

                    # Volatility
                    vol_dict[comp_ticker] = subset_series(calculate_historical_volatility(
                        comp_prices, window=vol_window, interval=interval
                    ).dropna())
                else:
                    st.warning(f"Could not fetch data for {comp_ticker}. Skipping.")

            # Subset data_dict for the chart candlestick/volume visualization
            chart_data_dict = {}
            for t, df in data_dict.items():
                try:
                    df_idx = df.index.tz_localize(None)
                except TypeError:
                    df_idx = df.index
                chart_data_dict[t] = df[df_idx >= chart_start_ts]

            # Build and show chart
            fig = build_three_panel_chart(base_ticker, chart_data_dict, momentum_dict, corr_dict, vol_dict, interval,
                                          overlay_enabled=overlay_enabled, overlay_scale=overlay_scale,
                                          y_scale=y_scale)
            st.plotly_chart(fig, use_container_width=True, theme=None)
