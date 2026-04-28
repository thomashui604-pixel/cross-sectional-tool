import pandas as pd
import streamlit as st
from data.fetcher import fetch_data
from compute.momentum import calculate_relative_momentum
from compute.correlation import calculate_rolling_correlation
from compute.volatility import calculate_historical_volatility
from compute.signals import compute_signals
from charts.builder import build_three_panel_chart, build_regime_scatter

st.set_page_config(page_title="Cross-Sectional Momentum & Correlation Tool", layout="wide")

st.title("Cross-Sectional Momentum & Correlation Tool")

# Sidebar Controls
st.sidebar.header("Data Parameters")
st.sidebar.caption("Equities: SPY, QQQ  |  Futures: ES=F, NQ=F, CL=F, GC=F")
base_ticker = st.sidebar.text_input("Base Security", value="SPY").upper().strip()

st.sidebar.subheader("Comparison Securities (Up to 12)")
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

st.sidebar.header("Signal Settings")
with st.sidebar.expander("Configure Signals", expanded=False):
    signal_persistence = st.slider(
        "Confirm Bars",
        min_value=2, max_value=5, value=3,
        help="Bars a crossover or extreme must hold before it is flagged.",
    )
    signal_extremes_pct = st.slider(
        "Extreme Threshold (%ile)",
        min_value=80, max_value=95, value=90,
        help="Momentum above this percentile = extreme high; below (100 − this) = extreme low.",
    )
    signal_corr_high = st.slider(
        "Corr. High Zone",
        min_value=0.40, max_value=0.90, value=0.60, step=0.05,
        help="Correlation above this threshold = high-correlation regime.",
    )
    signal_corr_low = st.slider(
        "Corr. Low Zone",
        min_value=0.10, max_value=0.50, value=0.40, step=0.05,
        help="Correlation below this threshold = idiosyncratic / low-correlation regime.",
    )
    signal_vol_ratio_high = st.slider("Vol Ratio High", min_value=1.2, max_value=2.5, value=1.5, step=0.1)
    signal_vol_ratio_low  = st.slider("Vol Ratio Low",  min_value=0.40, max_value=0.80, value=0.67, step=0.01)

# ── Setup type emoji map (display only) ───────────────────────────────────────
_SETUP_EMOJI = {
    "Trend Leader":               "🟢 Trend Leader",
    "Idiosyncratic Outperformer": "🔵 Idiosync. Outperformer",
    "Momentum Long":              "🟡 Momentum Long",
    "Convergence":                "🟠 Convergence",
    "Idiosyncratic Lag":          "🔴 Idiosync. Lag",
    "Momentum Short":             "🔴 Momentum Short",
    "Neutral":                    "⚪ Neutral",
    "—":                          "—",
}


def _style_scorecard(df: pd.DataFrame) -> pd.Styler:
    """Apply colour gradients to the numeric scorecard columns."""
    styler = df.style.format(
        {"Momentum": "{:.3f}", "Mom %ile": "{:.0f}", "Correlation": "{:.3f}", "Vol Ratio": "{:.2f}"},
        na_rep="—",
    )
    mom_vals = df["Momentum"].dropna() if "Momentum" in df.columns else pd.Series(dtype=float)
    mom_abs  = float(mom_vals.abs().max()) if not mom_vals.empty else 1.0
    mom_abs  = max(mom_abs, 0.01)

    if "Momentum" in df.columns:
        styler = styler.background_gradient(subset=["Momentum"], cmap="RdYlGn", vmin=-mom_abs, vmax=mom_abs)
    if "Mom %ile" in df.columns:
        styler = styler.background_gradient(subset=["Mom %ile"], cmap="RdYlGn", vmin=0, vmax=100)
    if "Correlation" in df.columns:
        styler = styler.background_gradient(subset=["Correlation"], cmap="coolwarm", vmin=-1, vmax=1)
    if "Vol Ratio" in df.columns:
        styler = styler.background_gradient(subset=["Vol Ratio"], cmap="YlOrRd", vmin=0.5, vmax=2.0)
    return styler


# ── Application Logic ──────────────────────────────────────────────────────────
if st.button("Generate Chart", type="primary"):
    if interval == "1h" and timeframe in ["2y", "5y"]:
        st.warning("Note: Hourly data in yfinance is capped at roughly 730 days. Data may be truncated.")

    all_tickers = [base_ticker] + comp_tickers

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

        tf_days_map = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 730, "5y": 1825}
        chart_days  = tf_days_map.get(timeframe, 365)

        tf_padding  = {"1mo": "3mo", "3mo": "6mo", "6mo": "1y", "1y": "2y", "2y": "5y", "5y": "10y"}
        fetch_period = tf_padding.get(timeframe, "5y")

        if interval == "1h":
            fetch_period = "730d"

        data_dict = fetch_data(all_tickers, interval, fetch_period)

        if data_dict is None or base_ticker not in data_dict:
            st.error(f"Failed to fetch data for the Base Security ({base_ticker}). Please check the ticker or timeframe.")
        else:
            base_prices = data_dict[base_ticker]['Close']

            latest_ts      = base_prices.index.max()
            chart_start_ts = (latest_ts - timedelta(days=chart_days)).tz_localize(None)

            momentum_dict      = {}   # subsetted — for chart
            momentum_full_dict = {}   # full series — for signal percentiles / event detection
            corr_dict          = {}
            corr_full_dict     = {}
            vol_dict           = {}

            def subset_series(s):
                try:
                    s_idx = s.index.tz_localize(None)
                except TypeError:
                    s_idx = s.index
                return s[s_idx >= chart_start_ts]

            vol_dict[base_ticker] = subset_series(
                calculate_historical_volatility(base_prices, window=vol_window, interval=interval).dropna()
            )

            for comp_ticker in comp_tickers:
                if comp_ticker in data_dict:
                    comp_prices = data_dict[comp_ticker]['Close']

                    mom = calculate_relative_momentum(
                        base_prices, comp_prices,
                        lookback=mo_lookback, ema_span=mo_ema_span,
                        vol_scaled=vol_scaled, vol_lookback=vol_lookback,
                    )
                    momentum_full_dict[comp_ticker] = mom.dropna()
                    momentum_dict[comp_ticker]      = subset_series(mom.dropna())

                    corr = calculate_rolling_correlation(
                        base_prices, comp_prices,
                        window=corr_window, return_interval=corr_return_interval,
                        return_type=corr_return_type.lower(),
                        base_ticker=base_ticker, comp_ticker=comp_ticker,
                    )
                    corr_full_dict[comp_ticker] = corr.dropna()
                    corr_dict[comp_ticker]      = subset_series(corr.dropna())

                    vol_dict[comp_ticker] = subset_series(
                        calculate_historical_volatility(comp_prices, window=vol_window, interval=interval).dropna()
                    )
                else:
                    st.warning(f"Could not fetch data for {comp_ticker}. Skipping.")

            chart_data_dict = {}
            for t, df in data_dict.items():
                try:
                    df_idx = df.index.tz_localize(None)
                except TypeError:
                    df_idx = df.index
                chart_data_dict[t] = df[df_idx >= chart_start_ts]

            # ── Signals & Scorecard ────────────────────────────────────────────
            trend_window = max(3, mo_lookback // 4)
            signals = compute_signals(
                base_ticker=base_ticker,
                comp_tickers=[t for t in comp_tickers if t in momentum_full_dict],
                momentum_full_dict=momentum_full_dict,
                corr_full_dict=corr_full_dict,
                vol_dict=vol_dict,
                mo_lookback=mo_lookback,
                persistence_bars=signal_persistence,
                corr_high_thresh=signal_corr_high,
                corr_low_thresh=signal_corr_low,
                vol_ratio_high=signal_vol_ratio_high,
                vol_ratio_low=signal_vol_ratio_low,
                extremes_pct_high=float(signal_extremes_pct),
                extremes_pct_low=float(100 - signal_extremes_pct),
            )

            with st.expander("📊 Signals & Scorecard", expanded=True):
                # ── Breadth ───────────────────────────────────────────────────
                breadth = signals["breadth"]
                n       = breadth["total"]
                count   = breadth["count"]
                trend   = breadth["trend"]

                if n > 0:
                    ratio = count / n
                    if ratio >= 2 / 3:
                        breadth_label = "Broad Leadership"
                        breadth_color = "🟢"
                    elif ratio <= 1 / 3:
                        breadth_label = "Broadly Lagging"
                        breadth_color = "🔴"
                    else:
                        breadth_label = "Mixed / Transitional"
                        breadth_color = "🟡"

                    trend_arrow = "▲" if trend > 0 else ("▼" if trend < 0 else "→")
                    delta_str   = f"{trend_arrow} {abs(trend)} vs {trend_window}B ago"

                    bc1, bc2 = st.columns([1, 3])
                    with bc1:
                        st.metric(
                            label=f"**{base_ticker}** beats",
                            value=f"{count} / {n}",
                            delta=delta_str,
                        )
                    with bc2:
                        st.markdown(
                            f"**{breadth_color} {breadth_label}**  \n"
                            f"*>⅔ = broad leadership · <⅓ = broadly lagging · between = mixed*"
                        )

                st.divider()

                # ── Scorecard table + Regime scatter ─────────────────────────
                sc_col, reg_col = st.columns([1.4, 1])

                with sc_col:
                    st.subheader("Scorecard")
                    scorecard = signals["scorecard"]
                    if scorecard:
                        sc_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in scorecard]
                        sc_df   = pd.DataFrame(sc_rows)
                        if "Setup" in sc_df.columns:
                            sc_df["Setup"] = sc_df["Setup"].map(lambda x: _SETUP_EMOJI.get(x, x))
                        st.dataframe(_style_scorecard(sc_df), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No comparable data available.")

                with reg_col:
                    st.subheader("Regime Map")
                    regime_fig = build_regime_scatter(
                        signals["scorecard"],
                        corr_high=signal_corr_high,
                        corr_low=signal_corr_low,
                    )
                    st.plotly_chart(regime_fig, use_container_width=True, theme=None)

                st.divider()

                # ── Events log ────────────────────────────────────────────────
                st.subheader("Extremes & Crossovers")
                events = signals["events"]
                if events:
                    ev_df = pd.DataFrame(events)
                    st.dataframe(ev_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No notable events detected in the current lookback window.")

                st.caption(
                    f"*Signals use the last completed bar (iloc[−2]) to avoid partial-bar distortion. "
                    f"Crossovers require {signal_persistence} confirming bars. "
                    f"Percentile window: 252 bars. Extremes threshold: {signal_extremes_pct}th / "
                    f"{100 - signal_extremes_pct}th pct.*"
                )

            # ── Chart ─────────────────────────────────────────────────────────
            fig = build_three_panel_chart(
                base_ticker, chart_data_dict, momentum_dict, corr_dict, vol_dict, interval,
                overlay_enabled=overlay_enabled, overlay_scale=overlay_scale, y_scale=y_scale,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
