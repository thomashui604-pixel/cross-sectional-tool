import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .theme import apply_light_professional_theme, PALETTE, CANDLE_INCREASING, CANDLE_DECREASING, BASE_VOL_COLOR


def _scaled_yrange(series_list, scale):
    """
    Return [lo, hi] for a y-axis, computed from the union of all series values
    and scaled around the midpoint by `scale` (< 1 zooms in, > 1 zooms out).
    Returns None if no data is present.
    """
    vals = []
    for s in series_list:
        if s is not None and not s.empty:
            clean = s.dropna()
            if not clean.empty:
                vals.extend(clean.tolist())
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 1e-10:
        span = max(abs(hi), 1e-6) * 0.2
    center = (hi + lo) / 2
    half = span / 2 * scale
    # Add a small natural padding (3%) so data doesn't sit flush against the axis edges.
    pad = span * 0.03
    return [center - half - pad, center + half + pad]

def build_three_panel_chart(
    base_ticker,
    data_dict,
    momentum_dict,
    corr_dict,
    vol_dict,
    interval,
    title="Cross-Sectional Analysis",
    overlay_enabled=False,
    overlay_scale="pct",
    y_scale=1.0,
):
    """
    Builds the 4-panel synchronized Plotly chart.

    Args:
        base_ticker (str): The symbol of the base security.
        data_dict (dict): Dictionary of {ticker: DataFrame}.
        momentum_dict (dict): Dictionary of {ticker: pd.Series} for momentum.
        corr_dict (dict): Dictionary of {ticker: pd.Series} for correlation.
        vol_dict (dict): Dictionary of {ticker: pd.Series} for historical volatility.
        interval (str): The data interval (e.g., '1d', '1wk').
        title (str): Chart title.
        overlay_enabled (bool): Whether to overlay comparison tickers on the price chart.
        overlay_scale (str): 'pct' for % change from chart start, 'price' for independent price axes.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            f"{base_ticker} Price",
            f"Relative Momentum (vs {base_ticker})",
            f"Rolling Correlation (vs {base_ticker})",
            "Historical Volatility (Annualised %)"
        )
    )

    comparison_tickers = [t for t in data_dict.keys() if t != base_ticker]

    # ── Panel 1: Base security price chart ────────────────────────────────────
    if base_ticker in data_dict and not data_dict[base_ticker].empty:
        base_df = data_dict[base_ticker]
        open_col  = base_df['Open']  if 'Open'  in base_df.columns else base_df['Close']
        high_col  = base_df['High']  if 'High'  in base_df.columns else base_df['Close']
        low_col   = base_df['Low']   if 'Low'   in base_df.columns else base_df['Close']
        close_col = base_df['Close']

        if overlay_enabled and overlay_scale == "pct":
            # Normalise base OHLC to % change from the first close in the chart window.
            first_close = close_col.dropna().iloc[0]
            fig.add_trace(
                go.Candlestick(
                    x=base_df.index,
                    open=(open_col  / first_close - 1) * 100,
                    high=(high_col  / first_close - 1) * 100,
                    low=(low_col    / first_close - 1) * 100,
                    close=(close_col / first_close - 1) * 100,
                    name=base_ticker,
                    increasing_line_color=CANDLE_INCREASING,
                    decreasing_line_color=CANDLE_DECREASING,
                    showlegend=False,
                ),
                row=1, col=1,
            )
            # Zero baseline so the chart reads as "flat = no change"
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
        else:
            # Raw price candlestick (default, or price-independent overlay mode)
            fig.add_trace(
                go.Candlestick(
                    x=base_df.index,
                    open=open_col,
                    high=high_col,
                    low=low_col,
                    close=close_col,
                    name=base_ticker,
                    increasing_line_color=CANDLE_INCREASING,
                    decreasing_line_color=CANDLE_DECREASING,
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── Panel 1: Comparison overlay lines ─────────────────────────────────────
    # Hidden y-axes for independent price mode (y5, y6, … up to y16 for 12 comps).
    # make_subplots already owns y, y2, y3, y4, so overlay axes start at y5.
    overlay_yaxis_defs = {}

    if overlay_enabled:
        for i, comp_ticker in enumerate(comparison_tickers):
            if comp_ticker not in data_dict or data_dict[comp_ticker].empty:
                continue
            comp_close = data_dict[comp_ticker]['Close'].dropna()
            if comp_close.empty:
                continue
            color = PALETTE[i % len(PALETTE)]

            if overlay_scale == "pct":
                first_comp = comp_close.iloc[0]
                y_vals = (comp_close / first_comp - 1) * 100
                fig.add_trace(
                    go.Scatter(
                        x=y_vals.index,
                        y=y_vals.values,
                        mode='lines',
                        name=comp_ticker,
                        line=dict(color=color, width=2),
                        legendgroup=comp_ticker,
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
            else:
                # Each comparison ticker gets its own invisible y-axis so its price
                # shape is rendered correctly without distorting the base price axis.
                yaxis_idx = 5 + i
                fig.add_trace(
                    go.Scatter(
                        x=comp_close.index,
                        y=comp_close.values,
                        mode='lines',
                        name=comp_ticker,
                        line=dict(color=color, width=2),
                        legendgroup=comp_ticker,
                        showlegend=False,
                        xaxis='x',
                        yaxis=f'y{yaxis_idx}',
                    )
                )
                overlay_yaxis_defs[f'yaxis{yaxis_idx}'] = dict(
                    overlaying='y',
                    visible=False,
                    anchor='x',
                )

        if overlay_yaxis_defs:
            fig.update_layout(**overlay_yaxis_defs)

    # ── Panels 2 & 3: Momentum and Correlation lines ───────────────────────────
    if momentum_dict and any(not mo.empty for mo in momentum_dict.values()):
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

    fig.add_hline(y=0,    line_dash="dash", line_color="gray",      line_width=1, row=3, col=1)
    fig.add_hline(y=0.5,  line_dash="dot",  line_color="lightgray", line_width=1, row=3, col=1)
    fig.add_hline(y=-0.5, line_dash="dot",  line_color="lightgray", line_width=1, row=3, col=1)

    for i, comp_ticker in enumerate(comparison_tickers):
        color = PALETTE[i % len(PALETTE)]

        # Momentum — legend entry lives on the correlation trace
        if comp_ticker in momentum_dict and not momentum_dict[comp_ticker].empty:
            mo_series = momentum_dict[comp_ticker]
            fig.add_trace(
                go.Scatter(
                    x=mo_series.index,
                    y=mo_series.values,
                    mode='lines',
                    name=comp_ticker,
                    line=dict(color=color, width=2),
                    legendgroup=comp_ticker,
                    showlegend=False,
                ),
                row=2, col=1,
            )

        # Correlation — carries the single legend entry per ticker
        if comp_ticker in corr_dict and not corr_dict[comp_ticker].empty:
            corr_series = corr_dict[comp_ticker]
            fig.add_trace(
                go.Scatter(
                    x=corr_series.index,
                    y=corr_series.values,
                    mode='lines',
                    name=comp_ticker,
                    line=dict(color=color, width=2),
                    legendgroup=comp_ticker,
                    showlegend=True,
                ),
                row=3, col=1,
            )

    # ── Panel 4: Historical volatility ────────────────────────────────────────
    if base_ticker in vol_dict and not vol_dict[base_ticker].empty:
        vol_base = vol_dict[base_ticker]
        fig.add_trace(
            go.Scatter(
                x=vol_base.index,
                y=vol_base.values,
                mode='lines',
                name=base_ticker,
                line=dict(color=BASE_VOL_COLOR, width=2),
                legendgroup='_base_vol',
                showlegend=True,
            ),
            row=4, col=1,
        )

    for i, comp_ticker in enumerate(comparison_tickers):
        color = PALETTE[i % len(PALETTE)]
        if comp_ticker in vol_dict and not vol_dict[comp_ticker].empty:
            vol_series = vol_dict[comp_ticker]
            fig.add_trace(
                go.Scatter(
                    x=vol_series.index,
                    y=vol_series.values,
                    mode='lines',
                    name=comp_ticker,
                    line=dict(color=color, width=2),
                    legendgroup=comp_ticker,
                    showlegend=False,
                ),
                row=4, col=1,
            )

    # ── X-axis rangebreaks ─────────────────────────────────────────────────────
    is_futures = any(t.endswith('=F') for t in data_dict.keys())
    rangebreaks = []
    if interval == "1d":
        rangebreaks.append(dict(bounds=["sat", "mon"]))
    elif interval == "1h" and not is_futures:
        rangebreaks.append(dict(bounds=["sat", "mon"]))

    fig.update_xaxes(rangebreaks=rangebreaks)
    fig.update_xaxes(rangeslider_visible=False, showticklabels=True)

    # ── Y-axis titles ──────────────────────────────────────────────────────────
    panel1_ylabel = "% Change" if (overlay_enabled and overlay_scale == "pct") else "Price"
    fig.update_yaxes(title_text=panel1_ylabel, row=1, col=1)
    fig.update_yaxes(title_text="Momentum",    row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=3, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=4, col=1)

    # ── Y-axis scaling ─────────────────────────────────────────────────────────
    # When y_scale == 1.0 let Plotly autorange; any other value sets an explicit
    # range so the scale persists across re-renders.
    if y_scale == 1.0:
        fig.update_yaxes(autorange=True)
    else:
        # Panel 1 — price or % change
        if base_ticker in data_dict and not data_dict[base_ticker].empty:
            base_close = data_dict[base_ticker]['Close']
            if overlay_enabled and overlay_scale == "pct":
                first_close = base_close.dropna().iloc[0]
                p1_series = [(base_close / first_close - 1) * 100]
                for ct in comparison_tickers:
                    if ct in data_dict and not data_dict[ct].empty:
                        cs = data_dict[ct]['Close'].dropna()
                        if not cs.empty:
                            p1_series.append((cs / cs.iloc[0] - 1) * 100)
            else:
                p1_series = [base_close]
            r = _scaled_yrange(p1_series, y_scale)
            if r:
                fig.update_yaxes(range=r, row=1, col=1)

        # Panel 2 — momentum
        r = _scaled_yrange(list(momentum_dict.values()), y_scale)
        if r:
            fig.update_yaxes(range=r, row=2, col=1)

        # Panel 3 — correlation (data lives in [-1, 1]; scale around that)
        r = _scaled_yrange(list(corr_dict.values()), y_scale)
        if r:
            # Never let the axis exceed the natural [-1, 1] bounds when zooming out
            r = [max(r[0], -1.05), min(r[1], 1.05)]
            fig.update_yaxes(range=r, row=3, col=1)

        # Panel 4 — volatility
        r = _scaled_yrange(list(vol_dict.values()), y_scale)
        if r:
            fig.update_yaxes(range=[max(r[0], 0), r[1]], row=4, col=1)

    fig.update_layout(
        title=title,
        height=1400,
        xaxis_rangeslider_visible=False,
    )

    fig = apply_light_professional_theme(fig)
    return fig


def build_regime_scatter(scorecard: list, corr_high: float = 0.6, corr_low: float = 0.4) -> go.Figure:
    """
    Scatter of each comparable in momentum × correlation space.
    Quadrant shading indicates regime zones without imposing hard category boundaries.
    Accepts the raw scorecard list from compute_signals (uses _mom and _corr private fields).
    """
    rows = [r for r in scorecard if r.get("_mom") is not None and r.get("_corr") is not None]

    fig = go.Figure()

    if not rows:
        fig.update_layout(
            xaxis=dict(title="Correlation", range=[-1, 1]),
            yaxis=dict(title="Momentum"),
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="#F8F9FA",
            margin=dict(l=50, r=20, t=20, b=40),
        )
        return fig

    tickers = [r["Ticker"] for r in rows]
    moms    = [r["_mom"]   for r in rows]
    corrs   = [r["_corr"]  for r in rows]
    setups  = [r["Setup"]  for r in rows]

    mom_abs = max((abs(v) for v in moms), default=1.0) or 1.0
    pad     = mom_abs * 0.30
    y_min   = -(mom_abs + pad)
    y_max   =  (mom_abs + pad)

    SETUP_COLORS = {
        "Trend Leader":               "#1a9641",
        "Idiosyncratic Outperformer": "#00acc1",
        "Momentum Long":              "#66c2a5",
        "Convergence":                "#f46d43",
        "Idiosyncratic Lag":          "#9e0142",
        "Momentum Short":             "#d73027",
        "Neutral":                    "#9e9e9e",
        "—":                          "#cccccc",
    }

    # Quadrant background shading
    shading = [
        (-1,        corr_low,  0,     y_max,  "rgba(0,172,193,0.07)"),   # Idiosync. Outperformer
        (corr_high, 1,         0,     y_max,  "rgba(26,150,65,0.07)"),    # Trend Leader
        (corr_high, 1,         y_min, 0,      "rgba(244,109,67,0.07)"),   # Convergence
        (-1,        corr_low,  y_min, 0,      "rgba(158,1,66,0.07)"),     # Idiosync. Lag
    ]
    for x0, x1, y0, y1, color in shading:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, line_width=0, layer="below",
        )

    # Reference lines
    fig.add_hline(y=0,         line_color="#aaa", line_width=1, line_dash="solid")
    fig.add_vline(x=corr_high, line_color="#bbb", line_width=1, line_dash="dot")
    fig.add_vline(x=corr_low,  line_color="#bbb", line_width=1, line_dash="dot")

    # One trace per point so each can have its own colour
    for ticker, mom, corr, setup in zip(tickers, moms, corrs, setups):
        color = SETUP_COLORS.get(setup, "#888888")
        fig.add_trace(go.Scatter(
            x=[corr], y=[mom],
            mode="markers+text",
            text=[ticker],
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(size=11, color=color, line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"Corr: {corr:.3f}<br>"
                f"Mom: {mom:.3f}<br>"
                f"Setup: {setup}<extra></extra>"
            ),
        ))

    # Faint quadrant labels
    label_y_top = y_max * 0.75
    label_y_bot = y_min * 0.75
    for x_c, lbl_top, lbl_bot in [
        (0.80,  "Trend Leader",    "Convergence"),
        (-0.70, "Idiosync. Out.",  "Idiosync. Lag"),
    ]:
        fig.add_annotation(x=x_c, y=label_y_top, text=lbl_top,
                           showarrow=False, font=dict(size=9, color="#bbb"), xanchor="center")
        fig.add_annotation(x=x_c, y=label_y_bot, text=lbl_bot,
                           showarrow=False, font=dict(size=9, color="#bbb"), xanchor="center")

    fig.update_layout(
        xaxis=dict(title="Correlation", range=[-1, 1], gridcolor="#eee", zeroline=False),
        yaxis=dict(title="Momentum", range=[y_min, y_max], gridcolor="#eee", zeroline=False),
        paper_bgcolor="white",
        plot_bgcolor="#F8F9FA",
        height=320,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    return fig
