import pandas as pd
import numpy as np


def _safe_iloc(series: pd.Series, idx: int):
    """Return series.iloc[idx], or None if out of bounds or NaN."""
    try:
        val = series.iloc[idx]
        return None if pd.isna(val) else float(val)
    except IndexError:
        return None


def _trend_dir(series: pd.Series, window: int) -> str:
    """Return ↑ / → / ↓ based on change between iloc[-2] and iloc[-2 - window]."""
    s = series.dropna()
    if len(s) < window + 2:
        return "→"
    current = _safe_iloc(s, -2)
    past = _safe_iloc(s, -2 - window)
    if current is None or past is None:
        return "→"
    delta = current - past
    threshold = max(abs(current) * 0.05, 1e-6)
    if delta > threshold:
        return "↑"
    if delta < -threshold:
        return "↓"
    return "→"


def _percentile_of(series: pd.Series, window: int = 252) -> float:
    """
    Percentile rank of iloc[-2] within its own prior `window` bars.
    Purely backward-looking: compares iloc[-2] against iloc[-window-1 : -1].
    """
    s = series.dropna()
    if len(s) < 3:
        return 50.0
    current = _safe_iloc(s, -2)
    if current is None:
        return 50.0
    start = max(-window - 1, -len(s))
    history = s.iloc[start:-1]
    if history.empty:
        return 50.0
    return float((history <= current).mean() * 100)


def _classify_setup(
    mom: float,
    corr: float,
    mom_pct: float,
    corr_high: float = 0.6,
    corr_low: float = 0.4,
    neutral_half_band: float = 20.0,
) -> str:
    """Classify setup type from momentum direction, correlation, and percentile."""
    if 50 - neutral_half_band <= mom_pct <= 50 + neutral_half_band:
        return "Neutral"
    if mom > 0:
        if corr >= corr_high:
            return "Trend Leader"
        if corr <= corr_low:
            return "Idiosyncratic Outperformer"
        return "Momentum Long"
    else:
        if corr >= corr_high:
            return "Convergence"
        if corr <= corr_low:
            return "Idiosyncratic Lag"
        return "Momentum Short"


def _zero_cross_event(series: pd.Series, persistence: int = 3, lookback: int = 30):
    """
    Find the most recent zero-cross in `series` that held for `persistence` completed
    bars, searching back at most `lookback` bars.  Returns a dict or None.
    """
    s = series.dropna().iloc[:-1]  # completed bars only
    if len(s) < persistence + 2:
        return None
    signs = np.sign(s.values)
    search_from = max(1, len(s) - lookback)

    for i in range(len(s) - persistence, search_from - 1, -1):
        if i <= 0:
            break
        if signs[i] == 0 or signs[i - 1] == 0:
            continue
        if signs[i] != signs[i - 1]:
            post = signs[i: i + persistence]
            if len(post) == persistence and np.all(post == signs[i]):
                direction = "positive" if signs[i] > 0 else "negative"
                return {
                    "event": f"Zero-cross → {direction}",
                    "bars_ago": len(s) - i,
                    "value": round(float(s.iloc[i]), 4),
                }
    return None


def _percentile_extreme_event(
    series: pd.Series,
    pct_window: int = 252,
    high_pct: float = 90.0,
    low_pct: float = 10.0,
):
    """Flag if the last completed bar sits at a percentile extreme."""
    s = series.dropna()
    if len(s) < 10:
        return None
    pct = _percentile_of(s, pct_window)
    if pct >= high_pct:
        return {"event": f"Momentum extreme (≥{high_pct:.0f}th pct)", "bars_ago": 0, "value": round(pct, 1)}
    if pct <= low_pct:
        return {"event": f"Momentum extreme (≤{low_pct:.0f}th pct)", "bars_ago": 0, "value": round(pct, 1)}
    return None


def _corr_shift_event(
    series: pd.Series,
    high_thresh: float = 0.6,
    neg_thresh: float = -0.3,
    persistence: int = 3,
    lookback: int = 30,
):
    """Most recent correlation threshold crossing that held for `persistence` bars."""
    s = series.dropna().iloc[:-1]
    if len(s) < persistence + 2:
        return None

    best = None
    best_bars_ago = lookback + 1

    for thresh in [high_thresh, neg_thresh]:
        above = s.values > thresh if thresh >= 0 else s.values < thresh
        search_from = max(1, len(s) - lookback)

        for i in range(len(s) - persistence, search_from - 1, -1):
            if i <= 0:
                break
            if above[i] != above[i - 1]:
                post = above[i: i + persistence]
                if len(post) == persistence and np.all(post == above[i]):
                    bars_ago = len(s) - i
                    if bars_ago < best_bars_ago:
                        best_bars_ago = bars_ago
                        # For positive thresholds above[i] means corr just crossed up;
                        # for negative thresholds above[i] means corr < thresh (crossed down).
                        if thresh >= 0:
                            direction = "rose above" if above[i] else "fell below"
                        else:
                            direction = "fell below" if above[i] else "rose above"
                        best = {
                            "event": f"Corr {direction} {thresh:+.1f}",
                            "bars_ago": bars_ago,
                            "value": round(float(s.iloc[i]), 3),
                        }
    return best


def _vol_ratio_extreme_event(
    base_vol: pd.Series,
    comp_vol: pd.Series,
    high_thresh: float = 1.5,
    low_thresh: float = 0.67,
    persistence: int = 3,
):
    """Flag if base/comp vol ratio has been at an extreme for `persistence` completed bars."""
    df = pd.DataFrame({"b": base_vol, "c": comp_vol}).dropna()
    df = df[df["c"] > 0]
    if len(df) < persistence + 1:
        return None
    ratio = (df["b"] / df["c"]).iloc[:-1]
    if len(ratio) < persistence:
        return None
    last_n = ratio.iloc[-persistence:]
    current = round(float(ratio.iloc[-1]), 2)
    if (last_n >= high_thresh).all():
        return {"event": f"Vol ratio extreme (base/comp ≥{high_thresh:.1f}×)", "bars_ago": 0, "value": current}
    if (last_n <= low_thresh).all():
        return {"event": f"Vol ratio extreme (base/comp ≤{low_thresh:.1f}×)", "bars_ago": 0, "value": current}
    return None


def compute_signals(
    base_ticker: str,
    comp_tickers: list,
    momentum_full_dict: dict,
    corr_full_dict: dict,
    vol_dict: dict,
    mo_lookback: int = 20,
    pct_window: int = 252,
    persistence_bars: int = 3,
    corr_high_thresh: float = 0.6,
    corr_low_thresh: float = 0.4,
    corr_regime_neg: float = -0.3,
    vol_ratio_high: float = 1.5,
    vol_ratio_low: float = 0.67,
    extremes_pct_high: float = 90.0,
    extremes_pct_low: float = 10.0,
    systemic_threshold: int = 6,
) -> dict:
    """
    Derive actionable signals from pre-computed momentum, correlation, and volatility series.

    Uses iloc[-2] (last *completed* bar) throughout to avoid partial-bar distortion
    during live market hours.  Rolling percentiles use only backward-looking history.

    Returns:
        {
            "breadth":   {"count": int, "total": int, "trend": int},
            "scorecard": [row_dict, ...],   # one per comparable; private fields prefixed _
            "events":    [event_dict, ...], # up to 10, sorted by recency (bars_ago asc)
        }
    """
    trend_window = max(3, mo_lookback // 4)
    valid = [t for t in comp_tickers if t in momentum_full_dict and t in corr_full_dict]

    # ── Breadth ───────────────────────────────────────────────────────────────
    now_pos = 0
    past_pos = 0
    for t in valid:
        mom = momentum_full_dict[t].dropna()
        if len(mom) >= 2 and mom.iloc[-2] > 0:
            now_pos += 1
        if len(mom) >= trend_window + 2 and mom.iloc[-2 - trend_window] > 0:
            past_pos += 1

    breadth = {"count": now_pos, "total": len(valid), "trend": now_pos - past_pos}

    # ── Scorecard ─────────────────────────────────────────────────────────────
    base_vol = vol_dict.get(base_ticker, pd.Series(dtype=float))
    scorecard = []

    for t in valid:
        mom  = momentum_full_dict[t].dropna()
        corr = corr_full_dict[t].dropna()
        comp_vol = vol_dict.get(t, pd.Series(dtype=float))

        mom_val  = _safe_iloc(mom,  -2) if len(mom)  >= 2 else None
        corr_val = _safe_iloc(corr, -2) if len(corr) >= 2 else None
        mom_pct  = _percentile_of(mom, pct_window)
        mom_trend  = _trend_dir(mom,  trend_window)
        corr_trend = _trend_dir(corr, trend_window)

        vol_ratio = None
        if len(base_vol) >= 2 and len(comp_vol) >= 2:
            bv = _safe_iloc(base_vol, -2)
            cv = _safe_iloc(comp_vol, -2)
            if bv is not None and cv is not None and cv > 0:
                vol_ratio = round(bv / cv, 2)

        setup = "—"
        if mom_val is not None and corr_val is not None:
            setup = _classify_setup(mom_val, corr_val, mom_pct, corr_high_thresh, corr_low_thresh)

        scorecard.append({
            "Ticker":    t,
            "Momentum":  round(mom_val,  3) if mom_val  is not None else None,
            "Mom %ile":  round(mom_pct,  0),
            "Trend":     mom_trend,
            "Correlation": round(corr_val, 3) if corr_val is not None else None,
            "Corr Trend":  corr_trend,
            "Vol Ratio":   vol_ratio,
            "Setup":       setup,
            # Private — used by regime scatter, excluded from table display
            "_mom":  mom_val,
            "_corr": corr_val,
        })

    # ── Events ────────────────────────────────────────────────────────────────
    events_by_type: dict = {}

    for t in valid:
        mom      = momentum_full_dict[t].dropna()
        corr     = corr_full_dict[t].dropna()
        comp_vol = vol_dict.get(t, pd.Series(dtype=float))

        checks = [
            ("Zero-cross",       _zero_cross_event,        (mom, persistence_bars)),
            ("Momentum extreme", _percentile_extreme_event, (mom, pct_window, extremes_pct_high, extremes_pct_low)),
            ("Corr shift",       _corr_shift_event,         (corr, corr_high_thresh, corr_regime_neg, persistence_bars)),
            ("Vol ratio",        _vol_ratio_extreme_event,  (base_vol, comp_vol, vol_ratio_high, vol_ratio_low, persistence_bars)),
        ]
        for key, fn, args in checks:
            evt = fn(*args)
            if evt:
                events_by_type.setdefault(key, []).append((t, evt))

    # Breadth threshold crossing
    n = len(valid)
    if n > 0:
        majority = int(n * 2 / 3) + 1
        minority = int(n * 1 / 3)
        prev_count = breadth["count"] - breadth["trend"]
        if breadth["trend"] > 0 and breadth["count"] >= majority > prev_count:
            events_by_type.setdefault("Breadth", []).append((
                "Basket",
                {"event": f"Breadth reached majority ({breadth['count']}/{n})", "bars_ago": trend_window, "value": float(breadth["count"])},
            ))
        elif breadth["trend"] < 0 and breadth["count"] <= minority < prev_count:
            events_by_type.setdefault("Breadth", []).append((
                "Basket",
                {"event": f"Breadth fell to minority ({breadth['count']}/{n})", "bars_ago": trend_window, "value": float(breadth["count"])},
            ))

    # Assemble — collapse systemic events when ≥ systemic_threshold tickers fire the same type
    all_events = []
    for event_type, ticker_events in events_by_type.items():
        if len(ticker_events) >= min(systemic_threshold, max(n, 1)):
            avg_val  = float(np.mean([e["value"] for _, e in ticker_events]))
            min_bars = min(e["bars_ago"] for _, e in ticker_events)
            all_events.append({
                "Ticker":   "Basket",
                "Event":    f"{event_type} — {len(ticker_events)}/{n} tickers (systemic)",
                "Bars Ago": min_bars,
                "Value":    round(avg_val, 3),
            })
        else:
            for ticker, evt in ticker_events:
                all_events.append({
                    "Ticker":   ticker,
                    "Event":    evt["event"],
                    "Bars Ago": evt["bars_ago"],
                    "Value":    evt["value"],
                })

    all_events.sort(key=lambda x: x["Bars Ago"])
    return {
        "breadth":   breadth,
        "scorecard": scorecard,
        "events":    all_events[:10],
    }
