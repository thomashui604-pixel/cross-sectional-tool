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
    # History = `window` bars BEFORE the current (-2) bar, exclusive of current.
    # iloc[a:-2] excludes positions -2 and -1 — i.e. the current bar and any
    # partial bar — so the comparison is purely backward-looking.
    start = max(-window - 2, -len(s))
    history = s.iloc[start:-2]
    if history.empty:
        return 50.0
    return float((history <= current).mean() * 100)


def _classify_setup(
    mom: float,
    corr: float,
    mom_pct: float,
    corr_high: float = 0.6,
    corr_low: float = 0.4,
    neutral_half_band: float = 10.0,
    crowded_pct: float = 90.0,
) -> str:
    """Classify setup type from momentum direction, correlation, and percentile."""
    # Labels read from the BASE's perspective: "Base Leads / Lags" describes
    # which side of the pair is winning, with the parenthetical qualifying how
    # (with the tape = high corr; idio = low corr; mild = mid-range corr;
    # crowded = leadership at extreme percentile + high corr, mean-revert risk).
    if 50 - neutral_half_band <= mom_pct <= 50 + neutral_half_band:
        return "Neutral"
    if mom > 0:
        if mom_pct >= crowded_pct and corr >= corr_high:
            return "Base Leads (Crowded)"
        if corr >= corr_high:
            return "Base Leads (with tape)"
        if corr <= corr_low:
            return "Base Leads (idio)"
        return "Base Leads (mild)"
    else:
        if corr >= corr_high:
            return "Base Lags (with tape)"
        if corr <= corr_low:
            return "Base Lags (idio)"
        return "Base Lags (mild)"


def _regime_metrics(series: pd.Series) -> dict:
    """
    Identify the currently-running directional momentum regime.

    Returns:
        {
          "age": int|None,            # bars since last sign flip of completed bars
          "peak_value": float|None,   # peak |momentum| within regime (signed)
          "peak_bars_ago": int|None,  # bars since that peak
        }
        age is None when no flip is found within the available series (regime is
        older than history, or the current sign is zero).
    """
    s = series.dropna().iloc[:-1]  # completed bars only
    if len(s) < 2:
        return {"age": None, "peak_value": None, "peak_bars_ago": None}
    signs = np.sign(s.values)
    last_sign = signs[-1]
    age = None
    if last_sign != 0:
        for i in range(len(s) - 2, -1, -1):
            if signs[i] != 0 and signs[i] != last_sign:
                # Regime began at bar i+1 (first bar with the new sign).
                age = (len(s) - 1) - (i + 1)
                break
    if age is not None:
        seg = s.iloc[-(age + 1):]
    else:
        seg = s.iloc[-min(60, len(s)):]
    if seg.empty:
        return {"age": age, "peak_value": None, "peak_bars_ago": None}
    peak_pos = int(seg.abs().values.argmax())
    return {
        "age": age,
        "peak_value": float(seg.iloc[peak_pos]),
        "peak_bars_ago": (len(seg) - 1) - peak_pos,
    }


def _stretch_sigma(series: pd.Series, window: int = 20) -> float:
    """Current momentum / rolling std of momentum.  Sign preserved.

    Magnitude = how many σ the signal sits from zero (i.e. how stretched);
    sign = current direction (so positive = positive momentum)."""
    s = series.dropna()
    if len(s) < window + 2:
        return None
    current = _safe_iloc(s, -2)
    if current is None:
        return None
    recent = s.iloc[-window - 2:-2]
    sigma = float(recent.std())
    if sigma == 0 or pd.isna(sigma):
        return None
    return current / sigma


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
                # bars_ago = distance from current completed bar (s.iloc[-1]) to the cross.
                # 0 = cross occurred on the most recent completed bar.
                return {
                    "event": f"Zero-cross → {direction}",
                    "bars_ago": (len(s) - 1) - i,
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
                    bars_ago = (len(s) - 1) - i
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
    pct_window_short: int = 63,
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
        mom_pct   = _percentile_of(mom, pct_window)
        mom_pct_s = _percentile_of(mom, pct_window_short)
        mom_trend  = _trend_dir(mom,  trend_window)
        corr_trend = _trend_dir(corr, trend_window)
        regime  = _regime_metrics(mom)
        stretch = _stretch_sigma(mom)

        vol_ratio = None
        if len(base_vol) >= 2 and len(comp_vol) >= 2:
            bv = _safe_iloc(base_vol, -2)
            cv = _safe_iloc(comp_vol, -2)
            if bv is not None and cv is not None and cv > 0:
                vol_ratio = round(bv / cv, 2)

        setup = "—"
        if mom_val is not None and corr_val is not None:
            setup = _classify_setup(mom_val, corr_val, mom_pct, corr_high_thresh, corr_low_thresh)

        if regime["age"] is not None and regime["peak_bars_ago"] is not None:
            regime_str = f"{regime['age']}B (pk {regime['peak_bars_ago']}B)"
        else:
            regime_str = "—"

        scorecard.append({
            "Ticker":    t,
            "Momentum":  round(mom_val,  3) if mom_val  is not None else None,
            "%ile L":    round(mom_pct,    0),
            "%ile S":    round(mom_pct_s,  0),
            "Trend":     mom_trend,
            "Stretch σ": round(stretch, 2) if stretch is not None else None,
            "Regime":    regime_str,
            "Correlation": round(corr_val, 3) if corr_val is not None else None,
            "Corr Trend":  corr_trend,
            "Vol Ratio":   vol_ratio,
            "Setup":       setup,
            # Private — used by regime scatter / headline, excluded from table display
            "_mom":  mom_val,
            "_corr": corr_val,
            "_mom_pct": mom_pct,
            "_mom_trend": mom_trend,
            "_regime_age": regime["age"],
            "_regime_peak_bars_ago": regime["peak_bars_ago"],
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

    # Assemble — collapse systemic events when ≥ systemic_threshold tickers fire the same type.
    # Skip systemic collapse for baskets smaller than the threshold (a 2-ticker basket where both
    # fire isn't meaningfully "systemic" — show them individually).
    all_events = []
    for event_type, ticker_events in events_by_type.items():
        if n >= systemic_threshold and len(ticker_events) >= systemic_threshold:
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

    # ── Headline ─────────────────────────────────────────────────────────────
    # One short string per comparable, framed from the BASE's perspective.
    # Positive momentum = base outperforming comparable (see compute/momentum.py:37).
    # Designed for the regime-check workflow (≤3 comparables) where breadth
    # is meaningless but a one-line read of the tape is what you actually want.
    headline = []
    for row in scorecard:
        mom_val = row.get("_mom")
        if mom_val is None:
            headline.append(f"{base_ticker} vs {row['Ticker']}: insufficient data")
            continue
        direction = "outperforming" if mom_val > 0 else "underperforming"
        pct       = int(row["%ile L"]) if row["%ile L"] is not None else 50
        trend     = row.get("_mom_trend", "→")
        age       = row.get("_regime_age")
        age_str   = f"{age}B" if age is not None else "≥series"
        headline.append(
            f"**{base_ticker}** {direction} **{row['Ticker']}** "
            f"({pct}th %ile, {trend}, {age_str})"
        )

    return {
        "breadth":   breadth,
        "scorecard": scorecard,
        "events":    all_events[:10],
        "headline":  headline,
    }
