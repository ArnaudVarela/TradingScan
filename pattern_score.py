# pattern_score.py — Scoring /100 des setups de PRÉ-EXPLOSION (quantifiable, prix/volume réels).
#
# Philosophie (garde-fou CLAUDE.md) : pas de reconnaissance de "forme" au jugé, mais des proxys
# MESURABLES de fin d'accumulation / cup&handle / VCP :
#   - contexte de tendance saine (stage 2)     - base tendue près des plus-hauts
#   - contraction de volatilité (VCP/squeeze)   - volume qui s'assèche puis prêt à exploser
#   - proximité du pivot (PRÉ-cassure)          - RSI/MACD constructifs (pas encore surchauffe)
#   - force relative vs marché                   - pénalité si déjà étendu (post-explosion)
#
# Chaque composante est dans [0,1], agrégée en pondéré, ×100. NaN-safe partout.
from __future__ import annotations
import os
import numpy as np
import pandas as pd

MIN_BARS = int(os.getenv("PAT_MIN_BARS", "150"))

# ----------------- helpers de zones (NaN-safe) -----------------

def _f(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None

def _ramp_up(x, a, b):
    """0 si x<=a, 1 si x>=b, linéaire entre (NaN -> 0)."""
    x = _f(x)
    if x is None or b <= a:
        return 0.0
    return float(min(1.0, max(0.0, (x - a) / (b - a))))

def _ramp_down(x, a, b):
    """1 si x<=a, 0 si x>=b (proximité : plus petit = mieux)."""
    return 1.0 - _ramp_up(x, a, b)

def _window(x, a, b, c, d):
    """Trapèze sweet-spot : 0 sous a, monte a->b, plateau b..c, descend c->d, 0 au-delà."""
    x = _f(x)
    if x is None:
        return 0.0
    if x < a or x > d:
        return 0.0
    if x < b:
        return _ramp_up(x, a, b)
    if x <= c:
        return 1.0
    return _ramp_down(x, c, d)

# ----------------- indicateurs -----------------

def _ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=1).mean()

def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ma_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd_hist(close):
    macd = _ema(close, 12) - _ema(close, 26)
    sig = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    return macd - sig

def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _bb_width(close, n=20):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return (4.0 * sd) / (mid + 1e-9)  # (upper-lower)/mid = 4*std/mid

def _pct_rank(series, lookback=126):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return 0.5
    w = s.tail(lookback)
    return float((w <= w.iloc[-1]).mean())

# ----------------- score principal -----------------

def preexplosion_score(df: pd.DataFrame, mkt: pd.DataFrame | None = None, min_bars: int = MIN_BARS) -> dict | None:
    if df is None or len(df) < min_bars or not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        return None
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    last = _f(c.iloc[-1])
    if last is None or last <= 0:
        return None

    e20, e50, e200 = _ema(c, 20).iloc[-1], _ema(c, 50).iloc[-1], _ema(c, 200).iloc[-1]
    slope50 = (_ema(c, 50).iloc[-1] - _ema(c, 50).iloc[-10]) / (abs(_ema(c, 50).iloc[-10]) + 1e-9)
    above200 = last > (e200 or last)
    up_align = (e20 or 0) > (e50 or 0) > (e200 or 0)
    c_trend = 0.4 * float(up_align) + 0.3 * float(above200) + 0.3 * _ramp_up(slope50, 0.0, 0.05)

    # Base / pivot sur 50 séances
    L = 50
    base_high = _f(h.iloc[-L:].max()) or last
    base_low = _f(l.iloc[-L:].min()) or last
    depth = (base_high - base_low) / (base_high + 1e-9)
    dist_to_high = (base_high - last) / (base_high + 1e-9)          # 0 = au plus-haut
    c_pivot = _ramp_down(dist_to_high, 0.0, 0.10)                    # proche du pivot = mieux
    c_base = _window(depth, 0.06, 0.12, 0.32, 0.50)                  # base ni trop plate ni cassée

    # VCP / squeeze
    bbw = _bb_width(c, 20)
    bbw_pct = _pct_rank(bbw, 126)                                    # bas = tendu
    atr14 = _atr(df, 14)
    atr_ratio = _f(atr14.iloc[-1] / (atr14.iloc[-40:].mean() + 1e-9))
    c_vcp = 0.6 * (1.0 - bbw_pct) + 0.4 * _ramp_down(atr_ratio if atr_ratio else 1.0, 0.75, 1.15)

    # Volume : dry-up + accumulation (OBV)
    vol20 = _f(v.iloc[-20:].mean()) or 0.0
    vol60 = _f(v.iloc[-60:].mean()) or 1.0
    dryup = _ramp_down(vol20 / (vol60 + 1e-9), 0.75, 1.10)           # récent < long terme = assèchement
    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    obv_slope = (obv.iloc[-1] - obv.iloc[-20]) / (abs(obv.iloc[-20]) + 1e-9)
    c_volume = 0.5 * dryup + 0.5 * _ramp_up(obv_slope, -0.05, 0.15)

    # Momentum prêt (RSI zone constructive + MACD hist qui remonte)
    rsi = _f(_rsi(c).iloc[-1]) or 50.0
    c_rsi = _window(rsi, 42, 50, 65, 78)
    hist = _macd_hist(c)
    hist_now, hist_prev = _f(hist.iloc[-1]) or 0.0, _f(hist.iloc[-4]) or 0.0
    c_macd = 0.5 * float(hist_now > hist_prev) + 0.5 * _ramp_up(hist_now - hist_prev, -0.05, 0.20)
    c_momentum = 0.5 * c_rsi + 0.5 * c_macd

    # Force relative vs marché
    c_rs = 0.5
    if mkt is not None and not mkt.empty and "close" in mkt.columns:
        m = pd.to_numeric(mkt["close"], errors="coerce").reindex(c.index).ffill()
        rs_line = (c / (m + 1e-9)).dropna()
        if len(rs_line) > 20:
            c_rs = _pct_rank(rs_line, 126)

    parts = {
        "trend": c_trend, "base": c_base, "vcp": c_vcp, "volume": c_volume,
        "pivot": c_pivot, "momentum": c_momentum, "rs": c_rs,
    }
    w = {"trend": 0.18, "base": 0.15, "vcp": 0.15, "volume": 0.12, "pivot": 0.18, "momentum": 0.12, "rs": 0.10}
    raw = sum(w[k] * parts[k] for k in w)

    # Pénalité : déjà étendu (post-explosion) -> on veut le PRÉ
    ext = (last - (e20 or last)) / ((e20 or last) + 1e-9)
    overext = _ramp_up(ext, 0.10, 0.28)
    if rsi > 80:
        overext = max(overext, 0.6)
    raw *= (1.0 - 0.5 * overext)
    # Gate : vraie tendance baissière -> ce n'est pas une base de pré-explosion
    if (not above200) and slope50 < 0:
        raw *= 0.30

    score = round(100.0 * max(0.0, min(1.0, raw)), 1)
    label = _label(parts, dist_to_high, depth, bbw_pct, dryup, overext, above200, slope50)
    return {
        "score": score,
        "label": label,
        "components": {k: round(parts[k], 3) for k in parts},
        "metrics": {
            "price": round(last, 2), "rsi": round(rsi, 1),
            "macd_hist": round(hist_now, 3), "vol_dryup": round(dryup, 2),
            "dist_to_high_pct": round(dist_to_high * 100, 1),
            "base_depth_pct": round(depth * 100, 1), "bbwidth_pct": round(bbw_pct * 100, 0),
            "overext": round(overext, 2),
        },
    }

def _label(p, dist_to_high, depth, bbw_pct, dryup, overext, above200, slope50):
    if (not above200) and slope50 < 0:
        return "Tendance baissière (pas un setup)"
    if overext >= 0.5:
        return "Étendu (déjà parti)"
    near = dist_to_high <= 0.06
    if bbw_pct <= 0.25 and near:
        return "Squeeze / VCP prêt à casser"
    if 0.15 <= depth <= 0.45 and near and p["trend"] >= 0.5:
        return "Cup & Handle probable"
    if dryup >= 0.6 and near and p["trend"] >= 0.5:
        return "Fin d'accumulation"
    if p["trend"] >= 0.5 and dist_to_high > 0.10:
        return "En base (loin du pivot)"
    if p["momentum"] >= 0.6 and near:
        return "Momentum en formation"
    return "Setup en formation"
