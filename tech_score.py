# tech_score.py  — version sans dépendance externe
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

# ---- indicateurs basiques (pure pandas)

def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = ema(up, length)
    roll_down = ema(down, length)
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line.rename(f"MACD_{fast}_{slow}_{signal}"), \
           hist.rename(f"MACDh_{fast}_{slow}_{signal}"), \
           signal_line.rename(f"MACDs_{fast}_{slow}_{signal}")

def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    # Wilder's ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean() / atr
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx.rename(f"ADX_{length}")

# ---- pipeline “features + label”

def compute_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: index datetime, colonnes: open, high, low, close, volume
    retourne df avec colonnes techniques
    """
    out = df.copy()
    out["rsi"] = rsi(out["close"], 14)
    m, h, s = macd(out["close"], 12, 26, 9)
    out = pd.concat([out, m, h, s], axis=1)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["adx"] = adx(out["high"], out["low"], out["close"], 14)
    return out

def tech_label_from_features(df_feat: pd.DataFrame) -> Tuple[str, float]:
    # utilise la dernière ligne non-nan si possible
    dff = df_feat.dropna()
    last = dff.iloc[-1] if len(dff) else df_feat.iloc[-1]

    rsi_v  = last.get("rsi")
    macd_v = last.get("MACD_12_26_9")
    macds  = last.get("MACDs_12_26_9")
    ema20  = last.get("ema20")
    ema50  = last.get("ema50")
    adx_v  = last.get("ADX_14")
    close  = last.get("close")

    def nz(x):  # non-zero / non-null helper
        return (x is not None) and not (isinstance(x, float) and np.isnan(x))

    s_rsi = 1.0 if (nz(rsi_v) and rsi_v >= 55) else (0.5 if (nz(rsi_v) and rsi_v >= 50) else 0.0)
    s_macd = 1.0 if (nz(macd_v) and nz(macds) and macd_v > macds > 0) else (0.5 if (nz(macd_v) and nz(macds) and macd_v > macds) else 0.0)
    s_trend = 1.0 if (nz(ema20) and nz(ema50) and nz(close) and (ema20 > ema50 and close > ema20)) else (0.5 if (nz(ema20) and nz(ema50) and ema20 > ema50) else 0.0)
    s_adx = 1.0 if (nz(adx_v) and adx_v >= 20) else 0.0

    score = float((s_rsi + s_macd + s_trend + s_adx) / 4.0)

    if score >= 0.70:
        label = "STRONG_BUY"
    elif score >= 0.55:
        label = "BUY"
    elif score >= 0.45:
        label = "HOLD"
    else:
        label = "SELL"
    return label, score
