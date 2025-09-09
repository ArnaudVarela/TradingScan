# tech_score.py
from __future__ import annotations
import pandas as pd
import pandas_ta as ta
from typing import Tuple

def compute_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: index datetime, colonnes: open, high, low, close, volume
    retourne df avec colonnes techniques
    """
    out = df.copy()
    out["rsi"] = ta.rsi(out["close"], length=14)
    macd = ta.macd(out["close"])  # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    out = pd.concat([out, macd], axis=1)
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    adx = ta.adx(out["high"], out["low"], out["close"], length=14)
    if isinstance(adx, pd.DataFrame) and "ADX_14" in adx.columns:
        out["adx"] = adx["ADX_14"]
    else:
        out["adx"] = None
    return out

def tech_label_from_features(df_feat: pd.DataFrame) -> Tuple[str, float]:
    last = df_feat.dropna().iloc[-1] if len(df_feat.dropna()) else df_feat.iloc[-1]

    rsi = last.get("rsi")
    macd = last.get("MACD_12_26_9")
    macds = last.get("MACDs_12_26_9")
    ema20 = last.get("ema20")
    ema50 = last.get("ema50")
    adx = last.get("adx")
    close = last.get("close")

    # signaux
    s_rsi = 1.0 if (rsi is not None and rsi >= 55) else (0.5 if (rsi is not None and rsi >= 50) else 0.0)
    s_macd = 1.0 if (macd is not None and macds is not None and macd > macds > 0) else (0.5 if (macd is not None and macds is not None and macd > macds) else 0.0)
    s_trend = 1.0 if (ema20 is not None and ema50 is not None and close is not None and (ema20 > ema50 and close > ema20)) else (0.5 if (ema20 is not None and ema50 is not None and ema20 > ema50) else 0.0)
    s_adx = 1.0 if (adx is not None and adx >= 20) else 0.0

    score = (s_rsi + s_macd + s_trend + s_adx) / 4.0

    if score >= 0.70:
        label = "STRONG_BUY"
    elif score >= 0.55:
        label = "BUY"
    elif score >= 0.45:
        label = "HOLD"
    else:
        label = "SELL"
    return label, float(score)
