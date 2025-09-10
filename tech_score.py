# tech_score.py
# Calcule quelques features techniques et en déduit un label/score simple.

import numpy as np
import pandas as pd

# --- TA import safe (classic fork ou package original)
try:
    import pandas_ta as ta
except ImportError:
    try:
        import pandas_ta_classic as ta
    except ImportError as e:
        raise ImportError(
            "Neither 'pandas_ta' nor 'pandas_ta_classic' found. "
            "Add this to requirements.txt:\n"
            "git+https://github.com/xgboosted/pandas-ta-classic.git"
        ) from e


def compute_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: colonnes ['open','high','low','close','volume'], index datetime
    Retourne un DataFrame 'out' qui contient:
      close, MACD, RSI, SMA20/50/200
    """
    out = pd.DataFrame(index=df.index.copy())
    out["close"] = pd.to_numeric(df["close"], errors="coerce")

    macd_df = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        out = out.join(macd_df)

    out["RSI_14"] = ta.rsi(out["close"], length=14)

    out["SMA_20"] = ta.sma(out["close"], length=20)
    out["SMA_50"] = ta.sma(out["close"], length=50)
    out["SMA_200"] = ta.sma(out["close"], length=200)

    return out


def tech_label_from_features(out: pd.DataFrame) -> tuple[str, float]:
    """
    Heuristique légère pour label + score technique.
    """
    if out is None or out.empty:
        return "HOLD", 0.5

    c = out["close"].iloc[-1] if pd.notna(out["close"].iloc[-1]) else np.nan
    sma20  = out["SMA_20"].iloc[-1]  if "SMA_20"  in out.columns  else np.nan
    sma50  = out["SMA_50"].iloc[-1]  if "SMA_50"  in out.columns  else np.nan
    sma200 = out["SMA_200"].iloc[-1] if "SMA_200" in out.columns else np.nan
    rsi14  = out["RSI_14"].iloc[-1]  if "RSI_14"  in out.columns  else np.nan

    macd_col, macds_col, macdh_col = "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"
    macd  = out[macd_col].iloc[-1]  if macd_col  in out.columns else np.nan
    macds = out[macds_col].iloc[-1] if macds_col in out.columns else np.nan
    macdh = out[macdh_col].iloc[-1] if macdh_col in out.columns else np.nan

    score = 0.5
    if pd.notna(c) and pd.notna(sma20) and c > sma20:   score += 0.07
    if pd.notna(c) and pd.notna(sma50) and c > sma50:   score += 0.10
    if pd.notna(c) and pd.notna(sma200) and c > sma200: score += 0.13
    if pd.notna(macd) and pd.notna(macds):
        score += 0.10 if macd > macds else -0.05
    if pd.notna(macdh):
        score += 0.07 if macdh > 0 else -0.03
    if pd.notna(rsi14):
        if rsi14 >= 65: score += 0.08
        elif rsi14 <= 45: score -= 0.08

    score = float(max(0.0, min(1.0, score)))

    if score >= 0.8:
        label = "STRONG_BUY"
    elif score >= 0.6:
        label = "BUY"
    elif score <= 0.35:
        label = "SELL"
    else:
        label = "HOLD"

    return label, score
