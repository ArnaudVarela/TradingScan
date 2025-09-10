# tech_score.py
# Calcule quelques features techniques et en déduit un label/score simple.
import numpy as np
import pandas as pd
import pandas_ta as ta

def compute_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: colonnes ['open','high','low','close','volume'], index datetime
    Retourne un DataFrame 'out' qui contient:
      - close
      - MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
      - RSI_14
      - SMA_20, SMA_50, SMA_200
    """
    out = pd.DataFrame(index=df.index.copy())
    out["close"] = pd.to_numeric(df["close"], errors="coerce")

    macd_df = ta.macd(out["close"], fast=12, slow=26, signal=9)  # déjà bien nommées
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

    macd_col  = "MACD_12_26_9"
    macds_col = "MACDs_12_26_9"
    macdh_col = "MACDh_12_26_9"
    macd  = out[macd_col].iloc[-1]  if macd_col  in out.columns else np.nan
    macds = out[macds_col].iloc[-1] if macds_col in out.columns else np.nan
    macdh = out[macdh_col].iloc[-1] if macdh_col in out.columns else np.nan

    score = 0.5

    # au-dessus des MM = mieux
    above20  = pd.notna(c) and pd.notna(sma20)  and c > sma20
    above50  = pd.notna(c) and pd.notna(sma50)  and c > sma50
    above200 = pd.notna(c) and pd.notna(sma200) and c > sma200

    score += 0.07 * above20 + 0.10 * above50 + 0.13 * above200

    # MACD > signal et histogramme positif = momentum
    if pd.notna(macd) and pd.notna(macds):
        score += 0.10 if macd > macds else -0.05
    if pd.notna(macdh):
        score += 0.07 if macdh > 0 else -0.03

    # RSI douce: 45-65 neutre, >65 bullish, <45 bearish
    if pd.notna(rsi14):
        if rsi14 >= 65:
            score += 0.08
        elif rsi14 <= 45:
            score -= 0.08

    # clamp
    score = float(max(0.0, min(1.0, score)))

    # mapping label
    if score >= 0.8:
        label = "STRONG_BUY"
    elif score >= 0.6:
        label = "BUY"
    elif score <= 0.35:
        label = "SELL"
    else:
        label = "HOLD"

    return label, score
