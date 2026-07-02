# data_fetch.py — Téléchargement OHLC (yfinance) partagé par le screener thématique et la validation.
# Extrait de l'ancien backtest_signals.py pour découpler le runtime du code legacy.
import os
import time
from random import uniform

import numpy as np
import pandas as pd
import yfinance as yf

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "80"))
CHUNK_SLEEP_RANGE = (float(os.getenv("CHUNK_SLEEP_MIN", "1.0")), float(os.getenv("CHUNK_SLEEP_MAX", "3.0")))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "3.0"))
RETRY_JITTER_MAX = float(os.getenv("RETRY_JITTER_MAX", "5.0"))

def _ohlc_from_multi(df, ticker):
    try:
        sub = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        return None
    sub.columns = ["open", "high", "low", "close", "volume"]
    sub = sub.dropna(subset=["close"])
    if sub.empty:
        return None
    sub.index = pd.to_datetime(pd.Index(sub.index).date)
    return sub.sort_index()

def prefetch_ohlc(tickers, start, end, batch_size=BATCH_SIZE) -> dict:
    """Télécharge l'OHLC ajusté (open..volume) par batches, avec retry/backoff. -> {ticker: DataFrame}."""
    out = {}
    symbols = list(dict.fromkeys([str(t) for t in tickers if str(t).strip() not in ("", "nan")]))
    if not symbols:
        return out
    s = pd.to_datetime(start).strftime("%Y-%m-%d")
    e = (pd.to_datetime(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = yf.download(tickers=chunk, start=s, end=e, interval="1d",
                                 auto_adjust=True, progress=False, group_by="ticker", threads=True)
                if df is not None and not df.empty:
                    break
            except Exception as ex:
                print(f"[WARN] batch {i//batch_size+1}: {ex}")
            time.sleep(RETRY_BACKOFF_BASE * attempt + uniform(0.0, RETRY_JITTER_MAX))
        if df is None or df.empty:
            print(f"[WARN] batch vide pour {len(chunk)} tickers.")
        elif isinstance(df.columns, pd.MultiIndex):
            for t in chunk:
                sub = _ohlc_from_multi(df, t)
                if sub is not None:
                    out[t] = sub
        elif len(chunk) == 1:
            sub = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub.columns = ["open", "high", "low", "close", "volume"]
            sub = sub.dropna(subset=["close"])
            if not sub.empty:
                sub.index = pd.to_datetime(pd.Index(sub.index).date)
                out[chunk[0]] = sub.sort_index()
        time.sleep(uniform(*CHUNK_SLEEP_RANGE))
    print(f"[PREFETCH] {len(out)}/{len(symbols)} tickers avec OHLC.")
    return out

def _mcap_one(ticker: str):
    """Market cap USD d'un ticker via yfinance fast_info (best-effort, multi-clés)."""
    try:
        fi = yf.Ticker(ticker).fast_info
    except Exception:
        return None
    for k in ("market_cap", "marketCap"):
        try:
            v = fi[k]
            if v and float(v) > 0:
                return float(v)
        except Exception:
            pass
    # fallback: dernier prix * actions en circulation
    try:
        px = float(fi["last_price"]); sh = float(fi["shares"])
        if px > 0 and sh > 0:
            return px * sh
    except Exception:
        pass
    return None

def fetch_mcaps(tickers, pause: float = 0.0) -> dict:
    """Retourne {ticker: market_cap_usd}. best-effort (ticker absent si indisponible)."""
    out = {}
    for t in tickers:
        mc = _mcap_one(str(t))
        if mc:
            out[str(t)] = mc
        if pause:
            time.sleep(pause)
    return out


def spy_equity(spy_df, start, end) -> pd.DataFrame:
    """Equity SPY buy&hold base 100 sur [start,end] (utile pour la validation vs marché)."""
    if spy_df is None or spy_df.empty or pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["date", "equity"])
    px = spy_df.loc[(spy_df.index >= pd.to_datetime(start)) & (spy_df.index <= pd.to_datetime(end)), "close"].dropna()
    if px.empty:
        return pd.DataFrame(columns=["date", "equity"])
    base = float(px.iloc[0])
    if not np.isfinite(base) or base == 0:
        return pd.DataFrame(columns=["date", "equity"])
    return pd.DataFrame({"date": pd.to_datetime(px.index), "equity": (px / base * 100.0).astype(float).values})
