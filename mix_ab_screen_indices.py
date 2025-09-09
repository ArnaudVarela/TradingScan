#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mix_ab_screen_indices.py
Enrichit l'univers, filtre mcap < 75B, score & bucketise selon 3 piliers:
- Tech  : SMA50>SMA200 & Close>SMA50
- TV    : tradingview_ta summary (BUY / STRONG_BUY / NEUTRAL / SELL)
- Analyst: Finnhub consensus -> BUY / STRONG_BUY / HOLD / SELL

Sorties:
- dashboard/public/candidates_all_ranked.csv
- dashboard/public/confirmed_STRONGBUY.csv
- dashboard/public/anticipative_pre_signals.csv
- dashboard/public/event_driven_signals.csv
- dashboard/public/signals_history.csv (append journalier)
"""

from __future__ import annotations
import os
import time
import math
import csv
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import yfinance as yf
from tradingview_ta import TA_Handler, Interval, Exchange
import requests

ROOT = Path(".")
PUB = ROOT / "dashboard" / "public"
PUB.mkdir(parents=True, exist_ok=True)

UNIVERSE_CSV = ROOT / "universe_in_scope.csv"  # 1 col: ticker_yf
SECTOR_CATALOG = ROOT / "sector_catalog.csv"   # optionnel (pour sector/industry)

TODAY = date.today().isoformat()
NOW_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"

# ---------- util
def _log(msg: str): print(msg, flush=True)

def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def votes_bin(n: Optional[float]) -> str:
    if pd.isna(n): return ""
    n = int(n)
    if n >= 60: return "60+"
    if n >= 40: return "40+"
    if n >= 30: return "30+"
    if n >= 20: return "20+"
    if n >= 10: return "10+"
    if n >= 5:  return "5+"
    if n > 0:   return "1+"
    return "0"

# ---------- data sources
def yf_fast_block(tickers: List[str]) -> pd.DataFrame:
    """
    Batch yfinance for price, mcap, fast info, and 20d avg $ volume.
    """
    _log("[YF] computing 20d avg dollar vol…")
    # history for avg $ vol
    hist = yf.download(
        tickers=tickers,
        period="3mo", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True
    )
    out_rows = []
    for t in tickers:
        price = np.nan
        avg_dv = np.nan
        try:
            # price: last Close from history if available
            if isinstance(hist.columns, pd.MultiIndex):
                h = hist[t]
            else:
                h = hist
            if not h.empty:
                last_close = safe_float(h["Close"].dropna().iloc[-1])
                price = last_close
                avg_dv = (h["Close"] * h["Volume"]).dropna().tail(20).mean()
        except Exception:
            pass

        out_rows.append({"ticker_yf": t, "price": price, "avg_dollar_vol": avg_dv})

    df_hist = pd.DataFrame(out_rows).set_index("ticker_yf")

    # fast info for market cap (and a fallback lastPrice)
    infos = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            fi = getattr(ticker, "fast_info", None)
            last = np.nan
            mcap = np.nan
            if fi:
                last = safe_float(getattr(fi, "last_price", None))
                mcap = safe_float(getattr(fi, "market_cap", None))
            # fallback to info (slower) if needed
            if (pd.isna(mcap) or mcap == 0) or (pd.isna(last) and pd.isna(df_hist.loc[t, "price"])):
                info = ticker.info or {}
                mcap2 = safe_float(info.get("marketCap"))
                last2 = safe_float(info.get("currentPrice", info.get("regularMarketPrice")))
                if not pd.isna(mcap2): mcap = mcap2
                if pd.isna(df_hist.loc[t, "price"]) and not pd.isna(last2): 
                    df_hist.loc[t, "price"] = last2
            infos.append({"ticker_yf": t, "mcap_usd_final": mcap})
        except Exception:
            infos.append({"ticker_yf": t, "mcap_usd_final": np.nan})

    df_info = pd.DataFrame(infos).set_index("ticker_yf")
    merged = df_hist.join(df_info, how="outer").reset_index()
    return merged

def tv_fetch_summary(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    """
    TradingView summary score in [0,1] and label (BUY/STRONG_BUY/NEUTRAL/SELL).
    We try common exchanges; library infers many symbols.
    """
    try:
        handler = TA_Handler(
            symbol=ticker,
            screener="america",
            exchange="NASDAQ" if not ticker.endswith(".NY") else "NYSE",
            interval=Interval.INTERVAL_1_DAY
        )
        s = handler.get_analysis().summary  # dict: BUY/SELL/NEUTRAL counts & RECOMMENDATION
        # score: map to 0..1
        rec = (s.get("RECOMMENDATION") or "").upper()
        buy = safe_float(s.get("BUY"), 0.0)
        sell = safe_float(s.get("SELL"), 0.0)
        neu = safe_float(s.get("NEUTRAL"), 0.0)
        tot = buy + sell + neu
        tv_score = (buy - sell) / tot if tot > 0 else 0.0
        # squash to 0..1 scale
        tv_norm = (tv_score + 1) / 2
        label = "NEUTRAL"
        if rec in ("STRONG_BUY", "BUY", "SELL", "STRONG_SELL", "NEUTRAL"):
            label = rec
        return float(tv_norm), label
    except Exception:
        return (np.nan, "")

def finnhub_consensus(ticker: str) -> Tuple[str, Optional[float]]:
    """
    Analyst consensus bucket + number of estimates/votes.
    We map Finnhub 'recommendation trends' into BUY/STRONG_BUY/HOLD/...
    """
    if not FINNHUB_API_KEY:
        return ("", np.nan)
    try:
        url = f"{FINNHUB_BASE}/stock/recommendation?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return ("", np.nan)
        latest = data[0]
        sb = safe_float(latest.get("strongBuy"), 0)
        b  = safe_float(latest.get("buy"), 0)
        h  = safe_float(latest.get("hold"), 0)
        s  = safe_float(latest.get("sell"), 0)
        ss = safe_float(latest.get("strongSell"), 0)
        votes = sb + b + h + s + ss
        # consensus bucket: majority on strongBuy/buy -> label
        if sb >= max(b, h, s, ss) and sb >= 1:
            label = "STRONG_BUY"
        elif (sb + b) > max(h, s + ss) and (sb + b) >= 1:
            label = "BUY"
        elif h >= max(sb + b, s + ss) and h >= 1:
            label = "HOLD"
        elif (s + ss) > max(sb + b, h) and (s + ss) >= 1:
            label = "SELL"
        else:
            label = ""
        return (label, votes)
    except Exception:
        return ("", np.nan)

def compute_tech_flags(tickers: List[str]) -> Dict[str, bool]:
    """
    Tech pillar: SMA50>SMA200 and Close>SMA50 on Daily.
    """
    flags = {}
    # We already downloaded 3mo. Need more for SMA200
    hist = yf.download(
        tickers=tickers, period="1y", interval="1d",
        group_by="ticker", auto_adjust=False, progress=False, threads=True
    )
    for t in tickers:
        ok = False
        try:
            if isinstance(hist.columns, pd.MultiIndex):
                h = hist[t]
            else:
                h = hist
            close = h["Close"].dropna()
            if len(close) >= 200:
                sma50  = close.rolling(50).mean().iloc[-1]
                sma200 = close.rolling(200).mean().iloc[-1]
                last   = close.iloc[-1]
                ok = (last > sma50) and (sma50 > sma200)
        except Exception:
            ok = False
        flags[t] = bool(ok)
    return flags

# ---------- bucketing / scoring
def pillar_bools(tech_ok: bool, tv_label: str, an_label: str) -> Tuple[bool, bool, bool]:
    p_tech = bool(tech_ok)
    p_tv   = (tv_label.upper() in ("BUY", "STRONG_BUY"))
    p_an   = (an_label.upper() in ("BUY", "STRONG_BUY"))
    return p_tech, p_tv, p_an

def assign_bucket(p_tech: bool, p_tv: bool, p_an: bool, tv_label: str, an_label: str) -> str:
    cnt = int(p_tech) + int(p_tv) + int(p_an)
    if cnt == 3 and tv_label.upper() == "STRONG_BUY" and an_label.upper() in ("BUY","STRONG_BUY"):
        return "confirmed"
    if cnt == 2:
        return "pre_signal"
    return "event"

def rank_score(tv_norm: float, cnt: int, votes: Optional[float]) -> float:
    # 0.5 * TVscore + 0.4 * normalized pillars + 0.1 * votes bucket
    tv = 0 if pd.isna(tv_norm) else float(tv_norm)
    pill = cnt / 3.0
    vb = 0.0
    if not pd.isna(votes):
        v = float(votes)
        if v >= 60: vb = 1.0
        elif v >= 40: vb = 0.8
        elif v >= 30: vb = 0.6
        elif v >= 20: vb = 0.5
        elif v >= 10: vb = 0.4
        elif v >= 5: vb = 0.2
        else: vb = 0.0
    return round(0.5*tv + 0.4*pill + 0.1*vb, 6)

# ---------- main
def main():
    _log("[UNI] loading scope…")
    uni = pd.read_csv(UNIVERSE_CSV)
    tickers = uni["ticker_yf"].dropna().astype(str).str.upper().tolist()
    tickers = sorted(list(dict.fromkeys(tickers)))
    _log(f"[UNI] in-scope: {len(tickers)}")

    # sector/industry (optional)
    sector_map = {}
    if SECTOR_CATALOG.exists():
        sc = pd.read_csv(SECTOR_CATALOG)
        for _, r in sc.iterrows():
            sector_map[str(r["ticker"]).upper()] = (r.get("sector"), r.get("industry"))

    # --- yfinance block
    yf_df = yf_fast_block(tickers)

    # --- TradingView (bounded, retries)
    tv_rows = []
    filled = 0
    for t in tickers:
        sc, lab = tv_fetch_summary(t)
        if lab: filled += 1
        tv_rows.append({"ticker_yf": t, "tv_score": sc, "tv_reco": lab})
        # gentle pacing to avoid rate limits
        time.sleep(0.05)
    _log(f"[TV] filled {filled}/{len(tickers)}")

    tv_df = pd.DataFrame(tv_rows)

    # --- Finnhub analysts
    an_rows = []
    filled = 0
    for t in tickers:
        lab, votes = finnhub_consensus(t)
        if lab: filled += 1
        an_rows.append({"ticker_yf": t, "analyst_bucket": lab, "analyst_votes": votes})
        time.sleep(0.02)
    _log(f"[Finnhub] {filled}/{len(tickers)}")

    # --- Tech pillar
    tech_flags = compute_tech_flags(tickers)
    tech_df = pd.DataFrame([{"ticker_yf": t, "p_tech": bool(v)} for t, v in tech_flags.items()])

    # --- merge
    base = (yf_df
            .merge(tv_df, on="ticker_yf", how="left")
            .merge(an_df := pd.DataFrame(an_rows), on="ticker_yf", how="left")
            .merge(tech_df, on="ticker_yf", how="left"))

    # sector/industry
    base["sector"] = base["ticker_yf"].map(lambda t: sector_map.get(t, (None, None))[0])
    base["industry"] = base["ticker_yf"].map(lambda t: sector_map.get(t, (None, None))[1])

    # --- pillars & buckets
    pts = []
    buckets = []
    ranks = []
    vbins = []
    for _, r in base.iterrows():
        p_tech, p_tv, p_an = pillar_bools(bool(r.get("p_tech")), str(r.get("tv_reco") or ""), str(r.get("analyst_bucket") or ""))
        cnt = int(p_tech) + int(p_tv) + int(p_an)
        buck = assign_bucket(p_tech, p_tv, p_an, str(r.get("tv_reco") or ""), str(r.get("analyst_bucket") or ""))
        sc = rank_score(safe_float(r.get("tv_score"), 0.0), cnt, safe_float(r.get("analyst_votes"), np.nan))
        pts.append((p_tech, p_tv, p_an, cnt))
        buckets.append(buck)
        ranks.append(sc)
        vbins.append(votes_bin(r.get("analyst_votes")))

    base["p_tech"] = [x[0] for x in pts]
    base["p_tv"]   = [x[1] for x in pts]
    base["p_an"]   = [x[2] for x in pts]
    base["pillars_met"] = [x[3] for x in pts]
    base["bucket"] = buckets
    base["rank_score"] = ranks
    base["votes_bin"] = vbins

    # --- ONLY filter: MCAP < 75B
    base["mcap_usd_final"] = base["mcap_usd_final"].astype(float)
    pre_filter_rows = len(base)
    base = base[ (base["mcap_usd_final"].notna()) & (base["mcap_usd_final"] < 75e9) ]
    _log(f"[FILTER] MCAP<75B: {len(base)}/{pre_filter_rows} kept")

    # price/last
    base["last"] = base["price"]

    # tidy columns
    cols = ["ticker_yf","ticker_yf","price","last","mcap_usd_final","mcap","avg_dollar_vol",
            "tv_score","tv_reco","analyst_bucket","analyst_votes",
            "sector","industry","p_tech","p_tv","p_an","pillars_met","votes_bin","rank_score","bucket"]
    base["ticker_tv"] = base["ticker_yf"]
    base["mcap"] = base["mcap_usd_final"]  # legacy
    out = base[["ticker_yf","ticker_tv","price","last","mcap_usd_final","mcap","avg_dollar_vol",
                "tv_score","tv_reco","analyst_bucket","analyst_votes",
                "sector","industry","p_tech","p_tv","p_an","pillars_met","votes_bin","rank_score","bucket"]].copy()

    # --- save master
    out_sorted = out.sort_values(["rank_score","avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)
    out_sorted.to_csv(PUB / "candidates_all_ranked.csv", index=False)
    _log(f"[SAVE] candidates_all_ranked.csv | rows={len(out_sorted)} | cols={out_sorted.shape[1]}")

    # splits
    confirmed = out_sorted[out_sorted["bucket"]=="confirmed"]
    pre_sig   = out_sorted[out_sorted["bucket"]=="pre_signal"]
    event     = out_sorted[out_sorted["bucket"]=="event"]

    confirmed.to_csv(PUB / "confirmed_STRONGBUY.csv", index=False)
    pre_sig.to_csv(PUB / "anticipative_pre_signals.csv", index=False)
    event.to_csv(PUB / "event_driven_signals.csv", index=False)
    _log(f"[SAVE] confirmed={len(confirmed)} pre={len(pre_sig)} event={len(event)}")

    # history (append-last 100 unique)
    hist_path = PUB / "signals_history.csv"
    today_rows = out_sorted[["ticker_yf","bucket","tv_reco","analyst_bucket","sector"]].copy()
    today_rows.insert(0, "date", TODAY)
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist = pd.concat([hist, today_rows], ignore_index=True)
        # keep last 100 rows
        hist = hist.tail(100)
    else:
        hist = today_rows
    hist.to_csv(hist_path, index=False)
    _log("[SAVE] signals_history.csv updated")

if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    try:
        main()
    except Exception as e:
        _log(f"[FATAL] {e}")
        sys.exit(1)
