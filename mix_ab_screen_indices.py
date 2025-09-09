#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mix_ab_screen_indices.py — Robuste, tout-en-un.

Objectif:
- Charger l'univers (universe_in_scope.csv).
- Enrichir pour chaque ticker: prix/last, market cap, avg dollar volume (20d),
  reco TradingView, reco analystes (Finnhub si dispo), petit signal "tech".
- Calculer 3 "pillars": p_tech, p_tv, p_an + score de rang.
- Déterminer "bucket" (confirmed / pre_signal / event).
- Sauver les CSV attendus par le front (avec alias de colonnes).
- Tenir un historique minimal des signaux du jour.

Dépendances déjà dans requirements.txt:
  pandas, yfinance, tradingview_ta, requests
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import gzip
import uuid
import queue
import shutil
import random
import pathlib
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from tradingview_ta import TA_Handler, Interval, Exchange

# ---------- Chemins et constantes ----------
ROOT = pathlib.Path(__file__).resolve().parent
PUBLIC = ROOT / "dashboard" / "public"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

TODAY = dt.date.today().isoformat()
UTCNOW = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

UNIVERSE_CSV = ROOT / "universe_in_scope.csv"
SECTOR_CATALOG = ROOT / "sector_catalog.csv"

# fichiers de sortie (dans / et /dashboard/public pour compat)
OUT_FILES = [
    "candidates_all_ranked.csv",
    "confirmed_STRONGBUY.csv",
    "anticipative_pre_signals.csv",
    "event_driven_signals.csv",
    "signals_history.csv",
]

# limites API raisonnables (tradingview/finnhub)
TV_MAX_PER_RUN = 999999   # on vise tout l'univers; baisse si tu veux throttle
FH_MAX_PER_RUN = 999999

# ---------- utils cache ----------
def load_json(path: pathlib.Path, default=None):
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: pathlib.Path, data):
    try:
        tmp = path.with_suffix(path.suffix + f".tmp{uuid.uuid4().hex}")
        if path.suffix == ".gz":
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
        tmp.replace(path)
    except Exception as e:
        print(f"[WARN] failed save_json {path}: {e}", file=sys.stderr)

# caches
CACHE_YF_FAST = CACHE_DIR / "yf_fastinfo.json"
CACHE_YF_FAST_DATA = load_json(CACHE_YF_FAST, default={})

CACHE_DV20 = CACHE_DIR / "dv20_cache.json"
CACHE_DV20_DATA = load_json(CACHE_DV20, default={})

CACHE_FH = CACHE_DIR / "finnhub_profile.json"
CACHE_FH_DATA = load_json(CACHE_FH, default={})

# ---------- helpers ----------
def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def sleep_jitter(min_s=0.3, max_s=0.8):
    time.sleep(random.uniform(min_s, max_s))

# ---------- chargement univers & secteurs ----------
def load_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        print(f"[ERROR] {UNIVERSE_CSV} manquant", file=sys.stderr)
        sys.exit(1)
    uni = pd.read_csv(UNIVERSE_CSV)
    # colonne attendue: ticker_yf
    col = None
    for c in ["ticker_yf", "ticker", "Symbol", "symbol", "Ticker"]:
        if c in uni.columns:
            col = c
            break
    if col != "ticker_yf":
        uni = uni.rename(columns={col: "ticker_yf"})
    uni["ticker_yf"] = uni["ticker_yf"].astype(str).str.strip().str.upper()
    uni = uni.dropna(subset=["ticker_yf"]).drop_duplicates("ticker_yf").reset_index(drop=True)
    print(f"[UNI] in-scope: {len(uni)}")
    return uni

def load_sector_map() -> Dict[str, Tuple[str, str]]:
    if SECTOR_CATALOG.exists():
        cat = pd.read_csv(SECTOR_CATALOG)
        out = {}
        for _, r in cat.iterrows():
            t = str(r.get("ticker") or r.get("ticker_yf") or "").strip().upper()
            if not t:
                continue
            out[t] = (str(r.get("sector") or ""), str(r.get("industry") or ""))
        print(f"[CAT] sectors mapped: {len(out)}")
        return out
    print("[CAT] sector_catalog.csv absent -> secteurs=Unknown")
    return {}

# ---------- YFinance enrich ----------
def yf_fastinfo(t: str) -> Dict[str, Any]:
    # cache first
    if t in CACHE_YF_FAST_DATA:
        return CACHE_YF_FAST_DATA[t]
    out = {}
    try:
        info = yf.Ticker(t).fast_info
        # certains champs peuvent manquer suivant le ticker
        out = {
            "last_price": safe_float(info.get("last_price") or info.get("regular_market_price")),
            "market_cap": safe_float(info.get("market_cap")),
            "currency": info.get("currency"),
        }
    except Exception:
        out = {}
    CACHE_YF_FAST_DATA[t] = out
    return out

def yf_price_fallback(t: str) -> Optional[float]:
    try:
        # 5 derniers jours, on prend le dernier close non-NaN
        h = yf.download(t, period="5d", interval="1d", progress=False, threads=True)
        if isinstance(h, pd.DataFrame) and not h.empty:
            if "Adj Close" in h.columns:
                v = h["Adj Close"].dropna().iloc[-1]
            else:
                v = h["Close"].dropna().iloc[-1]
            return safe_float(v)
    except Exception:
        pass
    return None

def dv20_dollar_average(t: str) -> Optional[float]:
    # cache first
    item = CACHE_DV20_DATA.get(t)
    if item and item.get("date") == TODAY:
        return safe_float(item.get("dv20"))
    try:
        h = yf.download(t, period="2mo", interval="1d", progress=False, threads=True)
        if isinstance(h, pd.DataFrame) and not h.empty:
            close = (h.get("Adj Close") or h.get("Close")).astype(float)
            vol = h.get("Volume").astype(float)
            df = pd.DataFrame({"close": close, "vol": vol}).dropna()
            if len(df) >= 10:
                dollar = df["close"] * df["vol"]
                dv20 = float(dollar.tail(20).mean())
                CACHE_DV20_DATA[t] = {"date": TODAY, "dv20": dv20}
                return dv20
    except Exception:
        pass
    return None

# ---------- TradingView ----------
def tv_reco_for(t: str) -> Tuple[Optional[str], Optional[float]]:
    """
    renvoie (tv_reco_label, tv_score_norm)
    tv_score_norm approx dans [0..1] (STRONG_SELL -> 0, STRONG_BUY -> 1)
    """
    label = None
    score = None
    # heuristique échange – on teste NASDAQ puis NYSE puis NYSEARCA
    exchanges = ["NASDAQ", "NYSE", "NYSEARCA"]
    for ex in exchanges:
        try:
            h = TA_Handler(
                symbol=t,
                screener="america",
                exchange=ex,
                interval=Interval.INTERVAL_1_DAY,
            )
            s = h.get_analysis().summary  # dict {'RECOMMENDATION': 'BUY', 'BUY': xx, ...}
            label = str(s.get("RECOMMENDATION") or "").upper()
            # convertit sur un axe 0..1 : SELL=0.25, NEUTRAL=0.5, BUY=0.75, STRONG_BUY=1.0
            table = {
                "STRONG_SELL": 0.0,
                "SELL": 0.25,
                "NEUTRAL": 0.5,
                "BUY": 0.75,
                "STRONG_BUY": 1.0,
            }
            score = table.get(label, 0.5)
            return label, score
        except Exception:
            sleep_jitter(0.1, 0.2)
            continue
    return None, None

# ---------- Finnhub (analystes) ----------
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()

def finnhub_analyst_for(t: str) -> Tuple[Optional[str], Optional[int]]:
    """
    bucket: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
    votes: nombre total (approx) sur la fenêtre renvoyée par Finnhub
    """
    if not FINNHUB_KEY:
        return None, None

    if t in CACHE_FH_DATA and CACHE_FH_DATA[t].get("date") == TODAY:
        d = CACHE_FH_DATA[t]
        return d.get("bucket"), d.get("votes")

    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={t}&token={FINNHUB_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None, None
        arr = r.json() or []
        # Finnhub renvoie une série mensuelle, on agrège simple: moyenne des dernières 6 entrées
        if not arr:
            return None, None
        df = pd.DataFrame(arr)
        cols = ["strongBuy", "buy", "hold", "sell", "strongSell"]
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        last = df.head(6)[cols].sum()
        votes = int(last.sum())
        # bucket par majorité simple
        order = [
            ("STRONG_BUY", last["strongBuy"]),
            ("BUY", last["buy"]),
            ("HOLD", last["hold"]),
            ("SELL", last["sell"]),
            ("STRONG_SELL", last["strongSell"]),
        ]
        bucket = max(order, key=lambda x: x[1])[0] if votes > 0 else None
        CACHE_FH_DATA[t] = {"date": TODAY, "bucket": bucket, "votes": votes}
        return bucket, votes
    except Exception:
        return None, None

# ---------- petit signal technique "p_tech" ----------
def compute_ptech(row) -> bool:
    # très simple: close > SMA20 & SMA20 > SMA50
    t = row["ticker_yf"]
    try:
        h = yf.download(t, period="3mo", interval="1d", progress=False, threads=True)
        if isinstance(h, pd.DataFrame) and not h.empty:
            c = (h.get("Adj Close") or h.get("Close")).dropna()
            if len(c) < 50:
                return False
            sma20 = c.rolling(20).mean()
            sma50 = c.rolling(50).mean()
            last_close = c.iloc[-1]
            return bool(last_close > sma20.iloc[-1] and sma20.iloc[-1] > sma50.iloc[-1])
    except Exception:
        pass
    return False

# ---------- pipeline principal ----------
def main():
    print("[STEP] mix_ab_screen_indices starting…")
    uni = load_universe()
    sector_map = load_sector_map()

    # base dataframe
    base = pd.DataFrame({"ticker_yf": uni["ticker_yf"].values})
    base["ticker_yf"] = base["ticker_yf"].astype(str).str.upper()

    # --- YF fastinfo + fallbacks (prix & mcap) ---
    prices, mcaps, currency = {}, {}, {}
    for t in base["ticker_yf"]:
        fi = yf_fastinfo(t)
        p = safe_float(fi.get("last_price"))
        if not p:
            p = yf_price_fallback(t)
        prices[t] = p
        mcaps[t] = safe_float(fi.get("market_cap"))
        currency[t] = fi.get("currency")
        sleep_jitter(0.02, 0.05)

    base["price"] = base["ticker_yf"].map(prices)
    base["last"] = base["price"]
    base["mcap_usd_final"] = base["ticker_yf"].map(mcaps)

    # --- average dollar volume 20j ---
    dv = {}
    for t in base["ticker_yf"]:
        dv[t] = dv20_dollar_average(t)
        sleep_jitter(0.02, 0.05)
    base["avg_dollar_vol"] = base["ticker_yf"].map(dv)

    # --- TradingView reco (pour tous, avec cache implicite via TA_Handler rate-limit friendly) ---
    tv_label, tv_score = {}, {}
    for i, t in enumerate(base["ticker_yf"], 1):
        if i > TV_MAX_PER_RUN:
            tv_label[t], tv_score[t] = None, None
            continue
        lab, sc = tv_reco_for(t)
        tv_label[t], tv_score[t] = lab, sc
        if i % 25 == 0:
            sleep_jitter(0.5, 1.0)
        else:
            sleep_jitter(0.05, 0.15)

    base["tv_reco"] = base["ticker_yf"].map(tv_label)
    base["tv_score"] = base["ticker_yf"].map(tv_score)

    # --- Analystes (Finnhub) ---
    an_bucket, an_votes = {}, {}
    for i, t in enumerate(base["ticker_yf"], 1):
        if i > FH_MAX_PER_RUN:
            an_bucket[t], an_votes[t] = None, None
            continue
        b, v = finnhub_analyst_for(t)
        an_bucket[t], an_votes[t] = b, v
        if FINNHUB_KEY:
            if i % 55 == 0:
                sleep_jitter(1.0, 1.5)
            else:
                sleep_jitter(0.05, 0.15)

    base["analyst_bucket"] = base["ticker_yf"].map(an_bucket)
    base["analyst_votes"] = base["ticker_yf"].map(an_votes)

    # --- Secteurs/industries ---
    base["sector"] = base["ticker_yf"].map(lambda x: sector_map.get(x, ("Unknown", "Unknown"))[0])
    base["industry"] = base["ticker_yf"].map(lambda x: sector_map.get(x, ("Unknown", "Unknown"))[1])

    # --- Pillars ---
    # p_tech: calcul léger; pour accélérer tu peux vectoriser/mettre en cache si besoin
    print("[TECH] computing p_tech (SMA20>SMA50 & close>SMA20)…")
    base["p_tech"] = False
    for idx, row in base.iterrows():
        base.at[idx, "p_tech"] = compute_ptech(row)

    def good(label: Optional[str]) -> bool:
        return label in ("BUY", "STRONG_BUY")

    base["p_tv"] = base["tv_reco"].map(good).fillna(False)
    base["p_an"] = base["analyst_bucket"].map(good).fillna(False)

    # nombre de piliers
    base["pillars_met"] = base[["p_tech", "p_tv", "p_an"]].sum(axis=1).astype(int)

    # votes simplifiés pour le front
    def votes_bin(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        try:
            v = int(v)
        except Exception:
            return ""
        if v >= 20:
            return "20+"
        if v >= 10:
            return "10+"
        if v > 0:
            return str(v)
        return ""
    base["votes_bin"] = base["analyst_votes"].map(votes_bin)

    # score de rang (pondère techno + tv + analyst)
    base["rank_score"] = (
        base["p_tech"].astype(float) * 0.34
        + base["p_tv"].astype(float) * 0.33
        + base["p_an"].astype(float) * 0.33
    )
    # si tv_score est connu, le mélanger à p_tv pour raffiner
    base["rank_score"] = base["rank_score"] * 0.6 + base["tv_score"].fillna(0.5) * 0.4

    # --- Buckets ---
    # Confirmed: 3/3
    # Pre-signal: 1 ou 2 piliers
    # Event-driven: reco TV STRONG_BUY ou Analyst STRONG_BUY (même si pas d'autres piliers)
    def compute_bucket(r):
        if r["pillars_met"] >= 3:
            return "confirmed"
        if r.get("tv_reco") == "STRONG_BUY" or r.get("analyst_bucket") == "STRONG_BUY":
            return "event"
        if r["pillars_met"] >= 1:
            return "pre_signal"
        return "event"  # fourre-tout si rien (mais visible)
    base["bucket"] = base.apply(compute_bucket, axis=1)

    # --- Aliases pour compat front ---
    base["ticker"] = base["ticker_yf"]
    base["mcap"] = base["mcap_usd_final"]
    base["tech"] = base["p_tech"]
    base["tv"] = base["tv_reco"]
    base["analyst"] = base["analyst_bucket"]
    base["votes"] = base["votes_bin"]

    # Normalisation/NA
    for c in ["price", "last", "mcap_usd_final", "avg_dollar_vol"]:
        base[c] = base[c].where(pd.notnull(base[c]), None)
    # S'il manque encore des prix, on met 0 (le front sait l'afficher), mais on évite les NaN
    base["price"] = base["price"].fillna(0.0)
    base["last"] = base["last"].fillna(base["price"])

    # --- Export global trié par score ---
    base_sorted = base.sort_values(["rank_score", "avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)

    # split
    df_confirmed = base_sorted[base_sorted["bucket"] == "confirmed"].copy()
    df_pre = base_sorted[base_sorted["bucket"] == "pre_signal"].copy()
    df_event = base_sorted[base_sorted["bucket"] == "event"].copy()

    # signatures colonnes front (on laisse toutes, ce n’est pas gênant)
    cols_export = [
        "ticker_yf","ticker_tv","price","last","mcap_usd_final","mcap",
        "avg_dollar_vol","tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry","p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","bucket",
        # aliases front:
        "ticker","mcap","tech","tv","analyst","votes",
    ]
    # ticker_tv = même chose par défaut
    base_sorted["ticker_tv"] = base_sorted.get("ticker_tv", base_sorted["ticker_yf"])

    # Sauvegardes (root + public)
    def save_both(name: str, df: pd.DataFrame):
        for outdir in [ROOT, PUBLIC]:
            out = outdir / name
            df.to_csv(out, index=False)
        print(f"[SAVE] {name} | rows={len(df)} | cols={len(df.columns)}")

    save_both("candidates_all_ranked.csv", base_sorted[cols_export])

    save_both("confirmed_STRONGBUY.csv", df_confirmed[cols_export])
    save_both("anticipative_pre_signals.csv", df_pre[cols_export])
    save_both("event_driven_signals.csv", df_event[cols_export])

    # Historique des signaux du jour (pour la timeline/heatmap)
    hist_cols = ["date","ticker_yf","sector","bucket","tv_reco","analyst_bucket"]
    today_rows = []
    for _, r in base_sorted.iterrows():
        today_rows.append({
            "date": TODAY,
            "ticker_yf": r["ticker_yf"],
            "sector": r.get("sector","Unknown"),
            "bucket": r.get("bucket",""),
            "tv_reco": r.get("tv_reco",""),
            "analyst_bucket": r.get("analyst_bucket",""),
        })
    hist_today = pd.DataFrame(today_rows, columns=hist_cols)

    hist_path_root = ROOT / "signals_history.csv"
    hist_path_pub  = PUBLIC / "signals_history.csv"

    if hist_path_root.exists():
        old = pd.read_csv(hist_path_root)
        # on garde max 100 dernières lignes pour ne pas gonfler
        hist = pd.concat([old, hist_today], ignore_index=True).tail(100)
    else:
        hist = hist_today

    hist.to_csv(hist_path_root, index=False)
    hist.to_csv(hist_path_pub, index=False)
    print(f"[SAVE] signals_history.csv | rows={len(hist)}")

    # flush caches
    save_json(CACHE_YF_FAST, CACHE_YF_FAST_DATA)
    save_json(CACHE_DV20, CACHE_DV20_DATA)
    save_json(CACHE_FH, CACHE_FH_DATA)

    # petite synthèse couverture
    coverage = {
        "universe": int(len(base_sorted)),
        "tv_reco_filled": int(base_sorted["tv_reco"].notna().sum()),
        "finnhub_analyst": int(base_sorted["analyst_bucket"].notna().sum()),
        "mcap_final_nonnull": int(base_sorted["mcap_usd_final"].notna().sum()),
        "price_nonnull": int((base_sorted["price"]>0).sum()),
        "confirmed_count": int(len(df_confirmed)),
        "pre_count": int(len(df_pre)),
        "event_count": int(len(df_event)),
        "unique_total": int(len(hist["ticker_yf"].unique())),
    }
    print(f"[COVERAGE] {coverage}")

    print("[DONE] mix_ab_screen_indices finished.")

if __name__ == "__main__":
    main()
