#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix A/B Screener — robuste & tolérant aux échecs
------------------------------------------------
Objectif
  - Lire l'univers (déjà filtré) depuis universe_in_scope.csv
  - Enrichir TOUS les tickers avec: prix, mcap, secteur, industry, avg$vol 20j
  - Puis (si possible) TradingView et avis analystes (YF + Finnhub optionnel)
  - Classer et produire exactement les CSV attendus par le front:
      * dashboard/public/confirmed_STRONGBUY.csv
      * dashboard/public/anticipative_pre_signals.csv
      * dashboard/public/event_driven_signals.csv
      * dashboard/public/candidates_all_ranked.csv
      * dashboard/public/raw_candidates.csv (diag)
      * dashboard/public/debug_all_candidates.csv (diag trié)
      * dashboard/public/signals_history.csv (append jour)
  - Résilient: retries, timeouts, échanges alternatifs TV, skip propre des delistés.

Notes
  - Aucune hard fail si un provider manque. On remplit au max.
  - Les colonnes sont EXACTES pour coller au front actuel.
"""
from __future__ import annotations

import os, sys, time, json, math, textwrap, random
import warnings
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from tradingview_ta import TA_Handler, Interval
    _TV_OK = True
except Exception:
    _TV_OK = False
    class Interval:
        INTERVAL_1_DAY = None

# ===================== CONFIG =====================
PUBLIC_DIR = os.path.join("dashboard", "public")
CACHE_DIR  = "cache"

YF_HISTORY_PERIOD  = "3mo"      # pour avg$vol 20j
YF_HISTORY_INTERVAL= "1d"

TV_ENABLED = True
TV_INTERVAL = Interval.INTERVAL_1_DAY
TV_DELAY_SEC = 0.15
TV_MAX_RETRIES = 2

# Enrichir avec TV pour N tickers (par défaut par $vol décroissant)
# 0 ou None => tenter pour tout le monde (plus lent)
TV_TOPN = 800

# Analyses "analystes"
#  - Par défaut on se base sur yfinance (recommendationMean / numberOfAnalystOpinions)
#  - Optionnel: Finnhub (si FINNHUB_API_KEY dans env) pour backup
FINNHUB_ENABLED = True
FINNHUB_TIMEOUT = 12

# ===================== UTILS FS =====================

def _ensure_dir(path: str):
    d = path if path.endswith(os.sep) else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_csv(df: pd.DataFrame, fname: str, also_public: bool=True):
    df = df.copy()
    df.to_csv(fname, index=False)
    if also_public:
        dst = os.path.join(PUBLIC_DIR, os.path.basename(fname))
        _ensure_dir(dst)
        df.to_csv(dst, index=False)


def load_csv(fname: str) -> pd.DataFrame:
    return pd.read_csv(fname)


# ===================== DOMAIN HELPERS =====================

def yf_norm(s: str) -> str:
    return (s or "").strip().upper().replace(".", "-")


def tv_norm(s: str) -> str:
    return (s or "").strip().upper().replace("-", ".")


def map_exchange_for_tv(exch: str) -> str:
    e = (exch or "").upper()
    if "NASDAQ" in e: return "NASDAQ"
    if "NYSE"   in e: return "NYSE"
    if "ARCA"   in e or "AMEX" in e or "MKT" in e: return "AMEX"
    # défaut US
    return "NASDAQ"


def analyst_bucket_from_mean(x: Optional[float]) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    try:
        x = float(x)
    except Exception:
        return None
    if x < 1.6: return "Strong Buy"
    if x < 2.5: return "Buy"
    if x < 3.5: return "Hold"
    if x < 4.2: return "Sell"
    return "Strong Sell"


# ===================== CACHES =====================

class JsonCache:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def flush(self):
        _ensure_dir(self.path)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f)


# ===================== PROVIDERS =====================

def yf_fetch_basics(ticker: str, hist_cache: JsonCache) -> Tuple[dict, Optional[pd.DataFrame]]:
    """Retourne (basics, hist)
    basics keys: price,last,market_cap,exchange,country,info_mcap
    """
    tk = yf.Ticker(ticker)

    # Prix / mcap via fast_info (rapide)
    basics: Dict[str, Any] = {"price": None, "last": None, "market_cap": None,
                              "exchange": None, "country": None, "info_mcap": None}
    try:
        fi = getattr(tk, "fast_info", None)
        if fi is not None:
            basics["price"] = float(getattr(fi, "last_price", np.nan))
            basics["last"]  = basics["price"]
            basics["market_cap"] = float(getattr(fi, "market_cap", np.nan))
    except Exception:
        pass

    info = {}
    try:
        info = tk.get_info() or {}
    except Exception:
        info = {}

    basics["info_mcap"] = info.get("marketCap")
    basics["exchange"]  = info.get("exchange") or info.get("fullExchangeName")
    basics["country"]   = (info.get("country") or info.get("countryOfCompany") or "").strip()

    # Compléments / fallback prix via history
    hkey = f"hist::{ticker}::{YF_HISTORY_PERIOD}::{YF_HISTORY_INTERVAL}"
    hist = None
    try:
        cached = hist_cache.get(hkey)
        if cached is not None:
            hist = pd.DataFrame.from_records(cached)
        else:
            hist = tk.history(period=YF_HISTORY_PERIOD, interval=YF_HISTORY_INTERVAL,
                              auto_adjust=True, actions=False)
            if not hist.empty:
                hist_cache.set(hkey, hist.reset_index().to_dict(orient="records"))
    except Exception:
        hist = None

    # Normalise prix si fast_info vide
    try:
        if (basics["price"] is None or np.isnan(basics["price"])) and hist is not None and not hist.empty:
            basics["price"] = float(hist["Close"].iloc[-1])
            basics["last"]  = basics["price"]
    except Exception:
        pass

    # Market cap fallback
    if (basics["market_cap"] is None or (isinstance(basics["market_cap"], float) and np.isnan(basics["market_cap"]))) and basics["info_mcap"]:
        try:
            basics["market_cap"] = float(basics["info_mcap"])
        except Exception:
            pass

    return basics, hist


def compute_avg_dollar_vol(hist: Optional[pd.DataFrame]) -> Optional[float]:
    if hist is None or hist.empty or not {"Close","Volume"}.issubset(hist.columns):
        return None
    try:
        px = hist["Close"].astype(float)
        vol= hist["Volume"].astype(float)
        dv = (px * vol).round(6)
        last20 = dv.tail(20)
        if last20.empty: return None
        return float(last20.mean())
    except Exception:
        return None


def tv_fetch_reco(symbol_tv: str, exchange_hint: str) -> Tuple[Optional[str], Optional[float]]:
    if not _TV_OK or not TV_ENABLED:
        return None, None

    def _try(sym: str, ex: Optional[str]):
        try:
            h = TA_Handler(symbol=sym, screener="america", exchange=ex or "NASDAQ", interval=TV_INTERVAL)
            s = h.get_analysis().summary
            reco = s.get("RECOMMENDATION")
            score = None
            # convertir summary -> score approx (optional)
            if isinstance(s, dict):
                # Buy/Sell/Neutral counts non dispo directement, on approx
                mapping = {"STRONG_BUY": 0.51, "BUY": 0.49, "NEUTRAL": 0.0, "SELL": -0.49, "STRONG_SELL": -0.51}
                score = mapping.get(reco)
            return reco, score
        except Exception:
            return None, None

    # 1) hint
    reco, score = _try(symbol_tv, exchange_hint)
    if reco: return reco, score

    # 2) autres places US
    for ex in ("NASDAQ", "NYSE", "AMEX"):
        reco, score = _try(symbol_tv, ex)
        if reco: return reco, score
        time.sleep(TV_DELAY_SEC)

    return None, None


def finnhub_fetch(symbol: str) -> Tuple[Optional[str], Optional[int]]:
    if not FINNHUB_ENABLED or not os.environ.get("FINNHUB_API_KEY"):
        return None, None
    import urllib.request, urllib.error
    api = os.environ.get("FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={api}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=FINNHUB_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
        if isinstance(data, list) and data:
            latest = data[0]
            # Convertir en bucket simple
            buy = int(latest.get("buy") or 0) + int(latest.get("strongBuy") or 0)
            hold= int(latest.get("hold") or 0)
            sell= int(latest.get("sell") or 0) + int(latest.get("strongSell") or 0)
            total = buy+hold+sell
            if total <= 0:
                return None, None
            # règle simple
            if buy/total >= 0.55:
                return "Buy", total
            if sell/total >= 0.55:
                return "Sell", total
            return "Hold", total
    except Exception:
        return None, None
    return None, None


# ===================== RANK & PILLARS =====================

def compute_pillars(row: pd.Series) -> Tuple[bool,bool,bool,int,str]:
    p_tech = bool(row.get("p_tech") or False)
    p_tv   = bool(row.get("p_tv") or False)
    p_an   = bool(row.get("p_an") or False)
    pillars = int(p_tech) + int(p_tv) + int(p_an)
    votes = row.get("analyst_votes")
    if isinstance(votes,(int,float)) and votes >= 20: vb = "20+"
    elif isinstance(votes,(int,float)) and votes >= 10: vb = "10-19"
    elif isinstance(votes,(int,float)) and votes >= 5: vb = "5-9"
    elif isinstance(votes,(int,float)) and votes > 0: vb = "1-4"
    else: vb = "0"
    return p_tech, p_tv, p_an, pillars, vb


def rank_score_row(s: pd.Series) -> float:
    score = 0.0
    # TV
    tv = (s.get("tv_reco") or "").upper()
    if tv == "STRONG_BUY": score += 0.51
    elif tv == "BUY":      score += 0.49
    elif tv == "SELL":     score -= 0.49
    elif tv == "STRONG_SELL": score -= 0.51

    # Tech (bool)
    if bool(s.get("p_tech")): score += 0.35
    # Analyst bucket
    ab = s.get("analyst_bucket")
    if ab == "Strong Buy": score += 0.4
    elif ab == "Buy":      score += 0.2
    elif ab == "Sell":     score -= 0.2
    elif ab == "Strong Sell": score -= 0.4

    # Votes analystes
    av = s.get("analyst_votes")
    if isinstance(av,(int,float)) and av>0:
        score += min(float(av), 20.0) / 100.0  # +0.20 max

    # Capitalisation (favoriser mid)
    mc = s.get("mcap_usd_final") or s.get("mcap")
    try:
        mc = float(mc)
        if mc>0:
            score += max(0.0, 1.2 - math.log10(mc))  # ~[0..1.2]
    except Exception:
        pass
    return float(score)


# ===================== MAIN =====================

def main():
    print("[STEP] mix_ab_screen_indices — start…")

    # ---------- 1) Chargement univers & catalog ----------
    uni_path = "universe_in_scope.csv"
    if not os.path.exists(uni_path):
        raise SystemExit("universe_in_scope.csv introuvable — étape build_universe manquante.")
    uni = load_csv(uni_path)
    # attend soit colonne ticker_yf, sinon unique colonne avec tickers
    col = None
    for c in uni.columns:
        if str(c).lower() in {"ticker", "ticker_yf", "symbol", "ticker_yf"}:
            col = c; break
    if col is None:
        # si 1 seule colonne
        if len(uni.columns)==1:
            col = uni.columns[0]
        else:
            raise SystemExit("universe_in_scope.csv ne contient pas de colonne ticker.")
    uni["ticker_yf"] = uni[col].astype(str).map(yf_norm)
    uni["ticker_tv"] = uni["ticker_yf"].map(tv_norm)
    uni = uni.drop_duplicates("ticker_yf").reset_index(drop=True)
    print(f"[UNI] in-scope: {len(uni)}")

    # Sector catalog
    sector_map: Dict[str, Tuple[str,str]] = {}
    cat_path = "sector_catalog.csv"
    if os.path.exists(cat_path):
        try:
            cat = pd.read_csv(cat_path)
            cat["ticker"] = cat["ticker"].astype(str).str.upper().str.strip()
            for _, r in cat.iterrows():
                sector_map[str(r["ticker"]).upper()] = (
                    str(r.get("sector") or "Unknown"),
                    str(r.get("industry") or "Unknown"),
                )
            print(f"[CAT] loaded sector catalog: {len(sector_map)}")
        except Exception as e:
            print(f"[WARN] sector catalog illisible: {e}")

    def sector_from_catalog(tv_sym: str, yf_sym: str) -> Tuple[str,str]:
        key_tv = (tv_sym or "").upper()
        if key_tv in sector_map:
            return sector_map[key_tv]
        key_yf_as_tv = (yf_sym or "").upper().replace("-", ".")
        return sector_map.get(key_yf_as_tv, ("Unknown","Unknown"))

    # Caches
    _ensure_dir(os.path.join(CACHE_DIR, ""))
    cache_hist   = JsonCache(os.path.join(CACHE_DIR, "ta_daily_cache.json"))
    cache_yfinfo = JsonCache(os.path.join(CACHE_DIR, "yf_fastinfo.json"))  # non utilisé directement mais dispo

    rows: list[Dict[str, Any]] = []

    # ---------- 2) Pré-collect YF basics + avg$vol ----------
    print("[YF] prix/mcap/avg$vol…")
    for i, r in enumerate(uni.itertuples(index=False), 1):
        yf_sym = r.ticker_yf
        tv_sym = r.ticker_tv
        basics, hist = yf_fetch_basics(yf_sym, cache_hist)

        # ignorer les clairement delistés (pas de prix, pas d'histo)
        price = basics.get("price")
        if price is None or (isinstance(price,float) and np.isnan(price)):
            # garder quand même la ligne, mais marquer price=None pour debug
            avg_dv20 = compute_avg_dollar_vol(hist)
        else:
            avg_dv20 = compute_avg_dollar_vol(hist)

        # Exchange hint pour TV
        exch = basics.get("exchange") or ""
        tv_exch = map_exchange_for_tv(exch)

        # Analyst via YF
        tk = yf.Ticker(yf_sym)
        try:
            info = tk.get_info() or {}
        except Exception:
            info = {}
        analyst_mean  = info.get("recommendationMean")
        analyst_votes = info.get("numberOfAnalystOpinions")
        analyst_bucket= analyst_bucket_from_mean(analyst_mean)

        # Sectors
        sector, industry = sector_from_catalog(tv_sym, yf_sym)
        if not sector or sector == "nan": sector = "Unknown"
        if not industry or industry == "nan": industry = "Unknown"

        # p_tech: heuristique simple — prix > sma50 si hist OK
        p_tech = False
        try:
            if hist is not None and not hist.empty and len(hist) >= 50:
                close = hist["Close"].astype(float)
                sma50 = close.rolling(50).mean()
                if float(close.iloc[-1]) > float(sma50.iloc[-1]):
                    p_tech = True
        except Exception:
            p_tech = False

        rows.append({
            "ticker_yf": yf_sym,
            "ticker_tv": tv_sym,
            "price": price,
            "last": price,
            "mcap_usd_final": basics.get("market_cap"),
            "mcap": basics.get("market_cap"),
            "avg_dollar_vol": avg_dv20,
            # placeholders TV/analyst (complétés ensuite)
            "tv_score": None,
            "tv_reco": None,
            "analyst_bucket": analyst_bucket,
            "analyst_votes": analyst_votes,
            "sector": sector,
            "industry": industry,
            "p_tech": bool(p_tech),
            "p_tv": False,
            "p_an": bool(analyst_bucket in {"Buy","Strong Buy"}),
        })

        if i % 200 == 0:
            print(f"  [YF] {i}/{len(uni)}…")

    # ---------- 3) TV & Finnhub (optionnel) ----------
    df = pd.DataFrame(rows)

    # Ordre d'enrichissement: par avg$vol décroissant (tickers les plus liquides d'abord)
    df = df.sort_values(by=["avg_dollar_vol"], ascending=[False]).reset_index(drop=True)
    n_tv = TV_TOPN if TV_TOPN else len(df)

    if TV_ENABLED and _TV_OK:
        print(f"[TV] Enrichissement sur {min(n_tv, len(df))} tickers…")
        for i in range(min(n_tv, len(df))):
            sym_tv = df.at[i, "ticker_tv"]
            exch_hint = map_exchange_for_tv("")  # on n'a pas toujours l'exchange fiable ici
            # petit backoff aléatoire pour limiter les bursts
            time.sleep(TV_DELAY_SEC + random.random()*0.05)
            reco, score = tv_fetch_reco(sym_tv, exch_hint)
            if reco:
                df.at[i, "tv_reco"] = str(reco).upper()
                df.at[i, "p_tv"] = bool(str(reco).upper() in {"STRONG_BUY","BUY"})
            if score is not None:
                df.at[i, "tv_score"] = float(score)

    # Finnhub pour tout le monde mais soft-fail
    if FINNHUB_ENABLED and os.environ.get("FINNHUB_API_KEY"):
        print("[Finnhub] avis analystes (fallback)…")
        for i in range(len(df)):
            if pd.notna(df.at[i, "analyst_bucket"]) and str(df.at[i, "analyst_bucket"]).strip():
                continue  # on a déjà YF
            sym = df.at[i, "ticker_yf"].replace("-", ".")  # Finnhub attend APPL format NASDAQ? Ici on laisse simple
            bkt, votes = finnhub_fetch(sym)
            if bkt:
                # map basique vers nôtre échelle
                if bkt == "Buy": ab = "Buy"
                elif bkt == "Sell": ab = "Sell"
                else: ab = "Hold"
                df.at[i, "analyst_bucket"] = ab
                if pd.isna(df.at[i, "analyst_votes"]) or not df.at[i, "analyst_votes"]:
                    df.at[i, "analyst_votes"] = votes

    # ---------- 4) Finalisation / scoring ----------
    # Remplissages sûrs
    for col in ["price","last","mcap_usd_final","mcap","avg_dollar_vol","tv_score","analyst_votes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # p_an peut évoluer après Finnhub
    df["p_an"] = df["analyst_bucket"].isin(["Buy","Strong Buy"]).fillna(False)

    # Pillars / bins / rank
    bins, pillars = [], []
    for _, r in df.iterrows():
        p_tech, p_tv, p_an, n, vb = compute_pillars(r)
        bins.append(vb)
        pillars.append(n)
    df["pillars_met"] = pillars
    df["votes_bin"]  = bins

    df["rank_score"] = df.apply(rank_score_row, axis=1)

    # ---------- 5) Sauvegardes (diag + front) ----------
    # raw (diag brut)
    raw_cols = [
        "ticker_yf","ticker_tv","price","last","mcap_usd_final","mcap",
        "avg_dollar_vol","tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry","p_tech","p_tv","p_an","pillars_met","votes_bin","rank_score"
    ]
    save_csv(df[raw_cols], "raw_candidates.csv")

    # debug trié
    dbg = df.sort_values(["rank_score","mcap_usd_final"], ascending=[False, True]).reset_index(drop=True)
    save_csv(dbg[raw_cols], "debug_all_candidates.csv")

    # confirmed (strict TV STRONG_BUY + analyst Buy/Strong Buy)
    m_conf = (df["tv_reco"].eq("STRONG_BUY")) & (df["analyst_bucket"].isin(["Buy","Strong Buy"]))
    confirmed = df[m_conf].copy()

    # anticipative (tech +/ou TV positif)
    m_pre = (df["p_tech"].fillna(False)) | (df["tv_reco"].isin(["STRONG_BUY","BUY"]))
    pre = df[m_pre].copy()

    # event-driven (avis analystes connu)
    evt = df[df["analyst_bucket"].notna() & (df["analyst_bucket"].astype(str) != "")].copy()

    # Tri commun
    for d in (confirmed, pre, evt):
        d.sort_values(["rank_score","mcap_usd_final"], ascending=[False, True], inplace=True)

    # Colonnes EXACTES pour le front
    out_cols = raw_cols
    save_csv(confirmed[out_cols], "confirmed_STRONGBUY.csv")
    save_csv(pre[out_cols],        "anticipative_pre_signals.csv")
    save_csv(evt[out_cols],        "event_driven_signals.csv")

    # Mix comparatif
    confirmed2 = confirmed.copy(); confirmed2.insert(0, "bucket", "confirmed")
    pre2       = pre.copy();       pre2.insert(0, "bucket", "pre_signal")
    evt2       = evt.copy();       evt2.insert(0, "bucket", "event")
    all_out = pd.concat([confirmed2, pre2, evt2], ignore_index=True)
    # Historique journalier par ticker
    _append_signals_history(all_out[["bucket","ticker_yf","ticker_tv","sector","industry","price"]])

    # Le front attend parfois une colonne "bucket" dans candidates_all_ranked.csv
    # + exactement out_cols
    save_csv(all_out[["bucket"] + out_cols], "candidates_all_ranked.csv")

    # Stat coverage
    coverage = {
        "universe": int(len(df)),
        "tv_reco_filled": int(df["tv_reco"].notna().sum()),
        "analyst_known": int(df["analyst_bucket"].notna().sum()),
        "price_nonnull": int(df["price"].notna().sum()),
        "confirmed_count": int(len(confirmed)),
        "pre_count": int(len(pre)),
        "event_count": int(len(evt)),
        "unique_total": int(len(all_out["ticker_yf"].unique()))
    }
    print("[COVERAGE]", coverage)

    print(f"[SAVE] confirmed_STRONGBUY.csv | rows={len(confirmed)} cols={len(out_cols)}")
    print(f"[SAVE] anticipative_pre_signals.csv | rows={len(pre)} cols={len(out_cols)}")
    print(f"[SAVE] event_driven_signals.csv | rows={len(evt)} cols={len(out_cols)}")
    print(f"[SAVE] candidates_all_ranked.csv | rows={len(all_out)} cols={len(out_cols)+1}")
    print("[DONE] mix_ab_screen_indices")


def _append_signals_history(df_in: pd.DataFrame):
    """Append (date, bucket, ticker_yf,ticker_tv,sector,industry,price) dans signals_history.csv"""
    SH = "signals_history.csv"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if df_in is None or df_in.empty:
        # créer le fichier s'il n'existe pas
        if not os.path.exists(SH):
            save_csv(pd.DataFrame(columns=["date","ticker_yf","sector","bucket","tv_reco","analyst_bucket"]), SH)
        return

    block = df_in.copy()
    block.insert(0, "date", today)

    if os.path.exists(SH):
        try:
            prev = pd.read_csv(SH)
        except Exception:
            prev = pd.DataFrame(columns=["date","bucket","ticker_yf","ticker_tv","sector","industry","price"])
    else:
        prev = pd.DataFrame(columns=["date","bucket","ticker_yf","ticker_tv","sector","industry","price"])

    prev["__k"]  = prev["date"].astype(str)+"|"+prev["bucket"].astype(str)+"|"+prev["ticker_yf"].astype(str)
    block["__k"] = block["date"].astype(str)+"|"+block["bucket"].astype(str)+"|"+block["ticker_yf"].astype(str)

    out = pd.concat([prev, block], ignore_index=True)
    out = out.drop_duplicates(subset="__k").drop(columns="__k")

    save_csv(out, SH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
