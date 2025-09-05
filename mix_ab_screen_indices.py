# mix_ab_screen_indices.py
# Build a compact, fully-enriched screening set and publish rich CSVs for the dashboard.

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf

# --------- Paths / IO ---------
PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

def _save(df: pd.DataFrame, name: str, also_public: bool = True):
    df.to_csv(name, index=False)
    if also_public:
        (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

def _read_csv_required(p: str) -> pd.DataFrame:
    if not Path(p).exists():
        print(f"❌ Missing required file: {p}")
        sys.exit(2)
    return pd.read_csv(p)

# --------- Config ---------
TOP_N = int(os.getenv("SCREEN_TOPN", "120"))          # top par liquidité à enrichir à fond
TV_BATCH = 80                                         # batch size pour TradingView scanner
REQ_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.25

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "")

# --------- Helpers ---------
def _yf_symbol_to_tv(sym: str) -> str:
    # BRK-B -> BRK.B  ;  XYZ -> XYZ
    return (sym or "").replace("-", ".")

def _safe_num(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except Exception:
        return None

def _label_from_tv_score(score: float | None) -> str | None:
    # TV "RECOMMENDATION" approx: values ~ [-1..+1] selon API; on remappe
    # on garde "STRONG_BUY"/"BUY"/"NEUTRAL"/"SELL"/"STRONG_SELL"
    if score is None:
        return None
    if score >= 0.5:  return "STRONG_BUY"
    if score >= 0.2:  return "BUY"
    if score <= -0.5: return "STRONG_SELL"
    if score <= -0.2: return "SELL"
    return "NEUTRAL"

def _bucket_from_analyst(label: str | None) -> str | None:
    if not label: return None
    u = label.upper().replace(" ", "_")
    if u in {"BUY","STRONG_BUY","OUTPERFORM","OVERWEIGHT"}: return "BUY"
    if u in {"SELL","STRONG_SELL","UNDERPERFORM","UNDERWEIGHT"}: return "SELL"
    if u in {"HOLD","NEUTRAL"}: return "HOLD"
    return "HOLD"

# --------- TradingView batch scan (non-officiel, robuste) ---------
def tv_scan_batch(symbols_tv: List[str]) -> Dict[str, Dict]:
    """
    Retourne {symbol_tv: {"tv_score": float, "tv_reco": str}}.
    """
    out: Dict[str, Dict] = {}
    if not symbols_tv:
        return out

    url = "https://scanner.tradingview.com/america/scan"
    # On demande uniquement l'agrégat de recommandation
    payload = {
        "symbols": {
            "tickers": [f"NASDAQ:{s}" for s in symbols_tv] + [f"NYSE:{s}" for s in symbols_tv],
            "query": {"types": []}
        },
        "columns": ["Recommend.All"]
    }

    try:
        r = requests.post(url, json=payload, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            print(f"[TV] HTTP {r.status_code} -> skip batch")
            return out
        data = r.json()
        # la réponse list "data" contient entries avec "s" (symbol) et "d" (values)
        for row in data.get("data", []):
            sym = row.get("s", "")
            # sym format "NASDAQ:XYZ" → garde XYZ (on se base sur notre liste)
            base = sym.split(":")[-1]
            vals = row.get("d", [])
            tv_score = _safe_num(vals[0]) if vals else None
            tv_reco = _label_from_tv_score(tv_score)
            # on ne sait pas si cela correspond NASDAQ:XYZ ou NYSE:XYZ ; on set si pas déjà présent
            if base and base not in out:
                out[base] = {"tv_score": tv_score, "tv_reco": tv_reco}
        print(f"[TV] filled {len(out)}/{len(symbols_tv)}")
    except Exception as e:
        print(f"[TV] exception: {e}")
    return out

# --------- Finnhub (analystes) ---------
def finnhub_analyst(symbol_yf: str) -> Tuple[str | None, int | None]:
    """
    Retourne (analyst_bucket, analyst_votes) depuis les recos Finnhub (dernier mois).
    """
    if not FINNHUB_KEY:
        return (None, None)
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol_yf}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            return (None, None)
        arr = r.json()
        if not isinstance(arr, list) or not arr:
            return (None, None)
        # on prend la plus récente (première)
        rec = arr[0]
        # votes: somme des catégories
        votes = 0
        for k in ("strongBuy","buy","hold","sell","strongSell"):
            v = rec.get(k)
            if isinstance(v, int):
                votes += v
        # label simple: si (strongBuy+buy) >> (sell) -> BUY ; si inverse -> SELL ; sinon HOLD
        sb = (rec.get("strongBuy") or 0) + (rec.get("buy") or 0)
        ss = (rec.get("strongSell") or 0) + (rec.get("sell") or 0)
        if sb >= ss + 3:
            bucket = "BUY"
        elif ss >= sb + 3:
            bucket = "SELL"
        else:
            bucket = "HOLD"
        return (bucket, votes or None)
    except Exception:
        return (None, None)

# --------- Alpha Vantage OVERVIEW (fallback secteur/industrie) ---------
def alphav_overview(symbol_yf: str) -> Tuple[str | None, str | None]:
    if not ALPHAV_KEY:
        return (None, None)
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function":"OVERVIEW","symbol":symbol_yf,"apikey":ALPHAV_KEY}
        r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            return (None, None)
        j = r.json()
        sec = j.get("Sector") or None
        ind = j.get("Industry") or None
        return (sec, ind)
    except Exception:
        return (None, None)

# --------- Yahoo price (léger) ---------
def yf_price_fast(symbol_yf: str) -> float | None:
    try:
        y = yf.Ticker(symbol_yf)
        fi = getattr(y, "fast_info", None) or {}
        p = fi.get("last_price")
        return float(p) if p is not None else None
    except Exception:
        return None

# --------- Main pipeline ---------
def main():
    uni = _read_csv_required("universe_in_scope.csv")
    print(f"[UNI] in-scope: {len(uni)}")

    # mapping secteurs
    cat = _read_csv_required("sector_catalog.csv")  # contient au moins ticker,sector,industry
    cat_cols = [c.lower() for c in cat.columns]
    # normalise colonnes
    def pickcol(df, names):
        for n in names:
            if n in df.columns: return n
            # tolère case insensitive
            for c in df.columns:
                if c.lower() == n.lower(): return c
        return None
    tcol = pickcol(cat, ["ticker","ticker_yf","symbol"])
    scol = pickcol(cat, ["sector"])
    icol = pickcol(cat, ["industry","gics_sub_industry","sub_industry"])

    cat = cat.rename(columns={tcol:"ticker_yf", scol:"sector", icol:"industry"})[["ticker_yf","sector","industry"]].drop_duplicates("ticker_yf")

    # join mapping sur univers
    base = uni.merge(cat, on="ticker_yf", how="left")
    base["sector"] = base["sector"].fillna("Unknown")
    base["industry"] = base["industry"].fillna("Unknown")

    # tri par liquidité (avg_dollar_vol desc) puis mcap asc (pour départager)
    base["avg_dollar_vol"] = pd.to_numeric(base.get("avg_dollar_vol"), errors="coerce")
    base["mcap_usd"] = pd.to_numeric(base.get("mcap_usd"), errors="coerce")
    base = base.sort_values(["avg_dollar_vol","mcap_usd"], ascending=[False, True])
    cand = base.head(TOP_N).reset_index(drop=True)
    print(f"[PICK] top {len(cand)} by liquidity for deep enrichment")

    # ---------- Enrichissement cascade ----------
    # prix (Yahoo)
    prices = []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        prices.append(yf_price_fast(sym))
        if i % 50 == 0:
            print(f"[YF] price {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS/2)
    cand["price"] = prices

    # TradingView en batch
    syms_tv = [ _yf_symbol_to_tv(s) for s in cand["ticker_yf"] ]
    tv_out: Dict[str, Dict] = {}
    for i in range(0, len(syms_tv), TV_BATCH):
        chunk = [s for s in syms_tv[i:i+TV_BATCH] if s]
        if not chunk: 
            continue
        part = tv_scan_batch(chunk)
        tv_out.update(part)
        time.sleep(SLEEP_BETWEEN_CALLS)

    cand["tv_score"] = [ tv_out.get(_yf_symbol_to_tv(s),{}).get("tv_score") for s in cand["ticker_yf"] ]
    cand["tv_reco"]  = [ tv_out.get(_yf_symbol_to_tv(s),{}).get("tv_reco")  for s in cand["ticker_yf"] ]

    # Finnhub (analystes)
    an_bucket, an_votes = [], []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        b, v = finnhub_analyst(sym)
        an_bucket.append(b)
        an_votes.append(v)
        if i % 50 == 0:
            print(f"[FINNHUB] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    cand["analyst_bucket"] = an_bucket
    cand["analyst_votes"]  = an_votes

    # Fallback Alpha Vantage pour sector/industry si Unknown
    if ALPHAV_KEY:
        miss = cand["sector"].eq("Unknown").sum()
        if miss:
            print(f"[AV] trying to fill sector/industry for {miss} unknown…")
        sec_f, ind_f = [], []
        for i, row in cand.iterrows():
            if row["sector"] != "Unknown":
                sec_f.append(None); ind_f.append(None); continue
            sec, ind = alphav_overview(row["ticker_yf"])
            sec_f.append(sec); ind_f.append(ind)
            time.sleep(SLEEP_BETWEEN_CALLS)
        # uniquement si non-null -> remplace
        for i, (sec, ind) in enumerate(zip(sec_f, ind_f)):
            if pd.notna(sec) and isinstance(sec, str) and sec.strip():
                cand.loc[i, "sector"] = sec
            if pd.notna(ind) and isinstance(ind, str) and ind.strip():
                cand.loc[i, "industry"] = ind

    # ---------- Pillars & score ----------
    # Tech/TV pillar
    def is_strong_tv(v):
        return str(v or "").upper() == "STRONG_BUY"
    # Analyst pillar
    def is_bull_an(v):
        return _bucket_from_analyst(v) == "BUY"

    cand["p_tv"] = cand["tv_reco"].map(is_strong_tv)
    cand["p_an"] = cand["analyst_bucket"].map(is_bull_an)
    # (optionnel) p_tech_local si tu ajoutes plus tard → pour l’instant = p_tv
    cand["p_tech"] = cand["p_tv"]

    cand["pillars_met"] = cand[["p_tech","p_tv","p_an"]].fillna(False).sum(axis=1)
    # votes_bin (pour résumé)
    votes = pd.to_numeric(cand["analyst_votes"], errors="coerce")
    cand["votes_bin"] = pd.cut(votes.fillna(-1), bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])

    # score agrégé simple : tv_score pondéré + bonus pour pillars & votes
    s_tv = cand["tv_score"].fillna(0)
    s_p  = cand["pillars_met"].fillna(0)
    s_v  = votes.fillna(0).clip(lower=0, upper=30) / 30.0
    cand["rank_score"] = (s_tv * 0.6) + (s_p * 0.3) + (s_v * 0.1)

    # ---------- Buckets ----------
    # confirmed: STRONG_BUY TV + analyst BUY (ou >= 12 votes “bull” de manière large)
    cond_confirmed = (cand["tv_reco"].str.upper().eq("STRONG_BUY")) & (cand["analyst_bucket"].map(_bucket_from_analyst).eq("BUY"))
    # pre_signal: TV fort (BUY ou STRONG_BUY) mais analyst HOLD/low votes
    cond_pre = (cand["tv_reco"].str.upper().isin(["STRONG_BUY","BUY"])) & (~cond_confirmed)

    cand["bucket"] = "other"
    cand.loc[cond_pre, "bucket"] = "pre_signal"
    cand.loc[cond_confirmed, "bucket"] = "confirmed"

    # tri final pour export
    export_cols = [
        "ticker_yf","ticker_tv","price","mcap_usd","avg_dollar_vol",
        "tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry",
        "p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","bucket"
    ]
    # ticker_tv peut venir de universe_in_scope.csv
    if "ticker_tv" not in cand.columns:
        cand["ticker_tv"] = cand["ticker_yf"].map(_yf_symbol_to_tv)

    cand = cand[export_cols].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)

    # ---------- Exports ----------
    _save(cand, "candidates_all_ranked.csv")

    confirmed = cand[cand["bucket"] == "confirmed"].copy()
    _save(confirmed, "confirmed_STRONGBUY.csv")

    pre = cand[cand["bucket"] == "pre_signal"].copy()
    _save(pre, "anticipative_pre_signals.csv")

    # events placeholder (à brancher sur tes détections “news/events”)
    events = cand.head(0).copy()
    _save(events, "event_driven_signals.csv")

    # ---------- Couverture / Diagnostics ----------
    cov = {
        "topN": int(len(cand)),
        "tv_reco_filled": int(cand["tv_reco"].notna().sum()),
        "finnhub_filled": int(cand["analyst_bucket"].notna().sum()),
        "sector_known": int((cand["sector"]!="Unknown").sum()),
        "industry_known": int((cand["industry"]!="Unknown").sum()),
        "confirmed_count": int(len(confirmed)),
        "pre_count": int(len(pre)),
    }
    print("[COVERAGE]", cov)

if __name__ == "__main__":
    main()
