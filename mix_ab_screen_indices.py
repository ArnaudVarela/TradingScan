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
        for row in data.get("data", []):
            sym = row.get("s", "")
            base = sym.split(":")[-1]
            vals = row.get("d", [])
            tv_score = _safe_num(vals[0]) if vals else None
            tv_reco = _label_from_tv_score(tv_score)
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
        rec = arr[0]
        votes = 0
        for k in ("strongBuy","buy","hold","sell","strongSell"):
            v = rec.get(k)
            if isinstance(v, int):
                votes += v
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
    # 0) Univers filtré
    uni = _read_csv_required("universe_in_scope.csv")
    print(f"[UNI] in-scope: {len(uni)}")

    # 1) Mapping secteurs depuis sector_catalog (robuste aux noms de colonnes)
    cat = _read_csv_required("sector_catalog.csv")
    def pickcol(df, names):
        for n in names:
            if n in df.columns:
                return n
            for c in df.columns:
                if c.lower() == n.lower():
                    return c
        return None

    tcol = pickcol(cat, ["ticker", "ticker_yf", "symbol"])
    scol = pickcol(cat, ["sector", "gics sector", "gics_sector"])
    icol = pickcol(cat, ["industry", "gics sub-industry", "gics_sub_industry", "sub_industry"])

    if tcol is None:
        print("[CAT] WARNING: no ticker column found in sector_catalog.csv -> sector/industry fallback to Unknown")
        cat = pd.DataFrame(columns=["ticker_yf", "sector", "industry"])
    else:
        rename_map = {tcol: "ticker_yf"}
        if scol is not None: rename_map[scol] = "sector"
        if icol is not None: rename_map[icol] = "industry"
        cat = cat.rename(columns=rename_map)
        if "sector" not in cat.columns: cat["sector"] = "Unknown"
        if "industry" not in cat.columns: cat["industry"] = "Unknown"
        cat = cat[["ticker_yf", "sector", "industry"]].drop_duplicates("ticker_yf")

    base = uni.merge(cat, on="ticker_yf", how="left")
    if "sector" not in base.columns: base["sector"] = "Unknown"
    if "industry" not in base.columns: base["industry"] = "Unknown"
    base["sector"] = base["sector"].fillna("Unknown")
    base["industry"] = base["industry"].fillna("Unknown")

    # 2) Sélection TOP_N par liquidité
    base["avg_dollar_vol"] = pd.to_numeric(base.get("avg_dollar_vol"), errors="coerce")
    cand = (base.sort_values(["avg_dollar_vol","mcap_usd"], ascending=[False, True])
                 .head(TOP_N)
                 .reset_index(drop=True))
    # ticker_tv peut venir d'universe; sinon on le construit
    if "ticker_tv" not in cand.columns:
        cand["ticker_tv"] = cand["ticker_yf"].map(_yf_symbol_to_tv)

    # 3) Enrichissement cascade
    # 3.1 Prix (Yahoo)
    prices = []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        prices.append(yf_price_fast(sym))
        if i % 50 == 0:
            print(f"[YF] price {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS/2)
    cand["price"] = prices

    # 3.2 TradingView (batch)
    syms_tv = [_yf_symbol_to_tv(s) for s in cand["ticker_yf"]]
    tv_out: Dict[str, Dict] = {}
    for i in range(0, len(syms_tv), TV_BATCH):
        chunk = [s for s in syms_tv[i:i+TV_BATCH] if s]
        if not chunk: 
            continue
        part = tv_scan_batch(chunk)
        tv_out.update(part)
        time.sleep(SLEEP_BETWEEN_CALLS)

    cand["tv_score"] = [tv_out.get(_yf_symbol_to_tv(s), {}).get("tv_score") for s in cand["ticker_yf"]]
    cand["tv_reco"]  = [tv_out.get(_yf_symbol_to_tv(s), {}).get("tv_reco")  for s in cand["ticker_yf"]]

    # 3.3 Finnhub (analystes)
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

    # 3.4 Fallback Alpha Vantage pour sector/industry si Unknown
    if ALPHAV_KEY:
        miss = int(cand["sector"].eq("Unknown").sum())
        if miss:
            print(f"[AV] trying to fill sector/industry for {miss} unknown…")
        sec_f, ind_f = [], []
        for _, row in cand.iterrows():
            if row["sector"] != "Unknown":
                sec_f.append(None); ind_f.append(None); continue
            sec, ind = alphav_overview(row["ticker_yf"])
            sec_f.append(sec); ind_f.append(ind)
            time.sleep(SLEEP_BETWEEN_CALLS)
        for i, (sec, ind) in enumerate(zip(sec_f, ind_f)):
            if isinstance(sec, str) and sec.strip():
                cand.loc[i, "sector"] = sec
            if isinstance(ind, str) and ind.strip():
                cand.loc[i, "industry"] = ind

    # 4) Pillars & score
    def is_strong_tv(v):
        return str(v or "").upper() == "STRONG_BUY"
    def is_bull_an(v):
        return _bucket_from_analyst(v) == "BUY"

    cand["p_tv"] = cand["tv_reco"].map(is_strong_tv)
    cand["p_an"] = cand["analyst_bucket"].map(is_bull_an)
    cand["p_tech"] = cand["p_tv"]  # placeholder

    cand["pillars_met"] = cand[["p_tech","p_tv","p_an"]].fillna(False).sum(axis=1)

    votes = pd.to_numeric(cand["analyst_votes"], errors="coerce")
    cand["votes_bin"] = pd.cut(votes.fillna(-1), bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])

    s_tv = pd.to_numeric(cand["tv_score"], errors="coerce").fillna(0)
    s_p  = pd.to_numeric(cand["pillars_met"], errors="coerce").fillna(0)
    s_v  = votes.fillna(0).clip(lower=0, upper=30) / 30.0
    cand["rank_score"] = (s_tv * 0.6) + (s_p * 0.3) + (s_v * 0.1)

    # 5) Buckets
    tvu = cand["tv_reco"].astype(str).str.upper()
    cond_confirmed = (tvu.eq("STRONG_BUY")) & (cand["analyst_bucket"].map(_bucket_from_analyst).eq("BUY"))
    cond_pre       = tvu.isin(["STRONG_BUY","BUY"]) & (~cond_confirmed)

    cand["bucket"] = "other"
    cand.loc[cond_pre, "bucket"] = "pre_signal"
    cand.loc[cond_confirmed, "bucket"] = "confirmed"

    # 6) Tri & exports
    export_cols = [
        "ticker_yf","ticker_tv","price","mcap_usd","avg_dollar_vol",
        "tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry",
        "p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","bucket"
    ]
    if "ticker_tv" not in cand.columns:
        cand["ticker_tv"] = cand["ticker_yf"].map(_yf_symbol_to_tv)

    cand = cand[export_cols].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)

    _save(cand, "candidates_all_ranked.csv")

    confirmed = cand[cand["bucket"] == "confirmed"].copy()
    _save(confirmed, "confirmed_STRONGBUY.csv")

    pre = cand[cand["bucket"] == "pre_signal"].copy()
    _save(pre, "anticipative_pre_signals.csv")

    events = cand.head(0).copy()
    _save(events, "event_driven_signals.csv")

    # 7) signals_history minimal (pour l’Action/backtest)
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sig_hist = cand[["ticker_yf","sector","bucket","tv_reco","analyst_bucket"]].copy()
    sig_hist.insert(0, "date", today)
    _save(sig_hist, "signals_history.csv")

    # 8) Couverture / Diagnostics
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
