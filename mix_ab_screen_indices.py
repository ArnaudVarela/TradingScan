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
    # TV "RECOMMENDATION" approx: [-1..+1] → libellés
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
    # 1) Univers filtré (déjà <75B + secteurs) construit en amont
    base = _read_csv_required("universe_in_scope.csv")
    if "ticker_yf" not in base.columns:
        raise SystemExit("❌ universe_in_scope.csv must contain 'ticker_yf'")

    # 2) Mapping secteurs/industries
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
        if scol is not None:
            rename_map[scol] = "sector"
        if icol is not None:
            rename_map[icol] = "industry"
        cat = cat.rename(columns=rename_map)
        if "sector" not in cat.columns:
            cat["sector"] = "Unknown"
        if "industry" not in cat.columns:
            cat["industry"] = "Unknown"
        cat = cat[["ticker_yf", "sector", "industry"]].drop_duplicates("ticker_yf")

    # Merge mapping → base
    base = base.merge(cat, on="ticker_yf", how="left")
    if "sector" not in base.columns:
        base["sector"] = "Unknown"
    if "industry" not in base.columns:
        base["industry"] = "Unknown"
    base["sector"] = base["sector"].fillna("Unknown")
    base["industry"] = base["industry"].fillna("Unknown")

    # 3) Sélection Top N par liquidité pour enrichissement “lourd”
    if "avg_dollar_vol" in base.columns:
        base["avg_dollar_vol"] = pd.to_numeric(base["avg_dollar_vol"], errors="coerce")
    else:
        base["avg_dollar_vol"] = None

    base_sorted = base.sort_values("avg_dollar_vol", ascending=False, na_position="last").reset_index(drop=True)
    cand = base_sorted.head(TOP_N).copy()  # à enrichir fort

    print(f"[UNI] in-scope: {len(base)}  |  TopN to enrich: {len(cand)}")

    # 4) Enrichissement cascade (Top N)
    # 4.1 Prix (Yahoo)
    prices = []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        prices.append(yf_price_fast(sym))
        if i % 120 == 0:
            print(f"[YF.fast] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS/2)
    cand["price"] = prices

    # 4.2 TradingView (batch)
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

    # 4.3 Finnhub (analystes)
    an_bucket, an_votes = [], []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        b, v = finnhub_analyst(sym)
        an_bucket.append(b)
        an_votes.append(v)
        if i % 80 == 0:
            print(f"[FINNHUB] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    cand["analyst_bucket"] = an_bucket
    cand["analyst_votes"]  = an_votes

    # 4.4 Fallback Alpha Vantage pour sector/industry si Unknown côté cand
    if ALPHAV_KEY:
        miss = (cand["sector"].astype(str).str.strip().eq("Unknown")).sum()
        if miss:
            print(f"[AV] trying sector/industry for ~{miss} unknown…")
        sec_f, ind_f = [], []
        for _, row in cand.iterrows():
            if str(row["sector"]).strip() != "Unknown":
                sec_f.append(None); ind_f.append(None); continue
            sec, ind = alphav_overview(row["ticker_yf"])
            sec_f.append(sec); ind_f.append(ind)
            time.sleep(SLEEP_BETWEEN_CALLS)
        # remplace si non-null
        for i, (sec, ind) in enumerate(zip(sec_f, ind_f)):
            if isinstance(sec, str) and sec.strip():
                cand.loc[cand.index[i], "sector"] = sec
            if isinstance(ind, str) and ind.strip():
                cand.loc[cand.index[i], "industry"] = ind

    # 5) Construire l’enrichissement à merger dans la base (évite duplicates sur sector/industry)
    enrich_cols = [
        "ticker_yf",
        "price", "tv_score", "tv_reco",
        "analyst_bucket", "analyst_votes",
        # sector/industry présents dans cand (potentiellement corrigés) — on les passera en suffixe et coalescera
        "sector", "industry",
    ]
    enrich_df = cand[enrich_cols].copy()

    # 6) Merge enrichissement -> base avec coalescence propre (PATCH anti 'sector_x' collisions)
    base = base.merge(
        enrich_df,
        on="ticker_yf",
        how="left",
        suffixes=("", "_cand"),
    )

    # Coalesce sector/industry: préfère valeur cand quand dispo
    for col in ("sector", "industry"):
        rcol = f"{col}_cand"
        if rcol in base.columns:
            base[col] = base[col].where(base[rcol].isna() | (base[rcol].astype(str).str.strip() == ""), base[rcol])
            base.drop(columns=[rcol], inplace=True)

    # 7) Pillars & score (sur toute la base ; TopN mieux remplis)
    def is_strong_tv(v):
        return str(v or "").upper() == "STRONG_BUY"
    def is_bull_an(v):
        return _bucket_from_analyst(v) == "BUY"

    base["p_tv"]   = base["tv_reco"].map(is_strong_tv)
    base["p_an"]   = base["analyst_bucket"].map(is_bull_an)
    # pour l’instant p_tech = p_tv (si tu ajoutes une tech locale plus tard, remplace ici)
    base["p_tech"] = base["p_tv"]

    base["pillars_met"] = base[["p_tech","p_tv","p_an"]].fillna(False).sum(axis=1)
    votes = pd.to_numeric(base.get("analyst_votes", pd.Series(index=base.index)), errors="coerce")
    base["votes_bin"] = pd.cut(votes.fillna(-1), bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])

    s_tv = pd.to_numeric(base.get("tv_score", 0), errors="coerce").fillna(0)
    s_p  = base["pillars_met"].fillna(0)
    s_v  = votes.fillna(0).clip(lower=0, upper=30) / 30.0
    base["rank_score"] = (s_tv * 0.6) + (s_p * 0.3) + (s_v * 0.1)

    # 8) Buckets
    tv_up = base["tv_reco"].astype(str).str.upper()
    an_up = base["analyst_bucket"].astype(str).map(lambda x: (_bucket_from_analyst(x) or ""))

    cond_confirmed = tv_up.eq("STRONG_BUY") & (an_up == "BUY")
    cond_pre       = tv_up.isin(["STRONG_BUY","BUY"]) & (~cond_confirmed)

    base["bucket"] = "other"
    base.loc[cond_pre, "bucket"] = "pre_signal"
    base.loc[cond_confirmed, "bucket"] = "confirmed"

    # 9) Compléter ticker_tv si absent
    if "ticker_tv" not in base.columns:
        base["ticker_tv"] = base["ticker_yf"].map(_yf_symbol_to_tv)

    # 10) Exports (tri sur score)
    export_cols = [
        "ticker_yf","ticker_tv","price","mcap_usd","avg_dollar_vol",
        "tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry",
        "p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","bucket"
    ]
    # assure les colonnes manquantes
    for c in export_cols:
        if c not in base.columns:
            base[c] = None

    cand_all = base[export_cols].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)
    _save(cand_all, "candidates_all_ranked.csv")

    confirmed = cand_all[cand_all["bucket"] == "confirmed"].copy()
    _save(confirmed, "confirmed_STRONGBUY.csv")

    pre = cand_all[cand_all["bucket"] == "pre_signal"].copy()
    _save(pre, "anticipative_pre_signals.csv")

    # events placeholder
    events = cand_all.head(0).copy()
    _save(events, "event_driven_signals.csv")

    # 11) signals_history minimal (pour l'Action / backtest)
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sig_hist = cand_all[["ticker_yf", "sector", "bucket", "tv_reco", "analyst_bucket"]].copy()
    sig_hist.insert(0, "date", today)
    _save(sig_hist, "signals_history.csv")

    # 12) Couverture / Diagnostics
    cov = {
        "universe": int(len(base)),
        "topN_enriched": int(len(cand)),
        "tv_reco_filled": int(cand_all["tv_reco"].notna().sum()),
        "finnhub_filled": int(cand_all["analyst_bucket"].notna().sum()),
        "sector_known": int((cand_all["sector"].astype(str) != "Unknown").sum()),
        "industry_known": int((cand_all["industry"].astype(str) != "Unknown").sum()),
        "confirmed_count": int(len(confirmed)),
        "pre_count": int(len(pre)),
    }
    print("[COVERAGE]", cov)

if __name__ == "__main__":
    main()
