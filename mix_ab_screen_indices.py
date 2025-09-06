# mix_ab_screen_indices.py
# Build a compact, fully-enriched screening set and publish rich CSVs for the dashboard.

import os
import sys
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timezone

# ========= Paths / IO =========
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

# ========= Config =========
TOP_ENRICH_N = int(os.getenv("SCREEN_TOPN", "120"))  # second pass agressif
TV_BATCH = 80
REQ_TIMEOUT = 20
SLEEP_THIN = 0.12     # petit délai pour rester “courtois”
SLEEP_HEAVY = 0.35    # délai pour appels plus lourds (.info / AV / Finnhub)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "")

# ========= Utils =========
def _yf_to_tv(sym: str) -> str:
    # BRK-B -> BRK.B
    return (sym or "").replace("-", ".")

def _safe_num(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except Exception:
        return None

def _bucket_from_analyst(label: str | None) -> str | None:
    if not label:
        return None
    u = str(label).upper().replace(" ", "_")
    if u in {"BUY","STRONG_BUY","OUTPERFORM","OVERWEIGHT"}: return "BUY"
    if u in {"SELL","STRONG_SELL","UNDERPERFORM","UNDERWEIGHT"}: return "SELL"
    if u in {"HOLD","NEUTRAL"}: return "HOLD"
    return "HOLD"

def _label_from_tv_score(score: float | None) -> str | None:
    # map simple depuis Recommend.All (~[-1..+1])
    if score is None: return None
    if score >= 0.5:  return "STRONG_BUY"
    if score >= 0.2:  return "BUY"
    if score <= -0.5: return "STRONG_SELL"
    if score <= -0.2: return "SELL"
    return "NEUTRAL"

def _now_date():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ========= TradingView batch =========
def tv_scan_batch(symbols_tv: List[str]) -> Dict[str, Dict]:
    """
    Retourne {symbol_tv: {"tv_score": float, "tv_reco": str}}
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
            tv_reco  = _label_from_tv_score(tv_score)
            if base and base not in out:
                out[base] = {"tv_score": tv_score, "tv_reco": tv_reco}
        print(f"[TV] filled {len(out)}/{len(symbols_tv)}")
    except Exception as e:
        print(f"[TV] exception: {e}")
    return out

# ========= Finnhub =========
def finnhub_analyst(symbol_yf: str) -> Tuple[str | None, int | None]:
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
            if isinstance(v, int): votes += v
        sb = (rec.get("strongBuy") or 0) + (rec.get("buy") or 0)
        ss = (rec.get("strongSell") or 0) + (rec.get("sell") or 0)
        if sb >= ss + 3: bucket = "BUY"
        elif ss >= sb + 3: bucket = "SELL"
        else: bucket = "HOLD"
        return (bucket, votes or None)
    except Exception:
        return (None, None)

# ========= Alpha Vantage =========
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

# ========= yfinance helpers =========
def yf_price_fast(symbol_yf: str) -> float | None:
    try:
        y = yf.Ticker(symbol_yf)
        fi = getattr(y, "fast_info", None) or {}
        p = fi.get("last_price")
        return float(p) if p is not None else None
    except Exception:
        return None

def yf_fast_meta(symbol_yf: str) -> Tuple[float | None, float | None, float | None]:
    """
    return (price, mcap, avg_dollar_vol) depuis fast_info si possible
    """
    try:
        y = yf.Ticker(symbol_yf)
        fi = getattr(y, "fast_info", None) or {}
        price = _safe_num(fi.get("last_price"))
        mcap  = _safe_num(fi.get("market_cap"))
        vol10 = _safe_num(fi.get("ten_day_average_volume"))
        adv   = None
        if price is not None and vol10 is not None:
            adv = price * vol10
        return (price, mcap, adv)
    except Exception:
        return (None, None, None)

def yf_info_fallback(symbol_yf: str) -> Dict:
    """
    Fallback plus lourd (une poignée seulement) :
    retourne dict pouvant contenir: price, marketCap, sector, industry
    """
    try:
        y = yf.Ticker(symbol_yf)
        info = y.info or {}
        out = {}
        # current price
        for k in ("currentPrice","regularMarketPrice","previousClose"):
            if k in info and _safe_num(info[k]) is not None:
                out["price"] = float(info[k]); break
        # market cap
        if _safe_num(info.get("marketCap")) is not None:
            out["mcap_usd"] = float(info["marketCap"])
        # sector/industry
        if isinstance(info.get("sector"), str) and info["sector"].strip():
            out["sector"] = info["sector"].strip()
        if isinstance(info.get("industry"), str) and info["industry"].strip():
            out["industry"] = info["industry"].strip()
        return out
    except Exception:
        return {}

# ========= Main =========
def main():
    # 0) Charger univers + catalog
    uni = _read_csv_required("universe_in_scope.csv")
    print(f"[UNI] in-scope: {len(uni)}")

    cat = _read_csv_required("sector_catalog.csv")

    # helper pickcol (case-insensitive)
    def pickcol(df, names):
        for n in names:
            if n in df.columns:
                return n
            for c in df.columns:
                if c.lower() == n.lower():
                    return c
        return None

    tcol = pickcol(cat, ["ticker","ticker_yf","symbol"])
    scol = pickcol(cat, ["sector","gics sector","gics_sector"])
    icol = pickcol(cat, ["industry","gics sub-industry","gics_sub_industry","sub_industry"])

    if tcol is None:
        cat = pd.DataFrame(columns=["ticker_yf","sector","industry"])
    else:
        ren = {tcol:"ticker_yf"}
        if scol: ren[scol] = "sector"
        if icol: ren[icol] = "industry"
        cat = cat.rename(columns=ren)
        if "sector"   not in cat.columns:   cat["sector"] = "Unknown"
        if "industry" not in cat.columns: cat["industry"] = "Unknown"
        cat = cat[["ticker_yf","sector","industry"]].drop_duplicates("ticker_yf")

    # 1) Base = univers + mapping secteur
    base = uni.merge(cat, on="ticker_yf", how="left")
    if "sector" not in base.columns:   base["sector"] = "Unknown"
    if "industry" not in base.columns: base["industry"] = "Unknown"
    base["sector"]   = base["sector"].fillna("Unknown")
    base["industry"] = base["industry"].fillna("Unknown")

    # Assure ticker_tv
    if "ticker_tv" not in base.columns:
        base["ticker_tv"] = base["ticker_yf"].map(_yf_to_tv)

    # 2) Enrichissement “léger” (toute la liste)
    prices, mcaps, advs = [], [], []
    for i, sym in enumerate(base["ticker_yf"], 1):
        p, m, a = yf_fast_meta(sym)
        prices.append(p); mcaps.append(m); advs.append(a)
        if i % 120 == 0: print(f"[YF.fast] {i}/{len(base)}"); time.sleep(0.8)
        else: time.sleep(SLEEP_THIN)
    base["price"] = prices
    base["mcap_usd"] = mcaps
    base["avg_dollar_vol"] = advs

    # 3) TradingView reco (batch)
    tv_map: Dict[str, Dict] = {}
    syms_tv = [ _yf_to_tv(s) for s in base["ticker_yf"] ]
    for i in range(0, len(syms_tv), TV_BATCH):
        chunk = [s for s in syms_tv[i:i+TV_BATCH] if s]
        if not chunk: continue
        part = tv_scan_batch(chunk)
        tv_map.update(part)
        time.sleep(SLEEP_THIN)
    base["tv_score"] = [ tv_map.get(_yf_to_tv(s),{}).get("tv_score") for s in base["ticker_yf"] ]
    base["tv_reco"]  = [ tv_map.get(_yf_to_tv(s),{}).get("tv_reco")  for s in base["ticker_yf"] ]

    # 4) Finnhub analyst (léger sur toute la liste, beaucoup seront None en free)
    an_bucket, an_votes = [], []
    for i, sym in enumerate(base["ticker_yf"], 1):
        b, v = finnhub_analyst(sym)
        an_bucket.append(b); an_votes.append(v)
        if i % 80 == 0: print(f"[FINNHUB] {i}/{len(base)}"); time.sleep(0.8)
        else: time.sleep(SLEEP_HEAVY)
    base["analyst_bucket"] = an_bucket
    base["analyst_votes"]  = an_votes

    # 5) Compléter sector/industry Unknown via Alpha Vantage (léger sur toute la liste)
    if ALPHAV_KEY:
        fill_idx = base["sector"].eq("Unknown") | base["industry"].eq("Unknown")
        idxs = base.index[fill_idx].tolist()
        if idxs:
            print(f"[AV] trying sector/industry for ~{len(idxs)} rows…")
        for j, irow in enumerate(idxs, 1):
            sym = base.at[irow, "ticker_yf"]
            sec, ind = alphav_overview(sym)
            if isinstance(sec, str) and sec.strip(): base.at[irow, "sector"] = sec.strip()
            if isinstance(ind, str) and ind.strip(): base.at[irow, "industry"] = ind.strip()
            if j % 80 == 0: time.sleep(0.8)
            else: time.sleep(SLEEP_HEAVY)

    # ===== Second pass agressif sur Top N =====
    # Sélection Top N par liquidité (avg_dollar_vol) desc
    base["avg_dollar_vol"] = pd.to_numeric(base["avg_dollar_vol"], errors="coerce")
    # si adv vide, tri sur mcap desc comme fallback
    base["mcap_usd"] = pd.to_numeric(base["mcap_usd"], errors="coerce")
    cand = base.sort_values(
        by=["avg_dollar_vol","mcap_usd"], ascending=[False, False]
    ).head(TOP_ENRICH_N).copy()

    print(f"[2nd-pass] forcing fill on Top {len(cand)}")

    # 2nd-pass : forcer price/mcap via y.info quand manquant
    for i, row in cand[cand[["price","mcap_usd"]].isna().any(axis=1)].iterrows():
        sym = row["ticker_yf"]
        inf = yf_info_fallback(sym)
        if ("price" in inf) and pd.isna(cand.at[i,"price"]):
            cand.at[i,"price"] = inf["price"]
        if ("mcap_usd" in inf) and pd.isna(cand.at[i,"mcap_usd"]):
            cand.at[i,"mcap_usd"] = inf["mcap_usd"]
        # recalc ADV si possible
        if pd.isna(cand.at[i,"avg_dollar_vol"]) and pd.notna(cand.at[i,"price"]):
            # on ne relit pas vol10 ici (trop lourd), on laisse vide si pas dispo
            pass
        time.sleep(SLEEP_HEAVY)

    # 2nd-pass : sector/industry encore Unknown → fallback y.info
    mask_si = (cand["sector"].eq("Unknown")) | (cand["industry"].eq("Unknown"))
    for i, row in cand[mask_si].iterrows():
        sym = row["ticker_yf"]
        inf = yf_info_fallback(sym)
        if ("sector" in inf) and row["sector"] == "Unknown":
            cand.at[i,"sector"] = inf["sector"]
        if ("industry" in inf) and row["industry"] == "Unknown":
            cand.at[i,"industry"] = inf["industry"]
        time.sleep(SLEEP_HEAVY)

    # 2nd-pass : TV / Finnhub “trous” (petites touches)
    # TV : si tv_reco manquant pour certains, on essaie un mini-batch spécifique
    still_tv = cand[cand["tv_reco"].isna()]["ticker_yf"].map(_yf_to_tv).dropna().unique().tolist()
    for k in range(0, len(still_tv), TV_BATCH):
        ch = still_tv[k:k+TV_BATCH]
        part = tv_scan_batch(ch)
        for tvsym, obj in part.items():
            idx = cand.index[cand["ticker_tv"] == tvsym]
            if len(idx):
                cand.loc[idx, "tv_score"] = obj.get("tv_score")
                cand.loc[idx, "tv_reco"]  = obj.get("tv_reco")
        time.sleep(SLEEP_THIN)

    # Finnhub : compléter quelques trous (mais limité en free)
    still_an = cand[cand["analyst_bucket"].isna()]["ticker_yf"].tolist()
    for sym in still_an:
        b, v = finnhub_analyst(sym)
        if b is not None or v is not None:
            idx = cand.index[cand["ticker_yf"] == sym]
            cand.loc[idx, "analyst_bucket"] = b
            cand.loc[idx, "analyst_votes"]  = v
        time.sleep(SLEEP_HEAVY)

    # Remettre cand dans base (on veut exporter cand ordonné et enrichi)
    base = base.drop(columns=[c for c in ["price","mcap_usd","avg_dollar_vol","tv_score","tv_reco","analyst_bucket","analyst_votes"] if c in base.columns])
    base = base.merge(
        cand[["ticker_yf","price","mcap_usd","avg_dollar_vol","tv_score","tv_reco","analyst_bucket","analyst_votes","sector","industry"]],
        on="ticker_yf", how="left"
    )
    # Pour export principal on repartira de cand (topN bien rempli). On calcule aussi un export global si besoin.

    # ===== Pillars / Score / Buckets sur cand =====
    def is_strong_tv(v): return str(v or "").upper() == "STRONG_BUY"
    def is_bull_an(v):  return _bucket_from_analyst(v) == "BUY"

    cand["p_tv"]   = cand["tv_reco"].map(is_strong_tv)
    cand["p_an"]   = cand["analyst_bucket"].map(is_bull_an)
    cand["p_tech"] = cand["p_tv"]  # placeholder si tu ajoutes plus tard une tech locale

    cand["pillars_met"] = cand[["p_tech","p_tv","p_an"]].fillna(False).sum(axis=1)

    votes = pd.to_numeric(cand.get("analyst_votes", pd.Series([None]*len(cand))), errors="coerce")
    cand["votes_bin"] = pd.cut(votes.fillna(-1), bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])

    s_tv = pd.to_numeric(cand["tv_score"], errors="coerce").fillna(0.0)
    s_p  = pd.to_numeric(cand["pillars_met"], errors="coerce").fillna(0.0)
    s_v  = votes.fillna(0).clip(lower=0, upper=30) / 30.0
    cand["rank_score"] = (s_tv * 0.6) + (s_p * 0.3) + (s_v * 0.1)

    cond_confirmed = (cand["tv_reco"].str.upper().eq("STRONG_BUY")) & (cand["analyst_bucket"].map(_bucket_from_analyst).eq("BUY"))
    cond_pre = (cand["tv_reco"].str.upper().isin(["STRONG_BUY","BUY"])) & (~cond_confirmed)

    cand["bucket"] = "other"
    cand.loc[cond_pre, "bucket"] = "pre_signal"
    cand.loc[cond_confirmed, "bucket"] = "confirmed"

    # ticker_tv sûre
    if "ticker_tv" not in cand.columns:
        cand["ticker_tv"] = cand["ticker_yf"].map(_yf_to_tv)

    # ===== Exports =====
    export_cols = [
        "ticker_yf","ticker_tv","price","mcap_usd","avg_dollar_vol",
        "tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry",
        "p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","bucket"
    ]
    cand = cand[export_cols].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)

    _save(cand, "candidates_all_ranked.csv")
    _save(cand[cand["bucket"]=="confirmed"].copy(), "confirmed_STRONGBUY.csv")
    _save(cand[cand["bucket"]=="pre_signal"].copy(), "anticipative_pre_signals.csv")

    # events placeholder
    _save(cand.head(0).copy(), "event_driven_signals.csv")

    # signals_history minimal pour le backtest
    today = _now_date()
    sig_hist = cand.copy()
    sig_hist = sig_hist[["ticker_yf","sector","bucket","tv_reco","analyst_bucket"]]
    sig_hist.insert(0, "date", today)
    _save(sig_hist, "signals_history.csv")

    # Couverture
    cov = {
        "universe_in_scope": int(len(uni)),
        "top_enriched": int(len(cand)),
        "price_filled": int(cand["price"].notna().sum()),
        "mcap_filled": int(cand["mcap_usd"].notna().sum()),
        "tv_reco_filled": int(cand["tv_reco"].notna().sum()),
        "analyst_filled": int(cand["analyst_bucket"].notna().sum()),
        "sector_known": int((cand["sector"].fillna("Unknown")!="Unknown").sum()),
        "industry_known": int((cand["industry"].fillna("Unknown")!="Unknown").sum()),
        "confirmed_count": int((cand["bucket"]=="confirmed").sum()),
        "pre_count": int((cand["bucket"]=="pre_signal").sum()),
    }
    print("[COVERAGE]", cov)

if __name__ == "__main__":
    main()
