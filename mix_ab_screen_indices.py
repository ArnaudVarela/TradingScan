# mix_ab_screen_indices.py
# Build a compact, fully-enriched screening set and publish rich CSVs for the dashboard.

import os, sys, json, time, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf

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
        print(f"❌ Missing required file: {p}"); sys.exit(2)
    return pd.read_csv(p)

# --------- Config ---------
TOP_N = int(os.getenv("SCREEN_TOPN", "180"))   # top by liquidity to enrich heavily
TV_BATCH = 80
REQ_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.25

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "")

MCAP_MAX_USD = 75_000_000_000

# --------- helpers ---------
def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)

def _yf_symbol_to_tv(sym: str) -> str:
    return (sym or "").replace("-", ".")

def _safe_num(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except Exception:
        return None

def _label_from_tv_score(score: float | None) -> str | None:
    if score is None: return None
    if score >= 0.5:  return "STRONG_BUY"
    if score >= 0.2:  return "BUY"
    if score <= -0.5: return "STRONG_SELL"
    if score <= -0.2: return "SELL"
    return "NEUTRAL"

def _bucket_from_analyst(label) -> str | None:
    if label is None or _is_nan(label): return None
    u = str(label).upper().strip().replace(" ", "_")
    if u in {"BUY","STRONG_BUY","OUTPERFORM","OVERWEIGHT"}: return "BUY"
    if u in {"SELL","STRONG_SELL","UNDERPERFORM","UNDERWEIGHT"}: return "SELL"
    if u in {"HOLD","NEUTRAL"}: return "HOLD"
    return "HOLD"

# --------- TV batch ---------
def tv_scan_batch(symbols_tv: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if not symbols_tv: return out
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
            print(f"[TV] HTTP {r.status_code} -> skip batch"); return out
        data = r.json()
        for row in data.get("data", []):
            sym = row.get("s",""); base = sym.split(":")[-1]
            vals = row.get("d", [])
            tv_score = _safe_num(vals[0]) if vals else None
            tv_reco  = _label_from_tv_score(tv_score)
            if base and base not in out:
                out[base] = {"tv_score": tv_score, "tv_reco": tv_reco}
        print(f"[TV] filled {len(out)}/{len(symbols_tv)}")
    except Exception as e:
        print(f"[TV] exception: {e}")
    return out

# --------- Finnhub analyst & profile2 (mcap) ---------
def finnhub_analyst(symbol_yf: str) -> Tuple[str | None, int | None]:
    if not FINNHUB_KEY: return (None, None)
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol_yf}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200: return (None, None)
        arr = r.json()
        if not isinstance(arr, list) or not arr: return (None, None)
        rec = arr[0]
        votes = 0
        for k in ("strongBuy","buy","hold","sell","strongSell"):
            v = rec.get(k); votes += int(v) if isinstance(v, int) else 0
        sb = (rec.get("strongBuy") or 0) + (rec.get("buy") or 0)
        ss = (rec.get("strongSell") or 0) + (rec.get("sell") or 0)
        bucket = "BUY" if sb >= ss + 3 else ("SELL" if ss >= sb + 3 else "HOLD")
        return (bucket, votes or None)
    except Exception:
        return (None, None)

def finnhub_profile2_mcap(symbol_yf: str) -> float | None:
    if not FINNHUB_KEY: return None
    try:
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol_yf}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200: return None
        j = r.json()
        mc = j.get("marketCapitalization")
        if mc is None: return None
        mc = float(mc)
        # Normalize units: if the number looks like "in billions", convert to USD
        # Heuristic: < 10_000 → billions;  otherwise assume already in USD.
        if mc < 10_000:
            return mc * 1e9
        return mc
    except Exception:
        return None

# --------- Yahoo price fast + vectorized dv20 ---------
def yf_price_fast(symbol_yf: str) -> float | None:
    try:
        fi = (yf.Ticker(symbol_yf).fast_info or {})
        p = fi.get("last_price")
        return float(p) if p is not None else None
    except Exception:
        return None

def compute_dv20_vectorized(tickers: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not tickers: return out
    try:
        df = yf.download(tickers=tickers, period="3mo", interval="1d",
                         auto_adjust=True, progress=False, group_by="ticker", threads=True)
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    sub = df[t][["Close","Volume"]].dropna()
                    if len(sub) >= 5:
                        tail = sub.tail(20)
                        out[t] = float((tail["Close"] * tail["Volume"]).mean())
                except Exception:
                    pass
        else:
            sub = df[["Close","Volume"]].dropna()
            if len(sub) >= 5:
                out[tickers[0]] = float((sub.tail(20)["Close"] * sub.tail(20)["Volume"]).mean())
    except Exception:
        pass
    return out

# --------- Main pipeline ---------
def main():
    # 1) Universe (already sector-scoped + roughly <75B, with NaNs preserved)
    base = _read_csv_required("universe_in_scope.csv")
    if "ticker_yf" not in base.columns:
        raise SystemExit("❌ universe_in_scope.csv must contain 'ticker_yf'")

    # 2) Sector catalog merge (keeps everything)
    cat = _read_csv_required("sector_catalog.csv")
    def pickcol(df, names):
        for n in names:
            if n in df.columns: return n
            for c in df.columns:
                if c.lower() == n.lower(): return c
        return None
    tcol = pickcol(cat, ["ticker","ticker_yf","symbol"])
    scol = pickcol(cat, ["sector"])
    icol = pickcol(cat, ["industry","gics sub-industry","sub_industry"])
    if tcol is None:
        cat = pd.DataFrame(columns=["ticker_yf","sector","industry"])
    else:
        cat = cat.rename(columns={tcol:"ticker_yf", **({scol:"sector"} if scol else {}),
                                  **({icol:"industry"} if icol else {})})
        for c in ("sector","industry"):
            if c not in cat.columns: cat[c] = "Unknown"
        cat = cat[["ticker_yf","sector","industry"]].drop_duplicates("ticker_yf")

    base = base.merge(cat, on="ticker_yf", how="left", suffixes=("", "_cat"))
    # Fill sector/industry from catalog if missing
    for c in ("sector","industry"):
        r = f"{c}_cat"
        if r in base.columns:
            base[c] = base[c].fillna("Unknown")
            base[c] = base[c].where(base[r].isna() | (base[r].astype(str).str.strip()==""), base[r])
            base.drop(columns=[r], inplace=True)

    # 3) Ensure dv20 (liquidity) is present for TopN selection
    base["avg_dollar_vol"] = pd.to_numeric(base.get("avg_dollar_vol"), errors="coerce")
    if base["avg_dollar_vol"].isna().any():
        need = base.loc[base["avg_dollar_vol"].isna(), "ticker_yf"].tolist()
        dv_map = {}
        for i in range(0, len(need), 120):
            part = compute_dv20_vectorized(need[i:i+120])
            dv_map.update(part); time.sleep(0.2)
        if dv_map:
            base["avg_dollar_vol"] = base.apply(
                lambda r: r["avg_dollar_vol"] if pd.notna(r["avg_dollar_vol"]) else dv_map.get(r["ticker_yf"]),
                axis=1
            )

    # 4) Choose TopN to enrich heavily
    base_sorted = base.sort_values("avg_dollar_vol", ascending=False, na_position="last").reset_index(drop=True)
    cand = base_sorted.head(TOP_N).copy()
    print(f"[UNI] in-scope: {len(base)}  |  TopN to enrich: {len(cand)}")

    # 5) Prices (fast) for cand; fallback via history later if needed
    prices = []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        prices.append(yf_price_fast(sym))
        if i % 120 == 0: print(f"[YF.fast] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS/2)
    cand["price"] = prices

    # 6) TradingView (batch)
    syms_tv = [_yf_symbol_to_tv(s) for s in cand["ticker_yf"]]
    tv_out: Dict[str, Dict] = {}
    for i in range(0, len(syms_tv), TV_BATCH):
        chunk = [s for s in syms_tv[i:i+TV_BATCH] if s]
        if not chunk: continue
        part = tv_scan_batch(chunk)
        tv_out.update(part)
        time.sleep(SLEEP_BETWEEN_CALLS)
    cand["tv_score"] = [tv_out.get(_yf_symbol_to_tv(s), {}).get("tv_score") for s in cand["ticker_yf"]]
    cand["tv_reco"]  = [tv_out.get(_yf_symbol_to_tv(s), {}).get("tv_reco")  for s in cand["ticker_yf"]]

    # 7) Finnhub (analyst) + Finnhub (mcap)
    an_bucket, an_votes, mcaps = [], [], []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        b, v = finnhub_analyst(sym); an_bucket.append(b); an_votes.append(v)
        mcaps.append(finnhub_profile2_mcap(sym))
        if i % 80 == 0: print(f"[FINNHUB] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    cand["analyst_bucket"] = an_bucket
    cand["analyst_votes"]  = an_votes
    cand["mcap_usd_finnhub"] = mcaps

    # 8) Yahoo info fallback for MCAP where still missing (small subset)
    need_m = cand["mcap_usd_finnhub"].isna()
    if need_m.any():
        m2 = []
        for sym in cand.loc[need_m, "ticker_yf"]:
            val = None
            try:
                inf = (yf.Ticker(sym).info or {})
                val = inf.get("marketCap")
                if val is not None: val = float(val)
            except Exception:
                pass
            m2.append(val)
            time.sleep(SLEEP_BETWEEN_CALLS/2)
        cand.loc[need_m, "mcap_usd_yfinfo"] = m2

    # 9) Build final enrichment set and merge back to base
    cand["mcap_usd_final"] = cand["mcap_usd_finnhub"].fillna(cand.get("mcap_usd_yfinfo"))
    enrich_cols = ["ticker_yf","price","tv_score","tv_reco","analyst_bucket","analyst_votes",
                   "mcap_usd_final","sector","industry"]
    enrich_df = cand[enrich_cols].copy()

    base = base.merge(enrich_df, on="ticker_yf", how="left", suffixes=("", "_cand"))
    for col in ("sector","industry"):
        rcol = f"{col}_cand"
        if rcol in base.columns:
            base[col] = base[col].where(base[rcol].isna() | (base[rcol].astype(str).str.strip()==""), base[rcol])
            base.drop(columns=[rcol], inplace=True)

    # 10) Compute a simple technical pillar from TV (placeholder)
    def is_strong_tv(v) -> bool:
        if v is None or _is_nan(v): return False
        return str(v).upper().strip() == "STRONG_BUY"
    def is_bull_an(v) -> bool:
        return _bucket_from_analyst(v) == "BUY"

    base["p_tv"]   = base.get("tv_reco").map(is_strong_tv)
    base["p_an"]   = base.get("analyst_bucket").map(is_bull_an)
    base["p_tech"] = base["p_tv"]  # placeholder until you add your local TA rules
    base["pillars_met"] = base[["p_tech","p_tv","p_an"]].fillna(False).sum(axis=1)

    # 11) Ranking
    votes = pd.to_numeric(base.get("analyst_votes"), errors="coerce")
    base["votes_bin"] = pd.cut(votes.fillna(-1), bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])
    s_tv = pd.to_numeric(base.get("tv_score", 0), errors="coerce").fillna(0)
    s_p  = pd.to_numeric(base["pillars_met"], errors="coerce").fillna(0)
    s_v  = votes.fillna(0).clip(lower=0, upper=30) / 30.0
    base["rank_score"] = (s_tv * 0.6) + (s_p * 0.3) + (s_v * 0.1)

    # 12) Final MCAP coalescence and enforcement < 75B
    base["mcap_usd_final"] = pd.to_numeric(base.get("mcap_usd_final"), errors="coerce") \
                                .fillna(pd.to_numeric(base.get("mcap_usd"), errors="coerce"))
    base = base[(base["mcap_usd_final"].isna()) | (base["mcap_usd_final"] < MCAP_MAX_USD)].copy()

    # 13) Buckets
    tv_up = base.get("tv_reco").astype(str).str.upper()
    an_up = base.get("analyst_bucket").map(lambda x: (_bucket_from_analyst(x) or ""))
    cond_confirmed = tv_up.eq("STRONG_BUY") & (an_up == "BUY")
    cond_pre       = tv_up.isin(["STRONG_BUY","BUY"]) & (~cond_confirmed)
    base["bucket"] = "other"
    base.loc[cond_pre, "bucket"] = "pre_signal"
    base.loc[cond_confirmed, "bucket"] = "confirmed"

    # 14) Export
    if "ticker_tv" not in base.columns:
        base["ticker_tv"] = base["ticker_yf"].map(_yf_symbol_to_tv)

    export_cols = [
        "ticker_yf","ticker_tv","price","mcap_usd_final","avg_dollar_vol",
        "tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry",
        "p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","bucket"
    ]
    for c in export_cols:
        if c not in base.columns: base[c] = None

    cand_all = base[export_cols].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)
    _save(cand_all, "candidates_all_ranked.csv")

    confirmed = cand_all[cand_all["bucket"] == "confirmed"].head(10).copy()
    pre       = cand_all[cand_all["bucket"] == "pre_signal"].head(50).copy()
    events    = cand_all.head(100).copy()  # placeholder list for now

    _save(confirmed, "confirmed_STRONGBUY.csv")
    _save(pre, "anticipative_pre_signals.csv")
    _save(events, "event_driven_signals.csv")

    # Minimal signals_history for backtest
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sig_hist = cand_all[["ticker_yf","sector","bucket","tv_reco","analyst_bucket"]].copy()
    sig_hist.insert(0, "date", today)
    _save(sig_hist, "signals_history.csv")

    cov = {
        "universe": int(len(base)),
        "tv_reco_filled": int(cand_all["tv_reco"].notna().sum()),
        "finnhub_analyst": int(cand_all["analyst_bucket"].notna().sum()),
        "mcap_final_nonnull": int(cand_all["mcap_usd_final"].notna().sum()),
        "price_nonnull": int(cand_all["price"].notna().sum()),
        "confirmed_count": int(len(confirmed)),
        "pre_count": int(len(pre)),
    }
    print("[COVERAGE]", cov)

if __name__ == "__main__":
    main()
