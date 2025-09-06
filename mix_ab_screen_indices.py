# mix_ab_screen_indices.py
# Build a compact, fully-enriched screening set and publish rich CSVs for the dashboard.
# Ajouts majeurs:
# - Pilier technique local (SMA50/200, RSI14, MACD) via batch OHLC
# - Détection "event-driven" (gap-up, volume spike, breakout 20j)
# - TradingView: inclut AMEX + déduplication (garde le meilleur score)
# - Buckets stricts + exports Top10 confirmed / Top50 pre-signal / Top100 event

import os, sys, json, time, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf

# --------- Paths / IO ---------
PUBLIC_DIR = Path("dashboard/public"); PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

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
TOP_N = int(os.getenv("SCREEN_TOPN", "180"))   # élargi pour couvrir Top100 events
TV_BATCH = 80
REQ_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.25

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "")

# Buckets thresholds
MIN_LIQ_CONFIRMED   = float(os.getenv("MIN_LIQ_CONFIRM_USD", "5000000"))  # $5M/j
MIN_ANALYST_VOTES   = int(os.getenv("MIN_ANALYST_VOTES", "10"))

# --------- Helpers ---------
def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)

def _yf_symbol_to_tv(sym: str) -> str:
    return (sym or "").replace("-", ".")

def _safe_num(v):
    try:
        n = float(v); return n if math.isfinite(n) else None
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

# --------- TradingView batch scan (AMEX + dédup) ---------
def tv_scan_batch(symbols_tv: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if not symbols_tv: return out

    url = "https://scanner.tradingview.com/america/scan"
    payload = {
        "symbols": {
            "tickers": [f"NASDAQ:{s}" for s in symbols_tv] + [f"NYSE:{s}" for s in symbols_tv] + [f"AMEX:{s}" for s in symbols_tv],
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
            sym = row.get("s", ""); base = sym.split(":")[-1]
            vals = row.get("d", []); tv_score = _safe_num(vals[0]) if vals else None
            tv_reco = _label_from_tv_score(tv_score)
            if not base: continue
            prev = out.get(base)
            if prev is None or (tv_score is not None and (prev["tv_score"] is None or tv_score > prev["tv_score"])):
                out[base] = {"tv_score": tv_score, "tv_reco": tv_reco}
        print(f"[TV] filled {len(out)}/{len(symbols_tv)}")
    except Exception as e:
        print(f"[TV] exception: {e}")
    return out

# --------- Finnhub (analystes) ---------
def finnhub_analyst(symbol_yf: str) -> Tuple[str | None, int | None]:
    if not FINNHUB_KEY: return (None, None)
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol_yf}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200: return (None, None)
        arr = r.json()
        if not isinstance(arr, list) or not arr: return (None, None)
        rec = arr[0]; votes = 0
        for k in ("strongBuy","buy","hold","sell","strongSell"):
            v = rec.get(k); votes += v if isinstance(v,int) else 0
        sb = (rec.get("strongBuy") or 0) + (rec.get("buy") or 0)
        ss = (rec.get("strongSell") or 0) + (rec.get("sell") or 0)
        bucket = "BUY" if sb >= ss + 3 else ("SELL" if ss >= sb + 3 else "HOLD")
        return (bucket, votes or None)
    except Exception:
        return (None, None)

# --------- Alpha Vantage OVERVIEW (fallback secteur/industrie) ---------
def alphav_overview(symbol_yf: str) -> Tuple[str | None, str | None]:
    if not ALPHAV_KEY: return (None, None)
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function":"OVERVIEW","symbol":symbol_yf,"apikey":ALPHAV_KEY}
        r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
        if r.status_code != 200: return (None, None)
        j = r.json(); return (j.get("Sector") or None, j.get("Industry") or None)
    except Exception:
        return (None, None)

# --------- Yahoo price (léger) ---------
def yf_price_fast(symbol_yf: str) -> float | None:
    try:
        y = yf.Ticker(symbol_yf); fi = getattr(y, "fast_info", None) or {}
        p = fi.get("last_price"); return float(p) if p is not None else None
    except Exception:
        return None

# --------- Technicals batch (SMA/RSI/MACD + Events) ---------
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0); down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _compute_one_ta(df: pd.DataFrame) -> dict:
    # df with columns: Open, High, Low, Close, Volume
    if df is None or df.empty or len(df) < 50:
        return {"technical_local": None, "tech_score": None, "event_score": 0}
    c = df["Close"].astype(float); v = df["Volume"].astype(float)
    sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean() if len(c) >= 200 else pd.Series(index=c.index, dtype=float)
    rsi14 = _compute_rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; signal = macd.ewm(span=9, adjust=False).mean(); hist = macd - signal

    close, s50 = c.iloc[-1], sma50.iloc[-1]
    s200 = sma200.iloc[-1] if len(sma200) and not math.isnan(sma200.iloc[-1]) else None
    rsi = rsi14.iloc[-1] if not math.isnan(rsi14.iloc[-1]) else None
    h_now = hist.iloc[-1] if not math.isnan(hist.iloc[-1]) else None
    h_prev = hist.iloc[-2] if len(hist) >= 2 and not math.isnan(hist.iloc[-2]) else None

    strong = (close > s50) and (s200 is None or s50 > s200) and (h_now is not None and h_now > 0) and (h_prev is not None and h_now > h_prev) and (rsi is not None and 50 <= rsi <= 75)
    buy    = (close > s50) and (h_now is not None and h_now > 0)

    label = "STRONG_BUY" if strong else ("BUY" if buy else "NEUTRAL")
    tech_score = 1.0 if label == "STRONG_BUY" else (0.5 if label == "BUY" else 0.0)

    # Events: gap-up, volume spike, breakout 20d, big % day
    event_score = 0
    if len(c) >= 21:
        pct1d = (c.iloc[-1] / c.iloc[-2] - 1.0)
        vol_ratio = (v.iloc[-1] / max(1.0, v.tail(20).mean())) if v.tail(20).mean() else 0.0
        breakout20 = c.iloc[-1] >= c.tail(20).max()
        gap_up = None
        if "Open" in df.columns:
            gap_up = (df["Open"].iloc[-1] / c.iloc[-2] - 1.0)
        event_score += int(pct1d >= 0.05) + int(vol_ratio >= 1.8) + int(breakout20) + int(gap_up is not None and gap_up >= 0.03)

    return {"technical_local": label, "tech_score": tech_score, "event_score": event_score}

def compute_technicals_batch(symbols_yf: List[str], batch_size: int = 120) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i in range(0, len(symbols_yf), batch_size):
        chunk = [s for s in symbols_yf[i:i+batch_size] if s]
        if not chunk: continue
        try:
            df = yf.download(
                tickers=chunk, period="1y", interval="1d",
                auto_adjust=True, group_by="ticker", threads=True, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                for sym in chunk:
                    try:
                        sub = df[sym][["Open","High","Low","Close","Volume"]].dropna()
                        out[sym] = _compute_one_ta(sub)
                    except Exception:
                        out[sym] = {"technical_local": None, "tech_score": None, "event_score": 0}
            else:
                # single symbol path
                try:
                    sub = df[["Open","High","Low","Close","Volume"]].dropna()
                    out[chunk[0]] = _compute_one_ta(sub)
                except Exception:
                    out[chunk[0]] = {"technical_local": None, "tech_score": None, "event_score": 0}
        except Exception as e:
            print(f"[TA] batch exception: {e}")
        time.sleep(0.4)
    print(f"[TA] computed: {len(out)}/{len(symbols_yf)}")
    return out

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
            if n in df.columns: return n
            for c in df.columns:
                if c.lower() == n.lower(): return c
        return None
    tcol = pickcol(cat, ["ticker", "ticker_yf", "symbol"])
    scol = pickcol(cat, ["sector", "gics sector", "gics_sector"])
    icol = pickcol(cat, ["industry", "gics sub-industry", "gics_sub_industry", "sub_industry"])
    if tcol is None:
        print("[CAT] WARNING: no ticker column in sector_catalog.csv"); cat = pd.DataFrame(columns=["ticker_yf","sector","industry"])
    else:
        rename_map = {tcol:"ticker_yf"}; 
        if scol is not None: rename_map[scol] = "sector"
        if icol is not None: rename_map[icol] = "industry"
        cat = cat.rename(columns=rename_map)
        if "sector" not in cat.columns:   cat["sector"]   = "Unknown"
        if "industry" not in cat.columns: cat["industry"] = "Unknown"
        cat = cat[["ticker_yf","sector","industry"]].drop_duplicates("ticker_yf")

    base = base.merge(cat, on="ticker_yf", how="left")
    if "sector" not in base.columns:   base["sector"]   = "Unknown"
    if "industry" not in base.columns: base["industry"] = "Unknown"
    base["sector"] = base["sector"].fillna("Unknown"); base["industry"] = base["industry"].fillna("Unknown")

    # 3) tri liquidité + TopN
    base["avg_dollar_vol"] = pd.to_numeric(base.get("avg_dollar_vol", None), errors="coerce")
    base_sorted = base.sort_values("avg_dollar_vol", ascending=False, na_position="last").reset_index(drop=True)
    cand = base_sorted.head(TOP_N).copy()

    print(f"[UNI] in-scope: {len(base)}  |  TopN to enrich: {len(cand)}")

    # 4) Enrichissement cascade (Top N)
    # 4.1 Prix (Yahoo)
    prices = []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        prices.append(yf_price_fast(sym))
        if i % 120 == 0: print(f"[YF.fast] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS/2)
    cand["price"] = prices

    # 4.2 TradingView (batch) + AMEX + best-of
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

    # 4.3 Finnhub (analystes)
    an_bucket, an_votes = [], []
    for i, sym in enumerate(cand["ticker_yf"], 1):
        b, v = finnhub_analyst(sym); an_bucket.append(b); an_votes.append(v)
        if i % 80 == 0: print(f"[FINNHUB] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    cand["analyst_bucket"] = an_bucket
    cand["analyst_votes"]  = an_votes

    # 4.4 Fallback Alpha Vantage pour sector/industry si Unknown
    if ALPHAV_KEY:
        miss = (cand["sector"].astype(str).str.strip().eq("Unknown")).sum()
        if miss: print(f"[AV] trying sector/industry for ~{miss} unknown…")
        sec_f, ind_f = [], []
        for _, row in cand.iterrows():
            if str(row["sector"]).strip() != "Unknown":
                sec_f.append(None); ind_f.append(None); continue
            sec, ind = alphav_overview(row["ticker_yf"])
            sec_f.append(sec); ind_f.append(ind); time.sleep(SLEEP_BETWEEN_CALLS)
        for i, (sec, ind) in enumerate(zip(sec_f, ind_f)):
            if isinstance(sec, str) and sec.strip(): cand.loc[cand.index[i], "sector"] = sec
            if isinstance(ind, str) and ind.strip(): cand.loc[cand.index[i], "industry"] = ind

    # 4.5 Pilier technique local + Events (batch OHLC)
    ta_map = compute_technicals_batch(cand["ticker_yf"].tolist(), batch_size=100)
    cand["technical_local"] = [ta_map.get(s, {}).get("technical_local") for s in cand["ticker_yf"]]
    cand["tech_score"]      = [ta_map.get(s, {}).get("tech_score") for s in cand["ticker_yf"]]
    cand["event_score"]     = [ta_map.get(s, {}).get("event_score") for s in cand["ticker_yf"]]

    # 5) Merge enrichissement -> base
    enrich_cols = ["ticker_yf","price","tv_score","tv_reco","analyst_bucket","analyst_votes",
                   "sector","industry","technical_local","tech_score","event_score"]
    enrich_df = cand[enrich_cols].copy()
    base = base.merge(enrich_df, on="ticker_yf", how="left", suffixes=("", "_cand"))
    for col in ("sector","industry","technical_local","tech_score","event_score"):
        rcol = f"{col}_cand"
        if rcol in base.columns:
            base[col] = base[col].where(base[rcol].isna() | (base[rcol].astype(str).str.strip()==""), base[rcol])
            base.drop(columns=[rcol], inplace=True)

    # 6) Pillars & score
    def is_strong_tv(v) -> bool:
        return (str(v).upper().strip() == "STRONG_BUY") if v is not None and not _is_nan(v) else False
    def is_bull_an(v) -> bool:
        return _bucket_from_analyst(v) == "BUY"
    def is_strong_tech(v) -> bool:
        return (str(v).upper().strip() == "STRONG_BUY") if v is not None and not _is_nan(v) else False

    base["p_tv"]   = base.get("tv_reco", pd.Series(index=base.index)).map(is_strong_tv)
    base["p_an"]   = base.get("analyst_bucket", pd.Series(index=base.index)).map(is_bull_an)
    base["p_tech"] = base.get("technical_local", pd.Series(index=base.index)).map(is_strong_tech)
    base["pillars_met"] = base[["p_tech","p_tv","p_an"]].fillna(False).sum(axis=1)

    votes = pd.to_numeric(base.get("analyst_votes", pd.Series(index=base.index)), errors="coerce")
    base["votes_bin"] = pd.cut(votes.fillna(-1), bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])

    tv = pd.to_numeric(base.get("tv_score", 0), errors="coerce").fillna(0)           # [-1..+1]
    tech = pd.to_numeric(base.get("tech_score", 0), errors="coerce").fillna(0)       # [0, 0.5, 1]
    v_norm = votes.fillna(0).clip(lower=0, upper=30) / 30.0                          # [0..1]
    base["rank_score"] = (tv * 0.45) + (tech * 0.35) + (v_norm * 0.20)

    # 7) Buckets stricts
    tv_up = base.get("tv_reco", pd.Series(index=base.index)).astype(str).str.upper()
    an_up = base.get("analyst_bucket", pd.Series(index=base.index)).map(lambda x: (_bucket_from_analyst(x) or ""))
    tech_up = base.get("technical_local", pd.Series(index=base.index)).astype(str).str.upper()
    liq_ok  = pd.to_numeric(base.get("avg_dollar_vol", 0), errors="coerce").fillna(0) >= MIN_LIQ_CONFIRMED
    votes_ok = votes.fillna(0) >= MIN_ANALYST_VOTES

    cond_confirmed = tv_up.eq("STRONG_BUY") & (an_up == "BUY") & tech_up.isin(["STRONG_BUY","BUY"]) & liq_ok & votes_ok
    cond_pre       = tv_up.isin(["STRONG_BUY","BUY"]) & tech_up.isin(["STRONG_BUY","BUY"]) & (~cond_confirmed)

    base["bucket"] = "other"
    base.loc[cond_pre, "bucket"] = "pre_signal"
    base.loc[cond_confirmed, "bucket"] = "confirmed"

    # 8) Events: à partir du event_score (>=2), hors confirmed/pre
    event_mask = (pd.to_numeric(base.get("event_score", 0), errors="coerce").fillna(0) >= 2) & (~cond_pre) & (~cond_confirmed)
    base.loc[event_mask, "bucket"] = base.loc[event_mask, "bucket"].where(base.loc[event_mask, "bucket"]!="other", "event")

    # 9) Compléter ticker_tv si absent
    if "ticker_tv" not in base.columns:
        base["ticker_tv"] = base["ticker_yf"].map(_yf_symbol_to_tv)

    # 10) Exports
    export_cols = [
        "ticker_yf","ticker_tv","price","mcap_usd","avg_dollar_vol",
        "tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry","technical_local","tech_score",
        "p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","event_score","bucket"
    ]
    for c in export_cols:
        if c not in base.columns: base[c] = None

    cand_all = base[export_cols].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)
    _save(cand_all, "candidates_all_ranked.csv")

    confirmed = cand_all[cand_all["bucket"] == "confirmed"].copy().sort_values("rank_score", ascending=False)
    pre       = cand_all[cand_all["bucket"] == "pre_signal"].copy().sort_values("rank_score", ascending=False)
    events    = cand_all[cand_all["bucket"] == "event"].copy().sort_values(["event_score","rank_score","avg_dollar_vol"], ascending=[False,False,False])

    # Tops exacts
    top10_confirmed = confirmed.head(10).copy()
    top50_pre       = pre.head(50).copy()
    top100_event    = events.head(100).copy()

    _save(confirmed, "confirmed_STRONGBUY.csv")
    _save(pre, "anticipative_pre_signals.csv")
    _save(events, "event_driven_signals.csv")
    _save(top10_confirmed, "top10_confirmed.csv")
    _save(top50_pre, "top50_pre.csv")
    _save(top100_event, "top100_event.csv")

    # 11) signals_history (pour backtest) — inclut le pilier technique
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sig_hist = cand_all[["ticker_yf","sector","bucket","tv_reco","analyst_bucket","technical_local"]].copy()
    sig_hist.insert(0, "date", today)
    _save(sig_hist, "signals_history.csv")

    # 12) Couverture / Diagnostics
    cov = {
        "universe": int(len(base)),
        "topN_enriched": int(len(cand_all)),
        "tv_reco_filled": int(cand_all["tv_reco"].notna().sum()),
        "finnhub_filled": int(cand_all["analyst_bucket"].notna().sum()),
        "tech_pillar_filled": int(cand_all["technical_local"].notna().sum()),
        "sector_known": int((cand_all["sector"].astype(str) != "Unknown").sum()),
        "confirmed_count": int(len(confirmed)),
        "pre_count": int(len(pre)),
        "event_count": int(len(events)),
    }
    print("[COVERAGE]", cov)

if __name__ == "__main__":
    main()
