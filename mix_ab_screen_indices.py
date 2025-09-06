# mix_ab_screen_indices.py — v3.0
# Pipeline: charge l'univers (scopé secteurs), calcule TA pour tout l'in-scope (=> last_close & dv20),
# sélectionne TopN par dv20, enrichit TV+analystes, calcule MCAP via Finnhub profile2, ENFORCE <75B,
# remplit/complète le panel pour garder N noms, score & exports.

import os, sys, json, time, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf

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

TOP_N = int(os.getenv("SCREEN_TOPN", "180"))
TV_BATCH = 80
REQ_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.20

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "")

MIN_LIQ_CONFIRMED = float(os.getenv("MIN_LIQ_CONFIRM_USD", "5000000"))
MIN_ANALYST_VOTES = int(os.getenv("MIN_ANALYST_VOTES", "10"))

TARGET_SECTORS = {"information technology","financials","industrials","health care"}
MCAP_MAX_USD   = 75_000_000_000

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
    u = str(label).upper().strip().replace(" ","_")
    if u in {"BUY","STRONG_BUY","OUTPERFORM","OVERWEIGHT"}: return "BUY"
    if u in {"SELL","STRONG_SELL","UNDERPERFORM","UNDERWEIGHT"}: return "SELL"
    if u in {"HOLD","NEUTRAL"}: return "HOLD"
    return "HOLD"

# ---- TradingView scan (recommend) ----
def tv_scan_batch(symbols_tv: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if not symbols_tv: return out
    url = "https://scanner.tradingview.com/america/scan"
    payload = {
        "symbols":{"tickers":[f"NASDAQ:{s}" for s in symbols_tv] + [f"NYSE:{s}" for s in symbols_tv] + [f"AMEX:{s}" for s in symbols_tv],"query":{"types":[]}},
        "columns":["Recommend.All"]
    }
    try:
        r = requests.post(url, json=payload, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            print(f"[TV] HTTP {r.status_code} -> skip batch"); return out
        data = r.json()
        for row in data.get("data", []):
            sym = row.get("s",""); base = sym.split(":")[-1]
            vals = row.get("d",[]); tv_score = _safe_num(vals[0]) if vals else None
            tv_reco = _label_from_tv_score(tv_score)
            if not base: continue
            prev = out.get(base)
            if prev is None or (tv_score is not None and (prev["tv_score"] is None or tv_score > prev["tv_score"])):
                out[base] = {"tv_score": tv_score, "tv_reco": tv_reco}
        print(f"[TV] filled {len(out)}/{len(symbols_tv)}")
    except Exception as e:
        print(f"[TV] exception: {e}")
    return out

# ---- Finnhub helpers ----
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

def finnhub_profile2_mcap(symbol_yf: str) -> float | None:
    """Return market cap in USD via profile2 (value is in billions)."""
    if not FINNHUB_KEY: return None
    try:
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol_yf}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200: return None
        j = r.json()
        val = j.get("marketCapitalization")
        return float(val)*1e9 if val is not None else None
    except Exception:
        return None

# ---- TA ----
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0); down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _compute_one_ta(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 50:
        return {"technical_local": None, "tech_score": None, "event_score": 0, "last_close": None, "dv20": None}
    c = df["Close"].astype(float); v = df["Volume"].astype(float)
    last_close = float(c.iloc[-1])
    dv20 = float((c.tail(20) * v.tail(20)).mean()) if len(c) >= 20 else None

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

    event_score = 0
    if len(c) >= 21:
        pct1d = (c.iloc[-1] / c.iloc[-2] - 1.0)
        vol_ratio = (v.iloc[-1] / max(1.0, v.tail(20).mean())) if v.tail(20).mean() else 0.0
        breakout20 = c.iloc[-1] >= c.tail(20).max()
        gap_up = (df["Open"].iloc[-1] / c.iloc[-2] - 1.0) if "Open" in df.columns else None
        event_score += int(pct1d >= 0.05) + int(vol_ratio >= 1.8) + int(breakout20) + int(gap_up is not None and gap_up >= 0.03)

    return {"technical_local": label, "tech_score": tech_score, "event_score": event_score, "last_close": last_close, "dv20": dv20}

def compute_technicals_batch(symbols_yf: List[str], batch_size: int = 120) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i in range(0, len(symbols_yf), batch_size):
        chunk = [s for s in symbols_yf[i:i+batch_size] if s]
        if not chunk: continue
        try:
            df = yf.download(tickers=chunk, period="1y", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                for sym in chunk:
                    try:
                        sub = df[sym][["Open","High","Low","Close","Volume"]].dropna()
                        out[sym] = _compute_one_ta(sub)
                    except Exception:
                        out[sym] = {"technical_local": None, "tech_score": None, "event_score": 0, "last_close": None, "dv20": None}
            else:
                try:
                    sub = df[["Open","High","Low","Close","Volume"]].dropna()
                    out[chunk[0]] = _compute_one_ta(sub)
                except Exception:
                    out[chunk[0]] = {"technical_local": None, "tech_score": None, "event_score": 0, "last_close": None, "dv20": None}
        except Exception as e:
            print(f"[TA] batch exception: {e}")
        time.sleep(0.35)
    print(f"[TA] computed: {len(out)}/{len(symbols_yf)}")
    return out

def _load_universe() -> pd.DataFrame:
    if Path("universe_in_scope.csv").exists():
        u = pd.read_csv("universe_in_scope.csv")
        if len(u) > 0:
            print(f"[UNI] in-scope: {len(u)}"); return u
        print("[UNI] in-scope empty → fallback raw_universe.csv")
    if Path("raw_universe.csv").exists():
        u = pd.read_csv("raw_universe.csv")
        print(f"[UNI] raw_universe: {len(u)}"); return u
    print("❌ No universe files"); sys.exit(2)

def main():
    base = _load_universe()
    # Merge catalog si besoin
    if "sector" not in base.columns or base["sector"].astype(str).str.strip().eq("").all():
        if Path("sector_catalog.csv").exists():
            cat = pd.read_csv("sector_catalog.csv").rename(columns={"ticker":"ticker_yf"})
            base = base.merge(cat[["ticker_yf","sector","industry"]], on="ticker_yf", how="left")
    base["sector"] = base.get("sector","Unknown").fillna("Unknown").astype(str).str.lower()
    base = base[base["sector"].isin(TARGET_SECTORS)].copy()

    # ----- TA pour tout l'in-scope -> on a dv20 & last_close -----
    base["ticker_yf"] = base["ticker_yf"].astype(str)
    ta_map = compute_technicals_batch(base["ticker_yf"].tolist(), batch_size=120)
    base["last_close"] = [ta_map.get(s,{}).get("last_close") for s in base["ticker_yf"]]
    base["dv20"]       = [ta_map.get(s,{}).get("dv20") for s in base["ticker_yf"]]
    base["technical_local"] = [ta_map.get(s,{}).get("technical_local") for s in base["ticker_yf"]]
    base["tech_score"]      = [ta_map.get(s,{}).get("tech_score") for s in base["ticker_yf"]]
    base["event_score"]     = [ta_map.get(s,{}).get("event_score") for s in base["ticker_yf"]]

    base["dv20"] = pd.to_numeric(base["dv20"], errors="coerce")
    base = base.sort_values("dv20", ascending=False, na_position="last").reset_index(drop=True)

    # ----- sélection progressive pour garantir N < 75B -----
    picked_rows = []
    start_idx = 0
    while len(picked_rows) < TOP_N and start_idx < len(base):
        # on prend un bloc (pour limiter les appels) — taille adaptative
        block = base.iloc[start_idx:start_idx + max(200, TOP_N)].copy()
        start_idx += max(200, TOP_N)

        # enrichissements TV + analystes + MCAP (Finnhub)
        syms_tv = [ _yf_symbol_to_tv(s) for s in block["ticker_yf"] ]
        tv_out: Dict[str, Dict] = {}
        for i in range(0, len(syms_tv), TV_BATCH):
            chunk = [s for s in syms_tv[i:i+TV_BATCH] if s]
            if not chunk: continue
            tv_out.update(tv_scan_batch(chunk))
            time.sleep(SLEEP_BETWEEN_CALLS)
        block["tv_score"] = [ tv_out.get(_yf_symbol_to_tv(s), {}).get("tv_score") for s in block["ticker_yf"] ]
        block["tv_reco"]  = [ tv_out.get(_yf_symbol_to_tv(s), {}).get("tv_reco")  for s in block["ticker_yf"] ]

        an_bucket, an_votes = [], []
        for i, sym in enumerate(block["ticker_yf"], 1):
            b, v = finnhub_analyst(sym); an_bucket.append(b); an_votes.append(v)
            if i % 80 == 0: print(f"[FINNHUB] {i}/{len(block)}")
            time.sleep(SLEEP_BETWEEN_CALLS)
        block["analyst_bucket"] = an_bucket
        block["analyst_votes"]  = an_votes

        # MCAP via Finnhub profile2 (rapide)
        mcaps = []
        for i, sym in enumerate(block["ticker_yf"], 1):
            mcaps.append(finnhub_profile2_mcap(sym))
            if i % 100 == 0: print(f"[FINN-MCAP] {i}/{len(block)}")
            time.sleep(0.12 if FINNHUB_KEY else 0)  # throttle léger
        block["mcap_usd"] = mcaps

        # price = last_close
        block["price"] = block["last_close"]

        # enforce < 75B
        m_ok = pd.to_numeric(block["mcap_usd"], errors="coerce").fillna(float("inf")) < MCAP_MAX_USD
        block_ok = block[m_ok].copy()

        # on ajoute tant qu'on n'a pas TOP_N
        remaining = TOP_N - len(picked_rows)
        if remaining > 0:
            picked_rows.append(block_ok.head(remaining))

    if picked_rows:
        cand = pd.concat(picked_rows, ignore_index=True)
    else:
        cand = base.head(TOP_N).copy()
        cand["price"] = cand["last_close"]
        cand["mcap_usd"] = None
        cand["tv_score"] = None; cand["tv_reco"] = None
        cand["analyst_bucket"] = None; cand["analyst_votes"] = None

    # Post-process scoring & buckets
    def is_strong_tv(v) -> bool:   return (str(v).upper().strip()=="STRONG_BUY") if v is not None and not _is_nan(v) else False
    def is_bull_an(v) -> bool:     return _bucket_from_analyst(v) == "BUY"
    def is_strong_tech(v) -> bool: return (str(v).upper().strip()=="STRONG_BUY") if v is not None and not _is_nan(v) else False

    votes = pd.to_numeric(cand.get("analyst_votes", 0), errors="coerce")
    cand["p_tv"]   = cand.get("tv_reco", pd.Series(index=cand.index)).map(is_strong_tv)
    cand["p_an"]   = cand.get("analyst_bucket", pd.Series(index=cand.index)).map(is_bull_an)
    cand["p_tech"] = cand.get("technical_local", pd.Series(index=cand.index)).map(is_strong_tech)
    cand["pillars_met"] = cand[["p_tech","p_tv","p_an"]].fillna(False).sum(axis=1)
    cand["votes_bin"] = pd.cut(votes.fillna(-1), bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])

    tv  = pd.to_numeric(cand.get("tv_score", 0), errors="coerce").fillna(0)
    tech= pd.to_numeric(cand.get("tech_score", 0), errors="coerce").fillna(0)
    v_n = votes.fillna(0).clip(lower=0, upper=30) / 30.0
    cand["rank_score"] = (tv*0.45) + (tech*0.35) + (v_n*0.20)

    liq_ok   = pd.to_numeric(cand.get("dv20", 0), errors="coerce").fillna(0) >= MIN_LIQ_CONFIRMED
    votes_ok = votes.fillna(0) >= MIN_ANALYST_VOTES
    tv_up   = cand.get("tv_reco", pd.Series(index=cand.index)).astype(str).str.upper()
    an_up   = cand.get("analyst_bucket", pd.Series(index=cand.index)).map(lambda x: (_bucket_from_analyst(x) or ""))
    tech_up = cand.get("technical_local", pd.Series(index=cand.index)).astype(str).str.upper()

    cond_confirmed = tv_up.eq("STRONG_BUY") & (an_up == "BUY") & tech_up.isin(["STRONG_BUY","BUY"]) & liq_ok & votes_ok
    cond_pre       = tv_up.isin(["STRONG_BUY","BUY"]) & tech_up.isin(["STRONG_BUY","BUY"]) & (~cond_confirmed)

    cand["bucket"] = "other"
    cand.loc[cond_pre, "bucket"] = "pre_signal"
    cand.loc[cond_confirmed, "bucket"] = "confirmed"
    event_mask = (pd.to_numeric(cand.get("event_score", 0), errors="coerce").fillna(0) >= 2) & (~cond_pre) & (~cond_confirmed)
    cand.loc[event_mask, "bucket"] = cand.loc[event_mask, "bucket"].where(cand.loc[event_mask, "bucket"]!="other", "event")

    if "ticker_tv" not in cand.columns:
        cand["ticker_tv"] = cand["ticker_yf"].map(_yf_symbol_to_tv)

    export_cols = ["ticker_yf","ticker_tv","price","mcap_usd","dv20","tv_score","tv_reco",
                   "analyst_bucket","analyst_votes","sector","industry","technical_local",
                   "tech_score","p_tech","p_tv","p_an","pillars_met","votes_bin",
                   "rank_score","event_score","bucket"]
    for c in export_cols:
        if c not in cand.columns: cand[c] = None

    cand_all = cand[export_cols].copy().sort_values("rank_score", ascending=False).reset_index(drop=True)
    _save(cand_all, "candidates_all_ranked.csv")

    confirmed = cand_all[cand_all["bucket"]=="confirmed"].copy().sort_values("rank_score", ascending=False)
    pre       = cand_all[cand_all["bucket"]=="pre_signal"].copy().sort_values("rank_score", ascending=False)
    events    = cand_all[cand_all["bucket"]=="event"].copy().sort_values(["event_score","rank_score","dv20"], ascending=[False,False,False])

    _save(confirmed, "confirmed_STRONGBUY.csv")
    _save(pre, "anticipative_pre_signals.csv")
    _save(events, "event_driven_signals.csv")
    _save(confirmed.head(10), "top10_confirmed.csv")
    _save(pre.head(50), "top50_pre.csv")
    _save(events.head(100), "top100_event.csv")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sig_hist = cand_all[["ticker_yf","sector","bucket","tv_reco","analyst_bucket","technical_local"]].copy()
    sig_hist.insert(0, "date", today)
    _save(sig_hist, "signals_history.csv")

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
        "mcap_nonnull": int(cand_all["mcap_usd"].notna().sum()),
        "price_nonnull": int(cand_all["price"].notna().sum())
    }
    print("[COVERAGE]", cov)

if __name__ == "__main__":
    main()
