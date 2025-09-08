# mix_ab_screen_indices.py
# Build a compact, fully-enriched screening set and publish rich CSVs for the dashboard.

from __future__ import annotations
import os, json, time, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import yfinance as yf

from cache_layer import load_map, save_map, set_cached, now_utc_iso

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# --------- Config ---------
TOP_N = int(os.getenv("SCREEN_TOPN", "180"))   # top by liquidity to enrich heavily
TV_BATCH = 80
REQ_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.25

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "")

MCAP_MAX_USD = 75_000_000_000

# --------- I/O helpers ---------
def _save(df: pd.DataFrame, name: str, also_public: bool = True):
    try:
        p = Path(name)
        df.to_csv(p, index=False)
        if also_public:
            q = PUBLIC_DIR / name
            df.to_csv(q, index=False)
        print(f"[SAVE] {name} | rows={len(df)} | cols={len(df.columns)}")
    except Exception as e:
        print(f"[SAVE] failed for {name}: {e}")

def _read_csv_required(name: str) -> pd.DataFrame:
    for base in (Path("."), PUBLIC_DIR):
        p = base / name
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    raise SystemExit(f"❌ Required file not found or unreadable: {name}")

# --------- utils ---------
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

def _label_from_tv_score(score: float | None) -> Optional[str]:
    if score is None: return None
    if score >= 0.5:  return "STRONG_BUY"
    if score >= 0.2:  return "BUY"
    if score <= -0.5: return "STRONG_SELL"
    if score <= -0.2: return "SELL"
    return "NEUTRAL"

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

def finnhub_analyst(symbol_yf: str) -> Tuple[Optional[str], Optional[int]]:
    if not FINNHUB_KEY: return (None, None)
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol_yf}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200: return (None, None)
        arr = r.json()
        if not isinstance(arr, list) or not arr: return (None, None)
        rec = arr[0]
        buy  = int(rec.get("buy", 0))
        hold = int(rec.get("hold", 0))
        sell = int(rec.get("sell", 0))
        votes = buy + hold + sell
        if buy > max(hold, sell):
            return ("BUY", votes)
        if sell > max(hold, buy):
            return ("SELL", votes)
        return ("HOLD", votes)
    except Exception:
        return (None, None)

def finnhub_profile_mcap(symbol_yf: str) -> Optional[float]:
    if not FINNHUB_KEY: return None
    # cached
    prof_cache = load_map("finnhub_profile", ttl_hours=24)
    if symbol_yf in prof_cache and "mcap_usd" in prof_cache[symbol_yf]:
        return prof_cache[symbol_yf]["mcap_usd"]
    try:
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol_yf}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200: return None
        data = r.json()
        m = _safe_num(data.get("marketCapitalization"))
        m_usd = float(m) * 1_000_000 if m is not None else None
        prof_cache[symbol_yf] = {"mcap_usd": m_usd, "cached_at": now_utc_iso()}
        save_map("finnhub_profile", prof_cache)
        return m_usd
    except Exception:
        return None

def yf_price_fast(symbol_yf: str) -> Optional[float]:
    # cached fast_info.last_price
    fi_cache = load_map("yf_fastinfo", ttl_hours=24)
    rec = fi_cache.get(symbol_yf)
    if rec and rec.get("last_price") is not None:
        return _safe_num(rec.get("last_price"))
    try:
        info = yf.Ticker(symbol_yf).fast_info
        p = getattr(info, "last_price", None)
        lp = float(p) if p is not None else None
        fi_cache[symbol_yf] = {
            "last_price": lp,
            "market_cap": getattr(info, "market_cap", None),
            "ten_day_average_volume": getattr(info, "ten_day_average_volume", None),
            "cached_at": now_utc_iso()
        }
        save_map("yf_fastinfo", fi_cache)
        return lp
    except Exception:
        return None

def yf_avg_dollar_vol(tickers: List[str]) -> Dict[str, float]:
    """20d avg dollar volume via history. Cached per ticker (dv20)."""
    out: Dict[str, float] = {}
    dv_cache = load_map("dv20_cache", ttl_hours=72)
    to_fetch: List[str] = []

    for t in tickers:
        rec = dv_cache.get(t)
        if rec and rec.get("dv20") is not None:
            out[t] = float(rec["dv20"])
        else:
            to_fetch.append(t)

    if to_fetch:
        try:
            df = yf.download(tickers=to_fetch, period="3mo", interval="1d",
                             auto_adjust=True, progress=False, group_by="ticker", threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                for t in to_fetch:
                    try:
                        sub = df[t][["Close","Volume"]].dropna()
                        if len(sub) >= 5:
                            tail = sub.tail(20)
                            dv20 = float((tail["Close"] * tail["Volume"]).mean())
                            out[t] = dv20
                            dv_cache[t] = {"dv20": dv20, "asof": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
                    except Exception:
                        pass
            else:
                # single ticker case
                sub = df[["Close","Volume"]].dropna()
                if len(sub) >= 5:
                    dv20 = float((sub.tail(20)["Close"] * sub.tail(20)["Volume"]).mean())
                    t = to_fetch[0]
                    out[t] = dv20
                    dv_cache[t] = {"dv20": dv20, "asof": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
        except Exception:
            pass
        save_map("dv20_cache", dv_cache)
    return out

def _bucket_from_analyst(v: Optional[str]) -> Optional[str]:
    if not v: return None
    s = str(v).strip().upper()
    if s == "BUY": return "BUY"
    if s == "SELL": return "SELL"
    return "HOLD"

def _fill_mcap_for_rows(tickers: List[str], prices: List[Optional[float]], sleep=0.1) -> List[Optional[float]]:
    """Fill missing market caps (USD) using yfinance.info or sharesOutstanding * price."""
    out: List[Optional[float]] = []
    yf_info_cache = load_map("yf_info_mcap", ttl_hours=24)
    for sym, p in zip(tickers, prices):
        val = None
        if sym in yf_info_cache and "mcap" in yf_info_cache[sym]:
            val = yf_info_cache[sym]["mcap"]
        else:
            try:
                info = (yf.Ticker(sym).info or {})
                val = info.get("marketCap")
                if val is None:
                    sh = info.get("sharesOutstanding")
                    if sh is not None and p is not None:
                        try:
                            val = float(sh) * float(p)
                        except Exception:
                            val = None
                if val is not None:
                    val = float(val)
                yf_info_cache[sym] = {"mcap": val, "cached_at": now_utc_iso()}
            except Exception:
                pass
            time.sleep(sleep)
        out.append(val)
    save_map("yf_info_mcap", yf_info_cache)
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
            for col in df.columns:
                if col.lower() == n.lower():
                    return col
        return None

    sec_col = pickcol(cat, ["sector","gics_sector","gics sector"])
    ind_col = pickcol(cat, ["industry","gics_industry","gics industry"])
    tic_col = pickcol(cat, ["ticker_yf","symbol","ticker","code"])

    if sec_col: cat = cat.rename(columns={sec_col: "sector"})
    if ind_col: cat = cat.rename(columns={ind_col: "industry"})
    if tic_col: cat = cat.rename(columns={tic_col: "ticker_yf"})

    if "ticker_yf" not in cat.columns:
        raise SystemExit("❌ sector_catalog.csv must have a ticker column (e.g. symbol/ticker_yf)")

    base = base.merge(cat[["ticker_yf","sector","industry"]].drop_duplicates(),
                      on="ticker_yf", how="left")

    # 3) Avg dollar vol (fallback if not present)
    if "avg_dollar_vol" not in base.columns or base["avg_dollar_vol"].isna().all():
        syms = base["ticker_yf"].astype(str).tolist()
        print("[YF] computing 20d avg dollar vol…")
        avgd = yf_avg_dollar_vol(syms)
        base["avg_dollar_vol"] = base["ticker_yf"].map(avgd).astype(float)

    # 4) Choose TopN to enrich heavily
    base_sorted = base.sort_values("avg_dollar_vol", ascending=False, na_position="last").reset_index(drop=True)
    cand = base_sorted.head(TOP_N).copy()
    print(f"[UNI] in-scope: {len(base)}  |  TopN to enrich: {len(cand)}")

    # 5) Prices (fast) for cand; cached; fallback via history later if needed
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
        m = finnhub_profile_mcap(sym); mcaps.append(m)
        if i % 80 == 0: print(f"[Finnhub] {i}/{len(cand)}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    cand["analyst_bucket"]  = an_bucket
    cand["analyst_votes"]   = an_votes
    cand["mcap_usd_finnhub"] = mcaps

    # 8) yfinance.info mcap for cand (fallback)
    mcap_yf = []
    fi_cache = load_map("yf_fastinfo", ttl_hours=24)
    for i, sym in enumerate(cand["ticker_yf"], 1):
        # use fast_info.market_cap if present in cache to save one info() call
        val = None
        rec = fi_cache.get(sym)
        if rec and rec.get("market_cap") is not None:
            val = _safe_num(rec.get("market_cap"))
        if val is None:
            try:
                info = (yf.Ticker(sym).info or {})
                val = _safe_num(info.get("marketCap"))
            except Exception:
                val = None
        mcap_yf.append(val)
        time.sleep(SLEEP_BETWEEN_CALLS/2)
    cand["mcap_usd_yfinfo"] = mcap_yf

    # 9) Build final enrichment set and merge back to base
    cand["mcap_usd_final"] = pd.to_numeric(cand["mcap_usd_finnhub"], errors="coerce") \
                                .fillna(pd.to_numeric(cand["mcap_usd_yfinfo"], errors="coerce"))

    # ensure sector/industry exist in cand
    for col in ("sector", "industry"):
        if col not in cand.columns:
            if col in base.columns:
                cand = cand.merge(base[["ticker_yf", col]].drop_duplicates(), on="ticker_yf", how="left")
            else:
                cand[col] = None

    wanted = ["ticker_yf","price","tv_score","tv_reco",
              "analyst_bucket","analyst_votes","mcap_usd_final",
              "sector","industry","avg_dollar_vol"]
    enrich_cols = [c for c in wanted if c in cand.columns]
    enrich_df = cand[enrich_cols].copy()
    base = base.merge(enrich_df, on="ticker_yf", how="left", suffixes=("", "_cand"))

    # reconcile sector/industry
    for col in ("sector","industry"):
        rcol = f"{col}_cand"
        if rcol in base.columns:
            take = base[rcol].where(base[rcol].astype(str).str.strip() != "", base[col])
            base[col] = base[col].where(base[col].notna(), take)
            base.drop(columns=[rcol], inplace=True)
        elif col not in base.columns:
            base[col] = None

    # 9-bis) Fill missing market caps across the whole base (not only cand)
    if "price" not in base.columns:
        base["price"] = None
    need_mask = base["mcap_usd_final"].isna()
    if need_mask.any():
        mcap_filled = _fill_mcap_for_rows(
            base.loc[need_mask, "ticker_yf"].astype(str).tolist(),
            base.loc[need_mask, "price"].tolist(),
            sleep=SLEEP_BETWEEN_CALLS/2
        )
        base.loc[need_mask, "mcap_usd_final"] = pd.to_numeric(pd.Series(mcap_filled), errors="coerce")

    # 10) Tech pillars
    def is_strong_tv(v) -> bool:
        if v is None or _is_nan(v): return False
        return str(v).upper().strip() == "STRONG_BUY"
    def is_bull_an(v) -> bool:
        return _bucket_from_analyst(v) == "BUY"

    base["p_tv"]   = base.get("tv_reco").map(is_strong_tv)
    base["p_an"]   = base.get("analyst_bucket").map(is_bull_an)
    # pour l’instant p_tech = p_tv (visible dans CSV/front)
    base["p_tech"] = base["p_tv"]
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
    base = base[pd.to_numeric(base["mcap_usd_final"], errors="coerce").notna() &
                (base["mcap_usd_final"] < MCAP_MAX_USD)].copy()

    # 13) Buckets exclusifs (confirmed > pre_signal > event)
    tv_up = base.get("tv_reco").astype(str).str.upper()
    an_up = base.get("analyst_bucket").map(lambda x: (_bucket_from_analyst(x) or ""))

    is_confirmed = tv_up.eq("STRONG_BUY") & (an_up == "BUY")
    is_pre       = tv_up.isin(["STRONG_BUY","BUY"]) & (~is_confirmed)
    is_event     = ~is_confirmed & ~is_pre  # placeholder pour exploration

    # Tri global par score
    base_sorted2 = base.sort_values("rank_score", ascending=False).reset_index(drop=True)

    # Sélection exclusive : on retire les tickers déjà pris d’une catégorie quand on passe à la suivante
    df_conf = base_sorted2[is_confirmed.reindex(base_sorted2.index, fill_value=False)].copy()
    already = set(df_conf["ticker_yf"])

    df_pre  = base_sorted2[
        is_pre.reindex(base_sorted2.index, fill_value=False)
        & (~base_sorted2["ticker_yf"].isin(already))
    ].copy()
    already.update(df_pre["ticker_yf"])

    df_evt  = base_sorted2[
        is_event.reindex(base_sorted2.index, fill_value=False)
        & (~base_sorted2["ticker_yf"].isin(already))
    ].copy()

    # Aliases + colonnes export
    base_sorted2["ticker_tv"] = base_sorted2.get("ticker_tv") \
        if "ticker_tv" in base_sorted2.columns else base_sorted2["ticker_yf"].map(_yf_symbol_to_tv)
    base_sorted2["last"] = base_sorted2.get("price")
    base_sorted2["mcap"] = base_sorted2.get("mcap_usd_final")

    export_cols = [
        "ticker_yf","ticker_tv",
        "price","last",
        "mcap_usd_final","mcap",
        "avg_dollar_vol",
        "tv_score","tv_reco","analyst_bucket","analyst_votes",
        "sector","industry",
        "p_tech","p_tv","p_an","pillars_met","votes_bin",
        "rank_score","bucket"
    ]
    for c in export_cols:
        if c not in base_sorted2.columns: base_sorted2[c] = None

    # Ecriture candidates global
    cand_all = base_sorted2[export_cols[:-1]].copy()  # sans 'bucket'
    _save(cand_all, "candidates_all_ranked.csv")

    # Buckets exclusifs et tailles cibles
    conf_out = df_conf.assign(bucket="confirmed")[export_cols].head(10)
    pre_out  = df_pre.assign(bucket="pre_signal")[export_cols].head(50)
    # event : reste (top 100 uniques au total)
    remaining_for_events = max(0, 100 - len(conf_out) - len(pre_out))
    evt_out = df_evt.assign(bucket="event")[export_cols].head(remaining_for_events)

    _save(conf_out, "confirmed_STRONGBUY.csv")
    _save(pre_out, "anticipative_pre_signals.csv")
    _save(evt_out, "event_driven_signals.csv")

    # 14) Minimal signals_history pour backtest
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sig_hist = pd.concat([conf_out, pre_out, evt_out], ignore_index=True)
    sig_hist = sig_hist[["ticker_yf","sector","bucket","tv_reco","analyst_bucket"]].copy()
    sig_hist.insert(0, "date", today)
    _save(sig_hist, "signals_history.csv")

    cov = {
        "universe": int(len(base_sorted2)),
        "tv_reco_filled": int(cand_all["tv_reco"].notna().sum()),
        "finnhub_analyst": int(cand_all["analyst_bucket"].notna().sum()),
        "mcap_final_nonnull": int(cand_all["mcap_usd_final"].notna().sum()),
        "price_nonnull": int(cand_all["price"].notna().sum()),
        "confirmed_count": int(len(conf_out)),
        "pre_count": int(len(pre_out)),
        "event_count": int(len(evt_out)),
        "unique_total": int(sig_hist["ticker_yf"].nunique())
    }
    print("[COVERAGE]", cov)

if __name__ == "__main__":
    main()
