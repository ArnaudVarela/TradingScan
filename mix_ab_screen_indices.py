# mix_ab_screen_indices.py
# Pipeline screener autonome (sans providers payants)
# - OHLCV via yfinance + cache parquet/csv
# - TechScore local (pandas-ta) -> tv_reco/tv_score
# - Proxy fondamentaux via SEC (companyfacts) -> analyst_label/score/votes
# - MCAP < 75B strict (rejette NaN)
# - Sorties à la racine du repo

from __future__ import annotations
import os, math, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import yfinance as yf

# modules locaux (déjà fournis dans ton repo)
from sec_helpers import sec_ticker_map, sec_latest_shares_outstanding, _sec_get_json
from tech_score import compute_tech_features, tech_label_from_features
from analyst_proxy import analyst_label_from_fundamentals

# --- chemins
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUT_DIR = ROOT                                # sorties à la RACINE
UNIVERSE_CSV = ROOT / "universe_in_scope.csv" # fourni par build_universe.py
SECTOR_CATALOG = ROOT / "sector_catalog.csv"  # mapping ticker->sector/industry (si dispo)

# --- paramètres
MCAP_CAP = 75e9                      # filtre < 75B (strict)
MIN_PRICE_FOR_SCORE = 0.5            # ignorer penny stocks pour TechScore
OHLCV_WINDOW_DAYS = 240              # fenêtre OHLCV
AVG_DOLLAR_VOL_LOOKBACK = 20         # moyenne $ volume 20j
YF_TIMEOUT = 15

# ========================================================= Utils

def _log(msg: str):
    print(msg, flush=True)

def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)): 
            return None
        return float(x)
    except Exception:
        return None

def read_sector_catalog() -> Dict[str, Dict[str,str]]:
    if not SECTOR_CATALOG.exists():
        return {}
    try:
        df = pd.read_csv(SECTOR_CATALOG)
        res: Dict[str, Dict[str,str]] = {}
        for _, r in df.iterrows():
            t = str(r.get("ticker") or r.get("ticker_yf") or "").upper()
            if not t:
                continue
            sector = r.get("sector")
            industry = r.get("industry")
            res[t] = {
                "sector": sector if pd.notna(sector) else "Unknown",
                "industry": industry if pd.notna(industry) else "Unknown",
            }
        return res
    except Exception as e:
        _log(f"[WARN] sector_catalog read failed: {e}")
        return {}

# ========================================================= OHLCV cache parquet/csv (fallback)

def _p_parquet(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.parquet"

def _p_csv(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.csv"

def _read_cached_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    pp = _p_parquet(ticker)
    pc = _p_csv(ticker)
    if pp.exists():
        try:
            return pd.read_parquet(pp)
        except Exception:
            pass
    if pc.exists():
        try:
            return pd.read_csv(pc, parse_dates=["Date"], index_col="Date")
        except Exception:
            pass
    return None

def _write_cached_ohlcv(ticker: str, df: pd.DataFrame):
    pp = _p_parquet(ticker)
    pc = _p_csv(ticker)
    # tente parquet
    try:
        df.to_parquet(pp, index=True)
        if pc.exists():
            try: pc.unlink()
            except Exception: pass
        return
    except Exception:
        pass
    # fallback CSV
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(pc, index=True)

def fetch_ohlcv_yf(ticker: str, period_days: int = OHLCV_WINDOW_DAYS) -> Optional[pd.DataFrame]:
    """
    OHLCV (daily) via yfinance avec cache. Colonnes: open, high, low, close, volume
    """
    df = _read_cached_ohlcv(ticker)

    need_fetch = True
    if df is not None and len(df) >= 50:
        try:
            last = pd.to_datetime(df.index[-1])
            # si cache < 2 jours, on réutilise
            if (pd.Timestamp.utcnow().tz_localize("UTC") - last.tz_localize("UTC")).days < 2:
                need_fetch = False
        except Exception:
            pass

    if need_fetch:
        try:
            period = f"{max(period_days, 120)}d"
            yf_df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if yf_df is None or yf_df.empty:
                return df
            yf_df = yf_df.rename(
                columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
            )[["open","high","low","close","volume"]].dropna()
            yf_df.index = pd.to_datetime(yf_df.index)
            _write_cached_ohlcv(ticker, yf_df)
            df = yf_df
        except Exception as e:
            _log(f"[YF] fetch failed {ticker}: {e}")
            return df

    return df

def last_close(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty:
        return None
    return safe_float(df["close"].iloc[-1])

def avg_dollar_vol(df: Optional[pd.DataFrame], lookback: int = AVG_DOLLAR_VOL_LOOKBACK) -> Optional[float]:
    if df is None or len(df) < lookback:
        return None
    sub = df.iloc[-lookback:]
    return float((sub["close"] * sub["volume"]).mean())

# ========================================================= SEC companyfacts cache light

FACTS_PATH = CACHE_DIR / "sec_companyfacts_cache.json"

def _facts_load():
    try:
        with FACTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"_ts": 0, "data": {}}

def _facts_save(data):
    tmp = FACTS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(FACTS_PATH)

def sec_companyfacts(cik10: str, ttl_hours: float = 48.0) -> Optional[dict]:
    cache = _facts_load()
    now = time.time()
    key = f"CIK{cik10}"
    rec = cache["data"].get(key)
    if rec and now - rec.get("_ts", 0) < ttl_hours * 3600:
        return rec.get("facts")
    try:
        facts = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    except Exception:
        return rec.get("facts") if rec else None
    cache["data"][key] = {"_ts": now, "facts": facts}
    _facts_save(cache)
    return facts

def _series_from_facts(facts: dict, ns: str, tag: str, prefer_units=("USD","USD/shares")) -> List[float]:
    try:
        units = facts["facts"][ns][tag]["units"]
    except Exception:
        return []
    # choisir d'abord les unités “meilleures”
    unit_keys = sorted(
        units.keys(),
        key=lambda u: 0 if any(p.lower() in u.lower() for p in prefer_units) else 1
    )
    arr: List[float] = []
    for uk in unit_keys:
        a = units[uk]
        a_sorted = sorted(a, key=lambda x: (x.get("fy",0), x.get("fp",""), x.get("end","")))
        vals: List[float] = []
        for obs in a_sorted:
            v = obs.get("val")
            try:
                if v is not None:
                    vals.append(float(v))
            except Exception:
                continue
        if len(vals) >= 2:
            arr = vals
            break
        elif not arr and len(vals) > 0:
            arr = vals
    return arr

# ========================================================= Pipeline

def ensure_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise SystemExit(f"[FATAL] {UNIVERSE_CSV} introuvable. Lance d'abord build_universe.py")

    df = pd.read_csv(UNIVERSE_CSV)
    col = None
    for c in ["ticker_yf", "ticker", "Symbol", "symbol"]:
        if c in df.columns:
            col = c
            break
    if not col:
        raise SystemExit("[FATAL] universe_in_scope.csv: colonne ticker introuvable")

    df = df.rename(columns={col: "ticker_yf"})
    df["ticker_yf"] = df["ticker_yf"].astype(str).str.strip().str.upper()
    df = df[df["ticker_yf"].ne("") & df["ticker_yf"].ne("-")]
    df = df[df["ticker_yf"].str.match(r"^[A-Z0-9.\-]+$")]
    df = df.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)

    _log(f"[UNI] in-scope: {len(df)}")
    return df

def _yfinance_market_cap_fast(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(ticker).fast_info
        mc = info.get("market_cap")
        return safe_float(mc)
    except Exception:
        return None

def compute_rows(uni: pd.DataFrame) -> pd.DataFrame:
    secmap = sec_ticker_map()   # map TICKER->CIK10
    sectors = read_sector_catalog()

    rows: List[Dict[str, Any]] = []
    for _, r in uni.iterrows():
        t = r["ticker_yf"]

        # 1) Prix/Volume
        df = fetch_ohlcv_yf(t)
        lc = last_close(df)
        adv = avg_dollar_vol(df) or 0.0

        # 2) Tech local
        if df is not None and lc and lc >= MIN_PRICE_FOR_SCORE and len(df) >= 50:
            feats = compute_tech_features(df)
            tech_label, tech_score = tech_label_from_features(feats)
        else:
            tech_label, tech_score = "HOLD", 0.5

        # 3) SEC Shares + MCAP
        cik = secmap.get(t)
        shares = None
        if cik:
            shares = sec_latest_shares_outstanding(cik)

        mcap = None
        if lc is not None and shares is not None:
            mcap = float(lc) * float(shares)
        else:
            # fallback yfinance fast_info
            mcap = _yfinance_market_cap_fast(t)

        # 4) Proxy fondamentaux (Revenue/NI/Marge/EPS)
        analyst_label, analyst_score, analyst_votes = "HOLD", 0.5, 20
        if cik:
            facts = sec_companyfacts(cik)
            if facts:
                rev = _series_from_facts(facts, "us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", ("USD",))
                if not rev:
                    rev = _series_from_facts(facts, "us-gaap", "Revenues", ("USD",))
                ni  = _series_from_facts(facts, "us-gaap", "NetIncomeLoss", ("USD",))
                op  = _series_from_facts(facts, "us-gaap", "OperatingIncomeLoss", ("USD",))
                op_margin = []
                if rev and op and len(rev) == len(op):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        op_margin = [float(o)/abs(float(r)) if r else np.nan for o, r in zip(op, rev)]
                        op_margin = [x for x in op_margin if x == x]
                eps = _series_from_facts(facts, "us-gaap", "EarningsPerShareDiluted", ("USD/shares","USD"))
                analyst_label, analyst_score, analyst_votes = analyst_label_from_fundamentals(
                    rev, ni, op_margin, eps
                )

        # 5) Sector/Industry
        sec = sectors.get(t, {})
        sector = sec.get("sector", "Unknown")
        industry = sec.get("industry", "Unknown")

        # 6) TV pillar (remplacé par “Composite Tech”): label & score techniques locaux
        tv_reco = tech_label
        tv_score = float(tech_score)

        # 7) Pillars/buckets
        def favorable(label: str) -> bool:
            return label in ("BUY","STRONG_BUY")

        # éviter doublon: p_tech = label ; p_tv = score seuil
        p_tech = favorable(tech_label)
        p_tv   = (tv_score >= 0.66)
        p_an   = favorable(analyst_label)

        pillars_met = int(p_tech) + int(p_tv) + int(p_an)

        # votes_bin (cosmétique)
        if analyst_votes >= 30: votes_bin = "20+"
        elif analyst_votes >= 20: votes_bin = "10-20"
        else: votes_bin = "<10"

        # buckets
        if pillars_met == 3 and ("STRONG_BUY" in (tech_label, tv_reco, analyst_label)):
            bucket = "confirmed"
        elif pillars_met >= 2:
            bucket = "pre_signal"
        else:
            bucket = "event"

        # rank score
        adv_weight = math.log(1 + adv) / 20.0
        rank_score = float(
            (pillars_met / 3.0) * 0.6 +
            (tv_score) * 0.2 +
            (analyst_score) * 0.1 +
            min(0.1, adv_weight)
        )

        rows.append({
            "ticker_yf": t,
            "ticker_tv": t,
            "price": lc,
            "last": lc,
            "mcap_usd_final": mcap,
            "mcap": mcap,
            "avg_dollar_vol": adv,
            "tv_score": tv_score,
            "tv_reco": tv_reco,
            "analyst_bucket": analyst_label,
            "analyst_votes": float(analyst_votes),
            "sector": sector,
            "industry": industry,
            "p_tech": bool(p_tech),
            "p_tv": bool(p_tv),
            "p_an": bool(p_an),
            "pillars_met": int(pillars_met),
            "votes_bin": votes_bin,
            "rank_score": rank_score,
        })

    return pd.DataFrame(rows)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre strict:
    - MCAP non nul & < 75B
    - Prix non nul (sinon rien n’a de sens)
    """
    mask = df["mcap_usd_final"].notna() & (df["mcap_usd_final"] < MCAP_CAP) & df["price"].notna()
    return df[mask].copy()

def write_outputs(out: pd.DataFrame):
    out_sorted = out.sort_values(["rank_score","avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)
    out_sorted.to_csv(OUT_DIR / "candidates_all_ranked.csv", index=False)
    _log(f"[SAVE] candidates_all_ranked.csv | rows={len(out_sorted)} | cols={out_sorted.shape[1]}")

    confirmed = out_sorted[out_sorted["bucket"]=="confirmed"]
    pre_sig   = out_sorted[out_sorted["bucket"]=="pre_signal"]
    event     = out_sorted[out_sorted["bucket"]=="event"]

    confirmed.to_csv(OUT_DIR / "confirmed_STRONGBUY.csv", index=False)
    pre_sig.to_csv(OUT_DIR / "anticipative_pre_signals.csv", index=False)
    event.to_csv(OUT_DIR / "event_driven_signals.csv", index=False)

    _log(f"[SAVE] confirmed={len(confirmed)} pre={len(pre_sig)} event={len(event)}")

def update_signals_history(today: str, out: pd.DataFrame):
    path = OUT_DIR / "signals_history.csv"
    cols = ["date","ticker_yf","sector","bucket","tv_reco","analyst_bucket"]
    new = out[["ticker_yf","sector","bucket","tv_reco","analyst_bucket"]].copy()
    new.insert(0, "date", today)
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=cols)
        hist = pd.concat([old, new], ignore_index=True)
    else:
        hist = new
    hist = hist.tail(100)
    hist.to_csv(path, index=False)
    _log(f"[SAVE] signals_history.csv | rows={len(hist)}")

def main():
    _log("[STEP] mix_ab_screen_indices starting…")
    uni = ensure_universe()
    base = compute_rows(uni)
    base = apply_filters(base)

    # anti-NaN (cosmétique)
    for c in ["price","last","mcap_usd_final","avg_dollar_vol","tv_score","analyst_votes","rank_score"]:
        base[c] = base[c].astype(float).where(pd.notna(base[c]), None)
    base["bucket"] = base["bucket"].fillna("event")

    write_outputs(base)

    today = pd.Timestamp.utcnow().date().isoformat()
    update_signals_history(today, base)

    coverage = {
        "universe": len(uni),
        "price_nonnull": int(base["price"].notna().sum()),
        "mcap_final_nonnull": int(base["mcap_usd_final"].notna().sum()),
        "confirmed_count": int((base["bucket"]=="confirmed").sum()),
        "pre_count": int((base["bucket"]=="pre_signal").sum()),
        "event_count": int((base["bucket"]=="event").sum()),
        "unique_total": len(base),
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    main()
