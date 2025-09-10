# mix_ab_screen_indices.py
# Remplace complètement l'ancien script.
from __future__ import annotations
import os, sys, math, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

from sec_helpers import sec_ticker_map, sec_latest_shares_outstanding
from tech_score import compute_tech_features, tech_label_from_features
from analyst_proxy import analyst_label_from_fundamentals

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUT_DIR = ROOT   # <<< sorties à la racine du repo
UNIVERSE_CSV = ROOT / "universe_in_scope.csv"     # fourni par build_universe.py
SECTOR_CATALOG = ROOT / "sector_catalog.csv"      # mapping ticker->sector/industry (si dispo)

# Params
MCAP_CAP = 75e9                    # filtre < 75B
MIN_PRICE_FOR_SCORE = 0.5          # ignorer penny stocks pour certains signaux
OHLCV_WINDOW_DAYS = 240            # fenêtre pour indicateurs/avg$vol
AVG_DOLLAR_VOL_LOOKBACK = 20       # 20 jours
YF_CHUNK_DAYS = 365                # pour cache/rafraîchissement
YF_TIMEOUT = 15

# --------------------------------------------------------------------- utils

def _log(msg: str):
    print(msg, flush=True)

def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)): return None
        return float(x)
    except Exception:
        return None

def read_sector_catalog() -> Dict[str, Dict[str,str]]:
    if not SECTOR_CATALOG.exists():
        return {}
    try:
        df = pd.read_csv(SECTOR_CATALOG)
        res = {}
        for _, r in df.iterrows():
            t = str(r.get("ticker") or r.get("ticker_yf") or "").upper()
            if not t: continue
            res[t] = {
                "sector": r.get("sector") if pd.notna(r.get("sector")) else "Unknown",
                "industry": r.get("industry") if pd.notna(r.get("industry")) else "Unknown",
            }
        return res
    except Exception as e:
        _log(f"[WARN] sector_catalog read failed: {e}")
        return {}

# ------------------------------ OHLCV cache (Parquet)

def ohlcv_cache_path(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.parquet"

def fetch_ohlcv_yf(ticker: str, period_days: int = OHLCV_WINDOW_DAYS) -> Optional[pd.DataFrame]:
    """
    Télécharge OHLCV (daily) via yfinance, cache parquet.
    Retourne df indexé datetime avec colonnes: open, high, low, close, volume
    """
    path = ohlcv_cache_path(ticker)
    df = None
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = None

    need_fetch = True
    if df is not None and len(df) >= 50:
        # Si dernier point < 2 jours, réutilise
        if (pd.Timestamp.utcnow().tz_localize("UTC") - df.index[-1].tz_localize("UTC")).days < 2:
            need_fetch = False

    if need_fetch:
        try:
            # yfinance period form: '300d' etc.
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
            # timezone-naive to UTC-naive
            yf_df.index = pd.to_datetime(yf_df.index)
            yf_df.to_parquet(path, index=True)
            df = yf_df
        except Exception as e:
            _log(f"[YF] fetch failed {ticker}: {e}")
            return df

    return df

def last_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    return safe_float(df["close"].iloc[-1])

def avg_dollar_vol(df: pd.DataFrame, lookback: int = AVG_DOLLAR_VOL_LOOKBACK) -> Optional[float]:
    if df is None or len(df) < lookback:
        return None
    sub = df.iloc[-lookback:]
    return float((sub["close"] * sub["volume"]).mean())

# ------------------------------ SEC companyfacts loader (light cache)

FACTS_PATH = CACHE_DIR / "sec_companyfacts_cache.json"

def load_facts_cache():
    try:
        with FACTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"_ts": 0, "data": {}}

def save_facts_cache(data):
    tmp = FACTS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(FACTS_PATH)

def sec_companyfacts(cik10: str, ttl_hours: float = 48.0) -> Optional[dict]:
    from sec_helpers import _sec_get_json  # reuse headers
    cache = load_facts_cache()
    now = time.time()
    key = f"CIK{cik10}"
    rec = cache["data"].get(key)
    if rec and now - rec.get("_ts", 0) < ttl_hours * 3600:
        return rec.get("facts")

    try:
        facts = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    except Exception as e:
        return rec.get("facts") if rec else None

    cache["data"][key] = {"_ts": now, "facts": facts}
    save_facts_cache(cache)
    return facts

def _series_from_facts(facts: dict, ns: str, tag: str, prefer_units=("USD","USD/shares")) -> List[float]:
    try:
        units = facts["facts"][ns][tag]["units"]
    except Exception:
        return []
    # pick best unit
    unit_keys = sorted(units.keys(), key=lambda u: 0 if any(p.lower() in u.lower() for p in prefer_units) else 1)
    arr = []
    for uk in unit_keys:
        a = units[uk]
        # tri chrono
        a_sorted = sorted(a, key=lambda x: (x.get("fy",0), x.get("fp",""), x.get("end","")))
        vals = []
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

# ------------------------------ Pipeline

def ensure_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise SystemExit(f"[FATAL] {UNIVERSE_CSV} introuvable. Lance d'abord build_universe.py")
    df = pd.read_csv(UNIVERSE_CSV)
    col = None
    for c in ["ticker_yf","ticker","Symbol","symbol"]:
        if c in df.columns:
            col = c
            break
    if not col:
        raise SystemExit("[FATAL] universe_in_scope.csv: colonne ticker introuvable")
    uni = df[[col]].dropna().rename(columns={col:"ticker_yf"})
    uni["ticker_yf"] = uni["ticker_yf"].astype(str).str.upper().str.strip()
    uni = uni[uni["ticker_yf"] != ""].drop_duplicates()
    _log(f"[UNI] in-scope: {len(uni)}")
    return uni

def compute_rows(uni: pd.DataFrame) -> pd.DataFrame:
    secmap = sec_ticker_map()
    sectors = read_sector_catalog()

    rows = []
    for i, r in uni.iterrows():
        t = r["ticker_yf"]
        # 1) prix & vol
        df = fetch_ohlcv_yf(t)
        lc = last_close(df)
        adv = avg_dollar_vol(df) or 0.0

        # 2) TechScore local
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
        mcap = (lc * shares) if (lc is not None and shares is not None) else None

        # 4) Fundamentals proxy (Revenue / NI / Margin / EPS)
        analyst_label, analyst_score, analyst_votes = "HOLD", 0.5, 20
        if cik:
            facts = sec_companyfacts(cik)
            if facts:
                rev = _series_from_facts(facts, "us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", ("USD",))
                if not rev:
                    rev = _series_from_facts(facts, "us-gaap", "Revenues", ("USD",))
                ni  = _series_from_facts(facts, "us-gaap", "NetIncomeLoss", ("USD",))
                # marge op approx: OperatingIncome / Revenue
                op  = _series_from_facts(facts, "us-gaap", "OperatingIncomeLoss", ("USD",))
                op_margin = []
                if rev and op and len(rev) == len(op):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        op_margin = [float(o)/abs(float(r)) if r else np.nan for o, r in zip(op, rev)]
                        op_margin = [x for x in op_margin if x == x]  # drop NaN
                eps = _series_from_facts(facts, "us-gaap", "EarningsPerShareDiluted", ("USD/shares","USD"))
                analyst_label, analyst_score, analyst_votes = analyst_label_from_fundamentals(rev, ni, op_margin, eps)

        # 5) Sector/industry
        sec = sectors.get(t, {})
        sector = sec.get("sector", "Unknown")
        industry = sec.get("industry", "Unknown")

        # 6) TV pillar remplacé par “Composite Tech” (= notre label tech)
        tv_reco = tech_label
        tv_score = tech_score

        # 7) Pillars & bucket
        # p_tech -> label tech local ; p_tv -> bool (reco favorable) ; p_an -> bool
        def favorable(label: str) -> bool:
            return label in ("BUY","STRONG_BUY")

        p_tech = favorable(tech_label)
        p_tv   = favorable(tv_reco)
        p_an   = favorable(analyst_label)

        pillars_met = sum([p_tech, p_tv, p_an])

        # votes_bin (cosmétique) en fonction d'analyst_votes
        if analyst_votes >= 30: votes_bin = "20+"
        elif analyst_votes >= 20: votes_bin = "10-20"
        else: votes_bin = "<10"

        # bucket rules
        if pillars_met == 3 and ("STRONG_BUY" in (tech_label, tv_reco, analyst_label)):
            bucket = "confirmed"
        elif pillars_met >= 2:
            bucket = "pre_signal"
        else:
            bucket = "event"

        # rank score simple: poids piliers + tv_score + analyst_score + log(ADV)
        adv_weight = math.log(1 + adv) / 20.0  # scale
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
    # Filtre < 75B si mcap dispo ; sinon on garde (on filtrera plus tard quand mcap dispo)
    mcap_mask = df["mcap_usd_final"].fillna(-1) < MCAP_CAP
    return df[mcap_mask | df["mcap_usd_final"].isna()].copy()

def write_outputs(out: pd.DataFrame):
    # master trié
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
    # garde 100 lignes récentes max
    hist = hist.tail(100)
    hist.to_csv(path, index=False)
    _log(f"[SAVE] signals_history.csv | rows={len(hist)}")

def main():
    _log("[STEP] mix_ab_screen_indices starting…")
    uni = ensure_universe()
    base = compute_rows(uni)

    # mcap_final (copie), bucket calc already set, mais on applique filtre <75B
    base["mcap_usd_final"] = base["mcap_usd_final"]
    base = apply_filters(base)

    # Re-évaluer buckets après filtre (facultatif, ici on conserve)
    # (ex: si filtre enlève des gros titres, rien à recalculer pour les autres)

    # fallbacks anti-NaN
    for c in ["price","last","mcap_usd_final","avg_dollar_vol","tv_score","analyst_votes","rank_score"]:
        base[c] = base[c].astype(float).where(pd.notna(base[c]), None)

    # bucket string safety
    base["bucket"] = base["bucket"].fillna("event")

    write_outputs(base)

    # history
    today = pd.Timestamp.utcnow().date().isoformat()
    update_signals_history(today, base)

    # couverture
    unique_total = len(base)
    coverage = {
        "universe": len(uni),
        "price_nonnull": int(base["price"].notna().sum()),
        "mcap_final_nonnull": int(base["mcap_usd_final"].notna().sum()),
        "confirmed_count": int((base["bucket"]=="confirmed").sum()),
        "pre_count": int((base["bucket"]=="pre_signal").sum()),
        "event_count": int((base["bucket"]=="event").sum()),
        "unique_total": unique_total,
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    main()
