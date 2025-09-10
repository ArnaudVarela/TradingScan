# mix_ab_screen_indices.py
# Pipeline "screener" robuste (sans providers payants) avec:
# - OHLCV via yfinance + cache local (parquet -> fallback csv)
# - TechScore local (pandas-ta dans tech_score.py)
# - Proxy "analyst" via fondamentaux SEC (sec_helpers.py + analyst_proxy.py)
# - Filtre MCAP < 75B (strict configurable)
#
# Sorties (à la racine du repo):
#   - candidates_all_ranked.csv
#   - confirmed_STRONGBUY.csv
#   - anticipative_pre_signals.csv
#   - event_driven_signals.csv
#   - signals_history.csv

from __future__ import annotations
import os, math, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import yfinance as yf

from sec_helpers import sec_ticker_map, sec_latest_shares_outstanding, _sec_get_json
from tech_score import compute_tech_features, tech_label_from_features
from analyst_proxy import analyst_label_from_fundamentals

# ------------------------------ Paths & Params

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUT_DIR = ROOT  # sorties à la racine
UNIVERSE_CSV = ROOT / "universe_in_scope.csv"
SECTOR_CATALOG = ROOT / "sector_catalog.csv"

# Params
MCAP_CAP = 75e9                       # filtre < 75B
STRICT_MCAP_FILTER = True             # True: exclut les NaN mcap ; False: laisse passer NaN
MIN_PRICE_FOR_SCORE = 0.5             # ignorer penny stocks pour certains signaux
OHLCV_WINDOW_DAYS = 240               # fenêtre pour indicateurs/avg$vol
AVG_DOLLAR_VOL_LOOKBACK = 20          # 20 jours
YF_TIMEOUT = 15

NEEDED_COLS = ["open", "high", "low", "close", "volume"]

# --------------------------------------------------------------------- utils

def _log(msg: str):
    print(msg, flush=True)

def safe_float(x) -> Optional[float]:
    """Conversion float tolérante: Series -> iloc[0], NaN -> None."""
    if x is None:
        return None
    if isinstance(x, pd.Series):
        if x.empty:
            return None
        x = x.iloc[0]
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def read_sector_catalog() -> Dict[str, Dict[str, str]]:
    if not SECTOR_CATALOG.exists():
        return {}
    try:
        df = pd.read_csv(SECTOR_CATALOG)
        res: Dict[str, Dict[str, str]] = {}
        for _, r in df.iterrows():
            t = str(r.get("ticker") or r.get("ticker_yf") or "").upper().strip()
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

# ------------------------------ OHLCV cache (Parquet -> fallback CSV)

def ohlcv_cache_path_parquet(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.parquet"

def ohlcv_cache_path_csv(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.csv"

def _ensure_df(obj) -> Optional[pd.DataFrame]:
    """Force un DataFrame quotidien propre avec colonnes normalisées."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        obj = obj.to_frame().T
    if not isinstance(obj, pd.DataFrame):
        return None
    df = obj.copy()
    # Normalise colonnes yfinance
    rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df.columns = [rename_map.get(c, c) for c in df.columns]
    # Index datetime propre
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    # Si on a les colonnes attendues, on garde seulement celles-ci
    if all(c in df.columns for c in NEEDED_COLS):
        df = df[NEEDED_COLS]
    # Types numeriques
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]) if "close" in df.columns else df
    return df if not df.empty else None

def _read_cache_any(ticker: str) -> Optional[pd.DataFrame]:
    """Essaye parquet puis csv. Si parquet corrompu, le supprime."""
    p_parq = ohlcv_cache_path_parquet(ticker)
    if p_parq.exists():
        try:
            df = pd.read_parquet(p_parq)
            return _ensure_df(df)
        except Exception:
            # cache parquet illisible -> le supprimer
            try:
                p_parq.unlink(missing_ok=True)
            except Exception:
                pass

    p_csv = ohlcv_cache_path_csv(ticker)
    if p_csv.exists():
        try:
            df = pd.read_csv(p_csv, index_col=0)
            return _ensure_df(df)
        except Exception:
            try:
                p_csv.unlink(missing_ok=True)
            except Exception:
                pass
    return None

def _write_cache_any(ticker: str, df: pd.DataFrame):
    """Essaye parquet, sinon CSV (si pas de pyarrow/fastparquet)."""
    p_parq = ohlcv_cache_path_parquet(ticker)
    p_csv = ohlcv_cache_path_csv(ticker)
    try:
        df.to_parquet(p_parq, index=True)
        # on peut nettoyer vieux CSV
        try: p_csv.unlink(missing_ok=True)
        except Exception: pass
    except Exception:
        # fallback CSV
        try:
            df.to_csv(p_csv)
            try: p_parq.unlink(missing_ok=True)
            except Exception: pass
        except Exception:
            # Rien à faire: pas de cache
            pass

def fetch_ohlcv_yf(ticker: str, period_days: int = OHLCV_WINDOW_DAYS) -> Optional[pd.DataFrame]:
    """
    Télécharge OHLCV (daily) via yfinance, cache local.
    Retourne df indexé datetime avec colonnes: open, high, low, close, volume
    """
    df = _read_cache_any(ticker)

    need_fetch = True
    if df is not None and len(df) >= 50:
        last_ts = df.index[-1]
        # si données fraiches (< 2 jours)
        try:
            if (pd.Timestamp.utcnow().tz_localize("UTC") - last_ts.tz_localize("UTC")).days < 2:
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
            yf_df = _ensure_df(yf_df)
            if yf_df is None:
                return df  # fallback cache
            _write_cache_any(ticker, yf_df)
            df = yf_df
        except Exception as e:
            _log(f"[YF] fetch failed {ticker}: {e}")
            return df  # retourne éventuellement cache existant

    return df

def last_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "close" not in df.columns:
        return None
    return safe_float(df["close"].iloc[-1])

def avg_dollar_vol(df: pd.DataFrame, lookback: int = AVG_DOLLAR_VOL_LOOKBACK) -> Optional[float]:
    if df is None:
        return None
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    if df.empty or not all(c in df.columns for c in ["close", "volume"]):
        return None

    n = min(len(df), max(1, lookback))
    sub = df.tail(n)

    close = pd.to_numeric(sub["close"], errors="coerce")
    vol = pd.to_numeric(sub["volume"], errors="coerce")
    prod = close.mul(vol)

    val = prod.mean(skipna=True)
    return float(val) if pd.notna(val) else None

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
    cache = load_facts_cache()
    now = time.time()
    key = f"CIK{cik10}"
    rec = cache["data"].get(key)
    if rec and now - rec.get("_ts", 0) < ttl_hours * 3600:
        return rec.get("facts")

    try:
        facts = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    except Exception:
        # fallback cache si dispo
        return rec.get("facts") if rec else None

    cache["data"][key] = {"_ts": now, "facts": facts}
    save_facts_cache(cache)
    return facts

def _series_from_facts(facts: dict, ns: str, tag: str, prefer_units=("USD", "USD/shares")) -> List[float]:
    try:
        units = facts["facts"][ns][tag]["units"]
    except Exception:
        return []
    # choisir une unité "meilleure"
    unit_keys = sorted(units.keys(), key=lambda u: 0 if any(p.lower() in u.lower() for p in prefer_units) else 1)
    arr: List[float] = []
    for uk in unit_keys:
        a = units[uk]
        # tri chrono
        a_sorted = sorted(a, key=lambda x: (x.get("fy", 0), x.get("fp", ""), x.get("end", "")))
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

# ------------------------------ Pipeline

def ensure_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise SystemExit(f"[FATAL] {UNIVERSE_CSV} introuvable. Lance d'abord build_universe.py")
    df = pd.read_csv(UNIVERSE_CSV)

    # trouver la colonne ticker
    col = None
    for c in ["ticker_yf", "ticker", "Symbol", "symbol"]:
        if c in df.columns:
            col = c
            break
    if not col:
        raise SystemExit("[FATAL] universe_in_scope.csv: colonne ticker introuvable")

    uni = df[[col]].rename(columns={col: "ticker_yf"}).copy()

    # normaliser tickers
    uni["ticker_yf"] = uni["ticker_yf"].astype(str).str.strip().str.upper()

    # virer vides, tirets, et caractères non valides
    uni = uni[uni["ticker_yf"].ne("") & uni["ticker_yf"].ne("-")]
    uni = uni[uni["ticker_yf"].str.match(r"^[A-Za-z0-9.\-]+$")]

    # éviter les doublons
    uni = uni.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)

    _log(f"[UNI] in-scope: {len(uni)}")
    return uni

def compute_rows(uni: pd.DataFrame) -> pd.DataFrame:
    secmap = sec_ticker_map()
    sectors = read_sector_catalog()

    rows: List[Dict[str, Any]] = []
    for _, r in uni.iterrows():
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
                rev = _series_from_facts(
                    facts,
                    "us-gaap",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    ("USD",),
                )
                if not rev:
                    rev = _series_from_facts(facts, "us-gaap", "Revenues", ("USD",))
                ni = _series_from_facts(facts, "us-gaap", "NetIncomeLoss", ("USD",))
                op = _series_from_facts(facts, "us-gaap", "OperatingIncomeLoss", ("USD",))
                op_margin = []
                if rev and op and len(rev) == len(op):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        op_margin = [
                            float(o) / abs(float(r)) if r else np.nan for o, r in zip(op, rev)
                        ]
                        op_margin = [x for x in op_margin if x == x]  # drop NaN
                eps = _series_from_facts(
                    facts, "us-gaap", "EarningsPerShareDiluted", ("USD/shares", "USD")
                )
                analyst_label, analyst_score, analyst_votes = analyst_label_from_fundamentals(
                    rev, ni, op_margin, eps
                )

        # 5) Sector/industry
        sec = sectors.get(t, {})
        sector = sec.get("sector", "Unknown")
        industry = sec.get("industry", "Unknown")

        # 6) TV pillar remplacé par “Composite Tech” (= notre label tech)
        tv_reco = tech_label
        tv_score = tech_score

        # 7) Pillars & bucket
        def favorable(label: str) -> bool:
            return label in ("BUY", "STRONG_BUY")

        p_tech = favorable(tech_label)
        p_tv = favorable(tv_reco)
        p_an = favorable(analyst_label)

        pillars_met = int(p_tech) + int(p_tv) + int(p_an)

        # votes_bin (cosmétique)
        if analyst_votes >= 30:
            votes_bin = "20+"
        elif analyst_votes >= 20:
            votes_bin = "10-20"
        else:
            votes_bin = "<10"

        # bucket rules
        if pillars_met == 3 and ("STRONG_BUY" in (tech_label, tv_reco, analyst_label)):
            bucket = "confirmed"
        elif pillars_met >= 2:
            bucket = "pre_signal"
        else:
            bucket = "event"

        # rank score simple: poids piliers + tv_score + analyst_score + log(ADV)
        adv_weight = math.log(1 + max(0.0, adv)) / 20.0  # petite échelle
        rank_score = float(
            (pillars_met / 3.0) * 0.6 + (tv_score) * 0.2 + (analyst_score) * 0.1 + min(0.1, adv_weight)
        )

        rows.append(
            {
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
                "bucket": bucket,
            }
        )

    return pd.DataFrame(rows)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre MCAP < 75B. En mode strict, élimine aussi les NaN mcap."""
    if "mcap_usd_final" not in df.columns:
        return df
    if STRICT_MCAP_FILTER:
        mask = pd.to_numeric(df["mcap_usd_final"], errors="coerce").lt(MCAP_CAP)
        return df[mask].copy()
    else:
        # Laisse passer NaN (mais peut faire grossir les outputs)
        mcap = pd.to_numeric(df["mcap_usd_final"], errors="coerce")
        mask = mcap.isna() | mcap.lt(MCAP_CAP)
        return df[mask].copy()

def write_outputs(out: pd.DataFrame):
    out_sorted = out.sort_values(["rank_score", "avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)
    out_sorted.to_csv(OUT_DIR / "candidates_all_ranked.csv", index=False)
    _log(f"[SAVE] candidates_all_ranked.csv | rows={len(out_sorted)} | cols={out_sorted.shape[1]}")

    confirmed = out_sorted[out_sorted["bucket"] == "confirmed"]
    pre_sig = out_sorted[out_sorted["bucket"] == "pre_signal"]
    event = out_sorted[out_sorted["bucket"] == "event"]

    confirmed.to_csv(OUT_DIR / "confirmed_STRONGBUY.csv", index=False)
    pre_sig.to_csv(OUT_DIR / "anticipative_pre_signals.csv", index=False)
    event.to_csv(OUT_DIR / "event_driven_signals.csv", index=False)

    _log(f"[SAVE] confirmed={len(confirmed)} pre={len(pre_sig)} event={len(event)}")

def update_signals_history(today: str, out: pd.DataFrame):
    path = OUT_DIR / "signals_history.csv"
    cols = ["date", "ticker_yf", "sector", "bucket", "tv_reco", "analyst_bucket"]
    new = out[["ticker_yf", "sector", "bucket", "tv_reco", "analyst_bucket"]].copy()
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

    # Filtres
    base = apply_filters(base)

    # fallbacks anti-NaN pour colonnes numériques
    for c in ["price", "last", "mcap_usd_final", "avg_dollar_vol", "tv_score", "analyst_votes", "rank_score"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # bucket string safety
    if "bucket" in base.columns:
        base["bucket"] = base["bucket"].fillna("event")

    write_outputs(base)

    # history
    today = pd.Timestamp.utcnow().date().isoformat()
    update_signals_history(today, base)

    # couverture
    coverage = {
        "universe": len(uni),
        "price_nonnull": int(base["price"].notna().sum()) if "price" in base.columns else 0,
        "mcap_final_nonnull": int(base["mcap_usd_final"].notna().sum()) if "mcap_usd_final" in base.columns else 0,
        "confirmed_count": int((base["bucket"] == "confirmed").sum()) if "bucket" in base.columns else 0,
        "pre_count": int((base["bucket"] == "pre_signal").sum()) if "bucket" in base.columns else 0,
        "event_count": int((base["bucket"] == "event").sum()) if "bucket" in base.columns else 0,
        "unique_total": len(base),
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    # Future behavior explicite pour éviter les warnings silencieux
    pd.set_option("future.no_silent_downcasting", True)
    main()
