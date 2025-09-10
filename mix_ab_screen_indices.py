# mix_ab_screen_indices.py
# Pipeline "A/B screen" unifié : construit les features, applique les
# trois piliers (Tech local, TV proxy = Tech local, Analyst proxy),
# filtre MCAP < 75B (strict), classe par bucket, et écrit les CSV.

from __future__ import annotations
import os, sys, math, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

# ---- dépendances locales
#   - sec_helpers.py : sec_ticker_map, sec_latest_shares_outstanding, _sec_get_json
#   - tech_score.py  : compute_tech_features, tech_label_from_features
#   - analyst_proxy.py : analyst_label_from_fundamentals
from sec_helpers import sec_ticker_map, sec_latest_shares_outstanding
from tech_score import compute_tech_features, tech_label_from_features
from analyst_proxy import analyst_label_from_fundamentals

# --------------------------------------------------------------------- Config
ROOT = Path(__file__).parent.resolve()
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUT_DIR = ROOT
UNIVERSE_CSV = ROOT / "universe_in_scope.csv"  # fourni par build_universe.py
SECTOR_CATALOG = ROOT / "sector_catalog.csv"   # optionnel: ticker→sector/industry

# Paramètres généraux
MCAP_CAP = 75e9                # filtre < 75B (strict)
MCAP_STRICT = True             # True: NaN = exclu ; False: NaN passent le filtre
MIN_PRICE_FOR_SCORE = 0.5      # ignorer penny stocks pour TechScore
OHLCV_WINDOW_DAYS = 240        # fenêtre pour indicateurs/ADV
AVG_DOLLAR_VOL_LOOKBACK = 20   # 20 jours
YF_TIMEOUT = 20
YF_REFETCH_MAX_AGE_DAYS = 2    # si dernier point < 2 jours, on garde le cache

# Throttle léger YF (évite rate limit)
SLEEP_BETWEEN_TICKERS = 0.0    # mets 0.05/0.1 si rate limit

# --------------------------------------------------------------------- Utils
def _log(msg: str):
    print(msg, flush=True)

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (pd.Series, pd.Index)) and len(x) == 1:
            x = x.iloc[0]
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
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
            res[t] = {
                "sector": (r.get("sector") if pd.notna(r.get("sector")) else "Unknown"),
                "industry": (r.get("industry") if pd.notna(r.get("industry")) else "Unknown"),
            }
        return res
    except Exception as e:
        _log(f"[WARN] sector_catalog read failed: {e}")
        return {}

# --------------------------------------------------------------------- OHLCV cache

def _parquet_path(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.parquet"

def _csv_path(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.csv"

def _write_cache(df: pd.DataFrame, t: str) -> None:
    """Essaie Parquet, fallback CSV."""
    df = df.copy()
    try:
        df.to_parquet(_parquet_path(t), index=True)
        return
    except Exception:
        pass
    try:
        df.to_csv(_csv_path(t), index=True)
    except Exception as e:
        _log(f"[CACHE] write failed for {t}: {e}")

def _read_cache(t: str) -> Optional[pd.DataFrame]:
    """Lit Parquet si dispo, sinon CSV. Retourne None si rien/erreur."""
    p = _parquet_path(t)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    c = _csv_path(t)
    if c.exists():
        try:
            return pd.read_csv(c, index_col=0, parse_dates=True)
        except Exception:
            pass
    return None

def _sanitize(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in
    out = df_in.copy()

    # Aplatir & normaliser noms
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["|".join(map(str, t)) for t in out.columns]
    out.columns = [str(c) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="last")]

    rename_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    }
    for k, v in rename_map.items():
        if k in out.columns and v not in out.columns:
            out = out.rename(columns={k: v})

    # Fallbacks close
    if "close" not in out.columns and "adj_close" in out.columns:
        out["close"] = out["adj_close"]
    if "close" not in out.columns:
        cand = [c for c in out.columns if c.lower().strip().endswith("close")]
        if cand:
            out["close"] = out[cand[0]]

    # Index datetime
    out.index = pd.to_datetime(out.index, errors="coerce")

    # Cast numérique robuste
    for c in ["open", "high", "low", "close", "volume", "adj_close"]:
        if c in out.columns:
            col = out[c]
            if isinstance(col, pd.DataFrame):  # au cas où
                col = col.iloc[:, 0]
            col = pd.to_numeric(pd.Series(col.to_numpy().reshape(-1)), errors="coerce")
            out[c] = col.to_numpy()

    # Drop lignes sans prix exploitable
    if "close" in out.columns:
        out = out.dropna(subset=["close"])
    elif "adj_close" in out.columns:
        out = out.dropna(subset=["adj_close"])

    # Ne garder que colonnes utiles si présentes
    wanted = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in out.columns]
    out = out[wanted] if wanted else out

    # tri par index
    out = out.sort_index()
    return out

def fetch_ohlcv_yf(ticker: str, period_days: int = OHLCV_WINDOW_DAYS) -> Optional[pd.DataFrame]:
    """
    Télécharge/cached OHLCV daily via yfinance.
    Retourne df index datetime avec colonnes: open, high, low, close, volume (+adj_close si dispo)
    """
    # 1) cache
    df = _read_cache(ticker)
    if df is not None and not df.empty:
        try:
            last_idx = pd.to_datetime(df.index[-1]).tz_localize(None)
            age = (pd.Timestamp.utcnow().tz_localize(None) - last_idx).days
            if age < YF_REFETCH_MAX_AGE_DAYS:
                return _sanitize(df)
        except Exception:
            pass

    # 2) fetch
    try:
        period = f"{max(period_days, 120)}d"
        yf_df = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=False, progress=False, threads=False, timeout=YF_TIMEOUT
        )
        if yf_df is None or yf_df.empty:
            return _sanitize(df) if df is not None else None
        yf_df = _sanitize(yf_df)
        if yf_df is None or yf_df.empty:
            return _sanitize(df) if df is not None else None
        _write_cache(yf_df, ticker)
        return yf_df
    except Exception as e:
        _log(f"[YF] fetch failed {ticker}: {e}")
        # fallback: cache brut si dispo
        return _sanitize(df) if df is not None else None

def last_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    col = "close" if "close" in df.columns else ("adj_close" if "adj_close" in df.columns else None)
    if not col or df[col].isna().all():
        return None
    try:
        v = df[col].iloc[-1]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None

def avg_dollar_vol(df: pd.DataFrame, lookback: int = AVG_DOLLAR_VOL_LOOKBACK) -> Optional[float]:
    if df is None or len(df) < max(lookback, 1):
        return None
    price_col = "close" if "close" in df.columns else ("adj_close" if "adj_close" in df.columns else None)
    if not price_col or "volume" not in df.columns:
        return None
    sub = df.tail(lookback).copy()
    p = pd.to_numeric(sub[price_col], errors="coerce")
    v = pd.to_numeric(sub["volume"], errors="coerce")
    prod = (p * v).dropna()
    if prod.empty:
        return None
    return float(prod.mean())

# --------------------------------------------------------------------- SEC facts (cache léger)
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
    from sec_helpers import _sec_get_json  # réutilise headers & retry
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
    # Choix d’unité « acceptable »
    unit_keys = sorted(units.keys(), key=lambda u: 0 if any(p.lower() in u.lower() for p in prefer_units) else 1)
    arr: List[float] = []
    for uk in unit_keys:
        a = units[uk]
        a_sorted = sorted(a, key=lambda x: (x.get("fy", 0), str(x.get("fp","")), str(x.get("end",""))))
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

# --------------------------------------------------------------------- Pipeline

def ensure_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise SystemExit(f"[FATAL] {UNIVERSE_CSV} introuvable. Lance d'abord build_universe.py")
    df = pd.read_csv(UNIVERSE_CSV)

    # Trouver la colonne ticker
    col = None
    for c in ["ticker_yf", "ticker", "Symbol", "symbol"]:
        if c in df.columns:
            col = c
            break
    if not col:
        raise SystemExit("[FATAL] universe_in_scope.csv: colonne ticker introuvable")

    # Normaliser sous 'ticker_yf'
    if "ticker_yf" not in df.columns:
        df["ticker_yf"] = df[col].astype(str)
    df["ticker_yf"] = df["ticker_yf"].astype(str).str.strip().str.upper()

    # Filtrages de base
    df = df[df["ticker_yf"].ne("") & df["ticker_yf"].ne("-")]
    df = df[df["ticker_yf"].str.match(r"^[A-Z0-9.\-]+$")]
    df = df.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)

    _log(f"[UNI] in-scope: {len(df)}")
    return df

def _favorable(label: str) -> bool:
    return str(label).upper() in ("BUY", "STRONG_BUY")

def compute_rows(uni: pd.DataFrame) -> pd.DataFrame:
    secmap = sec_ticker_map()
    sectors = read_sector_catalog()

    rows: List[Dict[str, Any]] = []
    n = len(uni)
    for i, r in uni.iterrows():
        t = str(r["ticker_yf"]).upper().strip()

        try:
            # 1) Prix & ADV
            df = fetch_ohlcv_yf(t)
            lc = last_close(df)
            adv = avg_dollar_vol(df) or 0.0

            # 2) TechScore local
            if df is not None and lc and lc >= MIN_PRICE_FOR_SCORE and len(df) >= 50:
                feats = compute_tech_features(df)
                tech_label, tech_score = tech_label_from_features(feats)
            else:
                tech_label, tech_score = "HOLD", 0.5

            # 3) SEC shares & MCAP
            cik = secmap.get(t)
            shares = None
            if cik:
                shares = sec_latest_shares_outstanding(cik)
            mcap = (lc * shares) if (lc is not None and shares is not None) else None

            # 4) Fundamentals proxy (Revenue, NI, Margin, EPS)
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
                            op_margin = [x for x in op_margin if x == x]  # rm NaN
                    eps = _series_from_facts(facts, "us-gaap", "EarningsPerShareDiluted", ("USD/shares","USD"))
                    analyst_label, analyst_score, analyst_votes = analyst_label_from_fundamentals(rev, ni, op_margin, eps)

            # 5) Sector/industry
            sec = sectors.get(t, {})
            sector = sec.get("sector", "Unknown")
            industry = sec.get("industry", "Unknown")

            # 6) TV proxy = Tech local
            tv_reco = tech_label
            tv_score = tech_score

            # 7) Pillars & bucket
            p_tech = _favorable(tech_label)
            p_tv   = _favorable(tv_reco)
            p_an   = _favorable(analyst_label)
            pillars_met = int(sum([p_tech, p_tv, p_an]))

            if analyst_votes >= 30:
                votes_bin = "20+"
            elif analyst_votes >= 20:
                votes_bin = "10-20"
            else:
                votes_bin = "<10"

            if pillars_met == 3 and ("STRONG_BUY" in (tech_label, tv_reco, analyst_label)):
                bucket = "confirmed"
            elif pillars_met >= 2:
                bucket = "pre_signal"
            else:
                bucket = "event"

            adv_weight = math.log(1 + max(adv, 0.0)) / 20.0
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
                "bucket": bucket,
                "rank_score": rank_score,
            })

        except Exception as e:
            _log(f"[ROW] {t}: {e}")
        finally:
            if SLEEP_BETWEEN_TICKERS:
                time.sleep(SLEEP_BETWEEN_TICKERS)

        # log de progression (rare pour ne pas flood)
        if (i + 1) % 500 == 0 or (i + 1) == n:
            _log(f"[PROG] {i+1}/{n}")

    return pd.DataFrame(rows)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre strict < 75B. Si MCAP_STRICT=True, NaN sont exclus."""
    if df.empty:
        return df
    mcap = pd.to_numeric(df["mcap_usd_final"], errors="coerce")
    if MCAP_STRICT:
        mask = (mcap.notna()) & (mcap < MCAP_CAP)
    else:
        mask = (mcap.isna()) | (mcap < MCAP_CAP)
    out = df.loc[mask].copy()
    return out

def write_outputs(out: pd.DataFrame):
    out_sorted = out.sort_values(["rank_score","avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)
    (OUT_DIR / "candidates_all_ranked.csv").write_text("")  # assure perms si needed
    out_sorted.to_csv(OUT_DIR / "candidates_all_ranked.csv", index=False)
    _log(f"[SAVE] candidates_all_ranked.csv | rows={len(out_sorted)} | cols={out_sorted.shape[1]}")

    confirmed = out_sorted[out_sorted["bucket"] == "confirmed"]
    pre_sig   = out_sorted[out_sorted["bucket"] == "pre_signal"]
    event     = out_sorted[out_sorted["bucket"] == "event"]

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
            hist = pd.concat([old, new], ignore_index=True)
        except Exception:
            hist = new
    else:
        hist = new

    hist = hist.tail(100)  # borne
    hist.to_csv(path, index=False)
    _log(f"[SAVE] signals_history.csv | rows={len(hist)}")

def main():
    _log("[STEP] mix_ab_screen_indices starting…")
    uni = ensure_universe()
    base = compute_rows(uni)

    # Renommage/typages soft
    num_cols = ["price","last","mcap_usd_final","avg_dollar_vol","tv_score","analyst_votes","rank_score"]
    for c in num_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # Filtre MCAP
    base = apply_filters(base)

    # Sécurité bucket
    if "bucket" not in base.columns:
        base["bucket"] = "event"
    base["bucket"] = base["bucket"].fillna("event")

    write_outputs(base)

    today = pd.Timestamp.utcnow().date().isoformat()
    update_signals_history(today, base)

    coverage = {
        "universe": len(uni),
        "rows_after_filter": len(base),
        "price_nonnull": int(base["price"].notna().sum()) if "price" in base.columns else 0,
        "mcap_final_nonnull": int(base["mcap_usd_final"].notna().sum()) if "mcap_usd_final" in base.columns else 0,
        "confirmed_count": int((base["bucket"]=="confirmed").sum()),
        "pre_count": int((base["bucket"]=="pre_signal").sum()),
        "event_count": int((base["bucket"]=="event").sum()),
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    # éviter les warning pandas agressifs
    pd.set_option("future.no_silent_downcasting", True)
    main()
