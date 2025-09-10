# mix_ab_screen_indices.py
# Pipeline de scoring local (Tech + proxy "Analyst") + fusion SEC + YF
# Sorties CSV à la racine du repo.

from __future__ import annotations
import os, sys, math, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import yfinance as yf

# Modules locaux (fichiers à côté)
from sec_helpers import sec_ticker_map, sec_latest_shares_outstanding, _sec_get_json
from tech_score import compute_tech_features, tech_label_from_features
from analyst_proxy import analyst_label_from_fundamentals

# ------------------------------------------------------------------ Constantes

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUT_DIR = ROOT   # sorties CSV à la racine
UNIVERSE_CSV = ROOT / "universe_in_scope.csv"   # produit par build_universe.py
SECTOR_CATALOG = ROOT / "sector_catalog.csv"    # mapping optionnel ticker->(sector,industry)

# Paramètres
MCAP_CAP = float(os.getenv("MCAP_CAP", 75e9))     # filtre strict < 75B
KEEP_NAN_MCAP = bool(int(os.getenv("KEEP_NAN_MCAP", "0")))  # 0 = drop NaN mcap
MIN_PRICE_FOR_SCORE = 0.5
OHLCV_WINDOW_DAYS = 240
AVG_DOLLAR_VOL_LOOKBACK = 20
YF_TIMEOUT = 15

pd.set_option("future.no_silent_downcasting", True)

# ------------------------------------------------------------------ Utils

def _log(msg: str) -> None:
    print(msg, flush=True)

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (pd.Series, pd.Index)) and len(x) == 1:
            return float(x.iloc[0])
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dé-hierarchise les colonnes si multi-index, et cast en str pour éviter
    les warnings parquet (“mixed type columns”).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join(map(str, tup)) for tup in df.columns]
    df.columns = [str(c) for c in df.columns]
    return df

def _write_cache_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Écrit un parquet sans warning de colonnes mixtes.
    On aplatit et on s’assure que l’index est DatetimeIndex simple.
    """
    try:
        out = df.copy()
        out = _flatten_columns(out)
        # Force DatetimeIndex si possible
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        out.to_parquet(path, index=True)
    except Exception as e:
        _log(f"[CACHE] write parquet failed {path.name}: {e}")

# ------------------------------------------------------------------ Sector catalog

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
                "sector": sector if (isinstance(sector, str) and sector) else "Unknown",
                "industry": industry if (isinstance(industry, str) and industry) else "Unknown",
            }
        return res
    except Exception as e:
        _log(f"[WARN] sector_catalog read failed: {e}")
        return {}

# ------------------------------------------------------------------ OHLCV cache YF

def ohlcv_cache_path(t: str) -> Path:
    return CACHE_DIR / f"ohlcv_{t}.parquet"

def fetch_ohlcv_yf(ticker: str, period_days: int = OHLCV_WINDOW_DAYS) -> Optional[pd.DataFrame]:
    """
    Télécharge OHLCV daily via yfinance, avec cache parquet.
    Colonnes: open, high, low, close, volume
    """
    path = ohlcv_cache_path(ticker)
    df = None

    def _sanitize(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return df_in
        out = df_in.copy()

        # Aplatir et supprimer colonnes dupliquées
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = ["|".join(map(str, t)) for t in out.columns]
        out.columns = [str(c) for c in out.columns]
        out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="last")]

        # Renommer si besoin (cas cache ancien)
        rename_map = {
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
            "Adj Close": "adj_close"
        }
        for k, v in rename_map.items():
            if k in out.columns and v not in out.columns:
                out = out.rename(columns={k: v})

        # Garder seulement les colonnes OHLCV si présentes
        keep = [c for c in ["open","high","low","close","volume"] if c in out.columns]
        out = out[keep] if keep else out

        # Index → datetime
        out.index = pd.to_datetime(out.index, errors="coerce")

        # Forcer en numérique, même si df[c] est un DataFrame (ex: colonnes dupli)
        for c in ["open","high","low","close","volume"]:
            if c not in out.columns:
                continue
            col = out[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]  # on garde la dernière/unique colonne utile
            col = pd.to_numeric(pd.Series(col.to_numpy().reshape(-1)), errors="coerce")
            out[c] = col.to_numpy()

        # Drop lignes sans close/volume
        if "close" in out.columns and "volume" in out.columns:
            out = out.dropna(subset=["close", "volume"])

        return out

    # 1) Lire cache si dispo
    if path.exists():
        try:
            df = pd.read_parquet(path)
            df = _sanitize(df)
        except Exception:
            df = None

    # 2) Décider si on refetch
    need_fetch = True
    if df is not None and len(df) >= 50:
        try:
            last_ts = pd.to_datetime(df.index[-1], utc=True)
            now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
            if (now_utc - last_ts).days < 2:
                need_fetch = False
        except Exception:
            need_fetch = True

    # 3) Fetch YF si nécessaire
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
                # si on a un cache utilisable, on le renvoie
                return df
            yf_df = yf_df.rename(
                columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
            )
            # certaines installations YF remontent aussi "Adj Close"
            yf_df = yf_df[["open","high","low","close","volume"]]
            yf_df.index = pd.to_datetime(yf_df.index, errors="coerce")
            yf_df = _sanitize(yf_df)

            _write_cache_parquet(yf_df, path)
            df = yf_df
        except Exception as e:
            _log(f"[YF] fetch failed {ticker}: {e}")
            # On retombe sur le cache si présent
            return df

    return df

    # types sûrs
    if df is not None and not df.empty:
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close","volume"])
    return df

def last_close(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty:
        return None
    v = df["close"].iloc[-1]
    return safe_float(v)

def avg_dollar_vol(df: Optional[pd.DataFrame], lookback: int = AVG_DOLLAR_VOL_LOOKBACK) -> Optional[float]:
    if df is None or df.empty or len(df) < lookback:
        return None
    sub = df.iloc[-lookback:]
    close_s = pd.to_numeric(sub["close"], errors="coerce")
    vol_s = pd.to_numeric(sub["volume"], errors="coerce")
    prod = (close_s * vol_s).dropna()
    if prod.empty:
        return None
    return float(prod.mean())

# ------------------------------------------------------------------ SEC companyfacts (+ mini-cache)

FACTS_PATH = CACHE_DIR / "sec_companyfacts_cache.json"

def load_facts_cache() -> dict:
    try:
        with FACTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"_ts": 0, "data": {}}

def save_facts_cache(data: dict) -> None:
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
        facts = rec.get("facts") if rec else None

    cache["data"][key] = {"_ts": now, "facts": facts}
    save_facts_cache(cache)
    return facts

def _series_from_facts(facts: dict, ns: str, tag: str, prefer_units=("USD","USD/shares")) -> List[float]:
    try:
        units = facts["facts"][ns][tag]["units"]
    except Exception:
        return []
    # choisir l’unité la plus pertinente
    unit_keys = sorted(
        units.keys(),
        key=lambda u: 0 if any(p.lower() in u.lower() for p in prefer_units) else 1
    )
    out: List[float] = []
    for uk in unit_keys:
        arr = units[uk]
        arr = sorted(arr, key=lambda x: (x.get("fy",0), x.get("fp",""), x.get("end","")))
        vals: List[float] = []
        for obs in arr:
            v = obs.get("val")
            try:
                if v is not None:
                    vals.append(float(v))
            except Exception:
                pass
        if len(vals) >= 2:
            out = vals
            break
        elif not out and len(vals) > 0:
            out = vals
    return out

# ------------------------------------------------------------------ Universe

def ensure_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise SystemExit(f"[FATAL] {UNIVERSE_CSV} introuvable. Lance d'abord build_universe.py")

    df = pd.read_csv(UNIVERSE_CSV)
    # détecter la colonne ticker
    ticker_col = None
    for c in ["ticker_yf", "ticker", "Symbol", "symbol"]:
        if c in df.columns:
            ticker_col = c
            break
    if not ticker_col:
        raise SystemExit("[FATAL] universe_in_scope.csv: colonne ticker introuvable")

    # normalisation → nouvelle colonne "ticker_yf"
    df["ticker_yf"] = df[ticker_col].astype(str).str.strip().str.upper()

    # virer vides / tirets / exotique
    df = df[df["ticker_yf"].ne("") & df["ticker_yf"].ne("-")]
    df = df[df["ticker_yf"].str.match(r"^[A-Za-z0-9.\-]+$")]

    # uniques
    df = df.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)
    _log(f"[UNI] in-scope: {len(df)}")
    return df

# ------------------------------------------------------------------ Pipeline principal

def favorable(label: str) -> bool:
    return str(label).upper() in ("BUY", "STRONG_BUY")

def compute_rows(uni: pd.DataFrame) -> pd.DataFrame:
    secmap = sec_ticker_map()
    sectors = read_sector_catalog()

    rows = []
    for _, r in uni.iterrows():
        t = r["ticker_yf"]

        # 1) Prix & ADV
        df = fetch_ohlcv_yf(t)
        lc = last_close(df)
        adv = avg_dollar_vol(df) or 0.0

        # 2) TechScore local
        if df is not None and lc is not None and lc >= MIN_PRICE_FOR_SCORE and len(df) >= 50:
            feats = compute_tech_features(df)
            tech_label, tech_score = tech_label_from_features(feats)
        else:
            tech_label, tech_score = "HOLD", 0.5

        # 3) SEC → shares & MCAP
        cik = secmap.get(t)
        shares = None
        if cik:
            shares = sec_latest_shares_outstanding(cik)
        mcap = (lc * shares) if (lc is not None and shares is not None) else None

        # 4) Proxy fondamentaux → analyst_label/score/votes
        analyst_label, analyst_score, analyst_votes = "HOLD", 0.5, 20
        if cik:
            facts = sec_companyfacts(cik)
            if facts:
                rev = _series_from_facts(facts, "us-gaap",
                                         "RevenueFromContractWithCustomerExcludingAssessedTax", ("USD",))
                if not rev:
                    rev = _series_from_facts(facts, "us-gaap", "Revenues", ("USD",))
                ni = _series_from_facts(facts, "us-gaap", "NetIncomeLoss", ("USD",))
                op = _series_from_facts(facts, "us-gaap", "OperatingIncomeLoss", ("USD",))

                op_margin: List[float] = []
                if rev and op and len(rev) == len(op):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        op_margin = [
                            float(o) / abs(float(r)) if r not in (0, None) else np.nan
                            for o, r in zip(op, rev)
                        ]
                        op_margin = [x for x in op_margin if x == x]  # drop NaN

                eps = _series_from_facts(facts, "us-gaap", "EarningsPerShareDiluted", ("USD/shares","USD"))
                analyst_label, analyst_score, analyst_votes = analyst_label_from_fundamentals(
                    rev, ni, op_margin, eps
                )

        # 5) Sector/industry (fallback Unknown)
        si = sectors.get(t, {})
        sector = si.get("sector", "Unknown")
        industry = si.get("industry", "Unknown")

        # 6) TV proxy = notre Tech composite
        tv_reco = tech_label
        tv_score = tech_score

        # 7) Pillars & bucket
        p_tech = favorable(tech_label)
        p_tv   = favorable(tv_reco)
        p_an   = favorable(analyst_label)
        pillars_met = int(p_tech) + int(p_tv) + int(p_an)

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

        # Ranking simple (pondérations douces + liquidité)
        adv_weight = math.log(1.0 + max(0.0, adv)) / 20.0
        adv_weight = min(0.1, adv_weight)
        rank_score = float(
            (pillars_met / 3.0) * 0.6 +
            (tv_score) * 0.2 +
            (analyst_score) * 0.1 +
            adv_weight
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
            "pillars_met": pillars_met,
            "votes_bin": votes_bin,
            "rank_score": rank_score,
            "bucket": bucket,
        })

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ Filtres & sorties

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre strict: MCAP < 75B
    - Si KEEP_NAN_MCAP=1 → on conserve les NaN (pour inspection).
    - Sinon (défaut) → on les retire (plus propre pour tes signaux).
    """
    mcap = pd.to_numeric(df["mcap_usd_final"], errors="coerce")
    if KEEP_NAN_MCAP:
        mask = (mcap < MCAP_CAP) | mcap.isna()
    else:
        mask = (mcap < MCAP_CAP) & mcap.notna()
    return df[mask].copy()

def write_outputs(out: pd.DataFrame) -> None:
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

def update_signals_history(today: str, out: pd.DataFrame) -> None:
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

# ------------------------------------------------------------------ Main

def main() -> None:
    _log("[STEP] mix_ab_screen_indices starting…")

    # yfinance timeout (soft)
    yf.shared._DFS = {}  # reset cache interne si besoin
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    uni = ensure_universe()
    base = compute_rows(uni)

    # robustesse NaN
    for c in ["price","last","mcap_usd_final","avg_dollar_vol","tv_score","analyst_votes","rank_score"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # filtre MCAP
    base = apply_filters(base)

    # bucket safety
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
        "unique_total": int(len(base)),
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    main()
