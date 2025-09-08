# tools/cache_layer.py
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Dict, Any, Iterable
import pandas as pd

CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(exist_ok=True, parents=True)
CACHE_PATH = CACHE_DIR / "enrich_cache.parquet"

# TTL par champ (secondes)
TTL = {
    "sector": 3650*24*3600,     # ~10 ans (quasi permanent)
    "industry": 3650*24*3600,
    "price": 24*3600,           # 1 jour (assez pour CI quotidienne)
    "mcap_usd": 3*24*3600,      # 3 jours
    "tv": 6*3600,               # 6 heures
    "analyst": 3*24*3600,       # 3 jours
}

SCHEMA = [
    "ticker_yf",
    "sector","sector_ts",
    "industry","industry_ts",
    "price","price_ts",
    "mcap_usd","mcap_usd_ts",
    "tv_score","tv_reco","tv_ts",
    "analyst_bucket","analyst_votes","analyst_ts",
]

def _now() -> float: return time.time()

def load() -> pd.DataFrame:
    if CACHE_PATH.exists():
        try:
            df = pd.read_parquet(CACHE_PATH)
            # garantir schéma
            for c in SCHEMA:
                if c not in df.columns: df[c] = None
            return df[SCHEMA].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=SCHEMA)

def save(df: pd.DataFrame) -> None:
    try:
        df[SCHEMA].to_parquet(CACHE_PATH, index=False)
    except Exception:
        # fallback CSV si parquet indisponible
        df[SCHEMA].to_csv(CACHE_PATH.with_suffix(".csv"), index=False)

def _fresh(ts: Any, ttl_key: str) -> bool:
    if ts is None or pd.isna(ts): return False
    try:
        return float(ts) > (_now() - TTL[ttl_key])
    except Exception:
        return False

def get_many(df: pd.DataFrame, tickers: Iterable[str], fields: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Retourne {ticker: {field: value,...}} pour les champs 'fresh'."""
    fields = list(fields)
    out: Dict[str, Dict[str, Any]] = {}
    sub = df[df["ticker_yf"].isin(list(tickers))]
    for _, row in sub.iterrows():
        t = row["ticker_yf"]
        keep: Dict[str, Any] = {}
        for f in fields:
            if f == "sector" and _fresh(row.get("sector_ts"), "sector"):
                keep["sector"] = row.get("sector")
            elif f == "industry" and _fresh(row.get("industry_ts"), "industry"):
                keep["industry"] = row.get("industry")
            elif f == "price" and _fresh(row.get("price_ts"), "price"):
                keep["price"] = row.get("price")
            elif f == "mcap_usd" and _fresh(row.get("mcap_usd_ts"), "mcap_usd"):
                keep["mcap_usd"] = row.get("mcap_usd")
            elif f in ("tv_score","tv_reco") and _fresh(row.get("tv_ts"), "tv"):
                keep["tv_score"] = row.get("tv_score")
                keep["tv_reco"] = row.get("tv_reco")
            elif f in ("analyst_bucket","analyst_votes") and _fresh(row.get("analyst_ts"), "analyst"):
                keep["analyst_bucket"] = row.get("analyst_bucket")
                keep["analyst_votes"] = row.get("analyst_votes")
        if keep:
            out[t] = keep
    return out

def upsert(df: pd.DataFrame, ticker: str, **fields) -> pd.DataFrame:
    """fields peut contenir: sector, industry, price, mcap_usd, tv_score, tv_reco, analyst_bucket, analyst_votes"""
    ts_map = {
        "sector": "sector_ts",
        "industry": "industry_ts",
        "price": "price_ts",
        "mcap_usd": "mcap_usd_ts",
        "tv_score": "tv_ts",
        "tv_reco": "tv_ts",
        "analyst_bucket": "analyst_ts",
        "analyst_votes": "analyst_ts",
    }
    if ticker not in set(df["ticker_yf"]):
        df.loc[len(df)] = {c: None for c in SCHEMA}
        df.loc[len(df)-1, "ticker_yf"] = ticker
    idx = df.index[df["ticker_yf"] == ticker][0]
    now = _now()
    for k, v in fields.items():
        if k not in ts_map: continue
        df.at[idx, k] = v
        ts_col = ts_map[k]
        # si on écrit tv_score ou tv_reco on rafraîchit tv_ts une fois
        if ts_col == "tv_ts":
            df.at[idx, "tv_ts"] = now
        elif ts_col == "analyst_ts":
            df.at[idx, "analyst_ts"] = now
        else:
            df.at[idx, ts_col] = now
    return df
