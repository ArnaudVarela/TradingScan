# sector_tag.py — Tag secteur GICS des setups marché via yfinance .info, avec CACHE PERSISTANT (sector_map.csv).
# Le libellé est mappé sur la MÊME taxonomie que le panneau Rotation (sector_perf.py) pour croiser setups <-> rotation.
# Coût réseau borné par run (MAX_FETCH) ; le cache s'accumule sur plusieurs runs (le secteur d'un titre ne change ~jamais).
import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
CACHE = ROOT / "sector_map.csv"
SETUPS = ROOT / "market_setups.csv"
MAX_FETCH = int(os.getenv("SECTOR_MAX_FETCH", "300"))     # borne le coût/CI par run
THROTTLE = float(os.getenv("SECTOR_THROTTLE", "0.35"))    # anti rate-limit yfinance

# yfinance .info["sector"] (taxo Morningstar) -> GICS (libellés exacts de sector_perf.py / panneau Rotation)
YF_TO_GICS = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Industrials": "Industrials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
}

def _load_cache():
    if CACHE.exists():
        try:
            c = pd.read_csv(CACHE, keep_default_na=False)
            return {str(k).upper(): str(v) for k, v in zip(c["ticker"], c["sector"])}
        except Exception:
            pass
    return {}

def _fetch_sector(t):
    """GICS mappé si trouvé ; '' si le titre n'a pas de secteur (ETF/shell) ; None si échec réseau (à retenter)."""
    for attempt in range(2):
        try:
            info = yf.Ticker(t).info
            yfs = (info or {}).get("sector")
            if yfs:
                return YF_TO_GICS.get(yfs, yfs)
            return ""
        except Exception:
            if attempt == 0:
                time.sleep(1.5)   # rate limit probable -> pause + 1 retry
    return None

def main():
    if not SETUPS.exists():
        print("[SECTOR TAG] market_setups.csv absent — skip."); return
    d = pd.read_csv(SETUPS, keep_default_na=False)   # tickers 'NA'/'NAN'/'NULL' littéraux, cohérent avec le cache
    d["ticker"] = d["ticker"].astype(str).str.upper()
    cache = _load_cache()

    # priorité aux plus hauts scores, uniquement les tickers absents du cache
    ordered = d.sort_values("score", ascending=False)["ticker"].tolist()
    todo = [t for t in ordered if t not in cache][:MAX_FETCH]
    print(f"[SECTOR TAG] cache={len(cache)} · à fetch={len(todo)} (plafond {MAX_FETCH})")

    got = 0
    for t in todo:
        s = _fetch_sector(t)
        if s is not None:      # None = échec réseau -> on NE cache PAS (retry au prochain run)
            cache[t] = s
            got += 1
        time.sleep(THROTTLE)

    pd.DataFrame(sorted(cache.items()), columns=["ticker", "sector"]).to_csv(CACHE, index=False)

    d["sector"] = d["ticker"].map(cache).fillna("")
    d.to_csv(SETUPS, index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        d.to_csv(pub / "market_setups.csv", index=False)

    cov = int((d["sector"].astype(str).str.len() > 0).sum())
    print(f"[SECTOR TAG] +{got} fetchés · couverture {cov}/{len(d)} setups · cache total {len(cache)}")
    vc = d[d["sector"].astype(str).str.len() > 0]["sector"].value_counts()
    if not vc.empty:
        print(vc.to_string())

if __name__ == "__main__":
    main()
