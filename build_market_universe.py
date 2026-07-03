# build_market_universe.py — Univers LARGE small/micro/nano cap (< $5B, TOUS secteurs) via API Nasdaq (gratuit, sans clé).
# Complète l'univers thématique hard-tech : ici on scanne le marché entier sous $5B pour ne rater aucun secteur.
import os
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
MCAP_MIN = float(os.getenv("MARKET_MCAP_MIN", "1e8"))    # $100M (exclut les ultra-nano/shells)
MCAP_MAX = float(os.getenv("MARKET_MCAP_MAX", "5e9"))    # $5B
PRICE_MIN = float(os.getenv("MARKET_PRICE_MIN", "1.0"))  # exclut le sub-penny
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
SYM_OK = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")

def _num(s):
    try:
        return float(re.sub(r"[^0-9.]", "", str(s)))
    except Exception:
        return 0.0

def main():
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&offset=0&download=true"
    r = requests.get(url, headers=H, timeout=60)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    out = []
    for x in rows:
        sym = str(x.get("symbol", "")).strip().upper().replace("/", "-")   # BRK/B -> BRK-B (format yfinance)
        if not SYM_OK.match(sym):
            continue
        mc, px = _num(x.get("marketCap")), _num(x.get("lastsale"))
        if not (MCAP_MIN <= mc <= MCAP_MAX) or px < PRICE_MIN:
            continue
        out.append({"ticker": sym, "mcap_usd": mc, "price": round(px, 2), "name": str(x.get("name", ""))[:60]})
    cols = ["ticker", "mcap_usd", "price", "name"]
    if not out:
        pd.DataFrame(columns=cols).to_csv(ROOT / "market_universe.csv", index=False)
        print("[MARKET] 0 titre retenu (schéma Nasdaq changé ? filtre trop strict ?) -> market_universe.csv vide.")
        return
    df = pd.DataFrame(out).drop_duplicates("ticker").sort_values("mcap_usd", ascending=False).reset_index(drop=True)
    df.to_csv(ROOT / "market_universe.csv", index=False)
    print(f"[MARKET] {len(df)} titres <= ${MCAP_MAX/1e9:g}B (>= ${MCAP_MIN/1e6:g}M, prix >= ${PRICE_MIN:g}) -> market_universe.csv")
    print(df.head(8)[["ticker", "mcap_usd", "price", "name"]].to_string(index=False))

if __name__ == "__main__":
    main()
