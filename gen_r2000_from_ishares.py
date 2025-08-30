#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère russell2000.csv à partir des holdings iShares IWM (Russell 2000).
Sortie: fichier CSV avec une colonne "Ticker" (compatible avec ton pipeline).
"""

import io
import re
import sys
import time
import json
import gzip
import pandas as pd
import requests

CANDIDATE_URLS = [
    # Endpoints connus chez iShares (peuvent changer, on en teste plusieurs)
    # 1) Ajax CSV “historique”
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
    # 2) Endpoint JSON holdings
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/holdings",
    # 3) Autre pattern "downloadFile"
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=holdings",
]

UA = {
    "User-Agent":
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
}

def fetch_csv_bytes(url: str) -> bytes | None:
    r = requests.get(url, headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    ctype = (r.headers.get("Content-Type") or "").lower()
    # direct CSV
    if "text/csv" in ctype or r.text.strip().startswith(("Ticker,", "\"Ticker\",")):
        return r.content

    # parfois JSON -> on essaie d’en extraire un CSV
    if "application/json" in ctype or r.text.strip().startswith("{"):
        try:
            data = r.json()
            # certains endpoints renvoient un objet complexe; on tente de trouver la table des positions
            # iShares a souvent "holdings" ou "positions" dans le JSON
            for key in ("holdings", "positions", "basket", "data"):
                if key in data and isinstance(data[key], list) and data[key]:
                    df = pd.json_normalize(data[key])
                    return df.to_csv(index=False).encode("utf-8")
        except Exception:
            return None

    # parfois GZIP (rare)
    if r.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            return gzip.decompress(r.content)
        except Exception:
            pass

    # fallback: peut-être que c’est du CSV malgré un content-type exotique
    if b"," in r.content[:2048]:
        return r.content

    return None


def parse_tickers_from_csv_bytes(b: bytes) -> list[str]:
    # Essaye encodages usuels
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(b), encoding=enc)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return []

    # Harmonise les noms de colonnes (Ticker, Ticker Symbol, TickerSymbol…)
    def find_col(cands):
        cols_lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    ticker_col = find_col(["Ticker", "Ticker Symbol", "TickerSymbol", "Symbol"])
    if not ticker_col:
        # parfois la colonne s’appelle "Code" ou "Local Ticker"
        ticker_col = find_col(["Code", "Local Ticker", "LocalTicker"])
    if not ticker_col:
        # Dernière chance: colonne qui ressemble à 'ticker'
        for c in df.columns:
            if "ticker" in c.lower() or c.lower() in {"symbol", "code"}:
                ticker_col = c
                break

    if not ticker_col:
        # rien à faire
        return []

    s = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .replace({"nan": None, "None": None})
        .dropna()
    )

    # vire cash/derivés, lignes vides, unicodes chelous
    s = (
        s.str.replace("\u200b", "", regex=False)  # zero-width space
         .str.replace(r"\s+", "", regex=True)
    )

    # filtre symboles équitables plausibles (ex: ABC, ABCD, ABCD.A, BRK.B, etc.)
    s = s[s.str.match(r"^[A-Za-z][A-Za-z0-9\.\-]{0,9}$")]

    # certains CSV mettent des lignes “CASH”/“USD” → on les jarte
    bad = {"CASH", "USD", "FX", "FUT", "SWAP", "OPTION"}
    s = s[~s.str.upper().isin(bad)]

    tickers = sorted(set(s.tolist()))
    return tickers


def main():
    ticks: list[str] = []
    for i, url in enumerate(CANDIDATE_URLS, 1):
        try:
            print(f"[try {i}] GET {url}")
            b = fetch_csv_bytes(url)
            if not b:
                print("  -> no csv here")
                continue
            ts = parse_tickers_from_csv_bytes(b)
            print(f"  -> parsed {len(ts)} tickers")
            if ts:
                ticks = ts
                break
        except Exception as e:
            print(f"  -> fail: {type(e).__name__}: {e}")
        time.sleep(0.8)

    if not ticks:
        print("❌ Impossible de récupérer la liste IWM (iShares).")
        sys.exit(2)

    # Sauvegarde au format attendu par ton screener
    out = pd.DataFrame({"Ticker": ticks})
    out.to_csv("russell2000.csv", index=False)
    print(f"✅ Écrit russell2000.csv avec {len(out)} tickers.")


if __name__ == "__main__":
    main()
