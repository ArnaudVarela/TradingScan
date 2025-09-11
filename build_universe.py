#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_universe.py (robuste)
- SP500 via datasets repo (stable)
- Russell 1000 via iShares IWB holdings (fallbacks)
- Nasdaq (composite-ish) via plusieurs miroirs nasdaqtrader
- Tolérant aux erreurs: retries + fallback + union/dedup
- Met en cache les tickers précédents et écrit raw_universe.csv + universe_in_scope.csv

AUCUN filtre secteur / mcap ici (le filtre mcap < 75B est appliqué dans mix_ab_screen_indices.py).
"""

from __future__ import annotations
import os, io, time, csv
from pathlib import Path
from typing import List, Set, Tuple
import requests
import pandas as pd

ROOT = Path(".")
PUB = ROOT / "dashboard" / "public"
PUB.mkdir(parents=True, exist_ok=True)

RAW = ROOT / "raw_universe.csv"
SCOPE = ROOT / "universe_in_scope.csv"
CACHE_PREV = PUB / "universe_today.csv"   # pour comparer/delta si tu veux
TIMEOUT = 30
HDRS = {
    "User-Agent": "Mozilla/5.0 (universe-builder; +https://github.com/)",
    "Accept": "text/csv,application/json,*/*;q=0.8",
    "Connection": "close",
}

def log(msg: str): print(msg, flush=True)

def http_get(url: str, tries: int = 3, sleep: float = 1.5) -> requests.Response | None:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=HDRS, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(sleep * (k + 1))
    log(f"[WARN] GET failed for {url}: {last}")
    return None

# ---------- SP500
def fetch_sp500() -> Set[str]:
    # Dataset public et stable
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    r = http_get(url)
    if not r: return set()
    try:
        df = pd.read_csv(io.StringIO(r.text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        syms = df[col].astype(str).str.upper().str.strip().tolist()
        return set([s for s in syms if s and s.isascii()])
    except Exception as e:
        log(f"[WARN] SP500 parse failed: {e}")
        return set()

# ---------- Russell 1000 (via iShares IWB)
def fetch_russell1000() -> Set[str]:
    urls = [
        # CSV holdings IWB (généralement propre)
        "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
        # mirroir (parfois identique)
        "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf?switchLocale=y&siteEntryPassthrough=true",
    ]
    for url in urls:
        r = http_get(url)
        if not r:
            continue
        txt = r.text
        # Certains CSV iShares commencent par un bloc méta: on saute jusqu'à la ligne qui contient 'Ticker'
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        # trouve l’index de l’en-tête
        idx = None
        for i, ln in enumerate(lines[:50]):
            if "Ticker" in ln and "Name" in ln:
                idx = i
                break
        if idx is None:
            # parfois iShares renvoie HTML si protection; on essaie quand même un parse tolérant
            try:
                df = pd.read_csv(io.StringIO(txt), engine="python", on_bad_lines="skip")
            except Exception:
                continue
        else:
            df = pd.read_csv(io.StringIO("\n".join(lines[idx:])), engine="python", on_bad_lines="skip")
        # Col possibles: Ticker / Holdings Ticker
        col = None
        for c in df.columns:
            if str(c).lower() in ("ticker", "holdings ticker", "holding ticker"):
                col = c; break
        if col is None:
            # essaie de détecter la colonne ticker
            for c in df.columns:
                if "ticker" in str(c).lower():
                    col = c; break
        if col is None:
            log("[WARN] R1000: no ticker column detected")
            continue
        syms = (
            df[col]
            .dropna()
            .astype(str)
            .str.replace(".","-", regex=False)   # parfois des formats bizarres
            .str.upper().str.strip()
            .tolist()
        )
        syms = [s for s in syms if s and s.isascii()]
        if syms:
            return set(syms)
    log("[WARN] Russell1000 fetch failed (iShares).")
    return set()

# ---------- Nasdaq listed (mêmes tickers que composite ± ETFs), multi-miroirs


# ---------- main
def main():
    log("[STEP] build_universe starting…")

    spx = fetch_sp500()
    log(f"[SRC] SPX: {len(spx)} rows")

    r1k = fetch_russell1000()
    log(f"[SRC] R1000 (iShares/IWB): {len(r1k)} rows")

    nas = fetch_nasdaq()
    log(f"[SRC] Nasdaq listed: {len(nas)} rows")

    # Union + dédup
    uni: Set[str] = set()
    uni |= spx
    uni |= r1k
    uni |= nas

    # Nettoyages simples
    uni = {s.replace("/", "-").replace(".", "-").strip().upper() for s in uni if s}
    # remove obvious bads
    bad = {"N/A", "NA", "NONE"}
    uni = {s for s in uni if s not in bad}

    tickers = sorted(uni)
    log(f"[SAVE] raw_universe.csv | rows={len(tickers)}")
    pd.DataFrame({"ticker_yf": tickers}).to_csv(RAW, index=False)

    # Pas de filtre mcap ici — on garde la même liste pour l’étape suivante
    pd.DataFrame({"ticker_yf": tickers}).to_csv(SCOPE, index=False)
    # Copie dans public pour debug/trace (optionnel)
    pd.DataFrame({"ticker_yf": tickers}).to_csv(PUB / "universe_today.csv", index=False)

    log("[DONE] build_universe finished.")

if __name__ == "__main__":
    main()
