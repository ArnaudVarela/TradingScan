#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère russell2000.csv via fonds répliquant (ou approchant) le Russell 2000.

Plan:
  1) SPDR UCITS ZPRR (Russell 2000) - XLSX "Download Daily Holdings"
     - lit toutes les feuilles
     - auto-détection de la ligne d'entête
     - récupère RIC / Bloomberg Ticker / Ticker / Symbol
  2) (fallback) rien pour l’instant (Schwab 403 en CI). On sort si SPDR échoue.

Sortie: russell2000.csv (colonne unique "Ticker")
"""

import io
import re
import sys
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

SPDR_XLSX_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"

UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}

# -------- utilitaires nettoyage --------
BAD = {"USD", "CASH", "FX", "SWAP", "OPTION", "FUT", "FUTURES", "CURRENCY"}
RIC_SUFFIX = re.compile(r"\.[A-Za-z]{1,4}$")  # .OQ .N .K .L etc.
BLOOM_BG_SUFFIX = re.compile(r"\s+(US|UW|UQ|LN|GY|FP|NA|IM|SM|HK|JP|CN|SW|AU)\b", re.IGNORECASE)
EQUITY_SUFFIX = re.compile(r"\s+US\s+EQUITY$", re.IGNORECASE)

def clean_symbol(x: str) -> str | None:
    if not x:
        return None
    t = str(x).strip().replace("\u200b", "")
    if not t or t.upper() in BAD:
        return None
    # variants bloomberg: "AAPL US", "AAPL US Equity"
    t = EQUITY_SUFFIX.sub("", t)
    t = BLOOM_BG_SUFFIX.sub("", t)
    # variants RIC: "AAPL.OQ"
    t = RIC_SUFFIX.sub("", t)
    # enlever espaces internes
    t = re.sub(r"\s+", "", t)
    # garder symboles US plausibles
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]{0,9}", t):
        return t.upper()
    return None

def unique_sorted(seq):
    return sorted({x for x in seq if x})

# -------- extraction depuis DataFrame --------
CAND_COLS = [
    "Ticker", "Ticker Symbol", "TickerSymbol", "Symbol",
    "RIC", "Reuters Ticker", "Reuters RIC",
    "Bloomberg Ticker", "Bloomberg", "BBG Ticker", "BBG"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # aplatit MultiIndex + cast en str
    new = []
    for c in df.columns:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if pd.notna(x)]
            name = " ".join(parts).strip()
        else:
            name = str(c).strip()
        new.append(name)
    df2 = df.copy()
    df2.columns = new
    return df2

def find_header_row(df: pd.DataFrame) -> int | None:
    """
    Cherche la ligne qui contient une mention d’une colonne clé (ticker/ric/bloomberg/symbol).
    On scanne les ~40 premières lignes.
    """
    max_probe = min(len(df), 40)
    key_rx = re.compile(r"(ticker|ric|reuters|bloomberg|symbol)", re.IGNORECASE)
    for i in range(max_probe):
        row = df.iloc[i]
        # transforme tout en str pour comparer
        s = " | ".join(str(x) for x in row.values if pd.notna(x))
        if key_rx.search(s):
            return i
    return None

def extract_tickers_any(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    # détecte header si besoin
    hdr = find_header_row(df)
    if hdr is not None:
        df = df.iloc[hdr:].reset_index(drop=True)
        df.columns = [str(x).strip() for x in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True)
    df = normalize_cols(df)
    # choisi la meilleure colonne dispo
    lower_map = {c.lower(): c for c in df.columns}
    chosen = None
    # priorité: RIC, puis Bloomberg, puis Ticker/Symbol
    priority = [
        "ric", "reuters ric", "reuters ticker",
        "bloomberg ticker", "bloomberg", "bbg ticker", "bbg",
        "ticker", "ticker symbol", "tickersymbol", "symbol"
    ]
    for key in priority:
        if key in lower_map:
            chosen = lower_map[key]
            break
    if not chosen:
        # heuristique
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ["ric", "bloomberg", "ticker", "symbol", "code"]):
                chosen = c
                break
    if not chosen:
        return []

    ser = (
        df[chosen]
        .astype(str)
        .str.strip()
        .replace({"nan": None, "": None, "None": None})
        .dropna()
    )
    ticks = [clean_symbol(v) for v in ser.tolist()]
    return unique_sorted(ticks)

# -------- téléchargement SPDR --------
def session_with_retries() -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s

def fetch_spdr_zprr() -> list[str]:
    print(f"[SPDR] GET {SPDR_XLSX_URL}")
    s = session_with_retries()
    r = s.get(SPDR_XLSX_URL, headers=UA, timeout=60)
    if r.status_code != 200 or not r.content:
        print(f"  -> status {r.status_code}")
        return []
    # lit toutes les feuilles
    try:
        book = pd.read_excel(io.BytesIO(r.content), sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"  -> read_excel fail: {e}")
        return []
    all_ticks = []
    for name, df in (book or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        ticks = extract_tickers_any(df)
        if ticks:
            print(f"  -> sheet '{name}': {len(ticks)} tickers")
            all_ticks.extend(ticks)
    all_ticks = unique_sorted(all_ticks)
    print(f"  -> parsed {len(all_ticks)} tickers (SPDR XLSX)")
    return all_ticks

def main():
    ticks = fetch_spdr_zprr()
    if not ticks:
        print("❌ Impossible d’extraire des tickers depuis le XLSX SPDR.")
        sys.exit(2)
    pd.DataFrame({"Ticker": ticks}).to_csv("russell2000.csv", index=False)
    print(f"✅ Écrit russell2000.csv avec {len(ticks)} tickers (source: SPDR UCITS ZPRR).")

if __name__ == "__main__":
    main()
