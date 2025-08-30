#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère russell2000.csv via des sources publiques de fonds qui répliquent
ou approximent le Russell 2000.

Ordre des tentatives (du plus complet/fiable au moins complet) :
  1) SPDR UCITS ZPRR "Download Daily Holdings" (XLSX)
  2) Schwab SCHA page "All holdings" (HTML) — peut ne donner que 100 si pagination JS

Sortie : russell2000.csv (colonne unique "Ticker")
"""

import re
import sys
import time
import io
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from io import StringIO

# --- Sources ---
SPDR_XLSX_URL = (
    # lien "Download Daily Holdings" vu sur la page produit ZPRR (UCITS R2000)
    # si jamais ça change, on pourrait re-parcourir la page pour retrouver le lien.
    "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"
)
SCHWAB_ALL_HOLDINGS_URL = "https://www.schwabassetmanagement.com/allholdings/SCHA"

UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}

def session_with_retries() -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s

# --- Normalisation de tickers ---
RIC_SUFFIX_RX = re.compile(r"\.[A-Za-z]+$")  # ex: AAPL.OQ -> AAPL
BAD_TICKERS = {"USD", "CASH", "FX", "SWAP", "OPTION", "FUT"}

def clean_ticker(x: str) -> str | None:
    if not x:
        return None
    t = str(x).strip().upper().replace("\u200b", "")
    if t in BAD_TICKERS:
        return None
    # supprime suffixes RIC (".OQ", ".N", ".K", etc.)
    t = RIC_SUFFIX_RX.sub("", t)
    # supprime espaces internes
    t = re.sub(r"\s+", "", t)
    # garde symboles plausibles US
    if re.fullmatch(r"[A-Z][A-Z0-9\.\-]{0,9}", t):
        return t
    return None

def unique_sorted(seq):
    return sorted({x for x in seq if x})

# --- Parsers ---
def parse_tickers_from_dataframe(df: pd.DataFrame) -> list[str]:
    """Cherche des colonnes candidates et normalise."""
    if df is None or df.empty:
        return []
    # aplatis colonnes si MultiIndex
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if pd.notna(x)]
            name = " ".join(parts).strip()
        else:
            name = str(c).strip()
        new_cols.append(name)
    df = df.copy()
    df.columns = new_cols

    candidates = [
        "Ticker", "RIC", "Reuters Ticker", "Bloomberg Ticker", "Bloomberg", "Symbol",
        "Ticker Symbol", "TickerSymbol", "Local Ticker", "LocalTicker", "Code"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    col = None
    for name in candidates:
        if name.lower() in lower_map:
            col = lower_map[name.lower()]
            break
    if not col:
        # Heuristique: toute colonne dont le nom contient 'ticker' ou 'symbol' ou == 'code'
        for c in df.columns:
            lc = c.lower()
            if "ticker" in lc or lc in {"symbol", "code", "ric"}:
                col = c
                break
    if not col:
        return []

    ser = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"nan": None, "": None, "None": None})
        .dropna()
    )
    ticks = [clean_ticker(v) for v in ser.tolist()]
    return unique_sorted(ticks)

def fetch_spdr_zprr(session: requests.Session) -> list[str]:
    print(f"[SPDR] GET {SPDR_XLSX_URL}")
    r = session.get(SPDR_XLSX_URL, headers=UA, timeout=60)
    if r.status_code != 200 or not r.content:
        print(f"  -> status {r.status_code}")
        return []
    # XLSX -> DataFrame
    try:
        df = pd.read_excel(io.BytesIO(r.content), engine="openpyxl")
    except Exception as e:
        print(f"  -> read_excel fail: {e}")
        return []
    ticks = parse_tickers_from_dataframe(df)
    print(f"  -> parsed {len(ticks)} tickers (SPDR XLSX)")
    return ticks

def fetch_schwab_scha(session: requests.Session) -> list[str]:
    """Parse la 1ère page HTML (100 premières lignes). Fallback si SPDR indispo."""
    print(f"[SCHWAB] GET {SCHWAB_ALL_HOLDINGS_URL}")
    r = session.get(SCHWAB_ALL_HOLDINGS_URL, headers=UA, timeout=60)
    if r.status_code != 200:
        print(f"  -> status {r.status_code}")
        return []
    html = r.text
    # Les blocs affichent `Symbol` sur une ligne et le symbole sur la ligne suivante
    # On capture tout ce qui suit 'Symbol' jusqu'à fin de ligne, puis la prochaine ligne non vide.
    # Plus robuste: rechercher des motifs style '\nSymbol\n {ticker}\n'
    raw = re.findall(r"Symbol\s*([\s\S]*?)\n", html)
    # … mais selon la structure, c'est plus fiable d'extraire tous les tickers par regex globale :
    # (Lettres/chiffres/.- entre balises)
    # On s'aide du fait que la page montre "Symbol\n    TICKER"
    ticks = []
    for m in re.finditer(r"Symbol\s*</?[^>]*>\s*([A-Za-z0-9.\-]{1,10})", html):
        ticks.append(clean_ticker(m.group(1)))
    # fallback encore plus simple: toutes les occurrences après 'Symbol' en texte
    if not ticks:
        lines = [l.strip() for l in html.splitlines()]
        for i, ln in enumerate(lines):
            if ln.strip() == "Symbol" and i + 1 < len(lines):
                ticks.append(clean_ticker(lines[i + 1]))
    ticks = unique_sorted(ticks)
    print(f"  -> parsed {len(ticks)} tickers (Schwab page, peut être partiel)")
    return ticks

def main():
    s = session_with_retries()

    # 1) SPDR UCITS (idéal)
    try:
        spdr = fetch_spdr_zprr(s)
    except Exception as e:
        print(f"[SPDR] error: {type(e).__name__}: {e}")
        spdr = []

    # 2) Schwab SCHA (fallback partiel si pas de pagination)
    schwab = []
    if not spdr:
        try:
            schwab = fetch_schwab_scha(s)
        except Exception as e:
            print(f"[SCHWAB] error: {type(e).__name__}: {e}")

    tickers = spdr or schwab
    if not tickers:
        print("❌ Aucune source n'a renvoyé de tickers utilisables.")
        sys.exit(2)

    pd.DataFrame({"Ticker": tickers}).to_csv("russell2000.csv", index=False)
    src = "SPDR UCITS (ZPRR XLSX)" if spdr else "Schwab SCHA (HTML partiel)"
    print(f"✅ Écrit russell2000.csv avec {len(tickers)} tickers — source: {src}")

if __name__ == "__main__":
    main()
