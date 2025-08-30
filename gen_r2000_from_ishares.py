#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère russell2000.csv à partir des holdings iShares IWM (Russell 2000).
Stratégies en cascade:
  1) CSV AJAX (IWM_holdings)
  2) CSV AJAX (holdings)
  3) Parse HTML holdings (table)
  4) Parse JSON embarqué dans la page
Sortie: fichier CSV "russell2000.csv" avec 1 colonne "Ticker".
"""

import io
import re
import sys
import time
import json
import gzip
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

PRODUCT_URL = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
HOLDINGS_URL = PRODUCT_URL + "/holdings"

CSV_URLS = [
    PRODUCT_URL + "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
    PRODUCT_URL + "/1467271812596.ajax?fileType=csv&fileName=holdings&dataType=fund",
]

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Referer": HOLDINGS_URL,
    "DNT": "1",
    "Connection": "keep-alive",
}

XHR_HEADERS = dict(UA_HEADERS, **{
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
})

def session_with_retries() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def parse_tickers_from_df(df: pd.DataFrame) -> list[str]:
    # essaie différentes colonnes possibles
    cands = ["Ticker", "Ticker Symbol", "TickerSymbol", "Symbol", "Code", "Local Ticker", "LocalTicker"]
    col = None
    lower = {c.lower(): c for c in df.columns}
    for name in cands:
        if name.lower() in lower:
            col = lower[name.lower()]
            break
    if not col:
        # heuristique: une colonne qui ressemble à ticker/symbol
        for c in df.columns:
            lc = c.lower()
            if "ticker" in lc or lc in {"symbol", "code"}:
                col = c
                break
    if not col:
        return []

    s = (
        df[col].astype(str)
        .str.strip()
        .replace({"nan": None, "None": None})
        .dropna()
        .str.replace("\u200b", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    # filtre symboles plausibles (BRK.B, ABC, ABCD, ABCD-A, etc.)
    s = s[s.str.match(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")]
    # enlève cash/FX/deriv
    bad = {"CASH", "USD", "FX", "FUT", "SWAP", "OPTION"}
    s = s[~s.str.upper().isin(bad)]
    return sorted(set(s.tolist()))

def try_csv_endpoints(sess: requests.Session) -> list[str]:
    for i, url in enumerate(CSV_URLS, 1):
        try:
            print(f"[csv {i}] GET {url}")
            r = sess.get(url, headers=XHR_HEADERS, timeout=60)
            if r.status_code != 200:
                print(f"  -> status {r.status_code}")
                continue

            content = r.content
            # Gzip ?
            if r.headers.get("Content-Encoding", "").lower() == "gzip":
                try:
                    content = gzip.decompress(content)
                except Exception:
                    pass

            # Essaye CSV via pandas
            # Certaines réponses ont du BOM/encodage exotique → on essaie plusieurs encodings
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    if not df.empty:
                        tickers = parse_tickers_from_df(df)
                        print(f"  -> parsed {len(tickers)} tickers")
                        if tickers:
                            return tickers
                except Exception:
                    continue

            # Parfois c'est du JSON au lieu de CSV
            try:
                data = r.json()
                if isinstance(data, dict):
                    for key in ("holdings", "positions", "basket", "data"):
                        if key in data and isinstance(data[key], list) and data[key]:
                            df = pd.json_normalize(data[key])
                            tickers = parse_tickers_from_df(df)
                            print(f"  -> parsed {len(tickers)} tickers (json)")
                            if tickers:
                                return tickers
            except Exception:
                pass

        except Exception as e:
            print(f"  -> fail: {type(e).__name__}: {e}")
        time.sleep(0.7)
    return []

def try_html_table(sess: requests.Session) -> list[str]:
    print(f"[html] GET {HOLDINGS_URL}")
    r = sess.get(HOLDINGS_URL, headers=UA_HEADERS, timeout=60)
    if r.status_code != 200:
        print(f"  -> status {r.status_code}")
        return []
    # pandas.read_html peut trouver 0..N tableaux; on les teste
    tables = []
    try:
        tables = pd.read_html(r.text, extract_links="body")
    except Exception:
        # deuxième tentative sans extract_links
        try:
            tables = pd.read_html(r.text)
        except Exception:
            tables = []
    for idx, t in enumerate(tables):
        if isinstance(t, tuple) and len(t) == 2:
            t = t[0]  # si extract_links="body", pandas renvoie (val, link)
        if not isinstance(t, pd.DataFrame) or t.empty:
            continue
        tickers = parse_tickers_from_df(t)
        if tickers:
            print(f"  -> table {idx}: {len(tickers)} tickers")
            return tickers
    print("  -> no usable table")
    return []

def try_embedded_json(sess: requests.Session) -> list[str]:
    print(f"[json] GET {HOLDINGS_URL} (embedded)")
    r = sess.get(HOLDINGS_URL, headers=UA_HEADERS, timeout=60)
    if r.status_code != 200:
        print(f"  -> status {r.status_code}")
        return []
    html = r.text

    # Cherche des JSON embarqués contenant 'holdings'/'positions'
    patterns = [
        r"var\s+.*?=\s*(\{.*?\"holdings\".*?\});",
        r"window\.__APP__\s*=\s*(\{.*?\});",
        r"\"holdings\"\s*:\s*(\[[^\]]+\])",
        r"\"positions\"\s*:\s*(\[[^\]]+\])",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.DOTALL)
        if not m:
            continue
        blob = m.group(1)
        # nettoyage de JSON “relâché”
        blob = re.sub(r"(?s)//.*?$", "", blob)  # enlève commentaires
        blob = re.sub(r",\s*}", "}", blob)
        blob = re.sub(r",\s*]", "]", blob)
        try:
            data = json.loads(blob)
        except Exception:
            # Parfois on matche directement la liste [...]
            try:
                data = json.loads(f'{{"holdings": {blob}}}')
            except Exception:
                continue

        # normalise
        if isinstance(data, dict):
            for key in ("holdings", "positions", "data"):
                if key in data and isinstance(data[key], list) and data[key]:
                    df = pd.json_normalize(data[key])
                    tickers = parse_tickers_from_df(df)
                    if tickers:
                        print(f"  -> embedded {key}: {len(tickers)} tickers")
                        return tickers
        elif isinstance(data, list):
            df = pd.json_normalize(data)
            tickers = parse_tickers_from_df(df)
            if tickers:
                print(f"  -> embedded list: {len(tickers)} tickers")
                return tickers

    print("  -> no embedded JSON found")
    return []

def main():
    sess = session_with_retries()

    # Prime le cookie en visitant la page produit (parfois nécessaire)
    try:
        sess.get(PRODUCT_URL, headers=UA_HEADERS, timeout=30)
        time.sleep(0.5)
        sess.get(HOLDINGS_URL, headers=UA_HEADERS, timeout=30)
    except Exception:
        pass

    # Stratégie 1 & 2 : CSV AJAX
    tickers = try_csv_endpoints(sess)
    if not tickers:
        # Stratégie 3 : tableau HTML
        tickers = try_html_table(sess)
    if not tickers:
        # Stratégie 4 : JSON embarqué
        tickers = try_embedded_json(sess)

    if not tickers:
        print("❌ Impossible de récupérer la liste IWM (iShares) après 4 stratégies.")
        sys.exit(2)

    out = pd.DataFrame({"Ticker": tickers})
    out.to_csv("russell2000.csv", index=False)
    print(f"✅ Écrit russell2000.csv avec {len(out)} tickers.")

if __name__ == "__main__":
    main()
