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
from io import StringIO
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

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Colnames -> str; aplatit MultiIndex en 'lvl1 lvl2 ...' et strip."""
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
    return df

def parse_tickers_from_df(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    df = _normalize_columns(df)

    # essai de colonnes candidates
    cands = ["Ticker", "Ticker Symbol", "TickerSymbol", "Symbol", "Code", "Local Ticker", "LocalTicker"]
    lower = {str(c).lower(): c for c in df.columns}
    col = None
    for name in cands:
        if name.lower() in lower:
            col = lower[name.lower()]
            break
    if not col:
        for c in df.columns:
            lc = str(c).lower()
            if "ticker" in lc or lc in {"symbol", "code"}:
                col = c
                break
    if not col:
        return []

    s = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"nan": None, "None": None, "": None})
        .dropna()
        .str.replace("\u200b", "", regex=False)   # zero-width space
        .str.replace(r"\s+", "", regex=True)      # remove inner whitespaces
    )

    # symboles plausibles (BRK.B, ABC, ABCD, ABCD-A, ALPHA-1…)
    s = s[s.str.match(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")]

    # enlève cash/FX/derivatives courants
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
            if r.headers.get("Content-Encoding", "").lower() == "gzip":
                try:
                    content = gzip.decompress(content)
                except Exception:
                    pass

            # Essaye CSV via pandas (plusieurs encodages)
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    if not df.empty:
                        ticks = parse_tickers_from_df(df)
                        print(f"  -> parsed {len(ticks)} tickers")
                        if ticks:
                            return ticks
                except Exception:
                    continue

            # Parfois c'est du JSON au lieu de CSV
            try:
                data = r.json()
                if isinstance(data, dict):
                    for key in ("holdings", "positions", "basket", "data"):
                        if key in data and isinstance(data[key], list) and data[key]:
                            df = pd.json_normalize(data[key])
                            ticks = parse_tickers_from_df(df)
                            print(f"  -> parsed {len(ticks)} tickers (json)")
                            if ticks:
                                return ticks
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
    html = r.text

    # pandas.read_html depuis un buffer (évite FutureWarning)
    tables = []
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        tables = []
    for idx, t in enumerate(tables):
        if not isinstance(t, pd.DataFrame) or t.empty:
            continue
        ticks = parse_tickers_from_df(t)
        if ticks:
            print(f"  -> table {idx}: {len(ticks)} tickers")
            return ticks
    print("  -> no usable table")
    return []

def try_embedded_json(sess: requests.Session) -> list[str]:
    print(f"[json] GET {HOLDINGS_URL} (embedded)")
    r = sess.get(HOLDINGS_URL, headers=UA_HEADERS, timeout=60)
    if r.status_code != 200:
        print(f"  -> status {r.status_code}")
        return []
    html = r.text

    patterns = [
        r"window\.__APP__\s*=\s*(\{.*?\});",
        r"var\s+.*?=\s*(\{.*?\"holdings\".*?\});",
        r"\"holdings\"\s*:\s*(\[[^\]]+\])",
        r"\"positions\"\s*:\s*(\[[^\]]+\])",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.DOTALL)
        if not m:
            continue
        blob = m.group(1)
        blob = re.sub(r"(?s)//.*?$", "", blob)   # strip // comments
        blob = re.sub(r",\s*}", "}", blob)
        blob = re.sub(r",\s*]", "]", blob)
        try:
            data = json.loads(blob)
        except Exception:
            try:
                data = json.loads(f'{{"holdings": {blob}}}')
            except Exception:
                continue

        if isinstance(data, dict):
            for key in ("holdings", "positions", "data"):
                if key in data and isinstance(data[key], list) and data[key]:
                    df = pd.json_normalize(data[key])
                    ticks = parse_tickers_from_df(df)
                    if ticks:
                        print(f"  -> embedded {key}: {len(ticks)} tickers")
                        return ticks
        elif isinstance(data, list):
            df = pd.json_normalize(data)
            ticks = parse_tickers_from_df(df)
            if ticks:
                print(f"  -> embedded list: {len(ticks)} tickers")
                return ticks

    print("  -> no embedded JSON found")
    return []

def main():
    sess = session_with_retries()

    # Prime le cookie (parfois nécessaire)
    try:
        sess.get(PRODUCT_URL, headers=UA_HEADERS, timeout=30)
        time.sleep(0.4)
        sess.get(HOLDINGS_URL, headers=UA_HEADERS, timeout=30)
    except Exception:
        pass

    # Stratégies en cascade
    ticks = try_csv_endpoints(sess)
    if not ticks:
        ticks = try_html_table(sess)
    if not ticks:
        ticks = try_embedded_json(sess)

    if not ticks:
        print("❌ Impossible de récupérer la liste IWM (iShares) après 4 stratégies.")
        sys.exit(2)

    pd.DataFrame({"Ticker": ticks}).to_csv("russell2000.csv", index=False)
    print(f"✅ Écrit russell2000.csv avec {len(ticks)} tickers.")

if __name__ == "__main__":
    main()
