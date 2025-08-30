#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io, re, sys
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

SPDR_XLSX_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream,*/*",
}

BAD_VALUES = {"", "ISIN", "UNASSIGNED", "USD", "CASH", "FX", "SWAP", "OPTION", "FUT", "FUTURES", "CURRENCY", "N/A", "NA", "None"}
RIC_SUFFIX   = re.compile(r"\.[A-Za-z]{1,4}$")             # .OQ .N .K …
BBG_EXCH     = re.compile(r"\s+(US|UW|UQ|LN|GY|FP|NA|IM|SM|HK|JP|CN|SW|AU)\b", re.IGNORECASE)
BBG_EQUITY   = re.compile(r"\s+US\s+EQUITY$", re.IGNORECASE)
TICK_RX      = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")

CANDIDATE_KEYS = [
    "ric", "reuters ric", "reuters ticker",
    "bloomberg ticker", "bloomberg", "bbg ticker", "bbg",
    "ticker", "ticker symbol", "tickersymbol", "symbol", "code"
]

def sess():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s

def clean_symbol(x: str) -> str | None:
    if x is None: return None
    t = str(x).strip().replace("\u200b","")
    if not t or t.upper() in BAD_VALUES: return None
    t = BBG_EQUITY.sub("", t)
    t = BBG_EXCH.sub("", t)
    t = RIC_SUFFIX.sub("", t)
    t = re.sub(r"\s+", "", t)
    if TICK_RX.fullmatch(t): return t.upper()
    return None

def flatten_cols(cols):
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if pd.notna(x)]
            out.append(" ".join(parts).strip())
        else:
            out.append(str(c).strip())
    return out

def find_header_row(df: pd.DataFrame) -> int | None:
    probe = min(len(df), 50)
    key = re.compile(r"(ticker|ric|reuters|bloomberg|symbol|code)", re.IGNORECASE)
    for i in range(probe):
        s = " | ".join(str(x) for x in df.iloc[i].values if pd.notna(x))
        if key.search(s): return i
    return None

def score_column(series: pd.Series) -> tuple[int, list[str]]:
    vals = []
    for v in series.astype(str).tolist():
        v = v.strip()
        if not v or v.upper() in BAD_VALUES: 
            continue
        c = clean_symbol(v)
        if c: vals.append(c)
    uniq = sorted(set(vals))
    return len(uniq), uniq

def best_tickers_from_df(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty: return []
    # header detection
    hdr = find_header_row(df)
    if hdr is not None:
        df = df.iloc[hdr:].reset_index(drop=True)
        df.columns = [str(x).strip() for x in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True)
    df = df.copy()
    df.columns = flatten_cols(df.columns)

    # map colonnes -> score
    lower = {c.lower(): c for c in df.columns}
    cand_cols = []
    for key in CANDIDATE_KEYS:
        if key in lower:
            cand_cols.append(lower[key])
    if not cand_cols:
        # heuristique : toute colonne contenant ces mots
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ["ric","bloomberg","ticker","symbol","code"]):
                cand_cols.append(c)

    best_count, best_vals, best_col = 0, [], None
    for col in cand_cols:
        cnt, vals = score_column(df[col])
        print(f"    - try col '{col}': {cnt} tickers")
        if cnt > best_count:
            best_count, best_vals, best_col = cnt, vals, col

    if best_col:
        print(f"    -> choose '{best_col}' with {best_count} tickers")
    return best_vals

def main():
    print(f"[SPDR] GET {SPDR_XLSX_URL}")
    r = sess().get(SPDR_XLSX_URL, headers=UA, timeout=60)
    if r.status_code != 200 or not r.content:
        print(f"  -> status {r.status_code}")
        sys.exit(2)
    try:
        book = pd.read_excel(io.BytesIO(r.content), sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"  -> read_excel fail: {e}")
        sys.exit(2)

    all_ticks = set()
    for name, df in (book or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        print(f"[sheet] {name} rows={len(df)} cols={len(df.columns)}")
        ticks = best_tickers_from_df(df)
        print(f"  -> sheet '{name}' extracted {len(ticks)}")
        all_ticks.update(ticks)

    total = len(all_ticks)
    print(f"[TOTAL] extracted {total} unique tickers")
    if total < 300:  # sanity check: le R2000 devrait en fournir >> 1000
        print("❌ Extraction trop faible — structure du fichier SPDR probablement changée. CSV non écrit.")
        sys.exit(2)

    out = pd.DataFrame({"Ticker": sorted(all_ticks)})
    out.to_csv("russell2000.csv", index=False)
    print(f"✅ Écrit russell2000.csv avec {len(out)} tickers.")

if __name__ == "__main__":
    main()
