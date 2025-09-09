# build_universe.py
# Assemble a raw equity universe from several public sources, sanitize, and write CSVs.

from __future__ import annotations

import io
import os
import re
import time
import json
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

REQ_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
RETRIES = int(os.getenv("HTTP_RETRIES", "2"))
SLEEP_BETWEEN = float(os.getenv("HTTP_SLEEP", "0.4"))

# ---------------- Utils ----------------

def log(msg: str):
    print(msg, flush=True)

def _save_csv(df: pd.DataFrame, name: str, also_public: bool = True):
    try:
        p = Path(name)
        df.to_csv(p, index=False)
        if also_public:
            q = PUBLIC_DIR / name
            df.to_csv(q, index=False)
        log(f"[SAVE] {name} | rows={len(df)} | cols={len(df.columns)}")
    except Exception as e:
        log(f"[SAVE] failed for {name}: {e}")

def _best_ticker_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "ticker", "symbol", "Ticker", "Symbol", "code", "Code",
        "Ticker Symbol", "TickerSymbol", "Security", "Name", "company"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    # heuristique: première colonne object courte
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    return None

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,10}$")

NOISE_WORDS = set("""
USA UNITED UNITEDSTATES AMERICA AMERICAN EUROPE EURO EUROPEAN
ASIA AFRICA LATAM LATIN LATINAMERICA GLOBAL WORLD COUNTRY COUNTRIES
INDEX INDICES ETF ETFS FUNDS FUND
ISRAEL CHINA INDIA JAPAN GERMANY FRANCE ITALY SPAIN SWEDEN NORWAY
BELGIUM BRAZIL CANADA MEXICO IRELAND KOREA PANAMA BERMUDA
AIRLINES SOFTWARE ENERGY TOBACCO MEDIA CURRENCY REIT
""".split())

def _clean_one(sym: str) -> Optional[str]:
    if not isinstance(sym, str):
        return None
    s = sym.strip().upper()
    if not s:
        return None
    # drop obvious noise
    if s in NOISE_WORDS:
        return None
    # common garbage tokens in mirrors
    if s in {"ZCZZT", "ZWZZT", "ZOOZ", "ZBZZT", "ZXZZT"}:
        return None
    # remove spaces and slashes
    s = s.replace(" ", "").replace("/", "").replace("_", "-")
    # Yahoo uses '-' for share classes (BRK-B), many sources use '.'
    if "." in s and len(s) <= 6:
        # keep dot (some ADRs) but we’ll convert later if needed
        pass
    # must match strict ticker pattern
    if not _TICKER_RE.match(s):
        return None
    # forbid all-digit
    if s.isdigit():
        return None
    return s

def _sanitize_tickers(tickers: Iterable[str]) -> pd.DataFrame:
    out = []
    seen = set()
    for t in tickers:
        c = _clean_one(t)
        if not c: 
            continue
        # normalize to Yahoo style where applicable: class dot -> dash
        if "." in c:
            # preserve dots for four-letter NASDAQ with .W? etc, but prefer dash for class shares
            # If looks like BRK.B or BF.B → convert
            if len(c) <= 6 and c.count(".") == 1 and len(c.split(".")[-1]) == 1:
                c = c.replace(".", "-")
        if c not in seen:
            seen.add(c)
            out.append(c)
    return pd.DataFrame({"ticker_yf": out})

def _http_get(url: str) -> Optional[str]:
    last_err = None
    for i in range(RETRIES + 1):
        try:
            r = requests.get(url, timeout=REQ_TIMEOUT)
            if r.status_code == 200 and r.text:
                return r.text
            log(f"[HTTP] {url} -> {r.status_code}")
        except Exception as e:
            last_err = e
            log(f"[HTTP] {url} error: {e}")
        time.sleep(SLEEP_BETWEEN)
    if last_err:
        log(f"[HTTP] failed after retries: {last_err}")
    return None

def _read_csv_flex(text: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(io.StringIO(text))
        return df
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(text), sep=";")
            return df
        except Exception:
            return None

def _fetch_any(urls: List[str], source_name: str) -> pd.DataFrame:
    for url in urls:
        txt = _http_get(url)
        if not txt:
            continue
        df = _read_csv_flex(txt)
        if df is None or df.empty:
            log(f"[CSV] parse failed for {url}: empty/parse")
            continue
        # pick ticker column
        col = _best_ticker_col(df)
        if not col:
            log(f"[CSV] no ticker-like column in {url}")
            continue
        d = _sanitize_tickers(df[col].astype(str).tolist())
        if len(d) > 0:
            d["src"] = source_name
            log(f"[SRC] {source_name}: {len(d)} symbols from {url}")
            return d
    log(f"[SRC] {source_name} not available; returning empty df")
    return pd.DataFrame(columns=["ticker_yf", "src"])

# ---------------- Source loaders (with stable mirrors) ----------------

def _fetch_sp500() -> pd.DataFrame:
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/constituents.csv",
        "https://raw.githubusercontent.com/justetf-com/sp-500-etf-holdings/main/sp500_holdings.csv",
    ]
    return _fetch_any(urls, "SPX")

def _fetch_russell1000() -> pd.DataFrame:
    urls = [
        "https://raw.githubusercontent.com/ryanmagoon/russell-1000/main/russell1000.csv",
        "https://raw.githubusercontent.com/sebastianhab/russell-1000/main/r1000.csv",
    ]
    return _fetch_any(urls, "R1000")

def _fetch_nasdaq100() -> pd.DataFrame:
    urls = [
        "https://raw.githubusercontent.com/vega/vega-datasets/master/data/nasdaq-100-symbols.csv",
        "https://raw.githubusercontent.com/datasets/nasdaq-100/master/nasdaq100.csv",
    ]
    return _fetch_any(urls, "NDX")

def _fetch_nasdaq_composite() -> pd.DataFrame:
    urls = [
        "https://raw.githubusercontent.com/JerBouma/FinanceDatabase/master/database/constituents/nasdaq.csv",
        "https://raw.githubusercontent.com/sebastianhab/nasdaq-composite/master/nasdaqcomposite.csv",
    ]
    return _fetch_any(urls, "IXIC")

# -------------- Main --------------

DEFAULT_SEED = [
    # ultra-basics pour ne jamais sortir vide si tout tombe
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","AMD",
    "NFLX","ADBE","COST","PEP","KO","XOM","CVX","JPM","V","MA"
]

def main():
    log("[STEP] build_universe starting…")

    pools: List[pd.DataFrame] = []
    try:
        pools.append(_fetch_sp500())
    except Exception as e:
        log(f"[SRC] SPX error: {e}")
    time.sleep(SLEEP_BETWEEN)

    try:
        pools.append(_fetch_russell1000())
    except Exception as e:
        log(f"[SRC] R1000 error: {e}")
    time.sleep(SLEEP_BETWEEN)

    try:
        pools.append(_fetch_nasdaq100())
    except Exception as e:
        log(f"[SRC] NDX error: {e}")
    time.sleep(SLEEP_BETWEEN)

    try:
        pools.append(_fetch_nasdaq_composite())
    except Exception as e:
        log(f"[SRC] IXIC error: {e}")

    pools = [p for p in pools if p is not None and not p.empty]

    if not pools:
        # fallback: si ancien raw_universe existe, on le réutilise
        for base in (Path("."), PUBLIC_DIR):
            p = base / "raw_universe.csv"
            if p.exists():
                try:
                    df_old = pd.read_csv(p)
                    if "ticker_yf" in df_old.columns and len(df_old) > 0:
                        log("[FALLBACK] reuse existing raw_universe.csv")
                        raw = df_old[["ticker_yf"]].dropna().drop_duplicates().copy()
                        break
                except Exception:
                    pass
        else:
            # sinon, seed minimal
            log("[FALLBACK] using minimal seed universe")
            raw = _sanitize_tickers(DEFAULT_SEED)
    else:
        raw = pd.concat(pools, ignore_index=True)
        raw = raw[["ticker_yf"]].dropna().drop_duplicates().reset_index(drop=True)

    # post-clean: drop weird overlong or too short
    raw["ticker_yf"] = raw["ticker_yf"].astype(str).str.upper()
    raw = raw[raw["ticker_yf"].str.len().between(1, 11)]
    raw = raw.drop_duplicates().reset_index(drop=True)

    # save raw
    _save_csv(raw, "raw_universe.csv")

    # basic in-scope filter (keep everything here; downstream scripts can filter by mcap/ADR/etc.)
    in_scope = raw.copy()
    _save_csv(in_scope, "universe_in_scope.csv")

    # small preview for CI logs
    try:
        log('== HEAD raw_universe.csv ==')
        print(pd.read_csv("raw_universe.csv").head(5).to_string(index=False))
    except Exception:
        log("missing raw_universe.csv head")

    try:
        log('== HEAD universe_in_scope.csv ==')
        print(pd.read_csv("universe_in_scope.csv").head(10).to_string(index=False))
    except Exception:
        log("missing universe_in_scope.csv head")

if __name__ == "__main__":
    main()
