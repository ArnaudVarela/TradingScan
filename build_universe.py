# build_universe.py
# Construit un univers unique à partir de plusieurs indices publics,
# l'enrichit avec market cap & liquidité (Yahoo), filtre MCAP < 75B$,
# puis écrit universe_today.csv (racine + dashboard/public/).

import io
import os
import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

MCAP_MAX_USD = 75_000_000_000  # < 75B$
MIN_ROW_LEN = 1

UA = {"User-Agent": "Mozilla/5.0"}

def _save(df: pd.DataFrame, name: str):
    # racine
    df.to_csv(name, index=False)
    # public
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

def _norm_yf(t: str) -> str | None:
    if not isinstance(t, str): return None
    t = t.strip().upper()
    if not t or t in {"N/A", "-", "—"}: return None
    # BRK.B (TV) → BRK-B (Yahoo)
    t = t.replace(".", "-")
    t = re.sub(r"[^A-Z0-9\-]", "", t)
    return t or None

def _yf_to_tv(t: str) -> str:
    # BRK-B (Yahoo) → BRK.B (TradingView)
    return (t or "").replace("-", ".")

def _union(*series: Iterable[str]) -> list[str]:
    seen = {}
    for s in series:
        for x in s:
            n = _norm_yf(x)
            if n: seen.setdefault(n, True)
    return list(seen.keys())

def _wiki_table(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    return pd.read_html(r.text)

def get_sp500() -> list[str]:
    # Wikipedia "List of S&P 500 companies"
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    out = []
    for t in _wiki_table(url):
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c for c in cols):
            sym_col = t.columns[cols.index(next(c for c in cols if "symbol" in c))]
            out.extend(t[sym_col].dropna().astype(str).tolist())
            break
    print(f"[SRC] S&P500: {len(out)} tickers")
    return out

def get_r1000() -> list[str]:
    # Wikipedia "Russell 1000 Index"
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    out = []
    for t in _wiki_table(url):
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            try:
                idx = next(i for i,c in enumerate(cols) if "ticker" in c or "symbol" in c)
            except StopIteration:
                continue
            out.extend(t.iloc[:, idx].dropna().astype(str).tolist())
            break
    print(f"[SRC] R1000: {len(out)} tickers")
    return out

def get_nasdaq100() -> list[str]:
    # Wikipedia "NASDAQ-100"
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    out = []
    for t in _wiki_table(url):
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            try:
                idx = next(i for i,c in enumerate(cols) if "ticker" in c or "symbol" in c)
            except StopIteration:
                continue
            out.extend(t.iloc[:, idx].dropna().astype(str).tolist())
            break
    print(f"[SRC] NASDAQ-100: {len(out)} tickers")
    return out

def get_r2000_from_csv(path="russell2000.csv") -> list[str]:
    if not os.path.exists(path):
        print("[SRC] R2000: missing russell2000.csv → skip")
        return []
    try:
        df = pd.read_csv(path)
        # support either "Ticker" or first column
        col = "Ticker" if "Ticker" in df.columns else df.columns[0]
        vals = df[col].dropna().astype(str).tolist()
        print(f"[SRC] R2000(csv): {len(vals)} tickers")
        return vals
    except Exception as e:
        print(f"[SRC] R2000(csv) error: {e} → skip")
        return []

def enrich_with_yf(tickers: list[str]) -> pd.DataFrame:
    # yfinance fast_info : market_cap, last_price, ten_day_average_volume
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            y = yf.Ticker(t)
            fi = getattr(y, "fast_info", None) or {}
            mcap = fi.get("market_cap")
            price = fi.get("last_price")
            vol10 = fi.get("ten_day_average_volume")
            if mcap is None:
                # fallback très léger (evite .info si possible)
                # on garde None si pas dispo
                pass
            avg_dollar = None
            if isinstance(price, (int,float)) and isinstance(vol10, (int,float)):
                avg_dollar = float(price)*float(vol10)
            rows.append({"ticker_yf": t, "mcap_usd": mcap, "avg_dollar_vol": avg_dollar})
        except Exception:
            rows.append({"ticker_yf": t, "mcap_usd": None, "avg_dollar_vol": None})
        if i % 100 == 0:
            print(f"[YF] {i}/{len(tickers)} enriched…")
            time.sleep(1.0)
    df = pd.DataFrame(rows)
    return df

def main():
    r1000  = get_r1000()
    sp500  = get_sp500()
    ndx100 = get_nasdaq100()
    r2000  = get_r2000_from_csv()  # optionnel

    universe = _union(r1000, sp500, ndx100, r2000)
    if len(universe) < MIN_ROW_LEN:
        raise SystemExit("❌ universe empty")

    print(f"[UNION] unique tickers before YF: {len(universe)}")

    meta = enrich_with_yf(universe)
    meta["ticker_tv"] = meta["ticker_yf"].map(_yf_to_tv)

    # filtre MCAP < 75B (garde mcap manquants pour ne pas perdre des noms,
    # mais on préfère filtrer strictement)
    before = len(meta)
    meta_f = meta[pd.to_numeric(meta["mcap_usd"], errors="coerce").fillna(0) < MCAP_MAX_USD].copy()
    print(f"[FILTER] mcap < 75B: {len(meta_f)}/{before}")

    # tri simple (liquidité décroissante)
    meta_f["avg_dollar_vol"] = pd.to_numeric(meta_f["avg_dollar_vol"], errors="coerce")
    meta_f = meta_f.sort_values(["avg_dollar_vol","mcap_usd"], ascending=[False, True])

    # colonnes finales
    out = meta_f[["ticker_yf","ticker_tv","mcap_usd","avg_dollar_vol"]].reset_index(drop=True)

    _save(out, "universe_today.csv")

if __name__ == "__main__":
    main()
