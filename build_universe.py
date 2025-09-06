# build_universe.py
# Universe = Russell 1000 + S&P500 + NASDAQ-100 + NASDAQ Composite (robust fetch + cache)
# Enrichment:
#   - Yahoo fast_info for market cap & light liquidity
#   - dv20 fallback from history when needed (batched)
#   - optional sector fill for a small top-liquidity slice via yfinance.info and Alpha Vantage
# Filter:
#   - Sectors ∈ {Tech, Financials, Industrials, Health Care}
#   - Keep MCAP < 75B **but do not drop NaNs here** (the screener will coalesce/complete MCAP again)
# Outputs: raw_universe.csv, universe_in_scope.csv, universe_today.csv (root + dashboard/public)

import io
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------ CONFIG ------------
PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MCAP_MAX_USD = 75_000_000_000  # < 75B$
MIN_ROW_LEN = 1

USE_NASDAQ_COMPOSITE = True

ALLOWED_SECTORS = {
    "information technology","technology","tech","information technology services",
    "financials","financial","financial services",
    "industrials","industrial",
    "health care","healthcare","health services",
}

TOP_LIQ_FOR_SECTOR_ENRICH = int(os.getenv("TOP_LIQ_FOR_SECTOR_ENRICH", "180"))
YF_INFO_SLEEP = 0.15
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
AV_SLEEP = 12.5
MAX_AV_LOOKUPS = int(os.getenv("MAX_AV_LOOKUPS", "60"))

# If network is flaky you can force cache-only fetch for NASDAQ lists
NASDAQ_FETCH_MODE = os.getenv("NASDAQ_FETCH", "live").lower()  # "live" | "cache"

UA = {"User-Agent": "Mozilla/5.0"}

# ------------ HTTP session (retries) ------------
def _session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
    s.headers.update(UA)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

# ------------ IO helpers ------------
def _save(df: pd.DataFrame, name: str):
    if df is None:
        df = pd.DataFrame()
    df.to_csv(name, index=False)
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

# ------------ symbol helpers ------------
def _norm_yf(t: str) -> Optional[str]:
    if not isinstance(t, str): return None
    t = t.strip().upper()
    if not t or t in {"N/A","-","—"}: return None
    t = t.replace(".", "-")         # BRK.B -> BRK-B
    t = re.sub(r"[^A-Z0-9\-]", "", t)
    return t or None

def _yf_to_tv(t: str) -> str:
    return (t or "").replace("-", ".")

def _union(*series: Iterable[str]) -> list[str]:
    seen = {}
    for s in series:
        for x in s:
            n = _norm_yf(x)
            if n: seen.setdefault(n, True)
    return list(seen.keys())

def _norm_sector(s: str) -> str:
    return str(s or "").strip().lower()

# ------------ Wikipedia scrapers ------------
def _wiki_tables(url: str) -> list[pd.DataFrame]:
    r = _session().get(url, timeout=45)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))

def _extract_symbol_and_sector(df: pd.DataFrame) -> Tuple[List[str], Dict[str,str]]:
    cols_lower = [str(c).lower() for c in df.columns]
    sym_idx = next((i for i,c in enumerate(cols_lower) if ("symbol" in c or "ticker" in c)), None)
    sec_idx = next((i for i,c in enumerate(cols_lower) if ("sector" in c and "sub" not in c)), None)

    syms, sector_map = [], {}
    if sym_idx is None: return syms, sector_map
    sym_col = df.columns[sym_idx]

    if sec_idx is not None:
        sec_col = df.columns[sec_idx]
        for _, row in df.iterrows():
            sym = row.get(sym_col)
            yf_sym = _norm_yf(str(sym)) if pd.notna(sym) else None
            if yf_sym:
                syms.append(yf_sym)
                sec = row.get(sec_col)
                if pd.notna(sec):
                    sector_map[yf_sym] = str(sec).strip()
    else:
        for s in df[sym_col].dropna().astype(str):
            yf_sym = _norm_yf(s)
            if yf_sym: syms.append(yf_sym)

    return syms, sector_map

def get_sp500() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms:
            out_syms.extend(syms); out_map.update(smap); break
    print(f"[SRC] S&P500: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

def get_r1000() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms:
            out_syms.extend(syms); out_map.update(smap); break
    print(f"[SRC] R1000: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

def get_nasdaq100() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms:
            out_syms.extend(syms); out_map.update(smap); break
    print(f"[SRC] NASDAQ-100: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

# ------------ NASDAQ Composite (robust with cache) ------------
def _cache_path(name: str) -> Path:
    return CACHE_DIR / name

def _read_cached_text(name: str, ttl_hours=48) -> Optional[str]:
    p = _cache_path(name)
    if not p.exists(): return None
    try:
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if age > timedelta(hours=ttl_hours): return None
        return p.read_text()
    except Exception:
        return None

def _write_cache_text(name: str, text: str):
    p = _cache_path(name)
    p.write_text(text)

def _fetch_text(url: str) -> Optional[str]:
    try:
        r = _session().get(url, timeout=45)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None

def get_nasdaq_composite() -> list[str]:
    """
    Build from nasdaqlisted + nasdaqtraded.
    """
    if NASDAQ_FETCH_MODE == "cache":
        txt1 = _read_cached_text("nasdaqlisted.txt")
        txt2 = _read_cached_text("nasdaqtraded.txt")
    else:
        txt1 = _fetch_text("https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
        if txt1: _write_cache_text("nasdaqlisted.txt", txt1)
        else:    txt1 = _read_cached_text("nasdaqlisted.txt")
        txt2 = _fetch_text("https://nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt")
        if txt2: _write_cache_text("nasdaqtraded.txt", txt2)
        else:    txt2 = _read_cached_text("nasdaqtraded.txt")

    syms: list[str] = []

    def parse(txt: str, field="Symbol"):
        if not txt: return []
        lines = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
        cols = [c.strip() for c in lines[0].split("|")]
        if field not in cols: return []
        idx = cols.index(field)
        out = []
        for ln in lines[1:]:
            parts = ln.split("|")
            if len(parts) > idx:
                n = _norm_yf(parts[idx])
                if n: out.append(n)
        return out

    syms.extend(parse(txt1, "Symbol"))
    # Keep only NASDAQ-listed from nasdaqtraded by checking the ListingExchange field if present
    if txt2:
        lines = [ln for ln in txt2.splitlines() if ln and not ln.startswith("#")]
        cols = [c.strip() for c in lines[0].split("|")]
        sym_i = cols.index("Symbol") if "Symbol" in cols else None
        exch_i = cols.index("ListingExchange") if "ListingExchange" in cols else None
        if sym_i is not None and exch_i is not None:
            for ln in lines[1:]:
                parts = ln.split("|")
                if len(parts) > max(sym_i, exch_i) and parts[exch_i].strip().upper() == "Q":
                    n = _norm_yf(parts[sym_i])
                    if n: syms.append(n)

    uniq = list(dict.fromkeys(syms).keys())
    print(f"[SRC] NASDAQ Composite (robust): ~{len(uniq)} tickers")
    return uniq

# ------------ Yahoo fast enrich + dv20 fallback ------------
def _dv20_from_history(tickers: list[str]) -> Dict[str, float]:
    """
    Vectorized dv20 (last 20 trading days).
    """
    out: Dict[str, float] = {}
    if not tickers:
        return out
    # fetch ~3 months to be safe
    try:
        df = yf.download(
            tickers=tickers,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    sub = df[t][["Close","Volume"]].dropna()
                    if sub.empty or len(sub) < 5: continue
                    tail = sub.tail(20)
                    dv = float((tail["Close"] * tail["Volume"]).mean())
                    out[t] = dv
                except Exception:
                    pass
        else:
            # single ticker
            sub = df[["Close","Volume"]].dropna()
            if not sub.empty:
                tail = sub.tail(20)
                out[tickers[0]] = float((tail["Close"] * tail["Volume"]).mean())
    except Exception:
        pass
    return out

def enrich_with_yf(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, 1):
        mcap = None; price = None; vol10 = None
        try:
            y = yf.Ticker(t)
            fi = getattr(y, "fast_info", None) or {}
            mcap = fi.get("market_cap")
            price = fi.get("last_price")
            vol10 = fi.get("ten_day_average_volume")
        except Exception:
            pass
        avg_dollar = float(price) * float(vol10) if isinstance(price,(int,float)) and isinstance(vol10,(int,float)) else None
        rows.append({"ticker_yf": t, "mcap_usd": mcap, "avg_dollar_vol": avg_dollar})
        if i % 120 == 0:
            print(f"[YF] {i}/{len(tickers)} enriched…")
            time.sleep(0.6)
    meta = pd.DataFrame(rows)

    # dv20 fallback where missing
    need = meta[meta["avg_dollar_vol"].isna()]["ticker_yf"].tolist()
    if need:
        dv_map = {}
        # do in chunks of ~200 to stay fast
        for i in range(0, len(need), 200):
            chunk = need[i:i+200]
            part = _dv20_from_history(chunk)
            dv_map.update(part)
            time.sleep(0.2)
        if dv_map:
            meta["avg_dollar_vol"] = meta.apply(
                lambda r: r["avg_dollar_vol"] if pd.notna(r["avg_dollar_vol"]) else dv_map.get(r["ticker_yf"]), axis=1
            )
        print(f"[LIQ] dv20 filled: {len(dv_map)}")

    return meta

# ------------ Sector fill (yfinance.info -> Alpha Vantage) ------------
def try_fill_sector_with_yfinfo(df: pd.DataFrame, ticker_col="ticker_yf") -> pd.DataFrame:
    df = df.copy()
    sec, ind = [], []
    for i, sym in enumerate(df[ticker_col].tolist(), 1):
        s, it = None, None
        try:
            info = yf.Ticker(sym).info
            s = info.get("sector")
            it = info.get("industry")
        except Exception:
            pass
        sec.append(s); ind.append(it)
        if i % 25 == 0:
            print(f"[YF.info] {i}/{len(df)}")
        time.sleep(YF_INFO_SLEEP)
    df["sector_fill"] = sec
    df["industry_fill"] = ind
    return df

def try_fill_sector_with_av(df: pd.DataFrame, ticker_col="ticker_yf") -> pd.DataFrame:
    if not AV_KEY or df.empty:
        df["av_sector"] = None
        df["av_industry"] = None
        return df
    sec, ind = [], []
    done = 0
    for i, sym in enumerate(df[ticker_col].tolist(), 1):
        if done >= MAX_AV_LOOKUPS:
            sec.append(None); ind.append(None); continue
        try:
            url = "https://www.alphavantage.co/query"
            params = {"function":"OVERVIEW","symbol":sym,"apikey":AV_KEY}
            r = _session().get(url, params=params, timeout=30)
            if r.status_code == 200:
                j = r.json()
                sec.append(j.get("Sector"))
                ind.append(j.get("Industry"))
                done += 1
            else:
                sec.append(None); ind.append(None)
        except Exception:
            sec.append(None); ind.append(None)
        time.sleep(AV_SLEEP)
    df["av_sector"] = sec
    df["av_industry"] = ind
    return df

# ------------ MAIN ------------
def main():
    # 1) Sources
    r1_syms, r1_sec = get_r1000()
    sp_syms, sp_sec = get_sp500()
    nd_syms, nd_sec = get_nasdaq100()
    comp_syms = get_nasdaq_composite() if USE_NASDAQ_COMPOSITE else []

    universe = _union(r1_syms, sp_syms, nd_syms, comp_syms)
    if len(universe) < MIN_ROW_LEN:
        raise SystemExit("❌ universe empty")
    print(f"[UNION] unique tickers before YF: {len(universe)}")

    # 2) Yahoo fast (mcap/liquidity) + dv20 fallback
    meta = enrich_with_yf(universe)
    meta["ticker_tv"] = meta["ticker_yf"].map(_yf_to_tv)

    # 3) Sector map from Wikipedia for known sets
    sector_map: Dict[str, str] = {}
    sector_map.update(r1_sec); sector_map.update(sp_sec); sector_map.update(nd_sec)
    meta["sector"] = meta["ticker_yf"].map(lambda t: sector_map.get(t, ""))  # Unknown for NASDAQ-only names
    meta["industry"] = ""

    # RAW
    raw_cols = ["ticker_yf","ticker_tv","mcap_usd","avg_dollar_vol","sector","industry"]
    raw = meta[raw_cols].copy()
    _save(raw, "raw_universe.csv")

    # 4) Pre-filter < 75B but keep NaNs (we’ll coalesce in screener)
    mcap_num = pd.to_numeric(raw["mcap_usd"], errors="coerce")
    meta_lt75 = raw[(mcap_num.isna()) | (mcap_num < MCAP_MAX_USD)].copy()

    # 5) Sector completion for a small top-liquidity subset
    meta_lt75["avg_dollar_vol"] = pd.to_numeric(meta_lt75["avg_dollar_vol"], errors="coerce")
    need_fill = meta_lt75[meta_lt75["sector"].astype(str).str.strip().eq("")].copy()
    need_fill = need_fill.sort_values("avg_dollar_vol", ascending=False).head(TOP_LIQ_FOR_SECTOR_ENRICH)
    if not need_fill.empty:
        print(f"[ENRICH] sector for top {len(need_fill)} with empty sector (YF.info -> AV)…")
        step = try_fill_sector_with_yfinfo(need_fill, "ticker_yf")
        for src, dst in [("sector_fill","sector"),("industry_fill","industry")]:
            mask = step[src].notna() & step[src].astype(str).str.strip().ne("")
            meta_lt75.loc[step.index[mask], dst] = step.loc[mask, src].values
        rest = step[step["sector_fill"].isna() | (step["sector_fill"].astype(str).str.strip()=="")]
        if not rest.empty:
            rest2 = try_fill_sector_with_av(rest, "ticker_yf")
            mask2 = rest2["av_sector"].notna() & rest2["av_sector"].astype(str).str.strip().ne("")
            meta_lt75.loc[rest2.index[mask2], "sector"] = rest2.loc[mask2, "av_sector"].values
            meta_lt75.loc[rest2.index[mask2], "industry"] = rest2.loc[mask2, "av_industry"].values

    # 6) Final sector scope (Tech/Fin/Ind/HC)
    sec_norm = meta_lt75["sector"].map(_norm_sector)
    in_scope = meta_lt75[sec_norm.isin(ALLOWED_SECTORS)].copy()

    # Sorting: liquidity desc then mcap asc
    in_scope["mcap_usd"] = pd.to_numeric(in_scope["mcap_usd"], errors="coerce")
    in_scope["avg_dollar_vol"] = pd.to_numeric(in_scope["avg_dollar_vol"], errors="coerce")
    in_scope = in_scope.sort_values(["avg_dollar_vol","mcap_usd"], ascending=[False, True])

    # 7) Outputs
    _save(in_scope, "universe_in_scope.csv")
    _save(in_scope, "universe_today.csv")

    print(f"[FILTER] sectors∈(Tech/Fin/Ind/HC) & mcap<75B (NaN kept for later fill): {len(in_scope)}/{len(raw)}")

if __name__ == "__main__":
    main()
