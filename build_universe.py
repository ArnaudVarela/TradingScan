# build_universe.py — v2.2 (tz fix, faster, sector-first, scoped liquidity)
# Univers: R1000 + S&P500 + NASDAQ-100 + NASDAQ Composite (multi-URL + cache)
# Enrichissement rapide: Yahoo fast_info (peut manquer MCAP/prix).
# Optimisation: on filtre SECTEURS d'abord, puis on ne comble la liquidité (dv) que pour cet in-scope.
# Filtre MCAP < 75B en "souple" (inclut NaN), l'enforcement strict se fait dans le screener.
# Sorties: raw_universe.csv, universe_in_scope.csv, universe_today.csv (root + dashboard/public)

import io, os, re, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------ CONFIG ------------
PUBLIC_DIR = Path("dashboard/public"); PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR  = Path("cache");            CACHE_DIR.mkdir(parents=True, exist_ok=True)

MCAP_MAX_USD = 75_000_000_000  # < 75B$
MIN_ROW_LEN = 1
USE_NASDAQ_COMPOSITE = True
TARGET_SECTORS = {"information technology", "financials", "industrials", "health care"}

TOP_LIQ_FOR_SECTOR_ENRICH = int(os.getenv("TOP_LIQ_FOR_SECTOR_ENRICH", "180"))
YF_INFO_SLEEP = 0.12
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
AV_SLEEP = 12.5
MAX_AV_LOOKUPS = int(os.getenv("MAX_AV_LOOKUPS", "60"))

YF_FAST_CACHE_PATH = CACHE_DIR / "yf_fast_info.csv"
YF_CACHE_TTL_DAYS  = int(os.getenv("YF_CACHE_TTL_DAYS", "2"))
NASDAQ_FETCH_MODE  = os.getenv("NASDAQ_FETCH", "live").lower()
UA = {"User-Agent": "Mozilla/5.0"}

# ------------ IO ------------
def _save(df: pd.DataFrame, name: str):
    if df is None: df = pd.DataFrame()
    df.to_csv(name, index=False)
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

# ------------ symbol helpers ------------
def _norm_yf(t: str) -> Optional[str]:
    if not isinstance(t, str): return None
    t = t.strip().upper()
    if not t or t in {"N/A","-","—"}: return None
    t = t.replace(".", "-")
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

# ------------ HTTP ------------
def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, connect=3, read=3, backoff_factor=1.1,
                    status_forcelist=(429,500,502,503,504),
                    allowed_methods=frozenset(["GET"]),
                    raise_on_status=False, respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=12, pool_maxsize=12)
    s.mount("http://", adapter); s.mount("https://", adapter)
    s.headers.update(UA)
    return s

SESSION = _build_session()
def _get(url: str, timeout=(8, 30)) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=timeout); r.raise_for_status(); return r.text
    except Exception as e:
        print(f"[WARN] GET fail {url}: {e}"); return None

# ------------ WIKI ------------
def _wiki_tables(url: str) -> list[pd.DataFrame]:
    for attempt in range(3):
        txt = _get(url, timeout=(8, 40))
        if txt:
            try: return pd.read_html(io.StringIO(txt))
            except Exception as e: print(f"[WARN] read_html fail ({attempt+1}/3): {e}")
        time.sleep(1.0 + 0.8*attempt)
    return []

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
            sym = row.get(sym_col); sec = row.get(sec_col)
            yf_sym = _norm_yf(str(sym)) if pd.notna(sym) else None
            if yf_sym:
                syms.append(yf_sym)
                if pd.notna(sec): sector_map[yf_sym] = str(sec).strip()
        return syms, sector_map
    for s in df[sym_col].dropna().astype(str).tolist():
        yf_sym = _norm_yf(s)
        if yf_sym: syms.append(yf_sym)
    return syms, sector_map

def get_sp500() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms: out_syms.extend(syms); out_map.update(smap); break
    print(f"[SRC] S&P500: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

def get_r1000() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms: out_syms.extend(syms); out_map.update(smap); break
    print(f"[SRC] R1000: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

def get_nasdaq100() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms: out_syms.extend(syms); out_map.update(smap); break
    print(f"[SRC] NASDAQ-100: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

# ------------ NASDAQ Composite (NasdaqTrader) ------------
def _parse_pipe_file(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not lines: return []
    header = [c.strip() for c in lines[0].split("|")]
    rows = []
    for ln in lines[1:]:
        if ln.lower().startswith("file creation time"): continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != len(header): continue
        rows.append(dict(zip(header, parts)))
    return rows

def _read_cached(name: str) -> Optional[str]:
    p = CACHE_DIR / name
    if p.exists():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            print(f"[CACHE] loaded {name} ({len(txt)} bytes)"); return txt
        except Exception: return None
    return None

def _write_cache(name: str, text: str) -> None:
    try:
        (CACHE_DIR / name).write_text(text, encoding="utf-8")
        print(f"[CACHE] saved {name} ({len(text)} bytes)")
    except Exception as e:
        print(f"[CACHE][WARN] save failed {name}: {e}")

def _get_text_from_urls(urls: list[str], cache_name: str) -> Optional[str]:
    if NASDAQ_FETCH_MODE == "cache": return _read_cached(cache_name)
    for i, u in enumerate(urls, 1):
        txt = _get(u, timeout=(8, 35))
        if txt:
            _write_cache(cache_name, txt); print(f"[SRC] OK {u}"); return txt
        if i % 2 == 0: time.sleep(0.9*(i//2))
    print("[SRC] fall back to cache"); return _read_cached(cache_name)

def get_nasdaq_composite() -> list[str]:
    l_urls = [
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
        "https://www.ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
        "http://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "http://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
        "http://www.ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
    ]
    t_urls = [
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
        "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt",
        "https://www.ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt",
        "http://nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
        "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
        "http://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt",
        "http://www.ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt",
    ]
    symbols: list[str] = []; seen = set()

    txt_listed = _get_text_from_urls(l_urls, "nasdaqlisted.txt")
    if txt_listed:
        rows = _parse_pipe_file(txt_listed)
        sym_col = "Symbol" if rows and "Symbol" in rows[0] else None
        test_col = "Test Issue" if rows and "Test Issue" in rows[0] else None
        etf_col  = "ETF" if rows and "ETF" in rows[0] else None
        if sym_col:
            added = 0
            for row in rows:
                raw = row.get(sym_col); n = _norm_yf(raw) if raw else None
                if not n or n in seen: continue
                if test_col and str(row.get(test_col,"")).upper() == "Y": continue
                if etf_col  and str(row.get(etf_col,"")).upper() == "Y": continue
                seen.add(n); symbols.append(n); added += 1
            print(f"[SRC] nasdaqlisted: +{added} (total {len(symbols)})")

    txt_traded = _get_text_from_urls(t_urls, "nasdaqtraded.txt")
    if txt_traded:
        rows = _parse_pipe_file(txt_traded)
        sym_col = "NASDAQ Symbol" if rows and "NASDAQ Symbol" in rows[0] else None
        exch_col = "Listing Exchange" if rows and "Listing Exchange" in rows[0] else ("Exchange" if rows and "Exchange" in rows[0] else None)
        test_col = "Test Issue" if rows and "Test Issue" in rows[0] else None
        etf_col  = "ETF" if rows and "ETF" in rows[0] else None
        if sym_col:
            added = 0
            for row in rows:
                if exch_col and str(row.get(exch_col,"")).upper() != "Q": continue
                raw = row.get(sym_col); n = _norm_yf(raw) if raw else None
                if not n or n in seen: continue
                if test_col and str(row.get(test_col,"")).upper() == "Y": continue
                if etf_col  and str(row.get(etf_col,"")).upper() == "Y": continue
                seen.add(n); symbols.append(n); added += 1
            print(f"[SRC] nasdaqtraded (Q only): +{added} (total {len(symbols)})")

    print(f"[SRC] NASDAQ Composite (robust): ~{len(symbols)} tickers")
    return symbols

# ------------ Yahoo fast_info cache ------------
def _load_yf_cache() -> pd.DataFrame:
    if not YF_FAST_CACHE_PATH.exists():
        return pd.DataFrame(columns=["ticker_yf","market_cap","last_price","ten_day_avg_vol","asof"])
    try:
        df = pd.read_csv(YF_FAST_CACHE_PATH)
        if "asof" in df.columns:
            # IMPORTANT: rendre 'asof' tz-aware UTC
            df["asof"] = pd.to_datetime(df["asof"], errors="coerce", utc=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker_yf","market_cap","last_price","ten_day_avg_vol","asof"])

def _save_yf_cache(df: pd.DataFrame):
    try:
        df.to_csv(YF_FAST_CACHE_PATH, index=False)
        print(f"[CACHE] saved yf_fast_info: {len(df)} rows")
    except Exception as e:
        print(f"[CACHE][WARN] could not save yf_fast_info: {e}")

def _cache_is_fresh(ts) -> bool:
    """ts peut être string/naive/aware → on convertit en UTC-aware et on compare."""
    if pd.isna(ts): return False
    try:
        tsv = pd.to_datetime(ts, utc=True)  # tz-aware UTC
    except Exception:
        return False
    now = pd.Timestamp.now(tz=timezone.utc)
    return (now - tsv) <= timedelta(days=YF_CACHE_TTL_DAYS)

# ------------ Sector normalization ------------
_SECTOR_NORM_MAP = {
    "technology":"information technology","tech":"information technology",
    "information technology":"information technology","information technology services":"information technology",
    "financial":"financials","financial services":"financials","finance":"financials","financials":"financials",
    "industrial":"industrials","industrials":"industrials",
    "healthcare":"health care","health services":"health care","health care":"health care",
}
def _norm_sector(s: str) -> str:
    k = str(s or "").strip().lower()
    return _SECTOR_NORM_MAP.get(k, k)

# ------------ Enrichissements ------------
def enrich_with_yf(tickers: list[str]) -> pd.DataFrame:
    cache = _load_yf_cache()
    cache_idx = {str(s).upper(): i for i, s in enumerate(cache.get("ticker_yf", []))}
    rows, to_fetch = [], []
    for t in tickers:
        u = str(t).upper()
        i = cache_idx.get(u, None)
        if i is not None and _cache_is_fresh(cache.loc[i, "asof"]):
            rows.append({
                "ticker_yf": u,
                "mcap_usd": cache.loc[i, "market_cap"],
                "avg_dollar_vol": (cache.loc[i, "last_price"] or 0) * (cache.loc[i, "ten_day_avg_vol"] or 0)
            })
        else:
            to_fetch.append(u)

    print(f"[YF] cache hit: {len(rows)} | to fetch: {len(to_fetch)}")
    fetched = []
    for i, t in enumerate(to_fetch, 1):
        try:
            y = yf.Ticker(t); fi = getattr(y, "fast_info", None) or {}
            mcap = fi.get("market_cap"); price = fi.get("last_price"); vol10 = fi.get("ten_day_average_volume")
            avg_dollar = (float(price) * float(vol10)) if (isinstance(price,(int,float)) and isinstance(vol10,(int,float))) else None
            rows.append({"ticker_yf": t, "mcap_usd": mcap, "avg_dollar_vol": avg_dollar})
            fetched.append({"ticker_yf": t, "market_cap": mcap, "last_price": price, "ten_day_avg_vol": vol10, "asof": pd.Timestamp.now(tz=timezone.utc)})
        except Exception:
            rows.append({"ticker_yf": t, "mcap_usd": None, "avg_dollar_vol": None})
        if i % 120 == 0:
            print(f"[YF] {i}/{len(to_fetch)} enriched…"); time.sleep(0.7)

    if fetched:
        cache = pd.concat([cache, pd.DataFrame(fetched)], ignore_index=True)
        cache = cache.sort_values("asof").drop_duplicates(subset=["ticker_yf"], keep="last")
        _save_yf_cache(cache)

    return pd.DataFrame(rows)

def fill_liquidity_with_history_scoped(meta_scoped: pd.DataFrame, batch_size: int = 110) -> pd.DataFrame:
    """Complète avg_dollar_vol sur l'échantillon scindé (12j, moyenne 8 derniers jours)."""
    mets = meta_scoped.copy()
    missing = mets[mets["avg_dollar_vol"].isna()]["ticker_yf"].tolist()
    if not missing:
        return mets
    print(f"[LIQ] scoped liquidity via history for {len(missing)} tickers…")
    liq_map: Dict[str, float] = {}
    for i in range(0, len(missing), batch_size):
        chunk = [s for s in missing[i:i+batch_size] if isinstance(s, str)]
        if not chunk: continue
        try:
            df = yf.download(tickers=chunk, period="12d", interval="1d",
                             auto_adjust=True, group_by="ticker", threads=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                for sym in chunk:
                    try:
                        sub = df[sym][["Close","Volume"]].dropna()
                        if len(sub) >= 8:
                            dv = (sub["Close"] * sub["Volume"]).tail(8).mean()
                            if pd.notna(dv): liq_map[sym] = float(dv)
                    except Exception:
                        pass
            else:
                try:
                    sub = df[["Close","Volume"]].dropna()
                    if len(sub) >= 8:
                        dv = (sub["Close"]*sub["Volume"]).tail(8).mean()
                        if pd.notna(dv): liq_map[chunk[0]] = float(dv)
                except Exception:
                    pass
        except Exception as e:
            print(f"[LIQ] batch exception: {e}")
        time.sleep(0.35)
    if liq_map:
        mets.loc[mets["ticker_yf"].isin(liq_map.keys()), "avg_dollar_vol"] = mets["ticker_yf"].map(liq_map)
    print(f"[LIQ] filled: {len(liq_map)}")
    return mets

def try_fill_sector_with_yfinfo(df: pd.DataFrame, ticker_col="ticker_yf") -> pd.DataFrame:
    df = df.copy(); sec = []; ind = []
    for i, sym in enumerate(df[ticker_col].tolist(), 1):
        s, it = None, None
        try:
            info = yf.Ticker(sym).info
            s = info.get("sector"); it = info.get("industry")
        except Exception:
            pass
        sec.append(s); ind.append(it)
        if i % 25 == 0: print(f"[YF.info] {i}/{len(df)}")
        time.sleep(YF_INFO_SLEEP)
    df.loc[:, "sector_fill"] = sec
    df.loc[:, "industry_fill"] = ind
    return df

def try_fill_sector_with_av(df: pd.DataFrame, ticker_col="ticker_yf") -> pd.DataFrame:
    out = df.copy()
    if not AV_KEY or out.empty:
        out.loc[:, "av_sector"] = None
        out.loc[:, "av_industry"] = None
        return out
    sec, ind = [], []
    done = 0
    for i, sym in enumerate(out[ticker_col].tolist(), 1):
        if done >= MAX_AV_LOOKUPS:
            sec.append(None); ind.append(None); continue
        try:
            url = "https://www.alphavantage.co/query"
            params = {"function":"OVERVIEW","symbol":sym,"apikey":AV_KEY}
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                j = r.json(); sec.append(j.get("Sector")); ind.append(j.get("Industry")); done += 1
            else:
                sec.append(None); ind.append(None)
        except Exception:
            sec.append(None); ind.append(None)
        time.sleep(AV_SLEEP)
    out.loc[:, "av_sector"] = sec
    out.loc[:, "av_industry"] = ind
    return out

def main():
    # 1) Sources
    r1_syms, r1_sec = get_r1000()
    sp_syms, sp_sec = get_sp500()
    nd_syms, nd_sec = get_nasdaq100()
    comp_syms = get_nasdaq_composite() if USE_NASDAQ_COMPOSITE else []

    universe = _union(r1_syms, sp_syms, nd_syms, comp_syms)
    if len(universe) < MIN_ROW_LEN: raise SystemExit("❌ universe empty")
    print(f"[UNION] unique tickers before YF: {len(universe)}")

    # 2) fast_info (cap/liquidité) + cache
    meta = enrich_with_yf(universe)
    meta["ticker_tv"] = meta["ticker_yf"].map(_yf_to_tv)

    # 3) Secteurs (Wiki + catalog)
    sector_map: Dict[str,str] = {}
    sector_map.update({k:_norm_sector(v) for k,v in r1_sec.items()})
    sector_map.update({k:_norm_sector(v) for k,v in sp_sec.items()})
    sector_map.update({k:_norm_sector(v) for k,v in nd_sec.items()})
    meta["sector"]   = meta["ticker_yf"].map(lambda t: sector_map.get(t, ""))
    meta["industry"] = ""

    # Merge secteur catalog si présent
    try:
        if Path("sector_catalog.csv").exists():
            cat = pd.read_csv("sector_catalog.csv")
            tcol = next((c for c in cat.columns if c.lower() in {"ticker","ticker_yf","symbol"}), None)
            if tcol:
                cat = cat.rename(columns={tcol:"ticker_yf"})
                for c in ("sector","industry"):
                    if c not in cat.columns: cat[c] = ""
                cat["sector"] = cat["sector"].map(_norm_sector)
                cat = cat[["ticker_yf","sector","industry"]].drop_duplicates("ticker_yf")
                meta = meta.merge(cat, on="ticker_yf", how="left", suffixes=("", "_cat"))
                for c in ("sector","industry"):
                    mc = f"{c}_cat"
                    meta[c] = meta[c].where(meta[mc].isna() | (meta[mc].astype(str).str.strip()==""), meta[mc])
                    if mc in meta.columns: meta.drop(columns=[mc], inplace=True)
    except Exception as e:
        print(f"[WARN] sector_catalog merge skipped: {e}")

    # RAW (avant tout fallback de liquidité)
    raw_cols = ["ticker_yf","ticker_tv","mcap_usd","avg_dollar_vol","sector","industry"]
    raw = meta[raw_cols].copy()
    _save(raw, "raw_universe.csv")

    # 4) Pré-filtre secteurs (SECTOR-FIRST) — on réduit le scope avant de combler la liquidité
    meta["sector"] = meta["sector"].map(_norm_sector)
    meta_scoped = meta[meta["sector"].isin(TARGET_SECTORS)].copy()
    print(f"[SCOPE] sector-first scoped: {len(meta_scoped)} / {len(meta)}")

    # 5) Liquidité fallback (scoped only)
    meta_scoped["avg_dollar_vol"] = pd.to_numeric(meta_scoped["avg_dollar_vol"], errors="coerce")
    meta_scoped = fill_liquidity_with_history_scoped(meta_scoped, batch_size=110)

    # 6) MCAP < 75B (souple ici)
    mcap_num = pd.to_numeric(meta_scoped["mcap_usd"], errors="coerce")
    in_scope = meta_scoped[mcap_num.fillna(0) < MCAP_MAX_USD].copy()

    # tri simple
    in_scope["mcap_usd"] = pd.to_numeric(in_scope["mcap_usd"], errors="coerce")
    in_scope["avg_dollar_vol"] = pd.to_numeric(in_scope["avg_dollar_vol"], errors="coerce")
    in_scope = in_scope.sort_values(["avg_dollar_vol","mcap_usd"], ascending=[False, True])

    _save(in_scope, "universe_in_scope.csv")
    _save(in_scope.rename(columns={"ticker_yf":"ticker_yf"}), "universe_today.csv")
    print(f"[DONE] build_universe complete. in_scope={len(in_scope)}")

if __name__ == "__main__":
    main()
