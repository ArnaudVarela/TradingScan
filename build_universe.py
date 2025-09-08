# build_universe.py
# Robust US universe builder with caching + strict ticker sanitation.
# Universe sources (best-effort): Russell 1000, S&P 500, NASDAQ-100, NASDAQ Composite.
# Enrichment: yfinance fast_info (mcap/price), 20D avg dollar volume (dv20) from history.
# Filters: keep only sectors in {Technology, Financials, Industrials, Health Care} later;
#          here we DO NOT drop NaN mcap (the screener will coalesce again).
# Outputs: raw_universe.csv, universe_in_scope.csv, universe_today.csv (root + dashboard/public)

import os, re, time, math, json
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import pandas as pd
import requests
import yfinance as yf

# ---------- Caching ----------
try:
    # cache_layer.py expected to expose a simple dict-like API.
    # Minimal interface used here:
    #   cache.get(key) -> object | None
    #   cache.set(key, obj, ttl_hours: int | None = None) -> None
    #   cache.get_df(key) / cache.set_df(key, df, ttl_hours=None)
    from cache_layer import cache
except Exception:
    # Fallback no-op cache to stay runnable locally.
    class _NoCache:
        def get(self, k): return None
        def set(self, k, v, ttl_hours=None): pass
        def get_df(self, k): return None
        def set_df(self, k, v, ttl_hours=None): pass
    cache = _NoCache()

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

REQ_TIMEOUT = int(os.getenv("REQ_TIMEOUT", "20"))
SLEEP_BETWEEN_HTTP = float(os.getenv("SLEEP_BETWEEN_HTTP", "0.25"))
YF_BATCH = int(os.getenv("YF_BATCH", "50"))
MAX_UNIVERSE = int(os.getenv("MAX_UNIVERSE", "12000"))  # hard cap to avoid runaway batches

# ---------- Saving helpers ----------
def _save(df: pd.DataFrame, name: str, also_public: bool = True):
    try:
        p = Path(name)
        df.to_csv(p, index=False)
        if also_public:
            q = PUBLIC_DIR / name
            df.to_csv(q, index=False)
        print(f"[SAVE] {name} | rows={len(df)} | cols={len(df.columns)}")
    except Exception as e:
        print(f"[SAVE] failed for {name}: {e}")

# ---------- Fetch helpers ----------
def _get(url: str) -> requests.Response | None:
    try:
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code == 200:
            return r
        print(f"[HTTP] {url} -> {r.status_code}")
        return None
    except Exception as e:
        print(f"[HTTP] {url} exception: {e}")
        return None
    finally:
        time.sleep(SLEEP_BETWEEN_HTTP)

def _read_remote_csv(urls: List[str], use_cache_key: str) -> pd.DataFrame:
    # cache first
    cached = cache.get_df(use_cache_key)
    if cached is not None and len(cached) > 0:
        print(f"[CACHE] hit: {use_cache_key} ({len(cached)})")
        return cached.copy()

    for u in urls:
        r = _get(u)
        if r is None:
            continue
        try:
            df = pd.read_csv(pd.compat.StringIO(r.text))
        except Exception:
            try:
                df = pd.read_csv(pd.compat.StringIO(r.text), sep=";")
            except Exception:
                try:
                    df = pd.read_csv(pd.compat.StringIO(r.text), engine="python")
                except Exception as e:
                    print(f"[CSV] parse failed for {u}: {e}")
                    continue
        if len(df) > 0:
            cache.set_df(use_cache_key, df, ttl_hours=24)
            print(f"[SRC] {use_cache_key} <- {u} ({len(df)})")
            return df.copy()

    print(f"[SRC] {use_cache_key} not available; returning empty df")
    return pd.DataFrame()

def _sanitize_tickers(series: Iterable[str]) -> List[str]:
    """
    Keep only US-like Yahoo tickers:
      - uppercase A-Z, 0-9, dot, hyphen
      - length <= 6 (allow up to 7 if contains a dot for classes, e.g. BRK.B)
      - exclude words that are clearly not tickers (countries, indices, placeholders)
      - exclude those containing spaces, '/', '\', ':'
    """
    bad_tokens = {
        "USA","US","ISRAEL","IRELAND","CHINA","EU","EURO","PANAMA","BELGIUM","SWEDEN","BRAZIL",
        "ENERGY","SOFTWARE","AIRLINES","REIT","CURRENCY","MEDIA","INDEX","INDUSTRY","SECTOR",
        "ZCZZT","ZJZZT","ZVZZT","ZWZZT","ZXZZT","ZOOZ","ZOOZW","ZOOZT",
    }
    out: List[str] = []
    pat = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,6}$")  # len<=7
    for raw in series:
        if not isinstance(raw, str):
            continue
        t = raw.strip().upper()
        if not t or t in bad_tokens:
            continue
        if any(ch in t for ch in (" ", "/", "\\", ":", ",")):
            continue
        if len(t) > 7:
            continue
        if not pat.match(t):
            continue
        # Heuristic: drop obvious codes that are not US tickers (all letters+digits length 7 with no dot/hyphen and ending by digits)
        if len(t) >= 7 and "." not in t and "-" not in t:
            continue
        # Drop Yahoo region suffixes (we keep US only)
        if t.endswith((".L",".SW",".PA",".DE",".TO",".V",".HK",".AX",".SI",".MI",".VI",".F",".SS",".SZ",".TW",".KS",".KQ",".CO",".HE",".OL",".ST",".MC",".WA",".PR",".IR",".ME",".IL",".TA",".SA",".MX",".SN",".BO",".BK",".T",".SR",".ZM",".BE",".SG",".VI",".MC",".BR",".VX",".IS",".AT",".AS",".MF")):
            continue
        out.append(t)
    return sorted(set(out))

# ---------- Universe sources (multiple fallbacks; you can update URLs over time) ----------
def _fetch_sp500() -> pd.DataFrame:
    urls = [
        # GitHub mirror of Wikipedia S&P500 (common in many repos)
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        # Stooq mirror
        "https://stooq.com/t/?s=^spx&i=d",
    ]
    df = _read_remote_csv(urls, "SP500_SRC")
    # try common column names
    for c in ("Symbol","symbol","Ticker","ticker","code"):
        if c in df.columns:
            return pd.DataFrame({"ticker": df[c].astype(str)})
    return pd.DataFrame(columns=["ticker"])

def _fetch_russell1000() -> pd.DataFrame:
    urls = [
        "https://www.cboe.com/us/equities/market_statistics/russell_1000/",
        "https://www.alphaspread.com/security-list/usa/russell-1000",
    ]
    # a lot of pages above are HTML; if our CSV reader fails, we just return empty and rely on cache/other lists
    df = _read_remote_csv(urls, "R1000_SRC")
    for c in ("symbol","Symbol","Ticker","ticker","code"):
        if c in df.columns:
            return pd.DataFrame({"ticker": df[c].astype(str)})
    return pd.DataFrame(columns=["ticker"])

def _fetch_nasdaq100() -> pd.DataFrame:
    urls = [
        "https://datahub.io/core/nasdaq-listings/r/constituents.csv",
        "https://pkgstore.datahub.io/core/nasdaq-listings/constituents_csv/data/constituents_csv.csv",
    ]
    df = _read_remote_csv(urls, "NDX_SRC")
    for c in ("Symbol","symbol","Ticker","ticker","code"):
        if c in df.columns:
            return pd.DataFrame({"ticker": df[c].astype(str)})
    return pd.DataFrame(columns=["ticker"])

def _fetch_nasdaq_composite() -> pd.DataFrame:
    urls = [
        # community-maintained; if not reachable we just skip
        "https://raw.githubusercontent.com/robinhood-unofficial/pyrh/master/examples/nasdaq_screener.csv",
    ]
    df = _read_remote_csv(urls, "IXIC_SRC")
    for c in ("Symbol","symbol","Ticker","ticker","code"):
        if c in df.columns:
            return pd.DataFrame({"ticker": df[c].astype(str)})
    return pd.DataFrame(columns=["ticker"])

# ---------- yfinance enrichment ----------
def _yf_fast_info(tickers: List[str]) -> Dict[str, Dict[str, float | None]]:
    """
    Cheap per-ticker pulls for price + marketCap. Cached per symbol.
    """
    out: Dict[str, Dict[str, float | None]] = {}
    for i, t in enumerate(tickers, 1):
        ck = f"fi::{t}"
        cached = cache.get(ck)
        if cached is not None:
            out[t] = cached
            continue
        price = None; mcap = None
        try:
            info = yf.Ticker(t).fast_info
            price = float(getattr(info, "last_price", None)) if getattr(info, "last_price", None) is not None else None
            mcap = float(getattr(info, "market_cap", None)) if getattr(info, "market_cap", None) is not None else None
        except Exception:
            pass
        out[t] = {"price": price, "mcap_usd": mcap}
        cache.set(ck, out[t], ttl_hours=24)
        if i % 200 == 0:
            print(f"[YF.fast] {i}/{len(tickers)}")
        time.sleep(0.02)
    return out

def _yf_dv20_batch(tickers: List[str]) -> Dict[str, float]:
    """
    Batched 20-day avg dollar volume: mean(Close*Volume) over last ~20 sessions.
    Cached per day per ticker (key dv20::<YYYYMMDD>::TICKER).
    """
    out: Dict[str, float] = {}
    if not tickers:
        return out
    today_key = time.strftime("%Y%m%d")
    remaining = []
    # cache hits
    for t in tickers:
        ck = f"dv20::{today_key}::{t}"
        cached = cache.get(ck)
        if isinstance(cached, (int, float)) and math.isfinite(cached):
            out[t] = float(cached)
        else:
            remaining.append(t)

    if not remaining:
        return out

    for i in range(0, len(remaining), YF_BATCH):
        batch = remaining[i:i+YF_BATCH]
        try:
            df = yf.download(
                tickers=batch, period="3mo", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker", threads=True
            )
        except Exception as e:
            print(f"[YF.hist] batch {i//YF_BATCH+1} exception: {e}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            for t in batch:
                try:
                    sub = df[t][["Close","Volume"]].dropna()
                    if len(sub) >= 5:
                        tail = sub.tail(20)
                        dv = float((tail["Close"] * tail["Volume"]).mean())
                        out[t] = dv
                        cache.set(f"dv20::{today_key}::{t}", dv, ttl_hours=36)
                except Exception:
                    # skip silently
                    pass
        else:
            # Single ticker edge
            try:
                sub = df[["Close","Volume"]].dropna()
                if len(sub) >= 5:
                    dv = float((sub.tail(20)["Close"] * sub.tail(20)["Volume"]).mean())
                    out[batch[0]] = dv
                    cache.set(f"dv20::{today_key}::{batch[0]}", dv, ttl_hours=36)
            except Exception:
                pass

        print(f"[YF.hist] {min(i+YF_BATCH,len(remaining))}/{len(remaining)} dv20 filled")
        time.sleep(0.5)

    return out

# ---------- main ----------
def main():
    print("[STEP] build_universe starting…")

    # 1) Gather sources (best-effort; each is optional)
    # Try cache of the merged universe first (to avoid hammering sources)
    merged_cached = cache.get_df("UNIVERSE::MERGED")
    if merged_cached is None:
        pools: List[pd.DataFrame] = []
        try:
            pools.append(_fetch_sp500())
        except Exception: pass
        try:
            pools.append(_fetch_russell1000())
        except Exception: pass
        try:
            pools.append(_fetch_nasdaq100())
        except Exception: pass
        try:
            pools.append(_fetch_nasdaq_composite())
        except Exception: pass

        raw = pd.concat([p for p in pools if p is not None and len(p) > 0], ignore_index=True) \
               if pools else pd.DataFrame(columns=["ticker"])
        if "ticker" not in raw.columns:
            raw["ticker"] = []
        raw["ticker"] = raw["ticker"].astype(str)
        raw = raw.dropna().drop_duplicates()
        cache.set_df("UNIVERSE::MERGED", raw, ttl_hours=6)
    else:
        print(f"[CACHE] hit: UNIVERSE::MERGED ({len(merged_cached)})")
        raw = merged_cached.copy()

    if len(raw) == 0:
        # Absolute last resort: use yesterday's universe if present
        for base in (Path("."), PUBLIC_DIR):
            p = base / "universe_today.csv"
            if p.exists():
                print("[FALLBACK] Using previous universe_today.csv")
                raw = pd.read_csv(p)
                break

    # 2) Sanitize & hard cap
    raw["ticker"] = raw["ticker"].astype(str)
    tickers = _sanitize_tickers(raw["ticker"])
    if len(tickers) == 0:
        raise SystemExit("❌ No usable tickers after sanitation.")
    if len(tickers) > MAX_UNIVERSE:
        print(f"[CAP] trimming {len(tickers)} → {MAX_UNIVERSE}")
        tickers = tickers[:MAX_UNIVERSE]

    # Persist raw for debugging
    _save(pd.DataFrame({"ticker": tickers}), "raw_universe.csv")

    # 3) Map to yfinance symbols (US: same)
    df = pd.DataFrame({"ticker_yf": tickers})

    # 4) Enrichment: fast_info (price, mcap)
    fi = _yf_fast_info(tickers)
    df["price"] = df["ticker_yf"].map(lambda t: fi.get(t, {}).get("price"))
    df["mcap_usd"] = df["ticker_yf"].map(lambda t: fi.get(t, {}).get("mcap_usd"))

    # 5) dv20
    dv = _yf_dv20_batch(tickers)
    df["avg_dollar_vol"] = df["ticker_yf"].map(dv).astype(float)

    # 6) Light cleaning
    # (do NOT drop NaN mcap here; screener will complete via other sources)
    df = df.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)

    # Sorting for convenience
    df["avg_dollar_vol"] = pd.to_numeric(df["avg_dollar_vol"], errors="coerce")
    df = df.sort_values("avg_dollar_vol", ascending=False, na_position="last")

    # 7) Outputs for next stages
    _save(df, "universe_in_scope.csv")
    _save(df, "universe_today.csv")

    print(f"[DONE] universe_in_scope: {len(df)}  | NaN mcap kept for later coalescence")

if __name__ == "__main__":
    main()
