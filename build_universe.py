# build_universe.py
# Build the raw universe from sources + enrich with YF fast_info/dv20 + sector cache, then filter.
from __future__ import annotations
import os, io, time, math
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import pandas as pd
import requests
import yfinance as yf

from cache_layer import load_map, save_map, now_utc_iso

# ---------- config ----------
PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MCAP_MAX_USD = 75_000_000_000
REQ_TIMEOUT = 20
SLEEP = 0.1

# sectors retenus pour le scope
SECTOR_WHITELIST = {"Information Technology","Technology","Tech",
                    "Financials","Industrials","Health Care"}

ALPHAV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "")

# ---------- io helpers ----------
def _save(df: pd.DataFrame, name: str, also_public: bool = False):
    try:
        df.to_csv(name, index=False)
        if also_public:
            (PUBLIC_DIR / name).write_text(Path(name).read_text())
        print(f"[OK] wrote {name}: {len(df)} rows")
    except Exception as e:
        print(f"[SAVE] fail {name}: {e}")

# ---------- universe sources ----------
def _read_csv_local(name: str) -> Optional[pd.DataFrame]:
    for base in (Path("."), PUBLIC_DIR):
        p = base / name
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def load_russell_csvs() -> pd.DataFrame:
    # repo ships two files
    r1k = _read_csv_local("russell1000.csv")
    r2k = _read_csv_local("russell2000.csv")
    frames = []
    if r1k is not None:
        r1k = r1k.rename(columns={c: "ticker" if c.lower() in ("ticker","symbol") else c for c in r1k.columns})
        frames.append(r1k[["ticker"]])
    if r2k is not None:
        r2k = r2k.rename(columns={c: "ticker" if c.lower() in ("ticker","symbol") else c for c in r2k.columns})
        frames.append(r2k[["ticker"]])
    if not frames:
        return pd.DataFrame(columns=["ticker"])
    df = pd.concat(frames, ignore_index=True).dropna().drop_duplicates()
    return df

def load_nasdaq_lists() -> pd.DataFrame:
    # optional: if missing or 404, silently ignore
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    rows = []
    for url in urls:
        try:
            r = requests.get(url, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                content = r.text
                si = io.StringIO(content)
                tmp = pd.read_csv(si, sep="|")
                col = None
                for c in tmp.columns:
                    if c.lower() in ("symbol","activesymbols","ticker"):
                        col = c; break
                if col:
                    rows.append(tmp[[col]].rename(columns={col: "ticker"}))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame(columns=["ticker"])
    return pd.concat(rows, ignore_index=True).dropna().drop_duplicates()

def yf_fmt(t: str) -> str:
    t = (t or "").strip().upper()
    if t.endswith(".A"): t = t.replace(".A","-A")
    if t.endswith(".B"): t = t.replace(".B","-B")
    return t

# ---------- enrichment ----------
def enrich_fastinfo_and_dv20(tickers: List[str]) -> pd.DataFrame:
    fi_cache = load_map("yf_fastinfo", ttl_hours=24)
    dv_cache = load_map("dv20_cache", ttl_hours=72)

    rows = []
    to_fetch_fast: List[str] = []
    for t in tickers:
        rec = fi_cache.get(t)
        if rec and ("market_cap" in rec or "last_price" in rec or "ten_day_average_volume" in rec):
            mcap = rec.get("market_cap")
            lp   = rec.get("last_price")
            v10  = rec.get("ten_day_average_volume")
            adv  = (lp or 0) * (v10 or 0) if (lp is not None and v10 is not None) else None
            rows.append({"ticker_yf": t, "mcap_usd": mcap, "avg_dollar_vol": adv})
        else:
            to_fetch_fast.append(t)

    for i, t in enumerate(to_fetch_fast, 1):
        mcap = price = v10 = None
        try:
            fi = (yf.Ticker(t).fast_info) or {}
            mcap  = fi.get("market_cap")
            price = fi.get("last_price")
            v10   = fi.get("ten_day_average_volume")
        except Exception:
            pass
        fi_cache[t] = {"market_cap": mcap, "last_price": price, "ten_day_average_volume": v10, "cached_at": now_utc_iso()}
        adv = (price or 0) * (v10 or 0) if (price is not None and v10 is not None) else None
        rows.append({"ticker_yf": t, "mcap_usd": mcap, "avg_dollar_vol": adv})
        if i % 120 == 0:
            print(f"[YF] {i}/{len(to_fetch_fast)} enriched…")
            time.sleep(0.6)

    save_map("yf_fastinfo", fi_cache)

    meta = pd.DataFrame(rows)

    # dv20 fallback for missing avg_dollar_vol
    need = meta[meta["avg_dollar_vol"].isna()]["ticker_yf"].tolist()
    still = []
    for t in need:
        if dv_cache.get(t, {}).get("dv20") is not None:
            pass
        else:
            still.append(t)

    if still:
        try:
            df = yf.download(tickers=still, period="3mo", interval="1d",
                             auto_adjust=True, progress=False, group_by="ticker", threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                for t in still:
                    try:
                        sub = df[t][["Close","Volume"]].dropna()
                        if len(sub) >= 5:
                            dv20 = float((sub.tail(20)["Close"] * sub.tail(20)["Volume"]).mean())
                            dv_cache[t] = {"dv20": dv20, "asof": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
                    except Exception:
                        pass
            else:
                sub = df[["Close","Volume"]].dropna()
                if len(sub) >= 5:
                    dv20 = float((sub.tail(20)["Close"] * sub.tail(20)["Volume"]).mean())
                    dv_cache[still[0]] = {"dv20": dv20, "asof": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
        except Exception:
            pass
        save_map("dv20_cache", dv_cache)

    if need:
        meta["avg_dollar_vol"] = meta.apply(
            lambda r: r["avg_dollar_vol"] if pd.notna(r["avg_dollar_vol"])
            else (dv_cache.get(r["ticker_yf"], {}).get("dv20")), axis=1
        )
    print(f"[LIQ] dv20 filled: {sum(1 for t in need if (load_map('dv20_cache').get(t, {}).get('dv20') is not None))}")
    return meta

def sector_fill(df: pd.DataFrame, ticker_col="ticker_yf") -> pd.DataFrame:
    """Fill sector/industry using YF.info first, then optional AlphaVantage, both cached."""
    df = df.copy()
    sec_cache = load_map("sector_cache", ttl_hours=24*90)

    sectors, industries, missing = [], [], []
    symbols = df[ticker_col].astype(str).tolist()
    for sym in symbols:
        rec = sec_cache.get(sym)
        if rec and rec.get("sector"):
            sectors.append(rec["sector"]); industries.append(rec.get("industry"))
        else:
            sectors.append(None); industries.append(None); missing.append(sym)

    # yfinance.info
    for i, sym in enumerate(missing, 1):
        s = it = None
        try:
            info = yf.Ticker(sym).info
            s = info.get("sector"); it = info.get("industry")
        except Exception:
            pass
        if s:
            sec_cache[sym] = {"sector": s, "industry": it, "source": "yf", "cached_at": now_utc_iso()}
        if i % 25 == 0: print(f"[YF.info] {i}/{len(missing)}"); time.sleep(0.2)

    # AlphaVantage en complément si clé disponible
    still = [sym for sym in symbols if (sec_cache.get(sym, {}).get("sector") is None)]
    if ALPHAV_KEY and still:
        for i, sym in enumerate(still, 1):
            try:
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={sym}&apikey={ALPHAV_KEY}"
                r = requests.get(url, timeout=REQ_TIMEOUT)
                if r.status_code == 200:
                    data = r.json() if isinstance(r.json(), dict) else {}
                    s = data.get("Sector"); it = data.get("Industry")
                    if s:
                        sec_cache[sym] = {"sector": s, "industry": it, "source": "av", "cached_at": now_utc_iso()}
            except Exception:
                pass
            time.sleep(0.2)

    save_map("sector_cache", sec_cache)

    final_sec, final_ind = [], []
    for sym in symbols:
        rec = sec_cache.get(sym, {})
        final_sec.append(rec.get("sector"))
        final_ind.append(rec.get("industry"))
    df["sector"] = final_sec
    df["industry"] = final_ind
    return df

# ---------- pipeline ----------
def main():
    # 1) Sources
    ru = load_russell_csvs()
    nd = load_nasdaq_lists()
    union = pd.concat([ru, nd], ignore_index=True).dropna().drop_duplicates()
    union["ticker"] = union["ticker"].astype(str).str.upper().str.strip()
    union = union[~union["ticker"].str.contains("[^A-Z0-9\\-\\.]", regex=True)]
    union = union.rename(columns={"ticker": "ticker_yf"})
    union["ticker_yf"] = union["ticker_yf"].map(yf_fmt)

    print(f"[UNION] unique tickers before YF: {len(union)}")

    # 2) Enrich fast_info + dv20
    meta = enrich_fastinfo_and_dv20(union["ticker_yf"].tolist())
    base = union.merge(meta, on="ticker_yf", how="left")

    # 3) Raw universe (pour inspection)
    base_raw = base[["ticker_yf"]].copy()
    base_raw["ticker_tv"] = base_raw["ticker_yf"].str.replace("-", ".", regex=False)
    base_raw["mcap_usd"] = pd.to_numeric(base.get("mcap_usd"), errors="coerce")
    base_raw["avg_dollar_vol"] = pd.to_numeric(base.get("avg_dollar_vol"), errors="coerce")
    base_raw["sector"] = None
    base_raw["industry"] = None
    _save(base_raw[["ticker_yf","ticker_tv","mcap_usd","avg_dollar_vol","sector","industry"]], "raw_universe.csv")

    # 4) Focus top by liquidity to fetch sector with priority (optionnel)
    top_for_sector = base_raw.sort_values("avg_dollar_vol", ascending=False, na_position="last").head(180).copy()
    top_for_sector = sector_fill(top_for_sector, ticker_col="ticker_yf")

    # 5) Merge sector back into base_raw; puis broad fill sur le reste (cache rend ça très rapide après 1er run)
    base_raw = base_raw.merge(
        top_for_sector[["ticker_yf","sector","industry"]].drop_duplicates(),
        on="ticker_yf", how="left", suffixes=("","_top")
    )
    base_raw["sector"] = base_raw["sector"].where(base_raw["sector"].notna(), base_raw["sector_top"])
    base_raw["industry"] = base_raw["industry"].where(base_raw["industry"].notna(), base_raw["industry_top"])
    base_raw.drop(columns=[c for c in ("sector_top","industry_top") if c in base_raw.columns], inplace=True)

    # fill missing sector for the rest using cache/AV/YF (très peu d’API après 1er run)
    need_sec = base_raw[base_raw["sector"].isna()].copy()
    if len(need_sec):
        need_sec = sector_fill(need_sec, ticker_col="ticker_yf")
        base_raw = base_raw.drop(columns=["sector","industry"]).merge(
            need_sec[["ticker_yf","sector","industry"]], on="ticker_yf", how="left"
        )

    # 6) Universe in scope: secteurs whitelist et MCAP < 75B (NaN gardés pour fill ultérieur par mix script)
    in_scope_mask = base_raw["sector"].isin(SECTOR_WHITELIST) | base_raw["sector"].isna()
    in_scope = base_raw[in_scope_mask].copy()
    print(f"[FILTER] sectors∈(Tech/Fin/Ind/HC) & mcap<75B (NaN kept for later fill): {len(in_scope)}/{len(base_raw)}")

    _save(in_scope, "universe_in_scope.csv", also_public=True)
    _save(in_scope, "universe_today.csv", also_public=True)

if __name__ == "__main__":
    main()
