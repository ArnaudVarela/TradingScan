# build_sector_catalog.py — v1.2
import io, os, re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

PUBLIC_DIR = Path("dashboard/public"); PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent":"Mozilla/5.0"}

WIKI_R1K = "https://en.wikipedia.org/wiki/Russell_1000_Index"
SPDR_SPSM_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"

def _save(df: pd.DataFrame, name: str):
    df.to_csv(name, index=False)
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

def _norm_ticker(x: str) -> str:
    if x is None: return ""
    x = str(x).strip().upper().replace(".","-")
    x = re.sub(r"\s+","",x)
    return x

_SECTOR_NORM = {
    "technology":"information technology","tech":"information technology",
    "information technology":"information technology","information technology services":"information technology",
    "financial":"financials","financial services":"financials","finance":"financials","financials":"financials",
    "industrial":"industrials","industrials":"industrials",
    "healthcare":"health care","health services":"health care","health care":"health care",
}
def _norm_sector(s: str) -> str:
    k = str(s or "").strip().lower()
    return _SECTOR_NORM.get(k, k if k else "Unknown")

def _read_html_tables(url: str):
    r = requests.get(url, headers=UA, timeout=60); r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))

def from_wikipedia_r1k() -> pd.DataFrame:
    try:
        tables = _read_html_tables(WIKI_R1K)
    except Exception as e:
        print(f"[WARN] R1K fetch failed: {e}"); return pd.DataFrame(columns=["ticker","sector","industry"])
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(("symbol" in c or "ticker" in c) for c in cols) and any("sector" in c for c in cols):
            best = t; break
    if best is None:
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any(("symbol" in c or "ticker" in c) for c in cols):
                best = t; break
    if best is None:
        print("[WARN] R1K table not found"); return pd.DataFrame(columns=["ticker","sector","industry"])

    colmap = {}
    for c in best.columns:
        cl = str(c).lower()
        if "symbol" in cl or "ticker" in cl: colmap["ticker"] = c
        elif "sector" in cl and "sub" not in cl: colmap["sector"] = c
        elif ("sub" in cl and "industry" in cl) or ("gics" in cl and "industry" in cl): colmap.setdefault("industry", c)

    df = best.rename(columns={v:k for k,v in colmap.items()})[[*colmap]].copy()
    for c in ["ticker","sector","industry"]:
        if c not in df.columns: df[c] = ""
    df["ticker"] = df["ticker"].map(_norm_ticker)
    df = df[df["ticker"] != ""].drop_duplicates("ticker")
    df["sector"] = df["sector"].map(_norm_sector)
    df.loc[df["industry"]=="", "industry"] = "Unknown"
    print(f"[CAT] R1K mapping: {len(df)}")
    return df[["ticker","sector","industry"]]

def load_universe() -> pd.DataFrame:
    cand = ["universe_in_scope.csv", "raw_universe.csv", "universe_today.csv"]
    for p in cand:
        if os.path.exists(p):
            u = pd.read_csv(p); print(f"[UNI] {p}: {len(u)} rows")
            if "ticker_yf" in u.columns: u["ticker"] = u["ticker_yf"].map(_norm_ticker)
            elif "ticker" in u.columns:  u["ticker"] = u["ticker"].map(_norm_ticker)
            else: u["ticker"] = u.iloc[:,0].map(_norm_ticker)
            u = u[u["ticker"] != ""].drop_duplicates("ticker")
            return u[["ticker"]]
    raise SystemExit("❌ universe files not found")

def main():
    uni  = load_universe()
    r1k  = from_wikipedia_r1k()
    cat = r1k.sort_values("ticker").drop_duplicates("ticker")

    merged = uni.merge(cat, on="ticker", how="left")
    for c in ("sector","industry"):
        if c not in merged.columns: merged[c] = "Unknown"
    merged["sector"] = merged["sector"].map(_norm_sector)
    merged.loc[merged["industry"]=="", "industry"] = "Unknown"

    catalog = merged[["ticker","sector","industry"]].drop_duplicates("ticker")
    _save(catalog, "sector_catalog.csv")

    hist = catalog.copy()
    hist.insert(0, "date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    _save(hist, "sector_history.csv")

    breadth = catalog.groupby("sector", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    _save(breadth, "sector_breadth.csv")

if __name__ == "__main__":
    main()
