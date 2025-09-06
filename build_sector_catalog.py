# build_sector_catalog.py
# Merge sector/industry maps (Wikipedia R1K + SPDR SPSM) over current universe.
# Outputs: sector_catalog.csv, sector_history.csv, sector_breadth.csv

import io, os, re
from pathlib import Path
import pandas as pd
import requests

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}

WIKI_R1K = "https://en.wikipedia.org/wiki/Russell_1000_Index"
SPDR_SPSM_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"

def _save(df: pd.DataFrame, name: str):
    df.to_csv(name, index=False)
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

def _norm_ticker(x: str) -> str:
    if x is None: return ""
    x = str(x).strip().upper().replace(".", "-")
    x = re.sub(r"[^A-Z0-9\-]", "", x)
    return x

def _read_html_tables(url: str):
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))

def from_wikipedia_r1k() -> pd.DataFrame:
    tables = _read_html_tables(WIKI_R1K)
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(("symbol" in c or "ticker" in c) for c in cols) and any("sector" in c for c in cols):
            best = t; break
    if best is None:
        return pd.DataFrame(columns=["ticker","sector","industry"])
    colmap = {}
    for c in best.columns:
        cl = str(c).lower()
        if "symbol" in cl or "ticker" in cl: colmap["ticker"] = c
        elif "sector" in cl and "sub" not in cl: colmap["sector"] = c
        elif ("industry" in cl) or ("sub-industry" in cl): colmap.setdefault("industry", c)
    df = best.rename(columns=colmap)
    out = pd.DataFrame()
    out["ticker"] = df["ticker"].map(_norm_ticker)
    out["sector"] = df.get("sector", "").astype(str).str.strip()
    out["industry"] = df.get("industry", "").astype(str).str.strip()
    out = out[out["ticker"] != ""].drop_duplicates("ticker")
    print(f"[CAT] R1K mapping: {len(out)}")
    return out

def from_spdr_spsm() -> pd.DataFrame:
    r = requests.get(SPDR_SPSM_XLSX, headers=UA, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    sh = next((s for s in xls.sheet_names if "holding" in s.lower()), xls.sheet_names[0])
    df = pd.read_excel(xls, sheet_name=sh)
    cols = {str(c).lower(): c for c in df.columns}
    tick = cols.get("ticker") or cols.get("ticker symbol:") or cols.get("symbol") or cols.get("isin")
    sec  = cols.get("sector") or cols.get("gics sector") or cols.get("economic sector")
    sub  = cols.get("industry") or cols.get("sub-industry") or cols.get("gics sub-industry")
    if tick is None:
        return pd.DataFrame(columns=["ticker","sector","industry"])
    out = pd.DataFrame()
    out["ticker"] = df[tick].map(_norm_ticker)
    out["sector"] = df[sec].astype(str).str.strip() if sec in df.columns else ""
    out["industry"] = df[sub].astype(str).str.strip() if sub in df.columns else ""
    out = out[out["ticker"] != ""].drop_duplicates("ticker")
    print(f"[CAT] SPSM mapping: {len(out)}")
    return out

def load_universe() -> pd.DataFrame:
    for p in ["universe_in_scope.csv","universe_today.csv"]:
        if os.path.exists(p):
            u = pd.read_csv(p)
            print(f"[UNI] {p}: {len(u)} rows")
            if "ticker_yf" in u.columns:
                u["ticker"] = u["ticker_yf"].map(_norm_ticker)
            else:
                u["ticker"] = u.iloc[:,0].map(_norm_ticker)
            if "sector" not in u.columns:
                u["sector"] = ""
            out = u[["ticker","sector"]].drop_duplicates("ticker")
            return out
    raise SystemExit("❌ universe_in_scope.csv / universe_today.csv introuvable")

def main():
    uni = load_universe()
    r1k = from_wikipedia_r1k()
    spsm = from_spdr_spsm()
    cat = pd.concat([r1k, spsm], ignore_index=True).sort_values("ticker").drop_duplicates("ticker")
    merged = uni.merge(cat, on="ticker", how="left", suffixes=("", "_cat"))
    merged["sector"] = merged.apply(
        lambda r: (str(r.get("sector_cat","")).strip() or str(r.get("sector","")).strip() or "Unknown"),
        axis=1,
    )
    merged["industry"] = merged.get("industry","").fillna("").astype(str).str.strip().replace({"": "Unknown"})
    catalog = merged[["ticker","sector","industry"]].drop_duplicates("ticker")
    _save(catalog, "sector_catalog.csv")

    history = catalog.copy()
    _save(history, "sector_history.csv")

    breadth = catalog.groupby("sector", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    _save(breadth, "sector_breadth.csv")

if __name__ == "__main__":
    main()
