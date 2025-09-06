# build_sector_catalog.py
# Agrège secteurs/industries: Wikipedia R1K + SPDR SPSM (quand dispo)
# Normalise les libellés de secteur (GICS-like), garde l'univers complet
# Sorties: sector_catalog.csv, sector_history.csv (daté), sector_breadth.csv

import io
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

PUBLIC_DIR = Path("dashboard/public"); PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
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
    x = str(x).strip().upper()
    x = re.sub(r"\s+", "", x).replace(".", "-")
    return x

# ---- Sector normalization ----
_SECTOR_NORM = {
    "technology":"information technology", "tech":"information technology",
    "information technology":"information technology", "information technology services":"information technology",
    "financial":"financials", "financial services":"financials", "finance":"financials", "financials":"financials",
    "industrial":"industrials", "industrials":"industrials",
    "healthcare":"health care", "health services":"health care", "health care":"health care",
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

    out_cols = [k for k in ["ticker","sector","industry"] if k in colmap]
    df = best.rename(columns={v:k for k,v in colmap.items()})[out_cols].copy()

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].map(_norm_ticker)
        df = df[df["ticker"] != ""]
        df = df[df["ticker"].str.match(r"^[A-Z0-9\-\.]+$")]

    for c in ["sector","industry"]:
        if c in df.columns: df[c] = df[c].fillna("").astype(str).str.strip()
        else: df[c] = ""

    df["sector"] = df["sector"].map(_norm_sector)
    df.loc[df["industry"]=="", "industry"] = "Unknown"
    print(f"[CAT] R1K mapping: {len(df)}")
    return df[["ticker","sector","industry"]]

def from_spdr_spsm() -> pd.DataFrame:
    try:
        r = requests.get(SPDR_SPSM_XLSX, headers=UA, timeout=60); r.raise_for_status()
        xls = pd.ExcelFile(io.BytesIO(r.content))
    except Exception as e:
        print(f"[WARN] SPSM fetch failed: {e}")
        return pd.DataFrame(columns=["ticker","sector","industry"])
    sh = next((s for s in xls.sheet_names if "holding" in s.lower()), xls.sheet_names[0])
    df = pd.read_excel(xls, sheet_name=sh)

    cols = {str(c).lower(): c for c in df.columns}
    tick = cols.get("ticker") or cols.get("ticker symbol:") or cols.get("symbol") or cols.get("isin")
    sec  = cols.get("sector") or cols.get("gics sector") or cols.get("economic sector")
    sub  = cols.get("industry") or cols.get("sub-industry") or cols.get("gics sub-industry")

    if tick is None: return pd.DataFrame(columns=["ticker","sector","industry"])

    out = pd.DataFrame()
    out["ticker"] = df[tick].map(_norm_ticker)
    out["sector"] = df[sec].astype(str).str.strip() if sec in df.columns else ""
    out["industry"] = df[sub].astype(str).str.strip() if sub in df.columns else ""
    out = out[out["ticker"] != ""].drop_duplicates(subset=["ticker"], keep="first")
    out["sector"] = out["sector"].map(_norm_sector)
    out.loc[out["industry"]=="", "industry"] = "Unknown"
    print(f"[CAT] SPSM mapping: {len(out)}")
    return out[["ticker","sector","industry"]]

def load_universe() -> pd.DataFrame:
    cand = ["universe_in_scope.csv", "universe_today.csv"]
    for p in cand:
        if os.path.exists(p):
            u = pd.read_csv(p); print(f"[UNI] {p}: {len(u)} rows")
            out = u.rename(columns={"ticker":"ticker_yf"}) if "ticker" in u.columns else u
            if "ticker_yf" in out.columns: out["ticker"] = out["ticker_yf"].map(_norm_ticker)
            elif "Ticker" in out.columns:  out["ticker"] = out["Ticker"].map(_norm_ticker)
            else: out["ticker"] = out.iloc[:,0].map(_norm_ticker)
            if "sector" not in out.columns: out["sector"] = ""
            out = out[["ticker","sector"] + [c for c in out.columns if c not in {"ticker","sector"}]]
            out = out[out["ticker"] != ""].drop_duplicates(subset=["ticker"], keep="first")
            return out
    raise SystemExit("❌ universe_in_scope.csv / universe_today.csv introuvable")

def main():
    uni  = load_universe()     # ticker, sector (optionnel)
    r1k  = from_wikipedia_r1k()
    spsm = from_spdr_spsm()

    cat = pd.concat([r1k, spsm], ignore_index=True)
    cat = cat.sort_values(["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    print(f"[CAT] merged map unique: {len(cat)}")

    merged = uni.merge(cat, on="ticker", how="left", suffixes=("", "_cat"))

    if "sector_cat" not in merged.columns: merged["sector_cat"] = ""
    if "industry" not in merged.columns:   merged["industry"]   = ""

    def pick_sector(row):
        s_cat = str(row.get("sector_cat","")).strip()
        s_uni = str(row.get("sector","")).strip()
        return s_cat if s_cat else s_uni

    merged["sector"]   = merged.apply(pick_sector, axis=1)
    merged["industry"] = merged["industry"].fillna("").astype(str).str.strip()

    merged["sector"]   = merged["sector"].map(_norm_sector)
    merged.loc[merged["industry"]=="", "industry"] = "Unknown"

    catalog = merged[["ticker","sector","industry"]].drop_duplicates(subset=["ticker"], keep="first")
    _save(catalog, "sector_catalog.csv")

    # history daté (UTC date)
    history = catalog.copy()
    history.insert(0, "date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    _save(history, "sector_history.csv")

    breadth = catalog.groupby("sector", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    _save(breadth, "sector_breadth.csv")

if __name__ == "__main__":
    main()
