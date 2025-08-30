# build_sector_catalog.py
# Agrège les secteurs/industries depuis:
# - Wikipedia Russell 1000
# - SPDR SPSM (Small Cap) XLSX
# Sortie: sector_catalog.csv -> colonnes: ticker, sector, industry

import pandas as pd
import requests
import io
import re

WIKI_R1K = "https://en.wikipedia.org/wiki/Russell_1000_Index"
SPDR_SPSM_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"

def norm_ticker(x: str) -> str:
    if x is None: return ""
    x = str(x).strip().upper()
    x = re.sub(r"\s+", "", x)
    # uniformiser BRK.B style TV
    return x

def from_wikipedia_r1k() -> pd.DataFrame:
    ua = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(WIKI_R1K, headers=ua, timeout=45)
    r.raise_for_status()
    tables = pd.read_html(r.text)

    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c or "ticker" in c for c in cols) and any("sector" in c for c in cols):
            best = t
            break
    if best is None:
        # certains snapshots ont les intitulés GICS différents; tente le plus large
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("symbol" in c or "ticker" in c for c in cols):
                best = t
                break

    if best is None:
        return pd.DataFrame(columns=["ticker","sector","industry"])

    # map colonnes
    colmap = {}
    for c in best.columns:
        cl = str(c).lower()
        if "symbol" in cl or "ticker" in cl:
            colmap["ticker"] = c
        elif "sector" in cl:
            colmap["sector"] = c
        elif "sub" in cl and "industry" in cl:
            colmap["industry"] = c
        elif "gics" in cl and "industry" in cl and "sub" not in cl:
            # parfois 'GICS Industry' existe
            colmap.setdefault("industry", c)

    df = best.rename(columns={v:k for k,v in colmap.items() if v in best.columns})[list(colmap.keys())]
    df["ticker"] = df["ticker"].map(norm_ticker)
    df["sector"] = df.get("sector", "").fillna("").astype(str).str.strip()
    df["industry"] = df.get("industry", "").fillna("").astype(str).str.strip()
    df = df.dropna(subset=["ticker"])
    df = df[df["ticker"].str.match(r"^[A-Z.\-]+$")]
    return df[["ticker","sector","industry"]]

def from_spdr_spsm() -> pd.DataFrame:
    ua = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(SPDR_SPSM_XLSX, headers=ua, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    # feuille 'holdings' généralement
    name = [s for s in xls.sheet_names if "holding" in s.lower()]
    sh = name[0] if name else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sh)
    # essayer différents noms de colonnes
    cols = {str(c).lower(): c for c in df.columns}
    tick = cols.get("ticker") or cols.get("ticker symbol:") or cols.get("symbol") or cols.get("isin")
    sec  = cols.get("sector") or cols.get("gics sector") or cols.get("economic sector")
    sub  = cols.get("industry") or cols.get("sub-industry") or cols.get("gics sub-industry")
    out = pd.DataFrame(columns=["ticker","sector","industry"])
    if tick is None:
        return out
    out["ticker"] = df[tick].map(norm_ticker)
    out["sector"] = df[sec].astype(str).str.strip() if sec in df.columns else ""
    out["industry"] = df[sub].astype(str).str.strip() if sub in df.columns else ""
    out = out.dropna(subset=["ticker"])
    out = out[out["ticker"].str.match(r"^[A-Z.\-]+$")]
    return out

def main():
    r1k = from_wikipedia_r1k()
    spsm = from_spdr_spsm()
    cat = pd.concat([r1k, spsm], ignore_index=True)
    cat = (cat
           .sort_values(["ticker","sector"])
           .drop_duplicates(subset=["ticker"], keep="first")
           .reset_index(drop=True))
    # clean vides
    cat["sector"] = cat["sector"].replace({"nan":""})
    cat["industry"] = cat["industry"].replace({"nan":""})
    cat.to_csv("sector_catalog.csv", index=False)
    print(f"[OK] sector_catalog.csv écrit: {len(cat)} lignes "
          f"(R1K={len(r1k)}, SPSM={len(spsm)})")

if __name__ == "__main__":
    main()
