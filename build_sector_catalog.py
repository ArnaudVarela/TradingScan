# build_sector_catalog.py
# Mappe les tickers de universe_today.csv -> {sector, industry},
# en fusionnant R1000 (Wikipedia) + SPSM (SPDR Excel) + fallback Yahoo,
# puis filtre les secteurs cibles et écrit universe_in_scope.csv + breadth/history.

import io
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0"}

# Secteurs à conserver (exact wording GICS le plus courant)
IN_SCOPE_SECTORS = {
    "Industrials",
    "Information Technology",
    "Health Care",
    "Financials",
}

WIKI_R1K = "https://en.wikipedia.org/wiki/Russell_1000_Index"
SPDR_SPSM_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"

def _save(df: pd.DataFrame, name: str):
    df.to_csv(name, index=False)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

def _norm_yf(t: str) -> str | None:
    if not isinstance(t, str): return None
    t = t.strip().upper()
    if not t: return None
    return t.replace(".", "-")

def load_universe() -> pd.DataFrame:
    path = Path("universe_today.csv")
    if not path.exists():
        raise SystemExit("❌ universe_today.csv introuvable (lance d’abord build_universe.py)")
    df = pd.read_csv(path)
    if "ticker_yf" not in df.columns:
        raise SystemExit("❌ universe_today.csv doit contenir 'ticker_yf'")
    df["ticker_yf"] = df["ticker_yf"].astype(str).map(_norm_yf)
    return df.dropna(subset=["ticker_yf"]).drop_duplicates("ticker_yf")

def from_wikipedia_r1k() -> pd.DataFrame:
    r = requests.get(WIKI_R1K, headers=UA, timeout=45)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(("symbol" in c or "ticker" in c) for c in cols) and any("sector" in c for c in cols):
            best = t; break
    if best is None:
        return pd.DataFrame(columns=["ticker_yf","sector","industry"])
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
            colmap.setdefault("industry", c)
    tmp = best.rename(columns={v:k for k,v in colmap.items() if v in best.columns})[list(colmap.keys())]
    out = pd.DataFrame({
        "ticker_yf": tmp["ticker"].astype(str).map(_norm_yf),
        "sector": tmp.get("sector","").astype(str).str.strip(),
        "industry": tmp.get("industry","").astype(str).str.strip()
    })
    out = out.dropna(subset=["ticker_yf"]).drop_duplicates("ticker_yf")
    print(f"[CAT] R1K mapping: {len(out)}")
    return out

def from_spdr_spsm() -> pd.DataFrame:
    r = requests.get(SPDR_SPSM_XLSX, headers=UA, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    sh = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sh)
    cols = {str(c).lower(): c for c in df.columns}
    tick = cols.get("ticker") or cols.get("ticker symbol:") or cols.get("symbol") or cols.get("isin")
    sec  = cols.get("sector") or cols.get("gics sector") or cols.get("economic sector")
    sub  = cols.get("industry") or cols.get("sub-industry") or cols.get("gics sub-industry")
    if not tick:
        return pd.DataFrame(columns=["ticker_yf","sector","industry"])
    out = pd.DataFrame({
        "ticker_yf": df[tick].astype(str).map(_norm_yf),
        "sector": (df[sec].astype(str).str.strip() if sec in df.columns else ""),
        "industry": (df[sub].astype(str).str.strip() if sub in df.columns else "")
    })
    out = out.dropna(subset=["ticker_yf"]).drop_duplicates("ticker_yf")
    print(f"[CAT] SPSM mapping: {len(out)}")
    return out

def yahoo_fill_missing(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).get_info()  # .info est déprécié, get_info garde le comportement
            sector = info.get("sector")
            industry = info.get("industry")
            if sector or industry:
                rows.append({"ticker_yf": t, "sector": sector or "", "industry": industry or ""})
        except Exception:
            pass
        if i % 50 == 0:
            print(f"[YF] sector fill {i}/{len(tickers)}…")
            time.sleep(0.2)
    return pd.DataFrame(rows)

def main():
    uni = load_universe()
    print(f"[UNI] universe_today.csv: {len(uni)} rows")

    r1k = from_wikipedia_r1k()
    spsm = from_spdr_spsm()

    cat = pd.concat([r1k, spsm], ignore_index=True)
    cat = (cat.sort_values(["ticker_yf","sector"])
              .drop_duplicates(subset=["ticker_yf"], keep="first")
              .reset_index(drop=True))

    # Compléter les manquants via Yahoo (sur l'univers du jour seulement)
    missing = sorted(set(uni["ticker_yf"]) - set(cat["ticker_yf"]))
    print(f"[CAT] missing after R1K+SPSM: {len(missing)} to try via Yahoo…")
    if missing:
        miss_df = yahoo_fill_missing(missing)
        if not miss_df.empty:
            cat = pd.concat([cat, miss_df], ignore_index=True)
            cat = (cat.sort_values(["ticker_yf"])
                     .drop_duplicates(subset=["ticker_yf"], keep="first"))

    # Écrire le catalogue complet
    cat["sector"] = cat["sector"].fillna("").astype(str)
    cat["industry"] = cat["industry"].fillna("").astype(str)
    _save(cat[["ticker_yf","sector","industry"]], "sector_catalog.csv")

    # Projeter le mapping sur l'univers du jour
    merged = uni.merge(cat, on="ticker_yf", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["industry"] = merged["industry"].fillna("Unknown")

    # Filtre secteurs in-scope
    in_scope = merged[merged["sector"].isin(IN_SCOPE_SECTORS)].copy()
    _save(in_scope, "universe_in_scope.csv")

    # Breadth + History
    br = (in_scope.groupby("sector", dropna=False)["ticker_yf"]
                  .nunique().reset_index(name="count")
                  .sort_values("count", ascending=False))
    _save(br, "sector_breadth.csv")

    # Append weekly history (date=YYYY-WW for example) → on met la date du jour
    today = pd.Timestamp.today(tz="UTC").date()
    br_hist = br.copy()
    br_hist["date"] = str(today)
    hist_path = Path("sector_history.csv")
    if hist_path.exists():
        old = pd.read_csv(hist_path)
        out = pd.concat([old, br_hist], ignore_index=True)
    else:
        out = br_hist
    _save(out, "sector_history.csv")

if __name__ == "__main__":
    import time
    main()
