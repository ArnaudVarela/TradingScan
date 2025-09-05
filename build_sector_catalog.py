# build_sector_catalog.py
# Agrège les secteurs/industries depuis:
# - Wikipedia Russell 1000 (secteur/industry)
# - SPDR SPSM (Small Cap) XLSX (secteur/industry quand dispo)
# Puis merge sur l'univers (in_scope ou today), en conservant TOUT l'univers.
# Sorties: sector_catalog.csv, sector_history.csv, sector_breadth.csv
# (écrites en racine + dashboard/public)

import io
import os
from pathlib import Path
import re
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
    x = str(x).strip().upper()
    x = re.sub(r"\s+", "", x)
    # BRK.B (TV/Wiki) → BRK-B (Yahoo)
    x = x.replace(".", "-")
    return x

def _read_html_tables(url: str):
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    # StringIO pour éviter le FutureWarning
    return pd.read_html(io.StringIO(r.text))

def from_wikipedia_r1k() -> pd.DataFrame:
    """
    Retourne DataFrame colonnes: ticker, sector, industry (industry best-effort)
    """
    tables = _read_html_tables(WIKI_R1K)
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(("symbol" in c or "ticker" in c) for c in cols) and any("sector" in c for c in cols):
            best = t
            break
    if best is None:
        # fallback: prend toute table contenant une colonne symbole
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any(("symbol" in c or "ticker" in c) for c in cols):
                best = t
                break
    if best is None:
        print("[WARN] R1K table not found")
        return pd.DataFrame(columns=["ticker","sector","industry"])

    # map colonnes
    colmap = {}
    for c in best.columns:
        cl = str(c).lower()
        if "symbol" in cl or "ticker" in cl:
            colmap["ticker"] = c
        elif "sector" in cl and "sub" not in cl:
            colmap["sector"] = c
        elif ("sub" in cl and "industry" in cl) or ("gics" in cl and "industry" in cl):
            colmap.setdefault("industry", c)

    out_cols = [k for k in ["ticker","sector","industry"] if k in colmap]
    df = best.rename(columns={v:k for k,v in colmap.items()})[out_cols].copy()

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].map(_norm_ticker)
        df = df[df["ticker"] != ""]
        df = df[df["ticker"].str.match(r"^[A-Z0-9\-\.]+$")]

    for c in ["sector","industry"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
        else:
            df[c] = ""

    print(f"[CAT] R1K mapping: {len(df)}")
    return df[["ticker","sector","industry"]]

def from_spdr_spsm() -> pd.DataFrame:
    """
    Retourne DataFrame colonnes: ticker, sector, industry (si disponibles)
    """
    r = requests.get(SPDR_SPSM_XLSX, headers=UA, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    sh = None
    for s in xls.sheet_names:
        if "holding" in s.lower():
            sh = s
            break
    if sh is None:
        sh = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sh)

    # tente plusieurs noms
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
    out = out[out["ticker"] != ""]
    out = out.drop_duplicates(subset=["ticker"], keep="first")
    print(f"[CAT] SPSM mapping: {len(out)}")
    return out[["ticker","sector","industry"]]

def load_universe() -> pd.DataFrame:
    """
    Charge l'univers depuis universe_in_scope.csv (priorité) ou universe_today.csv.
    Normalise en colonnes: ticker, (sector optionnel)
    """
    cand = ["universe_in_scope.csv", "universe_today.csv"]
    for p in cand:
        if os.path.exists(p):
            u = pd.read_csv(p)
            print(f"[UNI] {p}: {len(u)} rows")
            # colonne symbole -> 'ticker'
            if "ticker" in u.columns:
                out = u.rename(columns={"ticker":"ticker_yf"})
            else:
                out = u
            if "ticker_yf" in out.columns:
                out["ticker"] = out["ticker_yf"].map(_norm_ticker)
            elif "Ticker" in out.columns:
                out["ticker"] = out["Ticker"].map(_norm_ticker)
            else:
                # tente première colonne
                out["ticker"] = out.iloc[:,0].map(_norm_ticker)

            # garde une éventuelle colonne sector existante
            if "sector" not in out.columns:
                out["sector"] = ""  # sera complété après merge

            out = out[["ticker","sector"] + [c for c in out.columns if c not in {"ticker","sector"}]]
            out = out[out["ticker"] != ""]
            out = out.drop_duplicates(subset=["ticker"], keep="first")
            return out
    raise SystemExit("❌ universe_in_scope.csv / universe_today.csv introuvable")

def main():
    uni = load_universe()  # ticker, sector (optionnel)

    r1k = from_wikipedia_r1k()     # ticker, sector, industry
    spsm = from_spdr_spsm()        # ticker, sector, industry

    # merge mapping (priorité R1K, puis SPSM pour compléter)
    cat = pd.concat([r1k, spsm], ignore_index=True)
    cat = cat.sort_values(["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    print(f"[CAT] merged map unique: {len(cat)}")

    # left-merge sur l'univers (conserver toutes les lignes de l'univers)
    merged = uni.merge(cat, on="ticker", how="left", suffixes=("", "_cat"))

    # Résoudre sector final: priorité au mapping cat si non vide, sinon univers
    def pick_sector(row):
        s_cat = str(row.get("sector_cat", "")).strip()
        s_uni = str(row.get("sector", "")).strip()
        return s_cat if s_cat else s_uni

    # Créer colonnes si absentes (évite KeyError)
    if "sector_cat" not in merged.columns:
        merged["sector_cat"] = ""
    if "industry" not in merged.columns:
        merged["industry"] = ""

    merged["sector"] = merged.apply(pick_sector, axis=1)
    merged["industry"] = merged["industry"].fillna("").astype(str).str.strip()

    # Nettoyage final
    merged["sector"] = merged["sector"].fillna("").astype(str).str.strip()
    merged.loc[merged["sector"] == "", "sector"] = "Unknown"
    merged.loc[merged["industry"] == "", "industry"] = "Unknown"

    # Sortie 1: catalog (ticker, sector, industry)
    catalog = merged[["ticker","sector","industry"]].drop_duplicates(subset=["ticker"], keep="first")
    _save(catalog, "sector_catalog.csv")

    # Sortie 2: history (snapshot daté simple)
    # On peut ajouter une date si besoin ; ici snapshot sans date (compat front)
    # Si tu veux datestamper, ajoute une colonne 'date' avec pd.Timestamp.utcnow().date()
    history = catalog.copy()
    _save(history, "sector_history.csv")

    # Sortie 3: breadth (compte ticker par secteur)
    breadth = (
        catalog.groupby("sector", dropna=False)
               .size()
               .reset_index(name="count")
               .sort_values("count", ascending=False)
    )
    _save(breadth, "sector_breadth.csv")

if __name__ == "__main__":
    main()
