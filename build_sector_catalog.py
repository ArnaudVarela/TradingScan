# build_sector_catalog.py
# -----------------------------------------------------------------------------
# Agrège les secteurs/industries depuis:
# - Wikipedia Russell 1000 (R1K)
# - SPDR SPSM (Small Cap) XLSX
# Sorties (racine + copie dashboard/public/):
#   - sector_catalog.csv   (ticker, sector, industry)
#   - sector_breadth.csv   (generated_at_utc, sector, count)
#   - sector_history.csv   (date, sector, count)  -> append du snapshot du jour
# -----------------------------------------------------------------------------

import io
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

WIKI_R1K = "https://en.wikipedia.org/wiki/Russell_1000_Index"
SPDR_SPSM_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"

PUBLIC_DIR = os.path.join("dashboard", "public")
CAT_CSV = "sector_catalog.csv"
BREADTH_CSV = "sector_breadth.csv"
HIST_CSV = "sector_history.csv"

UA = {"User-Agent": "Mozilla/5.0 (TradingScan/CI)"}


# ----------------------------- Utils I/O --------------------------------------
def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_root_and_public(df: pd.DataFrame, name: str, min_headers=None):
    """Écrit <name> à la racine + copie vers dashboard/public/ (headers-only si vide)."""
    # headers-only si demandé et df vide
    if (df is None or df.empty) and min_headers:
        df = pd.DataFrame(columns=min_headers)

    # root
    df.to_csv(name, index=False)
    print(f"[OK] root  {name}: {len(df)} lignes")

    # public
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    dst = os.path.join(PUBLIC_DIR, name)
    df.to_csv(dst, index=False)
    print(f"[OK] public {name}: {len(df)} lignes (copied)")


def _append_history(snapshot_long: pd.DataFrame, hist_path: str):
    """
    Ajoute au fichier d'historique (date, sector, count).
    Si inexistant, le crée.
    """
    cols = ["date", "sector", "count"]
    snapshot_long = snapshot_long.reindex(columns=cols)
    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
        except Exception:
            hist = pd.DataFrame(columns=cols)
        hist = pd.concat([hist, snapshot_long], ignore_index=True)
    else:
        hist = snapshot_long.copy()
    hist.to_csv(hist_path, index=False)
    # copie public
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    hist.to_csv(os.path.join(PUBLIC_DIR, os.path.basename(hist_path)), index=False)
    print(f"[OK] root/public {os.path.basename(hist_path)} mis à jour: +{len(snapshot_long)} lignes")


# ----------------------------- HTTP session -----------------------------------
def mk_sess() -> requests.Session:
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.headers.update(UA)
    return s


# ----------------------------- Normalisation ----------------------------------
def norm_ticker(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().upper()
    x = re.sub(r"\s+", "", x)
    # (tu peux ajouter ici des normalisations type BRK.B -> BRK.B inchangé)
    return x


# ----------------------------- Scrapers ---------------------------------------
def from_wikipedia_r1k() -> pd.DataFrame:
    sess = mk_sess()
    r = sess.get(WIKI_R1K, timeout=45)
    r.raise_for_status()
    # FutureWarning: read_html avec html literal → acceptable ici
    tables = pd.read_html(r.text)

    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c or "ticker" in c for c in cols) and any("sector" in c for c in cols):
            best = t
            break
    if best is None:
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("symbol" in c or "ticker" in c for c in cols):
                best = t
                break

    if best is None:
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

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

    take = [k for k in ["ticker", "sector", "industry"] if k in colmap]
    if not take or "ticker" not in take:
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

    df = best.rename(columns={v: k for k, v in colmap.items()})[take]
    df["ticker"] = df["ticker"].map(norm_ticker)
    df["sector"] = df.get("sector", "").fillna("").astype(str).str.strip()
    df["industry"] = df.get("industry", "").fillna("").astype(str).str.strip()
    df = df.dropna(subset=["ticker"])
    df = df[df["ticker"].str.match(r"^[A-Z.\-]+$")]
    return df[["ticker", "sector", "industry"]]


def from_spdr_spsm() -> pd.DataFrame:
    sess = mk_sess()
    r = sess.get(SPDR_SPSM_XLSX, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    # feuille 'holdings' généralement
    name = [s for s in xls.sheet_names if "holding" in s.lower()]
    sh = name[0] if name else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sh)

    cols = {str(c).lower(): c for c in df.columns}
    tick = cols.get("ticker") or cols.get("ticker symbol:") or cols.get("symbol") or cols.get("isin")
    sec = cols.get("sector") or cols.get("gics sector") or cols.get("economic sector")
    sub = cols.get("industry") or cols.get("sub-industry") or cols.get("gics sub-industry")

    out = pd.DataFrame(columns=["ticker", "sector", "industry"])
    if not tick:
        return out

    out["ticker"] = df[tick].map(norm_ticker)
    out["sector"] = df[sec].astype(str).str.strip() if sec in df.columns else ""
    out["industry"] = df[sub].astype(str).str.strip() if sub in df.columns else ""
    out = out.dropna(subset=["ticker"])
    out = out[out["ticker"].str.match(r"^[A-Z.\-]+$")]
    return out


# ----------------------------- Main -------------------------------------------
def main():
    try:
        r1k = from_wikipedia_r1k()
        print(f"[R1K] {len(r1k)} lignes")
    except Exception as e:
        print(f"[WARN] Wikipedia R1K fetch failed: {e}")
        r1k = pd.DataFrame(columns=["ticker", "sector", "industry"])

    try:
        spsm = from_spdr_spsm()
        print(f"[SPSM] {len(spsm)} lignes")
    except Exception as e:
        print(f"[WARN] SPDR SPSM fetch failed: {e}")
        spsm = pd.DataFrame(columns=["ticker", "sector", "industry"])

    cat = pd.concat([r1k, spsm], ignore_index=True)
    if cat.empty:
        print("[ERROR] Aucune donnée R1K/SPSM récupérée.")
        # on écrit quand même des fichiers vides (headers) pour ne pas casser le front
        _write_root_and_public(pd.DataFrame(columns=["ticker", "sector", "industry"]), CAT_CSV,
                               min_headers=["ticker", "sector", "industry"])
        _write_root_and_public(pd.DataFrame(columns=["generated_at_utc", "sector", "count"]), BREADTH_CSV,
                               min_headers=["generated_at_utc", "sector", "count"])
        _write_root_and_public(pd.DataFrame(columns=["date", "sector", "count"]), HIST_CSV,
                               min_headers=["date", "sector", "count"])
        sys.exit(1)

    cat = (
        cat.sort_values(["ticker", "sector"])
           .drop_duplicates(subset=["ticker"], keep="first")
           .reset_index(drop=True)
    )
    # clean vides
    cat["sector"] = cat["sector"].replace({"nan": ""})
    cat["industry"] = cat["industry"].replace({"nan": ""})

    # 1) catalog
    _write_root_and_public(cat[["ticker", "sector", "industry"]], CAT_CSV)

    # 2) breadth (snapshot du jour)
    breadth = (
        cat.groupby("sector", dropna=False)
           .size()
           .reset_index(name="count")
           .sort_values("sector", na_position="last")
    )
    breadth.insert(0, "generated_at_utc", _utc_stamp())
    _write_root_and_public(breadth, BREADTH_CSV)

    # 3) history (append)
    snap_long = breadth.rename(columns={"generated_at_utc": "date"})[["date", "sector", "count"]].copy()
    _append_history(snap_long, HIST_CSV)

    print(f"[OK] sector_catalog.csv écrit: {len(cat)} lignes "
          f"(R1K={len(r1k)}, SPSM={len(spsm)})")


if __name__ == "__main__":
    main()
