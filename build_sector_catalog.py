# build_sector_catalog.py — v1.3 (robuste aux variations Wikipédia)
# Objet :
#   - Charger l’univers courant (universe_in_scope.csv ou universe_today.csv)
#   - Construire un mapping ticker -> (sector, industry) à partir de sources publiques
#       * Wikipédia Russell 1000 (toutes tables, heuristiques de colonnes)
#       * (optionnel) un second provider si tu veux en ajouter un plus tard
#   - Fusionner proprement, sans planter si une source est vide
#   - Écrire : sector_catalog.csv, sector_history.csv, sector_breadth.csv
#
# Points clés :
#   - AUCUN KeyError si les entêtes changent : on cherche les colonnes par motifs.
#   - Si aucune source externe ne renvoie de mapping, on sort un catalogue
#     minimal (tickers + "Unknown") pour ne pas casser la chaîne.

import io
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}

WIKI_R1K = "https://en.wikipedia.org/wiki/Russell_1000_Index"


# ---------- IO helpers ----------
def _save(df: pd.DataFrame, name: str):
    df = df.copy() if df is not None else pd.DataFrame()
    df.to_csv(name, index=False)
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")


# ---------- Normalisation ----------
def _norm_ticker(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().upper().replace(".", "-")
    x = re.sub(r"[^A-Z0-9\-]", "", x)
    return x


def _clean_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


# ---------- Chargement univers ----------
def load_universe() -> pd.DataFrame:
    for p in ["universe_in_scope.csv", "universe_today.csv", "raw_universe.csv"]:
        if os.path.exists(p):
            u = pd.read_csv(p)
            print(f"[UNI] {p}: {len(u)} rows")
            # déterminer la colonne ticker
            tcol = None
            for cand in ["ticker_yf", "ticker", "symbol"]:
                if cand in u.columns:
                    tcol = cand
                    break
            if tcol is None:
                # fallback : 1ère colonne
                tcol = u.columns[0]
            out = pd.DataFrame({"ticker": u[tcol].map(_norm_ticker)})
            # sector s'il existe déjà (sinon vide)
            out["sector"] = u.get("sector", "")
            out["sector"] = out["sector"].fillna("").astype(str)
            out = out[out["ticker"] != ""].drop_duplicates("ticker")
            return out[["ticker", "sector"]]
    raise SystemExit("❌ universe_in_scope.csv / universe_today.csv / raw_universe.csv introuvable")


# ---------- Fetch & parse Wikipédia (robuste) ----------
def _read_html_tables(url: str) -> List[pd.DataFrame]:
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        return pd.read_html(io.StringIO(r.text))
    except Exception as e:
        print(f"[WARN] Wikipedia fetch failed: {e}")
        return []


def _pick_col(cols: List, *must_contain, exclude: Optional[List[str]] = None) -> Optional:
    """
    Trouve la 1ère colonne dont le libellé (lower) contient tous les tokens de must_contain,
    et ne contient aucun des tokens d'exclude.
    """
    exclude = exclude or []
    lowers = [(c, str(c).lower()) for c in cols]
    for c, cl in lowers:
        if all(tok in cl for tok in must_contain) and not any(bad in cl for bad in exclude):
            return c
    return None


def _extract_map_from_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Essaie d'extraire (ticker, sector, industry) d'une table quelconque.
    Renvoie DataFrame possédant *au minimum* la colonne 'ticker'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

    # Parfois pandas crée des colonnes MultiIndex -> aplatir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != ""]).strip() for tup in df.columns]

    cols = list(df.columns)
    # Chercher différentes variantes
    sym_col = (
        _pick_col(cols, "symbol") or
        _pick_col(cols, "ticker") or
        _pick_col(cols, "ticker", "symbol") or
        _pick_col(cols, "code")  # fallback large
    )
    if sym_col is None:
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

    sec_col = (
        _pick_col(cols, "gics", "sector") or
        _pick_col(cols, "sector", exclude=["sub"]) or
        _pick_col(cols, "sector")
    )
    ind_col = (
        _pick_col(cols, "gics", "sub", "industry") or
        _pick_col(cols, "sub-industry") or
        _pick_col(cols, "industry")
    )

    out = pd.DataFrame()
    out["ticker"] = df[sym_col].map(_norm_ticker)
    out["sector"] = df[sec_col].map(_clean_text) if sec_col in df.columns else ""
    out["industry"] = df[ind_col].map(_clean_text) if ind_col in df.columns else ""
    out = out[out["ticker"] != ""].drop_duplicates("ticker")

    return out[["ticker", "sector", "industry"]]


def from_wikipedia_r1k() -> pd.DataFrame:
    tables = _read_html_tables(WIKI_R1K)
    if not tables:
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

    collected = []
    for t in tables:
        try:
            part = _extract_map_from_table(t)
            if not part.empty:
                collected.append(part)
        except Exception:
            # table non conforme -> on ignore
            continue

    if not collected:
        print("[WARN] R1K: aucune table exploitable")
        return pd.DataFrame(columns=["ticker", "sector", "industry"])

    df = (
        pd.concat(collected, ignore_index=True)
        .sort_values("ticker")
        .drop_duplicates("ticker")
        .reset_index(drop=True)
    )
    print(f"[CAT] R1K mapping: {len(df)}")
    return df


# ---------- MAIN ----------
def main():
    # 1) Univers courant
    uni = load_universe()              # colonnes: ticker, sector (sector peut être vide)
    uni["ticker"] = uni["ticker"].map(_norm_ticker)

    # 2) Source externe : Wikipédia R1K
    r1k = from_wikipedia_r1k()         # colonnes: ticker, sector, industry (peut être vide)

    # 3) Fusion et normalisation
    if r1k.empty:
        # Aucun mapping externe -> on garde un catalogue minimal depuis l'univers
        catalog = uni.copy()
        if "industry" not in catalog.columns:
            catalog["industry"] = "Unknown"
        catalog["sector"] = catalog["sector"].replace("", "Unknown")
        catalog["industry"] = catalog["industry"].replace("", "Unknown")
        catalog = catalog[["ticker", "sector", "industry"]].drop_duplicates("ticker")
        print("[CAT] fallback: no external mapping available; using universe only.")
    else:
        merged = uni.merge(r1k, on="ticker", how="left", suffixes=("", "_r1k"))
        # priorité à R1K quand dispo, sinon univers
        merged["sector"] = merged.apply(
            lambda r: _clean_text(r["sector_r1k"]) if _clean_text(r.get("sector_r1k", "")) else _clean_text(r.get("sector", "")),
            axis=1
        )
        if "industry_r1k" not in merged.columns:
            merged["industry_r1k"] = ""
        merged["industry"] = merged["industry_r1k"].apply(_clean_text)
        merged["sector"] = merged["sector"].replace("", "Unknown")
        merged["industry"] = merged["industry"].replace("", "Unknown")
        catalog = merged[["ticker", "sector", "industry"]].drop_duplicates("ticker")

    # 4) Sorties
    _save(catalog, "sector_catalog.csv")

    hist = catalog.copy()
    hist.insert(0, "date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    _save(hist, "sector_history.csv")

    breadth = (
        catalog.groupby("sector", dropna=False).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    _save(breadth, "sector_breadth.csv")


if __name__ == "__main__":
    main()
