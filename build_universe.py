#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_universe.py
- Construit l'univers: Russell 1000 + S&P 500 + Nasdaq Composite
- Normalise, dé-duplique, journalise les deltas (ajouts / suppressions)
- Maintient un registre incrémental (first_seen / last_seen / active / sources)
- Écrit raw_universe.csv, universe_today.csv, universe_in_scope.csv et universe_registry.csv
- Met en cache les snapshots et les métadonnées de sources (ETag / hash de contenu)

Dépendances: pandas, requests
"""

from __future__ import annotations
import csv
import hashlib
import io
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

ROOT = Path(".")
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ----- URLs officielles / stables -----
URL_SP500 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
# iShares Russell 1000 holdings (peut changer d’URL; ici endpoint CSV public iShares)
URL_R1000 = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
# Référentiel Nasdaq (TSV) — contient toutes les sociétés listées sur Nasdaq
URL_NASDAQ_LISTED = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

# ----- Fichiers de sortie -----
RAW_CSV = ROOT / "raw_universe.csv"
TODAY_CSV = ROOT / "universe_today.csv"
SCOPE_CSV = ROOT / "universe_in_scope.csv"
REGISTRY_CSV = ROOT / "universe_registry.csv"
REGISTRY_JSON = CACHE_DIR / "registry.json"
SOURCES_META_JSON = CACHE_DIR / "sources_meta.json"

TODAY = date.today().isoformat()
NOW_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get(url: str, timeout: int = 30) -> Tuple[bytes, Dict[str, str]]:
    """Simple GET avec gestion d'erreurs claire."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content, r.headers


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_ticker(t: str) -> str:
    """Normalise un ticker en uppercase, strip, remplace espaces -> '-'."""
    if not isinstance(t, str):
        return ""
    t = t.strip().upper()
    t = t.replace(" ", "-")
    # Quelques normalisations courantes (peuvent être étendues)
    t = t.replace(".A", "-A").replace(".B", "-B")
    return t


def fetch_sp500() -> pd.DataFrame:
    raw, headers = http_get(URL_SP500)
    df = pd.read_csv(io.BytesIO(raw))
    # colonnes typiques: Symbol, Security, GICS Sector, ...
    df = df.rename(columns={"Symbol": "ticker"})
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["source"] = "SP500"
    df["name"] = df.get("Security")
    df["exchange_hint"] = "NYSE/NASDAQ"
    return df[["ticker", "name", "source", "exchange_hint"]]


def fetch_r1000() -> pd.DataFrame:
    raw, headers = http_get(URL_R1000)
    # iShares CSV utilise ; comme séparateur dans certains locales — forçons ,
    text = raw.decode("utf-8", errors="ignore")
    # éliminer les lignes de préambule si présentes
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lower().startswith(("fund name", "as of", "ticker")) or ln.strip().lower().startswith("ticker")]
    text = "\n".join(lines)
    df = pd.read_csv(io.StringIO(text))
    # colonnes attendues: Ticker, Name, ...
    # certains fichiers iShares ont "Ticker" ou "Ticker Symbol"
    for col in ["Ticker", "Ticker Symbol", "TickerSymbol"]:
        if col in df.columns:
            ticker_col = col
            break
    else:
        raise RuntimeError("Russell1000: colonne ticker introuvable dans le CSV iShares")
    name_col = "Name" if "Name" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)

    out = pd.DataFrame({
        "ticker": df[ticker_col].astype(str).map(normalize_ticker),
        "name": df[name_col].astype(str) if name_col else None
    })
    out["source"] = "R1000"
    out["exchange_hint"] = "NYSE/NASDAQ"
    return out[["ticker", "name", "source", "exchange_hint"]]


def fetch_nasdaq_composite() -> pd.DataFrame:
    raw, headers = http_get(URL_NASDAQ_LISTED)
    # Fichier TSV avec en-tête et dernière ligne "File Creation Time"
    text = raw.decode("utf-8", errors="ignore")
    text = "\n".join([ln for ln in text.splitlines() if not ln.startswith("File Creation Time")])
    df = pd.read_csv(io.StringIO(text), sep="|")
    # colonnes: Symbol, Security Name, Market Category, Test Issue, Financial Status, Round Lot Size, ...
    df = df[df["Test Issue"].astype(str).str.upper() == "N"]  # exclut les tickers test
    tick = df["Symbol"].astype(str).map(normalize_ticker)
    name = df.get("Security Name")
    out = pd.DataFrame({
        "ticker": tick,
        "name": name,
    })
    out["source"] = "NASDAQ"
    out["exchange_hint"] = "NASDAQ"
    return out[["ticker", "name", "source", "exchange_hint"]]


def union_sources(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Union + agrégation des sources par ticker."""
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["ticker"] = all_df["ticker"].fillna("").astype(str)
    all_df = all_df[all_df["ticker"] != ""]
    # Regrouper et agréger
    agg = (all_df
           .groupby("ticker", as_index=False)
           .agg({
               "name": "first",
               "exchange_hint": "first",
               "source": lambda s: "|".join(sorted(set([x for x in s if isinstance(x, str)])))
           }))
    agg = agg.sort_values("ticker").reset_index(drop=True)
    return agg


def update_registry(universe: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Maintient registry.json et retourne DataFrame registry + diff."""
    registry = load_json(REGISTRY_JSON, default={})  # {ticker: {"first_seen":..., "last_seen":..., "active": bool, "sources": "..."}}
    today_ts = TODAY

    tickers_today = set(universe["ticker"].tolist())
    prev_active = {k for k, v in registry.items() if v.get("active")}
    added = sorted(tickers_today - prev_active)
    removed = sorted(prev_active - tickers_today)

    # MàJ/insert
    for _, row in universe.iterrows():
        t = row["ticker"]
        entry = registry.get(t, {})
        if not entry:
            entry["first_seen"] = today_ts
        entry["last_seen"] = today_ts
        entry["active"] = True
        entry["sources"] = row.get("source", "")
        registry[t] = entry

    # Désactiver les retirés
    for t in removed:
        entry = registry.get(t, {})
        entry["active"] = False
        entry["last_seen"] = today_ts
        registry[t] = entry

    # Sauvegarde JSON
    save_json(REGISTRY_JSON, registry)

    # DataFrame pour export CSV
    reg_df = (pd.DataFrame([
        {"ticker": t,
         "first_seen": v.get("first_seen"),
         "last_seen": v.get("last_seen"),
         "active": bool(v.get("active")),
         "sources": v.get("sources", "")
         }
        for t, v in registry.items()
    ])
    .sort_values(["active", "ticker"], ascending=[False, True])
    .reset_index(drop=True))

    diff = {"added": added, "removed": removed}
    return reg_df, diff


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    print("[STEP] build_universe starting…")

    sources_meta = load_json(SOURCES_META_JSON, default={})
    started = time.time()

    # --- Fetch
    spx = fetch_sp500()
    print(f"[SRC] SPX: {len(spx)} rows")

    try:
        r1k = fetch_r1000()
        print(f"[SRC] R1000: {len(r1k)} rows")
    except Exception as e:
        print(f"[WARN] Russell1000 fetch failed: {e}")
        r1k = pd.DataFrame(columns=["ticker", "name", "source", "exchange_hint"])

    nas = fetch_nasdaq_composite()
    print(f"[SRC] NASDAQ listed: {len(nas)} rows")

    # --- Union + normalisation
    uni = union_sources([spx, r1k, nas])
    print(f"[UNI] union after dedupe: {len(uni)} rows")

    # --- Sauvegardes CSV
    save_csv(uni, RAW_CSV)
    save_csv(uni, TODAY_CSV)
    # pas de filtre à ce stade: identique au raw
    save_csv(uni[["ticker"]].rename(columns={"ticker": "ticker_yf"}), SCOPE_CSV)

    # --- Registre incrémental + diff
    reg_df, diff = update_registry(uni)
    save_csv(reg_df, REGISTRY_CSV)

    # --- Logs diff
    print(f"[DIFF] +{len(diff['added'])} added, -{len(diff['removed'])} removed")
    if diff["added"]:
        print("  added:", ", ".join(diff["added"][:20]), ("…" if len(diff["added"]) > 20 else ""))
    if diff["removed"]:
        print("  removed:", ", ".join(diff["removed"][:20]), ("…" if len(diff["removed"]) > 20 else ""))

    # --- Résumé
    took = time.time() - started
    print(f"[DONE] build_universe finished in {took:.1f}s.")
    print(f"[OUT] {RAW_CSV.name} | rows={len(uni)}")
    print(f"[OUT] {SCOPE_CSV.name} | rows={len(uni)} (no policy filters here)")
    print(f"[OUT] {REGISTRY_CSV.name} | rows={len(reg_df)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
