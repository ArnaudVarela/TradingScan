# gen_r2000_from_ishares.py
# -----------------------------------------------------------------------------
# Récupère les constituants des indices Russell 2000 (IWM) et Russell 1000 (IWB)
# via les CSV "holdings" d'iShares, écrit à la racine ET (en copie) dans
# dashboard/public/, avec logs et codes d'erreur explicites.
# -----------------------------------------------------------------------------

import os
import io
import sys
import csv
import time
import shutil
import typing as t
import pandas as pd
import requests

ROOT = os.getcwd()
PUBLIC_DIR = os.path.join(ROOT, "dashboard", "public")

IWM_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWM_holdings&dataType=fund"
)
IWB_URL = (
    "https://www.ishares.com/us/products/239724/ishares-russell-1000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (TradingScan/CI)"}


def _dl_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    # certains CSV iShares ont un préambule; pandas gère bien si on donne un buffer
    buf = io.StringIO(r.text)
    try:
        df = pd.read_csv(buf)
    except Exception:
        # fallback: essayer sep="," explicite
        buf.seek(0)
        df = pd.read_csv(buf, sep=",")
    return df


def _extract_tickers(df: pd.DataFrame) -> pd.DataFrame:
    # colonnes possibles selon le fonds: "Ticker", "Ticker Symbol", "Underlying Ticker"
    candidates = ["Ticker", "Ticker Symbol", "Underlying Ticker", "Symbol"]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise RuntimeError(f"Aucune colonne ticker parmi {candidates}. Colonnes: {list(df.columns)}")
    tickers = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": ""})
    )
    tickers = tickers[tickers != ""].drop_duplicates().sort_values().reset_index(drop=True)
    out = pd.DataFrame({"Ticker": tickers})
    return out


def _write_root_and_public(df: pd.DataFrame, name: str):
    # racine
    root_path = os.path.join(ROOT, name)
    df.to_csv(root_path, index=False)
    print(f"[OK] wrote {name} at repo root: {len(df)} rows")

    # copie vers public (utile si ton front lit encore dashboard/public/*)
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    pub_path = os.path.join(PUBLIC_DIR, name)
    shutil.copy2(root_path, pub_path)
    print(f"[OK] copied {name} -> dashboard/public/{name}")


def main():
    try:
        print("[INFO] Downloading IWM holdings (Russell 2000)…")
        df_iwm = _dl_csv(IWM_URL)
        r2k = _extract_tickers(df_iwm)
        _write_root_and_public(r2k, "russell2000.csv")
    except Exception as e:
        print(f"[ERROR] Russell 2000 fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Russell 1000 (optionnel; si ça échoue, on log et continue)
    try:
        print("[INFO] Downloading IWB holdings (Russell 1000)…")
        df_iwb = _dl_csv(IWB_URL)
        r1k = _extract_tickers(df_iwb)
        _write_root_and_public(r1k, "russell1000.csv")
    except Exception as e:
        print(f"[WARN] Russell 1000 fetch failed (non bloquant): {e}", file=sys.stderr)

    print("[DONE] Univers updated.")


if __name__ == "__main__":
    main()
