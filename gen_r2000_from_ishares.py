# gen_r2000_from_ishares.py
# Génère russell2000.csv (liste complète) depuis le fichier officiel SSGA (IWM ETF)

import os
import pandas as pd
import requests

OUTFILE = "russell2000.csv"
PUBLIC_DIR = os.path.join("dashboard", "public")

# Lien direct vers les holdings de l’ETF IWM (SPDR Russell 2000 ETF)
SSGA_IWM_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-iwm.xlsx"

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def fetch_r2000_from_ssga():
    """Télécharge et parse l'Excel officiel de SSGA pour IWM (Russell 2000 ETF)."""
    r = requests.get(SSGA_IWM_XLSX, timeout=60)
    r.raise_for_status()
    local_path = "iwm_holdings.xlsx"
    with open(local_path, "wb") as f:
        f.write(r.content)

    # Lecture Excel (souvent 2-3 lignes d'entête, on cherche la colonne 'Ticker')
    xls = pd.ExcelFile(local_path)
    df = None
    for sheet in xls.sheet_names:
        try:
            tmp = pd.read_excel(local_path, sheet_name=sheet)
            # Cherche une colonne contenant 'Ticker' ou 'Symbol'
            cols = [str(c).lower() for c in tmp.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                df = tmp
                break
        except Exception:
            continue

    if df is None or df.empty:
        raise RuntimeError("Impossible de trouver une colonne Ticker dans le fichier IWM téléchargé.")

    # Normalisation tickers
    ticker_col = None
    for c in df.columns:
        if "ticker" in str(c).lower() or "symbol" in str(c).lower():
            ticker_col = c
            break

    tickers = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .tolist()
    )

    # Nettoyage : uniquement A-Z . -
    tickers = [t for t in tickers if pd.Series([t]).str.match(r"^[A-Z.\-]+$").iloc[0]]
    tickers = list(dict.fromkeys(tickers))  # dédoublonne

    return tickers

def save_csv(tickers, path):
    df = pd.DataFrame({"Ticker": tickers})
    _ensure_dir(path)
    df.to_csv(path, index=False)
    dst = os.path.join(PUBLIC_DIR, os.path.basename(path))
    _ensure_dir(dst)
    df.to_csv(dst, index=False)

def main():
    try:
        tickers = fetch_r2000_from_ssga()
        print(f"[OK] Russell 2000 récupéré: {len(tickers)} tickers")
        save_csv(tickers, OUTFILE)
    except Exception as e:
        print(f"[ERROR] Impossible de générer russell2000.csv: {e}")

if __name__ == "__main__":
    main()
