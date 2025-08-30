# gen_r2000_from_ishares.py
# Génère russell2000.csv en scrapant les tickers du SPDR Russell 2000 ETF (IWM)
import pandas as pd
import requests
import re

URL_SPDR_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"
OUTPUT = "russell2000.csv"

def looks_like_ticker(val: str) -> bool:
    """Retourne True si val ressemble à un ticker US (AAPL, MSFT, BRK.B, etc.)"""
    if not isinstance(val, str):
        return False
    v = val.strip().upper()
    # Exclut les codes trop longs (ISIN/SEDOL) ou numériques purs
    if len(v) > 6: 
        return False
    if re.match(r"^[A-Z0-9.\-]+$", v):
        return True
    return False

def main():
    print(f"[SPDR] GET {URL_SPDR_XLSX}")
    r = requests.get(URL_SPDR_XLSX, timeout=60)
    r.raise_for_status()

    # Lecture du fichier Excel
    xls = pd.ExcelFile(r.content)
    print(f"  -> feuilles dispo: {xls.sheet_names}")

    best_col = None
    best_vals = []

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            print(f"  [skip] {sheet} ({e})")
            continue

        print(f"  [sheet] {sheet}: {df.shape[0]} rows, {df.shape[1]} cols")

        for col in df.columns:
            series = df[col].dropna().astype(str).str.strip()
            vals = [v for v in series if looks_like_ticker(v)]
            if len(vals) > len(best_vals):
                best_vals = vals
                best_col = col

    if not best_vals or len(set(best_vals)) < 1000:
        raise RuntimeError("❌ Impossible d’extraire suffisamment de tickers (structure modifiée).")

    # Nettoyage final
    uniq = sorted(set(v.upper() for v in best_vals if looks_like_ticker(v)))
    df_out = pd.DataFrame({"Ticker": uniq})
    df_out.to_csv(OUTPUT, index=False)
    print(f"[OK] {len(df_out)} tickers écrits dans {OUTPUT} (colonne '{best_col}')")

if __name__ == "__main__":
    main()
