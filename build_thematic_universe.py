# build_thematic_universe.py — Construit l'univers thématique (actions US des thèmes hard-tech).
#
# Hybride : listes curées (themes.json) + validation contre yfinance (drop delistés/invalides).
# Multi-thème autorisé. Sortie : thematic_universe.csv (ticker, themes, n_themes, valid).
# Le mapping industrie GICS (broadening auto) viendra en couche 2.
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
THEMES = json.loads((ROOT / "themes.json").read_text(encoding="utf-8"))["themes"]
BATCH = 100

def collect() -> dict:
    """ticker -> set(themes)."""
    tmap = {}
    for key, obj in THEMES.items():
        for t in obj["tickers"]:
            t = str(t).strip().upper()
            if t:
                tmap.setdefault(t, set()).add(key)
    return tmap

def validate(tickers: list) -> set:
    """Garde les tickers qui renvoient des données yfinance récentes."""
    valid = set()
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        try:
            df = yf.download(chunk, period="1mo", interval="1d", progress=False,
                             group_by="ticker", threads=True)
        except Exception as e:
            print(f"[WARN] batch {i//BATCH+1}: {e}")
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for t in chunk:
                try:
                    if not df[t]["Close"].dropna().empty:
                        valid.add(t)
                except Exception:
                    pass
        elif len(chunk) == 1:
            if "Close" in df.columns and not df["Close"].dropna().empty:
                valid.add(chunk[0])
        time.sleep(1.0)
    return valid

def main():
    tmap = collect()
    tickers = sorted(tmap)
    print(f"[THEMES] {len(tickers)} tickers curés uniques sur {len(THEMES)} thèmes")
    valid = validate(tickers)

    rows = [{"ticker": t, "themes": "|".join(sorted(tmap[t])), "n_themes": len(tmap[t]),
             "valid": t in valid} for t in tickers]
    out = pd.DataFrame(rows).sort_values(["valid", "n_themes", "ticker"], ascending=[False, False, True])
    out.to_csv(ROOT / "thematic_universe.csv", index=False)
    # miroir dashboard
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        out.to_csv(pub / "thematic_universe.csv", index=False)

    dropped = sorted([t for t in tickers if t not in valid])
    print(f"[OK] valides: {len(valid)} | droppés ({len(dropped)}): {dropped}")
    print("[COUNTS] par thème (valides uniquement) :")
    for key, obj in THEMES.items():
        n = sum(1 for t in obj["tickers"] if str(t).strip().upper() in valid)
        print(f"   {key:<22} {n:>3}   ({obj['label']})")
    multi = out[(out['valid']) & (out['n_themes'] >= 2)]
    print(f"[INFO] {len(multi)} tickers multi-thèmes (ex: {list(multi['ticker'].head(8))})")

if __name__ == "__main__":
    main()
