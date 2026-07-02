# scan_thematic.py — Scanne l'univers thématique et sort le classement /100 des setups de pré-explosion.
# Sortie : thematic_setups.csv (+ miroir dashboard), trié par score décroissant.
import os
from pathlib import Path

import pandas as pd

import data_fetch as DF   # prefetch_ohlc
import pattern_score as P

ROOT = Path(__file__).parent
LOOKBACK_YEARS = float(os.getenv("SCAN_YEARS", "2"))
MARKET = os.getenv("MARKET_TICKER", "SPY")

def main():
    uni = pd.read_csv(ROOT / "thematic_universe.csv")
    if "valid" in uni.columns:
        uni = uni[uni["valid"].astype(str).str.lower().isin(["true", "1"])]
    tickers = uni["ticker"].astype(str).str.upper().tolist()
    theme_map = dict(zip(uni["ticker"].astype(str).str.upper(), uni["themes"].astype(str)))
    print(f"[SCAN] {len(tickers)} titres thématiques")

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS + 1)
    pm = DF.prefetch_ohlc(tickers + [MARKET], start, end)
    spy = pm.get(MARKET)

    rows = []
    for t in tickers:
        df = pm.get(t)
        if df is None or df.empty:
            continue
        r = P.preexplosion_score(df, spy)
        if r is None:
            continue
        m = r["metrics"]
        rows.append({
            "ticker": t, "themes": theme_map.get(t, ""),
            "score": r["score"], "setup": r["label"],
            "price": m["price"], "rsi": m["rsi"], "macd_hist": m["macd_hist"],
            "dist_to_high_pct": m["dist_to_high_pct"], "base_depth_pct": m["base_depth_pct"],
            "bbwidth_pctile": m["bbwidth_pct"], "vol_dryup": m["vol_dryup"], "overext": m["overext"],
            **{f"c_{k}": v for k, v in r["components"].items()},
        })
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"[MCAP] récupération market cap pour {len(out)} titres...")
    mcaps = DF.fetch_mcaps(out["ticker"].tolist())
    out["mcap_usd"] = out["ticker"].map(mcaps)
    out.to_csv(ROOT / "thematic_setups.csv", index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        out.to_csv(pub / "thematic_setups.csv", index=False)

    print(f"[OK] {len(out)} setups scorés -> thematic_setups.csv")
    cols = ["ticker", "themes", "score", "setup", "price", "mcap_usd", "rsi", "dist_to_high_pct", "base_depth_pct", "vol_dryup"]
    print("\n=== TOP 25 SETUPS PRÉ-EXPLOSION (aujourd'hui) ===")
    print(out[cols].head(25).to_string(index=False))
    print("\n=== Répartition des scores ===")
    print(f"  >=70: {(out['score']>=70).sum()}  |  60-70: {((out['score']>=60)&(out['score']<70)).sum()}  "
          f"|  50-60: {((out['score']>=50)&(out['score']<60)).sum()}  |  <50: {(out['score']<50).sum()}")

if __name__ == "__main__":
    main()
