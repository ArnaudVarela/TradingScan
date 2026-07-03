# sector_perf.py — Rotation sectorielle GICS (via ETFs SPDR, yfinance, gratuit).
# Perf multi-horizons + RRG-lite (quadrants Leading/Weakening/Lagging/Improving) pour anticiper les rotations.
from pathlib import Path

import numpy as np
import pandas as pd

import data_fetch as DF

ROOT = Path(__file__).parent
SECTORS = {
    "XLK": "Information Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLE": "Energy", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}
BENCH = "SPY"

def _pct(s, n):
    return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if len(s) > n else np.nan

def _quad(ratio, mom):
    if ratio != ratio or mom != mom:
        return ""
    strong, accel = ratio >= 100, mom >= 0
    return ("Leading" if strong and accel else "Weakening" if strong else "Improving" if accel else "Lagging")

def main():
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=9)
    pm = DF.prefetch_ohlc(list(SECTORS) + [BENCH], start, end)
    spy = pm.get(BENCH)
    if spy is None or spy.empty:
        print("[SECTOR] pas de SPY"); return
    spyc = spy["close"]

    rows = []
    for etf, name in SECTORS.items():
        px = pm.get(etf)
        if px is None or px.empty:
            continue
        c = px["close"]
        rs = c / spyc.reindex(c.index).ffill()
        rs_norm = rs / rs.rolling(63).mean()                      # RS relatif à sa moyenne (RS-Ratio ~ RRG)
        rr = float(rs_norm.iloc[-1] * 100) if rs_norm.notna().iloc[-1] else np.nan
        rm = float((rs_norm.iloc[-1] / rs_norm.iloc[-11] - 1) * 100) if len(rs_norm) > 11 and rs_norm.notna().iloc[-11] else np.nan
        rows.append({
            "etf": etf, "sector": name,
            "chg_1d": round(_pct(c, 1), 2), "chg_5d": round(_pct(c, 5), 2),
            "chg_1m": round(_pct(c, 21), 2), "chg_3m": round(_pct(c, 63), 2),
            "rs_ratio": round(rr, 1) if rr == rr else "",
            "rs_mom": round(rm, 2) if rm == rm else "",
            "quadrant": _quad(rr, rm),
        })
    # Filet anti rate-limit : un ETF refusé par yfinance ne doit pas FAIRE DISPARAÎTRE le secteur
    # (sinon dashboard sans rotation ni confluence pour ce secteur). On reporte sa dernière valeur connue.
    got = {r["sector"] for r in rows}
    missing = [(etf, name) for etf, name in SECTORS.items() if name not in got]
    if missing and (ROOT / "sector_perf.csv").exists():
        try:
            prev = pd.read_csv(ROOT / "sector_perf.csv", keep_default_na=False)
            for etf, name in missing:
                old = prev[prev["sector"].astype(str) == name]
                if not old.empty:
                    rows.append(old.iloc[0].to_dict())
                    print(f"[SECTOR] {name} ({etf}) indisponible (yfinance) -> dernière valeur conservée")
        except Exception as e:
            print(f"[SECTOR] report précédent KO ({e})")
    if not rows:
        print("[SECTOR] aucune donnée exploitable -> sector_perf.csv inchangé"); return
    df = pd.DataFrame(rows).sort_values("chg_1m", ascending=False, na_position="last")
    df.to_csv(ROOT / "sector_perf.csv", index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        df.to_csv(pub / "sector_perf.csv", index=False)
    print(df.to_string(index=False))
    lead = df[df["quadrant"] == "Improving"]["sector"].tolist()
    print(f"\n[SECTOR] En rotation entrante (Improving) : {lead or '—'}")

if __name__ == "__main__":
    main()
