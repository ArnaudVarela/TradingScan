import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root importable
# validate_market_fallenangel.py — VALIDATION anti-survivorship/curation : rejoue le backtest
# fallen-angel + sorties asymétriques sur l'univers MARCHÉ <$5B (~2870 titres, tous secteurs),
# au lieu des 332 hard-tech curés. Détection de structure VECTORISÉE (rapide), fetch 6 ans caché.
import os
from pathlib import Path

import numpy as np
import pandas as pd

import data_fetch as DF
import backtest_exits as BX  # sim, dedup_signals, stats, buyhold

ROOT = _pl.Path(__file__).resolve().parents[1]
SCRATCH = Path(os.getenv("SCRATCH", ROOT))
PKL = SCRATCH / "market_ohlc_6y.pkl"
MONTHS = int(os.getenv("VM_MONTHS", "36"))     # fenêtre des signaux
MAX_TICKERS = int(os.getenv("VM_MAX", "0")) or None

def flags(px):
    c, h, l, v = px["close"], px["high"], px["low"], px["volume"]
    roll_hi3 = h.rolling(756, min_periods=252).max()
    dd = 1 - c / roll_hi3
    bh6 = h.rolling(126).max(); bl6 = l.rolling(126).min()
    brange = (bh6 - bl6) / c
    dbh = (bh6 - c) / bh6
    vg = v.rolling(63).mean() / v.shift(63).rolling(189).mean()
    strict = (dd >= 0.60) & (brange <= 0.60) & (vg >= 1.15) & (dbh <= 0.12)
    relax = (dd >= 0.50) & (brange <= 0.70) & (vg >= 1.05) & (dbh <= 0.15)
    return strict.fillna(False), relax.fillna(False)

def main():
    uni = pd.read_csv(ROOT / "market_universe.csv")
    tickers = uni["ticker"].astype(str).str.upper().tolist()
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    if PKL.exists():
        pm = pd.read_pickle(PKL); print(f"[VM] OHLC cache ({len(pm)} titres)")
    else:
        end = pd.Timestamp.today().normalize()
        pm = DF.prefetch_ohlc(tickers + ["SPY"], end - pd.DateOffset(years=6), end)
        pd.to_pickle(pm, PKL); print(f"[VM] OHLC fetché 6 ans ({len(pm)} titres)")

    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=MONTHS)
    srows, rrows = [], []
    n_ok = 0
    for t in tickers:
        px = pm.get(t)
        if px is None or px.empty or len(px) < 300:
            continue
        n_ok += 1
        st, rx = flags(px)
        for d in px.index[st.values & (px.index >= cutoff)]:
            srows.append({"date": d, "ticker": t})
        for d in px.index[rx.values & (px.index >= cutoff)]:
            rrows.append({"date": d, "ticker": t})
    strict = pd.DataFrame(srows); relax = pd.DataFrame(rrows)
    print(f"[VM] {n_ok} titres exploitables · signaux strict={len(strict)} · relax={len(relax)}\n")

    for name, sg in [("MARCHÉ <$5B — STRICT", strict), ("MARCHÉ <$5B — RELAX", relax)]:
        if sg.empty:
            print(f"===== {name} : aucun signal =====\n"); continue
        print(f"===== {name} ({sg['ticker'].nunique()} titres) =====")
        BX.stats(BX.buyhold(sg, pm, 20).assign(R=np.nan), "BASELINE buy&hold 20j")
        for lab, sp, tp, mh in [("stop8 trail15 hold60", 0.08, 0.15, 60),
                                ("stop8 trail25 hold90", 0.08, 0.25, 90),
                                ("stop12 trail20 hold90", 0.12, 0.20, 90),
                                ("stop10 SANS trail hold40", 0.10, 1.0, 40)]:
            tr = BX.dedup_signals(sg, pm, lambda px, d, sp=sp, tp=tp, mh=mh: BX.sim(px, d, sp, tp, mh))
            BX.stats(tr, f"EXITS {lab}")
        print()

if __name__ == "__main__":
    main()
