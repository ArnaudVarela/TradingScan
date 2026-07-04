import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root importable
# backtest_entry.py — Teste les RAFFINEMENTS D'ENTRÉE (#7 confirmation, #6 volume-pop) sur la structure fallen-angel.
# Mêmes sorties pour toutes les variantes (isole l'effet ENTRÉE). Cache marché, dé-chevauché, slippage réaliste.
import os
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_engine as E

ROOT = _pl.Path(__file__).resolve().parents[1]
SCRATCH = Path(os.getenv("SCRATCH", ROOT))
PKL = SCRATCH / "market_ohlc_6y.pkl"
COSTS = E.cost_model_from_env()          # BT_SLIPPAGE_BPS=50 via env
WAIT = int(os.getenv("BE_WAIT", "30"))   # fenêtre pour que le déclencheur se produise
VOLK = float(os.getenv("BE_VOLK", "2.0"))
STOP, TRAIL, HOLD = 0.08, 0.25, 90       # sorties fixes (meilleur PF de la validation)
MONTHS = int(os.getenv("BE_MONTHS", "36"))

def _arrays(px):
    c, h, l, v = px["close"], px["high"], px["low"], px["volume"]
    roll_hi3 = h.rolling(756, min_periods=252).max()
    dd = 1 - c / roll_hi3
    bh6 = h.rolling(126).max(); bl6 = l.rolling(126).min()
    brange = (bh6 - bl6) / c; dbh = (bh6 - c) / bh6
    vg = v.rolling(63).mean() / v.shift(63).rolling(189).mean()
    relax = ((dd >= 0.50) & (brange <= 0.70) & (vg >= 1.05) & (dbh <= 0.15)).fillna(False).values
    return (relax, bh6.values, v.rolling(20).mean().values,
            px["open"].values, h.values, l.values, c.values, v.values, px.index)

def _entry_i(mode, o, h, l, c, v, wi, R, avgv):
    if mode == "structure":
        return wi + 1 if wi + 1 < len(c) else None
    lim = min(wi + WAIT, len(c) - 1)
    for j in range(wi + 1, lim + 1):
        if mode == "confirm" and np.isfinite(c[j]) and np.isfinite(R) and c[j] > R:
            return j + 1 if j + 1 < len(c) else None
        if mode == "volpop" and np.isfinite(avgv[j]) and avgv[j] > 0 and v[j] >= VOLK * avgv[j]:
            return j + 1 if j + 1 < len(c) else None
    return None

def _exit(o, h, l, c, ei):
    if ei is None or ei >= len(c):
        return None
    entry = float(o[ei])
    if not np.isfinite(entry) or entry <= 0:
        return None
    risk = entry * STOP; stop = entry * (1 - STOP); peak = entry
    end = min(ei + HOLD, len(c) - 1)
    exp = exi = None
    for j in range(ei, end + 1):
        oj, hj, lj, cj = float(o[j]), float(h[j]), float(l[j]), float(c[j])
        if oj <= stop:
            exp, exi = oj, j; break
        if lj <= stop:
            exp, exi = stop, j; break
        peak = max(peak, hj); stop = max(stop, peak * (1 - TRAIL))
        if j == end:
            exp, exi = cj, j
    if exp is None:
        return None
    buy = entry * (1 + COSTS.slip) * (1 + COSTS.fee); sell = exp * (1 - COSTS.slip) * (1 - COSTS.fee)
    return {"ret": (sell / buy - 1) * 100, "R": (exp - entry) / risk, "exit_i": exi}

def run(pm, mode):
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=MONTHS)
    trades, watches = [], 0
    for t, px in pm.items():
        if t == "SPY" or px is None or len(px) < 300:
            continue
        relax, bh6, avgv, o, h, l, c, v, idx = _arrays(px)
        wis = [i for i in range(len(idx)) if relax[i] and idx[i] >= cutoff]
        watches += len(wis)
        last_exit = -1
        for wi in wis:
            if wi <= last_exit:
                continue
            ei = _entry_i(mode, o, h, l, c, v, wi, bh6[wi], avgv)
            if ei is None:
                continue
            tr = _exit(o, h, l, c, ei)
            if tr:
                trades.append(tr["ret"]); last_exit = tr["exit_i"]
    r = pd.Series(trades)
    return r, watches

def stats(r, watches, label):
    if len(r) < 10:
        print(f"  {label:12} n={len(r):4}  (trop faible)"); return
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / (abs(losses.sum()) + 1e-9)
    print(f"  {label:12} n={len(r):4}  déclench={len(r)/watches*100:4.0f}%  win={ (r>0).mean()*100:4.1f}%  "
          f"exp={r.mean():+5.2f}%  médiane={r.median():+6.2f}%  PF={pf:.2f}  "
          f"gain moy={wins.mean():+5.1f}%  perte moy={losses.mean():+5.1f}%")

def main():
    pm = pd.read_pickle(PKL)
    print(f"Sorties fixes : stop {STOP:.0%} / trail {TRAIL:.0%} / hold {HOLD}j · slippage {COSTS.slippage_bps:.0f}bps · "
          f"attente déclencheur {WAIT}j\n")
    for mode, lab in [("structure", "STRUCTURE"), ("confirm", "CONFIRM #7"), ("volpop", "VOL-POP #6")]:
        r, w = run(pm, mode)
        stats(r, w, lab)

if __name__ == "__main__":
    main()
