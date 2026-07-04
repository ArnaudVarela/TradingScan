import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root importable
# backtest_exits.py — TEST DÉCISIF : le filtre "fallen angel" + SORTIES ASYMÉTRIQUES est-il tradeable ?
# Simule des trades réels (entrée next-open, stop initial, trailing qui ratchete, max-hold, gestion des gaps),
# DÉ-CHEVAUCHÉS (pas de ré-entrée tant qu'un trade est ouvert) -> trades indépendants (anti pseudo-réplication).
# Compare plusieurs configs de sortie vs le buy-and-hold 20j, sur le sous-ensemble structurel.
import os
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_engine as E

ROOT = _pl.Path(__file__).resolve().parents[1]
SCRATCH = Path(os.getenv("SCRATCH", ROOT))
PKL = SCRATCH / "ohlc_thematic_6y.pkl"
COSTS = E.cost_model_from_env()

def sim(px, sig_date, stop_pct, trail_pct, max_hold):
    idx = px.index
    pos = int(idx.searchsorted(pd.Timestamp(sig_date), side="right")) - 1
    if pos < 0:
        return None
    ei = pos + 1
    if ei >= len(idx):
        return None
    o, h, l, c = px["open"].values, px["high"].values, px["low"].values, px["close"].values
    entry = float(o[ei])
    if not np.isfinite(entry) or entry <= 0:
        return None
    risk = entry * stop_pct
    stop = entry * (1.0 - stop_pct)
    peak = entry
    end_i = min(ei + max_hold, len(idx) - 1)
    ex_price = ex_i = reason = None
    for j in range(ei, end_i + 1):
        oj, hj, lj, cj = float(o[j]), float(h[j]), float(l[j]), float(c[j])
        if oj <= stop:                       # gap sous le stop -> sortie à l'open
            ex_price, reason, ex_i = oj, "gap", j; break
        if lj <= stop:                       # stop touché en séance
            ex_price, reason, ex_i = stop, "stop", j; break
        peak = max(peak, hj)
        stop = max(stop, peak * (1.0 - trail_pct))   # trailing qui ne recule jamais
        if j == end_i:
            ex_price, reason, ex_i = cj, "time", j
    if ex_price is None:
        return None
    buy = entry * (1 + COSTS.slip) * (1 + COSTS.fee)
    sell = ex_price * (1 - COSTS.slip) * (1 - COSTS.fee)
    return {"ret": (sell / buy - 1) * 100, "R": (ex_price - entry) / risk,
            "reason": reason, "days": ex_i - ei, "exit_date": idx[ex_i]}

def dedup_signals(sig, pm, sim_fn):
    """Trades indépendants : pour chaque ticker, pas de ré-entrée tant que le trade précédent est ouvert."""
    trades = []
    for t, grp in sig.groupby("ticker"):
        px = pm.get(t)
        if px is None or px.empty:
            continue
        last_exit = None
        for d in sorted(pd.to_datetime(grp["date"])):
            if last_exit is not None and d <= last_exit:
                continue
            tr = sim_fn(px, d)
            if tr:
                trades.append({**tr, "ticker": t, "date": d})
                last_exit = tr["exit_date"]
    return pd.DataFrame(trades)

def stats(tr, label):
    if len(tr) < 10:
        print(f"  {label:34} n={len(tr):3}  (échantillon trop faible)"); return
    r = tr["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / (abs(losses.sum()) + 1e-9)
    exp = r.mean()
    rexp = tr["R"].mean() if "R" in tr else np.nan
    wr = (r > 0).mean() * 100
    aw = wins.mean() if len(wins) else 0
    al = losses.mean() if len(losses) else 0
    print(f"  {label:34} n={len(tr):3}  win={wr:4.1f}%  exp={exp:+5.2f}%  "
          f"médiane={r.median():+5.2f}%  R-exp={rexp:+.2f}  PF={pf:.2f}  gain moy={aw:+5.1f}%  perte moy={al:+5.1f}%")

def buyhold(sig, pm, h=20):
    trades = []
    for t, grp in sig.groupby("ticker"):
        px = pm.get(t)
        if px is None or px.empty:
            continue
        last = None
        for d in sorted(pd.to_datetime(grp["date"])):
            if last is not None and d <= last:
                continue
            r = E.simulate_trade(px, d, h, entry_mode="next_open", costs=COSTS)
            if r and r["status"] == "closed":
                trades.append({"ret": r["ret_net_pct"], "R": np.nan, "ticker": t,
                               "exit_date": pd.Timestamp(r["exit_date"])})
                last = pd.Timestamp(r["exit_date"])
    return pd.DataFrame(trades)

def main():
    pm = pd.read_pickle(PKL)
    m = pd.read_csv(ROOT / "structure_matrix.csv")
    strict = m[m["fallen_angel"] == 1][["date", "ticker"]]
    relax = m[(m["dd_from_hi"] >= 0.50) & (m["base_range6"] <= 0.70)
              & (m["vol_growth"] >= 1.05) & (m["dist_base_hi6"] <= 0.15)][["date", "ticker"]]
    print(f"signaux : strict={len(strict)} obs, relax={len(relax)} obs\n")

    for name, sg in [("STRICT (dd>=60% base tendue vol+15% retest<12%)", strict),
                     ("RELAX  (dd>=50% base<=70% vol+5% retest<15%)", relax)]:
        print(f"===== {name} =====")
        stats(buyhold(sg, pm, 20).pipe(lambda d: d.assign(R=np.nan)), "BASELINE buy&hold 20j")
        for lab, sp, tp, mh in [("stop8 trail15 hold60", 0.08, 0.15, 60),
                                ("stop8 trail25 hold90", 0.08, 0.25, 90),
                                ("stop12 trail20 hold90", 0.12, 0.20, 90),
                                ("stop10 SANS trail hold40", 0.10, 1.0, 40)]:
            tr = dedup_signals(sg, pm, lambda px, d, sp=sp, tp=tp, mh=mh: sim(px, d, sp, tp, mh))
            stats(tr, f"EXITS {lab}")
        print()

if __name__ == "__main__":
    main()
