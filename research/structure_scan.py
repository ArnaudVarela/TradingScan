import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root importable
# structure_scan.py — Teste (1) l'horizon court 5j et (2) la STRUCTURE "fallen angel / accumulation" :
# titre défoncé depuis un pic pluriannuel + longue base tendue + volume en croissance + re-test de résistance.
# Walk-forward point-in-time, 6 ans d'historique, rendements forward à 5j ET 20j. OHLC caché (pickle).
import os
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_engine as E
import data_fetch as DF
import pattern_score as P

ROOT = _pl.Path(__file__).resolve().parents[1]
SCRATCH = Path(os.getenv("SCRATCH", ROOT))
CACHE = SCRATCH / "ohlc_thematic_6y.pkl"
MONTHS = int(os.getenv("SS_MONTHS", "36"))     # dates de test
STEP = int(os.getenv("SS_STEP", "5"))
BIG20, BIG5 = 20.0, 10.0
COSTS = E.cost_model_from_env()

def _feats(sl):
    c, h, l, v = sl["close"], sl["high"], sl["low"], sl["volume"]
    n = len(c); last = float(c.iloc[-1])
    def r(k):
        return float(last / c.iloc[-1 - k] - 1.0) if n > k else np.nan
    win_hi = float(h.max()); win_lo = float(l.min())
    base_hi6 = float(h.iloc[-126:].max()); base_lo6 = float(l.iloc[-126:].min())
    vol63 = float(v.iloc[-63:].mean())
    vol_prior = float(v.iloc[-252:-63].mean()) if n > 252 else np.nan
    vol60 = float(v.iloc[-60:].mean()); vol20 = float(v.iloc[-20:].mean())
    atr = P._atr(sl, 14); bbw = P._bb_width(c, 20)
    return {
        # --- structure "fallen angel" ---
        "dd_from_hi": 1.0 - last / (win_hi + 1e-9),               # défoncé depuis le pic de la fenêtre
        "off_low": last / (win_lo + 1e-9) - 1.0,                  # remonté du plus-bas
        "base_range6": (base_hi6 - base_lo6) / (last + 1e-9),     # largeur base 6M (bas = tendue)
        "dist_base_hi6": (base_hi6 - last) / (base_hi6 + 1e-9),   # distance au re-test de résistance (bas = proche)
        "vol_growth": vol63 / (vol_prior + 1e-9) if vol_prior == vol_prior else np.nan,  # volume en croissance
        "n_years": n / 252.0,
        # --- génériques (rappel) ---
        "ret_1m": r(21), "ret_3m": r(64), "ret_6m": r(126),
        "rvol10": float(v.iloc[-10:].mean()) / (vol60 + 1e-9),
        "atr_expand": float(atr.iloc[-1]) / (float(atr.iloc[-60:].mean()) + 1e-9),
        "bbw_pct": P._pct_rank(bbw, 126),
        "dryup": vol20 / (vol60 + 1e-9),
        "rsi": float(P._rsi(c).iloc[-1]),
        "price": last,
    }

def _fallen_angel(f):
    """Structure exacte de l'hypothèse : défoncé + base tendue + volume en hausse + proche de la résistance."""
    try:
        return int(f["dd_from_hi"] >= 0.60 and f["base_range6"] <= 0.60
                   and f["vol_growth"] >= 1.15 and f["dist_base_hi6"] <= 0.12)
    except Exception:
        return 0

def build():
    uni = pd.read_csv(ROOT / "thematic_universe.csv")
    if "valid" in uni.columns:
        uni = uni[uni["valid"].astype(str).str.lower().isin(["true", "1"])]
    tickers = uni["ticker"].astype(str).str.upper().tolist()
    if CACHE.exists():
        pm = pd.read_pickle(CACHE); print(f"[SS] OHLC cache ({len(pm)} titres)")
    else:
        end = pd.Timestamp.today().normalize()
        pm = DF.prefetch_ohlc(tickers + ["SPY"], end - pd.DateOffset(years=6), end)
        pd.to_pickle(pm, CACHE); print(f"[SS] OHLC fetché 6 ans ({len(pm)} titres)")
    spy = pm.get("SPY"); cal = spy.index
    dates = list(cal[cal >= (cal[-1] - pd.DateOffset(months=MONTHS))][::STEP])
    print(f"[SS] {len(dates)} dates x {len(tickers)} titres · forward 5j & 20j")
    rows = []
    for i, t in enumerate(tickers):
        px = pm.get(t)
        if px is None or px.empty:
            continue
        for d in dates:
            sl = px.loc[px.index <= d]
            if len(sl) < 260:
                continue
            t20 = E.simulate_trade(px, d, 20, entry_mode="next_open", costs=COSTS)
            t5 = E.simulate_trade(px, d, 5, entry_mode="next_open", costs=COSTS)
            if t20 is None or t20["status"] != "closed" or t5 is None or t5["status"] != "closed":
                continue
            f = _feats(sl)
            f.update({"ticker": t, "date": d, "ret5": t5["ret_net_pct"], "ret20": t20["ret_net_pct"],
                      "big5": int(t5["ret_net_pct"] >= BIG5), "big20": int(t20["ret_net_pct"] >= BIG20),
                      "fallen_angel": _fallen_angel(f)})
            rows.append(f)
        if (i + 1) % 60 == 0:
            print(f"  ... {i+1}/{len(tickers)}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "structure_matrix.csv", index=False)
    return df

def report(df):
    b5, b20 = df["big5"].mean() * 100, df["big20"].mean() * 100
    print(f"\n[SS] {len(df)} obs · base P(5j>=10%)={b5:.1f}% · base P(20j>=20%)={b20:.1f}%")
    feats = [k for k in df.columns if k not in ("ticker", "date", "ret5", "ret20", "big5", "big20", "fallen_angel")]
    print(f"\n  {'feature':14}{'lift5(Q5)':>11}{'lift20(Q5)':>12}")
    out = []
    for f in feats:
        s = pd.to_numeric(df[f], errors="coerce"); m = s.notna()
        if m.sum() < 300:
            continue
        try:
            q = pd.qcut(s[m], 5, labels=False, duplicates="drop")
        except Exception:
            continue
        l5 = df["big5"][m][q == q.max()].mean() * 100 - b5
        l20 = df["big20"][m][q == q.max()].mean() * 100 - b20
        out.append((f, l5, l20))
    out.sort(key=lambda x: -x[2])
    for f, l5, l20 in out:
        print(f"  {f:14}{l5:>+10.1f}%{l20:>+11.1f}%")
    # --- le test clé : le filtre composite "fallen angel" ---
    fa = df[df["fallen_angel"] == 1]
    print(f"\n  === STRUCTURE 'FALLEN ANGEL' (défoncé>=60% + base tendue + volume+15% + retest<12%) ===")
    print(f"  n={len(fa)} obs ({len(fa)/len(df)*100:.1f}% de l'univers)")
    if len(fa) >= 30:
        print(f"  P(5j>=10%)  = {fa['big5'].mean()*100:5.1f}%  (base {b5:.1f}%  -> lift {fa['big5'].mean()*100-b5:+.1f}%)")
        print(f"  P(20j>=20%) = {fa['big20'].mean()*100:5.1f}%  (base {b20:.1f}%  -> lift {fa['big20'].mean()*100-b20:+.1f}%)")
        print(f"  ret médian 20j = {fa['ret20'].median():+.2f}%  (univers {df['ret20'].median():+.2f}%)")
        print(f"  winrate 20j    = {(fa['ret20']>0).mean()*100:.1f}%  (univers {(df['ret20']>0).mean()*100:.1f}%)")
    else:
        print("  (échantillon trop faible pour conclure)")

if __name__ == "__main__":
    df = build()
    if not df.empty:
        report(df)
