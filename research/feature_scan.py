import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root importable
# feature_scan.py — Découverte de features prédisant la QUEUE DE DROITE (ret>=20% à 20j).
# Walk-forward point-in-time : ~15 features candidates + rendement forward (next-open, net). OHLC mis en cache (pickle).
# Sortie : feature_matrix.csv + table de LIFT univarié (P(gain>=20%) par quintile de feature, corr avec ret).
import os
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_engine as E
import data_fetch as DF
import pattern_score as P  # helpers _atr, _bb_width, _rsi, _ema, _pct_rank

ROOT = _pl.Path(__file__).resolve().parents[1]
SCRATCH = Path(os.getenv("SCRATCH", ROOT))
CACHE = SCRATCH / "ohlc_thematic.pkl"
MONTHS = int(os.getenv("FS_MONTHS", "24"))
STEP = int(os.getenv("FS_STEP", "5"))
H = int(os.getenv("FS_H", "20"))
BIG = float(os.getenv("FS_BIG", "20"))     # seuil "gros mouvement" en %
COSTS = E.cost_model_from_env()

def _feats(sl):
    c, h, l, v = sl["close"], sl["high"], sl["low"], sl["volume"]
    o = sl["open"]
    n = len(c); last = float(c.iloc[-1])
    def r(k):
        return float(last / c.iloc[-1 - k] - 1.0) if n > k else np.nan
    vol5, vol10, vol20, vol60 = [float(v.iloc[-k:].mean()) for k in (5, 10, 20, 60)]
    atr = P._atr(sl, 14); atr_now = float(atr.iloc[-1]); atr_base = float(atr.iloc[-60:].mean())
    bbw = P._bb_width(c, 20); bbw_now = float(bbw.iloc[-1]); bbw_20 = float(bbw.iloc[-20]) if n > 20 else bbw_now
    hi252 = float(h.iloc[-252:].max()); lo252 = float(l.iloc[-252:].min())
    hi20 = float(h.iloc[-20:].max()); lo20 = float(l.iloc[-20:].min())
    ema50 = float(P._ema(c, 50).iloc[-1]); ema200 = float(P._ema(c, 200).iloc[-1]) if n >= 200 else ema50
    rsi = float(P._rsi(c).iloc[-1])
    return {
        "ret_1m": r(21), "ret_3m": r(64), "ret_6m": r(126),
        "rvol5": vol5 / (vol60 + 1e-9), "rvol10": vol10 / (vol60 + 1e-9),
        "vol_spike": float(v.iloc[-5:].max()) / (vol60 + 1e-9),
        "atr_expand": atr_now / (atr_base + 1e-9),
        "bbw_pct": P._pct_rank(bbw, 126),
        "bbw_expand": bbw_now / (bbw_20 + 1e-9),
        "dist_hi52": (hi252 - last) / (hi252 + 1e-9),
        "dist_lo52": (last - lo252) / (lo252 + 1e-9),
        "range20_pos": (last - lo20) / (hi20 - lo20 + 1e-9),
        "rsi": rsi,
        "price": last,
        "log_dvol": float(np.log10((c.iloc[-20:] * v.iloc[-20:]).mean() + 1)),
        "up20": float((c.diff().iloc[-20:] > 0).mean()),
        "gap": float(o.iloc[-1] / c.iloc[-2] - 1.0) if n > 1 else 0.0,
        "above50": float(last > ema50), "above200": float(last > ema200),
        "dryup": vol20 / (vol60 + 1e-9),
    }

def build():
    uni = pd.read_csv(ROOT / "thematic_universe.csv")
    if "valid" in uni.columns:
        uni = uni[uni["valid"].astype(str).str.lower().isin(["true", "1"])]
    tickers = uni["ticker"].astype(str).str.upper().tolist()
    if CACHE.exists():
        pm = pd.read_pickle(CACHE); print(f"[FS] OHLC depuis cache ({len(pm)} titres)")
    else:
        end = pd.Timestamp.today().normalize()
        pm = DF.prefetch_ohlc(tickers + ["SPY"], end - pd.DateOffset(months=MONTHS + 15), end)
        pd.to_pickle(pm, CACHE); print(f"[FS] OHLC fetché + caché ({len(pm)} titres)")
    spy = pm.get("SPY")
    cal = spy.index
    dates = list(cal[cal >= (cal[-1] - pd.DateOffset(months=MONTHS))][::STEP])
    print(f"[FS] {len(dates)} dates x {len(tickers)} titres, horizon {H}j, seuil gros mvt >= {BIG}%")

    rows = []
    for i, t in enumerate(tickers):
        px = pm.get(t)
        if px is None or px.empty:
            continue
        for d in dates:
            sl = px.loc[px.index <= d]
            if len(sl) < 210:
                continue
            tr = E.simulate_trade(px, d, H, entry_mode="next_open", costs=COSTS)
            if tr is None or tr["status"] != "closed":
                continue
            f = _feats(sl)
            f.update({"ticker": t, "date": d, "ret": tr["ret_net_pct"], "big": int(tr["ret_net_pct"] >= BIG)})
            rows.append(f)
        if (i + 1) % 60 == 0:
            print(f"  ... {i+1}/{len(tickers)}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "feature_matrix.csv", index=False)
    return df

def report(df):
    base = df["big"].mean() * 100
    print(f"\n[FS] {len(df)} obs · base rate P(gain>={BIG:.0f}%)={base:.1f}% · corr features -> ret & lift sur P(gros mvt)")
    feats = [k for k in df.columns if k not in ("ticker", "date", "ret", "big")]
    res = []
    for f in feats:
        s = pd.to_numeric(df[f], errors="coerce")
        m = s.notna() & df["ret"].notna()
        if m.sum() < 200:
            continue
        rho = float(np.corrcoef(s[m].rank(), df["ret"][m].rank())[0, 1])
        # quintile haut vs bas -> P(big)
        q = pd.qcut(s[m], 5, labels=False, duplicates="drop")
        big = df["big"][m]
        p_hi = big[q == q.max()].mean() * 100
        p_lo = big[q == 0].mean() * 100
        res.append((f, rho, p_lo, p_hi, p_hi - base))
    res.sort(key=lambda x: -x[4])
    print(f"  {'feature':14}{'rho(ret)':>10}{'P(big)|Q1':>11}{'P(big)|Q5':>11}{'lift Q5':>9}")
    for f, rho, plo, phi, lift in res:
        print(f"  {f:14}{rho:>+10.3f}{plo:>10.1f}%{phi:>10.1f}%{lift:>+8.1f}%")

if __name__ == "__main__":
    df = build()
    if not df.empty:
        report(df)
