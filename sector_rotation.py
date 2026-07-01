# sector_rotation.py — Backtest HONNÊTE d'une rotation sectorielle (relative strength + gate risk-off)
#
# Alpha source alternative au momentum-stock (qui s'est révélé être du beta+).
# Univers = 11 ETFs SPDR sectoriels. À chaque rebalance : top-N secteurs par momentum relatif,
# équipondérés ; cash si SPY sous sa MM200 (gate risk-off, évite les bears).
# Réutilise backtest_engine (entrée next-open, frais/slippage, anti look-ahead, equity marquée jour/jour).
import os
import numpy as np
import pandas as pd

import backtest_engine as E
import backtest_signals as BS  # réutilise prefetch_ohlc / spy_equity

SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC"]
MARKET = os.getenv("MARKET_TICKER", "SPY")
TOP_N = int(os.getenv("ROT_TOP_N", "3"))
REBAL = int(os.getenv("ROT_REBAL_DAYS", "21"))       # ~mensuel
LOOKBACKS = tuple(int(x) for x in os.getenv("ROT_LOOKBACKS", "63,126,252").split(","))
REPLAY_MONTHS = int(os.getenv("REPLAY_MONTHS", "60"))
COSTS = E.cost_model_from_env()

def momentum_score(px_slice):
    c = px_slice["close"]
    rs = [c.iloc[-1] / c.iloc[-lb - 1] - 1.0 for lb in LOOKBACKS if len(c) > lb]
    return float(np.mean(rs)) if rs else None

def spy_riskon(spy_slice):
    c = spy_slice["close"]
    if len(c) < 200:
        return True
    return bool(c.iloc[-1] >= c.tail(200).mean())

def build_signals(price_map, spy_df, use_gate: bool) -> pd.DataFrame:
    cal = spy_df.index
    start_test = cal[-1] - pd.DateOffset(months=REPLAY_MONTHS)
    dates = list(cal[cal >= start_test][::REBAL])
    rows = []
    for d in dates:
        spy_sl = spy_df.loc[spy_df.index <= d]
        if use_gate and not spy_riskon(spy_sl):
            continue  # risk-off -> cash
        scored = []
        for t in SECTOR_ETFS:
            px = price_map.get(t)
            if px is None:
                continue
            sl = px.loc[px.index <= d]
            if len(sl) < max(LOOKBACKS) + 5:
                continue
            m = momentum_score(sl)
            if m is not None and np.isfinite(m):
                scored.append((t, m))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[1])
        for t, _ in scored[:TOP_N]:
            rows.append({"date_signal": d.strftime("%Y-%m-%d"), "ticker": t,
                         "bucket": "confirmed", "cohort": "rotation", "source": "rotation"})
    return pd.DataFrame(rows)

def yearly(model_eq, spy_eq):
    def yret(e):
        e = e.copy(); e["date"] = pd.to_datetime(e["date"]); s = e.set_index("date")["equity"].astype(float)
        return {y: (g.iloc[-1] / g.iloc[0] - 1) * 100 for y, g in s.groupby(s.index.year)}
    my, sy = yret(model_eq), yret(spy_eq)
    print(f"  {'année':<8}{'Rotation':>12}{'SPY':>10}{'excès':>10}")
    for y in sorted(set(my) | set(sy)):
        m, s = my.get(y, float('nan')), sy.get(y, float('nan'))
        print(f"  {y:<8}{m:>+11.1f}%{s:>+9.1f}%{m - s:>+9.1f}%  {'BAT' if m > s else 'sous'}")

def mdd(eq):
    s = eq.set_index("date")["equity"].astype(float)
    return float(((s / s.cummax()) - 1).min() * 100)

def risk_stats(eq):
    s = eq.copy(); s["date"] = pd.to_datetime(s["date"]); s = s.set_index("date")["equity"].astype(float)
    r = s.pct_change().dropna()
    days = max((s.index[-1] - s.index[0]).days, 1)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (365.0 / days) - 1.0
    sharpe = (r.mean() / (r.std() or 1e-9)) * np.sqrt(252)
    dd = float(((s / s.cummax()) - 1).min())
    calmar = (cagr / abs(dd)) if dd < 0 else float("nan")
    return cagr * 100, sharpe, dd * 100, calmar

def run(price_map, spy_df, use_gate: bool, label: str):
    sig = build_signals(price_map, spy_df, use_gate)
    trades = E.build_trades(sig, price_map, [REBAL], entry_mode="next_open", costs=COSTS)
    # equity marquée aux prix réels (Sharpe/DD honnêtes) ; coûts négligeables en rotation mensuelle.
    eq = E.daily_marked_equity(trades, price_map, spy_df.index)
    if eq.empty:
        print(f"\n### {label}: aucune donnée"); return
    spy = BS.spy_equity(spy_df, eq["date"].min(), eq["date"].max())
    rtot = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100
    stot = (spy["equity"].iloc[-1] / spy["equity"].iloc[0] - 1) * 100
    rc, rsh, rdd, rcal = risk_stats(eq)
    sc, ssh, sdd, scal = risk_stats(spy)
    print(f"\n### {label}  (top{TOP_N}, rebal {REBAL}j, gate={'ON' if use_gate else 'OFF'})")
    print(f"  Rendement total  Rotation: {rtot:+.1f}%   SPY: {stot:+.1f}%   "
          f"excès: {rtot - stot:+.1f}%   -> {'BAT SPY' if rtot > stot else 'sous-perf'}")
    print(f"  Risque ajusté    Rotation: CAGR {rc:+.1f}% | Sharpe {rsh:.2f} | MaxDD {rdd:+.1f}% | Calmar {rcal:.2f}")
    print(f"                   SPY:      CAGR {sc:+.1f}% | Sharpe {ssh:.2f} | MaxDD {sdd:+.1f}% | Calmar {scal:.2f}")
    yearly(eq, spy)

def main():
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=7)
    price_map = BS.prefetch_ohlc(SECTOR_ETFS + [MARKET], start, end)
    spy_df = price_map.get(MARKET)
    if spy_df is None or spy_df.empty:
        print("⚠️ pas de SPY"); return
    print(f"[ROT] {len([t for t in SECTOR_ETFS if t in price_map])}/{len(SECTOR_ETFS)} ETFs, fenêtre {REPLAY_MONTHS}m")
    run(price_map, spy_df, use_gate=False, label="SANS gate régime")
    run(price_map, spy_df, use_gate=True, label="AVEC gate régime (cash si SPY < MM200)")

if __name__ == "__main__":
    main()
