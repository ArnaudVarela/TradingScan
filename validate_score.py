# validate_score.py — Valide le score /100 : les scores élevés précèdent-ils de vraies hausses ?
#
# Walk-forward point-in-time sur l'univers thématique : à chaque date, score(data<=d) puis
# rendement forward NET (entrée next-open) sur h jours, + excès vs SPY sur la même fenêtre.
# Agrégé par TRANCHE DE SCORE : si score^ => rendement/excès^ et winrate^ => le /100 est mérité.
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_engine as E
import data_fetch as DF
import pattern_score as P

ROOT = Path(__file__).parent
MONTHS = int(os.getenv("VAL_MONTHS", "36"))
STEP = int(os.getenv("VAL_STEP", "5"))            # grille hebdo
HORIZONS = [int(x) for x in os.getenv("VAL_HORIZONS", "5,10,20").split(",")]
SCORE_WIN = int(os.getenv("VAL_SCORE_WINDOW", "300"))
COSTS = E.cost_model_from_env()
MARKET = "SPY"

def main():
    uni = pd.read_csv(ROOT / "thematic_universe.csv")
    if "valid" in uni.columns:
        uni = uni[uni["valid"].astype(str).str.lower().isin(["true", "1"])]
    tickers = uni["ticker"].astype(str).str.upper().tolist()

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=6)
    pm = DF.prefetch_ohlc(tickers + [MARKET], start, end)
    spy = pm.get(MARKET)
    if spy is None or spy.empty:
        print("⚠️ pas de SPY"); return
    cal = spy.index
    dates = list(cal[cal >= (cal[-1] - pd.DateOffset(months=MONTHS))][::STEP])
    print(f"[VAL] {len(dates)} dates x {len(tickers)} titres, horizons {HORIZONS}")

    # rendement forward SPY par (date,h) = baseline marché
    spy_fwd = {}
    for d in dates:
        for h in HORIZONS:
            tr = E.simulate_trade(spy, d, h, entry_mode="next_open", costs=COSTS)
            spy_fwd[(d, h)] = tr["ret_net_pct"] if (tr and tr["status"] == "closed") else None

    rows = []
    t0 = time.time()
    for i, t in enumerate(tickers):
        px = pm.get(t)
        if px is None or px.empty:
            continue
        spy_by_d = {}
        for d in dates:
            sl = px.loc[px.index <= d]
            if len(sl) < 210:
                continue
            mkt_sl = spy.loc[spy.index <= d]
            r = P.preexplosion_score(sl.tail(SCORE_WIN), mkt_sl.tail(SCORE_WIN))
            if r is None:
                continue
            sc = r["score"]
            for h in HORIZONS:
                tr = E.simulate_trade(px, d, h, entry_mode="next_open", costs=COSTS)
                if tr is None or tr["status"] != "closed":
                    continue
                sf = spy_fwd.get((d, h))
                row = {"date": d, "ticker": t, "score": sc, "h": h,
                       "ret": tr["ret_net_pct"],
                       "excess": (tr["ret_net_pct"] - sf) if sf is not None else np.nan}
                row.update({f"c_{k}": v for k, v in r["components"].items()})
                rows.append(row)
        if (i + 1) % 40 == 0:
            print(f"  ... {i+1}/{len(tickers)} titres ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ aucune observation"); return
    df.to_csv(ROOT / "validate_score_obs.csv", index=False)

    bins = [0, 40, 50, 60, 70, 101]
    labels = ["<40", "40-50", "50-60", "60-70", "70+"]
    df["bucket"] = pd.cut(df["score"], bins=bins, labels=labels, right=False)

    print("\n================= VALIDATION DU SCORE /100 (net de frais) =================")
    for h in HORIZONS:
        sub = df[df["h"] == h]
        print(f"\n--- Horizon {h} jours (n={len(sub)}) ---")
        print(f"  {'tranche':<8}{'n':>7}{'winrate':>9}{'avg_ret':>9}{'med_ret':>9}{'avg_excès_SPY':>15}{'%>SPY':>8}")
        g = sub.groupby("bucket", observed=True)
        for b in labels:
            if b not in g.groups:
                continue
            s = sub[sub["bucket"] == b]
            wr = (s["ret"] > 0).mean() * 100
            ex = s["excess"].dropna()
            beat = (ex > 0).mean() * 100 if len(ex) else float("nan")
            print(f"  {b:<8}{len(s):>7}{wr:>8.1f}%{s['ret'].mean():>+8.2f}%{s['ret'].median():>+8.2f}%"
                  f"{ex.mean():>+14.2f}%{beat:>7.1f}%")
        # corrélation monotone (Spearman via rangs numpy, sans scipy)
        if len(sub) > 10:
            rho = float(np.corrcoef(sub["score"].rank(), sub["ret"].rank())[0, 1])
            print(f"  corrélation score<->rendement (Spearman): {rho:+.3f}")
    print("===========================================================================")

if __name__ == "__main__":
    main()
