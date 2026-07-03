# track_outcomes.py — Résout le rendement forward RÉEL des signaux loggés + rapport de fiabilité du score.
#
# Anti look-ahead : entrée au NEXT OPEN après la date de signal, résolution UNIQUEMENT quand la fenêtre h
# est écoulée (backtest_engine.simulate_trade renvoie 'open' sinon -> on ré-essaie un autre jour).
# Entrées : signals_log.csv (append-only, produit par scan_thematic). Sorties : signals_resolved.csv (détail)
# + score_reliability.csv (rapport glissant par bucket x horizon, lu par le dashboard).
import os
from pathlib import Path

import numpy as np
import pandas as pd

import data_fetch as DF
import backtest_engine as E

ROOT = Path(__file__).parent
SIGNALS_LOG = Path(os.getenv("SIGNALS_LOG_PATH", str(ROOT / "signals_log.csv")))
OUT = ROOT / "score_reliability.csv"
HORIZONS = [int(x) for x in os.getenv("OUT_HORIZONS", "5,10,20").split(",")]
WINDOW_DAYS = int(os.getenv("RELIABILITY_WINDOW_DAYS", "120"))   # fenêtre glissante d'évaluation
COSTS = E.cost_model_from_env()
MARKET = "SPY"
BUCKETS = ["70+", "60-70", "50-60", "<50"]
REPORT_COLS = ["window_days", "horizon", "bucket", "n", "winrate", "avg_ret", "med_ret", "avg_excess_spy", "pct_beat_spy"]

def _write(rep: pd.DataFrame):
    rep.to_csv(OUT, index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        rep.to_csv(pub / OUT.name, index=False)

def main():
    if not SIGNALS_LOG.exists():
        print(f"[OUTCOME] {SIGNALS_LOG.name} absent — rien à résoudre."); _write(pd.DataFrame(columns=REPORT_COLS)); return
    log = pd.read_csv(SIGNALS_LOG)
    if log.empty:
        print("[OUTCOME] log vide."); _write(pd.DataFrame(columns=REPORT_COLS)); return
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    log = log.dropna(subset=["date", "ticker"])
    tickers = sorted(log["ticker"].astype(str).str.upper().unique())
    start = log["date"].min() - pd.Timedelta(days=10)
    end = pd.Timestamp.today().normalize()
    print(f"[OUTCOME] {len(log)} signaux, {len(tickers)} tickers, {log['date'].min().date()} -> {log['date'].max().date()}")
    pm = DF.prefetch_ohlc(tickers + [MARKET], start, end)
    spy = pm.get(MARKET)

    rows = []
    for _, s in log.iterrows():
        px = pm.get(str(s["ticker"]).upper())
        if px is None or px.empty:
            continue
        for h in HORIZONS:
            tr = E.simulate_trade(px, s["date"], h, entry_mode="next_open", costs=COSTS)
            if tr is None or tr["status"] != "closed":
                continue   # fenêtre pas encore écoulée -> sera résolu un prochain jour
            spret = np.nan
            if spy is not None and not spy.empty:
                sp = E.simulate_trade(spy, s["date"], h, entry_mode="next_open", costs=COSTS)
                if sp is not None and sp["status"] == "closed":
                    spret = sp["ret_net_pct"]
            rows.append({"date": s["date"], "ticker": s["ticker"], "score": s["score"],
                         "bucket": s.get("bucket", ""), "h": h,
                         "ret": tr["ret_net_pct"], "excess": tr["ret_net_pct"] - spret})

    obs = pd.DataFrame(rows)
    resolved = int(len(obs))
    if obs.empty:
        print("[OUTCOME] aucun signal encore résolu (fenêtres non écoulées)."); _write(pd.DataFrame(columns=REPORT_COLS)); return
    obs.to_csv(ROOT / "signals_resolved.csv", index=False)

    # Rapport glissant par bucket x horizon
    cutoff = end - pd.Timedelta(days=WINDOW_DAYS)
    recent = obs[obs["date"] >= cutoff]
    report = []
    for h in HORIZONS:
        for b in BUCKETS:
            s = recent[(recent["h"] == h) & (recent["bucket"].astype(str) == b)]
            if s.empty:
                continue
            ex = s["excess"].dropna()
            report.append({
                "window_days": WINDOW_DAYS, "horizon": h, "bucket": b, "n": int(len(s)),
                "winrate": round(float((s["ret"] > 0).mean() * 100), 1),
                "avg_ret": round(float(s["ret"].mean()), 2),
                "med_ret": round(float(s["ret"].median()), 2),
                "avg_excess_spy": round(float(ex.mean()), 2) if len(ex) else np.nan,
                "pct_beat_spy": round(float((ex > 0).mean() * 100), 1) if len(ex) else np.nan,
            })
    rep = pd.DataFrame(report, columns=REPORT_COLS)
    _write(rep)

    # Spearman global (monotonie score <-> rendement) sur la fenêtre
    def spearman(a, b):
        m = a.notna() & b.notna()
        return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1]) if m.sum() > 10 else float("nan")
    print(f"[OUTCOME] {resolved} observations résolues.")
    for h in HORIZONS:
        sub = recent[recent["h"] == h]
        if len(sub) > 10:
            print(f"  h={h}j : Spearman score<->rendement = {spearman(sub['score'], sub['ret']):+.3f}")
    print("\n=== RAPPORT DE FIABILITÉ (fenêtre glissante) ===")
    print(rep.to_string(index=False) if not rep.empty else "  (pas encore assez de signaux résolus)")

if __name__ == "__main__":
    main()
