# backfill_signals.py — Amorce signals_log.csv en rejouant le score (corrigé) POINT-IN-TIME sur les mois passés.
#
# Objectif : donner immédiatement des données au suivi d'outcome (score_reliability) sans attendre des semaines.
# Anti look-ahead : à chaque date d, on score sur les barres <= d uniquement ; track_outcomes entre au next-open après d.
import os
from pathlib import Path

import pandas as pd

import data_fetch as DF
import pattern_score as P

ROOT = Path(__file__).parent
MONTHS = int(os.getenv("BACKFILL_MONTHS", "6"))
STEP = int(os.getenv("BACKFILL_STEP", "5"))            # grille hebdo
SCORE_WIN = int(os.getenv("SCORE_WINDOW", "300"))       # fenêtre glissante (comme le screener)
MIN_SCORE = float(os.getenv("LOG_MIN_SCORE", "50"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "1.0"))
MIN_DVOL = float(os.getenv("MIN_DOLLAR_VOL", "1000000"))
MARKET = os.getenv("MARKET_TICKER", "SPY")
SIGNALS_LOG = Path(os.getenv("SIGNALS_LOG_PATH", str(ROOT / "signals_log.csv")))

def _bucket(s):
    return "70+" if s >= 70 else "60-70" if s >= 60 else "50-60" if s >= 50 else "<50"

def main():
    uni = pd.read_csv(ROOT / "thematic_universe.csv")
    if "valid" in uni.columns:
        uni = uni[uni["valid"].astype(str).str.lower().isin(["true", "1"])]
    tickers = uni["ticker"].astype(str).str.upper().tolist()
    theme_map = dict(zip(uni["ticker"].astype(str).str.upper(), uni["themes"].astype(str)))

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=3)
    pm = DF.prefetch_ohlc(tickers + [MARKET], start, end)
    spy = pm.get(MARKET)
    if spy is None or spy.empty:
        print("[BACKFILL] pas de SPY"); return
    cal = spy.index
    dates = list(cal[cal >= (end - pd.DateOffset(months=MONTHS))][::STEP])
    print(f"[BACKFILL] {len(dates)} dates ({MONTHS}m, pas {STEP}j) x {len(tickers)} tickers")

    rows = []
    for d in dates:
        spy_sl = spy.loc[spy.index <= d].tail(SCORE_WIN)
        for t in tickers:
            px = pm.get(t)
            if px is None or px.empty:
                continue
            sl = px.loc[px.index <= d]
            if len(sl) < 150:
                continue
            r = P.preexplosion_score(sl.tail(SCORE_WIN), spy_sl)
            if r is None:
                continue
            m = r["metrics"]
            if (m.get("price") or 0) < MIN_PRICE or (m.get("avg_dollar_vol") or 0) < MIN_DVOL:
                continue
            if r["score"] < MIN_SCORE:
                continue
            rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": t, "score": r["score"],
                         "bucket": _bucket(r["score"]), "themes": theme_map.get(t, ""),
                         "price": m["price"], "mcap_usd": ""})

    new = pd.DataFrame(rows, columns=["date", "ticker", "score", "bucket", "themes", "price", "mcap_usd"])
    print(f"[BACKFILL] {len(new)} signaux historiques (score >= {MIN_SCORE})")
    if SIGNALS_LOG.exists():
        try:
            old = pd.read_csv(SIGNALS_LOG)
            merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=["date", "ticker"], keep="last")
        except Exception:
            merged = new
    else:
        merged = new
    merged = merged.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)
    merged.to_csv(SIGNALS_LOG, index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        merged.to_csv(pub / "signals_log.csv", index=False)
    print(f"[BACKFILL] signals_log.csv total {len(merged)}")

if __name__ == "__main__":
    main()
