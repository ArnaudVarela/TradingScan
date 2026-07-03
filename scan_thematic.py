# scan_thematic.py — Scanne l'univers thématique et sort le classement /100 des setups de pré-explosion.
# Sortie : thematic_setups.csv (+ miroir dashboard), trié par score décroissant.
import os
import datetime
from pathlib import Path

import pandas as pd

import data_fetch as DF   # prefetch_ohlc
import pattern_score as P

ROOT = Path(__file__).parent
LOOKBACK_YEARS = float(os.getenv("SCAN_YEARS", "2"))
MARKET = os.getenv("MARKET_TICKER", "SPY")
MIN_PRICE = float(os.getenv("MIN_PRICE", "1.0"))
MIN_DOLLAR_VOL = float(os.getenv("MIN_DOLLAR_VOL", "1000000"))   # $1M/j min (liquidité)
DROP_PARTIAL = int(os.getenv("SCAN_DROP_PARTIAL", "1"))          # exclure la bougie du jour en cours (mi-séance)
LOG_MIN_SCORE = float(os.getenv("LOG_MIN_SCORE", "50"))          # ne logge que les setups >= ce score
SIGNALS_LOG = ROOT / "signals_log.csv"

def append_signals_log(out, asof):
    """Journal append-only des signaux (base du suivi d'outcome). 1 ligne par (date, ticker)."""
    cols = ["date", "ticker", "score", "bucket", "themes", "price", "mcap_usd"]
    snap = out[out["score"] >= LOG_MIN_SCORE].copy()
    if snap.empty:
        print("[LOG] aucun signal >= seuil à logger"); return
    snap["date"] = pd.Timestamp(asof).strftime("%Y-%m-%d")
    snap["bucket"] = pd.cut(snap["score"], [0, 50, 60, 70, 101], labels=["<50", "50-60", "60-70", "70+"], right=False)
    snap = snap[cols]
    if SIGNALS_LOG.exists():
        try:
            old = pd.read_csv(SIGNALS_LOG)
            merged = pd.concat([old, snap], ignore_index=True).drop_duplicates(subset=["date", "ticker"], keep="last")
        except Exception:
            merged = snap
    else:
        merged = snap
    merged.to_csv(SIGNALS_LOG, index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        merged.to_csv(pub / "signals_log.csv", index=False)
    print(f"[LOG] signals_log.csv : +{len(snap)} signaux du {snap['date'].iloc[0]} (total {len(merged)})")

def _drop_partial(df):
    """Retire la bougie du jour EN COURS (partielle en mi-séance) -> score reproductible sur barres closes."""
    if not DROP_PARTIAL or df is None or getattr(df, "empty", True):
        return df
    et_today = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).date()
    if pd.Timestamp(df.index[-1]).date() >= et_today:
        return df.iloc[:-1]
    return df

def main():
    uni = pd.read_csv(ROOT / "thematic_universe.csv")
    if "valid" in uni.columns:
        uni = uni[uni["valid"].astype(str).str.lower().isin(["true", "1"])]
    tickers = uni["ticker"].astype(str).str.upper().tolist()
    theme_map = dict(zip(uni["ticker"].astype(str).str.upper(), uni["themes"].astype(str)))
    print(f"[SCAN] {len(tickers)} titres thématiques")

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS + 1)
    pm = DF.prefetch_ohlc(tickers + [MARKET], start, end)
    spy = _drop_partial(pm.get(MARKET))

    rows = []
    n_illiquid = 0
    for t in tickers:
        df = _drop_partial(pm.get(t))
        if df is None or df.empty:
            continue
        r = P.preexplosion_score(df, spy)
        if r is None:
            continue
        m = r["metrics"]
        # Filtre liquidité : écarte shells morts / sub-penny / nano illiquides (non actionnables)
        if (m.get("price") or 0) < MIN_PRICE or (m.get("avg_dollar_vol") or 0) < MIN_DOLLAR_VOL:
            n_illiquid += 1
            continue
        rows.append({
            "ticker": t, "themes": theme_map.get(t, ""),
            "score": r["score"], "setup": r["label"],
            "price": m["price"], "rsi": m["rsi"], "macd_hist": m["macd_hist"],
            "chg_1d": m.get("chg_1d"), "chg_7d": m.get("chg_7d"), "chg_1m": m.get("chg_1m"),
            "avg_dollar_vol": m.get("avg_dollar_vol"),
            "dist_to_high_pct": m["dist_to_high_pct"], "base_depth_pct": m["base_depth_pct"],
            "bbwidth_pctile": m["bbwidth_pct"], "vol_dryup": m["vol_dryup"], "overext": m["overext"],
            **{f"c_{k}": v for k, v in r["components"].items()},
        })
    print(f"[FILTRE] {n_illiquid} titres écartés (prix < ${MIN_PRICE:g} ou dollar-vol < ${MIN_DOLLAR_VOL:,.0f}/j)")
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"[MCAP] récupération market cap pour {len(out)} titres...")
    mcaps = DF.fetch_mcaps(out["ticker"].tolist())
    out["mcap_usd"] = out["ticker"].map(mcaps)
    if spy is not None and not spy.empty:
        append_signals_log(out, spy.index[-1])   # asof = dernière barre CLOSE (anti look-ahead)
    out.to_csv(ROOT / "thematic_setups.csv", index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        out.to_csv(pub / "thematic_setups.csv", index=False)

    print(f"[OK] {len(out)} setups scorés -> thematic_setups.csv")
    cols = ["ticker", "themes", "score", "setup", "price", "mcap_usd", "rsi", "dist_to_high_pct", "base_depth_pct", "vol_dryup"]
    print("\n=== TOP 25 SETUPS PRÉ-EXPLOSION (aujourd'hui) ===")
    print(out[cols].head(25).to_string(index=False))
    print("\n=== Répartition des scores ===")
    print(f"  >=70: {(out['score']>=70).sum()}  |  60-70: {((out['score']>=60)&(out['score']<70)).sum()}  "
          f"|  50-60: {((out['score']>=50)&(out['score']<60)).sum()}  |  <50: {(out['score']<50).sum()}")

if __name__ == "__main__":
    main()
