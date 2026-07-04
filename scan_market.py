# scan_market.py — Scan de pré-explosion sur l'univers LARGE (< $5B, tous secteurs). Sortie : market_setups.csv.
# Même score/filtre que le thématique ; mcap fournie par l'API Nasdaq (build_market_universe), pas de fast_info.
import os
import json
import datetime
from pathlib import Path

import pandas as pd

import data_fetch as DF
import pattern_score as P
import fallen_angel as FA

ROOT = Path(__file__).parent
LOOKBACK_YEARS = float(os.getenv("SCAN_YEARS", "2"))   # 3 ans fetchés : detect() utilise iloc[-756:] = pic 3 ans (fenêtre exacte du backtest)
MARKET = os.getenv("MARKET_TICKER", "SPY")
MIN_DOLLAR_VOL = float(os.getenv("MIN_DOLLAR_VOL", "1000000"))
DROP_PARTIAL = int(os.getenv("SCAN_DROP_PARTIAL", "1"))
MAX_TICKERS = int(os.getenv("MARKET_MAX_TICKERS", "0")) or None   # >0 = limite (tests)

THEMES = {}
try:
    _tj = json.loads((ROOT / "themes.json").read_text(encoding="utf-8"))["themes"]
    for k, o in _tj.items():
        for t in o["tickers"]:
            THEMES.setdefault(str(t).upper(), set()).add(k)
except Exception:
    pass

def _drop_partial(df):
    if not DROP_PARTIAL or df is None or getattr(df, "empty", True):
        return df
    et = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).date()
    if pd.Timestamp(df.index[-1]).date() >= et:
        return df.iloc[:-1]
    return df

COLS = ["ticker", "themes", "score", "setup", "price", "mcap_usd", "rsi", "macd_hist",
        "chg_1d", "chg_7d", "chg_1m",
        "avg_dollar_vol", "dist_to_high_pct", "base_depth_pct", "bbwidth_pctile", "vol_dryup", "overext"]

def _write(df):
    df.to_csv(ROOT / "market_setups.csv", index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        df.to_csv(pub / "market_setups.csv", index=False)

FA_COLS = ["ticker", "level", "price", "mcap_usd", "score", "setup",
           "dd_from_hi_pct", "base_range_pct", "vol_growth_pct", "dist_resist_pct", "resistance", "stop_ref"]

def _write_fa(df):
    df.to_csv(ROOT / "fallen_angel.csv", index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        df.to_csv(pub / "fallen_angel.csv", index=False)

def main():
    uni_path = ROOT / "market_universe.csv"
    if not uni_path.exists() or uni_path.stat().st_size == 0:
        print("[MARKET SCAN] market_universe.csv absent/vide -> market_setups.csv vide.")
        _write(pd.DataFrame(columns=COLS)); _write_fa(pd.DataFrame(columns=FA_COLS)); return
    try:
        uni = pd.read_csv(uni_path, keep_default_na=False)   # keep_default_na=False : tickers 'NA'/'NAN'/'NULL' littéraux
    except pd.errors.EmptyDataError:
        print("[MARKET SCAN] market_universe.csv illisible -> market_setups.csv vide.")
        _write(pd.DataFrame(columns=COLS)); _write_fa(pd.DataFrame(columns=FA_COLS)); return
    uni["ticker"] = uni["ticker"].astype(str).str.upper()
    mcap_map = dict(zip(uni["ticker"], uni["mcap_usd"]))
    tickers = uni["ticker"].tolist()
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    print(f"[MARKET SCAN] {len(tickers)} titres")

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS + 1)
    pm = DF.prefetch_ohlc(tickers + [MARKET], start, end)
    spy = _drop_partial(pm.get(MARKET))

    rows, fa_rows, n_illiquid = [], [], 0
    for t in tickers:
        df = _drop_partial(pm.get(t))
        if df is None or df.empty:
            continue
        r = P.preexplosion_score(df, spy)
        if r is None:
            continue
        m = r["metrics"]
        if (m.get("avg_dollar_vol") or 0) < MIN_DOLLAR_VOL:
            n_illiquid += 1
            continue
        rows.append({
            "ticker": t, "themes": "|".join(sorted(THEMES.get(t, []))),
            "score": r["score"], "setup": r["label"], "price": m["price"],
            "mcap_usd": mcap_map.get(t), "rsi": m["rsi"], "macd_hist": m["macd_hist"],
            "chg_1d": m.get("chg_1d"), "chg_7d": m.get("chg_7d"), "chg_1m": m.get("chg_1m"),
            "avg_dollar_vol": m.get("avg_dollar_vol"),
            "dist_to_high_pct": m["dist_to_high_pct"], "base_depth_pct": m["base_depth_pct"],
            "bbwidth_pctile": m["bbwidth_pct"], "vol_dryup": m["vol_dryup"], "overext": m["overext"],
            **{f"c_{k}": v for k, v in r["components"].items()},
        })
        fa = FA.detect(df)   # structure "fallen angel" (défoncé + base tendue + volume + retest) sur l'historique complet
        if fa:
            fa_rows.append({
                "ticker": t, "level": fa["level"], "price": m["price"], "mcap_usd": mcap_map.get(t),
                "score": r["score"], "setup": r["label"],
                "dd_from_hi_pct": fa["dd_from_hi_pct"], "base_range_pct": fa["base_range_pct"],
                "vol_growth_pct": fa["vol_growth_pct"], "dist_resist_pct": fa["dist_resist_pct"],
                "resistance": fa["resistance"], "stop_ref": fa["stop_ref"],
            })
    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=False).reset_index(drop=True) if not out.empty else pd.DataFrame(columns=COLS)
    _write(out)
    print(f"[MARKET SCAN] {len(out)} setups (illiquides écartés: {n_illiquid}) -> market_setups.csv")
    if not out.empty:
        print(f"  >=70: {(out['score']>=70).sum()}  |  60-70: {((out['score']>=60)&(out['score']<70)).sum()}")

    # --- Module Fallen Angel (candidats haute-variance, edge dans les sorties — paper-first) ---
    fa_df = pd.DataFrame(fa_rows, columns=FA_COLS)
    if not fa_df.empty:
        _lvl = {"strict": 0, "relax": 1}
        fa_df = fa_df.sort_values(by=["level", "vol_growth_pct"],
                                  key=lambda s: s.map(_lvl) if s.name == "level" else s,
                                  ascending=[True, False]).reset_index(drop=True)
    _write_fa(fa_df)
    print(f"[FALLEN ANGEL] {len(fa_df)} candidats"
          + (f" ({int((fa_df['level']=='strict').sum())} strict)" if not fa_df.empty else ""))

if __name__ == "__main__":
    main()
