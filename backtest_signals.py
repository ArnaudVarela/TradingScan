# backtest_signals.py — Backtest HONNÊTE (réécriture v2)
#
# Modes (env BACKTEST_MODE) :
#   - "replay"  : walk-forward point-in-time. Régénère les signaux avec le screener CORRIGÉ
#                 sur l'historique (uniquement données <= date), puis mesure les rendements
#                 forward. C'est le mode qui donne un VRAI verdict vs SPY.
#   - "forward" : legacy. Lit signals_history.csv + user_trades.csv et évalue les fenêtres écoulées.
#
# Toute la simulation (entrée next-open, frais+slippage, métriques) vit dans backtest_engine.py
# (pur & testé). Ici : chargement données OHLC + orchestration + écriture des sorties dashboard.
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from random import uniform

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_engine as E

# =================== SORTIES SÛRES (I/O helpers) ============================
PUBLIC_DIR = Path(".")

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _add_generated_at(df):
    if df is None or df.empty:
        return df
    if "generated_at_utc" not in df.columns:
        df = df.copy()
        df["generated_at_utc"] = _utc_stamp()
    return df

def _write_csv(df, path: Path, headers=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or getattr(df, "empty", True):
        cols = headers or (list(df.columns) if isinstance(df, pd.DataFrame) else [])
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        print(f"[WARN] {path.name}: empty -> headers only")
        return
    _add_generated_at(df).to_csv(path, index=False)
    print(f"[OK] {path.name}: {len(df)} rows")

def save_csv(df, fname: str, also_public: bool = True, headers=None):
    _write_csv(df, Path(fname), headers=headers)
    if also_public:
        _write_csv(df, PUBLIC_DIR / fname, headers=headers)

# ====== CONFIG (env-overridable, style repo) ===================================
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "replay")     # replay | forward
HORIZONS      = E.horizons_from_env()                     # défaut [1,3,5,10,20]
ENTRY_MODE    = E.entry_mode_from_env()                   # défaut next_open
COSTS         = E.cost_model_from_env()                   # défaut fee 1bp + slip 5bp / côté

MARKET_TICKER = os.getenv("MARKET_TICKER", "SPY")
UNIVERSE_CSV  = os.getenv("UNIVERSE_CSV", "universe_in_scope.csv")
MAX_TICKERS   = int(os.getenv("MAX_TICKERS", "0")) or None  # 0 = illimité

# Replay
REPLAY_MONTHS = int(os.getenv("REPLAY_MONTHS", "12"))     # fenêtre de test
REPLAY_STEP   = int(os.getenv("REPLAY_STEP", "5"))        # grille (5 = hebdo)
WARMUP_YEARS  = float(os.getenv("WARMUP_YEARS", "3"))     # historique total téléchargé
MIN_HISTORY   = int(os.getenv("MIN_HISTORY", "210"))      # barres min pour scorer (EMA200)
TARGET_CONFIRMED = float(os.getenv("TARGET_CONFIRMED_PCT", "0.02"))
TARGET_PRESIGNAL = float(os.getenv("TARGET_PRESIGNAL_PCT", "0.08"))

# yfinance batch
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "80"))
CHUNK_SLEEP_RANGE = (float(os.getenv("CHUNK_SLEEP_MIN", "1.0")), float(os.getenv("CHUNK_SLEEP_MAX", "3.0")))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "3.0"))
RETRY_JITTER_MAX = float(os.getenv("RETRY_JITTER_MAX", "5.0"))
CURATION_FILES = [PUBLIC_DIR / "user_trades.csv", Path("user_trades.csv")]
# ===============================================================================

def load_universe() -> list:
    for path in [UNIVERSE_CSV, "raw_universe.csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            col = next((c for c in ["ticker_yf", "ticker", "symbol"] if c in df.columns), df.columns[0])
            tickers = [str(t).strip().upper() for t in df[col].tolist() if str(t).strip() not in ("", "-", "nan")]
            tickers = list(dict.fromkeys(tickers))
            if tickers:
                print(f"[UNIV] {len(tickers)} tickers depuis {path} (colonne {col})")
                return tickers[:MAX_TICKERS] if MAX_TICKERS else tickers
    print("[UNIV] aucun univers trouvé.")
    return []

def _ohlc_from_multi(df, ticker):
    try:
        sub = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        return None
    sub.columns = ["open", "high", "low", "close", "volume"]
    sub = sub.dropna(subset=["close"])
    if sub.empty:
        return None
    sub.index = pd.to_datetime(pd.Index(sub.index).date)
    return sub.sort_index()

def prefetch_ohlc(tickers, start, end, batch_size=BATCH_SIZE) -> dict:
    """Télécharge l'OHLC ajusté (open..volume) par batches, avec retry/backoff."""
    out = {}
    symbols = list(dict.fromkeys([str(t) for t in tickers if str(t).strip() not in ("", "nan")]))
    if not symbols:
        return out
    s = pd.to_datetime(start).strftime("%Y-%m-%d")
    e = (pd.to_datetime(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = yf.download(tickers=chunk, start=s, end=e, interval="1d",
                                 auto_adjust=True, progress=False, group_by="ticker", threads=True)
                if df is not None and not df.empty:
                    break
            except Exception as ex:
                print(f"[WARN] batch {i//batch_size+1}: {ex}")
            time.sleep(RETRY_BACKOFF_BASE * attempt + uniform(0.0, RETRY_JITTER_MAX))
        if df is None or df.empty:
            print(f"[WARN] batch vide pour {len(chunk)} tickers.")
        elif isinstance(df.columns, pd.MultiIndex):
            for t in chunk:
                sub = _ohlc_from_multi(df, t)
                if sub is not None:
                    out[t] = sub
        elif len(chunk) == 1:
            sub = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub.columns = ["open", "high", "low", "close", "volume"]
            sub = sub.dropna(subset=["close"])
            if not sub.empty:
                sub.index = pd.to_datetime(pd.Index(sub.index).date)
                out[chunk[0]] = sub.sort_index()
        time.sleep(uniform(*CHUNK_SLEEP_RANGE))
    print(f"[PREFETCH] {len(out)}/{len(symbols)} tickers avec OHLC.")
    return out

def spy_equity(spy_df, start, end) -> pd.DataFrame:
    """Equity SPY buy&hold base 100 sur [start,end]."""
    if spy_df is None or spy_df.empty or pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["date", "equity"])
    px = spy_df.loc[(spy_df.index >= pd.to_datetime(start)) & (spy_df.index <= pd.to_datetime(end)), "close"].dropna()
    if px.empty:
        return pd.DataFrame(columns=["date", "equity"])
    base = float(px.iloc[0])
    if not np.isfinite(base) or base == 0:
        return pd.DataFrame(columns=["date", "equity"])
    return pd.DataFrame({"date": pd.to_datetime(px.index), "equity": (px / base * 100.0).astype(float).values})

# ============================ MODES ============================

SCORE_WINDOW = int(os.getenv("SCORE_WINDOW", "260"))  # fenêtre glissante ~ prod (OHLCV_WINDOW_DAYS)

def gen_signals_replay(price_map, spy_df) -> pd.DataFrame:
    """Régénère les signaux point-in-time avec le screener CORRIGÉ (injection de dépendance)."""
    import mix_ab_screen_indices as M

    def score_fn(df_slice, mkt_slice):
        try:
            mkt = mkt_slice.tail(SCORE_WINDOW) if mkt_slice is not None else None
            feats = M.compute_advanced_score(df_slice.tail(SCORE_WINDOW), mkt=mkt, adv_dv=None)
            return feats.get("score")
        except Exception:
            return None

    def regime_fn(mkt_slice):  # même fenêtre glissante que la prod
        return M._market_regime_factor(mkt_slice.tail(SCORE_WINDOW) if mkt_slice is not None else None)

    # Grille de dates (jours de bourse SPY) sur la fenêtre de test.
    cal = spy_df.index
    end = cal[-1]
    start_test = end - pd.DateOffset(months=REPLAY_MONTHS)
    test_days = cal[cal >= start_test]
    dates = list(test_days[::REPLAY_STEP])
    print(f"[REPLAY] {len(dates)} dates ({REPLAY_MONTHS}m, pas {REPLAY_STEP}j) x {len(price_map)} tickers")
    t0 = time.time()
    sig = E.generate_pointintime_signals(
        price_map, spy_df, dates, score_fn, regime_fn=regime_fn,
        target_confirmed=TARGET_CONFIRMED, target_presignal=TARGET_PRESIGNAL, min_history=MIN_HISTORY)
    print(f"[REPLAY] {len(sig)} signaux générés en {time.time()-t0:.1f}s")
    sig["sector"] = "Unknown"
    sig["source"] = "replay"
    sig["votes"] = 0
    return sig

def load_signals_forward() -> pd.DataFrame:
    """Legacy : signals_history.csv (auto) — évalué tel quel (fenêtres écoulées)."""
    for p in ["signals_history.csv", str(PUBLIC_DIR / "signals_history.csv")]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if {"date", "ticker_yf"}.issubset(df.columns):
                df = df.rename(columns={"date": "date_signal", "ticker_yf": "ticker"})
                df["ticker"] = df["ticker"].astype(str).str.upper()
                if "bucket" not in df.columns:
                    df["bucket"] = ""
                df["cohort"] = df.get("cohort", "P0_other")
                df["source"] = "auto"
                df["votes"] = 0
                return df[["date_signal", "ticker", "bucket", "cohort", "source", "votes"]]
    return pd.DataFrame(columns=["date_signal", "ticker", "bucket", "cohort", "source", "votes"])

# ============================ OUTPUTS ============================

def write_all_outputs(trades: pd.DataFrame, spy_df: pd.DataFrame):
    cal = spy_df.index  # calendrier de bourse pour l'equity marquée au jour le jour
    save_csv(trades, "backtest_trades.csv", headers=E.TRADE_COLS)

    summary = E.summarize(trades, by=("horizon_days", "bucket", "cohort"), ret_col="ret_pct")
    save_csv(summary, "backtest_summary.csv", headers=E.SUMMARY_COLS)

    verdicts = {}
    for h in HORIZONS:
        eq = E.equity_curve(trades, horizon=h, calendar=cal, ret_col="ret_pct")
        save_csv(eq, f"backtest_equity_{h}d.csv", headers=["date", "equity"])
        spy = spy_equity(spy_df, eq["date"].min() if not eq.empty else None,
                         eq["date"].max() if not eq.empty else None)
        save_csv(spy, f"backtest_benchmark_spy_{h}d.csv", headers=["date", "equity"])
        combo = (eq.rename(columns={"equity": "model"})
                 .merge(spy.rename(columns={"equity": "spy"}), on="date", how="inner")) if not eq.empty else pd.DataFrame(columns=["date", "model", "spy"])
        save_csv(combo, f"backtest_equity_{h}d_combo.csv", headers=["date", "model", "spy"])
        verdicts[f"{h}d"] = E.verdict_vs_spy(eq, spy)

    # Copie 10j + cohortes (contrat dashboard)
    prefer = 10 if 10 in HORIZONS else max(HORIZONS)
    eq10 = E.equity_curve(trades, horizon=prefer, calendar=cal, ret_col="ret_pct")
    save_csv(eq10, "backtest_equity_10d.csv", headers=["date", "equity"])
    for cohort in ["P3_confirmed", "P2_highconv", "P1_explore"]:
        sub = trades[(trades["cohort"] == cohort)]
        eqc = E.equity_curve(sub, horizon=prefer, calendar=cal, ret_col="ret_pct")
        save_csv(eqc, f"backtest_equity_10d_{cohort}.csv", headers=["date", "equity"])

    # Courbe "Mes Picks" (curation manuelle)
    user = trades[trades["source"] == "manual"]
    equ = E.equity_curve(user, horizon=prefer, calendar=cal, ret_col="ret_pct")
    save_csv(equ, "backtest_equity_10d_user.csv", headers=["date", "equity"])

    closed = trades[trades["status"] == "closed"]
    stats = {
        "mode": BACKTEST_MODE,
        "entry_mode": ENTRY_MODE,
        "fee_bps": COSTS.fee_bps, "slippage_bps": COSTS.slippage_bps,
        "n_trades": int(len(trades)),
        "n_closed": int(len(closed)),
        "n_open": int((trades["status"] == "open").sum()),
        "tickers_unique": int(trades["ticker"].nunique()) if not trades.empty else 0,
        "verdict_10d": verdicts.get("10d", {}),
        "verdicts": verdicts,
    }
    with open("backtest_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=float)
    with open(PUBLIC_DIR / "backtest_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=float)

    # Verdict lisible
    v = verdicts.get("10d", {})
    print("\n================= VERDICT (10j, net de frais) =================")
    print(f"  Modèle : {v.get('model_total_ret_pct', float('nan')):+.2f}%   "
          f"SPY : {v.get('spy_total_ret_pct', float('nan')):+.2f}%   "
          f"Excès : {v.get('excess_pct', float('nan')):+.2f}%   "
          f"-> BAT SPY : {'OUI' if v.get('beats_spy') else 'NON'}")
    if not summary.empty:
        cf = summary[(summary["bucket"] == "confirmed")]
        if not cf.empty:
            print("\n  Par horizon (bucket=confirmed) : winrate / avg_ret net")
            for _, r in cf.iterrows():
                print(f"    h={int(r['horizon_days']):>2}j  n={int(r['n_trades']):>4}  "
                      f"winrate={r['winrate']:.1f}%  avg={r['avg_ret']:+.2f}%  med={r['median_ret']:+.2f}%")
    print("===============================================================\n")

def main():
    tickers = load_universe()
    if not tickers:
        print("⚠️ Univers vide."); return

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=WARMUP_YEARS + 1)
    all_syms = list(dict.fromkeys(tickers + [MARKET_TICKER]))
    price_map = prefetch_ohlc(all_syms, start, end)
    spy_df = price_map.get(MARKET_TICKER)
    if spy_df is None or spy_df.empty:
        print("⚠️ Pas de données SPY -> impossible de calculer régime/benchmark."); return

    if BACKTEST_MODE == "replay":
        signals = gen_signals_replay({t: price_map[t] for t in tickers if t in price_map}, spy_df)
    else:
        signals = load_signals_forward()
        signals["date_signal"] = pd.to_datetime(signals["date_signal"], errors="coerce")

    if signals is None or signals.empty:
        print("⚠️ Aucun signal généré."); return

    trades = E.build_trades(signals, price_map, HORIZONS, entry_mode=ENTRY_MODE, costs=COSTS)
    if trades.empty:
        print("⚠️ Aucun trade généré."); return

    write_all_outputs(trades, spy_df)
    print("[DONE] Backtest honnête écrit.")

if __name__ == "__main__":
    main()
