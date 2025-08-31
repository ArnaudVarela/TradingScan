# backtest_signals.py
import os, math, time
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

PUBLIC_DIR = os.path.join("dashboard", "public")

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, fname: str):
    _ensure_dir(fname)
    df.to_csv(fname, index=False)
    dst = os.path.join(PUBLIC_DIR, fname)
    _ensure_dir(dst)
    df.to_csv(dst, index=False)

# --------- CONFIG BACKTEST ----------
HORIZONS = [5, 10, 20]   # jours ouvrés tenus
ENTRY_COL = "close"      # on prend le Close du jour du signal
MAX_TICKERS = None       # limite pour debug (ex: 500) ; None = illimité
SLEEP_BETWEEN_CALLS = 0.0

# ------------------------------------

def trading_days_shift(series: pd.Series, n: int):
    """Décale un index temporel de n jours de bourse (4h du mat cutoff yfinance)."""
    # Ici, on suppose des lignes quotidiennes; on prend l'index.
    # Pour +n, on prend la valeur à l'offset n (si dispo), sinon NaN.
    # series est 1D alignée sur l'index (Date).
    if series is None or series.empty:
        return pd.Series(index=series.index if series is not None else [])
    return series.shift(-n)

def get_daily_prices(symbol: str, start_date: str, end_date: str):
    """Télécharge le daily entre start_date-7j et end_date+40j pour marge."""
    # marges
    start_pad = pd.to_datetime(start_date) - pd.Timedelta(days=7)
    end_pad   = pd.to_datetime(end_date) + pd.Timedelta(days=40)
    hist = yf.Ticker(symbol).history(
        start=start_pad.strftime("%Y-%m-%d"),
        end=end_pad.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        actions=False
    )
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist[["Close"]].rename(columns={"Close": "close"})
    hist.index = pd.to_datetime(hist.index.date)
    return hist

def main():
    if not os.path.exists("signals_history.csv"):
        print("❌ signals_history.csv introuvable. Lance d'abord le screener.")
        return

    sig = pd.read_csv("signals_history.csv")
    if sig.empty:
        print("❌ signals_history.csv est vide.")
        return

    # Nettoyage / normalisation
    sig["date"] = pd.to_datetime(sig["date"]).dt.date
    sig["date"] = pd.to_datetime(sig["date"])
    sig["ticker_yf"] = sig["ticker_yf"].astype(str)
    sig["bucket"] = sig["bucket"].astype(str)

    # borne dates pour télécharger juste ce qu'il faut
    dmin = sig["date"].min()
    dmax = sig["date"].max() + pd.Timedelta(days=max(HORIZONS)+5)

    # Regrouper par ticker pour limiter les calls yfinance
    groups = sig.groupby("ticker_yf")
    trades = []  # une ligne par (signal, horizon)
    done = 0

    for ticker, g in groups:
        if MAX_TICKERS and done >= MAX_TICKERS:
            break
        # récupérer l'historique une fois
        try:
            px = get_daily_prices(ticker, dmin.strftime("%Y-%m-%d"), dmax.strftime("%Y-%m-%d"))
            if px.empty:
                continue
        except Exception:
            continue

        # pour chaque signal de ce ticker
        for _, row in g.iterrows():
            d0 = pd.to_datetime(row["date"]).normalize()
            if d0 not in px.index:
                # prendre le prochain jour de bourse disponible (premier index >= d0)
                # (si marché fermé le jour J)
                idx = px.index.searchsorted(d0)
                if idx >= len(px.index):
                    continue
                d0 = px.index[idx]

            entry = px.loc[d0, "close"] if d0 in px.index else np.nan
            if np.isnan(entry):
                continue

            # pour chaque horizon
            for h in HORIZONS:
                idx = px.index.searchsorted(d0)
                exit_idx = idx + h
                if exit_idx >= len(px.index):
                    # pas assez de barres futures
                    continue
                d_exit = px.index[exit_idx]
                exit_price = px.iloc[exit_idx]["close"]
                ret = (exit_price / entry) - 1.0

                trades.append({
                    "date_signal": d0.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sector": row.get("sector", "Unknown"),
                    "bucket": row.get("bucket", ""),
                    "horizon_days": h,
                    "entry": entry,
                    "exit": exit_price,
                    "date_exit": d_exit.strftime("%Y-%m-%d"),
                    "ret_pct": ret * 100.0
                })

        done += 1
        if SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    if not trades:
        print("⚠️ Aucune transaction simulée (pas assez d'historique futur).")
        return

    trades_df = pd.DataFrame(trades)
    # Sauvegarde trades détaillés
    save_csv(trades_df, "backtest_trades.csv")

    # Résumés par horizon / bucket / global
    summary = (
        trades_df
        .groupby(["horizon_days", "bucket"])
        .agg(
            n_trades=("ret_pct", "count"),
            winrate=("ret_pct", lambda s: (s > 0).mean() * 100.0),
            avg_ret=("ret_pct", "mean"),
            median_ret=("ret_pct", "median"),
            p95_ret=("ret_pct", lambda s: s.quantile(0.95)),
            p05_ret=("ret_pct", lambda s: s.quantile(0.05)),
        )
        .reset_index()
    )
    save_csv(summary, "backtest_summary.csv")

    # Courbe d’equity equal-weight par horizon (on empile les trades par date_exit)
    for h in HORIZONS:
        sub = trades_df[trades_df["horizon_days"] == h].copy()
        if sub.empty:
            continue
        # Regrouper par date_exit : moyenne des retours des trades sortis ce jour
        daily_ret = (
            sub.groupby("date_exit")["ret_pct"]
               .mean()
               .sort_index()
               .rename("daily_ret_pct")
        )
        # Transformer en equity : base 100, compound
        equity = (1.0 + (daily_ret / 100.0)).cumprod() * 100.0
        eq = pd.DataFrame({"date": pd.to_datetime(daily_ret.index), "equity": equity.values})
        save_csv(eq, f"backtest_equity_{h}d.csv")

    print("[OK] Backtest écrit : backtest_trades.csv, backtest_summary.csv, backtest_equity_[5|10|20]d.csv")

if __name__ == "__main__":
    main()
