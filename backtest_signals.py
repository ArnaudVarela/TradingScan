# backtest_signals.py
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

PUBLIC_DIR = os.path.join("dashboard", "public")

# --------- CONFIG BACKTEST ----------
HORIZONS = [5, 10, 20]      # jours ouvrés tenus
ENTRY_COL = "close"         # Close du jour du signal
MAX_TICKERS = None          # limite pour debug ; None = illimité
SLEEP_BETWEEN_CALLS = 0.0   # politesse YF
# ------------------------------------

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, fname: str, also_public: bool = True):
    """Écrit fname à la racine + copie dans dashboard/public/fname."""
    _ensure_dir(fname)
    df.to_csv(fname, index=False)
    if also_public:
        dst = os.path.join(PUBLIC_DIR, fname)
        _ensure_dir(dst)
        df.to_csv(dst, index=False)

def get_daily_prices(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Télécharge le daily entre start_date-7j et end_date+40j pour marge, auto_adjust=True."""
    start_pad = (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_pad   = (pd.to_datetime(end_date)   + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    hist = yf.Ticker(symbol).history(
        start=start_pad,
        end=end_pad,
        interval="1d",
        auto_adjust=True,
        actions=False
    )
    if hist is None or hist.empty:
        return pd.DataFrame()
    out = hist[["Close"]].rename(columns={"Close": "close"})
    out.index = pd.to_datetime(out.index.date)  # index date (naïf)
    return out

def main():
    src = "signals_history.csv"
    if not os.path.exists(src):
        print("❌ signals_history.csv introuvable. Lance d'abord le screener.")
        return

    sig = pd.read_csv(src)
    if sig.empty:
        print("❌ signals_history.csv est vide.")
        return

    # Normalisation colonnes d'entrée
    # requis: date (date du signal), ticker_yf
    if "date" not in sig.columns or "ticker_yf" not in sig.columns:
        print("❌ signals_history.csv doit contenir au minimum les colonnes: date, ticker_yf.")
        return

    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    sig["ticker_yf"] = sig["ticker_yf"].astype(str)
    if "bucket" not in sig.columns:
        sig["bucket"] = ""  # facultatif

    dmin = sig["date"].min()
    dmax = sig["date"].max() + pd.Timedelta(days=max(HORIZONS) + 5)

    trades = []  # une ligne par (signal, horizon)

    # Grouper par ticker pour minimiser les appels réseau
    for i, (ticker, g) in enumerate(sig.groupby("ticker_yf"), 1):
        if MAX_TICKERS and i > MAX_TICKERS:
            break
        try:
            px = get_daily_prices(ticker, dmin.strftime("%Y-%m-%d"), dmax.strftime("%Y-%m-%d"))
            if px.empty:
                # ex: delisted ou pas de data
                continue
        except Exception:
            continue

        # Pour chaque signal de ce ticker
        for _, row in g.iterrows():
            d0 = pd.to_datetime(row["date"]).normalize()

            # Trouver premier jour de bourse >= d0
            if d0 not in px.index:
                idx0 = px.index.searchsorted(d0)
                if idx0 >= len(px.index):
                    continue
                d0 = px.index[idx0]
            else:
                idx0 = px.index.get_indexer([d0])[0]

            entry = float(px.iloc[idx0][ENTRY_COL]) if idx0 < len(px.index) else np.nan
            if not np.isfinite(entry):
                continue

            # Pour chaque horizon : J+N barres daily
            for h in HORIZONS:
                exit_idx = idx0 + h
                if exit_idx >= len(px.index):
                    continue
                d_exit = px.index[exit_idx]
                exit_price = float(px.iloc[exit_idx][ENTRY_COL])
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

        if SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    if not trades:
        print("⚠️ Aucune transaction simulée (pas assez d'historique futur).")
        return

    trades_df = pd.DataFrame(trades)

    # 1) Journal de trades
    save_csv(trades_df, "backtest_trades.csv")

    # 2) KPIs par horizon / bucket
    summary = (
        trades_df
        .groupby(["horizon_days", "bucket"], dropna=False)
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

    # 3) Courbes d’equity (equal-weight) par horizon
    for h in HORIZONS:
        sub = trades_df[trades_df["horizon_days"] == h].copy()
        if sub.empty:
            continue
        daily_ret = (
            sub.groupby("date_exit")["ret_pct"]
               .mean()
               .sort_index()
               .rename("daily_ret_pct")
        )
        equity = (1.0 + (daily_ret / 100.0)).cumprod() * 100.0
        eq = pd.DataFrame({
            "date": pd.to_datetime(daily_ret.index),
            "equity": equity.values
        })
        save_csv(eq, f"backtest_equity_{h}d.csv")

        # Alias pour le front : on veut toujours backtest_equity_10d.csv
        if h == 10:
            save_csv(eq, "backtest_equity_10d.csv")

    print("[OK] Backtest écrit : backtest_trades.csv, backtest_summary.csv, backtest_equity_[5|10|20]d.csv (et alias 10d).")

if __name__ == "__main__":
    main()
