# backtest_signals.py
# ---------------------------------------------------------
# Backtest simple des signaux stockés dans signals_history.csv
# - Entrée: signals_history.csv  (date, ticker_yf, sector, bucket, ...)
# - Sorties:
#     backtest_trades.csv
#     backtest_summary.csv
#     backtest_equity_[1|3|5|10|20]d.csv
#   (chaque fichier est écrit à la racine ET dans dashboard/public/)
# ---------------------------------------------------------

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

# -------- CONFIG --------
HORIZONS = [1, 3, 5, 10, 20]   # jours de bourse tenus
PUBLIC_DIR = os.path.join("dashboard", "public")
ENTRY_COL = "close"            # on entre au close du jour du signal (ou prochain jour ouvré)
MAX_TICKERS = None             # None = no limit ; ex: 300 pour debug
SLEEP_BETWEEN_CALLS = 0.0      # petit sleep si besoin

# ------------------------

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, fname: str, also_public: bool = True):
    """Écrit le CSV à la racine + copie dans dashboard/public/."""
    _ensure_dir(fname)
    df.to_csv(fname, index=False)
    if also_public:
        dst = os.path.join(PUBLIC_DIR, fname)
        _ensure_dir(dst)
        df.to_csv(dst, index=False)

def read_csv_first_available(paths):
    """Essaie plusieurs chemins et renvoie (DataFrame, path_utilisé) ou (df_vide, None)."""
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p), p
            except Exception:
                pass
    return pd.DataFrame(), None

def get_daily_prices(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Télécharge le daily entre start_date-7j et end_date+40j (marges),
    ajuste l’index en dates-normalisées et renomme Close -> close.
    """
    start_pad = (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_pad   = (pd.to_datetime(end_date) + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(symbol).history(
            start=start_pad,
            end=end_pad,
            interval="1d",
            auto_adjust=True,
            actions=False,
        )
    except Exception as e:
        print(f"[WARN] download fail {symbol}: {e}")
        return pd.DataFrame()

    if hist is None or hist.empty:
        return pd.DataFrame()

    out = hist[["Close"]].rename(columns={"Close": "close"}).copy()
    # index -> date (sans heure)
    out.index = pd.to_datetime(out.index.date)
    return out

def placeholder_outputs_when_empty():
    """Crée des CSV vides (avec entêtes) pour ne pas casser le front lorsque 0 trade simulé."""
    save_csv(pd.DataFrame(columns=[
        "date_signal","ticker","sector","bucket","horizon_days","entry","exit","date_exit","ret_pct"
    ]), "backtest_trades.csv")

    save_csv(pd.DataFrame(columns=[
        "horizon_days","bucket","n_trades","winrate","avg_ret","median_ret","p95_ret","p05_ret"
    ]), "backtest_summary.csv")

    # equity 10j (attendue par le front) + les autres horizons pour être cohérent
    for h in HORIZONS:
        save_csv(pd.DataFrame(columns=["date","equity"]), f"backtest_equity_{h}d.csv")

def main():
    # 1) Charger les signaux (racine ou public)
    sig, used = read_csv_first_available(["signals_history.csv",
                                          os.path.join(PUBLIC_DIR, "signals_history.csv")])
    if sig.empty:
        print("❌ signals_history.csv introuvable ou vide. Lance d’abord le screener.")
        placeholder_outputs_when_empty()
        return

    print(f"[INFO] signals_history chargé depuis: {used}")
    # Normalisation colonnes minimales attendues
    required = {"date", "ticker_yf"}
    missing = [c for c in required if c not in sig.columns]
    if missing:
        print(f"❌ Colonnes manquantes dans signals_history.csv: {missing}")
        placeholder_outputs_when_empty()
        return

    # Typage/clean
    sig = sig.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    sig["ticker_yf"] = sig["ticker_yf"].astype(str)
    if "sector" not in sig.columns: sig["sector"] = "Unknown"
    if "bucket" not in sig.columns: sig["bucket"] = ""

    # 2) Fenêtre de téléchargement: du min(date) au max(date)+max(HORIZONS)+5j
    if sig["date"].empty:
        print("❌ Aucune date dans signals_history.csv.")
        placeholder_outputs_when_empty()
        return

    dmin = sig["date"].min()
    dmax = sig["date"].max() + pd.Timedelta(days=max(HORIZONS) + 5)

    # 3) Backtest par ticker (pour minimiser les téléchargements)
    trades = []
    groups = sig.groupby("ticker_yf")
    done = 0

    for ticker, g in groups:
        if MAX_TICKERS and done >= MAX_TICKERS:
            break

        px = get_daily_prices(ticker, dmin.strftime("%Y-%m-%d"), dmax.strftime("%Y-%m-%d"))
        if px.empty:
            # exemple: delisté → on skip
            print(f"[WARN] {ticker}: pas d’historique téléchargé.")
            continue

        idx_all = px.index

        for _, row in g.iterrows():
            d0 = pd.to_datetime(row["date"]).normalize()
            # si le jour du signal n’est pas un jour de bourse → prendre le prochain disponible
            if d0 not in idx_all:
                pos = idx_all.searchsorted(d0)
                if pos >= len(idx_all):
                    continue
                d0 = idx_all[pos]

            entry = float(px.loc[d0, "close"]) if d0 in px.index else np.nan
            if np.isnan(entry):
                continue

            for h in HORIZONS:
                pos = idx_all.searchsorted(d0)
                exit_pos = pos + h
                if exit_pos >= len(idx_all):
                    # pas assez d’obs futures
                    continue
                d_exit = idx_all[exit_pos]
                exit_price = float(px.iloc[exit_pos]["close"])
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
                    "ret_pct": ret * 100.0,
                })

        done += 1
        if SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    # 4) Sorties
    if not trades:
        print("⚠️ Aucune transaction simulée (pas assez d'historique futur).")
        placeholder_outputs_when_empty()
        return

    trades_df = pd.DataFrame(trades)
    save_csv(trades_df, "backtest_trades.csv")

    # Résumé par horizon & bucket
    def _p(series, q):
        try:
            return float(series.quantile(q))
        except Exception:
            return np.nan

    summary = (
        trades_df
        .groupby(["horizon_days", "bucket"], dropna=False)
        .agg(
            n_trades   = ("ret_pct", "count"),
            winrate    = ("ret_pct", lambda s: float((s > 0).mean() * 100.0)),
            avg_ret    = ("ret_pct", "mean"),
            median_ret = ("ret_pct", "median"),
            p95_ret    = ("ret_pct", lambda s: _p(s, 0.95)),
            p05_ret    = ("ret_pct", lambda s: _p(s, 0.05)),
        )
        .reset_index()
        .sort_values(["horizon_days", "bucket"])
    )
    save_csv(summary, "backtest_summary.csv")

    # Equity curves (equal-weight des trades sortis chaque jour)
    for h in HORIZONS:
        sub = trades_df[trades_df["horizon_days"] == h].copy()
        if sub.empty:
            # créer quand même un fichier vide pour le front
            save_csv(pd.DataFrame(columns=["date","equity"]), f"backtest_equity_{h}d.csv")
            continue

        daily_ret = (
            sub.groupby("date_exit")["ret_pct"]
               .mean()
               .sort_index()
               .rename("daily_ret_pct")
        )
        equity = (1.0 + (daily_ret / 100.0)).cumprod() * 100.0
        eq = pd.DataFrame({"date": pd.to_datetime(daily_ret.index), "equity": equity.values})
        save_csv(eq, f"backtest_equity_{h}d.csv")

    print("[OK] Backtest écrit : backtest_trades.csv, backtest_summary.csv, backtest_equity_[1|3|5|10|20]d.csv")

if __name__ == "__main__":
    main()
