# backtest_signals.py
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from random import uniform
import json

import numpy as np
import pandas as pd
import yfinance as yf

# =================== SORTIES SÛRES (I/O helpers) ============================
PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _add_generated_at(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    if "generated_at_utc" not in df.columns:
        df = df.copy()
        df["generated_at_utc"] = _utc_stamp()
    return df

def write_csv_public(df: pd.DataFrame | None, name: str, headers: list[str] | None = None):
    """Écrit dashboard/public/<name> (headers-only si vide) + timestamp si non vide."""
    out = PUBLIC_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or getattr(df, "empty", True):
        cols = headers or (list(df.columns) if isinstance(df, pd.DataFrame) else [])
        pd.DataFrame(columns=cols).to_csv(out, index=False)
        print(f"[WARN] public {name}: empty -> headers only")
        return
    df2 = _add_generated_at(df)
    df2.to_csv(out, index=False)
    print(f"[OK] public {name}: {len(df2)} rows")

def write_csv_root(df: pd.DataFrame | None, name: str, headers: list[str] | None = None):
    """Écrit <repo_root>/<name> (headers-only si vide) + timestamp si non vide."""
    out = Path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or getattr(df, "empty", True):
        cols = headers or (list(df.columns) if isinstance(df, pd.DataFrame) else [])
        pd.DataFrame(columns=cols).to_csv(out, index=False)
        print(f"[WARN] root  {name}: empty -> headers only")
        return
    df2 = _add_generated_at(df)
    df2.to_csv(out, index=False)
    print(f"[OK] root  {name}: {len(df2)} rows")

def save_csv(df: pd.DataFrame | None, fname: str, also_public: bool = True, headers: list[str] | None = None):
    """Écrit à la racine + (par défaut) copie dans dashboard/public/."""
    write_csv_root(df, fname, headers=headers)
    if also_public:
        write_csv_public(df, fname, headers=headers)
# ==============================================================================


# ====== CONFIG =================================================================
# Horizons en JOURS DE BOURSE
HORIZONS = [1, 3, 5, 10, 20]

# Filtre optionnel par bucket (sinon None)
FILTER_BUCKET = None  # ex: "confirmed" ou None

# Limiter le nombre de tickers téléchargés (debug/quota). None = illimité.
MAX_TICKERS = None

# Batch Yahoo
BATCH_SIZE = 80
CHUNK_SLEEP_RANGE = (2.0, 4.0)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 3.0
RETRY_JITTER_MAX = 5.0
# ===============================================================================


# ---------- Helpers I/O ----------
def read_csv_first_available(paths):
    """Tente de lire le 1er CSV dispo parmi paths; renvoie (df, path_utilisé)."""
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p), p
            except Exception:
                pass
    return pd.DataFrame(), None

def placeholder_outputs_when_empty():
    """Crée des fichiers vides/minimaux pour éviter de casser le front."""
    trades_cols  = ["date_signal","ticker","sector","bucket","cohort","votes","horizon_days","status","days_elapsed","entry","exit","date_exit","ret_pct","generated_at_utc"]
    summary_cols = ["horizon_days","bucket","cohort","votes_bin","n_trades","winrate","avg_ret","median_ret","p95_ret","p05_ret","generated_at_utc"]
    equity_cols  = ["date","equity","generated_at_utc"]

    save_csv(pd.DataFrame(columns=trades_cols),  "backtest_trades.csv")
    save_csv(pd.DataFrame(columns=summary_cols), "backtest_summary.csv")
    for h in HORIZONS:
        save_csv(pd.DataFrame(columns=equity_cols), f"backtest_equity_{h}d.csv")
        save_csv(pd.DataFrame(columns=equity_cols), f"backtest_benchmark_spy_{h}d.csv")
        save_csv(pd.DataFrame(columns=["date","model","spy","generated_at_utc"]), f"backtest_equity_{h}d_combo.csv")
    save_csv(pd.DataFrame(columns=equity_cols), "backtest_equity_10d.csv")
    for cohort in ["P3_confirmed", "P2_highconv", "P1_explore"]:
        save_csv(pd.DataFrame(columns=equity_cols), f"backtest_equity_10d_{cohort}.csv")

# ---------- SPY ----------
def compute_spy_benchmark(start_dt, end_dt):
    """
    Télécharge SPY en daily, normalise base 100 au premier point >= start_dt,
    et renvoie un DataFrame: date, equity (float).
    """
    start_pad = (pd.to_datetime(start_dt) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_pad   = (pd.to_datetime(end_dt)   + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        hist = yf.Ticker("SPY").history(
            start=start_pad, end=end_pad, interval="1d",
            auto_adjust=True, actions=False
        )
    except Exception:
        return pd.DataFrame(columns=["date","equity"])

    if hist is None or hist.empty:
        return pd.DataFrame(columns=["date","equity"])

    px = hist[["Close"]].rename(columns={"Close": "close"})
    px.index = pd.to_datetime(px.index.date)
    mask = (px.index >= pd.to_datetime(start_dt)) & (px.index <= pd.to_datetime(end_dt))
    px = px.loc[mask].copy()
    if px.empty:
        return pd.DataFrame(columns=["date","equity"])

    base = float(px["close"].iloc[0])
    px["equity"] = (px["close"] / base) * 100.0
    out = px.reset_index()[["Date","equity"]].rename(columns={"Date":"date"})
    return out

# ---------- Téléchargement des prix en lot ----------
def prefetch_prices(tickers, start_date, end_date, batch_size=BATCH_SIZE):
    """
    Télécharge les prix daily pour une liste de tickers en CHUNKS via yf.download.
    Retourne un dict {ticker: DataFrame(close)} avec index = dates normalisées.
    Gestion retries + jitter + pause entre chunks pour éviter le throttling Yahoo.
    """
    all_px = {}
    # unique tout en gardant l'ordre
    symbols = list(dict.fromkeys([str(t) for t in tickers if pd.notna(t) and str(t).strip() != ""]))
    if MAX_TICKERS is not None:
        symbols = symbols[:MAX_TICKERS]

    if not symbols:
        return all_px

    start_pad = (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_pad   = (pd.to_datetime(end_date)   + pd.Timedelta(days=40)).strftime("%Y-%m-%d")

    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i+batch_size]
        df = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = yf.download(
                    tickers=chunk,
                    start=start_pad,
                    end=end_pad,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                if df is not None and not df.empty:
                    break
            except Exception as e:
                print(f"[WARN] batch {i//batch_size+1}: exception {e}")

            sleep_s = RETRY_BACKOFF_BASE * attempt + uniform(0.0, RETRY_JITTER_MAX)
            print(f"[INFO] retry chunk {i//batch_size+1}/{(len(symbols)+batch_size-1)//batch_size} dans {sleep_s:.1f}s…")
            time.sleep(sleep_s)

        if df is None or df.empty:
            print(f"[WARN] chunk vide/échec pour {len(chunk)} tickers.")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                for t in chunk:
                    try:
                        sub = df[t][["Close"]].rename(columns={"Close": "close"}).dropna()
                        sub.index = pd.to_datetime(sub.index.date)
                        if not sub.empty:
                            all_px[t] = sub
                    except Exception:
                        pass
            else:
                try:
                    sub = df[["Close"]].rename(columns={"Close": "close"}).dropna()
                    sub.index = pd.to_datetime(sub.index.date)
                    if not sub.empty and len(chunk) == 1:
                        all_px[chunk[0]] = sub
                except Exception:
                    pass

        sleep_chunk = uniform(*CHUNK_SLEEP_RANGE)
        time.sleep(sleep_chunk)

    print(f"[INFO] Prefetch terminé: {len(all_px)}/{len(symbols)} tickers avec historique.")
    return all_px

# ---------- Construction equity ----------
def equity_from_daily_returns(daily_ret_pct: pd.Series) -> pd.DataFrame:
    """Transforme une série de retours (%) par date en courbe base 100."""
    if daily_ret_pct is None or daily_ret_pct.empty:
        return pd.DataFrame(columns=["date","equity"])
    daily_ret = daily_ret_pct.sort_index() / 100.0
    equity = (1.0 + daily_ret).cumprod() * 100.0
    return pd.DataFrame({"date": pd.to_datetime(equity.index), "equity": equity.values})

# ---------- Dérivation des piliers / cohortes ----------
def _is_strong(s):
    if not isinstance(s, str):
        return False
    v = s.strip().upper()
    return v in {"STRONG BUY", "STRONGBUY", "STRONG_BUY"}

def add_pillars_and_cohorts(sig: pd.DataFrame) -> pd.DataFrame:
    s = sig.copy()

    tech_col = "technical_local"  # ex: "Strong Buy"
    tv_col   = "tv_reco"          # ex: "STRONG_BUY"
    an_col   = "analyst_bucket"   # ex: "Buy"
    votes_col = None
    for cand in ["votes", "analyst_votes", "rank_votes"]:
        if cand in s.columns:
            votes_col = cand
            break

    s["p_tech"] = s.get(tech_col, pd.Series("", index=s.index)).apply(_is_strong)
    s["p_tv"]   = s.get(tv_col,   pd.Series("", index=s.index)).apply(_is_strong)
    s["p_an"]   = s.get(an_col,   pd.Series("", index=s.index)).astype(str).str.strip().str.upper().eq("BUY")

    s["pillars_met"] = (s[["p_tech","p_tv","p_an"]].fillna(False)).sum(axis=1)

    if votes_col is None:
        s["votes"] = 0
    else:
        s["votes"] = pd.to_numeric(s[votes_col], errors="coerce").fillna(0).astype(int)

    s["votes_bin"] = pd.cut(
        s["votes"], bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"]
    )

    s["cohort"] = np.select(
        [
            s["pillars_met"] >= 3,
            (s["pillars_met"] == 2) & (s["votes"] >= 15),
            (s["pillars_met"] == 1) & (s["votes"] >= 20),
        ],
        ["P3_confirmed", "P2_highconv", "P1_explore"],
        default="P0_other"
    )
    return s

# ---------- Main ----------
def main():
    # 1) Charger les signaux (racine ou public)
    sig, used = read_csv_first_available([
        "signals_history.csv",
        PUBLIC_DIR / "signals_history.csv"
    ])
    if sig.empty:
        print("❌ signals_history.csv introuvable ou vide. Lance d’abord le screener.")
        placeholder_outputs_when_empty()
        return

    print(f"[INFO] signals_history chargé depuis: {used}")

    # Colonnes minimales
    required = {"date", "ticker_yf"}
    missing = [c for c in required if c not in sig.columns]
    if missing:
        print(f"❌ Colonnes manquantes dans signals_history.csv: {missing}")
        placeholder_outputs_when_empty()
        return

    # 2) Normalisation
    sig = sig.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    sig["ticker_yf"] = sig["ticker_yf"].astype(str)
    if "sector" not in sig.columns: sig["sector"] = "Unknown"
    if "bucket" not in sig.columns: sig["bucket"] = ""

    # 2b) Dériver piliers/cohortes
    sig = add_pillars_and_cohorts(sig)

    # 3) Filtre éventuel par bucket (optionnel)
    if FILTER_BUCKET:
        before = len(sig)
        sig = sig[sig["bucket"].astype(str).str.lower() == str(FILTER_BUCKET).lower()]
        print(f"[INFO] Filtre bucket = {FILTER_BUCKET} → {len(sig)} signaux (avant={before})")

    if sig.empty:
        print("⚠️ Aucun signal après filtre. Création de placeholders.")
        placeholder_outputs_when_empty()
        return

    # 4) Fenêtre de prix à télécharger
    dmin = sig["date"].min()
    dmax = sig["date"].max() + pd.Timedelta(days=max(HORIZONS) + 5)

    # 4b) Prefetch des prix en lots
    tickers_unique = sig["ticker_yf"].astype(str).unique().tolist()
    px_map = prefetch_prices(tickers_unique, dmin.strftime("%Y-%m-%d"), dmax.strftime("%Y-%m-%d"), batch_size=BATCH_SIZE)

    # 5) Construire les trades (closed + open)
    trades = []  # une ligne par (signal, horizon)
    closed_count = 0
    open_count = 0

    for ticker, g in sig.groupby("ticker_yf"):
        prices = px_map.get(ticker)
        if prices is None or prices.empty:
            continue

        for _, row in g.iterrows():
            d0 = pd.to_datetime(row["date"])
            # index du 1er prix >= date signal
            idx = prices.index.searchsorted(d0)
            if idx >= len(prices.index):
                continue
            d_entry = prices.index[idx]
            entry = float(prices.iloc[idx]["close"])
            if not np.isfinite(entry):
                continue

            last_idx = len(prices.index) - 1
            for h in HORIZONS:
                exit_idx = idx + h
                if exit_idx <= last_idx:
                    # CLOSED
                    d_exit = prices.index[exit_idx]
                    exit_price = float(prices.iloc[exit_idx]["close"])
                    if not np.isfinite(exit_price):
                        continue
                    ret_pct = (exit_price / entry - 1.0) * 100.0
                    status = "closed"
                    days_elapsed = h
                    closed_count += 1
                else:
                    # OPEN (horizon non atteint) -> utiliser dernier prix dispo
                    d_exit = prices.index[last_idx]
                    exit_price = float(prices.iloc[last_idx]["close"])
                    if not np.isfinite(exit_price):
                        continue
                    ret_pct = (exit_price / entry - 1.0) * 100.0
                    status = "open"
                    days_elapsed = int(last_idx - idx)
                    open_count += 1

                trades.append({
                    "date_signal": d_entry.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sector": row.get("sector", "Unknown"),
                    "bucket": row.get("bucket", ""),
                    "cohort": row.get("cohort", "P0_other"),
                    "votes": int(row.get("votes", 0)),
                    "horizon_days": h,
                    "status": status,
                    "days_elapsed": days_elapsed,
                    "entry": entry,
                    "exit": exit_price,
                    "date_exit": pd.to_datetime(d_exit).strftime("%Y-%m-%d"),
                    "ret_pct": ret_pct
                })

    if not trades:
        print("⚠️ Aucune transaction simulée (pas assez d’historique futur).")
        placeholder_outputs_when_empty()
        return

    trades_df = pd.DataFrame(trades)
    save_csv(trades_df, "backtest_trades.csv")

    # 6) Résumés par horizon / bucket / cohort / votes_bin (CLOSED uniquement)
    sig_key = sig.copy()
    sig_key["date_signal"] = sig_key["date"].dt.strftime("%Y-%m-%d")
    sig_key = sig_key[["date_signal","ticker_yf","votes_bin"]].rename(columns={"ticker_yf":"ticker"})

    trades_closed = trades_df[trades_df["status"] == "closed"].copy()
    trades_closed = trades_closed.merge(sig_key, on=["date_signal","ticker"], how="left")

    summary = (
        trades_closed
        .groupby(["horizon_days","bucket","cohort","votes_bin"], dropna=False)
        .agg(
            n_trades   = ("ret_pct", "count"),
            winrate    = ("ret_pct", lambda s: float((s > 0).mean() * 100.0) if len(s) else 0.0),
            avg_ret    = ("ret_pct", lambda s: float(s.mean()) if len(s) else 0.0),
            median_ret = ("ret_pct", lambda s: float(s.median()) if len(s) else 0.0),
            p95_ret    = ("ret_pct", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
            p05_ret    = ("ret_pct", lambda s: float(s.quantile(0.05)) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values(["horizon_days","cohort","bucket","votes_bin"])
    )
    save_csv(summary, "backtest_summary.csv")

    # 7) Equity curves par horizon (CLOSED uniquement)
    for h in HORIZONS:
        sub = trades_closed[trades_closed["horizon_days"] == h]
        if sub.empty:
            save_csv(pd.DataFrame(columns=["date","equity"]), f"backtest_equity_{h}d.csv")
            save_csv(pd.DataFrame(columns=["date","equity"]), f"backtest_benchmark_spy_{h}d.csv")
            save_csv(pd.DataFrame(columns=["date","model","spy"]), f"backtest_equity_{h}d_combo.csv")
            continue

        daily_ret = sub.groupby("date_exit")["ret_pct"].mean().sort_index()
        eq = equity_from_daily_returns(daily_ret)
        save_csv(eq, f"backtest_equity_{h}d.csv")

        spy = compute_spy_benchmark(eq["date"].min(), eq["date"].max())
        save_csv(spy, f"backtest_benchmark_spy_{h}d.csv")

        combo = (
            eq.rename(columns={"equity": "model"})
              .merge(spy.rename(columns={"equity": "spy"}), on="date", how="inner")
        )
        save_csv(combo, f"backtest_equity_{h}d_combo.csv")

    # 8) Equity par COHORTE (10d, CLOSED uniquement)
    if 10 in HORIZONS:
        for cohort in ["P3_confirmed", "P2_highconv", "P1_explore"]:
            subc = trades_closed[(trades_closed["horizon_days"] == 10) & (trades_closed["cohort"] == cohort)]
            if subc.empty:
                save_csv(pd.DataFrame(columns=["date","equity"]), f"backtest_equity_10d_{cohort}.csv")
                continue
            daily_ret = subc.groupby("date_exit")["ret_pct"].mean().sort_index()
            eqc = equity_from_daily_returns(daily_ret)
            save_csv(eqc, f"backtest_equity_10d_{cohort}.csv")

    # 9) Copie de sûreté pour le front (10j)
    prefer = 10
    fallback = max(HORIZONS)
    src = f"backtest_equity_{prefer}d.csv" if prefer in HORIZONS else f"backtest_equity_{fallback}d.csv"
    eq10, _ = read_csv_first_available([src, PUBLIC_DIR / src])
    save_csv(eq10, "backtest_equity_10d.csv")

    # 10) Stats JSON pour logs/diag
    stats = {
        "signals_loaded": int(len(sig)),
        "tickers_unique": int(sig["ticker_yf"].nunique()),
        "date_min_signal": str(sig["date"].min().date()),
        "date_max_signal": str(sig["date"].max().date()),
        "trades_total": int(len(trades_df)),
        "trades_closed": int(len(trades_closed)),
        "trades_open": int(len(trades_df) - len(trades_closed)),
    }
    for h in HORIZONS:
        stats[f"trades_{h}d_closed"] = int((trades_closed["horizon_days"] == h).sum())
    print("[STATS] backtest:", stats)

    with open("backtest_stats.json", "w") as f:
        json.dump(stats, f)
    with open(PUBLIC_DIR / "backtest_stats.json", "w") as f:
        json.dump(stats, f)

    print("[OK] Backtest écrit : trades (open & closed), summary/equity (closed only), stats JSON.")

if __name__ == "__main__":
    main()
