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
    write_csv_root(df, fname, headers=headers)
    if also_public:
        write_csv_public(df, fname, headers=headers)
# ==============================================================================


# ====== CONFIG =================================================================
HORIZONS = [1, 3, 5, 10, 20]
FILTER_BUCKET = None
MAX_TICKERS = None

BATCH_SIZE = 80
CHUNK_SLEEP_RANGE = (2.0, 4.0)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 3.0
RETRY_JITTER_MAX = 5.0

# Fichier de curation (Mes Picks)
CURATION_FILES = [
    PUBLIC_DIR / "user_trades.csv",
    Path("user_trades.csv"),
]
# ===============================================================================


# ---------- Helpers I/O ----------
def read_csv_first_available(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p), str(p)
            except Exception:
                pass
    return pd.DataFrame(), None

def placeholder_outputs_when_empty():
    trades_cols  = ["date_signal","ticker","sector","bucket","cohort","votes","horizon_days","status","days_elapsed","entry","exit","date_exit","ret_pct","notes","source","generated_at_utc"]
    summary_cols = ["horizon_days","bucket","cohort","n_trades","winrate","avg_ret","median_ret","p95_ret","p05_ret","generated_at_utc"]
    equity_cols  = ["date","equity","generated_at_utc"]

    save_csv(pd.DataFrame(columns=trades_cols),  "backtest_trades.csv")
    save_csv(pd.DataFrame(columns=summary_cols), "backtest_summary.csv")
    for h in HORIZONS:
        save_csv(pd.DataFrame(columns=equity_cols), f"backtest_equity_{h}d.csv")
        save_csv(pd.DataFrame(columns=equity_cols), f"backtest_benchmark_spy_{h}d.csv")
        save_csv(pd.DataFrame(columns=["date","model","spy","generated_at_utc"]), f"backtest_equity_{h}d_combo.csv")
    save_csv(pd.DataFrame(columns=equity_cols), "backtest_equity_10d.csv")
    save_csv(pd.DataFrame(columns=equity_cols), "backtest_equity_10d_user.csv")  # NEW (Mes Picks)
    for cohort in ["P3_confirmed", "P2_highconv", "P1_explore"]:
        save_csv(pd.DataFrame(columns=equity_cols), f"backtest_equity_10d_{cohort}.csv")

# ---------- SPY ----------
def compute_spy_benchmark(start_dt, end_dt):
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

# ---------- Téléchargement des prix ----------
def prefetch_prices(tickers, start_date, end_date, batch_size=BATCH_SIZE):
    all_px = {}
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
        time.sleep(uniform(*CHUNK_SLEEP_RANGE))
    print(f"[INFO] Prefetch terminé: {len(all_px)}/{len(symbols)} tickers avec historique.")
    return all_px

# ---------- Equity ----------
def equity_from_daily_returns(daily_ret_pct: pd.Series) -> pd.DataFrame:
    if daily_ret_pct is None or daily_ret_pct.empty:
        return pd.DataFrame(columns=["date","equity"])
    daily_ret = daily_ret_pct.sort_index() / 100.0
    equity = (1.0 + daily_ret).cumprod() * 100.0
    return pd.DataFrame({"date": pd.to_datetime(equity.index), "equity": equity.values})

# ---------- Curation ----------
def load_curated_picks() -> pd.DataFrame:
    cols = ["date_signal","ticker","cohort","bucket","horizons","notes"]
    for p in CURATION_FILES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df = df.rename(columns={
                    "symbol": "ticker",
                    "ticker_yf": "ticker",
                    "date": "date_signal"
                })
                for c in cols:
                    if c not in df.columns:
                        df[c] = ""
                df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
                df["date_signal"] = pd.to_datetime(df["date_signal"], errors="coerce").dt.normalize()
                df["cohort"] = df["cohort"].fillna("").astype(str)
                df["bucket"] = df["bucket"].fillna("").astype(str)
                df["horizons"] = df["horizons"].fillna("").astype(str)
                df["notes"] = df["notes"].fillna("").astype(str)
                df = df[cols]
                df["source"] = "manual"
                print(f"[CURATE] loaded {len(df)} picks from {p}")
                return df
            except Exception as e:
                print(f"[CURATE][WARN] failed to read {p}: {e}")
    print("[CURATE] no user_trades.csv found")
    return pd.DataFrame(columns=["date_signal","ticker","cohort","bucket","horizons","notes","source"])

# ---------- Cohortes ----------
def _is_strong(s):
    if not isinstance(s, str):
        return False
    v = s.strip().upper()
    return v in {"STRONG BUY", "STRONGBUY", "STRONG_BUY"}

def add_pillars_and_cohorts(sig: pd.DataFrame) -> pd.DataFrame:
    s = sig.copy()
    tech_col = "technical_local"
    tv_col   = "tv_reco"
    an_col   = "analyst_bucket"
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
    s["votes_bin"] = pd.cut(s["votes"], bins=[-1,9,14,19,999], labels=["≤9","10–14","15–19","20+"])
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
    # === 1) Charger signaux AUTO ===
    sig, used = read_csv_first_available([
        "signals_history.csv",
        PUBLIC_DIR / "signals_history.csv"
    ])
    auto_ok = not sig.empty and all(c in sig.columns for c in ["date","ticker_yf"])
    if auto_ok:
        sig = sig.copy()
        sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
        sig["ticker_yf"] = sig["ticker_yf"].astype(str).str.upper()
        if "sector" not in sig.columns: sig["sector"] = "Unknown"
        if "bucket" not in sig.columns: sig["bucket"] = ""
        sig = add_pillars_and_cohorts(sig)
        if FILTER_BUCKET:
            sig = sig[sig["bucket"].astype(str).str.lower() == str(FILTER_BUCKET).lower()]
        sig["source"] = "auto"

    # === 2) Charger curation manuelle (Mes Picks) ===
    cur = load_curated_picks()  # columns: date_signal,ticker,cohort,bucket,horizons,notes,source=manual

    # === 3) Construire SEEDS séparés ===
    seeds_auto = pd.DataFrame(columns=["date_signal","ticker","cohort","bucket","horizons","source"])
    if auto_ok and not sig.empty:
        s = sig.rename(columns={"date": "date_signal", "ticker_yf": "ticker"})
        s["horizons"] = ""
        seeds_auto = s[["date_signal","ticker","cohort","bucket","horizons","source"]].copy()

    seeds_user = pd.DataFrame(columns=["date_signal","ticker","cohort","bucket","horizons","notes","source"])
    if not cur.empty:
        seeds_user = cur.copy()

    if seeds_auto.empty and seeds_user.empty:
        print("⚠️ Aucun seed (ni auto ni manuel).")
        placeholder_outputs_when_empty()
        return

    # Fenêtre prix -> couvrir auto+user
    seeds_all = pd.concat([seeds_auto, seeds_user], ignore_index=True) if not seeds_user.empty else seeds_auto.copy()
    seeds_all = seeds_all.dropna(subset=["ticker"])
    seeds_all["ticker"] = seeds_all["ticker"].astype(str).str.upper()

    dmin = seeds_all["date_signal"].dropna().min()
    if pd.isna(dmin):
        dmin = (pd.Timestamp.today().normalize() - pd.Timedelta(days=40))
    dmax = (seeds_all["date_signal"].dropna().max() if seeds_all["date_signal"].notna().any() else pd.Timestamp.today().normalize()) + pd.Timedelta(days=max(HORIZONS) + 5)

    tickers_unique = seeds_all["ticker"].unique().tolist()
    px_map = prefetch_prices(tickers_unique, dmin.strftime("%Y-%m-%d"), dmax.strftime("%Y-%m-%d"))

    # === 4) Construire TRADES (séparément pour auto et user) ===
    def build_trades(seeds, default_sector="Unknown"):
        out = []
        for ticker, g in seeds.groupby("ticker"):
            prices = px_map.get(ticker)
            if prices is None or prices.empty:
                continue
            for _, row in g.iterrows():
                d0 = row["date_signal"]
                idx = 0 if pd.isna(d0) else prices.index.searchsorted(d0)
                if idx >= len(prices.index):
                    continue
                d_entry = prices.index[idx]
                entry = float(prices.iloc[idx]["close"])
                if not np.isfinite(entry):
                    continue
                last_idx = len(prices.index) - 1
                # horizons personnalisés ?
                hlist = HORIZONS
                if row.get("horizons"):
                    try:
                        hlist = [int(x) for x in str(row["horizons"]).replace(",", "|").split("|") if x.strip().isdigit()]
                    except Exception:
                        hlist = HORIZONS
                for h in hlist:
                    exit_idx = idx + h
                    if exit_idx <= last_idx:
                        d_exit = prices.index[exit_idx]
                        exit_price = float(prices.iloc[exit_idx]["close"])
                        ret_pct = (exit_price / entry - 1.0) * 100.0
                        status = "closed"; days_elapsed = h
                    else:
                        d_exit = prices.index[last_idx]
                        exit_price = float(prices.iloc[last_idx]["close"])
                        ret_pct = (exit_price / entry - 1.0) * 100.0
                        status = "open"; days_elapsed = int(last_idx - idx)
                    out.append({
                        "date_signal": pd.to_datetime(d_entry).strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": row.get("sector", default_sector),
                        "bucket": row.get("bucket", ""),
                        "cohort": row.get("cohort", "P0_other"),
                        "votes": 0,  # on ne mélange pas ici
                        "horizon_days": int(h),
                        "status": status,
                        "days_elapsed": int(days_elapsed),
                        "entry": float(entry),
                        "exit": float(exit_price),
                        "date_exit": pd.to_datetime(d_exit).strftime("%Y-%m-%d"),
                        "ret_pct": float(ret_pct),
                        "notes": row.get("notes",""),
                        "source": row.get("source","auto"),
                    })
        return pd.DataFrame(out)

    # Auto trades
    auto_trades = pd.DataFrame()
    if not seeds_auto.empty:
        auto_trades = build_trades(seeds_auto)
    # User trades
    user_trades = pd.DataFrame()
    if not seeds_user.empty:
        user_trades = build_trades(seeds_user)

    # Dédup Mes Picks (si doublons exacts sur (date_signal,ticker,horizon))
    if not user_trades.empty:
        user_trades = (
            user_trades.sort_values(
                ["date_signal","ticker","horizon_days","days_elapsed"],
                ascending=[True, True, True, False]
            )
            .drop_duplicates(subset=["date_signal","ticker","horizon_days"], keep="first")
            .reset_index(drop=True)
        )

    # Concat pour audit complet
    trades_df = pd.concat([auto_trades, user_trades], ignore_index=True)
    if trades_df.empty:
        print("⚠️ Aucun trade généré.")
        placeholder_outputs_when_empty()
        return

    save_csv(trades_df, "backtest_trades.csv")

    # === 5) Résumé (CLOSED seulement) — sur AUTO uniquement pour coller au graphe auto
    auto_closed = auto_trades[auto_trades["status"] == "closed"].copy()
    summary = (
        auto_closed
        .groupby(["horizon_days","bucket","cohort"], dropna=False)
        .agg(
            n_trades   = ("ret_pct", "count"),
            winrate    = ("ret_pct", lambda s: float((s > 0).mean() * 100.0) if len(s) else 0.0),
            avg_ret    = ("ret_pct", lambda s: float(s.mean()) if len(s) else 0.0),
            median_ret = ("ret_pct", lambda s: float(s.median()) if len(s) else 0.0),
            p95_ret    = ("ret_pct", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
            p05_ret    = ("ret_pct", lambda s: float(s.quantile(0.05)) if len(s) else 0.0),
        )
        .reset_index()
        .sort_values(["horizon_days","cohort","bucket"])
    )
    save_csv(summary, "backtest_summary.csv")

    # === 6) Equity (AUTO only) par horizon + SPY + combo
    for h in HORIZONS:
        sub = auto_closed[auto_closed["horizon_days"] == h]
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

    # === 7) Equity AUTO par COHORTE (10j) ===
    if 10 in HORIZONS:
        for cohort in ["P3_confirmed", "P2_highconv", "P1_explore"]:
            subc = auto_closed[(auto_closed["horizon_days"] == 10) & (auto_closed["cohort"] == cohort)]
            if subc.empty:
                save_csv(pd.DataFrame(columns=["date","equity"]), f"backtest_equity_10d_{cohort}.csv")
                continue
            daily_ret = subc.groupby("date_exit")["ret_pct"].mean().sort_index()
            eqc = equity_from_daily_returns(daily_ret)
            save_csv(eqc, f"backtest_equity_10d_{cohort}.csv")

    # === 8) Copie de sûreté AUTO (10j) ===
    prefer = 10
    fallback = max(HORIZONS)
    src = f"backtest_equity_{prefer}d.csv" if prefer in HORIZONS else f"backtest_equity_{fallback}d.csv"
    eq10, _ = read_csv_first_available([src, PUBLIC_DIR / src])
    save_csv(eq10, "backtest_equity_10d.csv")

    # === 9) Courbe "Mes Picks" (10 jours, CLOSED only) ===
    if 10 in HORIZONS:
        user_closed = user_trades[user_trades["status"] == "closed"].copy()
        user10 = user_closed[user_closed["horizon_days"] == 10]
        if user10.empty:
            save_csv(pd.DataFrame(columns=["date","equity"]), "backtest_equity_10d_user.csv")
        else:
            daily_ret_user = user10.groupby("date_exit")["ret_pct"].mean().sort_index()
            eq_user = equity_from_daily_returns(daily_ret_user)
            save_csv(eq_user, "backtest_equity_10d_user.csv")
    else:
        save_csv(pd.DataFrame(columns=["date","equity"]), "backtest_equity_10d_user.csv")

    # === 10) Stats JSON ===
    stats = {
        "auto_trades": int(len(auto_trades)),
        "user_trades": int(len(user_trades)),
        "auto_closed": int((auto_trades["status"]=="closed").sum()),
        "user_closed": int((user_trades["status"]=="closed").sum()),
        "tickers_unique": int(seeds_all["ticker"].nunique()),
    }
    print("[STATS] backtest:", stats)
    with open("backtest_stats.json", "w") as f:
        json.dump(stats, f)
    with open(PUBLIC_DIR / "backtest_stats.json", "w") as f:
        json.dump(stats, f)

    print("[OK] Backtest écrit : auto (trades/summary/equity) + courbe Mes Picks 10j.")

if __name__ == "__main__":
    main()
