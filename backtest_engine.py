# backtest_engine.py — Moteur de backtest HONNÊTE, pur et testable.
#
# Principes (garde-fous CLAUDE.md) :
#   - Aucune I/O, aucun réseau : entrée = signaux datés + prix OHLC ; sortie = trades nets.
#   - Anti look-ahead : entrée au NEXT OPEN (jamais au close du jour de signal).
#   - Coûts explicites : frais + slippage par côté, paramétrables (env-overridable).
#   - NaN-safe : un prix manquant/incalculable -> trade ignoré (jamais un faux rendement).
#
# Ce module est volontairement séparé de backtest_signals.py pour être testé
# de façon déterministe (sur données synthétiques, sans réseau).
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# =========================== Paramètres ===============================

@dataclass(frozen=True)
class CostModel:
    """Coûts par CÔTÉ, en points de base (1 bp = 0.01%)."""
    fee_bps: float = 1.0        # commission par côté
    slippage_bps: float = 5.0   # slippage par côté (achat plus haut, vente plus bas)

    @property
    def fee(self) -> float:
        return self.fee_bps / 1e4

    @property
    def slip(self) -> float:
        return self.slippage_bps / 1e4


def cost_model_from_env() -> CostModel:
    return CostModel(
        fee_bps=float(os.getenv("BT_FEE_BPS", "1.0")),
        slippage_bps=float(os.getenv("BT_SLIPPAGE_BPS", "5.0")),
    )


def entry_mode_from_env() -> str:
    # "next_open" (défaut, honnête) | "close" (legacy, look-ahead)
    return os.getenv("BT_ENTRY_MODE", "next_open")


def horizons_from_env() -> list[int]:
    return [int(x) for x in os.getenv("BT_HORIZONS", "1,3,5,10,20").split(",") if x.strip().isdigit()]


# Colonnes des trades (compat dashboard : ret_pct = rendement NET).
TRADE_COLS = [
    "date_signal", "ticker", "sector", "bucket", "cohort", "votes",
    "horizon_days", "status", "days_elapsed", "entry", "exit", "date_exit",
    "ret_pct", "ret_gross_pct", "notes", "source",
]
SUMMARY_COLS = [
    "horizon_days", "bucket", "cohort", "n_trades",
    "winrate", "avg_ret", "median_ret", "p95_ret", "p05_ret",
]

# =========================== Simulation ===============================

def _px(prices: pd.DataFrame, pos: int, col: str) -> Optional[float]:
    """Prix à la position `pos` pour la colonne `col`, fallback sur close si absent/NaN."""
    if col in prices.columns:
        v = prices.iloc[pos][col]
        if pd.notna(v):
            return float(v)
    if "close" in prices.columns:
        v = prices.iloc[pos]["close"]
        if pd.notna(v):
            return float(v)
    return None


def simulate_trade(
    prices: pd.DataFrame,
    signal_date,
    horizon: int,
    entry_mode: str = "next_open",
    costs: Optional[CostModel] = None,
) -> Optional[dict]:
    """
    Simule un trade unique.
      prices : DataFrame indexé par DatetimeIndex trié, colonnes >= {'open','close'} (open optionnel).
      signal_date : jour où le signal est OBSERVÉ (info dispo à la clôture de ce jour).
    Retour None si l'entrée est impossible (pas de barre d'entrée disponible = pas de look-ahead).
    """
    if prices is None or prices.empty or "close" not in prices.columns:
        return None
    costs = costs or CostModel()
    idx = prices.index
    last = len(idx) - 1

    # Dernière barre dont la date <= signal_date (gère les dates non ouvrées / week-ends).
    pos = int(idx.searchsorted(pd.Timestamp(signal_date), side="right")) - 1
    if pos < 0:
        return None

    if entry_mode == "close":
        entry_pos = pos
        entry_raw = _px(prices, entry_pos, "close")   # legacy : close du jour de signal (look-ahead)
    else:  # next_open (défaut, honnête)
        entry_pos = pos + 1
        if entry_pos > last:
            return None  # pas encore de barre d'entrée -> on n'invente rien
        entry_raw = _px(prices, entry_pos, "open")
    if entry_raw is None or not np.isfinite(entry_raw) or entry_raw <= 0:
        return None

    exit_pos = entry_pos + int(horizon)
    if exit_pos <= last:
        status, ex_pos = "closed", exit_pos
    else:
        status, ex_pos = "open", last   # fenêtre non écoulée -> exclu des métriques closed
    exit_raw = _px(prices, ex_pos, "close")
    if exit_raw is None or not np.isfinite(exit_raw):
        return None

    # Coûts par côté : on achète plus haut, on vend plus bas, commission des deux côtés.
    buy = entry_raw * (1.0 + costs.slip) * (1.0 + costs.fee)
    sell = exit_raw * (1.0 - costs.slip) * (1.0 - costs.fee)
    ret_net = (sell / buy - 1.0) * 100.0
    ret_gross = (exit_raw / entry_raw - 1.0) * 100.0

    return {
        "entry_date": idx[entry_pos], "entry": float(entry_raw),
        "exit_date": idx[ex_pos], "exit": float(exit_raw),
        "horizon_days": int(horizon), "status": status,
        "days_held": int(ex_pos - entry_pos),
        "ret_gross_pct": float(ret_gross), "ret_net_pct": float(ret_net),
    }


def build_trades(
    signals: pd.DataFrame,
    price_map: dict,
    horizons: list[int],
    entry_mode: str = "next_open",
    costs: Optional[CostModel] = None,
) -> pd.DataFrame:
    """
    signals : colonnes requises {date_signal, ticker} ; optionnelles {sector,bucket,cohort,votes,notes,source}.
    price_map : {ticker -> DataFrame OHLC}.
    """
    costs = costs or CostModel()
    if signals is None or signals.empty:
        return pd.DataFrame(columns=TRADE_COLS)
    rows = []
    for _, s in signals.iterrows():
        t = str(s["ticker"])
        prices = price_map.get(t)
        if prices is None or getattr(prices, "empty", True):
            continue
        for h in horizons:
            r = simulate_trade(prices, s["date_signal"], h, entry_mode=entry_mode, costs=costs)
            if r is None:
                continue
            rows.append({
                "date_signal": pd.to_datetime(s["date_signal"]).strftime("%Y-%m-%d"),
                "ticker": t,
                "sector": s.get("sector", "Unknown") if hasattr(s, "get") else "Unknown",
                "bucket": s.get("bucket", "") if hasattr(s, "get") else "",
                "cohort": s.get("cohort", "P0_other") if hasattr(s, "get") else "P0_other",
                "votes": int(pd.to_numeric(s.get("votes", 0), errors="coerce") or 0) if hasattr(s, "get") else 0,
                "horizon_days": r["horizon_days"],
                "status": r["status"],
                "days_elapsed": r["days_held"],
                "entry": r["entry"], "exit": r["exit"],
                "date_exit": pd.to_datetime(r["exit_date"]).strftime("%Y-%m-%d"),
                "ret_pct": r["ret_net_pct"],          # NET (honnête) — colonne lue par le dashboard
                "ret_gross_pct": r["ret_gross_pct"],  # transparence (avant coûts)
                "notes": s.get("notes", "") if hasattr(s, "get") else "",
                "source": s.get("source", "auto") if hasattr(s, "get") else "auto",
            })
    return pd.DataFrame(rows, columns=TRADE_COLS)


# =========================== Agrégats ===============================

def summarize(trades: pd.DataFrame, by=("horizon_days", "bucket", "cohort"), ret_col: str = "ret_pct") -> pd.DataFrame:
    """Métriques par groupe, sur les trades CLOSED uniquement."""
    by = list(by)
    if trades is None or trades.empty:
        return pd.DataFrame(columns=[*by, "n_trades", "winrate", "avg_ret", "median_ret", "p95_ret", "p05_ret"])
    closed = trades[trades["status"] == "closed"]
    if closed.empty:
        return pd.DataFrame(columns=[*by, "n_trades", "winrate", "avg_ret", "median_ret", "p95_ret", "p05_ret"])
    g = closed.groupby(by, dropna=False)[ret_col]
    out = g.agg(
        n_trades="count",
        winrate=lambda s: float((s > 0).mean() * 100.0),
        avg_ret="mean",
        median_ret="median",
        p95_ret=lambda s: float(s.quantile(0.95)),
        p05_ret=lambda s: float(s.quantile(0.05)),
    ).reset_index()
    return out.sort_values(by).reset_index(drop=True)


def equity_curve(trades: pd.DataFrame, horizon: int, calendar=None, ret_col: str = "ret_pct") -> pd.DataFrame:
    """
    Equity base 100 d'un panier ÉQUIPONDÉRÉ rééquilibré quotidiennement.

    Chaque jour on est investi à parts égales dans toutes les positions OUVERTES ce jour-là ;
    le rendement du jour = moyenne des taux quotidiens géométriques des positions actives
    (0 = cash s'il n'y a aucune position). Chaque trade compose EXACTEMENT son propre rendement
    net sur ses jours actifs ((1+g)^n = 1+R) -> pas d'inflation par compoundage de rendements
    multi-jours à fréquence journalière (bug de l'ancienne courbe).

    calendar : DatetimeIndex des jours de bourse (défaut : bdate_range des trades).
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date", "equity"])
    sub = trades[(trades["status"] == "closed") & (trades["horizon_days"] == horizon)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "equity"])
    sub["entry_d"] = pd.to_datetime(sub["date_signal"])   # entrée réelle = next open (jour suivant)
    sub["exit_d"] = pd.to_datetime(sub["date_exit"])
    if calendar is not None and len(calendar):
        cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(calendar).unique())))
    else:
        cal = pd.bdate_range(sub["entry_d"].min(), sub["exit_d"].max())

    from collections import defaultdict
    contrib = defaultdict(list)
    for _, t in sub.iterrows():
        active = cal[(cal > t["entry_d"]) & (cal <= t["exit_d"])]  # jours détenus (post next-open)
        n = len(active)
        if n == 0:
            continue
        g = (1.0 + float(t[ret_col]) / 100.0) ** (1.0 / n) - 1.0
        for d in active:
            contrib[d].append(g)
    if not contrib:
        return pd.DataFrame(columns=["date", "equity"])
    lo, hi = min(contrib), max(contrib)
    span = cal[(cal >= lo) & (cal <= hi)]
    daily = np.array([float(np.mean(contrib[d])) if d in contrib else 0.0 for d in span])
    eq = np.cumprod(1.0 + daily) * 100.0
    return pd.DataFrame({"date": pd.to_datetime(span), "equity": eq.astype(float)})


def daily_marked_equity(trades: pd.DataFrame, price_map: dict, calendar) -> pd.DataFrame:
    """
    Equity base-100 marquée aux PRIX RÉELS quotidiens (close-to-close), entrée à l'open.
    Contrairement à equity_curve (qui lisse le rendement d'un trade sur ses jours -> Sharpe/DD
    faussés), celle-ci utilise la vraie trajectoire journalière -> volatilité, Sharpe et drawdown
    HONNÊTES. Portefeuille équipondéré : chaque jour = moyenne des rendements réels des positions
    ouvertes (0 = cash). Brut de frais (les coûts ~bps sont négligeables pour la forme/le risque).
    """
    empty = pd.DataFrame(columns=["date", "equity"])
    if trades is None or trades.empty:
        return empty
    closed = trades[trades["status"] == "closed"]
    if closed.empty:
        return empty
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(calendar).unique())))
    from collections import defaultdict
    contrib = defaultdict(list)
    for _, t in closed.iterrows():
        px = price_map.get(str(t["ticker"]))
        if px is None or px.empty:
            continue
        entry_d = pd.Timestamp(t["date_signal"])
        exit_d = pd.Timestamp(t["date_exit"])
        held = px.loc[(px.index > entry_d) & (px.index <= exit_d)]
        if held.empty:
            continue
        closes = pd.to_numeric(held["close"], errors="coerce").astype(float).values
        opens = (pd.to_numeric(held["open"], errors="coerce").astype(float).values
                 if "open" in held.columns else closes)
        idxd = held.index
        for i in range(len(closes)):
            if i == 0:
                base = opens[0] if (np.isfinite(opens[0]) and opens[0] > 0) else closes[0]
                r = closes[0] / base - 1.0
            else:
                r = closes[i] / closes[i - 1] - 1.0
            if np.isfinite(r):
                contrib[idxd[i]].append(r)
    if not contrib:
        return empty
    lo, hi = min(contrib), max(contrib)
    span = cal[(cal >= lo) & (cal <= hi)]
    daily = np.array([float(np.mean(contrib[d])) if d in contrib else 0.0 for d in span])
    eq = np.cumprod(1.0 + daily) * 100.0
    return pd.DataFrame({"date": pd.to_datetime(span), "equity": eq.astype(float)})


def generate_pointintime_signals(
    price_map: dict,
    spy: Optional[pd.DataFrame],
    dates,
    score_fn,
    regime_fn=None,
    target_confirmed: float = 0.02,
    target_presignal: float = 0.08,
    min_history: int = 210,
) -> pd.DataFrame:
    """
    Génère des signaux POINT-IN-TIME (walk-forward) — le cœur du backtest honnête.

    Pour chaque date d de `dates`, on n'utilise QUE les barres <= d (anti look-ahead) :
      - regime = regime_fn(spy_slice<=d)
      - pour chaque ticker : base = score_fn(px_slice<=d, spy_slice<=d) ; tv = clamp(base*regime)
      - buckets par top-% cross-sectionnel (mime les seuils adaptatifs ~2% / ~8%).

    score_fn et regime_fn sont INJECTÉS (en prod : le screener corrigé) pour rester testable.
    Retour : DataFrame {date_signal, ticker, bucket, tv_score, cohort}.
    """
    out_cols = ["date_signal", "ticker", "bucket", "tv_score", "cohort"]
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        spy_slice = None
        if spy is not None and not spy.empty:
            spy_slice = spy.loc[spy.index <= d]
        regime = regime_fn(spy_slice) if regime_fn is not None else 1.0
        if regime is None or not np.isfinite(regime):
            regime = 1.0

        scored = []
        for t, px in price_map.items():
            if px is None or px.empty:
                continue
            sl = px.loc[px.index <= d]        # <-- garde-fou anti look-ahead
            if len(sl) < min_history:
                continue
            base = score_fn(sl, spy_slice)
            if base is None or not np.isfinite(base):
                continue
            tv = base * regime
            tv = 0.0 if not np.isfinite(tv) else float(max(0.0, min(1.0, tv)))
            scored.append((t, tv))

        if not scored:
            continue
        sdf = pd.DataFrame(scored, columns=["ticker", "tv_score"]).sort_values(
            "tv_score", ascending=False).reset_index(drop=True)
        n = len(sdf)
        n_conf = max(1, int(round(n * target_confirmed)))
        n_pre = max(0, int(round(n * target_presignal)))
        bucket = np.array(["none"] * n, dtype=object)
        bucket[:n_conf] = "confirmed"
        bucket[n_conf:n_conf + n_pre] = "pre_signal"
        sdf["bucket"] = bucket
        sig = sdf[sdf["bucket"].isin(["confirmed", "pre_signal"])].copy()
        sig["date_signal"] = d.strftime("%Y-%m-%d")
        sig["cohort"] = np.where(sig["bucket"] == "confirmed", "P3_confirmed", "P1_explore")
        rows.append(sig[out_cols])

    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.concat(rows, ignore_index=True)


def verdict_vs_spy(model_eq: pd.DataFrame, spy_eq: pd.DataFrame) -> dict:
    """Rendement total du modèle vs SPY sur la fenêtre, + verdict 'bat SPY : oui/non'."""
    def total(e):
        if e is None or e.empty or "equity" not in e.columns or len(e) < 1:
            return float("nan")
        base = float(e["equity"].iloc[0])
        if not np.isfinite(base) or base == 0:
            return float("nan")
        return (float(e["equity"].iloc[-1]) / base - 1.0) * 100.0
    m, s = total(model_eq), total(spy_eq)
    excess = (m - s) if (np.isfinite(m) and np.isfinite(s)) else float("nan")
    return {
        "model_total_ret_pct": m,
        "spy_total_ret_pct": s,
        "excess_pct": excess,
        "beats_spy": bool(np.isfinite(excess) and excess > 0),
    }
