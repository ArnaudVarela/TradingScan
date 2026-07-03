# track_performance.py — Suivi de performance PAR TICKER depuis sa 1re détection (thématique + marché large).
# Enrichit thematic_setups.csv & market_setups.csv avec : first_scan_date, first_scan_price, pct_since_first, days_tracked.
# Registre persistant perf_tracker.csv : la 1re détection (date+prix) est IMMUABLE par (univers, ticker).
# Le thématique est bootstrappé depuis l'historique signals_log (prix/date réels) -> % dès le 1er jour ; le marché part d'aujourd'hui.
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRACKER = ROOT / "perf_tracker.csv"
SIGNALS_LOG = ROOT / "signals_log.csv"
PUB = ROOT / "dashboard" / "public"
REG_COLS = ["universe", "ticker", "first_date", "first_price", "first_score"]

def _today_et():
    return (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).date().isoformat()

def _load_registry():
    """Absent -> registre vide (démarrage). Présent mais illisible/incomplet -> on ARRÊTE (RuntimeError)
    pour NE JAMAIS écraser silencieusement les 1res détections immuables (l'étape CI est non bloquante :
    lever ici laisse le perf_tracker.csv committé intact au lieu de le re-stamper avec first_date=today)."""
    if not TRACKER.exists():
        return pd.DataFrame(columns=REG_COLS)
    try:
        r = pd.read_csv(TRACKER, keep_default_na=False)
    except Exception as e:
        raise RuntimeError(f"perf_tracker.csv présent mais illisible ({e}) — arrêt pour protéger les 1res détections") from e
    missing = {"universe", "ticker", "first_date", "first_price"} - set(r.columns)
    if missing:
        raise RuntimeError(f"perf_tracker.csv incomplet, colonnes critiques manquantes {missing} — arrêt")
    for c in REG_COLS:                       # backfill des colonnes NON critiques (ex. first_score) après évolution de schéma
        if c not in r.columns:
            r[c] = ""
    r["ticker"] = r["ticker"].astype(str).str.upper()
    return r[REG_COLS]

def _bootstrap_thematic(reg):
    """1re détection thématique depuis signals_log (prix/date réels), pour les tickers pas encore en registre."""
    if not SIGNALS_LOG.exists():
        return reg
    try:
        s = pd.read_csv(SIGNALS_LOG, keep_default_na=False)
    except Exception:
        return reg
    if s.empty or not {"date", "ticker", "price"}.issubset(s.columns):
        return reg
    s["ticker"] = s["ticker"].astype(str).str.upper()
    first = s.loc[s.groupby("ticker")["date"].idxmin()]   # ligne à la date la + ancienne par ticker
    have = set(reg[reg["universe"] == "thematic"]["ticker"])
    add = [{"universe": "thematic", "ticker": r["ticker"], "first_date": r["date"],
            "first_price": r["price"], "first_score": r.get("score", "")}
           for _, r in first.iterrows() if r["ticker"] not in have]
    return pd.concat([reg, pd.DataFrame(add)], ignore_index=True) if add else reg

def _register_new(reg, setups_path, universe, today):
    """Ajoute au registre les tickers du scan du jour absents (1re détection = aujourd'hui)."""
    if not setups_path.exists():
        return reg
    try:
        d = pd.read_csv(setups_path, keep_default_na=False)
    except Exception:
        return reg
    if d.empty or "ticker" not in d.columns:
        return reg
    have = set(reg[reg["universe"] == universe]["ticker"])
    add = []
    for _, r in d.iterrows():
        t = str(r["ticker"]).upper()
        if not t or t in have:
            continue
        add.append({"universe": universe, "ticker": t, "first_date": today,
                    "first_price": r.get("price", ""), "first_score": r.get("score", "")})
        have.add(t)
    return pd.concat([reg, pd.DataFrame(add)], ignore_index=True) if add else reg

def _enrich(setups_path, universe, reg, today):
    if not setups_path.exists():
        return
    try:
        d = pd.read_csv(setups_path, keep_default_na=False)
    except Exception:
        return
    if d.empty or "ticker" not in d.columns:
        return
    sub = reg[reg["universe"] == universe]
    fp = dict(zip(sub["ticker"], pd.to_numeric(sub["first_price"], errors="coerce")))
    fd = dict(zip(sub["ticker"], sub["first_date"].astype(str)))
    tks = d["ticker"].astype(str).str.upper()
    cur = pd.to_numeric(d["price"], errors="coerce") if "price" in d.columns else pd.Series(np.nan, index=d.index)
    d["first_scan_date"] = tks.map(fd).fillna("")
    fpx = tks.map(fp)
    d["first_scan_price"] = fpx.round(2)
    d["pct_since_first"] = (((cur / fpx) - 1) * 100).replace([np.inf, -np.inf], np.nan).round(2)
    def _days(fdate):
        try:
            return (datetime.date.fromisoformat(today) - datetime.date.fromisoformat(str(fdate))).days
        except Exception:
            return ""
    d["days_tracked"] = d["first_scan_date"].map(_days)
    d.to_csv(setups_path, index=False)
    if PUB.exists():
        d.to_csv(PUB / setups_path.name, index=False)
    tracked = int((pd.to_numeric(d["days_tracked"], errors="coerce") > 0).sum())
    print(f"[PERF] {universe}: {len(d)} setups enrichis ({tracked} avec >=1j d'historique)")

def main():
    today = _today_et()
    reg = _load_registry()
    reg = _bootstrap_thematic(reg)
    reg = _register_new(reg, ROOT / "thematic_setups.csv", "thematic", today)
    reg = _register_new(reg, ROOT / "market_setups.csv", "market", today)
    reg["ticker"] = reg["ticker"].astype(str).str.upper()
    reg = reg.drop_duplicates(subset=["universe", "ticker"], keep="first")[REG_COLS]
    reg.to_csv(TRACKER, index=False)
    if PUB.exists():
        reg.to_csv(PUB / "perf_tracker.csv", index=False)
    _enrich(ROOT / "thematic_setups.csv", "thematic", reg, today)
    _enrich(ROOT / "market_setups.csv", "market", reg, today)
    print(f"[PERF] registre perf_tracker.csv : {len(reg)} tickers suivis "
          f"({(reg['universe']=='thematic').sum()} thématiques, {(reg['universe']=='market').sum()} marché)")

if __name__ == "__main__":
    main()
