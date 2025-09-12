# mix_ab_screen_indices.py — Robust + MCAP OFF (root outputs)
from __future__ import annotations

import os, math, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

UNIVERSE_CSV   = ROOT / "universe_in_scope.csv"   # produit par build_universe.py
SECTOR_CATALOG = ROOT / "sector_catalog.csv"      # optionnel

OUT_DIR = ROOT  # sorties à la racine

# --------- Paramètres (env overridable)
DISABLE_MCAP = int(os.getenv("DISABLE_MCAP", "1"))  # 1 = pas de SEC/YF cap, pas de filtre mcap
MCAP_CAP = float(os.getenv("MCAP_CAP", 75e9))       # ignoré si DISABLE_MCAP=1
OHLCV_WINDOW_DAYS = int(os.getenv("OHLCV_WINDOW_DAYS", 240))    # historique (>=200 pour EMA200)
AVG_DOLLAR_VOL_LOOKBACK = int(os.getenv("AVG_DOLLAR_VOL_LB", 20))
CACHE_FRESH_DAYS = int(os.getenv("CACHE_FRESH_DAYS", 2))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 60))                   # batch yf
SLEEP_BETWEEN_BATCHES = float(os.getenv("SLEEP_BETWEEN_BATCHES", "0.0"))

YF_INTERVAL = "1d"
YF_THREADS  = False  # on pilote nos propres batches

# =========================== utils & io ============================

def _log(msg: str) -> None:
    print(msg, flush=True)

def _now_ts() -> float:
    return time.time()

def _is_recent(ts: float, fresh_days: int = CACHE_FRESH_DAYS) -> bool:
    return (_now_ts() - ts) < fresh_days * 86400

def _csv_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"ohlcv_{safe}.csv"

def _csv_cache_meta_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"ohlcv_{safe}.meta.json"

def _save_csv_cache(ticker: str, df: pd.DataFrame) -> None:
    # Attend un index datetime (name='date') + colonnes open/high/low/close/volume
    if df is None or df.empty:
        return
    out = df.copy()
    # Sécurise noms
    out = out.rename(columns={
        "Open":"open","High":"high","Low":"low",
        "Close":"close","Adj Close":"adj_close","Volume":"volume"
    })
    needed = ["open","high","low","close","volume"]
    for c in needed:
        if c not in out.columns:
            return  # on ne cache que si complet
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[needed].dropna()
    if out.empty:
        return
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out.index.name = "date"

    # Écriture atomique
    p = _csv_cache_path(ticker)
    tmp = p.with_suffix(".csv.tmp")
    out.reset_index()[["date"] + needed].to_csv(tmp, index=False)
    Path(tmp).replace(p)

    meta = {"ts": _now_ts(), "rows": int(len(out))}
    mp = _csv_cache_meta_path(ticker)
    mp.write_text(json.dumps(meta), encoding="utf-8")

def _load_csv_cache(ticker: str) -> Optional[pd.DataFrame]:
    p = _csv_cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty or "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        for c in ["open","high","low","close","volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["open","high","low","close","volume"]].dropna()
        return df if not df.empty else None
    except Exception:
        return None

def _cache_is_fresh(ticker: str) -> bool:
    mp = _csv_cache_meta_path(ticker)
    if not mp.exists():
        return False
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        ts = float(meta.get("ts", 0))
        return _is_recent(ts)
    except Exception:
        return False

def _read_sector_catalog() -> Dict[str, Dict[str,str]]:
    if not SECTOR_CATALOG.exists():
        return {}
    try:
        df = pd.read_csv(SECTOR_CATALOG)
    except Exception as e:
        _log(f"[WARN] sector_catalog read failed: {e}")
        return {}
    res: Dict[str, Dict[str,str]] = {}
    for _, r in df.iterrows():
        t = str(r.get("ticker") or r.get("ticker_yf") or "").upper().strip()
        if not t:
            continue
        res[t] = {
            "sector":   (r.get("sector")   if pd.notna(r.get("sector"))   else "Unknown"),
            "industry": (r.get("industry") if pd.notna(r.get("industry")) else "Unknown"),
        }
    return res

def _ensure_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise SystemExit(f"[FATAL] {UNIVERSE_CSV} introuvable. Lance d'abord build_universe.py")
    df = pd.read_csv(UNIVERSE_CSV)
    col = None
    for c in ["ticker_yf","ticker","Symbol","symbol"]:
        if c in df.columns:
            col = c
            break
    if not col:
        raise SystemExit("[FATAL] universe_in_scope.csv: colonne ticker introuvable")

    uni = pd.DataFrame({"ticker_yf": df[col].astype(str).str.strip().str.upper()})
    uni = uni[uni["ticker_yf"].ne("") & uni["ticker_yf"].ne("-")]
    uni = uni[uni["ticker_yf"].str.match(r"^[A-Z0-9.\-]+$")]
    uni = uni.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)
    _log(f"[UNI] in-scope: {len(uni)}")
    return uni

# ====================== yfinance batch loader ======================

def _tickers_needing_fetch(tickers: List[str]) -> List[str]:
    need = []
    for t in tickers:
        if not _cache_is_fresh(t) or _load_csv_cache(t) is None:
            need.append(t)
    return need

def _normalize_yf_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    # Cas MultiIndex (ticker, field) ou (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        return df  # l’appelant découpera
    # Single index (1 ticker)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    if not keep:
        return None
    out = df[keep].dropna()
    if out.empty:
        return None
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out.index.name = "date"
    return out

def _yf_download_batch(tickers: List[str], period_days: int) -> None:
    if not tickers:
        return
    period = f"{max(period_days, 220)}d"  # >=200 pour EMA200
    try:
        data = yf.download(
            tickers=tickers,                # list, pas string
            period=period,
            interval=YF_INTERVAL,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=YF_THREADS,
        )
        # MultiIndex attendu : level0=ticker OU field
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            for t in tickers:
                sub = None
                try:
                    if t in level0:
                        sub = data[t]
                    else:
                        # autre forme (field, ticker)
                        sub = data.xs(t, axis=1, level=1, drop_level=False)
                        fields = ["Open","High","Low","Close","Volume"]
                        sub = sub.copy()
                        sub.columns = [c[0] for c in sub.columns]  # 'Open' etc.
                        sub = sub[fields]
                except Exception:
                    sub = None
                if sub is None or sub.empty:
                    continue
                sub = sub.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                keep = [c for c in ["open","high","low","close","volume"] if c in sub.columns]
                out = sub[keep].dropna()
                if not out.empty:
                    out.index = pd.to_datetime(out.index, errors="coerce")
                    out = out[~out.index.isna()].sort_index()
                    out.index.name = "date"
                    _save_csv_cache(t, out)
        else:
            # Single-ticker retourné (rare quand tickers==1)
            out = _normalize_yf_df(data)
            if out is not None:
                _save_csv_cache(tickers[0], out)
    except Exception as e:
        _log(f"[YF] batch fetch failed ({len(tickers)}): {e} — fallback per-ticker")
        # Fallback par ticker (plus lent mais robuste)
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period=period, interval=YF_INTERVAL, auto_adjust=False)
                out = _normalize_yf_df(hist)
                if out is not None:
                    _save_csv_cache(t, out)
            except Exception:
                continue

def _ensure_ohlcv_cached(tickers: List[str], period_days: int = OHLCV_WINDOW_DAYS) -> None:
    need = _tickers_needing_fetch(tickers)
    if not need:
        return
    for i in range(0, len(need), BATCH_SIZE):
        chunk = need[i:i+BATCH_SIZE]
        _log(f"[YF] fetching batch {i//BATCH_SIZE+1}/{math.ceil(len(need)/BATCH_SIZE)} size={len(chunk)}")
        _yf_download_batch(chunk, period_days)
        if SLEEP_BETWEEN_BATCHES > 0:
            time.sleep(SLEEP_BETWEEN_BATCHES)

def _get_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    return _load_csv_cache(ticker)

# ====================== indicators & scores ========================

def _ema(s: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_tech_label(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Renvoie (label, score) – STRONG_BUY / BUY / HOLD / SELL
    Plus strict : exige >=200 barres pour éviter NaN sur EMA200.
    """
    if df is None or df.empty or "close" not in df.columns:
        return "HOLD", 0.5
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) < 200:
        return "HOLD", 0.5

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    macd, macds, hist = _macd(close)
    rsi14 = _rsi(close, 14)

    def last_ok(s: pd.Series) -> Optional[float]:
        v = pd.to_numeric(s, errors="coerce").dropna()
        return float(v.iloc[-1]) if len(v) else None

    e20, e50, e200 = last_ok(ema20), last_ok(ema50), last_ok(ema200)
    macd_l, macd_s, h = last_ok(macd), last_ok(macds), last_ok(hist)
    rsi_l = last_ok(rsi14)

    if None in (e20, e50, e200, macd_l, macd_s, h, rsi_l):
        return "HOLD", 0.5

    bull_trend = (e20 > e50 > e200)
    macd_pos = (macd_l > macd_s) and (h > 0)
    rsi_ok = 45 <= rsi_l <= 70

    score = 0.0
    score += 0.45 if bull_trend else 0.0
    score += 0.35 if macd_pos else 0.0
    score += 0.20 if rsi_ok else 0.0

    if score >= 0.80:
        label = "STRONG_BUY"
    elif score >= 0.60:
        label = "BUY"
    elif score >= 0.40:
        label = "HOLD"
    else:
        label = "SELL"

    return label, float(round(score, 4))

def _last_close(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty or "close" not in df.columns:
        return None
    v = pd.to_numeric(df["close"], errors="coerce").dropna()
    return float(v.iloc[-1]) if len(v) else None

def _avg_dollar_vol(df: Optional[pd.DataFrame], lb: int = AVG_DOLLAR_VOL_LOOKBACK) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "close" not in df.columns or "volume" not in df.columns:
        return None
    sub = df.tail(lb).copy()
    if sub.empty:
        return None
    c = pd.to_numeric(sub["close"], errors="coerce")
    v = pd.to_numeric(sub["volume"], errors="coerce")
    prod = (c * v).dropna()
    return float(prod.mean()) if len(prod) else None

# =========================== pipeline ==============================

def _build_rows(uni: pd.DataFrame) -> pd.DataFrame:
    tickers = uni["ticker_yf"].tolist()

    # 1) OHLCV cache upfront (batch)
    _ensure_ohlcv_cached(tickers, OHLCV_WINDOW_DAYS)

    # 2) Sector catalog (optionnel)
    sectors = _read_sector_catalog()

    rows = []
    price_map: Dict[str, Optional[float]] = {}
    adv_map: Dict[str, Optional[float]]   = {}

    # 3) prix/ADV + tech
    for t in tickers:
        df = _get_ohlcv(t)
        lc = _last_close(df)
        adv = _avg_dollar_vol(df) or 0.0
        price_map[t] = lc
        adv_map[t] = adv

        if df is not None and lc is not None and len(df) >= 200 and lc >= 0.5:
            tv_reco, tv_score = compute_tech_label(df)
        else:
            tv_reco, tv_score = "HOLD", 0.5

        secinfo = sectors.get(t, {})
        sector = secinfo.get("sector", "Unknown")
        industry = secinfo.get("industry", "Unknown")

        analyst_bucket = "HOLD"   # placeholders
        analyst_votes  = 20.0

        def favorable(label: str) -> bool:
            return label in ("BUY","STRONG_BUY")

        p_tech = favorable(tv_reco)
        p_tv   = favorable(tv_reco)
        p_an   = favorable(analyst_bucket)

        pillars_met = int(p_tech) + int(p_tv) + int(p_an)
        if pillars_met == 3 and tv_reco == "STRONG_BUY":
            bucket = "confirmed"
        elif pillars_met >= 2:
            bucket = "pre_signal"
        else:
            bucket = "event"

        adv_weight = math.log1p(float(max(0.0, adv))) / 20.0
        rank_score = float(
            (pillars_met / 3.0) * 0.6 +
            (tv_score) * 0.3 +
            min(0.1, adv_weight)
        )

        rows.append({
            "ticker_yf": t,
            "ticker_tv": t,
            "price": lc,
            "last": lc,
            "mcap_usd_final": np.nan,  # MCAP OFF
            "mcap": np.nan,
            "avg_dollar_vol": adv,
            "tv_score": tv_score,
            "tv_reco": tv_reco,
            "analyst_bucket": analyst_bucket,
            "analyst_votes": analyst_votes,
            "sector": sector,
            "industry": industry,
            "p_tech": bool(p_tech),
            "p_tv": bool(p_tv),
            "p_an": bool(p_an),
            "pillars_met": int(pillars_met),
            "votes_bin": "10-20",
            "rank_score": rank_score,
            "bucket": bucket,
        })

    base = pd.DataFrame(rows)

    # 4) MCAP désactivé → rien d'autre à faire
    return base

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if DISABLE_MCAP:
        return df.copy()
    # (non utilisé si DISABLE_MCAP=1)
    m = pd.to_numeric(df["mcap_usd_final"], errors="coerce")
    mask = m.notna() & np.isfinite(m) & (m < MCAP_CAP)
    return df.loc[mask].copy()

def _safe_write_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    Path(tmp).replace(path)

def _write_outputs(out: pd.DataFrame) -> None:
    out_sorted = out.sort_values(["rank_score","avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)
    _safe_write_csv(OUT_DIR / "candidates_all_ranked.csv", out_sorted)
    _log(f"[SAVE] candidates_all_ranked.csv | rows={len(out_sorted)} | cols={out_sorted.shape[1]}")

    confirmed = out_sorted[out_sorted["bucket"]=="confirmed"]
    pre_sig   = out_sorted[out_sorted["bucket"]=="pre_signal"]
    event     = out_sorted[out_sorted["bucket"]=="event"]

    _safe_write_csv(OUT_DIR / "confirmed_STRONGBUY.csv", confirmed)
    _safe_write_csv(OUT_DIR / "anticipative_pre_signals.csv", pre_sig)
    _safe_write_csv(OUT_DIR / "event_driven_signals.csv", event)
    _log(f"[SAVE] confirmed={len(confirmed)} pre={len(pre_sig)} event={len(event)}")

def _update_signals_history(today: str, out: pd.DataFrame) -> None:
    path = OUT_DIR / "signals_history.csv"
    cols = ["date","ticker_yf","sector","bucket","tv_reco","analyst_bucket"]
    new = out[["ticker_yf","sector","bucket","tv_reco","analyst_bucket"]].copy()
    new.insert(0, "date", today)
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=cols)
        hist = pd.concat([old, new], ignore_index=True)
    else:
        hist = new
    hist = hist.tail(100)
    _safe_write_csv(path, hist)
    _log(f"[SAVE] signals_history.csv | rows={len(hist)}")

def main():
    _log("[STEP] mix_ab_screen_indices starting…")
    pd.set_option("future.no_silent_downcasting", True)

    uni = _ensure_universe()
    tickers = uni["ticker_yf"].tolist()

    # Pré-cache OHLCV
    _ensure_ohlcv_cached(tickers, OHLCV_WINDOW_DAYS)

    base = _build_rows(uni)

    # Types
    float_cols = ["price","last","mcap_usd_final","avg_dollar_vol","tv_score","analyst_votes","rank_score"]
    for c in float_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # Filtre (no-op si MCAP OFF)
    filtered = _apply_filters(base)

    # Sorties
    _write_outputs(filtered)

    # Historique
    today = pd.Timestamp.utcnow().date().isoformat()
    _update_signals_history(today, filtered)

    coverage = {
        "universe": int(len(uni)),
        "post_filter": int(len(filtered)),
        "price_nonnull": int(base["price"].notna().sum()),
        "mcap_final_nonnull": int(base["mcap_usd_final"].notna().sum()),
        "confirmed_count": int((filtered["bucket"]=="confirmed").sum()),
        "pre_count": int((filtered["bucket"]=="pre_signal").sum()),
        "event_count": int((filtered["bucket"]=="event").sum()),
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    main()
