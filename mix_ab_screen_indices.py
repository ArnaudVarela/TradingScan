# mix_ab_screen_indices.py
from __future__ import annotations

import os, math, json, time, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# --- Local deps (assumés présents)
from sec_helpers import sec_ticker_map, sec_latest_shares_outstanding

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

UNIVERSE_CSV   = ROOT / "universe_in_scope.csv"   # produit par build_universe.py
SECTOR_CATALOG = ROOT / "sector_catalog.csv"      # optionnel

OUT_DIR = ROOT  # sorties à la racine

# --------- Paramètres (env overridable)
MCAP_CAP = float(os.getenv("MCAP_CAP", 75e9))                   # filtre strict < 75B
OHLCV_WINDOW_DAYS = int(os.getenv("OHLCV_WINDOW_DAYS", 240))    # historique
AVG_DOLLAR_VOL_LOOKBACK = int(os.getenv("AVG_DOLLAR_VOL_LB", 20))
CACHE_FRESH_DAYS = int(os.getenv("CACHE_FRESH_DAYS", 2))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 60))                   # batch yf
MAX_SEC_WORKERS = int(os.getenv("MAX_SEC_WORKERS", 8))          # workers pour SEC

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
    return CACHE_DIR / f"ohlcv_{ticker}.csv"

def _csv_cache_meta_path(ticker: str) -> Path:
    return CACHE_DIR / f"ohlcv_{ticker}.meta.json"

def _save_csv_cache(ticker: str, df: pd.DataFrame) -> None:
    p = _csv_cache_path(ticker)
    p.parent.mkdir(exist_ok=True, parents=True)
    # sécurise colonnes
    cols = ["date","open","high","low","close","volume"]
    out = df.copy()
    out = out.rename(columns=str).reset_index()
    if "Date" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"Date":"date"})
    for c_old, c_new in [("Open","open"),("High","high"),("Low","low"),
                         ("Close","close"),("Adj Close","adj_close"),("Volume","volume")]:
        if c_old in out.columns and c_new not in out.columns:
            out = out.rename(columns={c_old:c_new})
    # garde les colonnes utiles si présentes
    keep = [c for c in cols if c in out.columns]
    out[keep].to_csv(p, index=False)
    with _csv_cache_meta_path(ticker).open("w", encoding="utf-8") as f:
        json.dump({"ts": _now_ts(), "rows": int(len(out))}, f)

def _load_csv_cache(ticker: str) -> Optional[pd.DataFrame]:
    p = _csv_cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        if "date" in df.columns:
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
        meta = json.loads(mp.read_text())
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
    res = {}
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

def _yf_download_batch(tickers: List[str], period_days: int) -> None:
    """
    Télécharge en une fois un paquet de tickers via yf.download
    et remplit le cache CSV individuel.
    """
    if not tickers:
        return
    period = f"{max(period_days, 120)}d"
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval=YF_INTERVAL,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=YF_THREADS,
        )
        if isinstance(data.columns, pd.MultiIndex):
            # multi-index: (field across tickers) OU (ticker, field) selon version
            # standard: columns level0=ticker, level1=OHLCV
            for t in tickers:
                if t in data.columns.get_level_values(0):
                    sub = data[t].copy()
                else:
                    # autre forme: level0 = field, level1 = ticker
                    try:
                        sub = data.xs(t, axis=1, level=1).copy()
                    except Exception:
                        sub = None
                if sub is None or sub.empty:
                    continue
                sub = sub.rename(columns={"Open":"open","High":"high","Low":"low",
                                          "Close":"close","Volume":"volume"})
                keep = [c for c in ["open","high","low","close","volume"] if c in sub.columns]
                if not keep:
                    continue
                out = sub[keep].dropna()
                if not out.empty:
                    out.index.name = "date"
                    _save_csv_cache(t, out)
        else:
            # single-index (cas 1 ticker)
            sub = data.rename(columns={"Open":"open","High":"high","Low":"low",
                                       "Close":"close","Volume":"volume"})
            keep = [c for c in ["open","high","low","close","volume"] if c in sub.columns]
            out = sub[keep].dropna()
            if not out.empty:
                out.index.name = "date"
                _save_csv_cache(tickers[0], out)
    except Exception as e:
        _log(f"[YF] batch fetch failed ({len(tickers)}): {e}")

def _ensure_ohlcv_cached(tickers: List[str], period_days: int = OHLCV_WINDOW_DAYS) -> None:
    need = _tickers_needing_fetch(tickers)
    if not need:
        return
    # batches
    for i in range(0, len(need), BATCH_SIZE):
        chunk = need[i:i+BATCH_SIZE]
        _log(f"[YF] fetching batch {i//BATCH_SIZE+1}/{math.ceil(len(need)/BATCH_SIZE)} size={len(chunk)}")
        _yf_download_batch(chunk, period_days)

def _get_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    return _load_csv_cache(ticker)

# ====================== indicators & scores ========================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
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
    """
    if df is None or df.empty or "close" not in df.columns:
        return "HOLD", 0.5
    close = pd.to_numeric(df["close"], errors="coerce")
    if close.isna().all() or len(close) < 60:
        return "HOLD", 0.5
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    macd, macds, hist = _macd(close)
    rsi14 = _rsi(close, 14)

    # signaux
    bull_trend = (ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1])
    macd_pos = macd.iloc[-1] > macds.iloc[-1] and hist.iloc[-1] > 0
    rsi_ok = 45 <= rsi14.iloc[-1] <= 70

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

# ====================== SEC helpers (mcap) =========================

def _compute_mcap_from_sec(t: str, cik: Optional[str], last_price: Optional[float]) -> Optional[float]:
    if last_price is None or not cik:
        return None
    try:
        shares = sec_latest_shares_outstanding(cik)
        if shares is None or shares <= 0:
            return None
        return float(last_price * float(shares))
    except Exception:
        return None

def _bulk_mcap_from_sec(tickers: List[str], price_map: Dict[str, Optional[float]], cik_map: Dict[str, str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {t: None for t in tickers}
    # Threaded but raisonnable (API SEC)
    with ThreadPoolExecutor(max_workers=MAX_SEC_WORKERS) as ex:
        futs = {}
        for t in tickers:
            p = price_map.get(t)
            cik = cik_map.get(t)
            futs[ex.submit(_compute_mcap_from_sec, t, cik, p)] = t
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                out[t] = fut.result()
            except Exception:
                out[t] = None
    return out

# =========================== pipeline ==============================

def _build_rows(uni: pd.DataFrame) -> pd.DataFrame:
    tickers = uni["ticker_yf"].tolist()

    # 1) OHLCV cache upfront (batch)
    _ensure_ohlcv_cached(tickers, OHLCV_WINDOW_DAYS)

    # 2) Sector catalog (optionnel)
    sectors = _read_sector_catalog()

    # 3) SEC map (CIK)
    try:
        cik_map = sec_ticker_map()  # {TICKER: CIK10}
    except Exception as e:
        _log(f"[WARN] sec_ticker_map failed: {e}")
        cik_map = {}

    rows = []
    price_map: Dict[str, Optional[float]] = {}
    adv_map: Dict[str, Optional[float]]   = {}

    # 4) prix/ADV + tech
    for t in tickers:
        df = _get_ohlcv(t)
        lc = _last_close(df)
        adv = _avg_dollar_vol(df) or 0.0
        price_map[t] = lc
        adv_map[t] = adv

        if df is not None and lc is not None and len(df) >= 60 and lc >= 0.5:
            tv_reco, tv_score = compute_tech_label(df)
        else:
            tv_reco, tv_score = "HOLD", 0.5

        secinfo = sectors.get(t, {})
        sector = secinfo.get("sector", "Unknown")
        industry = secinfo.get("industry", "Unknown")

        # placeholders analyst (proxy plus tard si besoin)
        analyst_bucket = "HOLD"
        analyst_votes = 20.0
        analyst_score = 0.5

        # bucket provisoire (sera définitif après MCAP filtre)
        def favorable(label: str) -> bool:
            return label in ("BUY","STRONG_BUY")
        p_tech = favorable(tv_reco)     # notre “tech composite”
        p_tv   = favorable(tv_reco)     # miroir TV
        p_an   = favorable(analyst_bucket)

        pillars_met = int(p_tech) + int(p_tv) + int(p_an)
        if pillars_met == 3 and tv_reco == "STRONG_BUY":
            bucket = "confirmed"
        elif pillars_met >= 2:
            bucket = "pre_signal"
        else:
            bucket = "event"

        adv_weight = math.log(1 + (adv or 0.0)) / 20.0
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
            "mcap_usd_final": None,  # encore vide ici
            "mcap": None,
            "avg_dollar_vol": adv or 0.0,
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

    # 5) MCAP via SEC (en parallèle, uniquement pour tickers où prix valide)
    candidates = [t for t in tickers if price_map.get(t) not in (None, np.nan)]
    mcap_map = _bulk_mcap_from_sec(candidates, price_map, cik_map)
    base["mcap_usd_final"] = base["ticker_yf"].map(mcap_map)
    base["mcap"] = base["mcap_usd_final"]

    return base

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    # Filtre STRICT : mcap non-null ET < cap
    mask = df["mcap_usd_final"].notna() & (df["mcap_usd_final"] < MCAP_CAP)
    return df.loc[mask].copy()

def _write_outputs(out: pd.DataFrame) -> None:
    out_sorted = out.sort_values(["rank_score","avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)
    (OUT_DIR / "candidates_all_ranked.csv").write_text(out_sorted.to_csv(index=False))
    _log(f"[SAVE] candidates_all_ranked.csv | rows={len(out_sorted)} | cols={out_sorted.shape[1]}")

    confirmed = out_sorted[out_sorted["bucket"]=="confirmed"]
    pre_sig   = out_sorted[out_sorted["bucket"]=="pre_signal"]
    event     = out_sorted[out_sorted["bucket"]=="event"]

    (OUT_DIR / "confirmed_STRONGBUY.csv").write_text(confirmed.to_csv(index=False))
    (OUT_DIR / "anticipative_pre_signals.csv").write_text(pre_sig.to_csv(index=False))
    (OUT_DIR / "event_driven_signals.csv").write_text(event.to_csv(index=False))
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
    path.write_text(hist.to_csv(index=False))
    _log(f"[SAVE] signals_history.csv | rows={len(hist)}")

def main():
    _log("[STEP] mix_ab_screen_indices starting…")
    uni = _ensure_universe()

    tickers = uni["ticker_yf"].tolist()
    # Optim : pré-cache OHLCV
    _ensure_ohlcv_cached(tickers, OHLCV_WINDOW_DAYS)

    base = _build_rows(uni)

    # Nettoyages types
    float_cols = ["price","last","mcap_usd_final","avg_dollar_vol","tv_score","analyst_votes","rank_score"]
    for c in float_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # Applique filtre strict mcap
    filtered = _apply_filters(base)

    # Sorties
    _write_outputs(filtered)

    # Historique
    today = pd.Timestamp.utcnow().date().isoformat()
    _update_signals_history(today, filtered)

    coverage = {
        "universe": len(uni),
        "post_filter": len(filtered),
        "price_nonnull": int(base["price"].notna().sum()),
        "mcap_final_nonnull": int(base["mcap_usd_final"].notna().sum()),
        "confirmed_count": int((filtered["bucket"]=="confirmed").sum()),
        "pre_count": int((filtered["bucket"]=="pre_signal").sum()),
        "event_count": int((filtered["bucket"]=="event").sum()),
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    main()
