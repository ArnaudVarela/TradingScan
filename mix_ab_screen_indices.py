# mix_ab_screen_indices.py — Advanced Scoring (MCAP OFF, root outputs)
from __future__ import annotations

import os, math, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# =========================== Paths & Params ===========================

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

UNIVERSE_CSV   = ROOT / "universe_in_scope.csv"   # produit par build_universe.py
SECTOR_CATALOG = ROOT / "sector_catalog.csv"      # optionnel

OUT_DIR = ROOT  # sorties à la racine (GitHub Pages peut lire ici chez toi)

# --------- Paramètres (env overridable)
DISABLE_MCAP = int(os.getenv("DISABLE_MCAP", "1"))  # 1 = pas de SEC/YF cap, pas de filtre mcap
OHLCV_WINDOW_DAYS = int(os.getenv("OHLCV_WINDOW_DAYS", 240))    # historique (>=200 pour EMA200)
AVG_DOLLAR_VOL_LOOKBACK = int(os.getenv("AVG_DOLLAR_VOL_LB", 20))
CACHE_FRESH_DAYS = int(os.getenv("CACHE_FRESH_DAYS", 2))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 60))                   # batch yf
SLEEP_BETWEEN_BATCHES = float(os.getenv("SLEEP_BETWEEN_BATCHES", "0.0"))

YF_INTERVAL = "1d"
YF_THREADS  = False  # on pilote nos propres batches

# Scoring / buckets
MARKET_TICKER = os.getenv("MARKET_TICKER", "SPY")   # pour RS globale
CONFIRMED_THRESHOLD  = float(os.getenv("CONFIRMED_THRESHOLD", "0.82"))
PRESIGNAL_THRESHOLD  = float(os.getenv("PRESIGNAL_THRESHOLD", "0.64"))
EVENT_THRESHOLD      = float(os.getenv("EVENT_THRESHOLD", "0.50"))
ADV_CONFIRMED_MIN    = float(os.getenv("ADV_CONFIRMED_MIN", "500000"))   # $ min pour confirmed
ADV_PRESIGNAL_MIN    = float(os.getenv("ADV_PRESIGNAL_MIN", "250000"))   # $ min pour pre_signal

# =========================== utils & io ===============================

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

# ====================== Indicators & helpers ========================

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

def _pct_rank(x: pd.Series, lookback: int = 126) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 3:
        return 0.5
    window = x.tail(lookback)
    if len(window) < 3:
        return 0.5
    last = float(window.iloc[-1])
    rank = (window <= last).mean()
    return float(max(0.0, min(1.0, rank)))

def _zscore_last(x: pd.Series, lb: int = 63) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if len(x) < lb:
        return 0.0
    w = x.tail(lb)
    mu = float(w.mean()); sd = float(w.std(ddof=0)) or 1e-9
    return float((w.iloc[-1] - mu) / sd)

def _squash01(v: float, lo: float, hi: float) -> float:
    if hi == lo: return 0.0
    x = (v - lo) / (hi - lo)
    return float(0.0 if x < 0 else 1.0 if x > 1 else x)

# ====================== Advanced scoring ===========================

def compute_advanced_score(df: pd.DataFrame, mkt: Optional[pd.DataFrame]=None, adv_dv: Optional[float]=None) -> dict:
    """
    Retourne un dict:
      {
        'score': float 0..1,
        'label': 'STRONG_BUY'|'BUY'|'HOLD'|'SELL',
        # sous-scores / flags
        's_trend','s_momo','s_vol','s_volu','s_rs','s_entry',
        'breakout','pullback','squeeze_on','days_to_macd_cross','overextended'
      }
    """
    out = {'score':0.5,'label':'HOLD','s_trend':0,'s_momo':0,'s_vol':0,'s_volu':0,'s_rs':0,'s_entry':0,
           'breakout':False,'pullback':False,'squeeze_on':False,'days_to_macd_cross':None,'overextended':False}
    if df is None or df.empty:
        return out
    req = ["open","high","low","close","volume"]
    if any(c not in df.columns for c in req):
        return out
    if len(df) < 200:
        return out

    d = df.copy()
    c = pd.to_numeric(d["close"], errors="coerce")
    h = pd.to_numeric(d["high"], errors="coerce")
    l = pd.to_numeric(d["low"], errors="coerce")
    v = pd.to_numeric(d["volume"], errors="coerce")
    if c.isna().any(): c = c.fillna(method="ffill")

    # EMAs & slopes
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200= c.ewm(span=200,adjust=False, min_periods=200).mean()
    if ema200.isna().all(): return out

    e20, e50, e200 = ema20.iloc[-1], ema50.iloc[-1], ema200.iloc[-1]
    if len(ema50) >= 5 and not ema50.iloc[-5] == 0:
        slope50 = (ema50.iloc[-1] - ema50.iloc[-5]) / (abs(ema50.iloc[-5]) + 1e-9)
    else:
        slope50 = 0.0

    # MACD / RSI
    fast = c.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = fast - slow
    macd_sig  = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_sig
    rsi = _rsi(c, 14)

    # BBands / ATR
    ma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std(ddof=0)
    bb_up = ma20 + 2*sd20
    bb_lo = ma20 - 2*sd20
    bb_width = (bb_up - bb_lo) / (ma20 + 1e-9)
    bb_pctile = _pct_rank(bb_width, 126)  # faible => squeeze
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atrp = (atr14 / (c + 1e-9)).fillna(0)

    # Volume / OBV
    obv = ((np.sign(c.diff().fillna(0)) * v).fillna(0)).cumsum()
    if len(obv) >= 10 and not abs(obv.iloc[-10]) == 0:
        obv_slope = (obv.iloc[-1] - obv.iloc[-10]) / (abs(obv.iloc[-10]) + 1e-9)
    else:
        obv_slope = 0.0
    vol20 = v.rolling(20, min_periods=1).mean()
    vol60 = v.rolling(60, min_periods=1).mean()
    vol_ratio = float((vol20.iloc[-1] + 1e-9) / (vol60.iloc[-1] + 1e-9))
    adv_ratio = None
    if adv_dv is not None and adv_dv > 0:
        # normalisation douce de la liquidité
        adv_ratio = _squash01(math.log1p(adv_dv), math.log1p(5e5), math.log1p(2e7))  # 0.5M → 20M+

    # Relative Strength vs marché
    s_rs = 0.5
    if mkt is not None and not mkt.empty and "close" in mkt.columns:
        mclose = pd.to_numeric(mkt["close"], errors="coerce").reindex(c.index).fillna(method="ffill")
        rs_line = (c / (mclose + 1e-9))
        rs_chg20 = float((rs_line.iloc[-1] / (rs_line.iloc[-20] + 1e-9)) - 1) if len(rs_line) >= 21 else 0.0
        rs_chg63 = float((rs_line.iloc[-1] / (rs_line.iloc[-63] + 1e-9)) - 1) if len(rs_line) >= 64 else 0.0
        # Pctile de la somme des retours → proxy momentum cumulé relatif
        rs_pctile = _pct_rank(rs_line.pct_change().dropna().cumsum(), 252) if rs_line.notna().sum() > 10 else 0.5
        s_rs = 0.45*_squash01(rs_chg20, -0.1, 0.1) + 0.35*_squash01(rs_chg63, -0.2, 0.2) + 0.20*rs_pctile

    # Squeeze & entry triggers
    squeeze_on = (bb_pctile <= 0.25) and (atrp.iloc[-1] < atrp.rolling(63, min_periods=5).median().iloc[-1] if len(atrp) >= 63 else True)
    days_to_cross = None
    if len(macd_line) >= 3 and len(macd_sig) >= 3:
        diff = float((macd_line.iloc[-1] - macd_sig.iloc[-1]))
        slope = float((macd_line.iloc[-1] - macd_line.iloc[-3]) - (macd_sig.iloc[-1] - macd_sig.iloc[-3])) / 2.0
        if slope != 0:
            est = -diff / slope
            days_to_cross = float(est) if 0 < est < 10 else None

    # Breakout / Pullback
    hhv20 = c.rolling(20, min_periods=1).max()
    prev_hhv20 = hhv20.iloc[-2] if len(hhv20) >= 2 else hhv20.iloc[-1]
    breakout = (c.iloc[-1] > prev_hhv20) and (v.iloc[-1] > 1.3*vol20.iloc[-1])
    dist_ema20 = float((c.iloc[-1] - e20) / (c.iloc[-1] + 1e-9))
    rsi_last = float(_rsi(c, 14).iloc[-1]) if len(c) >= 14 else 50.0
    pullback = (e20 > e50 > e200) and (-0.02 <= dist_ema20 <= 0.01) and (45 <= rsi_last <= 60)

    # Overextension penalty
    overextended = (len(sd20) > 0 and c.iloc[-1] > ( (ma20.iloc[-1] if not np.isnan(ma20.iloc[-1]) else c.iloc[-1]) + 2*sd20.iloc[-1] + 0.3*(sd20.iloc[-1] or 0) )) or (rsi_last > 75)

    # Sous-scores 0..1
    s_trend = 0.55*_squash01((e20/e50)-1, 0.0, 0.06) + 0.30*_squash01((e50/e200)-1, 0.0, 0.12) + 0.15*_squash01(slope50, 0.0, 0.06)
    s_momo  = 0.55*_squash01(_zscore_last(macd_hist, 63), 0.0, 2.5) + 0.45*(1.0 - abs(rsi_last - 60.0)/60.0)
    s_vol   = 0.6*(1.0 - bb_pctile) + 0.4*(1.0 - _pct_rank(atrp, 126))  # petit = mieux
    s_volu  = 0.6*_squash01(vol_ratio, 0.8, 1.4) + 0.4*_squash01(obv_slope, 0.0, 0.15)
    if adv_ratio is not None: s_volu = 0.7*s_volu + 0.3*adv_ratio
    s_entry = 0.0
    if breakout: s_entry += 0.8
    if pullback: s_entry = max(s_entry, 0.65)
    s_rs    = float(max(0.0, min(1.0, s_rs)))

    # Score global (poids)
    score = (
        0.28*s_trend +
        0.22*s_momo  +
        0.22*s_rs    +
        0.14*s_volu  +
        0.10*s_vol   +
        0.04*s_entry
    )
    # Boosts “proches” / “propre”
    if breakout: score += 0.03
    if (days_to_cross is not None) and (days_to_cross <= 5): score += 0.03
    if pullback: score += 0.02
    if overextended: score -= 0.04

    score = float(max(0.0, min(1.0, round(score, 4))))

    # Label (technique) basé sur le score (pour compat)
    if score >= 0.80: label = "STRONG_BUY"
    elif score >= 0.60: label = "BUY"
    elif score >= 0.40: label = "HOLD"
    else: label = "SELL"

    out.update(dict(
        score=score, label=label, s_trend=float(round(s_trend,4)), s_momo=float(round(s_momo,4)),
        s_vol=float(round(s_vol,4)), s_volu=float(round(s_volu,4)), s_rs=float(round(s_rs,4)),
        s_entry=float(round(s_entry,4)), breakout=bool(breakout), pullback=bool(pullback),
        squeeze_on=bool(squeeze_on), days_to_macd_cross=days_to_cross, overextended=bool(overextended)
    ))
    return out

def _bucket_from_score(features: dict, adv_dv: Optional[float]) -> str:
    s = features.get('score', 0.5)
    over = features.get('overextended', False)
    breakout = features.get('breakout', False)
    squeeze = features.get('squeeze_on', False)
    dtc = features.get('days_to_macd_cross', None)

    adv_ok_confirmed = (adv_dv or 0.0) >= ADV_CONFIRMED_MIN
    adv_ok_pre       = (adv_dv or 0.0) >= ADV_PRESIGNAL_MIN

    # Now
    if s >= CONFIRMED_THRESHOLD and not over and adv_ok_confirmed and (breakout or s >= (CONFIRMED_THRESHOLD + 0.04)):
        return "confirmed"
    # Soon
    if (PRESIGNAL_THRESHOLD <= s < CONFIRMED_THRESHOLD) and not over and adv_ok_pre:
        if breakout or squeeze or (dtc is not None and dtc <= 5):
            return "pre_signal"
    # Watch
    if s >= EVENT_THRESHOLD or squeeze or (dtc is not None and dtc <= 7):
        return "event"
    return "event"

# =========================== pipeline ==============================

def _build_rows(uni: pd.DataFrame) -> pd.DataFrame:
    tickers = uni["ticker_yf"].tolist()

    # 1) OHLCV cache upfront (batch)
    _ensure_ohlcv_cached(tickers, OHLCV_WINDOW_DAYS)

    # 2) Sector catalog (optionnel)
    sectors = _read_sector_catalog()

    # 3) Marché (pour RS)
    mkt_df = None
    try:
        _ensure_ohlcv_cached([MARKET_TICKER], OHLCV_WINDOW_DAYS)
        mkt_df = _get_ohlcv(MARKET_TICKER)
    except Exception:
        mkt_df = None

    rows = []

    for t in tickers:
        df = _get_ohlcv(t)
        lc = _last_close(df)
        adv = _avg_dollar_vol(df) or 0.0

        feats = compute_advanced_score(df, mkt=mkt_df, adv_dv=adv)
        tv_reco = feats['label']              # réutilise ta colonne
        tv_score = feats['score']
        bucket = _bucket_from_score(feats, adv)

        secinfo = sectors.get(t, {})
        sector = secinfo.get("sector", "Unknown")
        industry = secinfo.get("industry", "Unknown")

        rows.append({
            "ticker_yf": t,
            "ticker_tv": t,
            "price": lc,
            "last": lc,
            "mcap_usd_final": np.nan,   # MCAP OFF
            "mcap": np.nan,
            "avg_dollar_vol": adv,
            "tv_score": tv_score,
            "tv_reco": tv_reco,
            "analyst_bucket": "HOLD",   # placeholders non utilisés par le score
            "analyst_votes": np.nan,
            "sector": sector,
            "industry": industry,
            "p_tech": None,
            "p_tv": None,
            "p_an": None,
            "pillars_met": None,
            "votes_bin": None,
            "rank_score": tv_score,     # aligne tri
            "bucket": bucket,
            # (bonus) expose sous-scores/flags pour debug/tri UI
            "s_trend": feats["s_trend"],
            "s_momo": feats["s_momo"],
            "s_vol": feats["s_vol"],
            "s_volu": feats["s_volu"],
            "s_rs": feats["s_rs"],
            "s_entry": feats["s_entry"],
            "breakout": feats["breakout"],
            "pullback": feats["pullback"],
            "squeeze_on": feats["squeeze_on"],
            "days_to_macd_cross": feats["days_to_macd_cross"],
            "overextended": feats["overextended"],
        })

    base = pd.DataFrame(rows)
    return base

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    # MCAP OFF → pas de filtre
    # On peut optionnellement filtrer les penny stocks / données manquantes
    out = df.copy()
    out = out[out["price"].notna()]  # garde prix valides
    return out

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
    cols = ["date","ticker_yf","sector","bucket","tv_reco","analyst_bucket","tv_score"]
    new = out[["ticker_yf","sector","bucket","tv_reco","analyst_bucket","tv_score"]].copy()
    new.insert(0, "date", today)
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=cols)
        hist = pd.concat([old, new], ignore_index=True)
    else:
        hist = new
    hist = hist.tail(1000)
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

    # Types sûrs
    float_cols = ["price","last","mcap_usd_final","avg_dollar_vol","tv_score","analyst_votes","rank_score"]
    for c in float_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # Applique filtres de base
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
        "confirmed_count": int((filtered["bucket"]=="confirmed").sum()),
        "pre_count": int((filtered["bucket"]=="pre_signal").sum()),
        "event_count": int((filtered["bucket"]=="event").sum()),
    }
    _log(f"[COVERAGE] {coverage}")

if __name__ == "__main__":
    main()
