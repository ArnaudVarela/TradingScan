# mix_ab_screen_indices.py — Advanced, Adaptive, Sector-neutral, Regime-aware
# MCAP display ON (no filtering), robust cache, mirroring to dashboard/public

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

OUT_DIR = ROOT  # sorties à la racine (GitHub Pages lit ici chez toi)

# --------- Paramètres (env overridable)
DISABLE_MCAP = int(os.getenv("DISABLE_MCAP", "1"))  # legacy: pas utilisé pour l'affichage ici
OHLCV_WINDOW_DAYS = int(os.getenv("OHLCV_WINDOW_DAYS", 260))     # >=200 pour EMA200
AVG_DOLLAR_VOL_LOOKBACK = int(os.getenv("AVG_DOLLAR_VOL_LB", 20))
CACHE_FRESH_DAYS = int(os.getenv("CACHE_FRESH_DAYS", 2))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 60))                   # batch yf
SLEEP_BETWEEN_BATCHES = float(os.getenv("SLEEP_BETWEEN_BATCHES", "0.0"))

YF_INTERVAL = "1d"
YF_THREADS  = False  # on pilote nos propres batches

# Scoring / buckets / régulation
MARKET_TICKER = os.getenv("MARKET_TICKER", "SPY")   # RS globale & régime
CONFIRMED_THRESHOLD_BASE  = float(os.getenv("CONFIRMED_THRESHOLD", "0.82"))
PRESIGNAL_THRESHOLD_BASE  = float(os.getenv("PRESIGNAL_THRESHOLD", "0.64"))
EVENT_THRESHOLD           = float(os.getenv("EVENT_THRESHOLD", "0.50"))
CONFIRMED_FLOOR           = float(os.getenv("CONFIRMED_FLOOR", "0.76"))   # plancher pour adaptation
ADV_CONFIRMED_MIN         = float(os.getenv("ADV_CONFIRMED_MIN", "500000"))  # $ min confirmed
ADV_PRESIGNAL_MIN         = float(os.getenv("ADV_PRESIGNAL_MIN", "250000"))  # $ min pre_signal

# Cibles adaptatives (éviter confirmed=0)
TARGET_CONFIRMED_PCT = float(os.getenv("TARGET_CONFIRMED_PCT", "0.02"))  # ~2% de l’univers
TARGET_PRESIGNAL_PCT = float(os.getenv("TARGET_PRESIGNAL_PCT", "0.08"))  # ~8%
MIN_CONFIRMED_COUNT  = int(os.getenv("MIN_CONFIRMED_COUNT", "5"))
MIN_PRESIGNAL_COUNT  = int(os.getenv("MIN_PRESIGNAL_COUNT", "20"))

# Hystérésis légère (collant)
HYSTERESIS_DAYS  = int(os.getenv("HYSTERESIS_DAYS", "3"))
HYSTERESIS_RELAX = float(os.getenv("HYSTERESIS_RELAX", "0.02"))  # relaxe les seuils si présent récemment

# --- Market cap display (no filtering)
ENABLE_MCAP_DISPLAY = int(os.getenv("ENABLE_MCAP_DISPLAY", "1"))
ENABLE_MCAP_DISPLAY_SEC = int(os.getenv("ENABLE_MCAP_DISPLAY_SEC", "0"))  # 1 = autorise fallback SEC (US only)
MAX_SEC_WORKERS = int(os.getenv("MAX_SEC_WORKERS", "4"))

# SEC helpers (optionnels)
try:
    from sec_helpers import sec_ticker_map, sec_latest_shares_outstanding
except Exception:
    sec_ticker_map = None
    sec_latest_shares_outstanding = None

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
            return
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[needed].dropna()
    if out.empty:
        return
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out.index.name = "date"

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

def _infer_sectors_from_universe() -> Dict[str, Dict[str,str]]:
    if not UNIVERSE_CSV.exists():
        return {}
    try:
        df = pd.read_csv(UNIVERSE_CSV)
    except Exception:
        return {}
    sec_col = next((c for c in ["sector","Sector","GICS Sector","gics_sector"] if c in df.columns), None)
    ind_col = next((c for c in ["industry","Industry","GICS Industry","gics_industry"] if c in df.columns), None)
    tik_col = next((c for c in ["ticker_yf","ticker","Symbol","symbol"] if c in df.columns), None)
    if not tik_col:
        return {}
    out = {}
    for _, r in df.iterrows():
        t = str(r[tik_col]).upper().strip()
        if not t:
            continue
        out[t] = {
            "sector": str(r.get(sec_col, "Unknown")) if sec_col else "Unknown",
            "industry": str(r.get(ind_col, "Unknown")) if ind_col else "Unknown",
        }
    return out

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
    if isinstance(df.columns, pd.MultiIndex):
        return df  # l’appelant découpera
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
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            for t in tickers:
                sub = None
                try:
                    if t in level0:
                        sub = data[t]
                    else:
                        sub = data.xs(t, axis=1, level=1, drop_level=False)
                        fields = ["Open","High","Low","Close","Volume"]
                        sub = sub.copy()
                        sub.columns = [c[0] for c in sub.columns]
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
            out = _normalize_yf_df(data)
            if out is not None:
                _save_csv_cache(tickers[0], out)
    except Exception as e:
        _log(f"[YF] batch fetch failed ({len(tickers)}): {e} — fallback per-ticker")
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
    mu = float(w.mean())
    sd = float(w.std(ddof=0)) or 1e-9
    return float((w.iloc[-1] - mu) / sd)

def _squash01(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    x = (v - lo) / (hi - lo)
    return float(0.0 if x < 0 else 1.0 if x > 1 else x)

def _tech_label_from_features(feats: dict) -> str:
    if feats.get("breakout"):
        return "Breakout"
    if feats.get("pullback"):
        return "Pullback on trend"
    if feats.get("squeeze_on"):
        return "Squeeze setup"
    dtc = feats.get("days_to_macd_cross")
    if dtc is not None and dtc <= 3:
        return "MACD cross imminent"
    if feats.get("overextended"):
        return "Overextended"
    subs = {k: feats.get(k, 0.0) for k in ("s_trend","s_momo","s_rs")}
    best = max(subs, key=subs.get)
    return {"s_trend":"Uptrend","s_momo":"Momentum","s_rs":"RS strong"}[best]

# ====================== FX & MCAP (display) =========================

_FX_CACHE = {}

def _fx_to_usd(currency: str) -> Optional[float]:
    cur = (currency or "USD").upper()
    if cur == "USD":
        return 1.0
    cached = _FX_CACHE.get(cur)
    if cached and (_now_ts() - cached[1]) < 3600:
        return cached[0]
    pair1 = f"{cur}USD=X"
    pair2 = f"USD{cur}=X"
    rate = None
    for pair in (pair1, pair2):
        try:
            hist = yf.download(pair, period="10d", interval="1d", progress=False, threads=False)
            if not hist.empty and "Close" in hist.columns:
                px = float(hist["Close"].dropna().iloc[-1])
                rate = px if pair == pair1 else (1.0 / px)
                break
        except Exception:
            continue
    if rate:
        _FX_CACHE[cur] = (rate, _now_ts())
    return rate

def _infer_mcap_usd(ticker: str, last_price: Optional[float]) -> Optional[float]:
    """Renvoie une estimation de MCAP en USD (yfinance fast_info > price*shares ; fallback SEC optionnel)."""
    if not ENABLE_MCAP_DISPLAY:
        return None

    mcap = None
    currency = "USD"

    # 1) yfinance fast_info
    try:
        fi = yf.Ticker(ticker).fast_info
        if fi:
            mcap = fi.get("market_cap") or None
            currency = (fi.get("currency") or "USD")
            if mcap is None:
                shares = fi.get("shares")
                if last_price is not None and shares:
                    mcap = float(last_price) * float(shares)
    except Exception:
        pass

    # 2) SEC fallback (optionnel, US only)
    if mcap is None and ENABLE_MCAP_DISPLAY_SEC and last_price is not None and sec_ticker_map and sec_latest_shares_outstanding:
        try:
            if not hasattr(_infer_mcap_usd, "_cik_map"):
                setattr(_infer_mcap_usd, "_cik_map", sec_ticker_map())
            cik = getattr(_infer_mcap_usd, "_cik_map", {}).get(ticker)
            if cik:
                shares = sec_latest_shares_outstanding(cik)
                if shares and shares > 0:
                    mcap = float(last_price) * float(shares)
        except Exception:
            pass

    # 3) FX → USD
    if mcap is not None and (currency or "USD").upper() != "USD":
        fx = _fx_to_usd(currency)
        if fx:
            mcap = float(mcap) * float(fx)

    return float(mcap) if mcap is not None else None

# ====================== Regime & advanced scoring ===================

def _market_regime_factor(mkt: Optional[pd.DataFrame]) -> float:
    """Renvoie un facteur multiplicatif ~ [0.90 ... 1.05] selon risk-on/off."""
    if mkt is None or mkt.empty or "close" not in mkt.columns:
        return 1.0
    c = pd.to_numeric(mkt["close"], errors="coerce").dropna()
    if len(c) < 200:
        return 1.0
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200= c.ewm(span=200,adjust=False, min_periods=200).mean()
    slope50 = (ema50.iloc[-1] - ema50.iloc[-5]) / (abs(ema50.iloc[-5]) + 1e-9)
    vol63   = c.pct_change().rolling(63).std(ddof=0).iloc[-1]
    dd      = 1.0 - (c.iloc[-1] / c.rolling(252).max().iloc[-1] + 1e-9)

    s_trend = _squash01((ema50.iloc[-1]/(ema200.iloc[-1]+1e-9))-1, -0.03, 0.06)
    s_slope = _squash01(slope50, -0.04, 0.06)
    s_vol   = 1.0 - _squash01(vol63, 0.015, 0.05)
    s_dd    = 1.0 - _squash01(dd, 0.05, 0.25)

    base = 0.38*s_trend + 0.22*s_slope + 0.20*s_vol + 0.20*s_dd  # [0..1]
    return float(round(0.90 + 0.15*base, 4))

def compute_advanced_score(df: pd.DataFrame, mkt: Optional[pd.DataFrame]=None, adv_dv: Optional[float]=None) -> dict:
    """
    Score absolu continu (0..1) + flags d'entrée, RS, volume, etc.
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
    if c.isna().any():
        c = c.ffill()

    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200= c.ewm(span=200,adjust=False, min_periods=200).mean()
    if ema200.isna().all():
        return out

    slope50 = (ema50.iloc[-1] - ema50.iloc[-5]) / (abs(ema50.iloc[-5]) + 1e-9) if len(ema50)>=5 else 0.0

    fast = c.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = fast - slow
    macd_sig  = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_sig
    rsi = _rsi(c, 14)
    rsi_last = float(rsi.iloc[-1]) if len(rsi) else 50.0

    ma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std(ddof=0)
    bb_up = ma20 + 2*sd20
    bb_lo = ma20 - 2*sd20
    bb_width = (bb_up - bb_lo) / (ma20 + 1e-9)
    bb_pctile = _pct_rank(bb_width, 126)
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atrp = (atr14 / (c + 1e-9)).fillna(0)

    obv = ((np.sign(c.diff().fillna(0)) * v).fillna(0)).cumsum()
    obv_slope = (obv.iloc[-1] - obv.iloc[-10]) / (abs(obv.iloc[-10]) + 1e-9) if len(obv)>=10 else 0.0
    vol20 = v.rolling(20, min_periods=1).mean()
    vol60 = v.rolling(60, min_periods=1).mean()
    vol_ratio = float((vol20.iloc[-1] + 1e-9) / (vol60.iloc[-1] + 1e-9))
    adv_ratio = None
    if adv_dv is not None and adv_dv > 0:
        adv_ratio = _squash01(math.log1p(adv_dv), math.log1p(5e5), math.log1p(2e7))  # 0.5M → 20M+

    # RS vs marché
    s_rs = 0.5
    if mkt is not None and not mkt.empty and "close" in mkt.columns:
        mclose = pd.to_numeric(mkt["close"], errors="coerce").reindex(c.index).ffill()
        rs_line = (c / (mclose + 1e-9))
        rs_chg20 = float((rs_line.iloc[-1] / (rs_line.iloc[-20] + 1e-9)) - 1) if len(rs_line) >= 21 else 0.0
        rs_chg63 = float((rs_line.iloc[-1] / (rs_line.iloc[-63] + 1e-9)) - 1) if len(rs_line) >= 64 else 0.0
        rs_pctile = _pct_rank(rs_line.pct_change().dropna().cumsum(), 252) if rs_line.notna().sum() > 10 else 0.5
        s_rs = 0.45*_squash01(rs_chg20, -0.1, 0.1) + 0.35*_squash01(rs_chg63, -0.2, 0.2) + 0.20*rs_pctile

    # Triggers
    squeeze_on = (bb_pctile <= 0.25) and (atrp.iloc[-1] < atrp.rolling(63, min_periods=5).median().iloc[-1] if len(atrp) >= 63 else True)
    days_to_cross = None
    if len(macd_line) >= 3 and len(macd_sig) >= 3:
        diff = float((macd_line.iloc[-1] - macd_sig.iloc[-1]))
        slope = float((macd_line.iloc[-1] - macd_line.iloc[-3]) - (macd_sig.iloc[-1] - macd_sig.iloc[-3])) / 2.0
        if slope != 0:
            est = -diff / slope
            days_to_cross = float(est) if 0 < est < 10 else None

    hhv20 = c.rolling(20, min_periods=1).max()
    prev_hhv20 = hhv20.iloc[-2] if len(hhv20) >= 2 else hhv20.iloc[-1]
    breakout = (c.iloc[-1] > prev_hhv20) and (v.iloc[-1] > 1.3*vol20.iloc[-1])
    dist_ema20 = float((c.iloc[-1] - (ema20.iloc[-1] or c.iloc[-1])) / (c.iloc[-1] + 1e-9))
    pullback = (ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]) and (-0.02 <= dist_ema20 <= 0.01) and (45 <= rsi_last <= 60)
    overextended = (len(sd20)>0 and c.iloc[-1] > ((ma20.iloc[-1] or c.iloc[-1]) + 2*sd20.iloc[-1] + 0.3*(sd20.iloc[-1] or 0))) or (rsi_last > 75)

    # Sous-scores
    s_trend = 0.55*_squash01((ema20.iloc[-1]/(ema50.iloc[-1]+1e-9))-1, 0.0, 0.06) \
            + 0.30*_squash01((ema50.iloc[-1]/(ema200.iloc[-1]+1e-9))-1, 0.0, 0.12) \
            + 0.15*_squash01(slope50, 0.0, 0.06)
    s_momo  = 0.55*_squash01(_zscore_last(macd_hist, 63), 0.0, 2.5) \
            + 0.45*(1.0 - abs(rsi_last - 60.0)/60.0)
    s_vol   = 0.6*(1.0 - bb_pctile) + 0.4*(1.0 - _pct_rank(atrp, 126))
    s_volu  = 0.6*_squash01(vol_ratio, 0.8, 1.4) + 0.4*_squash01(obv_slope, 0.0, 0.15)
    if adv_ratio is not None: s_volu = 0.7*s_volu + 0.3*adv_ratio
    s_entry = 0.0
    if breakout: s_entry += 0.8
    if pullback: s_entry = max(s_entry, 0.65)
    s_rs    = float(max(0.0, min(1.0, s_rs)))

    # Score global + boosts adoucis
    score = (0.28*s_trend + 0.22*s_momo + 0.22*s_rs + 0.14*s_volu + 0.10*s_vol + 0.04*s_entry)
    if breakout: score += 0.02
    if (days_to_cross is not None) and (days_to_cross <= 3): score += 0.02
    if pullback: score += 0.01
    if overextended: score -= 0.04
    score = float(max(0.0, min(1.0, round(score, 4))))

    # Label compat (non utilisé pour bucket final)
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

# =========================== pipeline ==============================

def _apply_sector_neutral_rank(df: pd.DataFrame) -> pd.Series:
    """Percentile intra-secteur du tv_score (réduit le biais sectoriel)."""
    df = df.copy()
    df["sector"] = df["sector"].fillna("Unknown")
    ranks = []
    for _, g in df.groupby("sector"):
        if "tv_score" in g.columns and len(g):
            r = g["tv_score"].rank(pct=True, method="average")
            ranks.append(r)
    if len(ranks):
        out = pd.concat(ranks).sort_index()
        return out.astype(float)
    return pd.Series(np.nan, index=df.index)

def _hysteresis_mask(history_path: Path, days: int) -> set:
    """Tickers apparus dans les N derniers jours (confirmed ou pre_signal)."""
    keep = set()
    if not history_path.exists() or days <= 0:
        return keep
    try:
        hist = pd.read_csv(history_path)
        if not {"date","ticker_yf","bucket"}.issubset(hist.columns):
            return keep
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days+1)
        recent = hist[hist["date"] >= cutoff]
        recent = recent[recent["bucket"].isin(["confirmed","pre_signal"])]
        keep = set(recent["ticker_yf"].astype(str))
    except Exception:
        pass
    return keep

def _build_rows(uni: pd.DataFrame) -> pd.DataFrame:
    tickers = uni["ticker_yf"].tolist()

    # 1) OHLCV cache upfront
    _ensure_ohlcv_cached(tickers, OHLCV_WINDOW_DAYS)

    # 2) Sector catalog + fallback universe
    sectors = _read_sector_catalog()
    fallback = _infer_sectors_from_universe()
    for k, v in fallback.items():
        sectors.setdefault(k, v)

    # 3) Marché (pour RS & régime)
    mkt_df = None
    try:
        _ensure_ohlcv_cached([MARKET_TICKER], OHLCV_WINDOW_DAYS)
        mkt_df = _get_ohlcv(MARKET_TICKER)
    except Exception:
        mkt_df = None
    regime = _market_regime_factor(mkt_df)

    rows = []
    for t in tickers:
        df = _get_ohlcv(t)
        lc = _last_close(df)
        adv = _avg_dollar_vol(df) or 0.0

        feats = compute_advanced_score(df, mkt=mkt_df, adv_dv=adv)

        # applique le régime au score absolu
        tv_score = float(max(0.0, min(1.0, round(feats['score'] * regime, 4))))
        tv_reco  = ("STRONG_BUY" if tv_score>=0.80 else "BUY" if tv_score>=0.60 else "HOLD" if tv_score>=0.40 else "SELL")

        # Tech label & piliers (placeholder p_an affiné plus bas)
        tech_label = _tech_label_from_features(feats)
        p_tv = tv_score >= 0.60
        p_tech = (
            feats["breakout"] or feats["pullback"] or feats["squeeze_on"]
            or (feats["days_to_macd_cross"] is not None and feats["days_to_macd_cross"] <= 3)
            or (feats["s_trend"] >= 0.60 and not feats["overextended"])
        )
        p_an = (feats["s_rs"] >= 0.60)  # sera renforcé avec sector_pct après coup
        pillars_met = int(bool(p_tv)) + int(bool(p_tech)) + int(bool(p_an))

        secinfo = sectors.get(t, {})
        sector = secinfo.get("sector", "Unknown")
        industry = secinfo.get("industry", "Unknown")

        # MCAP display only
        mcap_val = _infer_mcap_usd(t, lc)

        rows.append({
            "ticker_yf": t,
            "ticker_tv": t,
            "price": lc,
            "last": lc,
            "mcap_usd_final": mcap_val,
            "mcap": mcap_val,                     # utilisé par ta UI
            "avg_dollar_vol": adv,
            "tv_score": tv_score,
            "tv_reco": tv_reco,
            "analyst_bucket": "HOLD",
            "analyst_votes": 0,                   # pour afficher "0 votes"
            "sector": sector,
            "industry": industry,
            "rank_score": tv_score,               # provisoire
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

            # UI helpers
            "tech_label": tech_label,
            "tech": tech_label,                   # alias pour la colonne "Tech"
            "p_tech": bool(p_tech),
            "p_tv": bool(p_tv),
            "p_an": bool(p_an),
            "pillars_met": int(pillars_met),
            "votes_bin": "0",
        })

    base = pd.DataFrame(rows)

    # Sector-neutral RS percentile + liquidity score
    base["sector_pct"] = _apply_sector_neutral_rank(base).fillna(0.5)
    base["liq_score"]  = base["avg_dollar_vol"].apply(lambda x: _squash01(math.log1p(x or 0.0), math.log1p(5e5), math.log1p(2e7)))
    # Rank final
    base["rank_score"] = (0.65*base["tv_score"] + 0.25*base["sector_pct"] + 0.10*base["liq_score"]).astype(float)

    # p_an affiné avec sector_pct, puis recalcul des piliers
    base["p_an"] = (base["s_rs"] >= 0.60) & (base["sector_pct"] >= 0.70)
    base["pillars_met"] = base[["p_tv","p_tech","p_an"]].sum(axis=1).astype(int)

    return base

def _adaptive_thresholds(df: pd.DataFrame, hist_keep: set) -> Tuple[float, float]:
    """
    Adapte les seuils pour approcher les cibles (confirmed ~2%, pre ~8%) SANS descendre
    sous des planchers de qualité. (corrigé : on relève le seuil si trop de candidats)
    """
    n = len(df)
    if n == 0:
        return (CONFIRMED_THRESHOLD_BASE, PRESIGNAL_THRESHOLD_BASE)

    target_conf = max(MIN_CONFIRMED_COUNT, int(TARGET_CONFIRMED_PCT * n))
    target_pre  = max(MIN_PRESIGNAL_COUNT, int(TARGET_PRESIGNAL_PCT * n))

    elig_conf = df[(~df["overextended"]) & (df["avg_dollar_vol"] >= ADV_CONFIRMED_MIN)]
    elig_pre  = df[(~df["overextended"]) & (df["avg_dollar_vol"] >= ADV_PRESIGNAL_MIN)]

    # Confirmed — seuil = quantile haut (garder ~target_conf au-dessus)
    if len(elig_conf):
        q_conf = max(0.0, 1.0 - target_conf / len(elig_conf))
        q_conf_score = float(elig_conf["tv_score"].quantile(q_conf))
        confirmed_thr = max(CONFIRMED_THRESHOLD_BASE, CONFIRMED_FLOOR, q_conf_score)
    else:
        confirmed_thr = CONFIRMED_THRESHOLD_BASE

    # Pre-signal — couloir sous confirmed
    if len(elig_pre):
        tot_target = min(len(elig_pre), target_conf + target_pre)
        q_pre_all  = max(0.0, 1.0 - tot_target / len(elig_pre))
        q_pre_score = float(elig_pre["tv_score"].quantile(q_pre_all))
        presignal_thr = max(EVENT_THRESHOLD, min(q_pre_score, confirmed_thr - 0.01))
    else:
        presignal_thr = max(EVENT_THRESHOLD, min(PRESIGNAL_THRESHOLD_BASE, confirmed_thr - 0.01))

    # Hystérésis douce
    if len(hist_keep):
        confirmed_thr = max(CONFIRMED_FLOOR, confirmed_thr - HYSTERESIS_RELAX)
        presignal_thr = max(EVENT_THRESHOLD, min(confirmed_thr - 0.01, presignal_thr - HYSTERESIS_RELAX))

    # sécurité
    presignal_thr = min(presignal_thr, confirmed_thr - 0.01)

    return (float(confirmed_thr), float(presignal_thr))

def _assign_buckets(df: pd.DataFrame, confirmed_thr: float, presignal_thr: float) -> pd.Series:
    def decide(row):
        s = float(row["tv_score"] or 0.0)
        over = bool(row["overextended"])
        adv  = float(row["avg_dollar_vol"] or 0.0)
        breakout = bool(row["breakout"])
        pullback = bool(row["pullback"])
        squeeze = bool(row["squeeze_on"])
        dtc = row["days_to_macd_cross"]
        adv_ok_conf = adv >= ADV_CONFIRMED_MIN
        adv_ok_pre  = adv >= ADV_PRESIGNAL_MIN

        # --- Zone haute : s >= confirmed_thr
        if s >= confirmed_thr and not over and adv_ok_conf:
            strong_enough = breakout or pullback or (pd.notna(dtc) and dtc is not None and dtc <= 1) or (s >= confirmed_thr + 0.06)
            if strong_enough:
                return "confirmed"
            if s >= presignal_thr and adv_ok_pre:
                return "pre_signal"

        # --- Couloir pré-signal : presignal_thr <= s < confirmed_thr
        if (presignal_thr <= s < confirmed_thr) and not over and adv_ok_pre:
            if breakout or squeeze or (pd.notna(dtc) and dtc is not None and dtc <= 5) or pullback:
                return "pre_signal"

        # --- Sinon : event si un minimum de potentiel / setup
        if s >= EVENT_THRESHOLD or squeeze or (pd.notna(dtc) and dtc is not None and dtc <= 7):
            return "event"
        return "event"
    return df.apply(decide, axis=1)

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["price"].notna()]  # garde prix valides
    return out

def _safe_write_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    Path(tmp).replace(path)

def _mirror_to_public(names: List[str]) -> None:
    """Copie (atomique) vers dashboard/public si ce dossier existe."""
    public_dir = ROOT / "dashboard" / "public"
    if not public_dir.exists():
        return
    for name in names:
        src = OUT_DIR / name
        dst = public_dir / name
        if src.exists():
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            tmp.write_bytes(src.read_bytes())
            Path(tmp).replace(dst)

def _write_outputs(out: pd.DataFrame, run_ts: str, diag: dict) -> None:
    out_sorted = out.sort_values(["rank_score","avg_dollar_vol"], ascending=[False, False]).reset_index(drop=True)

    # Insère run_ts UNE fois
    if "run_ts" not in out_sorted.columns:
        out_sorted.insert(0, "run_ts", run_ts)

    _safe_write_csv(OUT_DIR / "candidates_all_ranked.csv", out_sorted)
    _log(f"[SAVE] candidates_all_ranked.csv | rows={len(out_sorted)} | cols={out_sorted.shape[1]}")

    confirmed = out_sorted[out_sorted["bucket"]=="confirmed"].copy()
    pre_sig   = out_sorted[out_sorted["bucket"]=="pre_signal"].copy()
    event     = out_sorted[out_sorted["bucket"]=="event"].copy()

    _safe_write_csv(OUT_DIR / "confirmed_STRONGBUY.csv", confirmed)
    _safe_write_csv(OUT_DIR / "anticipative_pre_signals.csv", pre_sig)
    _safe_write_csv(OUT_DIR / "event_driven_signals.csv", event)

    # debug & diag
    _safe_write_csv(OUT_DIR / "debug_all_candidates.csv", out_sorted)
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    _mirror_to_public([
        "candidates_all_ranked.csv",
        "confirmed_STRONGBUY.csv",
        "anticipative_pre_signals.csv",
        "event_driven_signals.csv",
        "debug_all_candidates.csv",
        "diagnostics.json",
    ])

    _log(f"[SAVE] confirmed={len(confirmed)} pre={len(pre_sig)} event={len(event)}")

def _update_signals_history(today: str, out: pd.DataFrame) -> None:
    path = OUT_DIR / "signals_history.csv"
    cols = ["date","ticker_yf","sector","bucket","tv_reco","tv_score"]
    new = out[["ticker_yf","sector","bucket","tv_reco","tv_score"]].copy()
    new.insert(0, "date", today)
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=cols)
        hist = pd.concat([old, new], ignore_index=True)
    else:
        hist = new
    hist = hist.tail(2000)
    _safe_write_csv(path, hist)
    _mirror_to_public(["signals_history.csv"])
    _log(f"[SAVE] signals_history.csv | rows={len(hist)}")

def main():
    _log("[STEP] mix_ab_screen_indices starting…")
    pd.set_option("future.no_silent_downcasting", True)

    uni = _ensure_universe()

    # Pré-cache OHLCV (accélère)
    _ensure_ohlcv_cached(uni["ticker_yf"].tolist(), OHLCV_WINDOW_DAYS)

    base = _build_rows(uni)
    base = _apply_filters(base)

    # Hystérésis & seuils adaptatifs
    hist_keep = _hysteresis_mask(OUT_DIR / "signals_history.csv", HYSTERESIS_DAYS)
    confirmed_thr, presignal_thr = _adaptive_thresholds(base, hist_keep)

    # Sanity pré-attribution (projection brute par score)
    _est_conf = int((base['tv_score'] >= confirmed_thr).sum())
    _est_pre  = int(((base['tv_score'] >= presignal_thr) & (base['tv_score'] < confirmed_thr)).sum())
    _log(f"[ADAPT] thresholds used: confirmed={confirmed_thr:.3f} pre={presignal_thr:.3f} (est. counts: conf≥thr={_est_conf}, pre-range={_est_pre})")

    # Buckets
    base["bucket"] = _assign_buckets(base, confirmed_thr, presignal_thr)

    # Sorties
    run_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    today = pd.Timestamp.utcnow().date().isoformat()

    diag = {
        "run_ts": run_ts,
        "thresholds": {
            "confirmed_used": confirmed_thr,
            "presignal_used": presignal_thr,
            "event_used": EVENT_THRESHOLD
        },
        "liquidity_mins": {
            "ADV_CONFIRMED_MIN": ADV_CONFIRMED_MIN,
            "ADV_PRESIGNAL_MIN": ADV_PRESIGNAL_MIN
        },
        "coverage": {
            "universe": int(len(uni)),
            "post_filter": int(len(base)),
            "confirmed_count": int((base["bucket"]=="confirmed").sum()),
            "pre_count": int((base["bucket"]=="pre_signal").sum()),
            "event_count": int((base["bucket"]=="event").sum())
        }
    }

    _write_outputs(base, run_ts, diag)
    _update_signals_history(today, base)

    # Health-check : top 10 confirmed (ticker, score, RS%, ADV, MCAP)
    confirmed = base[base["bucket"]=="confirmed"].sort_values(["rank_score","avg_dollar_vol"], ascending=[False, False]).head(10)
    if not confirmed.empty:
        _log("[TOP] confirmed (ticker, score, sec%, ADV$, MCAP$):")
        for _, r in confirmed.iterrows():
            adv = int(r['avg_dollar_vol'] or 0)
            mcap = ("{:,}".format(int(r['mcap'])) if pd.notna(r['mcap']) else "—")
            _log(f"  {r['ticker_yf']:<8}  score={r['tv_score']:.3f}  sec%={r['sector_pct']:.2f}  ADV={adv:,}  MCAP={mcap}")
    else:
        _log("[TOP] confirmed: (aucun)")

    _log(f"[THRESHOLDS] confirmed={confirmed_thr:.3f} pre={presignal_thr:.3f} event={EVENT_THRESHOLD:.3f}")

if __name__ == "__main__":
    main()
