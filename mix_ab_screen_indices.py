# mix_ab_screen_indices.py
# Screener V2 — univers filtré (<75B + secteurs), cascade prix (YF→Finnhub→AV),
# cascade analyst/reco (Yahoo→TradingView→Finnhub), agrégation & exports.

import os
import time
import math
import json
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ======================== CONFIG ==============================

# Fichier d'entrée (créé par build_universe.py + build_sector_catalog.py)
UNI_PATH = Path("universe_in_scope.csv")

# Batch Yahoo pour prix
YF_BATCH_SIZE = 80
YF_THREADS = True
YF_CHUNK_SLEEP = (1.2, 2.5)  # pause entre chunks pour éviter throttling

# Fenêtre de TA (jours calendaires ~ double pour marge)
TA_LOOKBACK = 260  # ~1 an de bourse

# Seed & slow APIs
TA_TOP_PERCENT = 0.20
TA_TOP_MIN = 250
TA_TOP_MAX = 500

# Externals — toggle
USE_YF_ANALYST = True      # yfinance.get_info().recommendationKey
USE_TV         = True      # TradingView scanner (best effort)
USE_FINNHUB    = True
USE_AV         = True      # Alpha Vantage fallback pour PRIX

# API keys (ENV > défauts “gratuits”)
FINNHUB_API_KEY      = os.environ.get("FINNHUB_API_KEY", "d2sfah1r01qiq7a429ugd2sfah1r01qiq7a429v0").strip()
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "U0GKL47NOC9PS0UL").strip()

FINNHUB_BASE = "https://finnhub.io/api/v1"
AV_BASE      = "https://www.alphavantage.co/query"
TV_SCAN_URL  = "https://scanner.tradingview.com/america/scan"

# Règles agrégées
ANALYST_MIN_VOTES = 10
CONFIRMED_FROM_TA_TOP = 10
CONFIRMED_ANALYST_IS_BUY = {"BUY", "STRONG BUY", "OUTPERFORM", "OVERWEIGHT"}

# I/O
PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
}

# ===============================================================

def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _save(df: pd.DataFrame, name: str):
    if df is None: df = pd.DataFrame()
    df.to_csv(name, index=False)
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

# ------------------------- UNIVERSE ----------------------------

def load_universe() -> pd.DataFrame:
    if not UNI_PATH.exists():
        raise SystemExit("❌ universe_in_scope.csv introuvable – lance d’abord build_universe.py puis build_sector_catalog.py")
    df = pd.read_csv(UNI_PATH)
    req = {"ticker_yf", "sector"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"❌ colonnes manquantes dans universe_in_scope.csv: {miss}")
    df["ticker_yf"] = df["ticker_yf"].astype(str)
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    df = df.dropna(subset=["ticker_yf"]).drop_duplicates("ticker_yf")
    print(f"[UNI] loaded {len(df)} tickers (<75B + sectors in-scope)")
    return df

# --------------------- PRICE PROVIDERS ------------------------

class ProviderState:
    def __init__(self, name, cooldown_sec=60):
        self.name = name
        self.cooldown_sec = cooldown_sec
        self.next_ok = 0.0
        self.errors = 0

    def available(self):
        return time.time() >= self.next_ok

    def penalize(self, factor=1.0):
        self.errors += 1
        backoff = min(300, self.cooldown_sec * (1 + self.errors) * factor)
        self.next_ok = time.time() + backoff
        print(f"[{self.name}] cooldown {int(backoff)}s (errors={self.errors})")

    def reward(self):
        self.errors = max(0, self.errors - 1)
        self.next_ok = time.time()

def yf_download_batch(tickers, start, end) -> dict[str, pd.DataFrame]:
    res = {}
    for i in range(0, len(tickers), YF_BATCH_SIZE):
        chunk = tickers[i:i+YF_BATCH_SIZE]
        try:
            df = yf.download(
                tickers=chunk, start=start, end=end,
                interval="1d", auto_adjust=True, actions=False,
                threads=YF_THREADS, group_by="ticker", progress=False
            )
        except Exception as e:
            print(f"[YF] chunk err: {e}")
            df = None

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                for t in chunk:
                    try:
                        sub = df[t][["Close"]].rename(columns={"Close": "close"}).dropna()
                        sub.index = pd.to_datetime(sub.index)
                        if not sub.empty:
                            res[t] = sub
                    except Exception:
                        pass
            else:
                try:
                    sub = df[["Close"]].rename(columns={"Close": "close"}).dropna()
                    sub.index = pd.to_datetime(sub.index)
                    if len(chunk) == 1 and not sub.empty:
                        res[chunk[0]] = sub
                except Exception:
                    pass

        time.sleep(random.uniform(*YF_CHUNK_SLEEP))
    return res

def finnhub_daily_series(ticker, start_ts, end_ts):
    if not (USE_FINNHUB and FINNHUB_API_KEY): return None
    try:
        r = requests.get(
            f"{FINNHUB_BASE}/stock/candle",
            params={"symbol": ticker, "resolution": "D", "from": start_ts, "to": end_ts, "token": FINNHUB_API_KEY},
            headers=UA, timeout=15
        )
        j = r.json()
        if j.get("s") != "ok": return None
        ts = np.array(j["t"], dtype="int64")
        close = np.array(j["c"], dtype="float64")
        if len(ts) == 0: return None
        ds = pd.DataFrame({"close": close}, index=pd.to_datetime(ts, unit="s"))
        return ds
    except Exception:
        return None

def av_daily_series(ticker):
    if not (USE_AV and ALPHAVANTAGE_API_KEY): return None
    try:
        r = requests.get(AV_BASE, params={
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "apikey": ALPHAVANTAGE_API_KEY,
            "outputsize": "compact"
        }, headers=UA, timeout=20)
        j = r.json()
        key = next((k for k in j.keys() if "Time Series" in k), None)
        if not key: return None
        df = pd.DataFrame(j[key]).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={"5. adjusted close": "close", "4. close": "close"})
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[["close"]].dropna()
        return df
    except Exception:
        return None

def get_prices_with_cascade(tickers: list[str], lookback_days=260) -> dict[str, pd.DataFrame]:
    start = (pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=lookback_days*2)).normalize()
    end   = (pd.Timestamp.today(tz="UTC") + pd.Timedelta(days=3)).normalize()

    yf_state  = ProviderState("YF", cooldown_sec=60)
    fin_state = ProviderState("FINNHUB", cooldown_sec=30)
    av_state  = ProviderState("ALPHAVANTAGE", cooldown_sec=60)

    have = {}
    try:
        have = yf_download_batch(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        yf_state.reward()
    except Exception as e:
        print(f"[YF] batch exception: {e}")
        yf_state.penalize(1.5)

    missing = [t for t in tickers if t not in have]

    if missing and fin_state.available():
        fts, tts = int(start.timestamp()), int(end.timestamp())
        for t in list(missing):
            ds = finnhub_daily_series(t, fts, tts)
            if ds is not None and not ds.empty:
                have[t] = ds
        fin_state.reward()
        missing = [t for t in missing if t not in have]
    else:
        if missing:
            print("[FINNHUB] skip (cooldown/disabled)")

    if missing and av_state.available():
        for t in list(missing):
            ds = av_daily_series(t)
            if ds is not None and not ds.empty:
                have[t] = ds
            time.sleep(12.5)  # free tier 5 req/min
        av_state.reward()
        missing = [t for t in missing if t not in have]
    else:
        if missing:
            print("[AV] skip (cooldown/disabled)")

    print(f"[PRICES] got {len(have)}/{len(tickers)} with cascade (YF→Finnhub→AV). Missing: {len(missing)}")
    return have

# --------------------- TA & SCORING ----------------------------

def ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["close","rsi","sma20","sma50","sma200","macd","signal","hist","stoch"])
    s = df.copy()
    c = s["close"]
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll = 14
    ma_up = up.rolling(roll, min_periods=roll).mean()
    ma_down = down.rolling(roll, min_periods=roll).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    sma20  = c.rolling(20,  min_periods=5).mean()
    sma50  = c.rolling(50,  min_periods=10).mean()
    sma200 = c.rolling(200, min_periods=20).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    low14  = c.rolling(14, min_periods=5).min()
    high14 = c.rolling(14, min_periods=5).max()
    stoch = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    out = s.copy()
    out["rsi"] = rsi
    out["sma20"] = sma20
    out["sma50"] = sma50
    out["sma200"] = sma200
    out["macd"] = macd
    out["signal"] = signal
    out["hist"] = hist
    out["stoch"] = stoch
    return out

def local_score_from_indic(ind: pd.DataFrame) -> float:
    if ind is None or ind.empty:
        return 0.0
    last = ind.iloc[-1]
    pts = 0.0
    rsi = last.get("rsi", np.nan)
    if np.isfinite(rsi):
        if rsi >= 70: pts += 25
        elif rsi >= 60: pts += 18
        elif rsi >= 55: pts += 12
        elif rsi >= 50: pts += 6
    s20, s50, s200 = last.get("sma20", np.nan), last.get("sma50", np.nan), last.get("sma200", np.nan)
    if all(np.isfinite(x) for x in (s20,s50,s200)):
        if s20 > s50 > s200: pts += 25
        elif s20 > s50: pts += 12
    close = last.get("close", np.nan)
    if np.isfinite(close) and np.isfinite(s20) and np.isfinite(s50) and np.isfinite(s200):
        if close > s20: pts += 8
        if close > s50: pts += 8
        if close > s200: pts += 8
    h = last.get("hist", np.nan)
    if np.isfinite(h):
        if h > 0: pts += 10
    st = last.get("stoch", np.nan)
    if np.isfinite(st):
        if st >= 80: pts += 6
        elif st >= 60: pts += 3
    return float(min(100.0, pts))

def local_label(ind: pd.DataFrame) -> str:
    if ind is None or ind.empty:
        return ""
    last = ind.iloc[-1]
    close = last.get("close", np.nan)
    s20, s50, s200 = last.get("sma20", np.nan), last.get("sma50", np.nan), last.get("sma200", np.nan)
    rsi = last.get("rsi", np.nan)
    macd, signal = last.get("macd", np.nan), last.get("signal", np.nan)
    if all(np.isfinite(x) for x in (close, s20, s50, s200, rsi, macd, signal)):
        if (s20 > s50 > s200) and (close > s20 > s50) and (rsi >= 60) and (macd > signal):
            return "Strong Buy"
        if (close > s50) and (rsi >= 55):
            return "Buy"
        if (close < s50) and (rsi <= 45):
            return "Sell"
    return "Neutral"

# ------------------------ ANALYSTS/RECO ------------------------

def fetch_yf_analyst(ticker: str) -> dict:
    if not USE_YF_ANALYST:
        return {}
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return {}
    out = {}
    rec = info.get("recommendationKey")
    if isinstance(rec, str):
        out["analyst_bucket"] = rec.replace("_", " ").upper()
    out["analyst_votes"] = None
    return out

def guess_tv_tickers(yf_symbol: str) -> list[str]:
    """
    Heuristique simple : on tente NASDAQ:SYMBOL puis NYSE:SYMBOL (US).
    Tu peux enrichir plus tard si besoin.
    """
    sym = yf_symbol.replace("-", ".").upper()
    return [f"NASDAQ:{sym}", f"NYSE:{sym}"]

def map_tv_recommend_to_bucket(x: float | int | None) -> str | None:
    """
    TradingView 'Recommend.All' ~ [-1..+1]. Mapping heuristique en bucket.
      >= +0.5  -> STRONG BUY
      >= +0.2  -> BUY
      >  -0.2  -> HOLD
      <= -0.5  -> STRONG SELL
      else     -> SELL
    """
    if x is None or not np.isfinite(x):
        return None
    v = float(x)
    if v >= 0.5:  return "STRONG BUY"
    if v >= 0.2:  return "BUY"
    if v > -0.2:  return "HOLD"
    if v <= -0.5: return "STRONG SELL"
    return "SELL"

def fetch_tv_recommend(yf_symbol: str) -> dict:
    if not USE_TV:
        return {}
    tickers = guess_tv_tickers(yf_symbol)
    payload = {
        "symbols": {"tickers": tickers, "query": {"types": []}},
        "columns": ["Recommend.All"]
    }
    try:
        r = requests.post(TV_SCAN_URL, json=payload, headers=UA, timeout=15)
        if r.status_code != 200:
            return {}
        j = r.json()
        data = j.get("data") or []
        # on prend la 1ère ligne valide
        for row in data:
            arr = row.get("d") or []
            if not arr:
                continue
            rec_all = arr[0]
            bucket = map_tv_recommend_to_bucket(rec_all)
            if bucket:
                # On expose sous tv_reco, et on peut aussi s'en servir comme analyst_bucket fallback
                return {"tv_reco": bucket.replace(" ", "_").upper(), "tv_score": float(rec_all)}
        return {}
    except Exception:
        return {}

def fetch_finnhub_rating(ticker: str) -> dict:
    if not (USE_FINNHUB and FINNHUB_API_KEY):
        return {}
    try:
        r = requests.get(
            f"{FINNHUB_BASE}/stock/recommendation",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            headers=UA, timeout=15
        )
        arr = r.json()
        if not isinstance(arr, list) or not arr:
            return {}
        row = arr[0]
        buy = (row.get("strongBuy") or 0) + (row.get("buy") or 0)
        sell= (row.get("strongSell") or 0) + (row.get("sell") or 0)
        hold= (row.get("hold") or 0)
        votes = int(buy + sell + hold)
        if buy >= max(sell, hold):
            bucket = "BUY" if buy > hold else "HOLD"
        elif sell > buy:
            bucket = "SELL"
        else:
            bucket = "HOLD"
        return {"analyst_bucket": bucket, "analyst_votes": votes}
    except Exception:
        return {}

# ---------------------- AGGREGATION ----------------------------

def aggregate_row(base: dict) -> dict:
    local_score = base.get("local_score", 0.0) or 0.0

    # analyst bucket/votes (priorité: Finnhub si votes, sinon Yahoo)
    analyst_bucket = base.get("analyst_bucket")
    analyst_votes  = base.get("analyst_votes")

    # tv_reco (texte) peut aider si on n’a PAS d’analyst_bucket fiable
    tv_reco = base.get("tv_reco")
    tv_score = base.get("tv_score")

    # score “analyst” (bucket → 0..100)
    def analyst_bucket_to_score(b):
        if not b: return 0.0
        x = str(b).upper()
        if x in {"STRONG BUY", "BUY", "OUTPERFORM", "OVERWEIGHT"}:
            return 100.0
        if x in {"HOLD"}: return 50.0
        if x in {"SELL", "STRONG SELL", "UNDERPERFORM"}: return 0.0
        return 50.0

    analyst_score = analyst_bucket_to_score(analyst_bucket)

    # si pas d’analyst_bucket mais on a tv_reco -> on l’utilise
    if not analyst_bucket and tv_reco:
        analyst_score = analyst_bucket_to_score(tv_reco.replace("_", " "))

    # pondération simple
    rank_score = 0.70 * float(local_score) + 0.30 * float(analyst_score)

    out = dict(base)
    out["rank_score"] = round(rank_score, 2)
    out["analyst_bucket"] = analyst_bucket
    out["analyst_votes"] = analyst_votes
    out["tv_reco"] = tv_reco
    out["tv_score"] = tv_score
    return out

def determine_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["bucket"] = []
        return df
    df = df.copy()
    df["__ta_rank"] = df["local_score"].rank(method="first", ascending=False)

    def is_best_analyst(row):
        # analyst_bucket prioritaire ; sinon on tolère tv_reco comme proxy
        b = str(row.get("analyst_bucket") or row.get("tv_reco") or "").upper().replace("_", " ")
        v = row.get("analyst_votes")
        return (b in CONFIRMED_ANALYST_IS_BUY) and (v is None or v >= ANALYST_MIN_VOTES)

    df["bucket"] = np.where(
        (df["__ta_rank"] <= CONFIRMED_FROM_TA_TOP) & (df.apply(is_best_analyst, axis=1)),
        "confirmed",
        ""
    )
    df.loc[(df["bucket"] == "") & (df["__ta_rank"] <= 100), "bucket"] = "pre_signal"
    has_any_analyst = df["analyst_bucket"].notna() | df["tv_reco"].notna()
    df.loc[(df["bucket"] == "") & has_any_analyst, "bucket"] = "event"
    return df.drop(columns="__ta_rank")

# ------------------------ HISTORY ------------------------------

def append_signals_history(df: pd.DataFrame):
    out_cols = [
        "date","ticker_yf","sector",
        "local_label","tv_reco","analyst_bucket","analyst_votes",
        "local_score","rank_score"
    ]
    today = pd.Timestamp.today(tz="UTC").date().isoformat()
    snippet = df.copy()
    snippet["date"] = today
    snippet = snippet[out_cols].copy()
    path = Path("signals_history.csv")
    if path.exists():
        old = pd.read_csv(path)
    else:
        old = pd.DataFrame(columns=out_cols)
    merged = pd.concat([old, snippet], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date","ticker_yf"], keep="last")
    merged.to_csv("signals_history.csv", index=False)
    merged.to_csv(PUBLIC_DIR / "signals_history.csv", index=False)
    print(f"[OK] signals_history.csv appended ({len(snippet)} today, total {len(merged)})")

# -------------------------- MAIN -------------------------------

def main():
    uni = load_universe()
    tickers = uni["ticker_yf"].astype(str).tolist()

    # Pass 1: TA locale avec cascade prix
    px_map = get_prices_with_cascade(tickers, lookback_days=TA_LOOKBACK)

    rows = []
    for t in tickers:
        px = px_map.get(t)
        if px is None or px.empty:
            continue
        ind = ta_indicators(px)
        score = local_score_from_indic(ind)
        label = local_label(ind)
        last_price = float(ind["close"].iloc[-1]) if not ind.empty else np.nan

        rows.append({
            "ticker_yf": t,
            "sector": uni.loc[uni["ticker_yf"] == t, "sector"].iloc[0] if (uni["ticker_yf"] == t).any() else "Unknown",
            "price": last_price,
            "local_score": score,
            "local_label": label,
        })

    base = pd.DataFrame(rows)
    if base.empty:
        _save(pd.DataFrame(columns=["ticker_yf","sector","price","local_score","local_label"]), "candidates_all_ranked.csv")
        _save(pd.DataFrame(columns=[]), "confirmed_STRONGBUY.csv")
        _save(pd.DataFrame(columns=[]), "anticipative_pre_signals.csv")
        _save(pd.DataFrame(columns=[]), "event_driven_signals.csv")
        return

    base = base.sort_values("local_score", ascending=False).reset_index(drop=True)
    k = max(TA_TOP_MIN, min(TA_TOP_MAX, int(math.ceil(len(base) * TA_TOP_PERCENT))))
    seed = base.head(k).copy()
    print(f"[SEED] top local_score: {k}/{len(base)}")

    # Pass 2: externes (cascade analyst/reco : Yahoo → TradingView → Finnhub)
    ext_rows = []
    for _, r in seed.iterrows():
        t = r["ticker_yf"]
        ext = {}

        # 1) Yahoo
        yfa = fetch_yf_analyst(t)
        if yfa.get("analyst_bucket"):
            ext.update(yfa)
        else:
            # 2) TradingView (proxy reco)
            tvr = fetch_tv_recommend(t)
            if tvr.get("tv_reco"):
                ext.update(tvr)
            # 3) Finnhub si toujours rien d'analyst
            if not yfa.get("analyst_bucket"):
                fin = fetch_finnhub_rating(t)
                # si Finnhub fournit un bucket, il prime sur TV pour analyst_bucket
                if fin.get("analyst_bucket"):
                    ext.update(fin)
                else:
                    # sinon on garde TV uniquement comme tv_reco
                    pass

        merged = dict(r)
        merged.update(ext)
        ext_rows.append(aggregate_row(merged))

    rich = pd.DataFrame(ext_rows)
    rich = determine_buckets(rich)
    ranked = rich.sort_values(["rank_score","local_score"], ascending=[False, False]).reset_index(drop=True)

    # Exports
    _save(ranked, "candidates_all_ranked.csv")

    confirmed = ranked[ranked["bucket"] == "confirmed"].copy()
    pre       = ranked[ranked["bucket"] == "pre_signal"].copy()
    events    = ranked[ranked["bucket"] == "event"].copy()

    for df in (confirmed, pre, events):
        if "tv_reco" not in df.columns: df["tv_reco"] = ""
        if "analyst_bucket" not in df.columns: df["analyst_bucket"] = ""
        if "analyst_votes" not in df.columns: df["analyst_votes"] = np.nan
        df["ticker_tv"] = df["ticker_yf"].str.replace("-", ".", regex=False)

    _save(confirmed, "confirmed_STRONGBUY.csv")
    _save(pre, "anticipative_pre_signals.csv")
    _save(events, "event_driven_signals.csv")

    append_signals_history(ranked)
    print("[DONE] screener V2 (cascade prix + analyst Yahoo→TV→Finnhub) complete.")

if __name__ == "__main__":
    main()
