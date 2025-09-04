# mix_ab_screen_indices.py
# Screener US — Gratuit, core local rapide + Finnhub (sniper) + TV (bonus capé)
# Sorties :
#   - confirmed_STRONGBUY.csv
#   - anticipative_pre_signals.csv
#   - event_driven_signals.csv
#   - candidates_all_ranked.csv   (inclut 'universe' = tout le seed filtré)
#   - universe_today.csv
#   - debug_all_candidates.csv
#   - sector_breadth.csv
#   - sector_history.csv
#   - signals_history.csv
#   - raw_candidates.csv
#   - tv_failures.csv (diagnostic TradingView)

import warnings, time, os, math, random
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tradingview_ta import TA_Handler, Interval
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ============= CONFIG =============

# --- Univers à inclure ---
INCLUDE_RUSSELL_1000 = True
INCLUDE_RUSSELL_2000 = True
INCLUDE_SP500        = True
INCLUDE_NASDAQ_COMPOSITE = True

# --- Fichiers fallback ---
R1K_FALLBACK_CSV = "russell1000.csv"        # 1 col: Ticker
R2K_FALLBACK_CSV = "russell2000.csv"        # 1 col: Ticker (généré par gen_r2000_from_ishares.py)
SP500_FALLBACK_CSV = "sp500.csv"            # 1 col: Ticker
NASDAQ_COMPOSITE_FALLBACK_CSV = "nasdaq_composite.csv"  # 1 col: Ticker

# --- Source Russell 2000 ---
# "csv"  : utilise exclusivement le CSV généré par gen_r2000_from_ishares.py
# "wiki" : force Wikipédia (⚠️ souvent incomplet)
# "auto" : tente wiki, puis fallback CSV si <1000 tickers
R2K_SOURCE = "csv"

# Fenêtre (suffisant pour SMA200/TA classiques)
PERIOD   = "1y"
INTERVAL = "1d"
TV_INTERVAL = Interval.INTERVAL_1_DAY

# Limites & priorisation
GATE_MIN_TECH_VOTES = 2                 # sert à prioriser les appels externes (pré-score)
MAX_MARKET_CAP      = 75_000_000_000    # < 75B
KEEP_FULL_UNIVERSE_IN_OUTPUTS = True    # garde tout le seed dans les outputs (même si externes manquants)

# === Ciblage externe sur un sous-ensemble (sniper mode) ===
EXTERNAL_TARGET_MODE = "topk"   # "topk" ou "toppct"
EXTERNAL_TOP_K = 150            # ~10–20% de l’univers
EXTERNAL_TOP_PCT = 0.15         # si mode "toppct", 15% de l’univers
EXTERNAL_TOP_MIN = 80           # minimum ciblé

# ---- Confirmed tuning (assoupli & réaliste) ----
CONFIRM_MIN_TECH_VOTES = 4              # TA forte
CONFIRM_MIN_ANALYST_VOTES = 5           # nb mini d'avis analystes pour valider "Buy"
CONFIRM_MIN_CORROBORATIONS = 1          # nombre d'arguments externes requis si final_signal == STRONG_BUY
MIN_AVG_DAILY_VOLUME = 150_000          # filtre liquidité (ten_day / 3m avg volume)

# Nasdaq Composite (API nasdaq.com)
NASDAQ_EXCLUDE_ETFS = True  # filtre basique des ETF/ETN/Funds/Trusts par le libellé

# TradingView : bonus capé (ratelimit fréquent)
TV_MAX_CALLS = 150
DELAY_BETWEEN_TV_CALLS_SEC = 0.10
TV_COOLDOWN_EVERY = 25
TV_COOLDOWN_SEC   = 2.0
TV_MAX_CONSEC_FAIL = 10
TV_FAIL_PAUSE_SEC  = 6.0
TV_MAX_PAUSE_TOTAL = 45.0

# Finnhub (gratuit)
USE_FINNHUB = True
FINNHUB_API_KEY = (os.environ.get("FINNHUB_API_KEY") or "d2sfah1r01qiq7a429ugd2sfah1r01qiq7a429v0").strip()
FINNHUB_RESOLUTION = "D"
FINNHUB_MAX_CALLS  = 350
FINNHUB_SLEEP_BETWEEN = 0.10
FINNHUB_TIMEOUT_SEC = 20

# Alpha Vantage (optionnel, désactivé)
USE_ALPHA_VANTAGE = False
ALPHAVANTAGE_API_KEY = (os.environ.get("ALPHAVANTAGE_API_KEY") or "85SZZGRDDJ6MUAEX").strip()
AV_MAX_CALLS = 20
AV_SLEEP_BETWEEN = 12.5

SECTOR_CATALOG_PATH = "sector_catalog.csv"  # optionnel

# --- Sorties : racine + copie dans dashboard/public/ pour le front
PUBLIC_DIR = os.path.join("dashboard", "public")

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, fname: str, also_public: bool = True):
    _ensure_dir(fname)
    df.to_csv(fname, index=False)
    if also_public:
        dst = os.path.join(PUBLIC_DIR, fname)
        _ensure_dir(dst)
        df.to_csv(dst, index=False)

# ============= HELPERS =============
def yf_norm(s:str)->str: return s.replace(".", "-")
def tv_norm(s:str)->str: return s.replace("-", ".")
def base_sym(tv_symbol: str)->str:
    return (tv_symbol or "").split(".")[0].upper()

def fetch_wikipedia_tickers(url: str):
    ua = {"User-Agent":"Mozilla/5.0"}
    r = requests.get(url, headers=ua, timeout=45)
    r.raise_for_status()
    tables = pd.read_html(r.text)

    def flatten(cols):
        out=[]
        for c in cols:
            if isinstance(c, tuple):
                out.append(" ".join([str(x) for x in c if pd.notna(x)]).strip())
            else:
                out.append(str(c).strip())
        return out

    for t in tables:
        cols = flatten(t.columns)
        lower = [c.lower() for c in cols]
        col_idx = None
        for i, name in enumerate(lower):
            if ("ticker" in name) or ("symbol" in name):
                col_idx = i; break
        if col_idx is None: continue
        ser = (t[t.columns[col_idx]].astype(str).str.strip()
               .str.replace(r"\s+","",regex=True)
               .str.replace("\u200b","",regex=False))
        vals = ser[ser.str.match(r"^[A-Za-z.\-]+$")].dropna().tolist()
        if vals: return vals
    raise RuntimeError(f"Aucune colonne Ticker/Symbol trouvée sur {url}")

def fetch_sp500_tickers():
    """Récupère la liste S&P 500 depuis Wikipédia (stable)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ua = {"User-Agent":"Mozilla/5.0"}
    r = requests.get(url, headers=ua, timeout=45)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c or "ticker" in c for c in cols):
            ser = t[t.columns[0]].astype(str).str.strip()
            vals = ser[ser.str.match(r"^[A-Za-z.\-]+$")].dropna().tolist()
            if vals:
                return vals
    return []

def fetch_nasdaq_composite():
    """
    Récupère tous les tickers listés sur le NASDAQ via l'endpoint screener.
    NB: c'est 'tout Nasdaq' (Composite-like), pas seulement le Nasdaq-100.
    """
    url = "https://api.nasdaq.com/api/screener/stocks"
    params = {"tableonly": "true", "limit": "9999", "exchange": "nasdaq"}  # 9999 pour tout ramener
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
        "Origin": "https://www.nasdaq.com",
        "Connection": "keep-alive",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=45)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        raise RuntimeError(f"Nasdaq API error: {e}")

    rows = (((js or {}).get("data") or {}).get("table") or {}).get("rows") or []
    out = []
    for row in rows:
        sym = str(row.get("symbol") or "").strip().upper()
        name = str(row.get("name") or "").strip().upper()
        if NASDAQ_EXCLUDE_ETFS:
            if any(tag in name for tag in ("ETF", "ETN", "FUND", "TRUST")):
                continue
        if sym and pd.notna(sym) and (len(sym) <= 8) and \
           (pd.Series([sym]).str.match(r"^[A-Z.\-]+$").iloc[0]):
            out.append(sym)
    return list(dict.fromkeys(out))  # dédoublonne en gardant l'ordre

def _read_tickers_csv(path: str, colname: str = "Ticker") -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV introuvable: {path}")
    df = pd.read_csv(path)
    if colname not in df.columns:
        colname = df.columns[0]
    ser = (
        df[colname].astype(str).str.strip().str.upper()
        .replace("", np.nan).dropna()
    )
    ser = ser[ser.str.match(r"^[A-Z.\-]+$")]
    return list(dict.fromkeys(ser.tolist()))

def load_universe()->pd.DataFrame:
    ticks = []

    # --- Russell 1000
    if INCLUDE_RUSSELL_1000:
        try:
            r1k = fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_1000_Index")
            ticks += r1k
            print(f"[UNIV] Russell 1000: {len(r1k)} tickers")
        except Exception as e:
            print(f"[WARN] Russell 1000 (Wiki) échec: {e}")
            if os.path.exists(R1K_FALLBACK_CSV):
                r1k_fb = pd.read_csv(R1K_FALLBACK_CSV)["Ticker"].astype(str).tolist()
                ticks += r1k_fb
                print(f"[UNIV] Russell 1000 fallback CSV: {len(r1k_fb)} tickers")

    # --- Russell 2000 (CSV prioritaire)
    if INCLUDE_RUSSELL_2000:
        if R2K_SOURCE.lower() == "csv":
            try:
                r2k = _read_tickers_csv(R2K_FALLBACK_CSV, colname="Ticker")
                ticks += r2k
                print(f"[UNIV] Russell 2000 (CSV): {len(r2k)} tickers")
            except Exception as e:
                print(f"[WARN] Russell 2000 CSV échec: {e} — aucun R2K ajouté")
        elif R2K_SOURCE.lower() == "wiki":
            try:
                r2k = fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_2000_Index")
                ticks += r2k
                print(f"[UNIV] Russell 2000 (Wiki): {len(r2k)} tickers")
            except Exception as e:
                print(f"[WARN] Russell 2000 (Wiki) échec: {e} — aucun R2K ajouté (pas de fallback CSV en mode wiki)")
        else:  # auto
            r2k = []
            try:
                r2k = fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_2000_Index")
                print(f"[UNIV] Russell 2000 (Wiki tentative): {len(r2k)} tickers")
            except Exception as e:
                print(f"[WARN] Russell 2000 (Wiki) échec: {e}")
            if len(r2k) < 1000:
                try:
                    r2k_csv = _read_tickers_csv(R2K_FALLBACK_CSV, colname="Ticker")
                    if r2k_csv:
                        r2k = r2k_csv
                        print(f"[UNIV] Russell 2000 (fallback CSV): {len(r2k)} tickers")
                except Exception as e:
                    print(f"[WARN] Russell 2000 fallback CSV échec: {e}")
            ticks += r2k

    # --- S&P 500
    if INCLUDE_SP500:
        try:
            sp = fetch_sp500_tickers()
            ticks += sp
            print(f"[UNIV] S&P 500: {len(sp)} tickers")
        except Exception as e:
            print(f"[WARN] S&P 500 (Wiki) échec: {e}")
            if os.path.exists(SP500_FALLBACK_CSV):
                sp_fb = pd.read_csv(SP500_FALLBACK_CSV)["Ticker"].astype(str).tolist()
                ticks += sp_fb
                print(f"[UNIV] S&P 500 fallback CSV: {len(sp_fb)} tickers")

    # --- Nasdaq Composite-like (API nasdaq.com)
    if INCLUDE_NASDAQ_COMPOSITE:
        try:
            ndaq = fetch_nasdaq_composite()
            ticks += ndaq
            print(f"[UNIV] Nasdaq Composite-like (API): {len(ndaq)} tickers (ETF filtered={NASDAQ_EXCLUDE_ETFS})")
        except Exception as e:
            print(f"[WARN] Nasdaq API échec: {e}")
            if os.path.exists(NASDAQ_COMPOSITE_FALLBACK_CSV):
                nzfb = pd.read_csv(NASDAQ_COMPOSITE_FALLBACK_CSV)["Ticker"].astype(str).tolist()
                ticks += nzfb
                print(f"[UNIV] Nasdaq fallback CSV: {len(nzfb)} tickers")

    if not ticks:
        raise RuntimeError("Impossible de charger l’univers (Wikipédia/CSV/Api).")

    # Normalisation TV/YF + dédoublonnage
    tv_syms = [tv_norm(s) for s in ticks]
    yf_syms = [yf_norm(s) for s in ticks]
    df = pd.DataFrame({"tv_symbol": tv_syms, "yf_symbol": yf_syms}).drop_duplicates().reset_index(drop=True)
    print(f"[UNIV] Total combiné (unique): {len(df)}")
    return df

def map_exchange_for_tv(exch: str)->str:
    e = (exch or "").upper()
    if "NASDAQ" in e: return "NASDAQ"
    if "NYSE"   in e: return "NYSE"
    if "ARCA"   in e: return "AMEX"
    if "BATS"   in e: return "AMEX"
    if "NYSEMKT" in e: return "AMEX"
    return "NASDAQ"

# ============= TA local & labels =============
def compute_local_technical_bucket(hist: pd.DataFrame):
    """Renvoie (bucket, votes, details) — tolérante aux NaN, nécessite >=60 barres."""
    if hist is None or hist.empty or len(hist) < 60:
        return None, None, {}

    close = hist["Close"]; high = hist["High"]; low = hist["Low"]
    try:
        s20  = ta.sma(close, length=20)
        s50  = ta.sma(close, length=50)
        s200 = ta.sma(close, length=200)
        rsi  = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        stoch= ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    except Exception:
        return None, None, {}

    try:
        price     = float(close.iloc[-1])
        s20_last  = float(s20.dropna().iloc[-1]); s50_last  = float(s50.dropna().iloc[-1])
        s200_last = float(s200.dropna().iloc[-1]) if s200 is not None and not s200.dropna().empty else np.nan
        rsi_last  = float(rsi.dropna().iloc[-1])

        macd_last = macds_last = np.nan
        if macd is not None and not macd.dropna().empty and {"MACD_12_26_9","MACDs_12_26_9"}.issubset(macd.columns):
            macd_last  = float(macd["MACD_12_26_9"].dropna().iloc[-1])
            macds_last = float(macd["MACDs_12_26_9"].dropna().iloc[-1])

        stoch_k = stoch_d = np.nan
        if stoch is not None and not stoch.dropna().empty and {"STOCHk_14_3_3","STOCHd_14_3_3"}.issubset(stoch.columns):
            stoch_k = float(stoch["STOCHk_14_3_3"].dropna().iloc[-1])
            stoch_d = float(stoch["STOCHd_14_3_3"].dropna().iloc[-1])
    except Exception:
        return None, None, {}

    votes = 0
    votes += 1 if price > s20_last else -1
    votes += 1 if s20_last > s50_last else -1
    if not np.isnan(s200_last): votes += 1 if s50_last > s200_last else -1
    if rsi_last >= 55: votes += 1
    elif rsi_last <= 45: votes -= 1
    if not np.isnan(macd_last) and not np.isnan(macds_last): votes += 1 if macd_last > macds_last else -1
    if not np.isnan(stoch_k) and not np.isnan(stoch_d):     votes += 1 if stoch_k > stoch_d else -1

    if votes >= 4: bucket = "Strong Buy"
    elif votes >= 2: bucket = "Buy"
    elif votes <= -4: bucket = "Strong Sell"
    elif votes <= -2: bucket = "Sell"
    else: bucket = "Neutral"

    details = dict(price=price, sma20=s20_last, sma50=s50_last, sma200=s200_last,
                   rsi=rsi_last, macd=macd_last, macds=macds_last, stoch_k=stoch_k, stoch_d=stoch_d)
    return bucket, int(votes), details

def local_label_from_indicators(det: dict, bucket: str) -> str:
    """Composite 'TV-like' local à partir des familles MAs / Oscillateurs (resserré)."""
    score = 0
    try:
        price, s20, s50, s200 = det.get("price"), det.get("sma20"), det.get("sma50"), det.get("sma200")
        rsi, macd, macds = det.get("rsi"), det.get("macd"), det.get("macds")
        k, d = det.get("stoch_k"), det.get("stoch_d")

        # MAs (tendance)
        if price is not None and s20 is not None:
            score += 1 if price > s20 else -1
        if s20 is not None and s50 is not None:
            score += 1 if s20 > s50 else -1
        if s50 is not None and s200 is not None and not np.isnan(s200):
            score += 1 if s50 > s200 else -1

        # Oscillateurs (RSI 60/40)
        if rsi is not None:
            score += 1 if rsi >= 60 else (-1 if rsi <= 40 else 0)
        if macd is not None and macds is not None and not np.isnan(macd) and not np.isnan(macds):
            score += 1 if macd > macds else -1
        if k is not None and d is not None and not np.isnan(k) and not np.isnan(d):
            score += 1 if k > d else -1
    except Exception:
        pass

    if score >= 5:   return "STRONG_BUY"
    if score >= 3:   return "BUY"
    if score <= -5:  return "STRONG_SELL"
    if score <= -3:  return "SELL"
    return "NEUTRAL"

def analyst_bucket_from_mean(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    if x < 1.6: return "Strong Buy"
    if x < 2.5: return "Buy"
    if x < 3.5: return "Hold"
    if x < 4.2: return "Sell"
    return "Strong Sell"

# ===== BULK HISTORY + FALLBACK =====
def _bulk_download_chunk(symbols, period="1y", interval="1d"):
    data = yf.download(
        symbols, period=period, interval=interval,
        auto_adjust=True, actions=False, progress=False, group_by="ticker", threads=True
    )
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        tickers_level = data.columns.get_level_values(0)
        for sym in symbols:
            if sym in tickers_level:
                try:
                    df = data[sym].dropna()
                    if not df.empty:
                        out[sym] = df
                except Exception:
                    continue
    else:
        df = data.dropna()
        if not df.empty and len(symbols) == 1:
            out[symbols[0]] = df
    return out

def bulk_download_history(yf_symbols, period="1y", interval="1d", chunk_size=200, fallback_limit=None):
    if not yf_symbols:
        return {}
    yf_symbols = list(dict.fromkeys(yf_symbols))  # dédoublonne

    all_hist = {}
    for i in range(0, len(yf_symbols), chunk_size):
        sub = yf_symbols[i:i+chunk_size]
        try:
            part = _bulk_download_chunk(sub, period=period, interval=interval)
            all_hist.update(part)
        except Exception:
            continue

    missing = [s for s in yf_symbols if s not in all_hist]
    count_fb = 0
    for sym in missing:
        if (fallback_limit is not None) and (count_fb >= fallback_limit):
            break
        try:
            df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=True, actions=False).dropna()
            if not df.empty and len(df) >= 60:
                all_hist[sym] = df
                count_fb += 1
        except Exception:
            continue

    return all_hist

# ========= FINNHUB: label composite (RSI + MACD) =========
def _fh_get_json(url, params, timeout=FINNHUB_TIMEOUT_SEC):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _score_from_rsi_macd(rsi_val, macd_val, macds_val):
    score = 0
    if rsi_val is not None:
        try:
            r = float(rsi_val)
            if r >= 60: score += 1
            elif r <= 40: score -= 1
        except Exception:
            pass
    try:
        m = float(macd_val) if macd_val is not None else None
        ms = float(macds_val) if macds_val is not None else None
        if m is not None and ms is not None:
            score += 1 if m > ms else -1
    except Exception:
        pass
    if score >= 2: return "STRONG_BUY"
    if score == 1: return "BUY"
    if score == 0: return "NEUTRAL"
    if score == -1: return "SELL"
    return "STRONG_SELL"

def fetch_finnhub_label(symbol: str):
    """Interroge Finnhub RSI(14) et MACD(12,26,9) et produit un label simple."""
    if (not USE_FINNHUB) or (not FINNHUB_API_KEY):
        return {"finnhub_label": None, "finnhub_rsi": None, "finnhub_macd": None, "finnhub_macds": None}

    base = "https://finnhub.io/api/v1/indicator"
    # RSI
    rsi_params = {"symbol": symbol, "resolution": FINNHUB_RESOLUTION, "indicator": "rsi", "timeperiod": 14, "token": FINNHUB_API_KEY}
    rsi_json = _fh_get_json(base, rsi_params)
    rsi_val = None
    if rsi_json and isinstance(rsi_json.get("rsi"), list) and rsi_json.get("rsi"):
        rsi_val = rsi_json["rsi"][-1]

    # MACD
    macd_params = {"symbol": symbol, "resolution": FINNHUB_RESOLUTION, "indicator": "macd",
                   "fastperiod": 12, "slowperiod": 26, "signalperiod": 9, "token": FINNHUB_API_KEY}
    macd_json = _fh_get_json(base, macd_params)
    macd_val = macds_val = None
    if macd_json:
        for k in ["macd","MACD","macdValue"]:
            if isinstance(macd_json.get(k), list) and macd_json[k]:
                macd_val = macd_json[k][-1]; break
        for k in ["macd_signal","signal","MACDs","macdSignal"]:
            if isinstance(macd_json.get(k), list) and macd_json[k]:
                macds_val = macd_json[k][-1]; break

    label = _score_from_rsi_macd(rsi_val, macd_val, macds_val)
    time.sleep(FINNHUB_SLEEP_BETWEEN)
    return {"finnhub_label": label, "finnhub_rsi": rsi_val, "finnhub_macd": macd_val, "finnhub_macds": macds_val}

# ========= TRADINGVIEW (bonus) =========
TV_EXCH_TRY = ["NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"]
_tv_cache = {}; _tv_fail_log=[]

def tv_symbol_variants(tv_symbol: str, yf_symbol: str):
    s=(tv_symbol or "").upper(); y=(yf_symbol or "").upper()
    v=[]
    if s: v.append(s)
    base=base_sym(s)
    if base and base not in v: v.append(base)
    if "." in s:
        dash=s.replace(".","-"); nodot=s.replace(".","")
        if dash not in v: v.append(dash)
        if nodot not in v: v.append(nodot)
    if "-" in y:
        dot=y.replace("-",".")
        if dot not in v: v.append(dot)
        if y not in v: v.append(y)
    out=[]
    for x in v:
        x=x.strip()
        if x and x not in out: out.append(x)
    return out

def _tv_try(symbol: str, exchange: str, interval=TV_INTERVAL, retries=1):
    key=(symbol,exchange,interval)
    if key in _tv_cache: return _tv_cache[key]
    last_err=None
    for attempt in range(retries+1):
        try:
            h=TA_Handler(symbol=symbol,screener="america",exchange=exchange,interval=interval)
            reco=h.get_analysis().summary.get("RECOMMENDATION")
            _tv_cache[key]=reco
            if DELAY_BETWEEN_TV_CALLS_SEC: time.sleep(DELAY_BETWEEN_TV_CALLS_SEC+random.uniform(0,0.03))
            return reco
        except Exception as e:
            last_err=str(e); time.sleep(0.04+0.06*attempt)
    _tv_cache[key]=None; _tv_fail_log.append({"symbol":symbol,"exchange":exchange,"err":last_err or "unknown"})
    return None

def get_tv_summary_max(tv_symbol: str, yf_symbol: str, exchange_hint: str):
    candidates=tv_symbol_variants(tv_symbol,yf_symbol)
    hint=map_exchange_for_tv(exchange_hint)
    exch_list=[hint]+[e for e in TV_EXCH_TRY if e!=hint]
    for sym in candidates:
        reco=_tv_try(sym,hint)
        if reco: return {"tv_reco":reco,"tv_symbol_used":sym,"tv_exchange_used":hint}
        for ex in exch_list:
            reco=_tv_try(sym,ex)
            if reco: return {"tv_reco":reco,"tv_symbol_used":sym,"tv_exchange_used":ex}
    return {"tv_reco":None,"tv_symbol_used":None,"tv_exchange_used":None}

# ============= RANK SCORE (vectorisé) =============
def compute_rank_score(df: pd.DataFrame) -> pd.Series:
    score = np.zeros(len(df), dtype=float)

    fs = df.get("final_signal").fillna("").astype(str).str.upper()
    score += np.select(
        [fs.eq("STRONG_BUY"), fs.eq("BUY"), fs.eq("NEUTRAL"), fs.eq("SELL"), fs.eq("STRONG_SELL")],
        [3.0,                1.8,          0.0,             -1.0,         -2.0],
        default=0.0
    )

    lb = df.get("local_label").fillna("").astype(str).str.upper()
    score += np.where(lb.eq("STRONG_BUY"), 0.8, 0.0)
    score += np.where(lb.eq("BUY"), 0.3, 0.0)

    fh = df.get("finnhub_label").fillna("").astype(str).str.upper()
    score += np.where(fh.eq("STRONG_BUY"), 0.5, 0.0)
    score += np.where(fh.eq("BUY"), 0.2, 0.0)

    tv = df.get("tv_reco").fillna("").astype(str).str.upper()
    score += np.where(tv.eq("STRONG_BUY"), 0.3, 0.0)

    ab = df.get("analyst_bucket")
    score += np.where(ab.eq("Strong Buy"), 1.2, 0.0)
    score += np.where(ab.eq("Buy"), 0.6, 0.0)

    av = pd.to_numeric(df.get("analyst_votes"), errors="coerce").fillna(0.0)
    score += np.minimum(av, 20.0) * 0.04

    mc = pd.to_numeric(df.get("market_cap"), errors="coerce")
    mc_bonus = np.maximum(0.0, 4.8 - np.log10(np.where(mc>0, mc, np.nan)))
    mc_bonus = np.where(np.isfinite(mc_bonus), mc_bonus, 0.0)
    score += mc_bonus

    vol = pd.to_numeric(df.get("avg_vol"), errors="coerce").fillna(0.0)
    score += np.where(vol < MIN_AVG_DAILY_VOLUME, -0.5, 0.0)

    return pd.Series(score, index=df.index, dtype=float)

# ============= MAIN =============
def main():
    print("Chargement de l’univers (R1K + R2K + S&P500 + Nasdaq)…")
    tickers_df = load_universe()
    print(f"Tickers dans l'univers (brut): {len(tickers_df)}")

    # ---- catalogue secteurs (optionnel) ----
    sector_map = {}
    if os.path.exists(SECTOR_CATALOG_PATH):
        try:
            cat = pd.read_csv(SECTOR_CATALOG_PATH)
            cat["ticker"] = cat["ticker"].astype(str).str.upper().str.strip()
            sector_map = {t: (s if isinstance(s,str) else "", i if isinstance(i,str) else "")
                          for t,s,i in cat[["ticker","sector","industry"]].itertuples(index=False, name=None)}
            print(f"[INFO] Sector catalog chargé: {len(sector_map)} tickers")
        except Exception as e:
            print(f"[WARN] Sector catalog illisible: {e}")

    def sector_from_catalog(sym_tv: str, sym_yf: str):
        key_tv = (sym_tv or "").upper()
        if key_tv in sector_map: return sector_map[key_tv]
        key_yf_tv = (sym_yf or "").upper().replace("-", ".")
        return sector_map.get(key_yf_tv, ("", ""))

    # === PASS 1: Bulk history → TA local rapide sur tout l’univers ===
    print("Téléchargement bulk des historiques (avec fallback)…")
    yf_list = tickers_df["yf_symbol"].tolist()
    bulk = bulk_download_history(yf_list, period=PERIOD, interval=INTERVAL, chunk_size=200, fallback_limit=None)
    print(f"[INFO] Historique dispo après bulk+fallback: {len(bulk)}/{len(yf_list)}")

    seed_rows = []
    for rec in tickers_df.itertuples(index=False):
        tv_sym, yf_sym = rec.tv_symbol, rec.yf_symbol
        hist = bulk.get(yf_sym)
        if hist is None or hist.empty:
            continue
        bucket, votes, det = compute_local_technical_bucket(hist)
        if not bucket:
            continue
        local_label = local_label_from_indicators(det, bucket)
        seed_rows.append({
            "ticker_tv": tv_sym, "ticker_yf": yf_sym,
            "technical_local": bucket, "tech_score": votes,
            "price": det.get("price"),
            "local_label": local_label
        })

    seed = pd.DataFrame(seed_rows)
    print(f"[INFO] Seed TA local (>=60 barres): {len(seed)}/{len(tickers_df)}")
    if seed.empty:
        for name in [
            "debug_all_candidates.csv","confirmed_STRONGBUY.csv","anticipative_pre_signals.csv",
            "event_driven_signals.csv","candidates_all_ranked.csv","sector_breadth.csv",
            "universe_today.csv","raw_candidates.csv"
        ]:
            save_csv(pd.DataFrame(), name)
        return

    # === Pré-score 100% local (rapide) pour cibler les API externes ===
    _lb = seed["local_label"].fillna("").astype(str).str.upper()
    _lb_bonus = np.select(
        [_lb.eq("STRONG_BUY"), _lb.eq("BUY"), _lb.eq("NEUTRAL"), _lb.eq("SELL"), _lb.eq("STRONG_SELL")],
        [2.0,                  1.0,           0.0,              -0.8,           -1.5],
        default=0.0
    )
    _ts = pd.to_numeric(seed["tech_score"], errors="coerce").fillna(0.0).clip(-6, 6)
    seed["local_pre_score"] = _ts + _lb_bonus
    seed.sort_values(["local_pre_score"], ascending=[False], inplace=True, kind="mergesort")
    seed.reset_index(drop=True, inplace=True)

    if EXTERNAL_TARGET_MODE == "toppct":
        shortlist_size = max(int(len(seed) * float(EXTERNAL_TOP_PCT)), int(EXTERNAL_TOP_MIN))
    else:  # "topk"
        shortlist_size = max(int(EXTERNAL_TOP_MIN), int(EXTERNAL_TOP_K))
    shortlist_size = min(shortlist_size, len(seed))
    shortlist = seed.head(shortlist_size)["ticker_yf"].tolist()
    shortlist_set = set(shortlist)
    print(f"[TARGET] external shortlist={len(shortlist)} / seed={len(seed)} (mode={EXTERNAL_TARGET_MODE})")

    # === PASS 2: Finnhub (priorité), TV (bonus capé), Yahoo info ===

    # 2b) FINNHUB — enrichit un sous-ensemble prioritaire
    finnhub_map = {}
    fh_calls = 0
    if USE_FINNHUB and FINNHUB_API_KEY:
        for rec in seed.itertuples(index=False):
            yf_sym = rec.ticker_yf
            if yf_sym in shortlist_set and fh_calls < FINNHUB_MAX_CALLS:
                finnhub_map[yf_sym] = fetch_finnhub_label(yf_sym)
                fh_calls += 1
            else:
                finnhub_map[yf_sym] = {"finnhub_label": None, "finnhub_rsi": None, "finnhub_macd": None, "finnhub_macds": None}
    else:
        for rec in seed.itertuples(index=False):
            finnhub_map[rec.ticker_yf] = {"finnhub_label": None, "finnhub_rsi": None, "finnhub_macd": None, "finnhub_macds": None}

    # 2c) TV bonus capé
    tv_budget = TV_MAX_CALLS
    tv_ok = tv_ko = 0; consec_fail = 0; pause_spent = 0.0
    tv_results = {}
    tv_calls = 0

    for rec in seed.itertuples(index=False):
        tv_sym, yf_sym = rec.ticker_tv, rec.ticker_yf
        if (yf_sym in shortlist_set) and (tv_calls < tv_budget):
            if (tv_calls > 0) and (tv_calls % TV_COOLDOWN_EVERY == 0) and (pause_spent < TV_MAX_PAUSE_TOTAL):
                time.sleep(TV_COOLDOWN_SEC); pause_spent += TV_COOLDOWN_SEC
            exch_hint = ""
            try:
                tk=yf.Ticker(yf_sym); fi=getattr(tk,"fast_info",None)
                exch_hint=getattr(fi,"exchange",None) or ""
                if not exch_hint:
                    try:
                        info=tk.get_info() or {}
                        exch_hint=info.get("exchange") or info.get("fullExchangeName") or ""
                    except Exception: pass
            except Exception: pass
            tv = get_tv_summary_max(tv_sym, yf_sym, exch_hint)
            tv_calls += 1
            if tv.get("tv_reco"): tv_ok += 1; consec_fail = 0
            else:
                tv_ko += 1; consec_fail += 1
                if consec_fail >= TV_MAX_CONSEC_FAIL and pause_spent < TV_MAX_PAUSE_TOTAL:
                    time.sleep(TV_FAIL_PAUSE_SEC); pause_spent += TV_FAIL_PAUSE_SEC; consec_fail = 0
            tv_results[yf_sym] = tv
        else:
            tv_results[yf_sym] = {"tv_reco": None, "tv_symbol_used": None, "tv_exchange_used": None}
    print(f"[TV] shortlist_used={tv_calls}, ok={tv_ok}, ko={tv_ko}, pause_total≈{pause_spent:.1f}s")

    # 2d) Yahoo info + assemble rows
    rows=[]
    for i, rec in enumerate(seed.itertuples(index=False), 1):
        tv_sym, yf_sym = rec.ticker_tv, rec.ticker_yf
        tech_bucket, tech_votes, last_price = rec.technical_local, rec.tech_score, rec.price
        local_label = rec.local_label

        try:
            tk = yf.Ticker(yf_sym)
            fi = getattr(tk, "fast_info", None)
            mcap = getattr(fi, "market_cap", None)

            # Appel Yahoo get_info UNIQUEMENT si utile (shortlist ou cap manquante)
            info = {}
            need_info = (yf_sym in shortlist_set) or (mcap is None)
            if need_info:
                try:
                    info = tk.get_info() or {}
                    if mcap is None:
                        mcap = info.get("marketCap")
                except Exception:
                    info = {}

            # Volume moyen (10j/3m)
            avg_vol = None
            try:
                if fi:
                    avg_vol = getattr(fi, "ten_day_average_volume", None) or getattr(fi, "three_month_average_volume", None)
                if avg_vol is None and isinstance(info, dict):
                    avg_vol = info.get("averageDailyVolume10Day") or info.get("averageDailyVolume3Month")
            except Exception:
                pass

            country = (info.get("country") or info.get("countryOfCompany") or "").strip()
            exch = info.get("exchange") or info.get("fullExchangeName") or (getattr(fi, "exchange", "") or "")
            is_us_exchange = str(exch).upper() in {"NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"}

            # Filtres (pour la sélection finale)
            out_of_scope = False
            if country and (country.upper() not in {"USA","US","UNITED STATES","UNITED STATES OF AMERICA"} and not is_us_exchange):
                out_of_scope = True
            if isinstance(mcap,(int,float)) and mcap >= MAX_MARKET_CAP:
                out_of_scope = True

            tv = tv_results.get(yf_sym) or {"tv_reco": None, "tv_symbol_used": None, "tv_exchange_used": None}
            fh = finnhub_map.get(yf_sym) or {"finnhub_label": None}

            # final_signal: Finnhub > TV > Local
            final_signal = fh.get("finnhub_label") or tv.get("tv_reco") or local_label

            sec_cat, ind_cat = sector_from_catalog(tv_sym, yf_sym)
            sector_val   = sec_cat or (info.get("sector") or "").strip() or "Unknown"
            industry_val = ind_cat or (info.get("industry") or "").strip() or "Unknown"

            analyst_mean  = info.get("recommendationMean") if need_info else None
            analyst_votes = info.get("numberOfAnalystOpinions") if need_info else None
            analyst_bucket = analyst_bucket_from_mean(analyst_mean) if need_info else None

            rows.append({
                "ticker_tv": tv_sym, "ticker_yf": yf_sym,
                "exchange": exch, "market_cap": mcap, "price": float(last_price) if last_price is not None else None,
                "sector": sector_val, "industry": industry_val,
                "technical_local": tech_bucket, "tech_score": int(tech_votes) if tech_votes is not None else None,
                "local_label": local_label,
                "finnhub_label": fh.get("finnhub_label"),
                "tv_reco": tv.get("tv_reco"),
                "tv_symbol_used": tv.get("tv_symbol_used"),
                "tv_exchange_used": tv.get("tv_exchange_used"),
                "final_signal": final_signal,
                "analyst_bucket": analyst_bucket,
                "analyst_mean": analyst_mean, "analyst_votes": analyst_votes,
                "avg_vol": avg_vol,
                "out_of_scope": out_of_scope
            })
        except Exception:
            continue

        if i % 50 == 0:
            seen = int(seed.loc[:i-1, "ticker_yf"].isin(shortlist_set).sum())
            print(f"{i}/{len(seed)} enrichis… shortlist_seen≈{seen}, FH used≈{fh_calls}/{FINNHUB_MAX_CALLS}, TV used≈{tv_calls}/{TV_MAX_CALLS}")

    if _tv_fail_log:
        save_csv(pd.DataFrame(_tv_fail_log), "tv_failures.csv")

    df = pd.DataFrame(rows)
    save_csv(df, "raw_candidates.csv")

    if df.empty:
        for name in [
            "debug_all_candidates.csv","confirmed_STRONGBUY.csv","anticipative_pre_signals.csv",
            "event_driven_signals.csv","candidates_all_ranked.csv","sector_breadth.csv","universe_today.csv"
        ]:
            save_csv(pd.DataFrame(), name)
        return

    # === Univers du jour ===
    if not KEEP_FULL_UNIVERSE_IN_OUTPUTS:
        df = df[~df["out_of_scope"]].copy()
    save_csv(df.drop(columns=["out_of_scope"]), "universe_today.csv")

    # === Scoring vectorisé ===
    df["rank_score"] = compute_rank_score(df)

    # === Colonnes de base ===
    base_cols = [
        "ticker_tv","ticker_yf","price","market_cap","sector","industry",
        "technical_local","tech_score","local_label","finnhub_label",
        "tv_reco","tv_symbol_used","tv_exchange_used",
        "final_signal","analyst_bucket","analyst_mean","analyst_votes","avg_vol","rank_score"
    ]

    # ---- Confirmed (assoupli & plus réaliste) ----
    an_ok = df["analyst_bucket"].isin({"Strong Buy", "Buy"}) & (
        pd.to_numeric(df["analyst_votes"], errors="coerce").fillna(0) >= CONFIRM_MIN_ANALYST_VOTES
    )
    fh_ok = df["finnhub_label"].isin({"BUY", "STRONG_BUY"})
    tv_ok = df["tv_reco"].eq("STRONG_BUY")
    ta_ok = (pd.to_numeric(df["tech_score"], errors="coerce").fillna(-99) >= CONFIRM_MIN_TECH_VOTES)
    liq_ok = (pd.to_numeric(df["avg_vol"], errors="coerce").fillna(0) >= MIN_AVG_DAILY_VOLUME)

    # Compteur de corroborations (diagnostic)
    corr_count = an_ok.astype(int) + fh_ok.astype(int) + tv_ok.astype(int) + ta_ok.astype(int)

    # Porte A : final STRONG_BUY + ≥1 corroboration
    mask_A = df["final_signal"].fillna("").astype(str).str.upper().eq("STRONG_BUY") & (corr_count >= 1)

    # Porte B : TV = STRONG_BUY + (analyst Buy/Strong Buy OU TA forte)
    mask_B = tv_ok & (an_ok | ta_ok)

    # Porte C : Analyst = Strong Buy (≥15 votes) + (FH ok OU TA forte)
    an_strong_enough = df["analyst_bucket"].eq("Strong Buy") & (
        pd.to_numeric(df["analyst_votes"], errors="coerce").fillna(0) >= 15
    )
    mask_C = an_strong_enough & (fh_ok | ta_ok)

    # Assemblage final + filtres généraux
    mask_confirm = (mask_A | mask_B | mask_C) & liq_ok & (~df["out_of_scope"])
    confirmed = df[mask_confirm].copy().sort_values(["rank_score", "market_cap"], ascending=[False, True])
    save_csv(confirmed[base_cols], "confirmed_STRONGBUY.csv")

    # Log de contrôle
    print(
        f"[CONFIRMED] via A={int(mask_A.sum())}, "
        f"via B={int((mask_B & liq_ok & (~df['out_of_scope'])).sum())}, "
        f"via C={int((mask_C & liq_ok & (~df['out_of_scope'])).sum())}, "
        f"total={len(confirmed)}"
    )

    # Pré-signaux
    mask_pre = (
        df["technical_local"].isin({"Buy","Strong Buy"})
        | df["local_label"].isin({"BUY","STRONG_BUY"})
        | df["finnhub_label"].isin({"BUY","STRONG_BUY"})
        | df["tv_reco"].eq("STRONG_BUY")
    ) & (~df["out_of_scope"]) & liq_ok
    pre = df[mask_pre].copy().sort_values(["rank_score","market_cap"], ascending=[False, True])
    save_csv(pre[base_cols], "anticipative_pre_signals.csv")

    # Event-driven
    evt = df[df["analyst_bucket"].notna() & (~df["out_of_scope"]) & liq_ok] \
            .copy().sort_values(["rank_score","market_cap"], ascending=[False, True])
    save_csv(evt[base_cols], "event_driven_signals.csv")

    # Reste de l’univers
    used_idx = pd.Index([]).union(confirmed.index).union(pre.index).union(evt.index)
    universe_rest = df.drop(index=used_idx).copy().sort_values(["rank_score","market_cap"], ascending=[False, True])

    all_out = pd.concat([
        confirmed.assign(candidate_type="confirmed"),
        pre.assign(candidate_type="anticipative"),
        evt.assign(candidate_type="event"),
        universe_rest.assign(candidate_type="universe"),
    ], ignore_index=True)
    all_out.sort_values(["candidate_type","rank_score","market_cap"], ascending=[True, False, True], inplace=True)
    save_csv(all_out[["candidate_type"]+base_cols], "candidates_all_ranked.csv")

    # === Sector breadth (sur df complet du jour, hors out_of_scope & avec liquidité) ===
    df_sc = df[~df["out_of_scope"] & liq_ok].copy()

    def _sector_universe(df_all: pd.DataFrame) -> pd.Series:
        return df_all.groupby(df_all["sector"].fillna("Unknown"))["ticker_yf"].nunique()
    def _sector_counts(df_part: pd.DataFrame) -> pd.Series:
        return df_part.groupby(df_part["sector"].fillna("Unknown"))["ticker_yf"].nunique()

    univ = _sector_universe(df_sc)
    c_conf = _sector_counts(confirmed).reindex(univ.index, fill_value=0)
    c_pre  = _sector_counts(pre).reindex(univ.index, fill_value=0)
    c_evt  = _sector_counts(evt).reindex(univ.index, fill_value=0)

    pct_conf = (c_conf / univ.replace(0, np.nan) * 100).fillna(0)
    pct_pre  = (c_pre  / univ.replace(0, np.nan) * 100).fillna(0)
    pct_evt  = (c_evt  / univ.replace(0, np.nan) * 100).fillna(0)

    breadth = (2*pct_conf + 1*pct_pre - 1*pct_evt)
    z_breadth = (breadth - breadth.mean()) / (breadth.std(ddof=1) or 1)

    sector_breadth = pd.DataFrame({
        "sector": univ.index,
        "universe": univ.values,
        "confirmed": c_conf.values,
        "pre": c_pre.values,
        "events": c_evt.values,
        "pct_confirmed": pct_conf.values,
        "pct_pre": pct_pre.values,
        "pct_events": pct_evt.values,
        "breadth": breadth.values,
        "z_breadth": z_breadth.values,
        "wow_pct_confirmed": 0.0,
    }).sort_values(["z_breadth","breadth"], ascending=[False, False])

    save_csv(sector_breadth, "sector_breadth.csv")

    # === Sector history (weekly snapshot) ===
    HISTORY_CSV = "sector_history.csv"
    def _norm_sector(x): return (x or "").strip() or "Unknown"
    def _count_by_sector(rows_df):
        out={}
        for _, r in rows_df.iterrows():
            sec=_norm_sector(r.get("sector")); out[sec]=out.get(sec,0)+1
        return out

    now_dt = datetime.now(timezone.utc); iso_year, iso_week, _ = now_dt.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"

    counts_confirmed = _count_by_sector(confirmed)
    counts_pre       = _count_by_sector(pre)
    counts_events    = _count_by_sector(evt)

    lines = []
    for sec, n in counts_confirmed.items(): lines.append((week_key, "confirmed", sec, int(n)))
    for sec, n in counts_pre.items():       lines.append((week_key, "pre", sec, int(n)))
    for sec, n in counts_events.items():    lines.append((week_key, "events", sec, int(n)))

    if os.path.exists(HISTORY_CSV):
        try: existing = pd.read_csv(HISTORY_CSV)
        except Exception: existing = pd.DataFrame(columns=["date","bucket","sector","count"])
    else:
        existing = pd.DataFrame(columns=["date","bucket","sector","count"])

    existing = existing[existing["date"] != week_key]
    new_df = pd.concat([existing, pd.DataFrame(lines, columns=["date","bucket","sector","count"])], ignore_index=True)
    new_df["__key"] = new_df["date"].astype(str)+"|"+new_df["bucket"].astype(str)+"|"+new_df["sector"].astype(str)
    new_df = new_df.drop_duplicates(subset="__key").drop(columns="__key")
    save_csv(new_df, HISTORY_CSV)

    # === Journalisation des signaux (historique par ticker) ===
    SIGNALS_CSV = "signals_history.csv"
    def _append_signals(bucket_name: str, df_bucket: pd.DataFrame):
        if df_bucket is None or df_bucket.empty:
            return pd.DataFrame(columns=["date","bucket","ticker_yf","ticker_tv","sector","industry","price"])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = df_bucket[["ticker_yf","ticker_tv","sector","industry","price"]].copy()
        out.insert(0, "bucket", bucket_name)
        out.insert(0, "date", now)
        return out

    hist_blocks = [
        _append_signals("confirmed", confirmed),
        _append_signals("pre",       pre),
        _append_signals("events",    evt)
    ]
    hist_all = pd.concat(hist_blocks, ignore_index=True)

    if os.path.exists(SIGNALS_CSV):
        try:
            prev = pd.read_csv(SIGNALS_CSV)
            prev["__k"] = prev["date"].astype(str)+"|"+prev["bucket"].astype(str)+"|"+prev["ticker_yf"].astype(str)
        except Exception:
            prev = pd.DataFrame(columns=["date","bucket","ticker_yf","ticker_tv","sector","industry","price"])
            prev["__k"] = ""
    else:
        prev = pd.DataFrame(columns=["date","bucket","ticker_yf","ticker_tv","sector","industry","price","__k"])

    hist_all["__k"] = hist_all["date"].astype(str)+"|"+hist_all["bucket"].astype(str)+"|"+hist_all["ticker_yf"].astype(str)
    merged = pd.concat([prev, hist_all], ignore_index=True)
    merged = merged.drop_duplicates(subset="__k").drop(columns="__k")
    save_csv(merged, SIGNALS_CSV)

    # Stats de remplissage (diagnostic)
    fill_tv = df["tv_reco"].notna().mean() if not df.empty else 0.0
    fill_fh = df["finnhub_label"].notna().mean() if not df.empty else 0.0
    fill_an = df["analyst_bucket"].notna().mean() if not df.empty else 0.0
    print(f"[OK] universe={len(df)} — confirmed={len(confirmed)}, pre={len(pre)}, event={len(evt)}, universe_rest={len(universe_rest)} — FH fill={fill_fh:.0%}, TV fill={fill_tv:.0%}, Analysts fill={fill_an:.0%}")

if __name__ == "__main__":
    main()
