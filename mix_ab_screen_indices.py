# mix_ab_screen_indices.py
# Screener US (Russell 1000 + 2000) — Gratuit, local + Finnhub + TV bonus
# Sorties :
#   - confirmed_STRONGBUY.csv
#   - anticipative_pre_signals.csv
#   - event_driven_signals.csv
#   - candidates_all_ranked.csv   (inclut 'universe' = tout le seed filtré US/MCAP)
#   - universe_today.csv          (univers complet du jour)
#   - debug_all_candidates.csv
#   - sector_breadth.csv
#   - sector_history.csv
#   - signals_history.csv
#   - raw_candidates.csv
#   - tv_failures.csv (diagnostic TradingView)

import warnings, time, os, math, random, json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tradingview_ta import TA_Handler, Interval
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ============= CONFIG =============
INCLUDE_RUSSELL_1000 = True
INCLUDE_RUSSELL_2000 = True
R1K_FALLBACK_CSV = "russell1000.csv"   # 1 col: Ticker
R2K_FALLBACK_CSV = "russell2000.csv"   # 1 col: Ticker

# Fenêtre (suffisant pour SMA200/TA classiques)
PERIOD   = "1y"
INTERVAL = "1d"
TV_INTERVAL = Interval.INTERVAL_1_DAY

# Limites & priorisation
GATE_MIN_TECH_VOTES = 2         # sert à prioriser les appels externes
MAX_MARKET_CAP      = 200_000_000_000     # < 200B
KEEP_FULL_UNIVERSE_IN_OUTPUTS = True      # garde tout le seed dans les outputs (même si externes manquants)

# TradingView : bonus capé (ratelimit fréquent)
TV_MAX_CALLS = 100
DELAY_BETWEEN_TV_CALLS_SEC = 0.10
TV_COOLDOWN_EVERY = 25
TV_COOLDOWN_SEC   = 2.0
TV_MAX_CONSEC_FAIL = 10
TV_FAIL_PAUSE_SEC  = 6.0
TV_MAX_PAUSE_TOTAL = 45.0

# Finnhub (gratuit) — on limite le volume pour rester <30mn
USE_FINNHUB = True
FINNHUB_API_KEY = (os.environ.get("FINNHUB_API_KEY") or "d2sfah1r01qiq7a429ugd2sfah1r01qiq7a429v0").strip()
FINNHUB_RESOLUTION = "D"          # Daily
FINNHUB_MAX_CALLS  = 250          # budget global (60 req/min free) → suffisant pour top tickers
FINNHUB_SLEEP_BETWEEN = 0.10      # petite pause anti-ban
FINNHUB_TIMEOUT_SEC = 20

# Alpha Vantage (optionnel, backup de Finnhub si besoin)
USE_ALPHA_VANTAGE = True
ALPHAVANTAGE_API_KEY = (os.environ.get("ALPHAVANTAGE_API_KEY") or "85SZZGRDDJ6MUAEX").strip()
AV_MAX_CALLS = 20
AV_SLEEP_BETWEEN = 12.5           # free tier 5 calls/min → 12s mini

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

def load_universe()->pd.DataFrame:
    ticks=[]
    if INCLUDE_RUSSELL_1000:
        try:
            ticks += fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_1000_Index")
        except:
            if os.path.exists(R1K_FALLBACK_CSV):
                ticks += pd.read_csv(R1K_FALLBACK_CSV)["Ticker"].astype(str).tolist()
    if INCLUDE_RUSSELL_2000:
        try:
            ticks += fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_2000_Index")
        except:
            if os.path.exists(R2K_FALLBACK_CSV):
                ticks += pd.read_csv(R2K_FALLBACK_CSV)["Ticker"].astype(str).tolist()

    if not ticks: raise RuntimeError("Impossible de charger l’univers (Wikipédia + CSV fallback).")

    tv_syms = [tv_norm(s) for s in ticks]
    yf_syms = [yf_norm(s) for s in ticks]
    df = pd.DataFrame({"tv_symbol": tv_syms, "yf_symbol": yf_syms}).drop_duplicates().reset_index(drop=True)
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
    """Composite 'TV-like' local à partir des familles MAs / Oscillateurs."""
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

        # Oscillateurs
        if rsi is not None:
            score += 1 if rsi >= 55 else (-1 if rsi <= 45 else 0)
        if macd is not None and macds is not None and not np.isnan(macd) and not np.isnan(macds):
            score += 1 if macd > macds else -1
        if k is not None and d is not None and not np.isnan(k) and not np.isnan(d):
            score += 1 if k > d else -1
    except Exception:
        pass

    if score >= 4:   return "STRONG_BUY"
    if score >= 2:   return "BUY"
    if score <= -4:  return "STRONG_SELL"
    if score <= -2:  return "SELL"
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
    rsi_params = {
        "symbol": symbol,
        "resolution": FINNHUB_RESOLUTION,
        "indicator": "rsi",
        "timeperiod": 14,
        "token": FINNHUB_API_KEY
    }
    rsi_json = _fh_get_json(base, rsi_params)
    rsi_val = None
    if rsi_json and isinstance(rsi_json.get("rsi"), list) and rsi_json.get("rsi"):
        rsi_val = rsi_json["rsi"][-1]

    # MACD
    macd_params = {
        "symbol": symbol,
        "resolution": FINNHUB_RESOLUTION,
        "indicator": "macd",
        "fastperiod": 12,
        "slowperiod": 26,
        "signalperiod": 9,
        "token": FINNHUB_API_KEY
    }
    macd_json = _fh_get_json(base, macd_params)
    macd_val = macds_val = None
    if macd_json:
        # Finnhub renvoie souvent: "macd", "macd_signal", "macd_hist"
        for k in ["macd","MACD","macdValue"]:
            if isinstance(macd_json.get(k), list) and macd_json[k]:
                macd_val = macd_json[k][-1]; break
        for k in ["macd_signal","signal","MACDs","macdSignal"]:
            if isinstance(macd_json.get(k), list) and macd_json[k]:
                macds_val = macd_json[k][-1]; break

    label = _score_from_rsi_macd(rsi_val, macd_val, macds_val)
    time.sleep(FINNHUB_SLEEP_BETWEEN)
    return {"finnhub_label": label, "finnhub_rsi": rsi_val, "finnhub_macd": macd_val, "finnhub_macds": macds_val}

# ========= ALPHA VANTAGE (backup optionnel) =========
def fetch_alpha_label(symbol: str):
    if (not USE_ALPHA_VANTAGE) or (not ALPHAVANTAGE_API_KEY):
        return {"av_label": None}
    try:
        # RSI
        rsi_url = "https://www.alphavantage.co/query"
        rsi_params = {"function":"RSI","symbol":symbol,"interval":"daily","time_period":"14","series_type":"close","apikey":ALPHAVANTAGE_API_KEY}
        r = requests.get(rsi_url, params=rsi_params, timeout=30); r.raise_for_status()
        j = r.json()
        rsi_val = None
        ta = j.get("Technical Analysis: RSI", {})
        if ta:
            last_key = sorted(ta.keys())[-1]
            rsi_val = float(ta[last_key]["RSI"])

        # MACD
        macd_params = {"function":"MACD","symbol":symbol,"interval":"daily","series_type":"close","fastperiod":"12","slowperiod":"26","signalperiod":"9","apikey":ALPHAVANTAGE_API_KEY}
        r = requests.get(rsi_url, params=macd_params, timeout=30); r.raise_for_status()
        j2 = r.json()
        macd_val = macds_val = None
        tb = j2.get("Technical Analysis: MACD", {})
        if tb:
            last_key = sorted(tb.keys())[-1]
            macd_val = float(tb[last_key]["MACD"])
            macds_val = float(tb[last_key]["MACD_Signal"])

        label = _score_from_rsi_macd(rsi_val, macd_val, macds_val)
        time.sleep(AV_SLEEP_BETWEEN)
        return {"av_label": label}
    except Exception:
        return {"av_label": None}

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

# ============= MAIN =============
def main():
    print("Chargement des tickers Russell 1000 + 2000…")
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

    # === PASS 2: Finnhub (priorité), TV (bonus capé), Yahoo info ===
    # 2a) Prioriser par score technique pour les appels externes
    seed = seed.sort_values(["tech_score"], ascending=False).reset_index(drop=True)

    # 2b) FINNHUB — on enrichit un sous-ensemble prioritaire
    fh_budget = min(FINNHUB_MAX_CALLS, len(seed)) if USE_FINNHUB and FINNHUB_API_KEY else 0
    finnhub_map = {}
    for idx, rec in enumerate(seed.itertuples(index=False), 1):
        yf_sym = rec.ticker_yf
        if idx <= fh_budget:
            finnhub_map[yf_sym] = fetch_finnhub_label(yf_sym)
        else:
            finnhub_map[yf_sym] = {"finnhub_label": None, "finnhub_rsi": None, "finnhub_macd": None, "finnhub_macds": None}

    # 2c) TV bonus capé
    tv_budget = min(TV_MAX_CALLS, len(seed))
    tv_ok = tv_ko = 0; consec_fail = 0; pause_spent = 0.0
    tv_results = {}

    for idx, rec in enumerate(seed.itertuples(index=False), 1):
        tv_sym, yf_sym = rec.ticker_tv, rec.ticker_yf
        if idx <= tv_budget:
            if idx % TV_COOLDOWN_EVERY == 0 and pause_spent < TV_MAX_PAUSE_TOTAL:
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
            if tv.get("tv_reco"): tv_ok += 1; consec_fail = 0
            else:
                tv_ko += 1; consec_fail += 1
                if consec_fail >= TV_MAX_CONSEC_FAIL and pause_spent < TV_MAX_PAUSE_TOTAL:
                    time.sleep(TV_FAIL_PAUSE_SEC); pause_spent += TV_FAIL_PAUSE_SEC; consec_fail = 0
            tv_results[yf_sym] = tv
        else:
            tv_results[yf_sym] = {"tv_reco": None, "tv_symbol_used": None, "tv_exchange_used": None}
    print(f"[TV] budget={tv_budget}, ok={tv_ok}, ko={tv_ko}, pause_total≈{pause_spent:.1f}s")

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
            info = {}
            if mcap is None or True:  # on lit souvent get_info pour analysts/secteur
                try:
                    info = tk.get_info() or {}
                    if mcap is None:
                        mcap = info.get("marketCap")
                except Exception:
                    info = {}

            country = (info.get("country") or info.get("countryOfCompany") or "").strip()
            exch = info.get("exchange") or info.get("fullExchangeName") or ""
            is_us_exchange = str(exch).upper() in {"NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"}

            # Filtres (pour la sélection finale) — l'univers complet sera conservé si flag actif
            out_of_scope = False
            if country and (country.upper() not in {"USA","US","UNITED STATES","UNITED STATES OF AMERICA"} and not is_us_exchange):
                out_of_scope = True
            if isinstance(mcap,(int,float)) and mcap >= MAX_MARKET_CAP:
                out_of_scope = True

            tv = tv_results.get(yf_sym) or {"tv_reco": None, "tv_symbol_used": None, "tv_exchange_used": None}
            fh = finnhub_map.get(yf_sym) or {"finnhub_label": None}

            # final_signal: Local > Finnhub > TV
            final_signal = local_label or fh.get("finnhub_label") or tv.get("tv_reco")

            sec_cat = ind_cat = ""
            if SECTOR_CATALOG_PATH and os.path.exists(SECTOR_CATALOG_PATH):
                # sector_from_catalog déjà déf
                pass
            sec_cat, ind_cat = sector_from_catalog(tv_sym, yf_sym)
            sector_val   = sec_cat or (info.get("sector") or "").strip() or "Unknown"
            industry_val = ind_cat or (info.get("industry") or "").strip() or "Unknown"

            analyst_mean  = info.get("recommendationMean")
            analyst_votes = info.get("numberOfAnalystOpinions")
            analyst_bucket = analyst_bucket_from_mean(analyst_mean)

            row = {
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
                "out_of_scope": out_of_scope
            }
            rows.append(row)
        except Exception:
            continue

        if i % 50 == 0:
            print(f"{i}/{len(seed)} enrichis… (FH used {min(i,fh_budget)}/{fh_budget}, TV used {min(i,tv_budget)}/{tv_budget})")

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
    # Si on veut conserver tout le seed dans les outputs (même hors US/MCAP), on “ramène” les lignes out_of_scope
    if KEEP_FULL_UNIVERSE_IN_OUTPUTS:
        # rien de spécial: df contient déjà tout ce qui a passé le TA seed; on garde même out_of_scope
        pass
    else:
        df = df[~df["out_of_scope"]].copy()

    # Sauvegarde univers_today
    save_csv(df.drop(columns=["out_of_scope"]), "universe_today.csv")

    # === Scoring & sorties ===
    dbg = df.copy()

    def rank_score_row(s: pd.Series) -> float:
        score = 0.0
        # Final signal (Local > Finnhub > TV)
        fs = (s.get("final_signal") or "").upper()
        if fs == "STRONG_BUY": score += 3.0
        elif fs == "BUY":      score += 1.8
        elif fs == "NEUTRAL":  score += 0.0
        elif fs == "SELL":     score -= 1.0
        elif fs == "STRONG_SELL": score -= 2.0

        # Renfort: local/finnhub/tv pris séparément (petits bonus)
        lb = (s.get("local_label") or "").upper()
        if lb == "STRONG_BUY": score += 0.8
        elif lb == "BUY":      score += 0.3

        fh = (s.get("finnhub_label") or "").upper()
        if fh == "STRONG_BUY": score += 0.5
        elif fh == "BUY":      score += 0.2

        tv = (s.get("tv_reco") or "").upper()
        if tv == "STRONG_BUY": score += 0.3

        # Analysts
        ab = s.get("analyst_bucket")
        if ab == "Strong Buy": score += 1.2
        elif ab == "Buy":      score += 0.6
        av = s.get("analyst_votes")
        if isinstance(av,(int,float)) and av>0: score += min(av, 20)*0.04

        # Market cap preference: léger bonus small/mid
        mc = s.get("market_cap")
        if isinstance(mc,(int,float)) and mc>0:
            score += max(0.0, 4.8 - math.log10(mc))

        return float(score)

    dbg["rank_score"] = dbg.apply(rank_score_row, axis=1)
    dbg.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(dbg.drop(columns=["out_of_scope"], errors="ignore"), "debug_all_candidates.csv")

    base_cols = [
        "ticker_tv","ticker_yf","price","market_cap","sector","industry",
        "technical_local","tech_score","local_label","finnhub_label",
        "tv_reco","tv_symbol_used","tv_exchange_used",
        "final_signal","analyst_bucket","analyst_mean","analyst_votes","rank_score"
    ]

    # Buckets (on peut garder les filtres US/MCAP implicitement via le score/final_signal)
    mask_confirm = (
        df["final_signal"].eq("STRONG_BUY")
    ) & (~df["out_of_scope"])
    confirmed = df[mask_confirm].copy()
    confirmed["rank_score"] = confirmed.apply(rank_score_row, axis=1)
    confirmed.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(confirmed[base_cols], "confirmed_STRONGBUY.csv")

    mask_pre = (
        df["technical_local"].isin({"Buy","Strong Buy"}) |
        df["local_label"].isin({"BUY","STRONG_BUY"}) |
        df["finnhub_label"].isin({"BUY","STRONG_BUY"}) |
        df["tv_reco"].eq("STRONG_BUY")
    ) & (~df["out_of_scope"])
    pre = df[mask_pre].copy()
    pre["rank_score"] = pre.apply(rank_score_row, axis=1)
    pre.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(pre[base_cols], "anticipative_pre_signals.csv")

    evt = df[df["analyst_bucket"].notna() & (~df["out_of_scope"])].copy()
    evt["rank_score"] = evt.apply(rank_score_row, axis=1)
    evt.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(evt[base_cols], "event_driven_signals.csv")

    # Reste de l’univers
    others_idx = df.index.difference(pd.concat([confirmed, pre, evt]).index)
    universe_rest = df.loc[others_idx].copy()
    universe_rest["rank_score"] = universe_rest.apply(rank_score_row, axis=1)
    universe_rest.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)

    all_out = pd.concat([
        confirmed.assign(candidate_type="confirmed"),
        pre.assign(candidate_type="anticipative"),
        evt.assign(candidate_type="event"),
        universe_rest.assign(candidate_type="universe"),
    ], ignore_index=True)
    all_out.sort_values(["candidate_type","rank_score","market_cap"], ascending=[True, False, True], inplace=True)
    save_csv(all_out[["candidate_type"]+base_cols], "candidates_all_ranked.csv")

    # === Sector breadth (sur df complet du jour, hors out_of_scope) ===
    df_sc = df[~df["out_of_scope"]].copy()
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
        "wow_pct_confirmed": 0.0,  # calcul WoW optionnel
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

    # Stats de remplissage
    fill_tv = df["tv_reco"].notna().mean() if not df.empty else 0.0
    fill_fh = df["finnhub_label"].notna().mean() if not df.empty else 0.0
    fill_an = df["analyst_bucket"].notna().mean() if not df.empty else 0.0
    print(f"[OK] universe={len(df)} — confirmed={len(confirmed)}, pre={len(pre)}, event={len(evt)}, universe_rest={len(universe_rest)} — FH fill={fill_fh:.0%}, TV fill={fill_tv:.0%}, Analysts fill={fill_an:.0%}")

if __name__ == "__main__":
    main()
