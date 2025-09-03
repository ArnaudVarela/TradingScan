# mix_ab_screen_indices.py
# Screener US (Russell 1000 + 2000)
# Sorties :
#   - confirmed_STRONGBUY.csv
#   - anticipative_pre_signals.csv
#   - event_driven_signals.csv
#   - candidates_all_ranked.csv
#   - debug_all_candidates.csv
#   - sector_history.csv
#   - raw_candidates.csv (diag brut)
#   - sector_breadth.csv
#
# Patch : + Investing (résumé technique D/W/M) + BarChart (opinion/score)
# - Ajout de features: inv_{d,w,m}_vote, inv_{d,w,m}_label, bc_score, bc_label
# - Rank score prend en compte ces features (pondération douce)
# - Scrapers robustes (session, headers, retries, cache TTL) + gating

import warnings, time, os, math, re, json, random
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tradingview_ta import TA_Handler, Interval

warnings.filterwarnings("ignore")

# ============= CONFIG =============
INCLUDE_RUSSELL_1000 = True
INCLUDE_RUSSELL_2000 = True
R1K_FALLBACK_CSV = "russell1000.csv"   # 1 col: Ticker
R2K_FALLBACK_CSV = "russell2000.csv"   # 1 col: Ticker

PERIOD   = "2y"
INTERVAL = "1d"
TV_INTERVAL = Interval.INTERVAL_1_DAY

MAX_MARKET_CAP = 200_000_000_000     # < 200B
DELAY_BETWEEN_TV_CALLS_SEC = 0.2

SECTOR_CATALOG_PATH = "sector_catalog.csv"  # optionnel

# --- Nouvelles sources (feature flags) ---
USE_INVESTING = True
USE_BARCHART  = True

# --- Gating : n’appeler Invest/BarChart que si prometteur ---
GATE_MIN_TECH_VOTES = 2  # si TA local votes >= 2
GATE_TV_ONLY_STRONG = True  # appelle BarChart/Investing si tv_reco STRONG_BUY

# --- Cache simple (TTL secondes) pour endpoints externes ---
CACHE_TTL_SECONDS = 60 * 60  # 1h

# --- Sorties : racine + copie dans dashboard/public/ pour le front
PUBLIC_DIR = os.path.join("dashboard", "public")

# ============= UTILS I/O =============
def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, fname: str, also_public: bool = True):
    """Écrit fname à la racine + copie dans dashboard/public/fname."""
    _ensure_dir(fname)
    df.to_csv(fname, index=False)
    if also_public:
        dst = os.path.join(PUBLIC_DIR, fname)
        _ensure_dir(dst)
        df.to_csv(dst, index=False)

# ============= HELPERS =============
def yf_norm(s:str)->str: return s.replace(".", "-")
def tv_norm(s:str)->str: return s.replace("-", ".")
def base_sym(s:str)->str:
    """Symbole 'propre' pour requêtes externes (AAPL, BRK.B, etc.)."""
    s = (s or "").upper().strip()
    s = s.replace("-", ".")
    # Certains sites n’aiment pas les suffixes .US/.NYQ, on garde tel quel pour US
    return s

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
    return pd.DataFrame({"tv_symbol": tv_syms, "yf_symbol": yf_syms}).drop_duplicates().reset_index(drop=True)

def map_exchange_for_tv(exch: str)->str:
    e = (exch or "").upper()
    if "NASDAQ" in e: return "NASDAQ"
    if "NYSE"   in e: return "NYSE"
    if "ARCA"   in e: return "AMEX"
    return "NASDAQ"

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

def analyst_bucket_from_mean(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    if x < 1.6: return "Strong Buy"
    if x < 2.5: return "Buy"
    if x < 3.5: return "Hold"
    if x < 4.2: return "Sell"
    return "Strong Sell"

def get_tv_summary(symbol: str, exchange: str):
    def try_once(sym, ex):
        try:
            h = TA_Handler(symbol=sym, screener="america", exchange=ex, interval=TV_INTERVAL)
            s = h.get_analysis().summary
            return {"tv_reco": s.get("RECOMMENDATION")}
        except Exception:
            return None
    first = try_once(symbol, exchange)
    if first and first.get("tv_reco"): return first
    for ex in ("NASDAQ", "NYSE", "AMEX"):
        alt = try_once(symbol, ex)
        if alt and alt.get("tv_reco"): return alt
    return {"tv_reco": None}

# ============= NEW: SCRAPERS & CACHE =============

class _TTLCache:
    def __init__(self, ttl_sec=CACHE_TTL_SECONDS):
        self.ttl = ttl_sec
        self.store = {}  # key -> (ts, value)
    def get(self, key):
        v = self.store.get(key)
        if not v: return None
        ts, val = v
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return val
    def set(self, key, val):
        self.store[key] = (time.time(), val)

CACHE = _TTLCache()

def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      f"(KHTML, like Gecko) Chrome/{random.randint(109, 121)}.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
    })
    return s

def _retry_get_json(sess, url, params=None, headers=None, referer=None, max_try=3, timeout=20):
    if referer:
        sess.headers.update({"Referer": referer})
    for i in range(max_try):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 429):
                time.sleep(1.5 + i)
            else:
                time.sleep(0.3 + 0.2*i)
        except Exception:
            time.sleep(0.5 + 0.2*i)
    return None

def _retry_get_text(sess, url, params=None, headers=None, referer=None, max_try=3, timeout=20):
    if referer:
        sess.headers.update({"Referer": referer})
    for i in range(max_try):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.text
            if r.status_code in (403, 429):
                time.sleep(1.5 + i)
            else:
                time.sleep(0.3 + 0.2*i)
        except Exception:
            time.sleep(0.5 + 0.2*i)
    return None

# ---- BarChart opinion ---------------------------------------------------------
def fetch_barchart_opinion(symbol:str):
    """
    Renvoie dict: {"bc_score": float|None, "bc_label": str|None}
    Essaie d'abord l'API proxifiée (JSON), puis fallback parse HTML.
    """
    if not USE_BARCHART:
        return {"bc_score": None, "bc_label": None}

    sym = base_sym(symbol)
    ck = f"barchart:{sym}"
    cached = CACHE.get(ck)
    if cached is not None:
        return cached

    sess = _session()
    # Priming cookies
    referer = f"https://www.barchart.com/stocks/quotes/{sym}/opinion"
    _ = _retry_get_text(sess, referer, timeout=12)

    # Try JSON proxy endpoint (commune à plusieurs pages)
    api_url = "https://www.barchart.com/proxies/core-api/v1/opinions/get"
    params = {"symbols": sym, "orderBy": "symbol"}
    headers = {"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"}

    data = _retry_get_json(sess, api_url, params=params, headers=headers, referer=referer, timeout=15)
    score = None; label = None

    def _extract_score_label(obj):
        sc = None; lb = None
        if not obj: return sc, lb
        # Cherche des champs communs
        for k in ("score","compositeScore","overallScore","opinionScore"):
            if isinstance(obj.get(k), (int,float)):
                sc = float(obj[k]); break
        for k in ("rating","signal","recommendation","opinion","label","composite"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                lb = v.strip(); break
        return sc, lb

    if data and isinstance(data, dict):
        # Structure typique: {"data":[{...}]}
        arr = data.get("data") or data.get("results") or []
        if arr and isinstance(arr, list):
            sc, lb = _extract_score_label(arr[0])
            score = sc; label = lb

    # Fallback: parse HTML pour récupérer label dans __BC_DATA__ (si dispo)
    if score is None or label is None:
        html = _retry_get_text(sess, referer, timeout=15)
        if html:
            # Cherche JSON dans window.__BC_DATA__ = {...};
            m = re.search(r"__BC_DATA__\s*=\s*({.*?})\s*;", html, re.DOTALL)
            if m:
                try:
                    j = json.loads(m.group(1))
                    # Heuristique: plonger récursivement pour trouver score/label
                    stack = [j]
                    seen = 0
                    while stack and seen < 2000:
                        seen += 1
                        cur = stack.pop()
                        if isinstance(cur, dict):
                            sc, lb = _extract_score_label(cur)
                            if score is None and sc is not None: score = sc
                            if (label is None or label.lower()=="none") and lb: label = lb
                            for v in cur.values():
                                if isinstance(v, (dict, list)): stack.append(v)
                        elif isinstance(cur, list):
                            for v in cur:
                                if isinstance(v, (dict, list)): stack.append(v)
                except Exception:
                    pass

    out = {"bc_score": score, "bc_label": label}
    CACHE.set(ck, out)
    # petite pause anti-throttle
    time.sleep(0.15 + random.random()*0.15)
    return out

# ---- Investing technical summary (D/W/M) --------------------------------------
def _map_summary_to_vote(txt:str):
    """Map 'Strong Buy/Buy/Neutral/Sell/Strong Sell' -> {+2,+1,0,-1,-2}"""
    if not txt: return None
    t = txt.strip().lower()
    if "strong buy" in t:  return +2
    if "buy" in t:         return +1
    if "strong sell" in t: return -2
    if "sell" in t:        return -1
    if "neutral" in t:     return 0
    return None

def _first_group(regex, text):
    m = re.search(regex, text, re.IGNORECASE|re.DOTALL)
    return m.group(1).strip() if m else None

def fetch_investing_summary(symbol:str):
    """
    Retourne: {
      "inv_d_label": str|None, "inv_w_label": str|None, "inv_m_label": str|None,
      "inv_d_vote": int|None,  "inv_w_vote": int|None,  "inv_m_vote": int|None
    }
    Stratégie:
      1) search -> récupérer slug/link (équities US) -> pages ‘-technical’ D/W/M
      2) parser label dans HTML (heuristique robuste)
    """
    if not USE_INVESTING:
        return {"inv_d_label":None,"inv_w_label":None,"inv_m_label":None,
                "inv_d_vote":None,"inv_w_vote":None,"inv_m_vote":None}

    sym = base_sym(symbol)
    ck = f"investing:{sym}"
    cached = CACHE.get(ck)
    if cached is not None:
        return cached

    sess = _session()
    # Petite chauffe pour cookie/csrf
    _ = _retry_get_text(sess, "https://www.investing.com/", timeout=12)

    # 1) Search endpoints (nouveau et ancien)
    slug_link = None
    # Essai API moderne
    j = _retry_get_json(sess, "https://api.investing.com/api/search/v2/global", params={"q": sym}, timeout=15)
    candidates = []
    if j and isinstance(j, dict):
        for k in ("quotes","all","data","result","results"):
            arr = j.get(k)
            if isinstance(arr, list):
                candidates += arr
    # Essai ancien endpoint
    if not candidates:
        try:
            r = sess.post("https://www.investing.com/search/service/SearchInner",
                          data={"search_text": sym, "limit": 20, "action":"getSymbols"},
                          headers={"X-Requested-With":"XMLHttpRequest"}, timeout=15)
            if r.status_code==200:
                jj = r.json()
                if isinstance(jj, dict):
                    for k in ("All","all","symbols","quotes"):
                        arr = jj.get(k)
                        if isinstance(arr, list):
                            candidates += arr
        except Exception:
            pass

    # Sélectionne un equity US si possible
    best = None
    for c in candidates:
        # diverses formes possibles
        t = (c.get("type") or c.get("pair_type") or c.get("instrument_type") or "").lower()
        sect = (c.get("sector") or c.get("sector_type") or "").lower()
        ex = (c.get("exchange") or c.get("exchange_name") or "").upper()
        is_equity = ("equities" in t) or (t=="equities") or ("stock" in t) or ("equity" in t)
        if is_equity and (("NASDAQ" in ex) or ("NYSE" in ex) or ("AMEX" in ex) or ("US" in ex)):
            best = c; break
    if not best and candidates:
        best = candidates[0]

    if best:
        link = best.get("link") or best.get("url") or best.get("symbol_url")
        if link and isinstance(link, str):
            # typiquement: /equities/apple-computer-inc
            slug_link = "https://www.investing.com" + link

    def _parse_summary_from_html(html_text):
        # essaie de trouver "Technical Summary: Strong Buy" ou équivalent dans la page
        # 1) JSON Next.js __NEXT_DATA__ (si présent)
        lab = _first_group(r'"technicalSummary"\s*:\s*"(Strong Buy|Buy|Neutral|Sell|Strong Sell)"', html_text)
        if lab: return lab
        # 2) Autres variantes de labels sérialisés
        lab = _first_group(r'"summary"\s*:\s*"(Strong Buy|Buy|Neutral|Sell|Strong Sell)"', html_text)
        if lab: return lab
        # 3) Recherche textuelle près de "Technical Summary"
        #    on prend le 1er label connu apparaissant dans la page (heuristique)
        m = re.search(r"(Strong Buy|Buy|Neutral|Sell|Strong Sell)", html_text, re.IGNORECASE)
        if m: return m.group(1).title()
        return None

    # 2) Récupération D/W/M
    d_lab = w_lab = m_lab = None
    if slug_link:
        # Plusieurs paramètres possibles selon versions du site
        urls = [
            (slug_link + "-technical?period=daily",   "daily"),
            (slug_link + "-technical?period=weekly",  "weekly"),
            (slug_link + "-technical?period=monthly", "monthly"),
            (slug_link + "-technical?timeFrame=Daily","daily"),
            (slug_link + "-technical?timeFrame=Weekly","weekly"),
            (slug_link + "-technical?timeFrame=Monthly","monthly"),
        ]
        got_d = got_w = got_m = False
        for u, kind in urls:
            if (got_d and kind=="daily") or (got_w and kind=="weekly") or (got_m and kind=="monthly"):
                continue
            html = _retry_get_text(sess, u, referer="https://www.investing.com/", timeout=15)
            if not html: continue
            lab = _parse_summary_from_html(html)
            if kind=="daily"   and lab: d_lab, got_d = lab, True
            if kind=="weekly"  and lab: w_lab, got_w = lab, True
            if kind=="monthly" and lab: m_lab, got_m = lab, True
            if got_d and got_w and got_m:
                break

    out = {
        "inv_d_label": d_lab, "inv_w_label": w_lab, "inv_m_label": m_lab,
        "inv_d_vote": _map_summary_to_vote(d_lab),
        "inv_w_vote": _map_summary_to_vote(w_lab),
        "inv_m_vote": _map_summary_to_vote(m_lab),
    }
    CACHE.set(ck, out)
    time.sleep(0.15 + random.random()*0.15)
    return out

# ============= RANKING (avec nouvelles features) =============
def rank_score_row(s: pd.Series) -> float:
    score = 0.0
    # TV
    if s.get("tv_reco") == "STRONG_BUY": score += 3.0
    # TA local
    tb = s.get("technical_local")
    if tb == "Strong Buy": score += 2.0
    elif tb == "Buy":     score += 1.0
    # Analystes (Yahoo)
    ab = s.get("analyst_bucket")
    if ab == "Strong Buy": score += 2.0
    elif ab == "Buy":      score += 1.0
    av = s.get("analyst_votes")
    if isinstance(av,(int,float)) and av>0: score += min(av, 20)*0.05
    # Market cap (préférence small/mid)
    mc = s.get("market_cap")
    if isinstance(mc,(int,float)) and mc>0:
        score += max(0.0, 5.0 - math.log10(mc))
    # --- NEW: Investing votes (D/W/M) -> bonus doux
    for k in ("inv_d_vote","inv_w_vote","inv_m_vote"):
        v = s.get(k)
        if isinstance(v, (int,float)):
            # vote ∈ {-2,-1,0,1,2} -> +0.5 par point positif / -0.25 par point négatif (moins punitif)
            score += 0.5*max(v,0) + (-0.25)*min(v,0)
    # --- NEW: BarChart score/label
    bc_score = s.get("bc_score")
    if isinstance(bc_score, (int,float)):
        # normalise 0..100 -> 0..2 (max) en douceur
        score += (float(bc_score)/100.0) * 2.0
    bc_label = (s.get("bc_label") or "").upper()
    if "STRONG BUY" in bc_label: score += 1.0
    elif bc_label == "BUY":      score += 0.5
    elif bc_label == "SELL":     score -= 0.5
    elif "STRONG SELL" in bc_label: score -= 1.0

    return score

# ============= MAIN =============
def main():
    print("Chargement des tickers Russell 1000 + 2000…")
    tickers_df = load_universe()
    print(f"Tickers dans l'univers: {len(tickers_df)}")

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

    rows=[]
    bc_hits = ivg_hits = 0
    for i, rec in enumerate(tickers_df.itertuples(index=False), 1):
        tv_sym, yf_sym = rec.tv_symbol, rec.yf_symbol
        try:
            tk = yf.Ticker(yf_sym)
            try: info = tk.get_info() or {}
            except: info = {}
            fi = getattr(tk, "fast_info", None)
            mcap = getattr(fi, "market_cap", None) if fi else None
            if mcap is None: mcap = info.get("marketCap")
            country = (info.get("country") or info.get("countryOfCompany") or "").strip()
            exch = info.get("exchange") or info.get("fullExchangeName") or ""
            tv_exchange = map_exchange_for_tv(exch)

            is_us_exchange = str(exch).upper() in {"NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"}
            if country and (country.upper() not in {"USA","US","UNITED STATES","UNITED STATES OF AMERICA"} and not is_us_exchange):
                continue
            if isinstance(mcap,(int,float)) and mcap >= MAX_MARKET_CAP:
                continue

            hist = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=True, actions=False)
            bucket, votes, det = compute_local_technical_bucket(hist)
            if not bucket: continue

            tv = get_tv_summary(tv_sym, tv_exchange)
            time.sleep(DELAY_BETWEEN_TV_CALLS_SEC)

            analyst_mean  = info.get("recommendationMean")
            analyst_votes = info.get("numberOfAnalystOpinions")
            analyst_bucket = analyst_bucket_from_mean(analyst_mean)

            sec_cat, ind_cat = sector_from_catalog(tv_sym, yf_sym)
            sector_val   = sec_cat or (info.get("sector") or "").strip() or "Unknown"
            industry_val = ind_cat or (info.get("industry") or "").strip() or "Unknown"

            # --- NEW: External votes (gated) ---
            inv = {"inv_d_label":None,"inv_w_label":None,"inv_m_label":None,
                   "inv_d_vote":None,"inv_w_vote":None,"inv_m_vote":None}
            bc  = {"bc_score":None,"bc_label":None}

            should_call_externals = False
            if votes is not None and votes >= GATE_MIN_TECH_VOTES:
                should_call_externals = True
            if GATE_TV_ONLY_STRONG and tv.get("tv_reco") == "STRONG_BUY":
                should_call_externals = True
            if analyst_bucket in {"Buy","Strong Buy"}:
                should_call_externals = True

            base = base_sym(tv_sym)
            if should_call_externals:
                if USE_INVESTING:
                    inv = fetch_investing_summary(base)
                    if any([inv.get("inv_d_vote") is not None, inv.get("inv_w_vote") is not None, inv.get("inv_m_vote") is not None]):
                        ivg_hits += 1
                if USE_BARCHART:
                    bc = fetch_barchart_opinion(base)
                    if bc.get("bc_score") is not None or bc.get("bc_label"):
                        bc_hits += 1

            rows.append({
                "ticker_tv": tv_sym, "ticker_yf": yf_sym,
                "exchange": exch, "market_cap": mcap, "price": det.get("price"),
                "sector": sector_val, "industry": industry_val,
                "technical_local": bucket, "tech_score": votes,
                "tv_reco": tv["tv_reco"],
                "analyst_bucket": analyst_bucket,
                "analyst_mean": analyst_mean, "analyst_votes": analyst_votes,
                # NEW: externals
                "inv_d_label": inv.get("inv_d_label"),
                "inv_w_label": inv.get("inv_w_label"),
                "inv_m_label": inv.get("inv_m_label"),
                "inv_d_vote": inv.get("inv_d_vote"),
                "inv_w_vote": inv.get("inv_w_vote"),
                "inv_m_vote": inv.get("inv_m_vote"),
                "bc_score": bc.get("bc_score"),
                "bc_label": bc.get("bc_label"),
            })
        except Exception:
            continue

        if i % 50 == 0:
            print(f"{i}/{len(tickers_df)} traités… (ivg_hits={ivg_hits}, bc_hits={bc_hits})")

    df = pd.DataFrame(rows)

    # dump brut pour diag, toujours
    save_csv(df, "raw_candidates.csv")
    print(f"[DIAG] rows collectées: {len(df)}")

    if df.empty:
        print("⚠️ Aucun titre collecté après filtres US + MCAP + TA local.")
        save_csv(pd.DataFrame(columns=[
            "ticker_tv","ticker_yf","price","market_cap","sector","industry",
            "technical_local","tv_reco","analyst_bucket"
        ]), "debug_all_candidates.csv")
        # produire quand même des fichiers vides pour le front
        empty_cols = ["ticker_tv","ticker_yf","price","market_cap","sector","industry",
                      "technical_local","tech_score","tv_reco","analyst_bucket",
                      "analyst_mean","analyst_votes","rank_score",
                      "inv_d_label","inv_w_label","inv_m_label",
                      "inv_d_vote","inv_w_vote","inv_m_vote","bc_score","bc_label"]
        empty = pd.DataFrame(columns=empty_cols)
        save_csv(empty, "confirmed_STRONGBUY.csv")
        save_csv(empty, "anticipative_pre_signals.csv")
        save_csv(empty, "event_driven_signals.csv")
        save_csv(pd.DataFrame(columns=["candidate_type"]+empty_cols), "candidates_all_ranked.csv")
        return

    # Diagnostic global (avant filtres finaux)
    dbg = df.copy()
    dbg["rank_score"] = dbg.apply(rank_score_row, axis=1)
    dbg.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(dbg, "debug_all_candidates.csv")

    # ====== 1) Confirmed (assoupli) ======
    mask_tv = df["tv_reco"].eq("STRONG_BUY")
    mask_an = df["analyst_bucket"].isin({"Strong Buy","Buy"})
    confirmed = df[mask_tv & mask_an].copy()
    confirmed["rank_score"] = confirmed.apply(rank_score_row, axis=1)
    confirmed.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    confirmed_cols = [
        "ticker_tv","ticker_yf","price","market_cap","sector","industry",
        "technical_local","tech_score","tv_reco",
        "analyst_bucket","analyst_mean","analyst_votes","rank_score",
        # NEW columns visibles dans les CSV
        "inv_d_label","inv_w_label","inv_m_label",
        "inv_d_vote","inv_w_vote","inv_m_vote",
        "bc_score","bc_label"
    ]
    save_csv(confirmed[confirmed_cols], "confirmed_STRONGBUY.csv")

    # ====== 2) Pré-signaux (forcés) ======
    mask_pre = df["technical_local"].isin({"Buy","Strong Buy"}) | df["tv_reco"].eq("STRONG_BUY")
    pre = df[mask_pre].copy()
    pre["rank_score"] = pre.apply(rank_score_row, axis=1)
    pre.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(pre[confirmed_cols], "anticipative_pre_signals.csv")

    # ====== 3) Event-driven (proxy : analystes connus) ======
    evt = df[df["analyst_bucket"].notna()].copy()
    evt["rank_score"] = evt.apply(rank_score_row, axis=1)
    evt.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(evt[confirmed_cols], "event_driven_signals.csv")

    # ====== 4) Mix comparatif ======
    all_out = pd.concat([
        confirmed.assign(candidate_type="confirmed"),
        pre.assign(candidate_type="anticipative"),
        evt.assign(candidate_type="event")
    ], ignore_index=True)
    all_out.sort_values(["rank_score","candidate_type","market_cap"], ascending=[False, True, True], inplace=True)
    all_cols = ["candidate_type"] + confirmed_cols
    save_csv(all_out[all_cols], "candidates_all_ranked.csv")

    def _sector_universe(df_all: pd.DataFrame) -> pd.Series:
        # nb d’actions vues par secteur dans l’univers filtré (US, cap, etc.)
        return df_all.groupby(df_all["sector"].fillna("Unknown"))["ticker_yf"].nunique()

    def _sector_counts(df_part: pd.DataFrame) -> pd.Series:
        return df_part.groupby(df_part["sector"].fillna("Unknown"))["ticker_yf"].nunique()

    # Univers par secteur (après filtres)
    univ = _sector_universe(df)

    # Comptes bruts par secteur
    c_conf = _sector_counts(confirmed).reindex(univ.index, fill_value=0)
    c_pre  = _sector_counts(pre).reindex(univ.index, fill_value=0)
    c_evt  = _sector_counts(evt).reindex(univ.index, fill_value=0)

    # Ratios (% du secteur en signal)
    pct_conf = (c_conf / univ.replace(0, np.nan) * 100).fillna(0)
    pct_pre  = (c_pre  / univ.replace(0, np.nan) * 100).fillna(0)
    pct_evt  = (c_evt  / univ.replace(0, np.nan) * 100).fillna(0)

    # Breadth score (simple et lisible)
    breadth = (2*pct_conf + 1*pct_pre - 1*pct_evt)

    # Momentum WoW depuis sector_history.csv (Si pas dispo -> 0)
    hist_path = "sector_history.csv"
    wow_conf = pd.Series(0.0, index=univ.index)
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        # dernier point semaine courante
        last_week = hist["date"].max()
        prev = hist[hist["date"] != last_week]
        if not prev.empty:
            prev_week = prev["date"].max()
            prev_conf = prev[(prev["date"]==prev_week) & (prev["bucket"]=="confirmed")] \
                          .set_index("sector")["count"]
            prev_univ = prev.groupby("sector")["count"].sum()
            prev_pct_conf = (prev_conf / prev_univ.replace(0,np.nan) * 100).fillna(0)
            wow_conf = (pct_conf - prev_pct_conf.reindex(univ.index).fillna(0))

    # Z-score du breadth (vs secteurs du jour)
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
        "wow_pct_confirmed": wow_conf.values,
    }).sort_values(["z_breadth","breadth"], ascending=[False, False])

    save_csv(sector_breadth, "sector_breadth.csv")

    # === 5) Sector history (weekly snapshot) ===============================
    from datetime import datetime, timezone
    HISTORY_CSV = "sector_history.csv"

    def _norm_sector(x):
        x = (x or "").strip() or "Unknown"
        return x

    def _count_by_sector(rows_df):
        out = {}
        for _, r in rows_df.iterrows():
            sec = _norm_sector(r.get("sector"))
            out[sec] = out.get(sec, 0) + 1
        return out

    now = datetime.now(timezone.utc)
    iso_year, iso_week, _ = now.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"

    counts_confirmed = _count_by_sector(confirmed)
    counts_pre       = _count_by_sector(pre)
    counts_events    = _count_by_sector(evt)

    lines = []
    for sec, n in counts_confirmed.items():
        lines.append((week_key, "confirmed", sec, int(n)))
    for sec, n in counts_pre.items():
        lines.append((week_key, "pre", sec, int(n)))
    for sec, n in counts_events.items():
        lines.append((week_key, "events", sec, int(n)))

    if os.path.exists(HISTORY_CSV):
        try:
            existing = pd.read_csv(HISTORY_CSV)
        except Exception:
            existing = pd.DataFrame(columns=["date","bucket","sector","count"])
    else:
        existing = pd.DataFrame(columns=["date","bucket","sector","count"])

    existing = existing[existing["date"] != week_key]
    new_df = pd.concat([existing, pd.DataFrame(lines, columns=["date","bucket","sector","count"])],
                       ignore_index=True)
    new_df["__key"] = new_df["date"].astype(str) + "|" + new_df["bucket"].astype(str) + "|" + new_df["sector"].astype(str)
    new_df = new_df.drop_duplicates(subset="__key").drop(columns="__key")
    save_csv(new_df, HISTORY_CSV)

    # === Journalisation des signaux (historique par ticker) ======================
    from datetime import datetime, timezone

    SIGNALS_CSV = "signals_history.csv"

    def _append_signals(bucket_name: str, df_bucket: pd.DataFrame):
        """Append (date, bucket, ticker_yf, ticker_tv, sector, industry, price)"""
        if df_bucket is None or df_bucket.empty:
            return pd.DataFrame(columns=["date","bucket","ticker_yf","ticker_tv","sector","industry","price"])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = df_bucket[["ticker_yf","ticker_tv","sector","industry","price"]].copy()
        out.insert(0, "bucket", bucket_name)
        out.insert(0, "date", now)
        return out

    hist_blocks = []
    hist_blocks.append(_append_signals("confirmed", confirmed))
    hist_blocks.append(_append_signals("pre",       pre))
    hist_blocks.append(_append_signals("events",    evt))
    hist_all = pd.concat(hist_blocks, ignore_index=True)

    # Dédupe (même date + même ticker_yf + bucket)
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

    # Sauvegarde racine + copie dans dashboard/public/
    save_csv(merged, SIGNALS_CSV)

    print(f"[OK] confirmed={len(confirmed)}, pre={len(pre)}, event={len(evt)}, all={len(all_out)} | ivg_hits={ivg_hits}, bc_hits={bc_hits}")

if __name__ == "__main__":
    main()
