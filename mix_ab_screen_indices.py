# mix_ab_screen_indices.py
# Screener US (Russell 1000 + 2000)
# Sorties :
#   - confirmed_STRONGBUY.csv
#   - anticipative_pre_signals.csv
#   - event_driven_signals.csv
#   - candidates_all_ranked.csv   (inclut désormais un bucket 'universe' = tout le seed)
#   - universe_today.csv          (diagnostic : tout l’univers avec TA/TV/analystes)
#   - debug_all_candidates.csv
#   - sector_breadth.csv
#   - sector_history.csv
#   - signals_history.csv
#   - raw_candidates.csv (diag brut)
#   - tv_failures.csv (diagnostic TradingView)

import warnings, time, os, math, random, re
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

# Gating = sert à PRIORISER les appels lents (ne filtre plus les sorties)
GATE_MIN_TECH_VOTES = 2

# TradingView budget & throttling
TV_MAX_CALLS = 450          # cap global d'appels TV par run (priorisés)
DELAY_BETWEEN_TV_CALLS_SEC = 0.10   # base delay + jitter
TV_COOLDOWN_EVERY = 25       # toutes les 25 requêtes, on fait une pause
TV_COOLDOWN_SEC   = 2.0
TV_MAX_CONSEC_FAIL = 12      # si trop d'échecs d'affilée -> pause
TV_FAIL_PAUSE_SEC  = 8.0
TV_MAX_PAUSE_TOTAL = 60.0    # ne pas dépasser Xs de pause cumulée liée à la protection

# Limites
MAX_MARKET_CAP = 200_000_000_000     # < 200B

# Sources
USE_ANALYSTS  = True
USE_INVESTING = False
USE_BARCHART  = True  # stub dessous (à brancher si tu as une clé API)

# Budget externes si tu actives Investing/BarChart (optionnel)
EXTERNAL_MAX   = 120
EXTERNAL_SPLIT = (0.4, 0.6)  # (Investing %, BarChart %)

SECTOR_CATALOG_PATH = "sector_catalog.csv"  # optionnel

# --- Sorties : racine + copie dans dashboard/public/ pour le front
PUBLIC_DIR = os.path.join("dashboard", "public")

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
    yf_symbols = list(dict.fromkeys(yf_symbols))  # dédoublonne en gardant l'ordre

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
            tk = yf.Ticker(sym)
            df = tk.history(period=period, interval=interval, auto_adjust=True, actions=False).dropna()
            if not df.empty and len(df) >= 60:
                all_hist[sym] = df
                count_fb += 1
        except Exception:
            continue

    return all_hist

def fast_ta_votes_from_hist(hist_df):
    return compute_local_technical_bucket(hist_df)

# ==== EXTERNAL STUBS (Investing / BarChart) ====
def fetch_investing_summary(base_symbol: str):
    return {"inv_d_label":None,"inv_w_label":None,"inv_m_label":None,
            "inv_d_vote":None,"inv_w_vote":None,"inv_m_vote":None}

def fetch_barchart_opinion(base_symbol: str):
    # stub : à brancher si tu as une clé API BarChart
    return {"bc_score":None, "bc_label":None}

# ========= TRADINGVIEW: MAX COVERAGE + ADAPTIVE THROTTLE =========
TV_EXCH_TRY = ["NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"]
_tv_cache = {}     # (symbol, exchange, interval) -> reco
_tv_fail_log = []  # liste des échecs (pour tv_failures.csv)

def tv_symbol_variants(tv_symbol: str, yf_symbol: str):
    s = (tv_symbol or "").upper()
    y = (yf_symbol or "").upper()
    variants = []
    if s: variants.append(s)
    base = base_sym(s)
    if base and base not in variants: variants.append(base)
    if "." in s:
        dash = s.replace(".", "-")
        if dash not in variants: variants.append(dash)
        nodot = s.replace(".", "")
        if nodot not in variants: variants.append(nodot)
    if "-" in y:
        dot = y.replace("-", ".")
        if dot not in variants: variants.append(dot)
        if y not in variants: variants.append(y)
    out = []
    for v in variants:
        v = v.strip()
        if v and v not in out:
            out.append(v)
    return out

def _tv_try(symbol: str, exchange: str, interval=TV_INTERVAL, retries=1):
    key = (symbol, exchange, interval)
    if key in _tv_cache:
        return _tv_cache[key]
    last_err = None
    for attempt in range(retries+1):
        try:
            h = TA_Handler(symbol=symbol, screener="america", exchange=exchange, interval=interval)
            s = h.get_analysis().summary
            reco = s.get("RECOMMENDATION")
            _tv_cache[key] = reco
            # petite pause + jitter
            base = DELAY_BETWEEN_TV_CALLS_SEC
            if base:
                time.sleep(base + random.uniform(0, 0.03))
            return reco
        except Exception as e:
            last_err = str(e)
            # backoff léger
            time.sleep(0.04 + 0.06*attempt)
    _tv_cache[key] = None
    _tv_fail_log.append({"symbol":symbol, "exchange":exchange, "err":last_err or "unknown"})
    return None

def get_tv_summary_max(tv_symbol: str, yf_symbol: str, exchange_hint: str):
    candidates_sym = tv_symbol_variants(tv_symbol, yf_symbol)
    hint = map_exchange_for_tv(exchange_hint)
    exch_list = [hint] + [e for e in TV_EXCH_TRY if e != hint]
    for sym in candidates_sym:
        reco = _tv_try(sym, hint)
        if reco:
            return {"tv_reco": reco, "tv_symbol_used": sym, "tv_exchange_used": hint}
        for ex in exch_list:
            reco = _tv_try(sym, ex)
            if reco:
                return {"tv_reco": reco, "tv_symbol_used": sym, "tv_exchange_used": ex}
    return {"tv_reco": None, "tv_symbol_used": None, "tv_exchange_used": None}

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
        bucket, votes, det = fast_ta_votes_from_hist(hist)
        if not bucket:
            continue
        seed_rows.append({
            "ticker_tv": tv_sym, "ticker_yf": yf_sym,
            "technical_local": bucket, "tech_score": votes,
            "price": det.get("price"),
        })

    seed = pd.DataFrame(seed_rows)
    print(f"[INFO] Seed avec TA local (>=60 barres): {len(seed)}/{len(tickers_df)}")

    if seed.empty:
        # sorties vides cohérentes
        for name in [
            "debug_all_candidates.csv","confirmed_STRONGBUY.csv","anticipative_pre_signals.csv",
            "event_driven_signals.csv","candidates_all_ranked.csv","sector_breadth.csv",
            "universe_today.csv","raw_candidates.csv"
        ]:
            save_csv(pd.DataFrame(), name)
        return

    # === PASS 2: Info Yahoo & TV (budgetisé/priorisé), externes sous budget ===
    # 2a) Prépare ranking pour PRIORISER les appels TV/externes
    seed = seed.sort_values(["tech_score"], ascending=[False]).reset_index(drop=True)

    # 2b) Appels TV priorisés (budget + throttle adaptatif)
    tv_budget = min(TV_MAX_CALLS, len(seed))
    tv_ok = tv_ko = 0
    consec_fail = 0
    pause_spent = 0.0

    # Pour éviter de perdre l'univers en sortie, on prépare un dict résultat par ticker
    tv_results = {}

    for idx, rec in enumerate(seed.itertuples(index=False), 1):
        tv_sym, yf_sym = rec.ticker_tv, rec.ticker_yf
        if idx <= tv_budget:
            # cooldown périodique
            if idx % TV_COOLDOWN_EVERY == 0 and pause_spent < TV_MAX_PAUSE_TOTAL:
                time.sleep(TV_COOLDOWN_SEC)
                pause_spent += TV_COOLDOWN_SEC

            # On a besoin de l'exchange hint -> via fast_info/get_info léger
            exch_hint = ""
            try:
                info = {}
                tk = yf.Ticker(yf_sym)
                fi = getattr(tk, "fast_info", None)
                exch_hint = getattr(fi, "exchange", None) or ""
                if not exch_hint:
                    try:
                        info = tk.get_info() or {}
                        exch_hint = info.get("exchange") or info.get("fullExchangeName") or ""
                    except Exception:
                        info = {}
            except Exception:
                pass

            tv = get_tv_summary_max(tv_sym, yf_sym, exch_hint)
            if tv.get("tv_reco"):
                tv_ok += 1
                consec_fail = 0
            else:
                tv_ko += 1
                consec_fail += 1
                # si avalanche d'échecs → petite pause (probable ratelimit)
                if consec_fail >= TV_MAX_CONSEC_FAIL and pause_spent < TV_MAX_PAUSE_TOTAL:
                    time.sleep(TV_FAIL_PAUSE_SEC)
                    pause_spent += TV_FAIL_PAUSE_SEC
                    consec_fail = 0
            tv_results[yf_sym] = tv
        else:
            # Pas d'appel TV (budget dépassé) → None
            tv_results[yf_sym] = {"tv_reco": None, "tv_symbol_used": None, "tv_exchange_used": None}

    print(f"[TV] budget={tv_budget}, ok={tv_ok}, ko={tv_ko}, pause_total≈{pause_spent:.1f}s")

    # 2c) Enrichissements Yahoo (analystes, secteurs, mcap) + externes sous budget
    rows = []
    used_inv = used_bc = 0
    if USE_INVESTING and USE_BARCHART:
        budget_inv = int(EXTERNAL_SPLIT[0]*EXTERNAL_MAX)
        budget_bc  = EXTERNAL_MAX - budget_inv
    else:
        budget_inv = EXTERNAL_MAX if USE_INVESTING else 0
        budget_bc  = EXTERNAL_MAX if USE_BARCHART else 0

    for i, rec in enumerate(seed.itertuples(index=False), 1):
        tv_sym, yf_sym = rec.ticker_tv, rec.ticker_yf
        tech_bucket, tech_votes, last_price = rec.technical_local, rec.tech_score, rec.price

        try:
            tk = yf.Ticker(yf_sym)
            fi = getattr(tk, "fast_info", None)
            mcap = getattr(fi, "market_cap", None)
            info = {}
            if mcap is None or USE_ANALYSTS:
                try:
                    info = tk.get_info() or {}
                    if mcap is None:
                        mcap = info.get("marketCap")
                except Exception:
                    info = {}

            country = (info.get("country") or info.get("countryOfCompany") or "").strip()
            exch = info.get("exchange") or info.get("fullExchangeName") or ""
            tv_exchange_hint = map_exchange_for_tv(exch)

            # Filtre US + Market Cap (pour la suite des sorties — mais on garde universe_today pour tous seed)
            is_us_exchange = str(exch).upper() in {"NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"}
            if country and (country.upper() not in {"USA","US","UNITED STATES","UNITED STATES OF AMERICA"} and not is_us_exchange):
                continue
            if isinstance(mcap,(int,float)) and mcap >= MAX_MARKET_CAP:
                continue

            tv = tv_results.get(yf_sym) or {"tv_reco": None, "tv_symbol_used": None, "tv_exchange_used": None}

            analyst_mean  = info.get("recommendationMean") if USE_ANALYSTS else None
            analyst_votes = info.get("numberOfAnalystOpinions") if USE_ANALYSTS else None
            analyst_bucket = analyst_bucket_from_mean(analyst_mean) if USE_ANALYSTS else None

            sec_cat, ind_cat = sector_from_catalog(tv_sym, yf_sym)
            sector_val   = sec_cat or (info.get("sector") or "").strip() or "Unknown"
            industry_val = ind_cat or (info.get("industry") or "").strip() or "Unknown"

            # Externals (optionnels et budgetisés)
            inv = {"inv_d_label":None,"inv_w_label":None,"inv_m_label":None,
                   "inv_d_vote":None,"inv_w_vote":None,"inv_m_vote":None}
            bc  = {"bc_score":None,"bc_label":None}

            promising = (tech_votes is not None and tech_votes >= GATE_MIN_TECH_VOTES)
            if promising and tv.get("tv_reco") == "STRONG_BUY":
                promising = True
            if promising and USE_ANALYSTS and analyst_bucket in {"Buy","Strong Buy"}:
                promising = True

            if promising:
                if USE_INVESTING and used_inv < budget_inv:
                    inv = fetch_investing_summary(base_sym(tv_sym)); used_inv += 1
                if USE_BARCHART and used_bc < budget_bc:
                    bc = fetch_barchart_opinion(base_sym(tv_sym)); used_bc += 1

            rows.append({
                "ticker_tv": tv_sym, "ticker_yf": yf_sym,
                "exchange": exch, "market_cap": mcap, "price": float(last_price) if last_price is not None else None,
                "sector": sector_val, "industry": industry_val,
                "technical_local": tech_bucket, "tech_score": int(tech_votes) if tech_votes is not None else None,
                "tv_reco": tv.get("tv_reco"),
                "tv_symbol_used": tv.get("tv_symbol_used"),
                "tv_exchange_used": tv.get("tv_exchange_used"),
                "analyst_bucket": analyst_bucket,
                "analyst_mean": analyst_mean, "analyst_votes": analyst_votes,
                "inv_d_label": inv.get("inv_d_label"), "inv_w_label": inv.get("inv_w_label"), "inv_m_label": inv.get("inv_m_label"),
                "inv_d_vote": inv.get("inv_d_vote"),   "inv_w_vote": inv.get("inv_w_vote"),   "inv_m_vote": inv.get("inv_m_vote"),
                "bc_score": bc.get("bc_score"), "bc_label": bc.get("bc_label"),
            })
        except Exception:
            continue

        if i % 50 == 0:
            print(f"{i}/{len(seed)} enrichis… (TV used {min(i,tv_budget)}/{tv_budget}, inv_used={used_inv}/{budget_inv}, bc_used={used_bc}/{budget_bc})")

    # Sauvegarde des échecs TV pour diagnostic
    if _tv_fail_log:
        save_csv(pd.DataFrame(_tv_fail_log), "tv_failures.csv")

    df = pd.DataFrame(rows)

    # dump brut diag
    save_csv(df, "raw_candidates.csv")

    # ========== Sorties ==========
    if df.empty:
        # fichiers vides cohérents
        for name in [
            "debug_all_candidates.csv","confirmed_STRONGBUY.csv","anticipative_pre_signals.csv",
            "event_driven_signals.csv","candidates_all_ranked.csv","sector_breadth.csv",
            "universe_today.csv"
        ]:
            save_csv(pd.DataFrame(), name)
        return

    # 0) universe_today.csv : TOUT l’univers vu aujourd’hui (après filtres US/MCAP)
    save_csv(df.copy(), "universe_today.csv")

    # Scoring
    dbg = df.copy()
    def rank_score_row(s: pd.Series) -> float:
        score = 0.0
        if s.get("tv_reco") == "STRONG_BUY": score += 3.0
        tb = s.get("technical_local")
        if tb == "Strong Buy": score += 2.0
        elif tb == "Buy":     score += 1.0
        ab = s.get("analyst_bucket")
        if ab == "Strong Buy": score += 2.0
        elif ab == "Buy":      score += 1.0
        av = s.get("analyst_votes")
        if isinstance(av,(int,float)) and av>0: score += min(av, 20)*0.05
        mc = s.get("market_cap")
        if isinstance(mc,(int,float)) and mc>0:
            score += max(0.0, 5.0 - math.log10(mc))
        bc_score = s.get("bc_score")
        if isinstance(bc_score,(int,float)):
            score += float(bc_score) * 0.01
        for k in ("inv_d_vote","inv_w_vote","inv_m_vote"):
            v = s.get(k)
            if isinstance(v,(int,float)):
                score += 0.2 * float(v)
        return score

    dbg["rank_score"] = dbg.apply(rank_score_row, axis=1)
    dbg.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(dbg, "debug_all_candidates.csv")

    confirmed_cols = [
        "ticker_tv","ticker_yf","price","market_cap","sector","industry",
        "technical_local","tech_score","tv_reco","tv_symbol_used","tv_exchange_used",
        "analyst_bucket","analyst_mean","analyst_votes","rank_score",
        "inv_d_label","inv_w_label","inv_m_label",
        "inv_d_vote","inv_w_vote","inv_m_vote","bc_score","bc_label"
    ]

    # 1) Confirmed
    mask_tv = df["tv_reco"].eq("STRONG_BUY")
    mask_an = df["analyst_bucket"].isin({"Strong Buy","Buy"}) if USE_ANALYSTS else True
    confirmed = df[mask_tv & mask_an].copy()
    confirmed["rank_score"] = confirmed.apply(rank_score_row, axis=1)
    confirmed.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(confirmed[confirmed_cols], "confirmed_STRONGBUY.csv")

    # 2) Pré-signaux
    mask_pre = df["technical_local"].isin({"Buy","Strong Buy"}) | df["tv_reco"].eq("STRONG_BUY")
    pre = df[mask_pre].copy()
    pre["rank_score"] = pre.apply(rank_score_row, axis=1)
    pre.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(pre[confirmed_cols], "anticipative_pre_signals.csv")

    # 3) Event-driven (proxy)
    evt = df[df["analyst_bucket"].notna()].copy() if USE_ANALYSTS else df.copy()
    evt["rank_score"] = evt.apply(rank_score_row, axis=1)
    evt.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    save_csv(evt[confirmed_cols], "event_driven_signals.csv")

    # 4) Mix comparatif — inclut désormais UNIVERSE
    others_idx = df.index.difference(pd.concat([confirmed, pre, evt]).index)
    universe_rest = df.loc[others_idx].copy()
    universe_rest["rank_score"] = universe_rest.apply(rank_score_row, axis=1)
    universe_rest.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)

    all_out = pd.concat([
        confirmed.assign(candidate_type="confirmed"),
        pre.assign(candidate_type="anticipative"),
        evt.assign(candidate_type="event"),
        universe_rest.assign(candidate_type="universe"),  # << NEW
    ], ignore_index=True)

    all_out.sort_values(["candidate_type","rank_score","market_cap"], ascending=[True, False, True], inplace=True)
    all_cols = ["candidate_type"] + confirmed_cols
    save_csv(all_out[all_cols], "candidates_all_ranked.csv")

    # 5) Sector breadth (sur df complet du jour)
    def _sector_universe(df_all: pd.DataFrame) -> pd.Series:
        return df_all.groupby(df_all["sector"].fillna("Unknown"))["ticker_yf"].nunique()

    def _sector_counts(df_part: pd.DataFrame) -> pd.Series:
        return df_part.groupby(df_part["sector"].fillna("Unknown"))["ticker_yf"].nunique()

    univ = _sector_universe(df)
    c_conf = _sector_counts(confirmed).reindex(univ.index, fill_value=0)
    c_pre  = _sector_counts(pre).reindex(univ.index, fill_value=0)
    c_evt  = _sector_counts(evt).reindex(univ.index, fill_value=0)

    pct_conf = (c_conf / univ.replace(0, np.nan) * 100).fillna(0)
    pct_pre  = (c_pre  / univ.replace(0, np.nan) * 100).fillna(0)
    pct_evt  = (c_evt  / univ.replace(0, np.nan) * 100).fillna(0)

    breadth = (2*pct_conf + 1*pct_pre - 1*pct_evt)
    z_breadth = (breadth - breadth.mean()) / (breadth.std(ddof=1) or 1)

    # Momentum WoW depuis sector_history.csv (Si pas dispo -> 0)
    hist_path = "sector_history.csv"
    wow_conf = pd.Series(0.0, index=univ.index)
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        last_week = hist["date"].max()
        prev = hist[hist["date"] != last_week]
        if not prev.empty:
            prev_week = prev["date"].max()
            prev_conf = prev[(prev["date"]==prev_week) & (prev["bucket"]=="confirmed")] \
                          .set_index("sector")["count"]
            prev_univ = prev.groupby("sector")["count"].sum()
            prev_pct_conf = (prev_conf / prev_univ.replace(0,np.nan) * 100).fillna(0)
            wow_conf = (pct_conf - prev_pct_conf.reindex(univ.index).fillna(0))

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

    # 6) Sector history (weekly snapshot)
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

    now_dt = datetime.now(timezone.utc)
    iso_year, iso_week, _ = now_dt.isocalendar()
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

    # 7) Journalisation des signaux (historique par ticker)
    SIGNALS_CSV = "signals_history.csv"

    def _append_signals(bucket_name: str, df_bucket: pd.DataFrame):
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
    fill_an = df["analyst_bucket"].notna().mean() if not df.empty else 0.0
    fill_bc = ((df["bc_label"].notna()) | (df["bc_score"].notna())).mean() if not df.empty else 0.0
    print(f"[OK] universe={len(df)} (après filtres US/MCAP), confirmed={len(confirmed)}, pre={len(pre)}, event={len(evt)}, universe_rest={len(universe_rest)}, all={len(all_out)} — TV fill={fill_tv:.0%}, Analysts fill={fill_an:.0%}, BC fill={fill_bc:.0%}")

if __name__ == "__main__":
    main()
