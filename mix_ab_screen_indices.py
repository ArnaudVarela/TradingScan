# mix_ab_screen_indices.py
# Screener US (Russell 1000 + 2000)
# Sorties :
#   - confirmed_STRONGBUY.csv         (TV=STRONG_BUY ∩ Analystes∈{Strong Buy, Buy})
#   - anticipative_pre_signals.csv    (pré-signaux : TA∈{Buy,Strong Buy} ou TV=STRONG_BUY)
#   - event_driven_signals.csv        (proxy événements : analystes connus)
#   - candidates_all_ranked.csv       (tous les candidats triés)
#   - debug_all_candidates.csv        (diagnostic avant filtres finaux)

import warnings, time, re, os, math
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

# ============= HELPERS =============
def yf_norm(s:str)->str: return s.replace(".", "-")
def tv_norm(s:str)->str: return s.replace("-", ".")

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
        try: ticks += fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_1000_Index")
        except:
            if os.path.exists(R1K_FALLBACK_CSV):
                ticks += pd.read_csv(R1K_FALLBACK_CSV)["Ticker"].astype(str).tolist()
    if INCLUDE_RUSSELL_2000:
        try: ticks += fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_2000_Index")
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

# === Indicateurs via API fonctionnelle pandas_ta ===
def compute_local_technical_bucket(hist: pd.DataFrame):
    """
    Renvoie (bucket, votes, details{price, sma20, sma50, sma200, rsi, macd, macds, stoch_k, stoch_d})
    Tolérante si SMA200 absente. Nécessite min 60 barres.
    """
    if hist is None or hist.empty or len(hist) < 60:
        return None, None, {}

    close = hist["Close"]; high = hist["High"]; low = hist["Low"]
    try:
        s20  = ta.sma(close, length=20)
        s50  = ta.sma(close, length=50)
        s200 = ta.sma(close, length=200)
        rsi  = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)      # cols: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        stoch= ta.stoch(high, low, close, k=14, d=3, smooth_k=3) # cols: STOCHk_14_3_3, STOCHd_14_3_3
    except Exception:
        return None, None, {}

    # valeurs finales
    try:
        price     = float(close.iloc[-1])
        s20_last  = float(s20.dropna().iloc[-1]); s50_last  = float(s50.dropna().iloc[-1])
        s200_last = float(s200.dropna().iloc[-1]) if s200 is not None and not s200.dropna().empty else np.nan
        rsi_last  = float(rsi.dropna().iloc[-1])

        macd_last = macds_last = np.nan
        if macd is not None and not macd.dropna().empty and \
           {"MACD_12_26_9","MACDs_12_26_9"}.issubset(macd.columns):
            macd_last  = float(macd["MACD_12_26_9"].dropna().iloc[-1])
            macds_last = float(macd["MACDs_12_26_9"].dropna().iloc[-1])

        stoch_k = stoch_d = np.nan
        if stoch is not None and not stoch.dropna().empty and \
           {"STOCHk_14_3_3","STOCHd_14_3_3"}.issubset(stoch.columns):
            stoch_k = float(stoch["STOCHk_14_3_3"].dropna().iloc[-1])
            stoch_d = float(stoch["STOCHd_14_3_3"].dropna().iloc[-1])
    except Exception:
        return None, None, {}

    # votes
    votes = 0
    votes += 1 if price > s20_last else -1
    votes += 1 if s20_last > s50_last else -1
    if not np.isnan(s200_last): votes += 1 if s50_last > s200_last else -1
    if rsi_last >= 55: votes += 1
    elif rsi_last <= 45: votes -= 1
    if not np.isnan(macd_last) and not np.isnan(macds_last):
        votes += 1 if macd_last > macds_last else -1
    if not np.isnan(stoch_k) and not np.isnan(stoch_d):
        votes += 1 if stoch_k > stoch_d else -1

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

def rank_score_row(s: pd.Series) -> float:
    """Score pour trier les candidats (plus haut = mieux)."""
    score = 0.0
    # TV
    if s.get("tv_reco") == "STRONG_BUY": score += 3.0
    # TA local
    tb = s.get("technical_local")
    if tb == "Strong Buy": score += 2.0
    elif tb == "Buy": score += 1.0
    # Analystes
    ab = s.get("analyst_bucket")
    if ab == "Strong Buy": score += 2.0
    elif ab == "Buy": score += 1.0
    # Votes analystes
    av = s.get("analyst_votes")
    if isinstance(av,(int,float)) and av>0: score += min(av, 20)*0.05
    # Plus petite cap favorisée
    mc = s.get("market_cap")
    if isinstance(mc,(int,float)) and mc>0:
        score += max(0.0, 5.0 - math.log10(mc))
    return score

# ============= MAIN =============
def main():
    print("Chargement des tickers Russell 1000 + 2000…")
    universe = load_universe()
    print(f"Tickers dans l'univers: {len(universe)}")

    rows=[]
    for i, rec in enumerate(universe.itertuples(index=False), 1):
        tv_sym, yf_sym = rec.tv_symbol, rec.yf_symbol
        try:
            tk = yf.Ticker(yf_sym)

            # infos émetteur (robuste)
            try: info = tk.get_info() or {}
            except: info = {}
            fi = getattr(tk, "fast_info", None)
            mcap = getattr(fi, "market_cap", None) if fi else None
            if mcap is None: mcap = info.get("marketCap")
            country = (info.get("country") or info.get("countryOfCompany") or "").strip()
            exch = info.get("exchange") or info.get("fullExchangeName") or ""
            tv_exchange = map_exchange_for_tv(exch)

            # US + cap
            is_us_exchange = str(exch).upper() in {"NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"}
            if country and (country.upper() not in {"USA","US","UNITED STATES","UNITED STATES OF AMERICA"} and not is_us_exchange):
                continue
            if isinstance(mcap,(int,float)) and mcap >= MAX_MARKET_CAP:
                continue

            # historique
            hist = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=True, actions=False)
            bucket, votes, det = compute_local_technical_bucket(hist)
            if not bucket: continue

            # TV
            tv = get_tv_summary(tv_sym, tv_exchange)
            time.sleep(DELAY_BETWEEN_TV_CALLS_SEC)

            # Analystes
            analyst_mean  = info.get("recommendationMean")
            analyst_votes = info.get("numberOfAnalystOpinions")
            analyst_bucket = analyst_bucket_from_mean(analyst_mean)

            rows.append({
                "ticker_tv": tv_sym, "ticker_yf": yf_sym,
                "exchange": exch, "market_cap": mcap, "price": det.get("price"),
                "technical_local": bucket, "tech_score": votes,
                "tv_reco": tv["tv_reco"],
                "analyst_bucket": analyst_bucket,
                "analyst_mean": analyst_mean, "analyst_votes": analyst_votes,
            })
        except Exception:
            continue

        if i % 50 == 0:
            print(f"{i}/{len(universe)} traités…")

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ Aucun titre collecté après filtres US + MCAP + TA local.")
        # on écrit tout de même un CSV diagnostic vide
        pd.DataFrame(columns=["ticker_tv","ticker_yf","price","market_cap","technical_local","tv_reco","analyst_bucket"]).to_csv("debug_all_candidates.csv", index=False)
        return

    # Diagnostic global (avant filtres finaux)
    dbg = df.copy()
    dbg["rank_score"] = dbg.apply(rank_score_row, axis=1)
    dbg.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    dbg.to_csv("debug_all_candidates.csv", index=False)

    # ====== 1) Confirmed (assoupli) ======
    mask_tv = df["tv_reco"].eq("STRONG_BUY")
    mask_an = df["analyst_bucket"].isin({"Strong Buy","Buy"})
    confirmed = df[mask_tv & mask_an].copy()
    confirmed["rank_score"] = confirmed.apply(rank_score_row, axis=1)
    confirmed.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    confirmed_cols = ["ticker_tv","ticker_yf","price","market_cap","technical_local","tech_score",
                      "tv_reco","analyst_bucket","analyst_mean","analyst_votes","rank_score"]
    confirmed[confirmed_cols].to_csv("confirmed_STRONGBUY.csv", index=False)

    # ====== 2) Pré-signaux (forcés) ======
    mask_pre = df["technical_local"].isin({"Buy","Strong Buy"}) | df["tv_reco"].eq("STRONG_BUY")
    pre = df[mask_pre].copy()
    pre["rank_score"] = pre.apply(rank_score_row, axis=1)
    pre.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    pre[confirmed_cols].to_csv("anticipative_pre_signals.csv", index=False)

    # ====== 3) Event-driven (proxy : analystes connus) ======
    evt = df[df["analyst_bucket"].notna()].copy()
    evt["rank_score"] = evt.apply(rank_score_row, axis=1)
    evt.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    evt[confirmed_cols].to_csv("event_driven_signals.csv", index=False)

    # ====== 4) Mix comparatif ======
    all_out = pd.concat([
        confirmed.assign(candidate_type="confirmed"),
        pre.assign(candidate_type="anticipative"),
        evt.assign(candidate_type="event")
    ], ignore_index=True)
    all_out.sort_values(["rank_score","candidate_type","market_cap"], ascending=[False, True, True], inplace=True)
    all_cols = ["candidate_type"] + confirmed_cols
    all_out[all_cols].to_csv("candidates_all_ranked.csv", index=False)

    print(f"[OK] confirmed={len(confirmed)}, pre={len(pre)}, event={len(evt)}, all={len(all_out)}")

if __name__ == "__main__":
    main()
