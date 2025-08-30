# mix_ab_screen_indices.py
# Screener US (Russell 1000 + 2000)
# Assoupli + pré-signaux + diagnostics + TRI par score
import warnings, time, os, re, math
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
    for t in tables:
        for c in t.columns:
            name = str(c).lower()
            if "ticker" in name or "symbol" in name:
                ser = (t[c].astype(str).str.strip())
                vals = [s for s in ser if re.match(r"^[A-Za-z.\-]+$", s)]
                if vals: return vals
    return []

def load_universe()->pd.DataFrame:
    ticks = []
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
    """Renvoie (bucket, votes, details{price}). Tolérant si SMA200 manquante."""
    if hist is None or hist.empty or len(hist) < 60:
        return None, None, {}
    close, high, low = hist["Close"], hist["High"], hist["Low"]
    s20 = close.ta.sma(20); s50 = close.ta.sma(50); s200 = close.ta.sma(200)
    rsi = close.ta.rsi(14)
    macd = close.ta.macd(12,26,9)
    stoch = ta.stoch(high,low,close)

    try:
        price = float(close.iloc[-1])
        votes = 0
        votes += 1 if price > float(s20.iloc[-1]) else -1
        votes += 1 if float(s20.iloc[-1]) > float(s50.iloc[-1]) else -1
        if s200 is not None and not s200.dropna().empty:
            votes += 1 if float(s50.iloc[-1]) > float(s200.iloc[-1]) else -1
        r = float(rsi.iloc[-1])
        if r >= 55: votes += 1
        elif r <= 45: votes -= 1
        if macd is not None and not macd.dropna().empty:
            if float(macd["MACD_12_26_9"].iloc[-1]) > float(macd["MACDs_12_26_9"].iloc[-1]): votes += 1
        if stoch is not None and not stoch.dropna().empty:
            if float(stoch["STOCHk_14_3_3"].iloc[-1]) > float(stoch["STOCHd_14_3_3"].iloc[-1]): votes += 1

        if votes >= 4: bucket = "Strong Buy"
        elif votes >= 2: bucket = "Buy"
        elif votes <= -4: bucket = "Strong Sell"
        elif votes <= -2: bucket = "Sell"
        else: bucket = "Neutral"
        return bucket, int(votes), {"price": price}
    except:
        return None, None, {}

def analyst_bucket_from_mean(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    if x < 1.6: return "Strong Buy"
    if x < 2.5: return "Buy"
    if x < 3.5: return "Hold"
    if x < 4.2: return "Sell"
    return "Strong Sell"

def get_tv_summary(symbol: str, exchange: str):
    try:
        h = TA_Handler(symbol=symbol, screener="america", exchange=exchange, interval=TV_INTERVAL)
        s = h.get_analysis().summary
        return {"tv_reco": s.get("RECOMMENDATION")}
    except:
        return {"tv_reco": None}

def rank_score_row(s: pd.Series) -> float:
    """Score de tri: plus haut = mieux."""
    score = 0.0
    # TradingView
    if s.get("tv_reco") == "STRONG_BUY": score += 3.0
    # Technique locale
    tb = s.get("technical")
    if tb == "Strong Buy": score += 2.0
    elif tb == "Buy": score += 1.0
    # Analystes
    ab = s.get("analyst_bucket")
    if ab == "Strong Buy": score += 2.0
    elif ab == "Buy": score += 1.0
    # Votes analystes (si dispo)
    av = s.get("analyst_votes")
    if isinstance(av, (int,float)) and av > 0:
        score += min(av, 20) * 0.05
    # Market cap (favorise plus petit)
    mc = s.get("market_cap")
    if isinstance(mc, (int,float)) and mc > 0:
        score += max(0.0, 5.0 - math.log10(mc))
    return score

# ============= MAIN =============
def main():
    universe = load_universe()
    print(f"Tickers dans l'univers: {len(universe)}")

    rows = []
    for i, rec in enumerate(universe.itertuples(index=False), 1):
        yf_sym = rec.yf_symbol; tv_sym = rec.tv_symbol
        tk = yf.Ticker(yf_sym)

        # infos émetteur
        try:
            info = tk.get_info() or {}
        except:
            info = {}

        mcap = info.get("marketCap")
        country = (info.get("country") or info.get("countryOfCompany") or "").upper()
        exch = info.get("exchange") or info.get("fullExchangeName") or ""

        # Filtres US + cap
        if country and country not in {"USA","UNITED STATES","US"}:
            continue
        if isinstance(mcap, (int,float)) and mcap >= MAX_MARKET_CAP:
            continue

        # historique pour TA local
        try:
            hist = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=True, actions=False)
        except:
            continue
        bucket, votes, det = compute_local_technical_bucket(hist)
        if not bucket:
            continue

        # TV
        tv = get_tv_summary(tv_sym, map_exchange_for_tv(exch))
        time.sleep(DELAY_BETWEEN_TV_CALLS_SEC)

        # Analystes
        analyst_mean  = info.get("recommendationMean")
        analyst_votes = info.get("numberOfAnalystOpinions")
        analyst_bucket = analyst_bucket_from_mean(analyst_mean)

        rows.append({
            "ticker_tv": tv_sym,
            "ticker_yf": yf_sym,
            "exchange": exch,
            "market_cap": mcap,
            "price": det.get("price"),
            "technical": bucket,
            "tech_votes": votes,
            "tv_reco": tv["tv_reco"],
            "analyst_bucket": analyst_bucket,
            "analyst_mean": analyst_mean,
            "analyst_votes": analyst_votes
        })

        if i % 50 == 0:
            print(f"{i}/{len(universe)} traités…")

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ Aucun titre collecté après filtres US + MCAP + TA local.")
        # on écrit un CSV vide pour debug
        pd.DataFrame(columns=["ticker_tv","ticker_yf","price","market_cap","technical","tv_reco","analyst_bucket"]).to_csv("debug_all_candidates.csv", index=False)
        return

    # --- Diagnostic intermédiaire (avant filtrage TV/analystes) ---
    df.assign(rank_score=df.apply(rank_score_row, axis=1))\
      .sort_values(["rank_score","market_cap"], ascending=[False, True])\
      .to_csv("debug_all_candidates.csv", index=False)

    # ====== 1) Confirmed (assoupli) ======
    mask_tv = df["tv_reco"].eq("STRONG_BUY")
    mask_an = df["analyst_bucket"].isin({"Strong Buy","Buy"})
    confirmed = df[mask_tv & mask_an].copy()
    confirmed["rank_score"] = confirmed.apply(rank_score_row, axis=1)
    confirmed.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    confirmed_cols = ["ticker_tv","ticker_yf","price","market_cap","technical","tech_votes",
                      "tv_reco","analyst_bucket","analyst_mean","analyst_votes","rank_score"]
    confirmed[confirmed_cols].to_csv("confirmed_STRONGBUY.csv", index=False)

    # ====== 2) Pré-signaux (forcés) ======
    # On garde si technique ∈ {Buy, Strong Buy} OU TV=STRONG_BUY, même sans analystes.
    mask_pre = df["technical"].isin({"Buy","Strong Buy"}) | df["tv_reco"].eq("STRONG_BUY")
    pre = df[mask_pre].copy()
    pre["rank_score"] = pre.apply(rank_score_row, axis=1)
    pre.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    pre[confirmed_cols].to_csv("anticipative_pre_signals.csv", index=False)

    # ====== 3) Event-driven (proxy simple : analystes connus) ======
    # (Tu pourras remplacer par un vrai calendrier earnings quand on branchera la source)
    evt = df[df["analyst_bucket"].notna()].copy()
    evt["rank_score"] = evt.apply(rank_score_row, axis=1)
    evt.sort_values(["rank_score","market_cap"], ascending=[False, True], inplace=True)
    evt[confirmed_cols].to_csv("event_driven_signals.csv", index=False)

    # ====== 4) Candidates all (comparatif) ======
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
