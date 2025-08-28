import pandas as pd, numpy as np, time, re, warnings
import yfinance as yf
import pandas_ta as ta
from tradingview_ta import TA_Handler, Interval
import requests

warnings.filterwarnings("ignore")

# =========================
#  CONFIG
# =========================
USE_WIKI_LISTS = True       # True: charge S&P500 & Nasdaq-100 depuis Wikipedia. False: charge depuis CSV perso ci-dessous.
CUSTOM_CSV_SP500 = "sp500.csv"     # colonnes attendues: Symbol
CUSTOM_CSV_NAS100 = "nas100.csv"   # colonnes attendues: Ticker

INTERVAL = "1d"             # interval pour yfinance
PERIOD = "6mo"              # historique minimum pour TA
TV_INTERVAL = Interval.INTERVAL_1_DAY

# Filtres finaux
ANALYST_GOOD_BUCKETS = {"Strong Buy", "Buy"}
TV_GOOD = {"STRONG_BUY"}     # tu peux inclure "BUY" si tu veux être plus large

# =========================
#  UTILS
# =========================
def load_sp500_and_nas100(use_wiki=True, sp500_csv=None, nas100_csv=None):
    if use_wiki:
        # S&P 500
        sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sp_symbols = sp["Symbol"].astype(str).str.strip().tolist()

        # Nasdaq-100
        # Sur la page NASDAQ-100, le tableau des tickers est souvent à l'index 3, mais on sécurise:
        nas_tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        nas = None
        for tbl in nas_tables:
            cols = [c.lower() for c in tbl.columns.astype(str)]
            if any("ticker" in c for c in cols):
                nas = tbl
                break
        if nas is None:
            raise RuntimeError("Impossible de trouver la table des tickers Nasdaq-100 sur Wikipedia.")
        # trouve la colonne ticker
        ticker_col = [c for c in nas.columns if re.search("ticker", str(c), re.I)]
        nas_symbols = nas[ticker_col[0]].astype(str).str.strip().tolist()
    else:
        sp = pd.read_csv(sp500_csv)
        nas = pd.read_csv(nas100_csv)
        sp_symbols = sp["Symbol"].astype(str).str.strip().tolist()
        nas_symbols = nas["Ticker"].astype(str).str.strip().tolist()

    # dédoublonne et normalise
    # Remarque: yfinance attend BRK-B au lieu de BRK.B ; gardons les deux formes: tv_symbol (original), yf_symbol (remap)
    def yf_norm(sym):
        return sym.replace(".", "-")  # ex: BRK.B -> BRK-B, BF.B -> BF-B

    tickers = sorted(set(sp_symbols + nas_symbols))
    df = pd.DataFrame({
        "tv_symbol": tickers,
        "yf_symbol": [yf_norm(t) for t in tickers]
    })
    return df

def compute_local_technical_bucket(hist: pd.DataFrame):
    """Retourne (bucket, score, details) sur la base d’indicateurs classiques."""
    close = hist["Close"]
    high, low = hist["High"], hist["Low"]

    # MAs
    sma20 = close.ta.sma(20)
    sma50 = close.ta.sma(50)
    sma200 = close.ta.sma(200)

    # RSI/MACD/Stoch
    rsi = close.ta.rsi(14)
    macd = close.ta.macd(12, 26, 9)  # cols: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    stoch = ta.stoch(high, low, close)  # cols: STOCHk_14_3_3, STOCHd_14_3_3

    if len(sma200.dropna()) == 0 or len(rsi.dropna()) == 0 or macd is None or stoch is None:
        return None, None, {}

    rsi_last = rsi.iloc[-1]
    macd_last = macd.iloc[-1]
    stoch_last = stoch.iloc[-1]

    votes = 0
    price = close.iloc[-1]
    s20, s50, s200 = sma20.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]

    # Trend/MAs
    votes += 1 if price > s20 else -1
    votes += 1 if s20 > s50 else -1
    votes += 1 if s50 > s200 else -1

    # RSI bias
    if rsi_last >= 55: votes += 1
    elif rsi_last <= 45: votes -= 1

    # MACD line > signal
    if macd_last["MACD_12_26_9"] > macd_last["MACDs_12_26_9"]: votes += 1
    else: votes -= 1

    # Stoch %K > %D
    if stoch_last["STOCHk_14_3_3"] > stoch_last["STOCHd_14_3_3"]: votes += 1
    else: votes -= 1

    # Bucket
    if votes >= 4: bucket = "Strong Buy"
    elif votes >= 2: bucket = "Buy"
    elif votes <= -4: bucket = "Strong Sell"
    elif votes <= -2: bucket = "Sell"
    else: bucket = "Neutral"

    details = dict(
        price=float(price),
        sma20=float(s20), sma50=float(s50), sma200=float(s200),
        rsi=float(rsi_last),
        macd=float(macd_last["MACD_12_26_9"]),
        macds=float(macd_last["MACDs_12_26_9"]),
        stoch_k=float(stoch_last["STOCHk_14_3_3"]),
        stoch_d=float(stoch_last["STOCHd_14_3_3"]),
    )
    return bucket, int(votes), details

def analyst_bucket_from_mean(x):
    # Yahoo: 1.0 Strong Buy ... 5.0 Sell
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x < 1.6: return "Strong Buy"
    if x < 2.5: return "Buy"
    if x < 3.5: return "Hold"
    if x < 4.2: return "Sell"
    return "Strong Sell"

def map_exchange_for_tv(yf_info_exch: str, ticker: str):
    """
    Essaie de déterminer l'exchange pour TradingView_ta.
    yfinance renvoie souvent 'NasdaqGS', 'NYSE', 'NYSEArca', etc.
    On mape grossièrement → 'NASDAQ' | 'NYSE' | 'AMEX'
    """
    if not yf_info_exch:
        # heuristique simple: beaucoup du Nasdaq-100 sont NASDAQ
        return "NASDAQ"
    e = yf_info_exch.upper()
    if "NASDAQ" in e: return "NASDAQ"
    if "NYSE" in e: return "NYSE"
    if "ARCA" in e: return "AMEX"  # TV classe souvent les ETF Arca côté AMEX
    # fallback
    return "NASDAQ"

def get_tv_summary(symbol: str, exchange: str):
    try:
        h = TA_Handler(
            symbol=symbol,
            screener="america",
            exchange=exchange,
            interval=TV_INTERVAL
        )
        s = h.get_analysis().summary  # dict: {'RECOMMENDATION': 'STRONG_BUY', 'BUY': xx, 'SELL': yy, ...}
        # uniformiser clés
        return {
            "tv_reco": s.get("RECOMMENDATION"),
            "tv_buy": s.get("BUY"),
            "tv_sell": s.get("SELL"),
            "tv_neutral": s.get("NEUTRAL"),
        }
    except Exception:
        return {"tv_reco": None, "tv_buy": None, "tv_sell": None, "tv_neutral": None}

# =========================
#  MAIN
# =========================
def main():
    print("Chargement des tickers S&P500 + Nasdaq-100…")
    tickers_df = load_sp500_and_nas100(
        use_wiki=USE_WIKI_LISTS,
        sp500_csv=CUSTOM_CSV_SP500,
        nas100_csv=CUSTOM_CSV_NAS100
    )

    rows = []
    n = len(tickers_df)
    for i, row in enumerate(tickers_df.itertuples(index=False), 1):
        tv_symbol = row.tv_symbol
        yf_symbol = row.yf_symbol

        try:
            tk = yf.Ticker(yf_symbol)
            info = {}
            try:
                info = tk.get_info() or {}
            except Exception:
                info = {}

            exch = info.get("exchange") or info.get("fullExchangeName") or ""
            tv_exchange = map_exchange_for_tv(exch, tv_symbol)

            hist = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=False, actions=False)
            if hist.empty or len(hist) < 60:
                # pas assez de données pour nos indicateurs
                continue

            # A) note technique locale
            local_bucket, local_score, local_details = compute_local_technical_bucket(hist)
            if local_bucket is None:
                continue

            # B) reco TradingView
            tv = get_tv_summary(tv_symbol, tv_exchange)

            # Analystes
            analyst_mean = info.get("recommendationMean")
            analyst_votes = info.get("numberOfAnalystOpinions")
            analyst_bucket = analyst_bucket_from_mean(analyst_mean)

            rows.append({
                "ticker_tv": tv_symbol,
                "ticker_yf": yf_symbol,
                "exchange_yf": exch,
                "exchange_tv": tv_exchange,
                "price": local_details.get("price"),
                "technical_local": local_bucket,
                "tech_score": local_score,
                "rsi": local_details.get("rsi"),
                "sma20": local_details.get("sma20"),
                "sma50": local_details.get("sma50"),
                "sma200": local_details.get("sma200"),
                "macd": local_details.get("macd"),
                "macds": local_details.get("macds"),
                "stoch_k": local_details.get("stoch_k"),
                "stoch_d": local_details.get("stoch_d"),
                "tv_reco": tv["tv_reco"],
                "tv_buy": tv["tv_buy"],
                "tv_neutral": tv["tv_neutral"],
                "tv_sell": tv["tv_sell"],
                "analyst_mean": analyst_mean,
                "analyst_votes": analyst_votes,
                "analyst_bucket": analyst_bucket,
            })

        except Exception as e:
            # on continue silencieusement pour robustesse
            continue

        if i % 25 == 0:
            print(f"{i}/{n} traités…")

        # Option: petit sleep pour éviter de spammer TradingView_ta
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if df.empty:
        print("Aucun résultat assemblé. Vérifie la connectivité et les dépendances.")
        return

    # 1) Strong Buy local + bonne note analystes
    strong_local = df.query("technical_local == 'Strong Buy'")
    strong_local_good_analysts = strong_local[strong_local["analyst_bucket"].isin(ANALYST_GOOD_BUCKETS)]
    strong_local_good_analysts = strong_local_good_analysts.sort_values(
        ["tech_score","analyst_votes"], ascending=[False, False]
    )
    strong_local_good_analysts.to_csv("strong_buy_local_and_good_analysts.csv", index=False)

    # 2) Intersection: Strong Buy local ET TV STRONG_BUY + bonne note analystes
    mask_tv_good = df["tv_reco"].isin(TV_GOOD)
    intersection = df[ (df["technical_local"] == "Strong Buy") & mask_tv_good & (df["analyst_bucket"].isin(ANALYST_GOOD_BUCKETS)) ]
    intersection = intersection.sort_values(["tech_score","analyst_votes"], ascending=[False, False])
    intersection.to_csv("intersection_local_TV_and_good_analysts.csv", index=False)

    # Affichage rapide
    print("\n=== Strong Buy local + bons analystes (top 30) ===")
    print(strong_local_good_analysts.head(30).to_string(index=False))

    print("\n=== INTERSECTION (Local Strong Buy ∩ TV STRONG_BUY) + bons analystes (top 30) ===")
    print(intersection.head(30).to_string(index=False))

if __name__ == "__main__":
    main()
