# Screener US (Russell 1000 + 2000)
# Filtres: US only, MarketCap < 200B, double Strong Buy (TradingView + analystes Yahoo)
# Sortie: tv_STRONGBUY__analyst_STRONGBUY__under200B.csv
import warnings, time, re, os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tradingview_ta import TA_Handler, Interval

warnings.filterwarnings("ignore")

# =========================
#  CONFIG
# =========================
# Univers: Russell 1000 + Russell 2000
INCLUDE_RUSSELL_1000 = True
INCLUDE_RUSSELL_2000 = True

# Fallback CSV (optionnel) si Wikipedia change trop souvent.
# Si présents à la racine du repo, ils seront utilisés en secours.
R1K_FALLBACK_CSV = "russell1000.csv"   # entête attendu: Ticker
R2K_FALLBACK_CSV = "russell2000.csv"   # entête attendu: Ticker

# Indicateurs locaux (période/interval Yahoo)
PERIOD = "6mo"
INTERVAL = "1d"
TV_INTERVAL = Interval.INTERVAL_1_DAY

# Filtres secteurs & cap
MAX_MARKET_CAP = 200_000_000_000  # < 200B$

# Fichier de sortie
OUTPUT_CSV = "tv_STRONGBUY__analyst_STRONGBUY__under200B.csv"

# Respect TradingView (lib non-officielle) : ne pas spammer
DELAY_BETWEEN_TV_CALLS_SEC = 0.05


# =========================
#  HELPERS
# =========================
def yf_norm(sym: str) -> str:
    """TradingView aime BRK.B ; yfinance aime BRK-B."""
    return sym.replace(".", "-")


def tv_norm(sym: str) -> str:
    """L’inverse pour TradingView_ta."""
    return sym.replace("-", ".")


def fetch_wikipedia_tickers(url: str):
    """
    Télécharge une page Wikipedia et tente d’extraire une colonne 'Ticker' OU 'Symbol'
    depuis n’importe quel tableau (en gérant MultiIndex d'entêtes).
    Renvoie la liste de tickers trouvés (strings).
    """
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"}
    resp = requests.get(url, headers=ua, timeout=45)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)

    def flatten_cols(cols):
        # Gère les MultiIndex d'entêtes
        out = []
        for c in cols:
            if isinstance(c, tuple):
                name = " ".join([str(x) for x in c if pd.notna(x)]).strip()
            else:
                name = str(c).strip()
            out.append(name)
        return out

    candidates = ("ticker", "symbol")  # wikipedia alterne entre Ticker / Symbol
    for t in tables:
        cols = flatten_cols(t.columns)
        lower = [c.lower() for c in cols]
        # Cherche une colonne qui matche 'ticker' OU 'symbol'
        col_idx = None
        for i, name in enumerate(lower):
            if any(key in name for key in candidates):
                col_idx = i
                break
        if col_idx is None:
            continue
        col_name = t.columns[col_idx]
        # Nettoie les valeurs
        ser = (
            t[col_name]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)   # supprime espaces internes (rare)
            .str.replace("\u200b", "", regex=False)  # zero-width space
        )
        # Filtre les lignes bizarres
        ser = ser[ser.str.match(r"^[A-Za-z.\-]+$")].dropna().tolist()
        if len(ser) > 0:
            return ser

    raise RuntimeError(f"Aucune colonne 'Ticker' ou 'Symbol' trouvée sur {url}")


def load_universe():
    """
    Charge Russell 1000 + Russell 2000 depuis Wikipedia (robuste sur 'Ticker'/'Symbol').
    Si échec, essaie les CSV locaux fallback (russell1000.csv / russell2000.csv).
    Renvoie DataFrame avec tv_symbol / yf_symbol.
    """
    tickers = []

    def add_from_url(url, label):
        try:
            lst = fetch_wikipedia_tickers(url)
            # Normalisation double (TV / YF)
            tickers.extend(lst)
        except Exception as e:
            print(f"[WARN] {label}: {e}")

    if INCLUDE_RUSSELL_1000:
        add_from_url("https://en.wikipedia.org/wiki/Russell_1000_Index", "Russell 1000")

    if INCLUDE_RUSSELL_2000:
        add_from_url("https://en.wikipedia.org/wiki/Russell_2000_Index", "Russell 2000")

    # Fallback CSV si nécessaire
    def add_from_csv(path, label):
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "Ticker" in df.columns:
                    vals = (
                        df["Ticker"].astype(str).str.strip()
                        .str.replace(r"\s+", "", regex=True)
                        .tolist()
                    )
                    if vals:
                        tickers.extend(vals)
                        print(f"[INFO] Fallback {label}: {len(vals)} tickers ajoutés depuis {path}")
        except Exception as e:
            print(f"[WARN] Fallback {label} CSV: {e}")

    if not tickers and INCLUDE_RUSSELL_1000:
        add_from_csv(R1K_FALLBACK_CSV, "R1K")
    if not tickers and INCLUDE_RUSSELL_2000:
        add_from_csv(R2K_FALLBACK_CSV, "R2K")

    if not tickers:
        raise RuntimeError("Impossible de charger l’univers (Wikipedia et CSV fallback indisponibles).")

    # Normalisation finale
    tv_syms = [tv_norm(s) for s in tickers]          # BRK.B pour TradingView
    yf_syms = [yf_norm(s) for s in tickers]          # BRK-B pour yfinance
    df = pd.DataFrame({"tv_symbol": tv_syms, "yf_symbol": yf_syms}).drop_duplicates().reset_index(drop=True)
    return df


def map_exchange_for_tv(yf_info_exch: str, ticker: str):
    """Map grossier pour TradingView_ta."""
    if not yf_info_exch:
        return "NASDAQ"
    e = yf_info_exch.upper()
    if "NASDAQ" in e:
        return "NASDAQ"
    if "NYSE" in e:
        return "NYSE"
    if "ARCA" in e:
        return "AMEX"
    return "NASDAQ"


def compute_local_technical_bucket(hist: pd.DataFrame):
    """Retourne (bucket, score, details) basé sur SMA20/50/200, RSI, MACD, Stoch."""
    if hist.empty or len(hist) < 60:
        return None, None, {}

    close = hist["Close"]; high = hist["High"]; low = hist["Low"]
    s20 = close.ta.sma(20); s50 = close.ta.sma(50); s200 = close.ta.sma(200)
    rsi = close.ta.rsi(14)
    macd = close.ta.macd(12, 26, 9)
    stoch = ta.stoch(high, low, close)

    if any(x is None or x.dropna().empty for x in (s20, s50, s200, rsi)) or macd is None or stoch is None:
        return None, None, {}

    price = close.iloc[-1]
    rsi_last = rsi.iloc[-1]
    macd_last = macd.iloc[-1]
    stoch_last = stoch.iloc[-1]

    votes = 0
    votes += 1 if price > s20.iloc[-1] else -1
    votes += 1 if s20.iloc[-1] > s50.iloc[-1] else -1
    votes += 1 if s50.iloc[-1] > s200.iloc[-1] else -1
    if rsi_last >= 55: votes += 1
    elif rsi_last <= 45: votes -= 1
    votes += 1 if macd_last["MACD_12_26_9"] > macd_last["MACDs_12_26_9"] else -1
    votes += 1 if stoch_last["STOCHk_14_3_3"] > stoch_last["STOCHd_14_3_3"] else -1

    if votes >= 4: bucket = "Strong Buy"
    elif votes >= 2: bucket = "Buy"
    elif votes <= -4: bucket = "Strong Sell"
    elif votes <= -2: bucket = "Sell"
    else: bucket = "Neutral"

    details = dict(
        price=float(price),
        sma20=float(s20.iloc[-1]), sma50=float(s50.iloc[-1]), sma200=float(s200.iloc[-1]),
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


def get_tv_summary(symbol: str, exchange: str):
    try:
        h = TA_Handler(
            symbol=symbol,
            screener="america",
            exchange=exchange,
            interval=TV_INTERVAL
        )
        s = h.get_analysis().summary
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
    print("Chargement des tickers Russell 1000 + 2000…")
    tickers_df = load_universe()
    print(f"Tickers dans l'univers: {len(tickers_df)}")

    rows = []
    for i, row in enumerate(tickers_df.itertuples(index=False), 1):
        tv_symbol = row.tv_symbol
        yf_symbol = row.yf_symbol

        try:
            tk = yf.Ticker(yf_symbol)
            try:
                info = tk.get_info() or {}
            except Exception:
                info = {}

            # Métadonnées nécessaires pour filtres
            sector = (info.get("sector") or "").strip()
            industry = (info.get("industry") or "").strip()
            country = (info.get("country") or info.get("countryOfCompany") or "").strip()
            mcap = info.get("marketCap")
            exch = info.get("exchange") or info.get("fullExchangeName") or ""
            tv_exchange = map_exchange_for_tv(exch, tv_symbol)

            # Filtres pays (on reste sur actions US)
            if country:
            if country.upper() not in {"USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA"}:
            continue


            # Filtres MarketCap
            if not isinstance(mcap, (int, float)) or mcap is None or mcap >= MAX_MARKET_CAP:
                continue

            # Données prix pour calcul TA local
            hist = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=False, actions=False)
            local_bucket, local_score, local_details = compute_local_technical_bucket(hist)
            if local_bucket is None:
                continue

            # TradingView recommendation
            tv = get_tv_summary(tv_symbol, tv_exchange)
            time.sleep(DELAY_BETWEEN_TV_CALLS_SEC)

            # Analystes Yahoo
            analyst_mean = info.get("recommendationMean")
            analyst_votes = info.get("numberOfAnalystOpinions")
            analyst_bucket = analyst_bucket_from_mean(analyst_mean)

            rows.append({
                "ticker_tv": tv_symbol,
                "ticker_yf": yf_symbol,
                "exchange_yf": exch,
                "exchange_tv": tv_exchange,
                "country": country,
                "sector": sector,
                "industry": industry,
                "market_cap": mcap,
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

        except Exception:
            # robustesse: on skip en silence
            continue

        if i % 50 == 0:
            print(f"{i}/{len(tickers_df)} traités…")

    df = pd.DataFrame(rows)
    if df.empty:
        print("Aucun titre après filtrages et collecte.")
        return

    # Filtres finaux: TradingView STRONG_BUY + Analystes Strong Buy
    mask_tv_strong = df["tv_reco"].isin({"STRONG_BUY"})
    mask_analyst_strong = df["analyst_bucket"].isin({"Strong Buy"})
    final_df = df[mask_tv_strong & mask_analyst_strong].copy()

    # Tri: score technique desc, analyst_votes desc, market_cap asc
    final_df.sort_values(["tech_score", "analyst_votes", "market_cap"],
                     ascending=[False, False, True], inplace=True)

    # Sauvegarde CSV
    final_df.to_csv(OUTPUT_CSV, index=False)


    # Aperçu console
    print("\n=== TV STRONG_BUY ∩ Analystes (Strong Buy) — US — <200B — Top 50 ===")
    cols_show = ["ticker_tv","ticker_yf","price","sector","industry","market_cap",
                 "technical_local","tech_score","tv_reco","analyst_bucket","analyst_mean","analyst_votes"]
    print(final_df[cols_show].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
