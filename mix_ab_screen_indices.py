# mix_ab_screen_indices.py
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
from collections import Counter

warnings.filterwarnings("ignore")

# =========================
#  DEBUG / FLAGS
# =========================
DEBUG_TA = True          # mettre False quand tout est bon
TA_PROBE_LIMIT = 15      # nombre max de lignes [TA_FAIL] imprimées
TA_FAIL_STATS = Counter()
TA_PROBE_LEFT = TA_PROBE_LIMIT

# =========================
#  CONFIG
# =========================
INCLUDE_RUSSELL_1000 = True
INCLUDE_RUSSELL_2000 = True

# Fallback CSV (optionnel) si Wikipedia change trop souvent.
R1K_FALLBACK_CSV = "russell1000.csv"   # entête attendu: Ticker
R2K_FALLBACK_CSV = "russell2000.csv"   # entête attendu: Ticker

# Indicateurs locaux (période/interval Yahoo)
PERIOD = "2y"     # important pour SMA200 et TA stables
INTERVAL = "1d"
TV_INTERVAL = Interval.INTERVAL_1_DAY

# Filtre Market Cap
MAX_MARKET_CAP = 200_000_000_000  # < 200B$

# Fichiers de sortie
OUTPUT_CSV = "tv_STRONGBUY__analyst_STRONGBUY__under200B.csv"

# Respect TradingView (lib non-officielle) : ne pas spammer
DELAY_BETWEEN_TV_CALLS_SEC = 0.2


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
    Télécharge la page Wikipedia et tente d’extraire une colonne 'Ticker' OU 'Symbol'
    depuis n’importe quel tableau (gère MultiIndex d'entêtes).
    Renvoie la liste de tickers trouvés (strings).
    """
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"}
    resp = requests.get(url, headers=ua, timeout=45)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)

    def flatten_cols(cols):
        out = []
        for c in cols:
            if isinstance(c, tuple):
                name = " ".join([str(x) for x in c if pd.notna(x)]).strip()
            else:
                name = str(c).strip()
            out.append(name)
        return out

    candidates = ("ticker", "symbol")
    for t in tables:
        cols = flatten_cols(t.columns)
        lower = [c.lower() for c in cols]
        col_idx = None
        for i, name in enumerate(lower):
            if any(key in name for key in candidates):
                col_idx = i
                break
        if col_idx is None:
            continue
        col_name = t.columns[col_idx]
        ser = (
            t[col_name]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
            .str.replace("\u200b", "", regex=False)
        )
        ser = ser[ser.str.match(r"^[A-Za-z.\-]+$")].dropna().tolist()
        if len(ser) > 0:
            return ser

    raise RuntimeError(f"Aucune colonne 'Ticker' ou 'Symbol' trouvée sur {url}")


def load_universe():
    tickers = []

    def add_from_url(url, label):
        try:
            lst = fetch_wikipedia_tickers(url)
            tickers.extend(lst)
        except Exception as e:
            print(f"[WARN] {label}: {e}")

    if INCLUDE_RUSSELL_1000:
        add_from_url("https://en.wikipedia.org/wiki/Russell_1000_Index", "Russell 1000")

    if INCLUDE_RUSSELL_2000:
        add_from_url("https://en.wikipedia.org/wiki/Russell_2000_Index", "Russell 2000")

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

    tv_syms = [tv_norm(s) for s in tickers]
    yf_syms = [yf_norm(s) for s in tickers]
    df = pd.DataFrame({"tv_symbol": tv_syms, "yf_symbol": yf_syms}).drop_duplicates().reset_index(drop=True)
    return df


def map_exchange_for_tv(yf_info_exch: str, ticker: str):
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

def compute_local_technical_bucket(hist: pd.DataFrame, symbol: str = None):
    """
    Renvoie (bucket, score, details) en étant tolérant:
    - Essaie d'abord pandas_ta (API fonctionnelle)
    - Si ça échoue, fallback en pur pandas/numpy
    Exige: Close + SMA20 + SMA50 + RSI (SMA200/MACD/Stoch optionnels)
    """
    global TA_FAIL_STATS, TA_PROBE_LEFT

    def fail(reason: str, extra: dict = None):
        TA_FAIL_STATS[reason] += 1
        if DEBUG_TA and TA_PROBE_LEFT > 0:
            TA_PROBE_LEFT -= 1
            msg = f"[TA_FAIL] {reason}"
            if symbol:
                msg += f" sym={symbol}"
            if extra:
                # on affiche tout le dict extra (inclut 'err' si présent)
                msg += " | " + ", ".join(f"{k}={v}" for k, v in extra.items())
            print(msg)
        return None, None, {}

    if hist is None or hist.empty:
        return fail("hist_empty")
    if len(hist) < 60:
        return fail("hist_too_short", {"len": len(hist)})

    for col in ("Close", "High", "Low"):
        if col not in hist.columns:
            return fail("missing_col", {"col": col})

    close = hist["Close"].astype(float)
    high  = hist["High"].astype(float)
    low   = hist["Low"].astype(float)

    # utilitaires
    def last_valid(s):
        if s is None: return None
        s2 = s.dropna()
        return None if s2.empty else s2.iloc[-1]

    # ---------- 1) Tentative avec pandas_ta (API fonctionnelle) ----------
    try:
        s20  = ta.sma(close, length=20)
        s50  = ta.sma(close, length=50)
        s200 = ta.sma(close, length=200)
        rsi  = ta.rsi(close, length=14)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)  # colonnes: MACD_12_26_9, MACDs_..., MACDh_...
        stoch_df = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)  # STOCHk_14_3_3, STOCHd_14_3_3

        s20_last   = last_valid(s20)
        s50_last   = last_valid(s50)
        s200_last  = last_valid(s200)
        rsi_last   = last_valid(rsi)
        macd_row   = last_valid(macd_df)
        stoch_row  = last_valid(stoch_df)

        if any(v is None for v in (last_valid(close), s20_last, s50_last, rsi_last)):
            raise RuntimeError("pta_missing_core")

        macd_val = macd_sig = None
        if isinstance(macd_row, pd.Series):
            macd_val = next((float(macd_row[k]) for k in macd_row.index if "MACD_" in k and not "s_" in k and not "h_" in k), None)
            macd_sig = next((float(macd_row[k]) for k in macd_row.index if "MACDs_" in k), None)

        stoch_k = stoch_d = None
        if isinstance(stoch_row, pd.Series):
            stoch_k = next((float(stoch_row[k]) for k in stoch_row.index if k.startswith("STOCHk")), None)
            stoch_d = next((float(stoch_row[k]) for k in stoch_row.index if k.startswith("STOCHd")), None)

    except Exception as e:
        # ---------- 2) Fallback sans pandas_ta ----------
        try:
            # SMA simples
            s20  = close.rolling(20).mean()
            s50  = close.rolling(50).mean()
            s200 = close.rolling(200).mean()

            # RSI 14 classique
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            roll_up = gain.rolling(14).mean()
            roll_down = loss.rolling(14).mean()
            rs = roll_up / roll_down
            rsi = 100 - (100 / (1 + rs))

            # MACD (EMA 12/26 + signal 9)
            def ema(series, span):
                return series.ewm(span=span, adjust=False).mean()
            macd_line = ema(close, 12) - ema(close, 26)
            macd_signal = ema(macd_line, 9)

            # Stoch 14,3,3
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            stoch_k_series = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_k_series = stoch_k_series.rolling(3).mean()
            stoch_d_series = stoch_k_series.rolling(3).mean()

            s20_last   = last_valid(s20)
            s50_last   = last_valid(s50)
            s200_last  = last_valid(s200)
            rsi_last   = last_valid(rsi)
            macd_val   = float(last_valid(macd_line))   if last_valid(macd_line)   is not None else None
            macd_sig   = float(last_valid(macd_signal)) if last_valid(macd_signal) is not None else None
            stoch_k    = float(last_valid(stoch_k_series)) if last_valid(stoch_k_series) is not None else None
            stoch_d    = float(last_valid(stoch_d_series)) if last_valid(stoch_d_series) is not None else None

            if any(v is None for v in (last_valid(close), s20_last, s50_last, rsi_last)):
                return fail("fallback_missing_core", {"err": type(e).__name__})
        except Exception as e2:
            return fail("fallback_exception", {"err": type(e).__name__, "err_fb": type(e2).__name__})

    # ---------- Votes ----------
    price_last = float(last_valid(close))
    votes = 0
    try:
        votes += 1 if float(price_last) > float(s20_last) else -1
        votes += 1 if float(s20_last) > float(s50_last) else -1
        if s200_last is not None and not (isinstance(s200_last, float) and np.isnan(s200_last)):
            votes += 1 if float(s50_last) > float(s200_last) else -1

        r = float(rsi_last)
        if r >= 55: votes += 1
        elif r <= 45: votes -= 1

        if macd_val is not None and macd_sig is not None:
            votes += 1 if macd_val > macd_sig else -1

        if 'stoch_k' in locals() and 'stoch_d' in locals():
            if (stoch_k is not None) and (stoch_d is not None):
                votes += 1 if stoch_k > stoch_d else -1
    except Exception as ev:
        return fail("vote_exception", {"err": type(ev).__name__})

    # Bucket
    if votes >= 4: bucket = "Strong Buy"
    elif votes >= 2: bucket = "Buy"
    elif votes <= -4: bucket = "Strong Sell"
    elif votes <= -2: bucket = "Sell"
    else: bucket = "Neutral"

    details = dict(
        price=price_last,
        sma20=float(s20_last),
        sma50=float(s50_last),
        sma200=float(s200_last) if s200_last is not None and not (isinstance(s200_last, float) and np.isnan(s200_last)) else float("nan"),
        rsi=float(rsi_last),
        macd=macd_val if macd_val is not None else float("nan"),
        macds=macd_sig if macd_sig is not None else float("nan"),
        stoch_k=stoch_k if 'stoch_k' in locals() and stoch_k is not None else float("nan"),
        stoch_d=stoch_d if 'stoch_d' in locals() and stoch_d is not None else float("nan"),
    )
    return bucket, int(votes), details

def analyst_bucket_from_mean(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
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
            return {
                "tv_reco": s.get("RECOMMENDATION"),
                "tv_buy": s.get("BUY"),
                "tv_sell": s.get("SELL"),
                "tv_neutral": s.get("NEUTRAL"),
            }
        except Exception:
            return None

    # 1er essai sur l’exchange déduit
    first = try_once(symbol, exchange)
    if first and first.get("tv_reco"):
        return first

    # Fallback: tente d’autres bourses US courantes
    for ex in ("NASDAQ", "NYSE", "AMEX"):
        alt = try_once(symbol, ex)
        if alt and alt.get("tv_reco"):
            return alt

    # Rien trouvé
    return {"tv_reco": None, "tv_buy": None, "tv_sell": None, "tv_neutral": None}


# =========================
#  MAIN
# =========================
def main():
    print("Chargement des tickers Russell 1000 + 2000…")
    tickers_df = load_universe()
    print(f"Tickers dans l'univers: {len(tickers_df)}")

    rows = []

    # Compteurs de diagnostic
    c_total = c_us = c_mcap = c_hist = c_ta = 0

    for i, row in enumerate(tickers_df.itertuples(index=False), 1):
        tv_symbol = row.tv_symbol
        yf_symbol = row.yf_symbol
        c_total += 1

        try:
            tk = yf.Ticker(yf_symbol)

            # Info lente (parfois vide)
            try:
                info = tk.get_info() or {}
            except Exception:
                info = {}

            # Info rapide (fiable) + market cap robuste
            fi = getattr(tk, "fast_info", None)
            mcap = None
            exch = info.get("exchange") or info.get("fullExchangeName") or ""
            if fi:
                mcap = getattr(fi, "market_cap", None) or mcap
                exch = getattr(fi, "exchange", exch) or exch
            if mcap is None:
                mcap = info.get("marketCap")  # fallback si fast_info vide

            sector = (info.get("sector") or "").strip()
            industry = (info.get("industry") or "").strip()
            country = (info.get("country") or info.get("countryOfCompany") or "").strip()

            # Détection US tolérante : si country vide mais exchange US, on garde
            is_us_exchange = str(exch).upper() in {"NASDAQ", "NYSE", "AMEX", "BATS", "NYSEARCA", "NYSEMKT"}
            if country:
                if country.upper() not in {"USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA"} and not is_us_exchange:
                    continue
            # Si country vide, on laisse passer; contrôle via exchange ci-dessus
            c_us += 1

            # Filtre MarketCap — n'exclut que si on connaît la cap ET qu'elle dépasse le seuil
            if isinstance(mcap, (int, float)) and mcap >= MAX_MARKET_CAP:
                continue
            c_mcap += 1

            tv_exchange = map_exchange_for_tv(exch, tv_symbol)

            # Petit délai pour éviter rate-limit Yahoo (avant history)
            time.sleep(0.05)
            hist = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=True, actions=False)
            if hist.empty or len(hist) < 60:
                continue
            c_hist += 1

            # TA locale (SMA200/MACD/Stoch optionnels)
            local_bucket, local_score, local_details = compute_local_technical_bucket(hist, symbol=yf_symbol)
            if local_bucket is None:
                continue
            c_ta += 1

            # TradingView recommendation (avec fallback d'exchange)
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
            continue

        if i % 50 == 0:
            print(f"{i}/{len(tickers_df)} traités…")

    # ===== fin de boucle =====
    df = pd.DataFrame(rows)

    print(f"[STEP] total tickers vus       : {c_total}")
    print(f"[STEP] après filtre pays (US)  : {c_us}")
    print(f"[STEP] après filtre MCAP       : {c_mcap}")
    print(f"[STEP] historiques OK (>=60)   : {c_hist}")
    print(f"[STEP] TA locale OK            : {c_ta}")
    if DEBUG_TA:
        top_fail = TA_FAIL_STATS.most_common(8)
        print("[TA_FAIL_STATS]", dict(top_fail))

    # --- Diagnostics / Debug même si vide ---
    if df.empty:
        print("Aucun titre après filtrages et collecte (df vide). Écriture de CSV vides pour debug.")
        empty_cols = ["ticker_tv","ticker_yf","price","sector","industry","market_cap",
                      "technical_local","tech_score","tv_reco","analyst_bucket","analyst_mean","analyst_votes"]
        pd.DataFrame(columns=empty_cols).to_csv("debug_tv_STRONGBUY.csv", index=False)
        pd.DataFrame(columns=empty_cols).to_csv("debug_analyst_STRONGBUY.csv", index=False)
        pd.DataFrame(columns=empty_cols).to_csv(OUTPUT_CSV, index=False)
        return

    # 0) combien de lignes ont des données analystes ?
    have_analyst = df["analyst_bucket"].notna().sum()
    print(f"[DEBUG] Titres avec note analystes dispo: {have_analyst}/{len(df)}")

    # 1) Comptes par étape
    mask_tv_strong = df["tv_reco"].isin({"STRONG_BUY"})
    mask_analyst_strong = df["analyst_bucket"].isin({"Strong Buy"})

    tv_only = df[mask_tv_strong].copy()
    analyst_only = df[mask_analyst_strong].copy()
    intersection = df[mask_tv_strong & mask_analyst_strong].copy()

    print(f"[DEBUG] TV=STRONG_BUY : {len(tv_only)}")
    print(f"[DEBUG] Analystes=Strong Buy : {len(analyst_only)}")
    print(f"[DEBUG] Intersection (double Strong Buy) : {len(intersection)}")

    # 2) Sauvegardes debug
    tv_only.sort_values(["tech_score","analyst_votes","market_cap"], ascending=[False, False, True], inplace=True)
    analyst_only.sort_values(["analyst_votes","market_cap"], ascending=[False, True], inplace=True)
    intersection.sort_values(["tech_score","analyst_votes","market_cap"], ascending=[False, False, True], inplace=True)

    tv_only.to_csv("debug_tv_STRONGBUY.csv", index=False)
    analyst_only.to_csv("debug_analyst_STRONGBUY.csv", index=False)

    # 3) Export final (intersection)
    intersection.to_csv(OUTPUT_CSV, index=False)

    # 4) Aperçu console
    print("\n=== INTERSECTION — TV STRONG_BUY ∩ Analystes Strong Buy — US — <200B — Top 50 ===")
    cols_show = ["ticker_tv","ticker_yf","price","sector","industry","market_cap",
                 "technical_local","tech_score","tv_reco","analyst_bucket","analyst_mean","analyst_votes"]
    print(intersection[cols_show].head(50).to_string(index=False))

    if intersection.empty:
        print("⚠️ Intersection vide : ouvre les fichiers 'debug_tv_STRONGBUY.csv' et 'debug_analyst_STRONGBUY.csv' pour voir où ça coince.")


if __name__ == "__main__":
    main()
