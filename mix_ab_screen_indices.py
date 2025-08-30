#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_ab_screen_indices_anticipative.py

Screener US (Russell 1000 + 2000) — version anticipative :
- Confirmed: double STRONG BUY (TradingView + analystes Yahoo)
- Anticipative: pré-signaux (MACD≈Signal, RSI~50, SMA20≈SMA50 ou just-cross) + volume (breakout/accumulation)
- Event-driven: proximité earnings (±7j) comme catalyseur

Sorties:
  - confirmed_STRONGBUY.csv
  - anticipative_pre_signals.csv
  - event_driven_signals.csv
  - candidates_all_ranked.csv (tous les signaux pour tri/QA)
"""

import os, re, time, math, warnings, datetime as dt
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import pandas_ta as ta
from tradingview_ta import TA_Handler, Interval

warnings.filterwarnings("ignore")

# =========================
#  CONFIG
# =========================
# Univers
INCLUDE_RUSSELL_1000 = True
INCLUDE_RUSSELL_2000 = True
R1K_FALLBACK_CSV = "russell1000.csv"   # 1 col "Ticker"
R2K_FALLBACK_CSV = "russell2000.csv"   # 1 col "Ticker"

# Filtres de base
MAX_MARKET_CAP = 200_000_000_000   # < 200B$
ALLOW_EMPTY_COUNTRY_IF_US_EXCH = True  # garde si exchange US quand country manquant

# Périodes & intervalles
PERIOD = "2y"       # pour indicateurs daily
INTERVAL = "1d"
TV_INTERVAL = Interval.INTERVAL_1_DAY

# Volume
VOL_MA_WINDOW = 20
VOL_BREAKOUT_MULT = 1.5        # vol_today > 1.5 * vol_ma20
VOL_ACCUM_MIN_STREAK = 3       # >ma20 pendant 3 jours = accumulation

# Pré-signaux (seuils “early”)
MACD_DELTA_MAX = 0.05          # |MACD - signal| < 0.05 ≈ croisement imminent
RSI_EARLY_MIN = 49.0           # franchissement zone 50
SMA_NEAR_PCT = 0.01            # |SMA20 - SMA50| / price < 1%
STOCH_NEUTRAL_LOW, STOCH_NEUTRAL_HIGH = 30, 70

# Calendrier (earnings)
EARNINGS_LOOKAHEAD_DAYS = 7

# TradingView (anti-rate-limit)
DELAY_BETWEEN_TV_CALLS_SEC = 0.15

# Fichiers de sortie
OUT_CONFIRMED = "confirmed_STRONGBUY.csv"
OUT_ANTICIP   = "anticipative_pre_signals.csv"
OUT_EVENTS    = "event_driven_signals.csv"
OUT_ALL       = "candidates_all_ranked.csv"


# =========================
#  HELPERS
# =========================
def yf_norm(sym: str) -> str:
    return sym.replace(".", "-")

def tv_norm(sym: str) -> str:
    return sym.replace("-", ".")

def fetch_wikipedia_tickers(url: str):
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"}
    r = requests.get(url, headers=ua, timeout=45)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    # essaie de trouver une colonne Ticker/Symbol
    for t in tables:
        cols = []
        for c in t.columns:
            cols.append((" ".join([str(x) for x in c if pd.notna(x)]) if isinstance(c, tuple) else str(c)).strip())
        low = [c.lower() for c in cols]
        pick = None
        for i, name in enumerate(low):
            if "ticker" in name or name == "symbol":
                pick = t.columns[i]; break
        if pick is None: 
            continue
        ser = (t[pick].astype(str).str.strip()
               .str.replace(r"\s+", "", regex=True)
               .str.replace("\u200b","", regex=False))
        ser = ser[ser.str.match(r"^[A-Za-z.\-]+$")].dropna().tolist()
        if ser:
            return ser
    raise RuntimeError("No ticker/symbol column on Wikipedia page")

def load_universe() -> pd.DataFrame:
    tickers = []
    def add_url(url, tag):
        try:
            tickers.extend(fetch_wikipedia_tickers(url))
        except Exception as e:
            print(f"[WARN] {tag}: {e}")
    if INCLUDE_RUSSELL_1000:
        add_url("https://en.wikipedia.org/wiki/Russell_1000_Index", "R1K")
    if INCLUDE_RUSSELL_2000:
        # Wikipedia R2K bouge souvent; on utilise CSV local si dispo
        if os.path.exists(R2K_FALLBACK_CSV):
            try:
                df = pd.read_csv(R2K_FALLBACK_CSV)
                if "Ticker" in df.columns:
                    tickers.extend(df["Ticker"].astype(str).str.strip().tolist())
                    print(f"[INFO] Fallback R2K CSV: {len(df)} tickers")
            except Exception as e:
                print(f"[WARN] R2K CSV: {e}")
        else:
            add_url("https://en.wikipedia.org/wiki/Russell_2000_Index", "R2K (wiki)")

    if INCLUDE_RUSSELL_1000 and not tickers and os.path.exists(R1K_FALLBACK_CSV):
        try:
            df = pd.read_csv(R1K_FALLBACK_CSV)
            if "Ticker" in df.columns:
                tickers.extend(df["Ticker"].astype(str).str.strip().tolist())
                print(f"[INFO] Fallback R1K CSV: {len(df)} tickers")
        except Exception as e:
            print(f"[WARN] R1K CSV: {e}")

    if not tickers:
        raise RuntimeError("Universe load failed (Wikipedia + CSV fallback).")

    tv_syms = [tv_norm(s) for s in tickers]
    yf_syms = [yf_norm(s) for s in tickers]
    return pd.DataFrame({"tv_symbol": tv_syms, "yf_symbol": yf_syms}).drop_duplicates().reset_index(drop=True)

def map_exchange_for_tv(yf_exch: str):
    if not yf_exch: return "NASDAQ"
    e = yf_exch.upper()
    if "NASDAQ" in e: return "NASDAQ"
    if "NYSE"   in e: return "NYSE"
    if "ARCA"   in e: return "AMEX"
    return "NASDAQ"

def last_valid(series_or_df):
    if series_or_df is None: return None
    if isinstance(series_or_df, pd.DataFrame):
        xx = series_or_df.dropna()
        return None if xx.empty else xx.iloc[-1]
    xx = series_or_df.dropna()
    return None if xx.empty else xx.iloc[-1]

def compute_indicators(hist: pd.DataFrame):
    if hist is None or hist.empty or len(hist) < 60:
        return None
    close = hist["Close"]; high = hist["High"]; low = hist["Low"]; vol = hist["Volume"]

    sma20 = close.ta.sma(20); sma50 = close.ta.sma(50); sma200 = close.ta.sma(200)
    rsi   = close.ta.rsi(14)
    macd  = close.ta.macd(12,26,9)       # DF: MACD_*, MACDs_* (ligne/signal)
    stoch = ta.stoch(high, low, close)   # DF: STOCHk_14_3_3, STOCHd_14_3_3
    obv   = ta.obv(close, vol)

    vol_ma = vol.rolling(VOL_MA_WINDOW).mean()
    ret = {
        "price": float(close.iloc[-1]),
        "sma20": float(last_valid(sma20)) if last_valid(sma20) is not None else np.nan,
        "sma50": float(last_valid(sma50)) if last_valid(sma50) is not None else np.nan,
        "sma200": float(last_valid(sma200)) if last_valid(sma200) is not None else np.nan,
        "rsi": float(last_valid(rsi)) if last_valid(rsi) is not None else np.nan,
        "macd": np.nan, "macds": np.nan,
        "stoch_k": np.nan, "stoch_d": np.nan,
        "vol": float(vol.iloc[-1]) if not math.isnan(vol.iloc[-1]) else np.nan,
        "vol_ma20": float(last_valid(vol_ma)) if last_valid(vol_ma) is not None else np.nan,
        "obv_slope10": np.nan
    }
    # MACD
    macd_row = last_valid(macd)
    if isinstance(macd_row, pd.Series):
        for k in ["MACD_12_26_9","MACD_12_26_9.0","MACD_12_26_9_MACD"]:
            if k in macd_row: ret["macd"] = float(macd_row[k]); break
        for k in ["MACDs_12_26_9","MACDs_12_26_9.0","MACD_12_26_9_SIGNAL"]:
            if k in macd_row: ret["macds"] = float(macd_row[k]); break
    # Stoch
    st_row = last_valid(stoch)
    if isinstance(st_row, pd.Series):
        for k in ["STOCHk_14_3_3","%K","STOCHk"]:
            if k in st_row: ret["stoch_k"] = float(st_row[k]); break
        for k in ["STOCHd_14_3_3","%D","STOCHd"]:
            if k in st_row: ret["stoch_d"] = float(st_row[k]); break
    # OBV slope 10 (approx)
    if isinstance(obv, pd.Series) and len(obv.dropna()) >= 11:
        obv10 = obv.dropna().iloc[-10:]
        ret["obv_slope10"] = float(obv10.iloc[-1] - obv10.iloc[0]) / 10.0
    return ret

def pre_signals(flags):
    """Renvoie (is_pre, votes, reasons) d’après indicateurs “early”."""
    price = flags["price"]; sma20=flags["sma20"]; sma50=flags["sma50"]; rsi=flags["rsi"]
    macd=flags["macd"]; macds=flags["macds"]; k=flags["stoch_k"]; d=flags["stoch_d"]

    votes = 0; reasons = []

    # MACD ≈ Signal
    if not np.isnan(macd) and not np.isnan(macds):
        if abs(macd - macds) <= MACD_DELTA_MAX:
            votes += 1; reasons.append("MACD≈Signal")

    # RSI franchit 50 (ou proche)
    if not np.isnan(rsi) and rsi >= RSI_EARLY_MIN:
        votes += 1; reasons.append("RSI≈50+")

    # SMA20 ≈ SMA50 (ou cross récent)
    if not np.isnan(sma20) and not np.isnan(sma50) and price > 0:
        near = abs(sma20 - sma50) / price <= SMA_NEAR_PCT
        crossed = sma20 > sma50
        if near or crossed:
            votes += 1; reasons.append("SMA20~SMA50/cross")

    # Stoch K croise D en zone neutre
    if not np.isnan(k) and not np.isnan(d):
        if STOCH_NEUTRAL_LOW <= k <= STOCH_NEUTRAL_HIGH and k > d:
            votes += 1; reasons.append("Stoch K>D neutre")

    return (votes >= 2), votes, ";".join(reasons)

def volume_features(flags, hist):
    """Retourne (vol_score, vol_breakout, accum_streak, obv_pos)"""
    vol = flags["vol"]; vol_ma = flags["vol_ma20"]; obv_slope = flags["obv_slope10"]
    vol_breakout = False; accum_streak = 0; obv_pos = False
    vol_score = 0

    if not np.isnan(vol) and not np.isnan(vol_ma) and vol_ma > 0:
        if vol > VOL_BREAKOUT_MULT * vol_ma:
            vol_breakout = True; vol_score += 2

        v = hist["Volume"]
        ma = v.rolling(VOL_MA_WINDOW).mean()
        streak = 0
        for vv, mm in zip(v.iloc[-10:], ma.iloc[-10:]):
            if not np.isnan(vv) and not np.isnan(mm) and vv > mm:
                streak += 1
        accum_streak = streak
        if streak >= VOL_ACCUM_MIN_STREAK:
            vol_score += 1

    if not np.isnan(obv_slope) and obv_slope > 0:
        obv_pos = True; vol_score += 1

    return vol_score, vol_breakout, accum_streak, obv_pos

def analyst_bucket_from_mean(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    if x < 1.6: return "Strong Buy"
    if x < 2.5: return "Buy"
    if x < 3.5: return "Hold"
    if x < 4.2: return "Sell"
    return "Strong Sell"

def get_tv_summary(symbol: str, exchange: str):
    def once(sym, ex):
        try:
            h = TA_Handler(symbol=sym, screener="america", exchange=ex, interval=TV_INTERVAL)
            s = h.get_analysis().summary
            return {"tv_reco": s.get("RECOMMENDATION"), "tv_buy": s.get("BUY"),
                    "tv_sell": s.get("SELL"), "tv_neutral": s.get("NEUTRAL")}
        except Exception:
            return None
    first = once(symbol, exchange)
    if first and first.get("tv_reco"): return first
    for ex in ("NASDAQ","NYSE","AMEX"):
        alt = once(symbol, ex)
        if alt and alt.get("tv_reco"): return alt
    return {"tv_reco": None, "tv_buy": None, "tv_sell": None, "tv_neutral": None}

def is_us_security(country: str, exch: str) -> bool:
    if country:
        if country.upper() in {"USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA"}:
            return True
        # si explicitement non-US, rejette (sauf AMEX/NYSE/NASDAQ fallback)
        if exch and str(exch).upper() in {"NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"} and ALLOW_EMPTY_COUNTRY_IF_US_EXCH:
            return True
        return False
    # pas de pays → on accepte si exchange US
    return str(exch).upper() in {"NASDAQ","NYSE","AMEX","BATS","NYSEARCA","NYSEMKT"} if exch else True

def get_earnings_date_safe(tk: yf.Ticker):
    try:
        cal = tk.get_calendar() if hasattr(tk, "get_calendar") else tk.calendar
        df = cal if isinstance(cal, pd.DataFrame) else None
        if df is None or df.empty: return None
        # colonnes possibles: 'Earnings Date' / 'Value'
        if "Earnings Date" in df.index:
            val = df.loc["Earnings Date"].values[0]
            return pd.to_datetime(val).date() if pd.notna(val) else None
        if "Value" in df.columns and "Earnings Date" in df.index:
            val = df.loc["Earnings Date","Value"]
            return pd.to_datetime(val).date() if pd.notna(val) else None
        # fallback: première date
        v = df.iloc[0,0]
        return pd.to_datetime(v).date() if pd.notna(v) else None
    except Exception:
        return None


# =========================
#  MAIN
# =========================
def main():
    today = dt.date.today()
    print("Chargement des tickers Russell 1000 + 2000…")
    universe = load_universe()
    print(f"Tickers dans l'univers: {len(universe)}")

    rows = []
    # stats
    c_total=c_us=c_mcap=c_hist=0

    for i, rec in enumerate(universe.itertuples(index=False), 1):
        tv_symbol = rec.tv_symbol
        yf_symbol = rec.yf_symbol
        c_total += 1

        try:
            tk = yf.Ticker(yf_symbol)

            # infos
            try:
                info = tk.get_info() or {}
            except Exception:
                info = {}
            fi = getattr(tk, "fast_info", None)
            exch = info.get("exchange") or info.get("fullExchangeName") or (fi.exchange if fi else "")
            mcap = (fi.market_cap if fi and getattr(fi, "market_cap", None) else info.get("marketCap"))
            country = (info.get("country") or info.get("countryOfCompany") or "").strip()
            sector = (info.get("sector") or "").strip()
            industry = (info.get("industry") or "").strip()

            if not is_us_security(country, exch):
                continue
            c_us += 1

            if isinstance(mcap, (int,float)) and mcap >= MAX_MARKET_CAP:
                continue
            c_mcap += 1

            # history
            time.sleep(0.03)
            hist = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=True, actions=False)
            if hist is None or hist.empty or len(hist) < 60:
                continue
            c_hist += 1

            flags = compute_indicators(hist)
            if not flags:
                continue

            # volume features
            vol_score, vol_breakout, accum_streak, obv_pos = volume_features(flags, hist)

            # pre-signals
            is_pre, pre_votes, pre_reasons = pre_signals(flags)

            # TV rec
            tv_exch = map_exchange_for_tv(exch)
            tv = get_tv_summary(tv_symbol, tv_exch)
            time.sleep(DELAY_BETWEEN_TV_CALLS_SEC)

            # analysts
            analyst_mean = info.get("recommendationMean")
            analyst_votes = info.get("numberOfAnalystOpinions")
            analyst_bucket = analyst_bucket_from_mean(analyst_mean)

            # earnings
            earn_date = get_earnings_date_safe(tk)
            days_to_earn = None
            if earn_date:
                days_to_earn = (earn_date - today).days

            rows.append({
                "ticker_tv": tv_symbol, "ticker_yf": yf_symbol,
                "exchange_yf": exch, "exchange_tv": tv_exch,
                "country": country, "sector": sector, "industry": industry,
                "market_cap": mcap, "price": flags["price"],
                "sma20": flags["sma20"], "sma50": flags["sma50"], "sma200": flags["sma200"],
                "rsi": flags["rsi"], "macd": flags["macd"], "macds": flags["macds"],
                "stoch_k": flags["stoch_k"], "stoch_d": flags["stoch_d"],
                "vol": flags["vol"], "vol_ma20": flags["vol_ma20"],
                "obv_slope10": flags["obv_slope10"],
                "vol_score": vol_score, "vol_breakout": vol_breakout,
                "vol_accum_streak10": accum_streak, "obv_pos": obv_pos,
                "pre_signal": is_pre, "pre_votes": pre_votes, "pre_reasons": pre_reasons,
                "tv_reco": tv["tv_reco"], "tv_buy": tv["tv_buy"], "tv_neutral": tv["tv_neutral"], "tv_sell": tv["tv_sell"],
                "analyst_mean": analyst_mean, "analyst_votes": analyst_votes, "analyst_bucket": analyst_bucket,
                "earnings_date": earn_date, "days_to_earnings": days_to_earn
            })

        except Exception:
            continue

        if i % 50 == 0:
            print(f"{i}/{len(universe)} traités…")

    print(f"[STEP] total vus: {c_total} | US: {c_us} | mcap<200B: {c_mcap} | history OK: {c_hist}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("Aucun titre collecté (df vide). Rien à exporter.")
        # on écrit des CSV vides pour éviter les commits inutiles
        for p in (OUT_CONFIRMED, OUT_ANTICIP, OUT_EVENTS, OUT_ALL):
            pd.DataFrame().to_csv(p, index=False)
        return

    # ==== Buckets ====
    mask_tv_strong      = df["tv_reco"].isin({"STRONG_BUY"})
    mask_analyst_strong = df["analyst_bucket"].isin({"Strong Buy"})
    confirmed = df[mask_tv_strong & mask_analyst_strong].copy()

    # anticipative: pre_signals ET volume (score >= 2 ou breakout) — même si pas STRONG_BUY encore
    mask_pre = df["pre_signal"] == True
    mask_vol = (df["vol_score"] >= 2) | (df["vol_breakout"] == True)
    anticip  = df[mask_pre & mask_vol].copy()

    # event-driven: earnings dans ±7 jours
    near_earn = df["days_to_earnings"].notna() & (df["days_to_earnings"].abs() <= EARNINGS_LOOKAHEAD_DAYS)
    events = df[near_earn].copy()

    # ==== Scores de tri ====
    def rank_score(s):
        # score simple pour classer (plus haut = mieux)
        score = 0.0
        # confirmed fort
        if s.get("tv_reco") == "STRONG_BUY": score += 3
        if s.get("analyst_bucket") == "Strong Buy": score += 2
        score += float(s.get("pre_votes") or 0) * 0.7
        score += float(s.get("vol_score") or 0) * 0.6
        # plus de votes analystes, mieux
        av = s.get("analyst_votes"); 
        if isinstance(av, (int,float)) and av > 0: score += min(av, 20) * 0.05
        # market cap plus petit → plus explosif
        mc = s.get("market_cap")
        if isinstance(mc, (int,float)) and mc>0:
            score += max(0.0, 5.0 - math.log10(mc))   # favor small/mid
        # earnings proche: petit bonus
        dte = s.get("days_to_earnings")
        if isinstance(dte, (int,float)) and abs(dte) <= EARNINGS_LOOKAHEAD_DAYS:
            score += 0.5
        return score

    for frame in (df, confirmed, anticip, events):
        frame["rank_score"] = frame.apply(rank_score, axis=1)

    # tri & export
    confirmed.sort_values(["rank_score","vol_score","analyst_votes","market_cap"], ascending=[False,False,False,True], inplace=True)
    anticip.sort_values(["rank_score","vol_score","pre_votes","market_cap"], ascending=[False,False,False,True], inplace=True)
    events.sort_values(["rank_score","days_to_earnings","market_cap"], ascending=[False,True,True], inplace=True)

    # candidates all
    all_candidates = pd.concat([
        confirmed.assign(candidate_type="confirmed"),
        anticip.assign(candidate_type="anticipative"),
        events.assign(candidate_type="event")
    ], ignore_index=True).sort_values(["rank_score","candidate_type"], ascending=[False,True])

    # colonnes utiles
    base_cols = ["ticker_tv","ticker_yf","price","sector","industry","market_cap",
                 "tv_reco","analyst_bucket","analyst_mean","analyst_votes",
                 "pre_signal","pre_votes","pre_reasons",
                 "vol_score","vol_breakout","vol_accum_streak10","obv_pos",
                 "rsi","sma20","sma50","sma200","macd","macds","stoch_k","stoch_d",
                 "earnings_date","days_to_earnings","rank_score"]

    confirmed[base_cols].to_csv(OUT_CONFIRMED, index=False)
    anticip[base_cols].to_csv(OUT_ANTICIP, index=False)
    events[base_cols].to_csv(OUT_EVENTS, index=False)
    all_candidates[["candidate_type"] + base_cols].to_csv(OUT_ALL, index=False)

    print(f"Exportés: {OUT_CONFIRMED} ({len(confirmed)}), {OUT_ANTICIP} ({len(anticip)}), {OUT_EVENTS} ({len(events)}), {OUT_ALL} ({len(all_candidates)})")


if __name__ == "__main__":
    main()
