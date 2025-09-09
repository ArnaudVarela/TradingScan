#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair candidates CSVs: fill price (robust) and market cap.
- Reads dashboard/public/candidates_all_ranked.csv if present
- Also tries confirmed_STRONGBUY.csv, anticipative_pre_signals.csv, event_driven_signals.csv if present
- Writes files in place

Safe to run multiple times.
Requires: pandas, yfinance
"""

import os
import sys
import math
import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

PUBLIC_DIR = Path(".")

# -------------------------- yfinance helpers -------------------------- #

def _get_fast_info_price(ticker: str) -> Optional[float]:
    """Try fast_info price with snake_case/camelCase compatibility."""
    try:
        fi = getattr(yf.Ticker(ticker), "fast_info", None) or {}
        p = fi.get("last_price", None)
        if p is None:
            p = fi.get("lastPrice", None)
        if p is None:
            return None
        return float(p)
    except Exception:
        return None

def _batch_history_close_last(tickers: List[str]) -> Dict[str, float]:
    """
    Batch fallback: last close from recent history.
    Returns mapping {ticker: last_close}.
    """
    out: Dict[str, float] = {}
    if not tickers:
        return out

    # yfinance supports comma-separated tickers
    try:
        df = yf.download(
            tickers=" ".join(tickers),
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return out

    # MultiIndex (per-ticker) or single
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                sub = df[t]["Close"].dropna()
                if len(sub):
                    out[t] = float(sub.iloc[-1])
            except Exception:
                pass
    else:
        # Single ticker edge-case
        try:
            sub = df["Close"].dropna()
            if len(sub):
                out[tickers[0]] = float(sub.iloc[-1])
        except Exception:
            pass

    return out

def _fill_prices(df: pd.DataFrame, yf_col_candidates=("ticker_yf","ticker","symbol")) -> pd.Series:
    """
    Returns a price Series aligned with df.index.
    - Tries df['price'] if already present and >0.
    - Else tries fast_info
    - Else batch history fallback
    """
    # Determine which column contains the Yahoo symbol
    ycol = None
    for c in yf_col_candidates:
        if c in df.columns:
            ycol = c
            break
    if ycol is None:
        # try to infer from index-like columns
        raise RuntimeError("No ticker column found (expected one of: %s)" % (yf_col_candidates,))

    prices = pd.to_numeric(df.get("price"), errors="coerce") if "price" in df.columns else pd.Series(index=df.index, dtype="float64")
    # Anything already valid (>0) we keep
    need_mask = (prices.isna()) | (prices <= 0)
    if not need_mask.any():
        return prices

    tickers_need = df.loc[need_mask, ycol].astype(str).str.strip().tolist()

    # 1) Try fast_info one by one (fast, low quota impact)
    fast_hits: Dict[str, float] = {}
    for t in tickers_need:
        p = _get_fast_info_price(t)
        if p is not None and p > 0:
            fast_hits[t] = p

    prices.loc[need_mask] = df.loc[need_mask, ycol].map(fast_hits)

    # 2) Remaining → batch history fallback
    still_need = prices.isna() | (prices <= 0)
    if still_need.any():
        miss = df.loc[still_need, ycol].astype(str).str.strip().unique().tolist()
        hist_map = _batch_history_close_last(miss)
        prices.loc[still_need] = df.loc[still_need, ycol].map(hist_map)

    # Normalize: if something is still NaN, leave it NaN (UI can show "—")
    return prices

def _coalesce_mcap(df: pd.DataFrame, price_col="price") -> pd.Series:
    """
    Build a robust mcap series:
    - Prefer existing explicit market cap columns if present (Finnhub/YF info)
    - Else compute from shares_outstanding * price when available
    - Never fill with 0 by default (leave NaN if unknown)
    """
    candidates = []
    for col in ["mcap_usd_final", "mcap_usd_finnhub", "mcap_usd_yfinfo", "market_cap", "mcap"]:
        if col in df.columns:
            candidates.append(pd.to_numeric(df[col], errors="coerce"))

    mcap = None
    if candidates:
        mcap = candidates[0].copy()
        for c in candidates[1:]:
            mcap = mcap.fillna(c)
    else:
        mcap = pd.Series(index=df.index, dtype="float64")

    # Try compute from shares_outstanding if available
    so = None
    for col in ["shares_outstanding", "sharesOutstanding", "float_shares", "free_float"]:
        if col in df.columns:
            so = pd.to_numeric(df[col], errors="coerce")
            break

    price = pd.to_numeric(df.get(price_col), errors="coerce")
    if so is not None and price is not None:
        calc = so * price
        mcap = mcap.fillna(calc)

    # Do NOT fillna(0) -> keep NaN so UI can render "—"
    return mcap

# -------------------------- main patch routine -------------------------- #

TARGET_FILES = [
    "candidates_all_ranked.csv",
    "confirmed_STRONGBUY.csv",
    "anticipative_pre_signals.csv",
    "event_driven_signals.csv",
]

def patch_file(path: str) -> bool:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return False

    # Price
    prices = _fill_prices(df)
    df["price"] = prices

    # MCap
    df["mcap_usd_final"] = _coalesce_mcap(df, price_col="price")

    # Optional: round for neat display (won't hurt ranking)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").round(4)
    if "mcap_usd_final" in df.columns:
        df["mcap_usd_final"] = pd.to_numeric(df["mcap_usd_final"], errors="coerce").round(0)

    df.to_csv(path, index=False)
    return True


def main():
    base = PUBLIC_DIR
    if not os.path.isdir(base):
        print(f"[repair] directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    any_patched = False
    for fname in TARGET_FILES:
        fpath = os.path.join(base, fname)
        if os.path.isfile(fpath):
            ok = patch_file(fpath)
            if ok:
                any_patched = True
                print(f"[repair] patched {fname}")
        else:
            # not an error; the file may not exist for all buckets
            pass

    if not any_patched:
        print("[repair] nothing to patch (no target CSVs found).")
    else:
        print("[repair] done.")

if __name__ == "__main__":
    main()
