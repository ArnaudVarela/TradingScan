# build_universe.py
# Construit un univers unique à partir de plusieurs indices publics,
# enrichit avec market cap & liquidité (Yahoo), associe SECTOR (Wikipedia),
# filtre MCAP < 75B$ + secteur (IT/HC/Financials/Industrials),
# puis écrit raw_universe.csv, universe_in_scope.csv, universe_today.csv
# (en racine + dashboard/public/).

import io
import os
import re
import time
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import pandas as pd
import requests
import yfinance as yf

PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

MCAP_MAX_USD = 75_000_000_000  # < 75B$
MIN_ROW_LEN = 1

# Secteurs “autorisés” (on tolère quelques variantes fréquentes)
SECTORS_IN = {
    "Information Technology", "Technology",
    "Health Care", "Healthcare",
    "Financials",
    "Industrials",
}

UA = {"User-Agent": "Mozilla/5.0"}

def _save(df: pd.DataFrame, name: str):
    if df is None:
        df = pd.DataFrame()
    # racine
    df.to_csv(name, index=False)
    # public
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

def _norm_yf(t: str) -> str | None:
    if not isinstance(t, str): return None
    t = t.strip().upper()
    if not t or t in {"N/A", "-", "—"}: return None
    # BRK.B (TV/Wiki) → BRK-B (Yahoo)
    t = t.replace(".", "-")
    t = re.sub(r"[^A-Z0-9\-]", "", t)
    return t or None

def _yf_to_tv(t: str) -> str:
    # BRK-B (Yahoo) → BRK.B (TradingView)
    return (t or "").replace("-", ".")

def _union(*series: Iterable[str]) -> list[str]:
    seen = {}
    for s in series:
        for x in s:
            n = _norm_yf(x)
            if n: seen.setdefault(n, True)
    return list(seen.keys())

def _wiki_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    # Wrap dans StringIO pour éviter le FutureWarning
    return pd.read_html(io.StringIO(r.text))

def _extract_symbol_and_sector(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """
    Essaie de trouver les colonnes 'symbol/ticker' et 'sector' (ou GICS Sector).
    Retourne (liste_tickers, mapping{ticker->sector})
    """
    cols_lower = [str(c).lower() for c in df.columns]
    sym_idx = None
    sec_idx = None

    for i, c in enumerate(cols_lower):
        if "symbol" in c or "ticker" in c:
            sym_idx = i
            break
    for i, c in enumerate(cols_lower):
        if "sector" in c and "sub" not in c:
            sec_idx = i
            break

    syms = []
    sector_map = {}
    if sym_idx is None:
        return syms, sector_map

    sym_col = df.columns[sym_idx]
    syms_raw = df[sym_col].dropna().astype(str).tolist()

    if sec_idx is not None:
        sec_col = df.columns[sec_idx]
        for _, row in df.iterrows():
            sym = row.get(sym_col)
            sec = row.get(sec_col)
            yf_sym = _norm_yf(str(sym)) if pd.notna(sym) else None
            if yf_sym:
                syms.append(yf_sym)
                if pd.notna(sec):
                    sector_map[yf_sym] = str(sec).strip()
        return syms, sector_map

    # pas de secteur dans cette table
    for s in syms_raw:
        yf_sym = _norm_yf(s)
        if yf_sym:
            syms.append(yf_sym)
    return syms, sector_map

def get_sp500() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms:
            out_syms.extend(syms)
            out_map.update(smap)
            break
    print(f"[SRC] S&P500: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

def get_r1000() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms:
            out_syms.extend(syms)
            out_map.update(smap)
            break
    print(f"[SRC] R1000: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

def get_nasdaq100() -> Tuple[List[str], Dict[str,str]]:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    out_syms, out_map = [], {}
    for t in _wiki_tables(url):
        syms, smap = _extract_symbol_and_sector(t)
        if syms:
            out_syms.extend(syms)
            out_map.update(smap)
            break
    print(f"[SRC] NASDAQ-100: {len(out_syms)} tickers (with sector for {len(out_map)})")
    return out_syms, out_map

def get_r2000_from_csv(path="russell2000.csv") -> List[str]:
    if not os.path.exists(path):
        print("[SRC] R2000: missing russell2000.csv → skip")
        return []
    try:
        df = pd.read_csv(path)
        col = "Ticker" if "Ticker" in df.columns else df.columns[0]
        vals = [s for s in df[col].dropna().astype(str).tolist() if _norm_yf(s)]
        print(f"[SRC] R2000(csv): {len(vals)} tickers")
        return vals
    except Exception as e:
        print(f"[SRC] R2000(csv) error: {e} → skip")
        return []

def enrich_with_yf(tickers: list[str]) -> pd.DataFrame:
    # yfinance fast_info : market_cap, last_price, ten_day_average_volume
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            y = yf.Ticker(t)
            fi = getattr(y, "fast_info", None) or {}
            mcap = fi.get("market_cap")
            price = fi.get("last_price")
            vol10 = fi.get("ten_day_average_volume")
            avg_dollar = None
            if isinstance(price, (int,float)) and isinstance(vol10, (int,float)):
                avg_dollar = float(price)*float(vol10)
            rows.append({"ticker_yf": t, "mcap_usd": mcap, "avg_dollar_vol": avg_dollar})
        except Exception:
            rows.append({"ticker_yf": t, "mcap_usd": None, "avg_dollar_vol": None})
        if i % 100 == 0:
            print(f"[YF] {i}/{len(tickers)} enriched…")
            time.sleep(1.0)
    return pd.DataFrame(rows)

def main():
    # 1) Sources
    r1_syms, r1_sec = get_r1000()
    sp_syms, sp_sec = get_sp500()
    nd_syms, nd_sec = get_nasdaq100()
    r2_syms         = get_r2000_from_csv()  # pas de secteur

    # 2) Union (normalisée Yahoo)
    universe = _union(r1_syms, sp_syms, nd_syms, r2_syms)
    if len(universe) < MIN_ROW_LEN:
        raise SystemExit("❌ universe empty")
    print(f"[UNION] unique tickers before YF: {len(universe)}")

    # 3) Enrichissement Yahoo (cap/liquidité)
    meta = enrich_with_yf(universe)

    # 4) Ajout du sector via mapping Wikipedia (R1000/S&P500/NASDAQ-100)
    sector_map = {}
    sector_map.update(r1_sec)
    sector_map.update(sp_sec)
    sector_map.update(nd_sec)

    meta["ticker_tv"] = meta["ticker_yf"].map(_yf_to_tv)
    meta["sector"]    = meta["ticker_yf"].map(lambda t: sector_map.get(t, ""))

    # 5) Sauvegarde RAW (avant filtre)
    raw_cols = ["ticker_yf","ticker_tv","mcap_usd","avg_dollar_vol","sector"]
    raw = meta[raw_cols].copy()
    _save(raw, "raw_universe.csv")

    # 6) Filtre MCAP + Secteurs
    mcap_num = pd.to_numeric(raw["mcap_usd"], errors="coerce").fillna(0)
    sec_norm = raw["sector"].astype(str).str.strip()

    in_scope = raw[
        (mcap_num < MCAP_MAX_USD) &
        (sec_norm.isin(SECTORS_IN))
    ].copy()

    # tri simple (liquidité décroissante, puis cap croissant)
    in_scope["avg_dollar_vol"] = pd.to_numeric(in_scope["avg_dollar_vol"], errors="coerce")
    in_scope = in_scope.sort_values(["avg_dollar_vol","mcap_usd"], ascending=[False, True])

    # 7) Sauvegardes attendues par le workflow
    _save(in_scope, "universe_in_scope.csv")
    # compat historique (ton ancien front lisait universe_today.csv)
    _save(in_scope.rename(columns={"ticker_yf":"ticker_yf", "sector":"sector"}), "universe_today.csv")

    print(f"[FILTER] mcap < 75B & sector in {sorted(SECTORS_IN)}: {len(in_scope)}/{len(raw)}")
    print("[DONE] build_universe complete.")

if __name__ == "__main__":
    main()
