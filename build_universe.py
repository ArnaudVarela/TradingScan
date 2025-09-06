# build_universe.py
# Construit un univers à partir de R1000 / S&P500 / NASDAQ-100 + NASDAQ Composite,
# enrichit avec Yahoo (mcap/liquidité), complète sector/industry sur un sous-ensemble,
# filtre final: MCAP < 75B & sector ∈ {Tech, Financials, Industrials, Health Care},
# écrit raw_universe.csv, universe_in_scope.csv, universe_today.csv (root + dashboard/public/).

import io
import os
import re
import time
from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

# ------------ CONFIG ------------
PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

MCAP_MAX_USD = 75_000_000_000  # < 75B$
MIN_ROW_LEN = 1

# Inclure NASDAQ Composite (grand univers NASDAQ)
USE_NASDAQ_COMPOSITE = True

# Secteurs acceptés (tolérance sur libellés)
ALLOWED_SECTORS = {
    # Tech
    "information technology", "technology", "tech", "information technology services",
    # Financials
    "financials", "financial", "financial services",
    # Industrials
    "industrials", "industrial",
    # Health Care
    "health care", "healthcare", "health services"
}

# Enrichissement du secteur pour un SOUS-ENSEMBLE (limiter la durée des runs)
TOP_LIQ_FOR_SECTOR_ENRICH = int(os.getenv("TOP_LIQ_FOR_SECTOR_ENRICH", "180"))  # top N par liquidité < 75B
YF_INFO_SLEEP = 0.15  # secondes entre appels .info
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
AV_SLEEP = 12.5       # 5 req/min pour la free tier -> ~12s
MAX_AV_LOOKUPS = int(os.getenv("MAX_AV_LOOKUPS", "60"))  # ~12 min max si à plein régime

UA = {"User-Agent": "Mozilla/5.0"}

# ------------ IO helpers ------------
def _save(df: pd.DataFrame, name: str):
    if df is None:
        df = pd.DataFrame()
    df.to_csv(name, index=False)
    (PUBLIC_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLIC_DIR / name, index=False)
    print(f"[OK] wrote {name}: {len(df)} rows")

# ------------ symbol helpers ------------
def _norm_yf(t: str) -> Optional[str]:
    if not isinstance(t, str):
        return None
    t = t.strip().upper()
    if not t or t in {"N/A", "-", "—"}:
        return None
    # Yahoo: BRK.B -> BRK-B
    t = t.replace(".", "-")
    # Nettoie hors [A-Z0-9-]
    t = re.sub(r"[^A-Z0-9\-]", "", t)
    return t or None

def _yf_to_tv(t: str) -> str:
    return (t or "").replace("-", ".")

def _union(*series: Iterable[str]) -> list[str]:
    seen = {}
    for s in series:
        for x in s:
            n = _norm_yf(x)
            if n:
                seen.setdefault(n, True)
    return list(seen.keys())

# ------------ HTTP helpers ------------
def _get(url: str, **kwargs) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=UA, timeout=45, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"[WARN] GET fail {url}: {e}")
        return None

# ------------ scraping wikipedia ------------
def _wiki_tables(url: str) -> list[pd.DataFrame]:
    # petit retry
    for attempt in range(3):
        r = _get(url)
        if r and r.text:
            try:
                return pd.read_html(io.StringIO(r.text))
            except Exception as e:
                print(f"[WARN] read_html fail ({attempt+1}/3): {e}")
        time.sleep(1.2 * (attempt + 1))
    return []

def _extract_symbol_and_sector(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    cols_lower = [str(c).lower() for c in df.columns]
    sym_idx, sec_idx = None, None
    for i, c in enumerate(cols_lower):
        if "symbol" in c or "ticker" in c:
            sym_idx = i
            break
    for i, c in enumerate(cols_lower):
        if "sector" in c and "sub" not in c:
            sec_idx = i
            break

    syms, sector_map = [], {}
    if sym_idx is None:
        return syms, sector_map

    sym_col = df.columns[sym_idx]
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

    for s in df[sym_col].dropna().astype(str).tolist():
        yf_sym = _norm_yf(s)
        if yf_sym:
            syms.append(yf_sym)
    return syms, sector_map

def get_sp500() -> Tuple[List[str], Dict[str, str]]:
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

def get_r1000() -> Tuple[List[str], Dict[str, str]]:
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

def get_nasdaq100() -> Tuple[List[str], Dict[str, str]]:
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

# ------------ NASDAQ Composite (NasdaqTrader) ------------
def _parse_pipe_file(text: str) -> list[dict]:
    """Parse un fichier pipe-delimited (NasdaqTrader .txt)."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not lines:
        return []
    header = [c.strip() for c in lines[0].split("|")]
    rows = []
    for ln in lines[1:]:
        # Ignore la ligne "File Creation Time: ..."
        if ln.lower().startswith("file creation time"):
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows

def get_nasdaq_composite() -> list[str]:
    """
    Liste des tickers NASDAQ à partir de NasdaqTrader (robuste):
    - essaie d'abord nasdaqlisted.txt,
    - puis fallback sur nasdaqtraded.txt (en filtrant Listing Exchange == 'Q').
    Filtre: Test Issue = 'N', ETF = 'N'.
    """
    urls = [
        "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt",  # fallback élargi
    ]
    symbols: list[str] = []
    seen = set()

    for u in urls:
        r = _get(u)
        if not r or not r.text:
            continue
        rows = _parse_pipe_file(r.text)
        if not rows:
            continue

        # Détection nom de colonne du symbole
        # nasdaqlisted -> "Symbol"
        # nasdaqtraded -> "NASDAQ Symbol"
        sym_col = None
        for cand in ("Symbol", "NASDAQ Symbol"):
            if rows and cand in rows[0]:
                sym_col = cand
                break
        if sym_col is None:
            continue

        # Colonnes facultatives
        test_col = "Test Issue" if "Test Issue" in rows[0] else None
        etf_col = "ETF" if "ETF" in rows[0] else None
        exch_col = None
        for cand in ("Listing Exchange", "Exchange"):
            if cand in rows[0]:
                exch_col = cand
                break

        count_added = 0
        for row in rows:
            raw = row.get(sym_col)
            n = _norm_yf(raw) if raw else None
            if not n or n in seen:
                continue
            # Filtre Test Issue
            if test_col and str(row.get(test_col, "")).upper() == "Y":
                continue
            # Filtre ETF
            if etf_col and str(row.get(etf_col, "")).upper() == "Y":
                continue
            # Si on est sur le fallback "nasdaqtraded", ne garde que l'exchange NASDAQ ('Q')
            if "nasdaqtraded" in u and exch_col:
                if str(row.get(exch_col, "")).upper() != "Q":
                    continue
            seen.add(n)
            symbols.append(n)

        # Si on a déjà beaucoup de symboles via nasdaqlisted, on peut s'arrêter
        if "nasdaqlisted" in u and len(symbols) > 1000:
            print(f"[SRC] nasdaqlisted: {len(symbols)} symbols")
        if "nasdaqtraded" in u:
            print(f"[SRC] nasdaqtraded (Q only): +{count_added} (total {len(symbols)})")

    print(f"[SRC] NASDAQ Composite (robust): ~{len(symbols)} tickers")
    return symbols

# ------------ Yahoo enrich (mcap/liquidité) ------------
def enrich_with_yf(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            y = yf.Ticker(t)
            fi = getattr(y, "fast_info", None) or {}
            mcap = fi.get("market_cap")
            price = fi.get("last_price")
            vol10 = fi.get("ten_day_average_volume")
            avg_dollar = None
            if isinstance(price, (int, float)) and isinstance(vol10, (int, float)):
                avg_dollar = float(price) * float(vol10)
            rows.append({"ticker_yf": t, "mcap_usd": mcap, "avg_dollar_vol": avg_dollar})
        except Exception:
            rows.append({"ticker_yf": t, "mcap_usd": None, "avg_dollar_vol": None})
        if i % 120 == 0:
            print(f"[YF] {i}/{len(tickers)} enriched…")
            time.sleep(0.8)
    return pd.DataFrame(rows)

# ------------ Sector fill (yfinance.info -> Alpha Vantage) ------------
def try_fill_sector_with_yfinfo(df: pd.DataFrame, ticker_col="ticker_yf") -> pd.DataFrame:
    df = df.copy()
    sec, ind = [], []
    for i, sym in enumerate(df[ticker_col].tolist(), 1):
        s, it = None, None
        try:
            info = yf.Ticker(sym).info  # peut être lent mais on limite au Top N
            s = info.get("sector")
            it = info.get("industry")
        except Exception:
            pass
        sec.append(s)
        ind.append(it)
        if i % 25 == 0:
            print(f"[YF.info] {i}/{len(df)}")
        time.sleep(YF_INFO_SLEEP)
    df["sector_fill"] = sec
    df["industry_fill"] = ind
    return df

def try_fill_sector_with_av(df: pd.DataFrame, ticker_col="ticker_yf") -> pd.DataFrame:
    if not AV_KEY or df.empty:
        df["av_sector"] = None
        df["av_industry"] = None
        return df

    sec, ind = [], []
    done = 0
    for i, sym in enumerate(df[ticker_col].tolist(), 1):
        if done >= MAX_AV_LOOKUPS:
            sec.append(None)
            ind.append(None)
            continue
        try:
            url = "https://www.alphavantage.co/query"
            params = {"function": "OVERVIEW", "symbol": sym, "apikey": AV_KEY}
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                j = r.json()
                sec.append(j.get("Sector"))
                ind.append(j.get("Industry"))
                done += 1
            else:
                sec.append(None)
                ind.append(None)
        except Exception:
            sec.append(None)
            ind.append(None)
        time.sleep(AV_SLEEP)  # respecter la rate-limit
    df["av_sector"] = sec
    df["av_industry"] = ind
    return df

# ------------ Utils ------------
def _norm_sector(s: str) -> str:
    return str(s or "").strip().lower()

# ------------ MAIN ------------
def main():
    # 1) Sources
    r1_syms, r1_sec = get_r1000()
    sp_syms, sp_sec = get_sp500()
    nd_syms, nd_sec = get_nasdaq100()
    comp_syms = get_nasdaq_composite() if USE_NASDAQ_COMPOSITE else []

    # Union
    universe = _union(r1_syms, sp_syms, nd_syms, comp_syms)
    if len(universe) < MIN_ROW_LEN:
        raise SystemExit("❌ universe empty")
    print(f"[UNION] unique tickers before YF: {len(universe)}")

    # 2) Yahoo fast (cap/liquidité)
    meta = enrich_with_yf(universe)
    meta["ticker_tv"] = meta["ticker_yf"].map(_yf_to_tv)

    # 3) Mapping secteurs connus (R1K/SPX/NDX depuis Wikipédia)
    sector_map: Dict[str, str] = {}
    sector_map.update(r1_sec)
    sector_map.update(sp_sec)
    sector_map.update(nd_sec)
    meta["sector"] = meta["ticker_yf"].map(lambda t: sector_map.get(t, ""))
    meta["industry"] = ""  # sera éventuellement rempli plus bas

    # RAW avant filtre
    raw_cols = ["ticker_yf", "ticker_tv", "mcap_usd", "avg_dollar_vol", "sector", "industry"]
    raw = meta[raw_cols].copy()
    _save(raw, "raw_universe.csv")

    # 4) Pré-filtre MCAP < 75B
    mcap_num = pd.to_numeric(raw["mcap_usd"], errors="coerce").fillna(0)
    meta_lt75 = raw[mcap_num < MCAP_MAX_USD].copy()

    # 5) Compléter SECTOR uniquement pour un sous-ensemble top liquide <75B avec secteur manquant
    meta_lt75["avg_dollar_vol"] = pd.to_numeric(meta_lt75["avg_dollar_vol"], errors="coerce")
    need_fill = meta_lt75[meta_lt75["sector"].astype(str).str.strip().eq("")].copy()
    need_fill = need_fill.sort_values("avg_dollar_vol", ascending=False).head(TOP_LIQ_FOR_SECTOR_ENRICH)

    if not need_fill.empty:
        print(f"[ENRICH] sector for top {len(need_fill)} <75B with empty sector (YF.info -> AlphaVantage)…")
        step = try_fill_sector_with_yfinfo(need_fill, "ticker_yf")
        # copy back if found
        for col_src, col_dst in [("sector_fill", "sector"), ("industry_fill", "industry")]:
            mask = step[col_src].notna() & step[col_src].astype(str).str.strip().ne("")
            meta_lt75.loc[step.index[mask], col_dst] = step.loc[mask, col_src].values

        # restants -> Alpha Vantage (si clé)
        rest = step[step["sector_fill"].isna() | (step["sector_fill"].astype(str).str.strip() == "")]
        if not rest.empty:
            rest2 = try_fill_sector_with_av(rest, "ticker_yf")
            mask2 = rest2["av_sector"].notna() & rest2["av_sector"].astype(str).str.strip().ne("")
            meta_lt75.loc[rest2.index[mask2], "sector"] = rest2.loc[mask2, "av_sector"].values
            meta_lt75.loc[rest2.index[mask2], "industry"] = rest2.loc[mask2, "av_industry"].values

    # 6) Filtre final: sector ∈ {Tech, Financials, Industrials, Health Care} & MCAP < 75B
    sec_norm = meta_lt75["sector"].map(_norm_sector)
    in_scope = meta_lt75[sec_norm.isin(ALLOWED_SECTORS)].copy()

    # tri: liquidité décroissante puis cap croissant (quand dispos)
    in_scope["mcap_usd"] = pd.to_numeric(in_scope["mcap_usd"], errors="coerce")
    in_scope["avg_dollar_vol"] = pd.to_numeric(in_scope["avg_dollar_vol"], errors="coerce")
    in_scope = in_scope.sort_values(["avg_dollar_vol", "mcap_usd"], ascending=[False, True])

    # 7) Sorties
    _save(in_scope, "universe_in_scope.csv")
    _save(in_scope.rename(columns={"ticker_yf": "ticker_yf"}), "universe_today.csv")

    print(f"[FILTER] sectors ∈ (Tech/Fin/Ind/HCare) & mcap < 75B: {len(in_scope)}/{len(raw)} (raw), {len(in_scope)}/{len(meta_lt75)} (<75B)")
    print("[DONE] build_universe complete.")

if __name__ == "__main__":
    main()
