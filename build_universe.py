# build_universe.py
# Assemble un univers US large/mid à partir de sources publiques robustes,
# avec fallbacks propres et sans boucles infinies.

import os
import re
import io
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

# ==== Config ================================================================

TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

RAW_CSV = Path("raw_universe.csv")
IN_SCOPE_CSV = Path("universe_in_scope.csv")

VALID_TICKER_RE = re.compile(r"^[A-Z0-9.-]{1,10}$")

# ==== Utils =================================================================

def _http_get(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f"[HTTP] {url} -> {r.status_code}")
            return None
        r.encoding = r.apparent_encoding or r.encoding or "utf-8"
        return r.text
    except Exception as e:
        print(f"[HTTP] fail {url}: {e}")
        return None


def _read_csv_text(text: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"[CSV] parse failed (text): {e}")
        return None


def _fetch_csv_first_ok(urls: Iterable[str], label: str) -> Optional[pd.DataFrame]:
    for u in urls:
        txt = _http_get(u)
        if not txt:
            continue
        df = _read_csv_text(txt)
        if df is None or df.empty:
            continue
        df["src"] = label
        print(f"[SRC] {label}: {u} OK rows={len(df)} cols={list(df.columns)}")
        return df
    print(f"[SRC] {label} not available; returning empty df")
    return None


def _read_html_tables(url: str) -> List[pd.DataFrame]:
    html = _http_get(url)
    if not html:
        return []
    try:
        return pd.read_html(io.StringIO(html))
    except Exception as e:
        print(f"[HTML] parse failed {url}: {e}")
        return []


def _sanitize_to_yahoo(sym: str) -> Optional[str]:
    if sym is None:
        return None
    s = str(sym).strip().upper()
    if s in {"", "NAN", "NULL", "NONE", "ISRAEL", "CHINA", "SWEDEN"}:
        return None
    # Yahoo: BRK.B -> BRK-B
    s = s.replace(".", "-")
    if not VALID_TICKER_RE.match(s):
        return None
    if len(s) > 10:
        return None
    blacklist = {"BANKS", "AIRLINES", "SOFTWARE", "CURRENCY", "ENERGY", "MEDIA"}
    if s in blacklist:
        return None
    return s


def _guess_ticker_col_by_values(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Devine la colonne ticker en scorant chaque colonne par le ratio de cellules
    qui ressemblent à des tickers Yahoo.
    """
    best_col = None
    best_score = 0.0
    for col in df.columns:
        try:
            ser = df[col]
            if ser.dtype == object or pd.api.types.is_string_dtype(ser):
                vals = ser.dropna().astype(str).str.strip()
            else:
                # nombreuses tables wiki ont des ints/objects — on cast en str
                vals = ser.dropna().astype(str).str.strip()
            if vals.empty:
                continue
            total = min(len(vals), 500)
            sample = vals.head(total)
            hits = sum(1 for v in sample if _sanitize_to_yahoo(v) is not None)
            score = hits / float(total)
            if hits >= 10 and score >= 0.30 and score > best_score:
                best_score = score
                best_col = col
        except Exception:
            continue
    if best_col is not None:
        print(f"[GUESS] picked '{best_col}' as ticker column (score={best_score:.2f})")
        return df[best_col]
    return None


def _extract_ticker_column(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    # Cast noms de colonnes en str pour éviter AttributeError: int has no attribute lower
    col_map = {str(c).lower().strip(): c for c in df.columns}
    candidates = ["symbol", "ticker", "ticker symbol", "ticker_symbol", "code", "sym"]
    for look in candidates:
        if look in col_map:
            return df[col_map[look]]
    # fallback: toute colonne contenant "symbol" dans le nom
    for k, orig in col_map.items():
        if "symbol" in k:
            return df[orig]
    # dernier recours: deviner par contenu
    return _guess_ticker_col_by_values(df)


def _df_from_tickers(tickers: Iterable[str], label: str) -> pd.DataFrame:
    vals = []
    for t in tickers:
        y = _sanitize_to_yahoo(t)
        if y:
            vals.append(y)
    out = pd.DataFrame({"ticker_yf": sorted(set(vals))})
    out["src"] = label
    return out


# ==== Sources ===============================================================

def load_spx() -> Optional[pd.DataFrame]:
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/constituents.csv",
    ]
    df = _fetch_csv_first_ok(urls, "SPX")
    if df is None:
        return None
    col = _extract_ticker_column(df)
    if col is None:
        if "Symbol" in df.columns:
            col = df["Symbol"]
        else:
            print("[SPX] no ticker column found")
            return None
    return _df_from_tickers(col.dropna().astype(str), "SPX")


def load_r1000() -> Optional[pd.DataFrame]:
    tables = _read_html_tables("https://en.wikipedia.org/wiki/Russell_1000_Index")
    tickers = []
    for tb in tables:
        col = _extract_ticker_column(tb)
        if col is not None:
            tickers.extend(col.dropna().astype(str).tolist())
    if not tickers:
        print("[R1000] no tickers parsed from Wikipedia")
        return None
    return _df_from_tickers(tickers, "R1000")


def load_ndx() -> Optional[pd.DataFrame]:
    tables = _read_html_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
    tickers = []
    for tb in tables:
        col = _extract_ticker_column(tb)
        if col is not None:
            tickers.extend(col.dropna().astype(str).tolist())
    if not tickers:
        print("[NDX] no tickers parsed from Wikipedia")
        return None
    return _df_from_tickers(tickers, "NDX")


# ==== Pipeline ==============================================================

def _save(df: pd.DataFrame, path: Path):
    try:
        df.to_csv(path, index=False)
        print(f"[SAVE] {path} | rows={len(df)} cols={len(df.columns)}")
    except Exception as e:
        print(f"[SAVE] failed {path}: {e}")


def main():
    print("[STEP] build_universe starting…")

    pools: List[pd.DataFrame] = []

    spx = load_spx()
    if spx is not None and not spx.empty:
        pools.append(spx)

    r1k = load_r1000()
    if r1k is not None and not r1k.empty:
        pools.append(r1k)

    ndx = load_ndx()
    if ndx is not None and not ndx.empty:
        pools.append(ndx)

    if not pools:
        if RAW_CSV.exists():
            try:
                prev = pd.read_csv(RAW_CSV)
                if "ticker_yf" in prev.columns and not prev.empty:
                    print("[FALLBACK] using existing raw_universe.csv")
                    pools.append(prev[["ticker_yf"]].assign(src="FALLBACK"))
                else:
                    print("[FALLBACK] raw_universe.csv exists but invalid/empty")
            except Exception as e:
                print(f"[FALLBACK] failed to read raw_universe.csv: {e}")

    if not pools:
        print("❌ No sources available and no valid fallback raw_universe.csv. Exiting.")
        return

    raw = pd.concat(pools, ignore_index=True)
    raw = raw.dropna(subset=["ticker_yf"])
    raw["ticker_yf"] = raw["ticker_yf"].astype(str).str.strip().str.upper()
    raw = raw[raw["ticker_yf"].str.match(VALID_TICKER_RE)]
    raw = raw[~raw["ticker_yf"].isin({"NAN", "NONE", "NULL"})]
    raw = raw.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)

    _save(raw[["ticker_yf"]], RAW_CSV)

    in_scope = raw[["ticker_yf"]].copy()
    _save(in_scope, IN_SCOPE_CSV)

    print("[DONE] build_universe finished.")


if __name__ == "__main__":
    main()
