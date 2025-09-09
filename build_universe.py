# build_universe.py
# Assemble un univers US large-cap/mid-cap à partir de sources publiques robustes,
# avec fallbacks propres et sans boucles infinies.

import os
import re
import io
import time
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

# où écrire
RAW_CSV = Path("raw_universe.csv")
IN_SCOPE_CSV = Path("universe_in_scope.csv")

# bornes simples pour filtrage basique (le screening affinera ensuite)
# (on laisse large ici — pas de données mcap ici, c’est un simple univers brut)
VALID_TICKER_RE = re.compile(r"^[A-Z0-9.-]{1,10}$")

# ==== Utils =================================================================

def _http_get(url: str) -> Optional[str]:
    """GET texte avec UA + timeout + gestion d'erreurs, sans raise."""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f"[HTTP] {url} -> {r.status_code}")
            return None
        # Essaye d'auto-détecter l'encodage
        r.encoding = r.apparent_encoding or r.encoding or "utf-8"
        return r.text
    except Exception as e:
        print(f"[HTTP] fail {url}: {e}")
        return None


def _read_csv_text(text: str) -> Optional[pd.DataFrame]:
    """Lecture CSV depuis un texte brut (corrige le pb pandas.compat.StringIO)."""
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"[CSV] parse failed (text): {e}")
        return None


def _fetch_csv_first_ok(urls: Iterable[str], label: str) -> Optional[pd.DataFrame]:
    """Essaie une liste d'URLs CSV; retourne le premier DataFrame non vide."""
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
    """read_html avec timeout via requests -> StringIO -> pandas.read_html."""
    html = _http_get(url)
    if not html:
        return []
    try:
        # read_html sait parser plusieurs tableaux; on cappe le parse pour éviter lenteur
        return pd.read_html(io.StringIO(html))
    except Exception as e:
        print(f"[HTML] parse failed {url}: {e}")
        return []


def _sanitize_to_yahoo(sym: str) -> Optional[str]:
    """
    Nettoie un ticker en format Yahoo:
      - majuscules
      - remplace '.' -> '-' (ex: BRK.B -> BRK-B)
      - supprime espaces/trim
      - filtre chaînes trop longues/invalides
    """
    if not sym:
        return None
    s = str(sym).strip().upper()
    # Exclusions rapides de pseudo-IDs
    if s in {"NAN", "NULL", "NONE", "ISRAEL", "CHINA", "SWEDEN"}:
        return None
    # remplace '.' par '-' pour compat Yahoo
    s = s.replace(".", "-")
    # retire suffixes fréquents non US (ex: ".MI", ".L" si jamais)
    s = re.sub(r"-[A-Z]{1,3}$", lambda m: m.group(0) if len(m.group(0)) == 2 else m.group(0), s)
    # garde uniquement ce qui ressemble à un ticker
    if not VALID_TICKER_RE.match(s):
        return None
    # filtres rapides contre trucs manifestement pas des tickers
    if len(s) < 1 or len(s) > 10:
        return None
    # évite des mots connus non-tickers (déjà majuscules)
    blacklist = {"BANKS", "AIRLINES", "SOFTWARE", "CURRENCY", "ENERGY", "MEDIA"}
    if s in blacklist:
        return None
    return s


def _extract_ticker_column(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Essaie d'extraire une colonne ticker depuis un DF (souvent 'Symbol', 'Ticker', 'Ticker symbol').
    Insensible à la casse et aux espaces.
    """
    if df is None or df.empty:
        return None
    lc = {c.lower().strip(): c for c in df.columns}
    candidates = [
        "symbol", "ticker", "ticker symbol", "ticker_symbol", "code", "sym"
    ]
    for look in candidates:
        if look in lc:
            return df[lc[look]]
    # parfois la colonne s'appelle 'Symbols'
    for k, orig in lc.items():
        if "symbol" in k:
            return df[orig]
    return None


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
    """
    S&P 500 : dataset GitHub fiable (constituents.csv).
    """
    urls = [
        # OK au moment de l’écriture / liens stables
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        # fallback (même repo, autre chemin historique)
        "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/constituents.csv",
    ]
    df = _fetch_csv_first_ok(urls, "SPX")
    if df is None:
        return None
    col = _extract_ticker_column(df)
    if col is None:
        # certains dumps ont 'Symbol' déjà
        if "Symbol" in df.columns:
            col = df["Symbol"]
        else:
            print("[SPX] no ticker column found")
            return None
    return _df_from_tickers(col.dropna().astype(str), "SPX")


def load_r1000() -> Optional[pd.DataFrame]:
    """
    Russell 1000 : Wikipedia (liste officielle, table parseable).
    """
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
    """
    Nasdaq-100 : Wikipedia (Components).
    """
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

    # 1) SPX
    spx = load_spx()
    if spx is not None and not spx.empty:
        pools.append(spx)

    # 2) R1000
    r1k = load_r1000()
    if r1k is not None and not r1k.empty:
        pools.append(r1k)

    # 3) NDX
    ndx = load_ndx()
    if ndx is not None and not ndx.empty:
        pools.append(ndx)

    # 4) Si rien n’a été récupéré en ligne, tente un fallback local existant
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
        # Rien du tout — on sort proprement pour éviter les « No objects to concatenate »
        print("❌ No sources available and no valid fallback raw_universe.csv. Exiting.")
        return

    # 5) Fusion + dédup
    raw = pd.concat(pools, ignore_index=True)
    raw = raw.dropna(subset=["ticker_yf"])
    raw["ticker_yf"] = raw["ticker_yf"].astype(str).str.strip().str.upper()

    # Filtrage basique anti-bruit
    raw = raw[raw["ticker_yf"].str.match(VALID_TICKER_RE)]
    raw = raw[~raw["ticker_yf"].isin({"NAN", "NONE", "NULL"})]

    # Dédup final
    raw = raw.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)

    # 6) Sauvegarde raw
    _save(raw[["ticker_yf"]], RAW_CSV)

    # 7) Construire universe_in_scope (pour la suite du pipeline)
    #    Ici on garde juste la colonne attendue; d’autres colonnes pourront être ajoutées plus tard.
    in_scope = raw[["ticker_yf"]].copy()
    _save(in_scope, IN_SCOPE_CSV)

    print("[DONE] build_universe finished.")

if __name__ == "__main__":
    main()
