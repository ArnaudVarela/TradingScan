# sec_helpers.py
# -----------------------------------------------------------------------------
# Utilitaires SEC robustes avec cache + fallbacks.
# - sec_ticker_map()  -> dict[str, dict]: map TICKER -> {cik, title, exchange?}
# - sec_latest_shares_outstanding(cik) -> float|None : dernières "CommonStockSharesOutstanding"
#
# Respecte la recommandation SEC d'User-Agent + email de contact :
#   export SEC_CONTACT="ton.email@domaine.tld"
# (ou définis SEC_CONTACT dans les "env" GitHub Actions).
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import time
import typing as t
from pathlib import Path

import requests

# ------------------------- Config & chemins cache -----------------------------

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers cache
TICKERS_CACHE = CACHE_DIR / "sec_company_tickers.json"   # cache de l'annuaire tickers
FACTS_CACHE_DIR = CACHE_DIR / "sec_companyfacts"         # 1 fichier par CIK
FACTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TTL (jours)
TICKERS_TTL_DAYS = 7         # annuaire tickers : hebdo suffit
FACTS_TTL_DAYS = 3           # facts : plus fréquent

# URLs officielles / alternatives
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKERS_ALTS = [
    # Alternative officielle (structure proche, groupé par exchange)
    "https://www.sec.gov/files/company_tickers_exchange.json",
]
SEC_TICKERS_FALLBACKS = [
    # miroirs communautaires (à utiliser en dernier recours)
    "https://raw.githubusercontent.com/secdatabase/sec-files/master/company_tickers.json",
    "https://raw.githubusercontent.com/davidtaoarw/edgar-company-tickers/master/company_tickers.json",
]

# API companyfacts
SEC_COMPANYFACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_pad}.json"

# Respect rate limiting SEC (gentle)
REQ_SLEEP_SEC = 0.20

# ------------------------- Helpers généraux ----------------------------------

def _sec_headers() -> dict:
    """
    Entêtes HTTP conformes aux guidelines SEC:
      - User-Agent avec contact
      - Accept/encoding
    """
    contact = os.environ.get("SEC_CONTACT", "").strip()
    if contact:
        ua = f"TradingScan/1.0 (contact: {contact})"
    else:
        # Mieux vaut quand même fournir quelque chose d'explicite
        ua = "TradingScan/1.0 (contact: arnaud.varela@gmail.com)"
    return {
        "User-Agent": ua,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        # La SEC n'exige pas de Referer, mais certains proxies peuvent l'apprécier
        "Referer": "https://www.sec.gov/",
    }


def _is_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    age_sec = time.time() - path.stat().st_mtime
    return age_sec < (ttl_days * 86400)


def _get_with_retries(url: str, timeout: float = 30.0, retries: int = 3) -> requests.Response:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=_sec_headers(), timeout=timeout)
            if r.status_code == 429:
                # Trop de requêtes — attendre un peu plus
                time.sleep(0.6 + i * 0.4)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            # petit backoff progressif
            time.sleep(0.4 + i * 0.6)
    # si toutes les tentatives échouent
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to GET {url} for unknown reasons")


def _download_json(url: str) -> t.Any:
    r = _get_with_retries(url)
    time.sleep(REQ_SLEEP_SEC)  # courtoisie
    # Certaines sources de fallback livrent déjà du JSON propre
    # D'autres livrent du texte à parser
    try:
        return r.json()
    except ValueError:
        return json.loads(r.text)


# ------------------------- Chargement annuaire tickers -----------------------

def _normalize_ticker(t: str) -> str:
    """
    Normalisation douce des tickers pour mapping :
      - upper
      - trim
      - version avec '.' <-> '-' traitée au lookup
    """
    return (t or "").strip().upper()


def _save_json(path: Path, data: t.Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        # ne jamais faire planter le run pour un échec d'écriture cache
        pass


def _load_json(path: Path) -> t.Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_ticker_dir() -> t.Any:
    """
    Charge l'annuaire des tickers depuis les sources officielles SEC (puis fallbacks).
    - Utilise cache frais si dispo,
    - Si tout échoue, tente le cache même périmé,
    - Sinon propage la dernière erreur.
    """
    if _is_fresh(TICKERS_CACHE, TICKERS_TTL_DAYS):
        try:
            return _load_json(TICKERS_CACHE)
        except Exception:
            # on refetch
            pass

    sources = [SEC_TICKERS_URL, *SEC_TICKERS_ALTS, *SEC_TICKERS_FALLBACKS]
    last_err: Exception | None = None
    for url in sources:
        try:
            data = _download_json(url)
            _save_json(TICKERS_CACHE, data)
            return data
        except Exception as e:
            last_err = e
            continue

    # toutes les sources échouent : tentons un cache obsolète
    if TICKERS_CACHE.exists():
        try:
            return _load_json(TICKERS_CACHE)
        except Exception:
            pass

    if last_err:
        raise last_err
    raise RuntimeError("SEC ticker directory unavailable and no cache on disk")


def _parse_ticker_dir(raw: t.Any) -> dict[str, dict]:
    """
    Uniformise la structure de l'annuaire SEC vers :
      map[ticker] = { 'cik': '##########', 'title': 'Company Name', 'exchange': 'NYSE/Nasdaq/...' }
    Les différents fichiers (company_tickers.json vs *_exchange.json vs fallbacks) ont des schémas proches.
    """
    result: dict[str, dict] = {}

    # Cas 1: format officiel "company_tickers.json" (dict indexé par '0','1',... avec objets)
    # Exemple d'item: {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    if isinstance(raw, dict) and raw and all(k.isdigit() for k in raw.keys()):
        for _, obj in raw.items():
            try:
                ticker = _normalize_ticker(obj.get("ticker", ""))
                if not ticker:
                    continue
                cik = str(obj.get("cik_str", "")).zfill(10)
                title = obj.get("title", "") or ""
                # Pas d'exchange direct dans ce format
                result[ticker] = {"cik": cik, "title": title, "exchange": obj.get("exchange") or ""}
            except Exception:
                continue
        return result

    # Cas 2: format "company_tickers_exchange.json": dict par exchange -> liste d'obj
    # Exemple: {"nasdaq": [{"cik_str":..., "ticker":"AAPL", "title":"Apple Inc."}, ...], "nyse": [...], ...}
    if isinstance(raw, dict) and any(isinstance(v, list) for v in raw.values()):
        for exch, arr in raw.items():
            if not isinstance(arr, list):
                continue
            for obj in arr:
                try:
                    ticker = _normalize_ticker(obj.get("ticker", ""))
                    if not ticker:
                        continue
                    cik = str(obj.get("cik_str", "")).zfill(10)
                    title = obj.get("title", "") or ""
                    result[ticker] = {"cik": cik, "title": title, "exchange": exch.upper()}
                except Exception:
                    continue
        if result:
            return result

    # Cas 3: fallbacks divers — tenter quelques heuristiques
    # 3a) liste d'objets [{"cik_str":..., "ticker":"...", "title":"..."}]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        for obj in raw:
            try:
                ticker = _normalize_ticker(obj.get("ticker", ""))
                if not ticker:
                    continue
                cik_val = obj.get("cik_str") or obj.get("cik") or obj.get("CIK") or ""
                cik = str(cik_val).zfill(10) if cik_val else ""
                title = obj.get("title") or obj.get("name") or obj.get("company") or ""
                exch = obj.get("exchange") or ""
                result[ticker] = {"cik": cik, "title": title, "exchange": exch}
            except Exception:
                continue
        if result:
            return result

    # 3b) dict déjà map[ticker]->obj
    if isinstance(raw, dict):
        for k, obj in raw.items():
            try:
                ticker = _normalize_ticker(k)
                if not ticker:
                    continue
                if isinstance(obj, dict):
                    cik_val = obj.get("cik_str") or obj.get("cik") or obj.get("CIK") or ""
                    cik = str(cik_val).zfill(10) if cik_val else ""
                    title = obj.get("title") or obj.get("name") or obj.get("company") or ""
                    exch = obj.get("exchange") or ""
                else:
                    cik = ""
                    title = str(obj)
                    exch = ""
                result[ticker] = {"cik": cik, "title": title, "exchange": exch}
            except Exception:
                continue
        if result:
            return result

    # si on arrive ici, impossible d'interpréter — on renvoie vide
    return result


def _with_ticker_variants(map_by_ticker: dict[str, dict]) -> dict[str, dict]:
    """
    Ajoute des variantes '.' <-> '-' pour maximiser les hits (ex: BRK.B vs BRK-B).
    """
    extra: dict[str, dict] = {}
    for tk, meta in map_by_ticker.items():
        if "." in tk:
            extra[tk.replace(".", "-")] = meta
        if "-" in tk:
            extra[tk.replace("-", ".")] = meta
    map_by_ticker.update(extra)
    return map_by_ticker


def sec_ticker_map() -> dict[str, dict]:
    """
    Renvoie un mapping TICKER -> {cik, title, exchange}
    (cache + fallbacks + variantes tickers).
    """
    raw = _load_ticker_dir()
    mp = _parse_ticker_dir(raw)
    mp = _with_ticker_variants(mp)
    return mp


# ------------------------- Company Facts & actions en circulation ------------

def _facts_cache_path(cik: str) -> Path:
    cik_pad = str(cik).zfill(10)
    return FACTS_CACHE_DIR / f"facts_{cik_pad}.json"


def _download_companyfacts(cik: str) -> t.Any:
    """
    Télécharge les "companyfacts" pour un CIK donné, avec cache et retries.
    """
    cik_pad = str(cik).zfill(10)
    cache_path = _facts_cache_path(cik_pad)

    if _is_fresh(cache_path, FACTS_TTL_DAYS):
        try:
            return _load_json(cache_path)
        except Exception:
            pass

    url = SEC_COMPANYFACTS_URL_TMPL.format(cik_pad=cik_pad)
    data = _download_json(url)
    _save_json(cache_path, data)
    return data


def _latest_unit_value(series_entry: dict, unit_key: str) -> float | None:
    """
    Extrait la dernière valeur disponible (par date) dans la série 'units[unit_key]'.
    Chaque point a la forme {'val': ..., 'fy': ..., 'fp': ..., 'form': '10-K/10-Q', 'end': 'YYYY-MM-DD', ...}
    """
    units = (series_entry or {}).get("units", {})
    datapoints = units.get(unit_key)
    if not isinstance(datapoints, list) or not datapoints:
        return None

    # trier par date 'end' (et fallback sur 'filed' si présent)
    def _key(dp):
        # format 'YYYY-MM-DD', sinon fallback string
        return dp.get("end") or dp.get("filed") or ""

    datapoints_sorted = sorted(datapoints, key=_key)
    for dp in reversed(datapoints_sorted):
        val = dp.get("val")
        if isinstance(val, (int, float)) and val is not None:
            return float(val)
    return None


def sec_latest_shares_outstanding(cik: str) -> float | None:
    """
    Renvoie la dernière valeur (float) des actions en circulation d'une société (CIK),
    en cherchant dans les companyfacts plusieurs clés fréquentes :
      - "CommonStockSharesOutstanding" (classique)
      - "CommonSharesOutstanding" (variation)
      - "WeightedAverageNumberOfSharesOutstandingBasic" (fallback proxy)
    Les unités à tester en priorité: 'shares' puis d'autres si nécessaire.
    """
    try:
        facts = _download_companyfacts(cik)
    except Exception:
        # si on a un cache périmé, tenter la lecture brute (mieux que rien)
        cache_path = _facts_cache_path(cik)
        if cache_path.exists():
            try:
                return _latest_shares_from_facts(_load_json(cache_path))
            except Exception:
                return None
        return None

    return _latest_shares_from_facts(facts)


def _latest_shares_from_facts(facts: dict) -> float | None:
    if not isinstance(facts, dict):
        return None
    # branche US-GAAP généralement
    facts_usgaap = (facts.get("facts") or {}).get("us-gaap") or {}

    # clés candidates
    KEYS = [
        "CommonStockSharesOutstanding",
        "CommonSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ]

    # unités candidates par ordre de préférence
    UNITS = ["shares", "SHARES", "pure"]

    for key in KEYS:
        entry = facts_usgaap.get(key)
        if not entry:
            continue
        for unit in UNITS:
            val = _latest_unit_value(entry, unit)
            if val is not None and val > 0:
                return float(val)
    return None


# ------------------------- Convenience : mcap proxy (optionnel) --------------

def sec_proxy_mcap_from_price_shares(price: float | None, shares_out: float | None) -> float | None:
    """
    Calcul simple de market cap si on a un prix (yfinance) + actions en circulation (SEC).
    """
    if price is None or shares_out is None:
        return None
    try:
        return float(price) * float(shares_out)
    except Exception:
        return None

# === à AJOUTER en bas de sec_helpers.py ===

def _sec_get_json(url: str):
    # compat rétro avec l'ancien mix_ab_screen_indices.py
    return _download_json(url)

def sec_companyfacts(cik: str):
    try:
        return _download_companyfacts(cik)
    except Exception:
        return None

