# sec_helpers.py
from __future__ import annotations
import json, os, time, pathlib, typing as t
from datetime import datetime
import requests

# =========================
# Config
# =========================
SEC_CONTACT = os.getenv("SEC_CONTACT", "you@example.com")  # set a real email in CI
UA = f"TradingScanBot/0.1 ({SEC_CONTACT}; research POC)"
HEADERS = {
    "User-Agent": UA,
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}
TIMEOUT = 20
RETRY_DELAYS = [2, 5, 10]  # seconds
CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ticker directory
TICKERS_CACHE = CACHE_DIR / "sec_company_tickers.json"
TICKERS_TTL_DAYS = 3
SEC_TICKERS_URL = "https://data.sec.gov/api/xbrl/company_tickers.json"
SEC_TICKERS_FALLBACKS = [
    "https://raw.githubusercontent.com/secdatabase/sec-files/master/company_tickers.json",
    "https://raw.githubusercontent.com/davidtaoarw/edgar-company-tickers/master/company_tickers.json",
]

# Company facts (per CIK) cache
FACTS_DIR = CACHE_DIR / "sec_company_facts"
FACTS_DIR.mkdir(parents=True, exist_ok=True)
FACTS_TTL_DAYS = 7
FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_session: requests.Session | None = None

def _session_get() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _session = s
    return _session

def _is_fresh(path: pathlib.Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    try:
        age = (datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)).days
        return age <= ttl_days
    except Exception:
        return False

def _download_json(url: str) -> t.Any:
    s = _session_get()
    last_exc = None
    for delay in [0] + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            r = s.get(url, timeout=TIMEOUT)
            time.sleep(0.2)  # be gentle with EDGAR
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
    raise last_exc

# =========================
# Ticker directory (ticker -> {cik,title})
# =========================
def _load_ticker_dir() -> t.Any:
    # cache hit?
    if _is_fresh(TICKERS_CACHE, TICKERS_TTL_DAYS):
        with open(TICKERS_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    # fetch primary
    data = None
    try:
        data = _download_json(SEC_TICKERS_URL)
    except Exception:
        # fallbacks
        for fb in SEC_TICKERS_FALLBACKS:
            try:
                data = _download_json(fb)
                break
            except Exception:
                pass
        if data is None:
            raise  # bubble up the primary error

    try:
        with open(TICKERS_CACHE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
    return data

def sec_ticker_map() -> dict[str, dict]:
    """
    Returns {'AAPL': {'cik': 320193, 'title': 'Apple Inc.'}, ...}
    """
    raw = _load_ticker_dir()
    entries = raw.values() if isinstance(raw, dict) else raw
    out: dict[str, dict] = {}
    for row in entries:
        try:
            tkr = str(row.get("ticker", "")).strip().upper()
            if not tkr:
                continue
            cik = row.get("cik_str")
            cik = int(cik) if cik not in (None, "") else None
            title = row.get("title") or ""
            out[tkr] = {"cik": cik, "title": title}
        except Exception:
            continue
    return out

# =========================
# Company facts per CIK (to derive latest shares outstanding)
# =========================
def _cik10(cik: int | str) -> str:
    """Pad CIK to 10 digits as required by EDGAR companyfacts endpoint."""
    c = int(str(cik).strip())
    return f"{c:010d}"

def _facts_cache_path(cik10: str) -> pathlib.Path:
    return FACTS_DIR / f"{cik10}.json"

def _load_company_facts(cik: int | str) -> dict:
    cik10 = _cik10(cik)
    p = _facts_cache_path(cik10)
    if _is_fresh(p, FACTS_TTL_DAYS):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    url = FACTS_URL_TMPL.format(cik=cik10)
    data = _download_json(url)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
    return data

def _parse_iso_date(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def _latest_value_from_series(series: list[dict]) -> float | None:
    """
    Given a list of facts (each with e.g. 'end', 'val'), return the most recent non-null val.
    """
    best_dt = None
    best_val = None
    for it in series:
        if "val" not in it:
            continue
        # prefer 'end' date; fall back to 'filed'
        dt = _parse_iso_date(it.get("end")) or _parse_iso_date(it.get("filed"))
        if dt is None:
            continue
        try:
            val = float(it["val"])
        except Exception:
            continue
        if (best_dt is None) or (dt > best_dt):
            best_dt, best_val = dt, val
    return best_val

def _latest_shares_from_facts(facts_json: dict) -> float | None:
    """
    Try several common us-gaap tags, prefer units 'shares'.
    """
    if not facts_json:
        return None
    facts = (facts_json.get("facts") or {}).get("us-gaap") or {}

    # Priority order of tags that commonly carry share counts
    CANDIDATE_TAGS = [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "SharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
    ]

    for tag in CANDIDATE_TAGS:
        node = facts.get(tag)
        if not node:
            continue
        units = node.get("units") or {}
        # first try "shares" unit (best), then any other unit as fallback
        candidates = []
        if "shares" in units:
            candidates = units["shares"]
        else:
            # sometimes it shows as "pure" or other; try the first available unit
            # but avoid obvious non-share units like "USD" etc.
            for unit, series in units.items():
                if "share" in unit.lower() or unit.lower() in ("pure",):
                    candidates = series
                    break
            if not candidates:
                # as last resort, just take first unit series
                for series in units.values():
                    candidates = series
                    break
        val = _latest_value_from_series(candidates)
        if val is not None and val > 0:
            return float(val)
    return None

def sec_latest_shares_outstanding(symbol_or_cik: str | int) -> float | None:
    """
    Returns the latest shares outstanding (float) from SEC Company Facts,
    or None if unavailable.
    Accepts either a ticker (str) or a CIK (int/str).
    """
    cik: int | None = None
    if isinstance(symbol_or_cik, (int,)) or str(symbol_or_cik).isdigit():
        cik = int(symbol_or_cik)
    else:
        tkr = str(symbol_or_cik).strip().upper()
        m = sec_ticker_map()
        meta = m.get(tkr)
        if not meta or not meta.get("cik"):
            return None
        cik = int(meta["cik"])

    try:
        facts = _load_company_facts(cik)
    except Exception:
        return None

    try:
        return _latest_shares_from_facts(facts)
    except Exception:
        return None
