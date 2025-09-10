# sec_helpers.py
from __future__ import annotations
import json, os, time, pathlib, typing as t
from datetime import datetime
import requests

# --- config
SEC_CONTACT = os.getenv("SEC_CONTACT", "arnaud.varela@gmail.com")  # set a real email in CI
UA = f"TradingScanBot/0.1 ({SEC_CONTACT}; research POC)"
HEADERS = {
    "User-Agent": UA,
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    # Host header is set automatically by requests
}
TIMEOUT = 20
RETRY_DELAYS = [2, 5, 10]  # seconds
CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "sec_company_tickers.json"
CACHE_TTL_DAYS = 3  # only refetch every 3 days unless cache missing

SEC_PRIMARY = "https://data.sec.gov/api/xbrl/company_tickers.json"
# very lightweight public mirror of that file in multiple GH repos;
# we use it only if the primary refuses us (should be rare with UA set)
SEC_FALLBACKS = [
    "https://raw.githubusercontent.com/secdatabase/sec-files/master/company_tickers.json",
    "https://raw.githubusercontent.com/davidtaoarw/edgar-company-tickers/master/company_tickers.json",
]

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
    for i, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            r = s.get(url, timeout=TIMEOUT)
            # SEC guideline: be gentle; even on success, tiny pause helps
            time.sleep(0.2)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
    raise last_exc  # bubble up after retries

def _load_cached_or_fetch() -> t.Any:
    # 1) cached and fresh? use it
    if _is_fresh(CACHE_FILE, CACHE_TTL_DAYS):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2) fetch primary
    try:
        data = _download_json(SEC_PRIMARY)
    except Exception:
        # 3) fallback mirrors
        for fb in SEC_FALLBACKS:
            try:
                data = _download_json(fb)
                break
            except Exception:
                data = None
        if data is None:
            raise  # re-raise the primary error

    # persist cache
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass  # cache write failure is non-fatal

    return data

def sec_ticker_map() -> dict[str, dict]:
    """
    Returns a dict:
      { 'AAPL': {'cik': 320193, 'title': 'Apple Inc.'}, ... }
    The SEC JSON is an array/dict of entries with 'ticker', 'cik_str', 'title'.
    We normalize to upper TICKER and int CIK.
    """
    raw = _load_cached_or_fetch()

    # The SEC file can be either a dict indexed by 0..N-1 or a list.
    if isinstance(raw, dict):
        entries = raw.values()
    else:
        entries = raw

    out: dict[str, dict] = {}
    for row in entries:
        try:
            tkr = str(row.get("ticker", "")).strip().upper()
            if not tkr:
                continue
            cik = int(row.get("cik_str")) if row.get("cik_str") not in (None, "") else None
            title = row.get("title") or ""
            out[tkr] = {"cik": cik, "title": title}
        except Exception:
            continue
    return out
