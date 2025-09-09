# sec_helpers.py
from __future__ import annotations
import json, time, os
from pathlib import Path
from typing import Optional, Dict, Any
import requests

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
MAP_PATH = CACHE_DIR / "sec_ticker_map.json"
SHARES_PATH = CACHE_DIR / "sec_shares_out.json"

SEC_HEADERS = {
    "User-Agent": "yourname@example.com (for research; GitHub: yourrepo)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

def _sec_get_json(url: str, timeout: float = 20.0) -> Any:
    r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def load_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)

def sec_ticker_map(ttl_hours: float = 24.0) -> Dict[str, str]:
    """
    Retourne {TICKER_UPPER -> CIK_10digits}. Cache 24h par défaut.
    """
    cache = load_json(MAP_PATH, {})
    now = time.time()
    if cache.get("_ts") and now - cache["_ts"] < ttl_hours * 3600 and "map" in cache:
        return cache["map"]

    data = _sec_get_json("https://data.sec.gov/api/xbrl/company_tickers.json")
    out = {}
    for _, row in data.items():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)

    save_json(MAP_PATH, {"_ts": now, "map": out})
    return out

def sec_latest_shares_outstanding(cik10: str, prefer_units=("shares","sharesclass"), ttl_hours: float = 48.0) -> Optional[float]:
    """
    Retourne la dernière valeur non-nulle de 'Shares Outstanding' (float) via XBRL, sinon None.
    Cache par CIK pendant 48h.
    """
    cache = load_json(SHARES_PATH, {"_ts": 0, "data": {}})
    key = f"CIK{cik10}"
    rec = cache["data"].get(key)
    now = time.time()
    if rec and now - rec.get("_ts", 0) < ttl_hours * 3600:
        return rec.get("shares")

    try:
        facts = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    except Exception:
        return rec.get("shares") if rec else None

    candidates = [
        ("us-gaap", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
    ]
    best = None
    for ns, tag in candidates:
        try:
            series = facts["facts"][ns][tag]["units"]
        except KeyError:
            continue
        # privilégie les unités contenant "share"
        # ex: "shares", "sharesClass"
        unit_keys = sorted(series.keys(), key=lambda u: 0 if any(p in u.lower() for p in prefer_units) else 1)
        for unit in unit_keys:
            arr = series[unit]
            arr_sorted = sorted(
                arr,
                key=lambda x: (
                    x.get("fy", 0),
                    x.get("fp", ""),
                    x.get("end", ""),
                    x.get("form", ""),
                ),
                reverse=True,
            )
            for obs in arr_sorted:
                val = obs.get("val")
                if val is not None:
                    try:
                        fv = float(val)
                    except Exception:
                        continue
                    if fv > 0:
                        best = fv
                        break
            if best is not None:
                break
        if best is not None:
            break

    cache["data"][key] = {"_ts": now, "shares": best}
    save_json(SHARES_PATH, cache)
    return best
