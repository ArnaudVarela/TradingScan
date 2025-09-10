# sec_helpers.py
import requests, time

SEC_HEADERS = {
    "User-Agent": "TradingScanBot/0.1 (arnaud.varela@gmail.com; research project, GitHub: TradingScan)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

def _sec_get_json(url: str):
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def sec_ticker_map():
    """
    Télécharge la map Ticker -> CIK depuis la SEC
    """
    data = _sec_get_json("https://data.sec.gov/api/xbrl/company_tickers.json")
    out = {}
    for _, row in data.items():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out

def sec_latest_shares_outstanding(cik10: str):
    data = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    candidates = [
        ("us-gaap", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
    ]
    best = None
    for ns, tag in candidates:
        try:
            series = data["facts"][ns][tag]["units"]
        except KeyError:
            continue
        for unit, arr in series.items():
            if "share" not in unit.lower():
                continue
            arr_sorted = sorted(arr, key=lambda x: (x.get("end","")), reverse=True)
            for obs in arr_sorted:
                val = obs.get("val")
                if val and float(val) > 0:
                    best = float(val)
                    break
            if best: break
        if best: break
    return best
