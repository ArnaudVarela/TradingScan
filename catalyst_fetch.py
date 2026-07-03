# catalyst_fetch.py — Overlay CATALYSEUR via SEC EDGAR (dépôts 8-K = événements matériels).
#
# Garde-fou : le catalyseur est une ANNOTATION en COLONNES SÉPARÉES de thematic_setups.csv,
# JAMAIS mélangé au score /100 (le /100 reste purement technique). Factuel (8-K), gratuit, sans clé.
# Non bloquant : si SEC échoue, on n'écrase rien de critique.
import os
import time
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

ROOT = Path(__file__).parent
UA = os.getenv("SEC_USER_AGENT", "TradingScan research arnaud.varela@gmail.com")
LOOKBACK_DAYS = int(os.getenv("CATALYST_DAYS", "30"))
CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
TICKERS_CACHE = CACHE / "sec_tickers.json"
MATERIAL = {"8-K", "8-K/A"}
# Codes d'items 8-K -> libellé lisible (les plus parlants pour un catalyseur)
ITEM_LABELS = {
    "1.01": "Contrat", "1.02": "Fin contrat", "1.05": "Cyberincident",
    "2.01": "Acquisition", "2.02": "Résultats", "2.03": "Financement",
    "3.01": "Delisting", "3.02": "Émission titres", "5.01": "Chgt contrôle",
    "5.02": "Direction", "5.07": "Vote actionnaires", "7.01": "Reg FD", "8.01": "Événement",
}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})

def _tickers_map():
    if TICKERS_CACHE.exists() and (time.time() - TICKERS_CACHE.stat().st_mtime) < 7 * 86400:
        try:
            return json.loads(TICKERS_CACHE.read_text())
        except Exception:
            pass
    r = SESSION.get("https://www.sec.gov/files/company_tickers.json", timeout=25)
    r.raise_for_status()
    m = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}
    try:
        TICKERS_CACHE.write_text(json.dumps(m))
    except Exception:
        pass
    return m

def _catalyst(cik):
    try:
        r = SESSION.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=25)
        r.raise_for_status()
        rec = r.json().get("filings", {}).get("recent", {})
    except Exception:
        return None
    forms = rec.get("form", []); dates = rec.get("filingDate", []); items = rec.get("items", [])
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=LOOKBACK_DAYS)
    best = None; n = 0
    for i, f in enumerate(forms):
        if f not in MATERIAL:
            continue
        try:
            fd = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except Exception:
            continue
        if fd < cutoff:
            continue
        n += 1
        if best is None or fd > best[0]:
            best = (fd, items[i] if i < len(items) else "")
    if best is None:
        return {"catalyst": "", "catalyst_days": "", "catalyst_n": 0}
    days = (datetime.now(timezone.utc).date() - best[0]).days
    codes = [c.strip() for c in str(best[1]).split(",") if c.strip()]
    labels = [ITEM_LABELS[c] for c in codes if c in ITEM_LABELS]
    return {"catalyst": labels[0] if labels else "8-K", "catalyst_days": days, "catalyst_n": n}

def main():
    f = ROOT / "thematic_setups.csv"
    if not f.exists():
        print("[CATALYST] pas de thematic_setups.csv"); return
    df = pd.read_csv(f)
    try:
        cmap = _tickers_map()
    except Exception as e:
        print(f"[CATALYST] map SEC indisponible ({e}) -> overlay ignoré (non bloquant)"); return

    out = {"catalyst": [], "catalyst_days": [], "catalyst_n": []}
    hits = 0
    for t in df["ticker"].astype(str).str.upper():
        cik = cmap.get(t)
        res = _catalyst(cik) if cik else None
        if res is None:
            res = {"catalyst": "", "catalyst_days": "", "catalyst_n": 0}
        if res["catalyst"]:
            hits += 1
        for k in out:
            out[k].append(res[k])
        time.sleep(0.12)   # poli avec SEC (<10 req/s)

    for k in out:
        df[k] = out[k]
    df.to_csv(f, index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        df.to_csv(pub / "thematic_setups.csv", index=False)
    print(f"[CATALYST] {hits}/{len(df)} titres avec un 8-K < {LOOKBACK_DAYS}j")

if __name__ == "__main__":
    main()
