# news_fetch.py — Phase 2 overlay news : CHALEUR SECTORIELLE (Google News RSS, sans clé)
# + BUZZ société (Finnhub, si FINNHUB_API_KEY). Colonnes/fichiers SÉPARÉS, jamais dans le /100.
# Best-effort et non bloquant (si une source échoue, on continue).
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ROOT = Path(__file__).parent
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
BUZZ_DAYS = int(os.getenv("BUZZ_DAYS", "7"))
BUZZ_MIN_SCORE = float(os.getenv("BUZZ_MIN_SCORE", "55"))   # buzz seulement sur les meilleurs setups (quota)
UA = {"User-Agent": "Mozilla/5.0 (compatible; TradingScan/1.0; research)"}

THEME_QUERIES = {
    "semiconducteurs": "semiconductor chip stocks",
    "quantique": "quantum computing stocks",
    "ia": "artificial intelligence AI stocks",
    "laser_photonique": "photonics laser lidar stocks",
    "robotique": "robotics automation stocks",
    "equipement_electrique": "electrical grid power equipment stocks",
    "production_energie": "nuclear uranium energy power stocks",
    "gestion_thermique": "data center cooling thermal management stocks",
    "espace": "space satellite rocket stocks",
    "defense": "defense military drone stocks",
    "new_tech": "emerging deep tech stocks",
}

# Chaleur sur les 11 secteurs GICS (libellés EXACTS de sector_perf.py -> jointure directe avec la Rotation).
GICS_QUERIES = {
    "Information Technology": "technology software semiconductor stocks",
    "Health Care": "healthcare biotech pharmaceutical stocks",
    "Financials": "bank financial insurance stocks",
    "Industrials": "industrial manufacturing aerospace defense stocks",
    "Consumer Discretionary": "consumer retail discretionary stocks",
    "Consumer Staples": "consumer staples food beverage stocks",
    "Energy": "energy oil gas uranium stocks",
    "Utilities": "utilities power electricity stocks",
    "Materials": "materials mining chemicals stocks",
    "Real Estate": "real estate REIT stocks",
    "Communication Services": "media telecom communication stocks",
}

def _gn_count(q):
    """Nombre d'articles Google News (48h) + 1er titre. ('', '') si échec (best-effort)."""
    try:
        url = ("https://news.google.com/rss/search?q="
               + requests.utils.quote(q + " when:2d") + "&hl=en-US&gl=US&ceid=US:en")
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        items = ET.fromstring(r.content).findall(".//item")
        return len(items), ((items[0].findtext("title") or "")[:150] if items else "")
    except Exception:
        return "", ""

def _write_heat(rows, key_col, fname, tag):
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / fname, index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        df.to_csv(pub / fname, index=False)
    tot = sum(x for x in df["heat"] if isinstance(x, (int, float)))
    print(f"[NEWS] {tag} : {len(df)} {key_col}s, {tot} articles (2j)")

def sector_heat():
    rows = []
    for key, q in THEME_QUERIES.items():
        n, top = _gn_count(q)
        rows.append({"theme": key, "heat": n, "top_headline": top})
        time.sleep(0.4)
    _write_heat(rows, "thème", "sector_heat.csv", "chaleur thèmes")

def gics_heat():
    rows = []
    for sector, q in GICS_QUERIES.items():
        n, top = _gn_count(q)
        rows.append({"sector": sector, "heat": n, "top_headline": top})
        time.sleep(0.4)
    _write_heat(rows, "secteur", "sector_heat_gics.csv", "chaleur GICS")

def finnhub_buzz():
    if not FINNHUB_KEY:
        print("[NEWS] FINNHUB_API_KEY absent -> buzz société ignoré (chaleur sectorielle OK)")
        return
    f = ROOT / "thematic_setups.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    frm = (datetime.now(timezone.utc).date() - timedelta(days=BUZZ_DAYS)).isoformat()
    to = datetime.now(timezone.utc).date().isoformat()
    buzz = []
    sess = requests.Session()
    fetched = 0
    for i, row in df.iterrows():
        t = str(row["ticker"]).upper()
        b = ""
        if float(row.get("score", 0) or 0) >= BUZZ_MIN_SCORE:
            try:
                r = sess.get(f"https://finnhub.io/api/v1/company-news?symbol={t}&from={frm}&to={to}&token={FINNHUB_KEY}", timeout=15)
                if r.ok:
                    b = len(r.json())
                fetched += 1
                time.sleep(1.05)   # respect 60 req/min
            except Exception:
                pass
        buzz.append(b)
    df["buzz"] = buzz
    df.to_csv(f, index=False)
    pub = ROOT / "dashboard" / "public"
    if pub.exists():
        df.to_csv(pub / "thematic_setups.csv", index=False)
    nz = int((pd.to_numeric(df["buzz"], errors="coerce") > 0).sum())
    print(f"[NEWS] buzz Finnhub : {fetched} titres interrogés, {nz} avec news récentes")

def main():
    try:
        sector_heat()
    except Exception as e:
        print(f"[NEWS] chaleur thèmes KO ({e})")
    try:
        gics_heat()
    except Exception as e:
        print(f"[NEWS] chaleur GICS KO ({e})")
    try:
        finnhub_buzz()
    except Exception as e:
        print(f"[NEWS] buzz société KO ({e})")

if __name__ == "__main__":
    main()
