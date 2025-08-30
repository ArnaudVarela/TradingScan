# gen_r2000_merge.py
import io, re, sys, requests
import pandas as pd
from requests.adapters import HTTPAdapter, Retry

URL_SPDR = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"
URL_SCHA = "https://www.schwabassetmanagement.com/allholdings/SCHA"
OUTPUT = "russell2000.csv"

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Referer": "https://www.schwabassetmanagement.com/"
}

EXCH_SUFFIX = re.compile(r"(\s+|\.)(US|UN|UW|UQ|UR|N|OQ|K|LN|GY|FP|NA|IM|SM|HK|JP|CN|SW|AU)\b.*", re.IGNORECASE)
RIC_LONG = re.compile(r"\.[A-Z]{2,4}$")
CLASS_DOT = re.compile(r"^[A-Z][A-Z0-9]{0,5}\.[A-Z]$")
TICK_RX = re.compile(r"^[A-Z][A-Z0-9]{0,7}(\.[A-Z])?$")

BAD_VALUES = {"", "ISIN", "USD", "CASH", "FX", "SWAP", "OPTION", "FUT", "FUTURES", "N/A", "NA", "NONE", "-", "—"}

def mk_sess():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def normalize_candidate(x: str) -> str|None:
    if not isinstance(x, str):
        return None
    v = x.strip().upper()
    if not v or v in BAD_VALUES:
        return None
    v = v.replace("/", ".").replace(" EQUITY","")
    v = EXCH_SUFFIX.sub("", v).strip()
    if RIC_LONG.search(v): v = RIC_LONG.sub("", v)
    v = re.sub(r"\s+", "", v)
    if TICK_RX.fullmatch(v) or CLASS_DOT.fullmatch(v):
        return v
    return None

def parse_spdr():
    sess = mk_sess()
    r = sess.get(URL_SPDR, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content), engine="openpyxl")
    ticks = set()
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        for col in df.columns:
            ser = df[col].dropna().astype(str)
            for raw in ser:
                t = normalize_candidate(raw)
                if t: ticks.add(t)
    print(f"[SPDR] {len(ticks)} tickers")
    return ticks

def parse_scha():
    sess = mk_sess()
    r = sess.get(URL_SCHA, headers=UA, timeout=60)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    ticks = set()
    for t in tables:
        for col in t.columns:
            ser = t[col].dropna().astype(str)
            for raw in ser:
                tck = normalize_candidate(raw)
                if tck: ticks.add(tck)
    print(f"[SCHA] {len(ticks)} tickers")
    return ticks

def main():
    spdr = parse_spdr()
try:
    scha = parse_scha()
except Exception as e:
    print(f"[WARN] SCHA fetch failed: {e}")
    scha = []
    union = sorted(spdr.union(scha))
    print(f"[MERGED] total unique tickers: {len(union)}")

    if len(union) < 1000:
        print("❌ Extraction trop faible — structure modifiée")
        sys.exit(2)

    pd.DataFrame({"Ticker": union}).to_csv(OUTPUT, index=False)
    print(f"✅ {OUTPUT} écrit avec {len(union)} tickers")

if __name__ == "__main__":
    main()
