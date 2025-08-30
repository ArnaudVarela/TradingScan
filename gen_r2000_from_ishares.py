# gen_r2000_from_ishares.py
# Génère russell2000.csv en scrapant les tickers depuis le SPDR Russell 2000 (UCITS)
import io
import re
import sys
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

URL_SPDR_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprr-gy.xlsx"
OUTPUT = "russell2000.csv"

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream,*/*",
}

# suffixes d’échange fréquents (Bloomberg/Reuters/etc.)
EXCH_SUFFIX = re.compile(r"(\s+|\.)(US|UN|UW|UQ|UR|N|OQ|K|LN|GY|FP|NA|IM|SM|HK|JP|CN|SW|AU)\b.*", re.IGNORECASE)
# RIC type .OQ .N .K (mais on veut CONSERVER .A/.B des classes d’actions US)
RIC_LONG = re.compile(r"\.[A-Z]{2,4}$")
# Classe d’action US (ex: BRK.B, BF.B) — on garde
CLASS_DOT = re.compile(r"^[A-Z][A-Z0-9]{0,5}\.[A-Z]$")
# ticker simple
TICK_RX = re.compile(r"^[A-Z][A-Z0-9]{0,6}(\.[A-Z])?$")

BAD_VALUES = {"", "ISIN", "UNASSIGNED", "USD", "CASH", "FX", "SWAP", "OPTION", "FUT", "FUTURES", "CURRENCY",
              "N/A", "NA", "NONE", "—", "-", "0"}

def mk_sess():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s

def normalize_candidate(x: str) -> str | None:
    if x is None:
        return None
    v = str(x).strip().upper()
    if not v or v in BAD_VALUES:
        return None

    v = v.replace("\u200b", "")
    v = v.replace(",", " ")
    v = v.replace(" EQUITY", "")
    v = v.replace(" COMMON STOCK", "")
    v = v.replace(" ORD", "")
    v = v.replace(" ADR", "")
    # convertir slash classe → point (BRK/B → BRK.B)
    v = v.replace("/", ".")

    # supprimer suffixes d’échange style " US", " UW", " UQ", " .OQ" etc.
    v = EXCH_SUFFIX.sub("", v).strip()

    # si RIC long ".OQ" etc -> supprime, mais NE PAS supprimer ".B" (classe)
    if RIC_LONG.search(v):
        v = RIC_LONG.sub("", v)

    # enlever espaces restants
    v = re.sub(r"\s+", "", v)

    # certains providers mettent un suffixe pays à 2 lettres collé (ex: AAPLUS)
    m = re.match(r"^([A-Z][A-Z0-9]{0,5})(US|UW|UQ|LN|GY|FP|NA|IM|SM|HK|JP|CN|SW|AU)$", v)
    if m:
        v = m.group(1)

    # Valider le format final
    if TICK_RX.fullmatch(v) or CLASS_DOT.fullmatch(v):
        return v
    return None

def extract_best_tickers(df: pd.DataFrame) -> tuple[str | None, list[str]]:
    best_col = None
    best_vals: list[str] = []
    cols = list(df.columns)

    # on tente d’abord des colonnes probables en priorité
    priority = [c for c in cols if any(k in str(c).lower() for k in ["ticker", "symbol", "ric", "bloomberg", "code"])]
    other = [c for c in cols if c not in priority]
    for col in priority + other:
        ser = df[col].dropna().astype(str)
        candidates = []
        for raw in ser:
            t = normalize_candidate(raw)
            if t:
                candidates.append(t)
        uniq = sorted(set(candidates))
        print(f"    - try col '{col}': {len(uniq)} uniq tickers")
        if len(uniq) > len(best_vals):
            best_vals = uniq
            best_col = str(col)

    return best_col, best_vals

def main():
    print(f"[SPDR] GET {URL_SPDR_XLSX}")
    r = mk_sess().get(URL_SPDR_XLSX, headers=UA, timeout=60)
    if r.status_code != 200 or not r.content:
        print(f"  -> status {r.status_code}")
        sys.exit(2)

    try:
        xls = pd.ExcelFile(io.BytesIO(r.content), engine="openpyxl")
    except Exception as e:
        print(f"  -> read_excel fail: {e}")
        sys.exit(2)

    all_ticks: set[str] = set()
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            print(f"  [skip] {sheet} ({e})")
            continue

        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        print(f"[sheet] {sheet} rows={len(df)} cols={len(df.columns)}")
        # tente de détecter si la 1ère ligne est un faux header (parfois le cas)
        # si la première ligne contient “Name/Ticker/ISIN/...”, on la bascule en header
        head_join = " ".join(str(x) for x in df.iloc[0].values if pd.notna(x)).lower()
        if any(k in head_join for k in ["ticker", "symbol", "isin", "sedol", "name", "security"]):
            # re-définir les colonnes et retirer la 1ère ligne
            df.columns = [str(x).strip() for x in df.iloc[0].values]
            df = df.iloc[1:].reset_index(drop=True)

        col, vals = extract_best_tickers(df)
        print(f"  -> sheet '{sheet}' chose '{col}' with {len(vals)}")
        all_ticks.update(vals)

    total = len(all_ticks)
    print(f"[TOTAL] extracted {total} unique tickers")
    if total < 1000:
        print("❌ Extraction trop faible — structure probablement modifiée. CSV non écrit.")
        sys.exit(2)

    out = pd.DataFrame({"Ticker": sorted(all_ticks)})
    out.to_csv(OUTPUT, index=False)
    print(f"✅ Écrit {OUTPUT} avec {len(out)} tickers.")

if __name__ == "__main__":
    main()
