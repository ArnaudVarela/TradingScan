# fetch_fear_greed.py
import os, io, json, time, random, datetime as dt
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry

# --- Sorties (root + public pour compat) ---
ROOT_DIR   = Path(".")
PUBLIC_DIR = Path("dashboard/public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

HIST_CSV   = ROOT_DIR / "fear_greed_history.csv"
OUT_CSV    = ROOT_DIR / "fear_greed.csv"
OUT_JSON   = ROOT_DIR / "fear_greed.json"

# on duplique aussi dans dashboard/public/
PUB_CSV    = PUBLIC_DIR / "fear_greed.csv"
PUB_JSON   = PUBLIC_DIR / "fear_greed.json"

CNN_URL    = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://money.cnn.com/data/fear-and-greed/",
    "Connection": "keep-alive",
}

def _now_utc_iso():
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _bucket_label(score: int) -> str:
    s = int(score)
    if s <= 24:  return "Extreme Fear"
    if s <= 44:  return "Fear"
    if s <= 55:  return "Neutral"
    if s <= 75:  return "Greed"
    return "Extreme Greed"

def _mk_session():
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.8,  # 0.8, 1.6, 2.4...
        status_forcelist=(418, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

def _parse_cnn_payload(j: dict) -> dict:
    """
    Retourne {score:int, asof:'YYYY-MM-DD'} depuis les variantes CNN.
    - Variante A: j['fear_and_greed'] = {'score':.., 'timestamp':..} OU série
    - Variante B: j['fear_and_greed_historical']['data'] = [{x:ms, y:int}, ...]
    - Variante C: j['previous_close'] = {'score':.., 'timestamp':..}
    Lève ValueError si aucune forme exploitable.
    """
    # 1) série 'fear_and_greed' (certaines versions mettent déjà [{x,y},...])
    if isinstance(j.get("fear_and_greed"), list) and j["fear_and_greed"]:
        last = j["fear_and_greed"][-1]
        score = int(last.get("y"))
        ts_ms = int(last.get("x"))
        asof  = dt.datetime.utcfromtimestamp(ts_ms/1000).strftime("%Y-%m-%d")
        return {"score": score, "asof": asof}

    if isinstance(j.get("fear_and_greed"), dict):
        d = j["fear_and_greed"]
        # soit c'est un "point courant"
        if "score" in d and "timestamp" in d:
            score = int(d["score"])
            ts_ms = int(d["timestamp"])
            asof  = dt.datetime.utcfromtimestamp(ts_ms/1000).strftime("%Y-%m-%d")
            return {"score": score, "asof": asof}
        # soit une sous-clé 'data'
        if isinstance(d.get("data"), list) and d["data"]:
            last = d["data"][-1]
            score = int(last.get("y"))
            ts_ms = int(last.get("x"))
            asof  = dt.datetime.utcfromtimestamp(ts_ms/1000).strftime("%Y-%m-%d")
            return {"score": score, "asof": asof}

    # 2) historique 'fear_and_greed_historical'
    hist = j.get("fear_and_greed_historical", {})
    if isinstance(hist, dict) and isinstance(hist.get("data"), list) and hist["data"]:
        last = hist["data"][-1]
        score = int(last.get("y"))
        ts_ms = int(last.get("x"))
        asof  = dt.datetime.utcfromtimestamp(ts_ms/1000).strftime("%Y-%m-%d")
        return {"score": score, "asof": asof}

    # 3) previous_close (fallback vu chez CNN)
    pc = j.get("previous_close")
    if isinstance(pc, dict) and "score" in pc and "timestamp" in pc:
        score = int(pc["score"])
        ts_ms = int(pc["timestamp"])
        asof  = dt.datetime.utcfromtimestamp(ts_ms/1000).strftime("%Y-%m-%d")
        return {"score": score, "asof": asof}

    raise ValueError("CNN payload not understood")

def fetch_cnn() -> dict:
    sess = _mk_session()

    # essai 1: endpoint par défaut
    r = sess.get(CNN_URL, headers=UA_HEADERS, timeout=20)
    if r.ok:
        return _parse_cnn_payload(r.json())

    # essai 2: variante avec start date (améliore parfois la fiabilité)
    start = (dt.datetime.utcnow() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
    r2 = sess.get(f"{CNN_URL}/{start}", headers=UA_HEADERS, timeout=20)
    if r2.ok:
        return _parse_cnn_payload(r2.json())

    # si toujours pas ok -> lève
    r.raise_for_status()
    raise RuntimeError("CNN fetch failed with no JSON")

def compute_streak(hist: pd.DataFrame, cur_label: str) -> int:
    """Nombre de jours consécutifs (fin) avec le même label (incluant aujourd'hui)."""
    if hist.empty:
        return 1
    streak = 0
    for _, row in hist.sort_values("date").iloc[::-1].iterrows():
        if str(row.get("label","")) == cur_label:
            streak += 1
        else:
            break
    return max(1, streak)

def write_outputs(score: int | None, label: str, asof: str | None, streak_days: int):
    # CSV courant (root + public)
    cur_df = pd.DataFrame([{
        "score": score, "label": label, "asof": asof, "generated_at_utc": _now_utc_iso()
    }])
    cur_df.to_csv(OUT_CSV, index=False)
    cur_df.to_csv(PUB_CSV, index=False)

    # JSON (root + public)
    payload = {"score": score, "label": label, "asof": asof, "streak_days": int(streak_days)}
    with open(OUT_JSON, "w") as f: json.dump(payload, f)
    with open(PUB_JSON, "w") as f: json.dump(payload, f)

def main():
    # 1) lire historique existant
    if HIST_CSV.exists():
        hist = pd.read_csv(HIST_CSV)
    else:
        hist = pd.DataFrame(columns=["date","score","label"])

    # 2) tenter CNN
    try:
        cur = fetch_cnn()
        score = int(cur["score"])
        asof  = str(cur["asof"])
        label = _bucket_label(score)

        # maj de l'historique (upsert par date)
        if (hist["date"] == asof).any():
            hist.loc[hist["date"] == asof, ["score","label"]] = [score, label]
        else:
            hist = pd.concat([hist, pd.DataFrame([{"date": asof, "score": score, "label": label}])], ignore_index=True)

        hist = hist.sort_values("date")
        hist.to_csv(HIST_CSV, index=False)

        streak = compute_streak(hist, label)
        write_outputs(score, label, asof, streak)
        print(f"[OK] Fear&Greed {asof}: score={score} ({label}), streak={streak}j")

    except Exception as e:
        print(f"[WARN] CNN fetch failed: {e}")

        # 3) fallback: dernière valeur connue
        if hist.empty:
            write_outputs(None, "N/A", None, 0)
            print("[FALLBACK] no history -> wrote placeholder")
            return

        last = hist.sort_values("date").iloc[-1]
        score = int(last["score"])
        label = str(last["label"])
        asof  = str(last["date"])
        streak = compute_streak(hist, label)
        write_outputs(score, label, asof, streak)
        print(f"[FALLBACK] reused last value {asof}: score={score} ({label}), streak={streak}j")

if __name__ == "__main__":
    main()
