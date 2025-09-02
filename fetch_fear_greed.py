# fetch_fear_greed.py
import os, json, datetime as dt, requests, pandas as pd

PUBLIC_DIR = "dashboard/public"
ROOT_SNAP  = "fear_greed.csv"          # snapshot du jour (RAW GitHub)
HIST_CSV   = "fear_greed_history.csv"  # historique (append/upsert)
ROOT_JSON  = "fear_greed.json"         # snapshot JSON root (copie aussi en public)
OUT_JSON   = os.path.join(PUBLIC_DIR, "fear_greed.json")

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def bucket_label(score: int) -> str:
    s = int(score)
    if s <= 24:  return "Extreme Fear"
    if s <= 44:  return "Fear"
    if s <= 55:  return "Neutral"
    if s <= 75:  return "Greed"
    return "Extreme Greed"

def _ts_to_ymd(ts_ms: int) -> str:
    return dt.datetime.utcfromtimestamp(int(ts_ms)/1000).strftime("%Y-%m-%d")

def fetch_cnn() -> dict:
    import time
    from requests.adapters import HTTPAdapter, Retry

    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Referer": "https://edition.cnn.com/markets/fear-and-greed",
        "Accept": "application/json,text/plain,*/*",
    })
    retry = Retry(total=2, backoff_factor=0.6, status_forcelist=[418,429,500,502,503,504])
    sess.mount("https://", HTTPAdapter(max_retries=retry))

    # 2 tentatives "propres" + 2 manuelles si 418 persiste
    for attempt in range(4):
        r = sess.get(url, timeout=20)
        if r.status_code == 418 and attempt < 3:
            time.sleep(1.2 * (attempt + 1))
            continue
        r.raise_for_status()
        j = r.json()

        # Plusieurs formats possibles dans le temps
        series = None
        if isinstance(j.get("fear_and_greed"), list):
            series = j["fear_and_greed"]
        elif isinstance(j.get("data"), list):
            series = j["data"]
        elif isinstance(j.get("fear_and_greed"), dict) and isinstance(j["fear_and_greed"].get("data"), list):
            series = j["fear_and_greed"]["data"]

        if series:
            last = series[-1]
            score = last.get("y") if "y" in last else last.get("score")
            ts    = last.get("x") if "x" in last else last.get("timestamp")
            if score is not None and ts is not None:
                asof = dt.datetime.utcfromtimestamp(int(ts)/1000).strftime("%Y-%m-%d")
                return {"score": int(score), "asof": asof}

        prev = j.get("previous_close") or {}
        if "score" in prev and "timestamp" in prev:
            asof = dt.datetime.utcfromtimestamp(int(prev["timestamp"])/1000).strftime("%Y-%m-%d")
            return {"score": int(prev["score"]), "asof": asof}

        # Si format non reconnu, on retente la boucle
        time.sleep(0.8)

    raise RuntimeError("CNN payload not understood after retries")

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

def write_root_and_public_json(payload: dict):
    with open(ROOT_JSON, "w") as f:
        json.dump(payload, f)
    ensure_dir(OUT_JSON)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f)

def main():
    ensure_dir(OUT_JSON)

    # Lire historique existant
    if os.path.exists(HIST_CSV):
        try:
            hist = pd.read_csv(HIST_CSV)
        except Exception:
            hist = pd.DataFrame(columns=["date","score","label"])
    else:
        hist = pd.DataFrame(columns=["date","score","label"])

    # Fetch du jour
    try:
        cur = fetch_cnn()
        print(f"[FG] fetched: score={cur['score']} asof={cur['asof']}")
    except Exception as e:
        print(f"[FG][WARN] fetch failed: {e}")
        if hist.empty:
            # placeholder neutre
            snap = pd.DataFrame([{
                "date": None, "score": None, "label": "N/A",
                "generated_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "streak_days": 0
            }])
            snap.to_csv(ROOT_SNAP, index=False)
            write_root_and_public_json({
                "score": None, "label":"N/A", "asof": None,
                "streak_days": 0,
                "generated_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            })
            print("[FG] wrote placeholders (no history).")
            return
        last = hist.sort_values("date").iloc[-1]
        payload = {
            "score": int(last["score"]),
            "label": str(last["label"]),
            "asof": str(last["date"]),
            "streak_days": compute_streak(hist, str(last["label"])),
            "generated_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        # snapshot CSV root (reutilise la dernière)
        snap = pd.DataFrame([{
            "date": payload["asof"],
            "score": payload["score"],
            "label": payload["label"],
            "streak_days": payload["streak_days"],
            "generated_at_utc": payload["generated_at_utc"],
        }])
        snap.to_csv(ROOT_SNAP, index=False)
        write_root_and_public_json(payload)
        print("[FG] reused last history entry due to fetch error.")
        return

    # Mise à jour label + history (UPSERT du jour)
    cur_label = bucket_label(cur["score"])
    today = cur["asof"]

    if (hist["date"] == today).any():
        hist.loc[hist["date"] == today, ["score","label"]] = [cur["score"], cur_label]
    else:
        hist = pd.concat([hist, pd.DataFrame([{"date": today, "score": cur["score"], "label": cur_label}])], ignore_index=True)

    hist = hist.sort_values("date")
    streak = compute_streak(hist, cur_label)

    # Écrit snapshot CSV root
    snap = pd.DataFrame([{
        "date": today,
        "score": cur["score"],
        "label": cur_label,
        "streak_days": int(streak),
        "generated_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }])
    snap.to_csv(ROOT_SNAP, index=False)

    # Écrit history
    hist.to_csv(HIST_CSV, index=False)

    # JSON root + public
    payload = {
        "score": cur["score"],
        "label": cur_label,
        "asof": today,
        "streak_days": int(streak),
        "generated_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_root_and_public_json(payload)

    print(f"[FG] wrote: {ROOT_SNAP} (snap) + {HIST_CSV} (history) + JSON (root/public)")

if __name__ == "__main__":
    main()
