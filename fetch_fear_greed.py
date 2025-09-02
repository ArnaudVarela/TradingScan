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
    """
    Endpoint public utilisé par CNN:
    https://production.dataviz.cnn.io/index/fearandgreed/graphdata
      - format 1: {"fear_and_greed":[{"x": 1717286400000, "y": 64}, ...]}
      - format 2: {"previous_close":{"score": 70, "timestamp": 1717286400000}}
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()

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
            return {"score": int(score), "asof": _ts_to_ymd(ts)}

    # fallback
    prev = j.get("previous_close") or {}
    if "score" in prev and "timestamp" in prev:
        return {"score": int(prev["score"]), "asof": _ts_to_ymd(prev["timestamp"])}

    raise RuntimeError("CNN payload not understood")

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
