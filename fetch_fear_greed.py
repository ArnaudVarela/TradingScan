# fetch_fear_greed.py
import os, json, datetime as dt, requests, pandas as pd, csv

HIST_CSV   = "fear_greed_history.csv"   # historique complet
OUT_JSON   = "fear_greed.json"          # snapshot JSON
OUT_CSV    = "fear_greed.csv"           # snapshot CSV (une seule ligne)

def bucket_label(score: int) -> str:
    s = int(score)
    if s <= 24:  return "Extreme Fear"
    if s <= 44:  return "Fear"
    if s <= 55:  return "Neutral"
    if s <= 75:  return "Greed"
    return "Extreme Greed"

def fetch_cnn() -> dict:
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {"User-Agent": "Mozilla/5.0 (TradingScan/CI)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    series = j.get("fear_and_greed") or j.get("data") or []
    if not series:
        score = j.get("previous_close", {}).get("score")
        ts    = j.get("previous_close", {}).get("timestamp")
        if score is None:
            raise RuntimeError("Pas de données CNN valides")
        return {"score": int(score), "asof": dt.datetime.utcfromtimestamp(int(ts)/1000).strftime("%Y-%m-%d")}
    last = series[-1]
    score = int(last.get("y"))
    ts    = int(last.get("x"))
    asof  = dt.datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d")
    return {"score": score, "asof": asof}

def compute_streak(hist: pd.DataFrame, cur_label: str) -> int:
    if hist.empty: return 1
    streak = 0
    for _, row in hist.sort_values("date").iloc[::-1].iterrows():
        if row["label"] == cur_label:
            streak += 1
        else:
            break
    return max(1, streak)

def main():
    if os.path.exists(HIST_CSV):
        hist = pd.read_csv(HIST_CSV)
    else:
        hist = pd.DataFrame(columns=["date","score","label"])

    try:
        cur = fetch_cnn()
    except Exception:
        if hist.empty:
            payload = {"score": None, "label":"N/A", "asof": None, "streak_days": 0}
        else:
            last = hist.sort_values("date").iloc[-1]
            cur_label = str(last["label"])
            streak = compute_streak(hist, cur_label)
            payload = {
                "score": int(last["score"]),
                "label": cur_label,
                "asof": str(last["date"]),
                "streak_days": streak
            }
        # Sauvegardes minimalistes
        with open(OUT_JSON, "w") as f: json.dump(payload, f)
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(["date","score","label","streak_days"])
            if payload["asof"]:
                writer.writerow([payload["asof"], payload["score"], payload["label"], payload["streak_days"]])
        return

    cur_label = bucket_label(cur["score"])
    today = cur["asof"]

    if (hist["date"] == today).any():
        hist.loc[hist["date"] == today, ["score","label"]] = [cur["score"], cur_label]
    else:
        hist = pd.concat([hist, pd.DataFrame([{"date": today, "score": cur["score"], "label": cur_label}])], ignore_index=True)

    streak = compute_streak(hist, cur_label)

    hist.sort_values("date").to_csv(HIST_CSV, index=False)

    payload = {"score": cur["score"], "label": cur_label, "asof": today, "streak_days": int(streak)}
    with open(OUT_JSON, "w") as f: json.dump(payload, f)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["date","score","label","streak_days"])
        writer.writerow([today, cur["score"], cur_label, streak])

if __name__ == "__main__":
    main()
