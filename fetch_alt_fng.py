# fetch_alt_fng.py
import json, os, pandas as pd, requests
from datetime import datetime, timezone

HIST = "fear_greed_history.csv"
OUT  = "dashboard/public/fear_greed.json"

def fetch():
    r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=15)
    r.raise_for_status()
    d = r.json().get("data", [{}])[0]
    ts = int(d.get("timestamp", 0))
    return {
        "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
        "score": int(d.get("value", 0)),
        "label": str(d.get("value_classification", "")),
    }

def streak_days(df, label):
    s = 0
    for _, row in df.sort_values("date").iloc[::-1].iterrows():
        if row["label"] == label: s += 1
        else: break
    return s

def main():
    cur = fetch()
    if os.path.exists(HIST):
        hist = pd.read_csv(HIST)
    else:
        hist = pd.DataFrame(columns=["date","score","label"])

    if (hist["date"] == cur["date"]).any():
        hist.loc[hist["date"] == cur["date"], ["score","label"]] = [cur["score"], cur["label"]]
    else:
        hist = pd.concat([hist, pd.DataFrame([cur])], ignore_index=True)

    hist.sort_values("date").to_csv(HIST, index=False)

    payload = dict(cur)
    payload["streak_days"] = streak_days(hist, cur["label"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f)

if __name__ == "__main__":
    main()
