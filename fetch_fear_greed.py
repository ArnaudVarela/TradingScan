# fetch_fear_greed.py
import os, json, datetime as dt, requests, pandas as pd

PUBLIC_DIR = os.path.join("dashboard", "public")
HIST_CSV   = "fear_greed_history.csv"
OUT_JSON   = os.path.join(PUBLIC_DIR, "fear_greed.json")

def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def bucket_label(score: int) -> str:
    s = int(score)
    if s <= 24:  return "Extreme Fear"
    if s <= 44:  return "Fear"
    if s <= 55:  return "Neutral"
    if s <= 75:  return "Greed"
    return "Extreme Greed"

def fetch_cnn() -> dict:
    # Endpoint public (non documenté) utilisé par CNN pour la page
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    series = j.get("fear_and_greed") or j.get("data") or []
    if not series:
        # fallback sur 'previous_close' s'il existe
        score = j.get("previous_close", {}).get("score")
        ts    = j.get("previous_close", {}).get("timestamp")
        if score is None: 
            raise RuntimeError("Pas de données CNN valides")
        return {"score": int(score), "asof": dt.datetime.utcfromtimestamp(int(ts)/1000).strftime("%Y-%m-%d")}
    last = series[-1]
    score = int(last.get("y"))
    ts    = int(last.get("x"))  # ms epoch
    asof  = dt.datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d")
    return {"score": score, "asof": asof}

def compute_streak(hist: pd.DataFrame, cur_label: str) -> int:
    """Nombre de jours consécutifs (fin) avec le même label (incluant aujourd'hui)."""
    if hist.empty: return 1
    streak = 0
    for _, row in hist.sort_values("date").iterrows():
        pass
    # on repart du plus récent
    for _, row in hist.sort_values("date").iloc[::-1].iterrows():
        if row["label"] == cur_label:
            streak += 1
        else:
            break
    return max(1, streak)

def main():
    ensure_dir(OUT_JSON)
    # Lire historique existant (optionnel)
    if os.path.exists(HIST_CSV):
        hist = pd.read_csv(HIST_CSV)
    else:
        hist = pd.DataFrame(columns=["date","score","label"])

    # Récup valeur du jour
    try:
        cur = fetch_cnn()
    except Exception as e:
        # En cas d'échec, réutiliser la dernière si disponible
        if hist.empty:
            # écrire un placeholder neutre pour le front
            with open(OUT_JSON, "w") as f:
                json.dump({"score": None, "label":"N/A", "asof": None, "streak_days": 0}, f)
            return
        last = hist.sort_values("date").iloc[-1]
        payload = {
            "score": int(last["score"]),
            "label": str(last["label"]),
            "asof": str(last["date"]),
            "streak_days": compute_streak(hist, str(last["label"]))
        }
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f)
        return

    cur_label = bucket_label(cur["score"])
    today = cur["asof"]

    # mettre à jour l'historique (si nouvelle date)
    if (hist["date"] == today).any():
        hist.loc[hist["date"] == today, ["score","label"]] = [cur["score"], cur_label]
    else:
        hist = pd.concat([hist, pd.DataFrame([{"date": today, "score": cur["score"], "label": cur_label}])], ignore_index=True)

    # streak
    streak = compute_streak(hist, cur_label)

    # Sauvegardes
    hist.sort_values("date").to_csv(HIST_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump({"score": cur["score"], "label": cur_label, "asof": today, "streak_days": int(streak)}, f)

if __name__ == "__main__":
    main()
