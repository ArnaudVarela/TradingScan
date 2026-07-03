# test_pattern_score.py — Tests de propriétés/invariants du score (sans réseau). Branché au CI (fail-fast).
import sys
import numpy as np
import pandas as pd

import pattern_score as P

ok = 0
fail = 0
def check(name, cond, got=None):
    global ok, fail
    if cond:
        ok += 1; print(f"  [OK]  {name}")
    else:
        fail += 1; print(f"  [XX]  {name}  got={got}")

def approx(a, b, tol=1e-3):
    return a is not None and b is not None and abs(a - b) <= tol

def mkdf(closes, vols=None):
    closes = np.asarray(closes, float); n = len(closes)
    idx = pd.bdate_range("2023-01-02", periods=n)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.01
    lows = np.minimum(opens, closes) * 0.99
    if vols is None: vols = np.full(n, 2e6)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": np.asarray(vols, float)}, index=idx)

# ---- Helpers : bornes & NaN-safety ----
print("== helpers _ramp/_window : bornes & NaN-safe ==")
check("_ramp_up borné [0,1]", 0 <= P._ramp_up(0.5, 0, 1) <= 1)
check("_ramp_up(nan)=0 (NaN-safe)", P._ramp_up(float("nan"), 0, 1) == 0.0)
check("_ramp_down(nan)=1", P._ramp_down(float("nan"), 0, 1) == 1.0)
check("_window plateau=1", approx(P._window(0.2, 0.06, 0.12, 0.32, 0.50), 1.0))
check("_window hors zone=0", P._window(0.60, 0.06, 0.12, 0.32, 0.50) == 0.0)
check("_window(nan)=0", P._window(float("nan"), 0, 1, 2, 3) == 0.0)

# ---- Patterns synthétiques : discrimination ----
print("== patterns synthétiques ==")
up = np.linspace(10, 50, 180); cup = 50 - 8*np.sin(np.linspace(0, np.pi, 40))
handle = np.linspace(50, 47, 6); tail = np.linspace(47, 48.5, 24)
c1 = np.concatenate([up, cup, handle, tail])
v1 = np.concatenate([np.full(180, 2e6), np.full(40, 1.5e6), np.full(30, 0.7e6)])
r1 = P.preexplosion_score(mkdf(c1, v1))
r2 = P.preexplosion_score(mkdf(np.linspace(50, 22, 250)))
up3 = np.linspace(10, 45, 230); spike = np.linspace(45, 72, 20)
r3 = P.preexplosion_score(mkdf(np.concatenate([up3, spike])))

check("cup&handle score élevé (>45)", r1["score"] > 45, r1["score"])
check("downtrend score bas (<25)", r2["score"] < 25, r2["score"])
check("pré-explosion > déjà-explosé", r1["score"] > r3["score"], f"{r1['score']} vs {r3['score']}")

# ---- Invariants du score ----
print("== invariants ==")
check("score dans [0,100]", all(0 <= r["score"] <= 100 for r in (r1, r2, r3)))
check("déterministe (même input -> même score)", P.preexplosion_score(mkdf(c1, v1))["score"] == r1["score"])
check("composantes dans [0,1]", all(0 <= v <= 1 for v in r1["components"].values()))
check("historique insuffisant -> None", P.preexplosion_score(mkdf(np.linspace(10, 12, 50))) is None)
check("avg_dollar_vol & n_bars présents", "avg_dollar_vol" in r1["metrics"] and "n_bars" in r1["metrics"])
# EMA200 non factice : un jeune titre (<200 barres) ne doit pas planter et rester borné
ry = P.preexplosion_score(mkdf(np.linspace(10, 20, 170)))
check("jeune titre (<200 barres) scoré sans crash & borné", ry is not None and 0 <= ry["score"] <= 100)

print(f"\nRESULT: {ok} OK / {fail} FAIL")
sys.exit(1 if fail else 0)
