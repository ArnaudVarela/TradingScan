# fallen_angel.py — Détection de la structure "fallen angel" VALIDÉE par backtest.
# (Audit : sur l'univers marché <$5B, 1100+ trades, PF ~1,3 net de coûts réalistes 50bps — edge modeste mais réel.)
#
# Structure = titre DÉFONCÉ depuis un pic pluriannuel + base 6M TENDUE + VOLUME en croissance + RE-TEST de résistance.
# Ce n'est PAS un prédicteur d'explosion : c'est une liste de candidats HAUTE-VARIANCE à trader avec un STOP défini
# (l'edge vient de la gestion du risque, pas de la sélection seule). Paper-first.
import numpy as np
import pandas as pd

MIN_BARS = 300   # ~15 mois : besoin d'un vrai contexte pluriannuel pour un "fallen angel"

def detect(df):
    """Retourne un dict de métriques si le titre est un candidat fallen-angel (niveau strict/relax), sinon None."""
    if df is None or len(df) < MIN_BARS or not {"high", "low", "close", "volume"}.issubset(df.columns):
        return None
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    n = len(c)
    last = float(c.iloc[-1])
    if not np.isfinite(last) or last <= 0:
        return None
    roll_hi = float(h.iloc[-756:].max()) if n >= 756 else float(h.max())   # pic ~3 ans (ou dispo)
    if not np.isfinite(roll_hi) or roll_hi <= 0:
        return None
    dd = 1.0 - last / roll_hi                                              # défoncé depuis le pic
    # NaN dans une fenêtre -> None (même sémantique que rolling(min_periods=window).fillna(False) du backtest,
    # sinon detect() pourrait sortir des candidats absents de la population backtestée -> invalide le PF~1,3).
    h6, l6 = h.iloc[-126:], l.iloc[-126:]                                  # base 6 mois
    if h6.isna().any() or l6.isna().any():
        return None
    bh6, bl6 = float(h6.max()), float(l6.min())
    if not np.isfinite(bh6) or bh6 <= 0:
        return None
    base_range = (bh6 - bl6) / last                                        # largeur de base (bas = tendue)
    dist_resist = (bh6 - last) / bh6                                       # distance au re-test (bas = proche)
    if n <= 252:
        return None
    vr, vp = v.iloc[-63:], v.iloc[-252:-63]                                # volume récent vs base (189 barres)
    if vr.isna().any() or vp.isna().any():
        return None
    vprior = float(vp.mean())
    if not (vprior > 0):
        return None
    vgrow = float(vr.mean()) / vprior

    relax = dd >= 0.50 and base_range <= 0.70 and vgrow >= 1.05 and dist_resist <= 0.15
    if not relax:
        return None
    strict = dd >= 0.60 and base_range <= 0.60 and vgrow >= 1.15 and dist_resist <= 0.12
    return {
        "level": "strict" if strict else "relax",
        "dd_from_hi_pct": round(dd * 100, 1),
        "base_range_pct": round(base_range * 100, 1),
        "vol_growth_pct": round((vgrow - 1.0) * 100, 0),
        "dist_resist_pct": round(dist_resist * 100, 1),
        "resistance": round(bh6, 2),
        "stop_ref": round(last * 0.90, 2),   # stop initial de référence -10% (l'edge = risque défini)
    }
