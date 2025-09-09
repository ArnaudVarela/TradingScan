# analyst_proxy.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np

# On réutilisera les companyfacts chargés via sec_helpers sec_latest_facts(...)
# Pour limiter le trafic, on n'ouvre pas de nouvelles routes ici.
# Le mix le plus stable: Revenue YoY, NetIncome YoY, marge op (approx), EPS TTM.

def _yoy(g: List[float]) -> Optional[float]:
    if g is None or len(g) < 2:
        return None
    try:
        prev, curr = float(g[-2]), float(g[-1])
        if prev == 0:
            return None
        return (curr - prev) / abs(prev)
    except Exception:
        return None

def analyst_label_from_fundamentals(
    revenue_series: List[float] | None,
    netincome_series: List[float] | None,
    opmargin_series: List[float] | None,
    eps_ttm_series: List[float] | None
) -> Tuple[str, float, int]:
    """
    Retourne (label, score[0..1], votes_int)
    - score = moyenne de 4 sous-scores (chaque sous-score ∈ {0, 0.5, 1})
    - votes_int = 10..30 en fonction du label (pour mimer un 'votes' analystes)
    """
    yoy_rev = _yoy(revenue_series or [])
    yoy_ni  = _yoy(netincome_series or [])
    yoy_eps = _yoy(eps_ttm_series or [])
    # marge: on prend la dernière valeur
    m_op = opmargin_series[-1] if opmargin_series else None

    def bucket_yoy(x):
        if x is None: return 0.0
        if x >= 0.15: return 1.0
        if x >= 0.05: return 0.5
        return 0.0

    def bucket_margin(m):
        if m is None: return 0.0
        if m >= 0.15: return 1.0
        if m >= 0.05: return 0.5
        return 0.0

    s_rev = bucket_yoy(yoy_rev)
    s_ni  = bucket_yoy(yoy_ni)
    s_eps = bucket_yoy(yoy_eps)
    s_mrg = bucket_margin(m_op)

    score = float((s_rev + s_ni + s_eps + s_mrg) / 4.0)

    if score >= 0.70:
        label, votes = "STRONG_BUY", 30
    elif score >= 0.55:
        label, votes = "BUY", 25
    elif score >= 0.45:
        label, votes = "HOLD", 20
    else:
        label, votes = "SELL", 15
    return label, score, votes
