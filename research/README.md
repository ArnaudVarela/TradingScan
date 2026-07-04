# research/ — Harness de l'audit stratégie (reproductibilité)

Scripts one-shot qui ont produit les conclusions de l'audit du screener. À lancer depuis la RACINE
du repo (`python research/<script>.py`) ; un shim ajoute la racine au PYTHONPATH.
Les sorties (feature_matrix.csv, structure_matrix.csv, ...) sont régénérables et gitignorées.
Le cache OHLC (pickles) va dans le scratchpad (env SCRATCH).

## Conclusions mesurées (dans l'ordre)
1. **feature_scan.py** — quelles features prédisent un gain >=20% (20j) ? Réponse : ~aucune
   (tous rho < 0,07). Prix bas / volatilité = plus de gros mouvements mais SYMÉTRIQUE (risque, pas alpha).
2. **structure_scan.py** — horizon 5j + structure "fallen angel" (défoncé + base + volume + retest).
   La structure a une queue haute plus grasse MAIS un winrate < 50% et une médiane négative.
3. **backtest_exits.py** — fallen angel + SORTIES ASYMÉTRIQUES (stop/trailing) sur le thématique.
   Les sorties transforment un buy&hold plat en PF ~2,0 -> l'edge est dans la GESTION, pas la sélection.
4. **validate_market_fallenangel.py** — même test sur le MARCHÉ <$5B (anti-survivorship/curation).
   L'edge survit mais fond : PF ~1,3 net de coûts réalistes (50bps). Modeste mais réel.
5. **backtest_entry.py** — raffinements d'entrée (#7 confirmation cassure, #6 volume-pop).
   La confirmation DÉGRADE (PF 1,19 -> 0,98, on chasse le breakout) ; le volume-pop est du bruit.
   -> l'entrée "à la structure" est déjà l'optimum.

## Verdict
La sélection/le timing d'entrée ne portent pas d'edge exploitable en swing sur cet univers.
Le seul edge mesuré (PF ~1,3) vient des SORTIES asymétriques appliquées aux candidats fallen-angel.
D'où le module `fallen_angel.py` (production) : générateur de candidats haute-variance, paper-first.
Note : une variante `preexplosion_score_v2` (score "plus contraction") a été testée et REJETÉE
(capture de queue de droite pire que v1) -> non conservée.
