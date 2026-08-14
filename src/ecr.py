"""
ecr.py
======
Exponential Consensus Ranking – abhaengigkeitsfrei.

Importiert NUR aus der Standardbibliothek. Damit laesst sich das Ranking
sowohl im Rescoring-Container (aus docking_rescore.py) als auch auf dem
Host ausfuehren (aus rescore_rank.py), ohne numpy, joblib, torch oder
gninatorch zu benoetigen.

Das ist der Grund fuer dieses Modul: die eigentliche Konsensrechnung ist
Sortieren und exp(). Sie an einen Container zu binden, der mehrere
Gigabyte PyTorch mitschleppt, haette bedeutet, dass ein Neuranken mit
anderen Gewichten denselben Apparat braucht wie das Scoring selbst.

Verfahren (Palacio-Rodriguez et al., Sci Rep 9, 5142, 2019):

    Alle Scores sind richtungskorrigiert (kleiner = besser).
    Pro Funktion j:  Rang r_j durch aufsteigende Sortierung
                     ECR_j = exp(-r_j / sigma),  sigma = N / sigma_fraction
    P(pose)   = Summe_j  w_j * ECR_j
    P(ligand) = max ueber die Posen des Liganden

Die Funktionen arbeiten ueber getattr/setattr auf beliebigen Objekten mit
den passenden Attributnamen. Sie funktionieren daher unveraendert mit den
PoseResult-Dataclasses aus docking_rescore.py wie mit den leichtgewichtigen
Zeilenobjekten, die rescore_rank.py aus den CSVs baut.
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------
# ECR-Key -> (Score-Attribut, Rang-Attribut, ECR-Attribut)
# ----------------------------------------------------------------------
FIELDS: dict[str, tuple[str, str, str]] = {
    "vina":              ("score_vina",              "rank_vina",              "ecr_vina"),
    "vinardo":           ("score_vinardo",           "rank_vinardo",           "ecr_vinardo"),
    "ad4":               ("score_ad4",               "rank_ad4",               "ecr_ad4"),
    "cnnaffinity":       ("score_cnnaffinity",       "rank_cnnaffinity",       "ecr_cnnaffinity"),
    "cnnscore":          ("score_cnnscore",          "rank_cnnscore",          "ecr_cnnscore"),
    "deltalinf9xgb":     ("score_deltalinf9xgb",     "rank_deltalinf9xgb",     "ecr_deltalinf9xgb"),
    "dense_cnnaffinity": ("score_dense_cnnaffinity", "rank_dense_cnnaffinity", "ecr_dense_cnnaffinity"),
    "dense_cnnscore":    ("score_dense_cnnscore",    "rank_dense_cnnscore",    "ecr_dense_cnnscore"),
}

# Funktionen, bei denen groesser besser ist und die beim Einlesen aus
# Rohdaten invertiert werden muessen. Innerhalb dieses Moduls sind die
# Werte bereits korrigiert – die Liste dient rescore_rank.py als Referenz.
LARGER_IS_BETTER = (
    "cnnaffinity", "cnnscore", "deltalinf9xgb",
    "dense_cnnaffinity", "dense_cnnscore",
)

ALL_KEYS = tuple(FIELDS)


def normalize_weights(active_scores, weight_map=None) -> dict[str, float]:
    """
    Gewichte auf Summe 1 normieren – ueber die AKTIVEN Funktionen.

    weight_map=None oder alle Werte 0 -> Gleichgewichtung 1/K.
    Das Normieren nur ueber die aktiven Funktionen sorgt dafuer, dass das
    Abschalten einer Funktion die uebrigen nicht still umgewichtet.
    """
    active = [k for k in active_scores if k in FIELDS]
    if not active:
        return {}
    if not weight_map:
        return {k: 1.0 / len(active) for k in active}

    vals = {k: max(0.0, float(weight_map.get(k, 0.0))) for k in active}
    total = sum(vals.values())
    if total <= 0.0:
        return {k: 1.0 / len(active) for k in active}
    return {k: v / total for k, v in vals.items()}


def compute_ecr(poses, sigma_fraction: float, active_scores,
                weights=None):
    """
    Berechnet Raenge, ECR-Terme und ecr_total fuer jede Pose.

    Veraendert die uebergebenen Objekte in place und gibt sie zurueck.
    Posen ohne Wert fuer eine Funktion erhalten dort keinen Rang und
    tragen 0 zu diesem Term bei – das benachteiligt sie, statt sie
    neutral zu behandeln.
    """
    n = len(poses)
    if n == 0:
        return poses

    sigma = max(n / sigma_fraction, 1.0)
    active = [k for k in active_scores if k in FIELDS]

    if weights is None:
        weights = normalize_weights(active)

    for key in active:
        s_attr, r_attr, e_attr = FIELDS[key]
        valid = sorted(
            ((i, getattr(poses[i], s_attr, None)) for i in range(n)
             if getattr(poses[i], s_attr, None) is not None),
            key=lambda x: x[1],          # kleiner = besser -> Rang 1
        )
        for rank0, (idx, _) in enumerate(valid):
            rank = rank0 + 1
            setattr(poses[idx], r_attr, rank)
            setattr(poses[idx], e_attr, math.exp(-rank / sigma))

    for pose in poses:
        pose.ecr_total = sum(
            weights.get(k, 0.0) * getattr(pose, FIELDS[k][2], 0.0)
            for k in active
        )

    return poses


def aggregate_ligands(poses, make_result=None):
    """
    P(ligand) = max(ecr_total) ueber die Posen des Liganden.

    make_result: optionale Fabrik (name, best_pose_objekt, ecr) -> Objekt.
    Ohne sie werden schlichte Dicts zurueckgegeben – so bleibt das Modul
    frei von den Dataclasses aus docking_rescore.py.

    Rueckgabe: absteigend nach ecr_score sortiert, Rang ab 1 gesetzt.
    """
    by_lig: dict = {}
    for p in poses:
        by_lig.setdefault(p.ligand, []).append(p)

    results = []
    for name, ps in by_lig.items():
        best = max(ps, key=lambda p: getattr(p, "ecr_total", 0.0))
        if make_result is not None:
            results.append(make_result(name, best, best.ecr_total))
        else:
            row = {"ligand": name, "best_pose": best.pose,
                   "ecr_score": best.ecr_total}
            for key in ALL_KEYS:
                row[f"score_{key}_best"] = getattr(best, FIELDS[key][0], None)
            results.append(row)

    def _score(r):
        return r["ecr_score"] if isinstance(r, dict) else r.ecr_score

    results.sort(key=_score, reverse=True)
    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            r["ecr_rank"] = i
        else:
            r.ecr_rank = i
    return results
