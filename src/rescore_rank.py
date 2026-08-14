#!/usr/bin/env python3
"""
rescore_rank.py
===============
Rangiert ein Target neu, ohne irgendetwas neu zu bewerten.

    python3 src/rescore_rank.py RESULTS/BRD4
    python3 src/rescore_rank.py RESULTS/BRD4 --weights vina=0.2,cnnscore=0.5,cnnaffinity=0.3
    python3 src/rescore_rank.py RESULTS/BRD4 --scores vina,cnnscore --out ranking_vina_cnn.csv

Grundlage sind die Zwischenstaende aus dem Blockmodus
(RESULTS/<target>/.rescore_partial/scores_*.csv). Sie enthalten die
Rohscores jeder Pose; Raenge und ECR-Terme haengen vom Gesamtsatz ab und
werden hier neu gerechnet.

Warum das ein eigenes Werkzeug ist: Das Scoring kostet Stunden GPU-Zeit,
das Ranking Sekunden CPU-Zeit. Wer die Konsensgewichte variieren will –
und genau das ist die interessante Frage – soll dafuer nicht die ganze
Maschinerie erneut anwerfen muessen.

Laeuft auf dem HOST, ohne Container: importiert nur ecr.py, das seinerseits
nur die Standardbibliothek braucht. Kein numpy, kein PyTorch, kein gnina.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ecr as ecr_mod  # noqa: E402


class Pose:
    """Leichtgewichtige Pose fuer die Rangrechnung."""
    __slots__ = (["ligand", "pose", "ecr_total"]
                 + [a for k in ecr_mod.ALL_KEYS for a in ecr_mod.FIELDS[k]])

    def __init__(self, ligand: str, pose: int):
        self.ligand = ligand
        self.pose = pose
        self.ecr_total = 0.0
        for key in ecr_mod.ALL_KEYS:
            s_attr, r_attr, e_attr = ecr_mod.FIELDS[key]
            setattr(self, s_attr, None)
            setattr(self, r_attr, None)
            setattr(self, e_attr, 0.0)


def load_partials(target_dir: Path) -> list[Pose]:
    """Liest alle Block-CSVs eines Targets ein."""
    pdir = target_dir / ".rescore_partial"
    if not pdir.is_dir():
        raise FileNotFoundError(
            f"Keine Zwischenstaende in {pdir}.\n"
            f"Sie entstehen nur im Blockmodus – [RESCORE] rescore_block_size "
            f"in rescore.ini auf einen Wert > 0 setzen und das Rescoring "
            f"einmal laufen lassen."
        )

    poses: list[Pose] = []
    files = sorted(pdir.glob("scores_*.csv"))
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                p = Pose(row["ligand"], int(row["pose"]))
                for key in ecr_mod.ALL_KEYS:
                    attr = ecr_mod.FIELDS[key][0]
                    raw = (row.get(attr) or "").strip()
                    if raw:
                        setattr(p, attr, float(raw))
                poses.append(p)
    print(f"  {len(poses):,} Posen aus {len(files)} Blockdateien gelesen")
    return poses


def parse_weights(spec: str) -> dict[str, float]:
    """'vina=0.2,cnnscore=0.5' -> {'vina': 0.2, 'cnnscore': 0.5}"""
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Ungueltiges Gewicht: '{part}' (erwartet key=wert)")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in ecr_mod.FIELDS:
            raise ValueError(
                f"Unbekannte Funktion '{k}'. Erlaubt: "
                f"{', '.join(ecr_mod.ALL_KEYS)}"
            )
        out[k] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ECR neu rechnen aus gespeicherten Rohscores")
    ap.add_argument("target_dir", type=Path,
                    help="RESULTS/<target>")
    ap.add_argument("--scores", default="",
                    help="Aktive Funktionen, kommagetrennt. Ohne Angabe alle, "
                         "fuer die Daten vorliegen.")
    ap.add_argument("--weights", default="",
                    help="Gewichte, z.B. vina=0.2,cnnscore=0.8. Ohne Angabe "
                         "gleichgewichtet. Wird auf Summe 1 normiert.")
    ap.add_argument("--sigma-fraction", type=float, default=4.0,
                    help="sigma = N_Posen / diesem Wert (default 4.0)")
    ap.add_argument("--out", default="",
                    help="Ausgabedatei. Ohne Angabe "
                         "ranking_<target>.csv im Target-Verzeichnis.")
    ap.add_argument("--top", type=int, default=10,
                    help="Wieviele Treffer auf der Konsole (default 10)")
    args = ap.parse_args()

    tdir = args.target_dir
    if not tdir.is_dir():
        print(f"FEHLER: {tdir} ist kein Verzeichnis", file=sys.stderr)
        return 1
    target_name = tdir.name

    try:
        poses = load_partials(tdir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 1

    if not poses:
        print("FEHLER: Keine Posen gelesen.", file=sys.stderr)
        return 1

    # Aktive Funktionen: explizit oder alle mit Daten
    if args.scores:
        active = [s.strip() for s in args.scores.split(",") if s.strip()]
        unknown = [s for s in active if s not in ecr_mod.FIELDS]
        if unknown:
            print(f"FEHLER: Unbekannte Funktion(en): {unknown}", file=sys.stderr)
            return 1
    else:
        active = [k for k in ecr_mod.ALL_KEYS
                  if any(getattr(p, ecr_mod.FIELDS[k][0]) is not None
                         for p in poses)]

    # Funktionen ohne Daten aussortieren – sonst zaehlen sie als 0 mit und
    # verwaessern die Gewichtung der uebrigen.
    with_data = [k for k in active
                 if any(getattr(p, ecr_mod.FIELDS[k][0]) is not None
                        for p in poses)]
    dropped = set(active) - set(with_data)
    if dropped:
        print(f"  Ohne Daten, ausgeschlossen: {', '.join(sorted(dropped))}")
    if not with_data:
        print("FEHLER: Keine der gewaehlten Funktionen hat Daten.",
              file=sys.stderr)
        return 1

    try:
        weight_map = parse_weights(args.weights) if args.weights else None
    except ValueError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 1

    weights = ecr_mod.normalize_weights(with_data, weight_map)
    sigma = max(len(poses) / args.sigma_fraction, 1.0)

    print(f"  Funktionen : {', '.join(with_data)}")
    print(f"  Gewichte   : " +
          ", ".join(f"{k}={v:.3f}" for k, v in weights.items()))
    print(f"  sigma      : {sigma:.2f}  ({len(poses):,} Posen / "
          f"{args.sigma_fraction})")

    ecr_mod.compute_ecr(poses, args.sigma_fraction, with_data, weights)
    ligands = ecr_mod.aggregate_ligands(poses)

    out_path = Path(args.out) if args.out else tdir / f"ranking_{target_name}.csv"
    cols = (["ecr_rank", "ligand", "ecr_score", "best_pose"]
            + [f"score_{k}_best" for k in with_data])
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in ligands:
            w.writerow([r["ecr_rank"], r["ligand"], f"{r['ecr_score']:.6f}",
                        r["best_pose"]]
                       + [("" if r[f"score_{k}_best"] is None
                           else f"{r[f'score_{k}_best']:.3f}")
                          for k in with_data])

    print(f"\n  {len(ligands):,} Liganden -> {out_path}\n")
    print(f"  {'Rang':>5}  {'Ligand':<28}{'ECR':>10}  {'Pose':>5}")
    print("  " + "-" * 52)
    for r in ligands[:args.top]:
        print(f"  {r['ecr_rank']:>5}  {r['ligand']:<28}"
              f"{r['ecr_score']:>10.5f}  {r['best_pose']:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
