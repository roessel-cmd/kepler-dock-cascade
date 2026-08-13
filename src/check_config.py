#!/usr/bin/env python3
"""
check_config.py
===============
Prueft die drei Stufen-INIs auf Konsistenz. Laeuft auf dem HOST, nicht
im Container – vor dem Start der Pipeline.

    python3 check_config.py config/

Der Preis der Modularisierung ist, dass [PATHS] in mehreren Dateien
steht. Die Uebergabepunkte zwischen den Stufen muessen aber exakt
uebereinstimmen:

    docking.ini  results_dir  ==  rescore.ini  results_dir
    docking.ini  target_dir   ==  rescore.ini  target_dir

Stufe 1 (sdf_to_pdbqt.sif) hat keine INI – sie wird per CLI
parametrisiert. Ihr --out-dir muss dem pdbqt_dir aus docking.ini
entsprechen; das prueft check_ligands.py.

Genau das prueft dieses Skript – plus ein paar Plausibilitaeten, die
sonst erst nach Stunden Laufzeit auffallen.
"""

from __future__ import annotations

import configparser
import sys
from pathlib import Path

STAGES = {
    "docking": "docking.ini",
    "rescore": "rescore.ini",
}

# (Stufe A, Stufe B, Key) – muessen identisch sein
SHARED = [
    ("docking", "rescore", "results_dir"),
    ("docking", "rescore", "target_dir"),
]


def load(path: Path) -> configparser.ConfigParser:
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read(path, encoding="utf-8")
    return p


def main() -> int:
    cfg_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "config")
    errors:   list[str] = []
    warnings: list[str] = []

    parsers = {}
    for stage, fname in STAGES.items():
        path = cfg_dir / fname
        if not path.exists():
            errors.append(f"FEHLT: {path}")
            continue
        parsers[stage] = load(path)

    if errors:
        for e in errors:
            print(f"  FEHLER  {e}")
        return 1

    # ── Uebergabepunkte ───────────────────────────────────────────────
    for a, b, key in SHARED:
        va = parsers[a].get("PATHS", key, fallback=None)
        vb = parsers[b].get("PATHS", key, fallback=None)
        if va is None or vb is None:
            errors.append(f"[PATHS] {key} fehlt in {a}.ini oder {b}.ini")
        elif va != vb:
            errors.append(
                f"[PATHS] {key} weicht ab: {a}.ini='{va}' vs {b}.ini='{vb}'"
            )
        else:
            print(f"  OK      {key}: {va}  ({a} → {b})")

    # ── Uni-Dock Plausibilitaet ───────────────────────────────────────
    d = parsers["docking"]
    mode = d.get("UNIDOCK", "search_mode", fallback="balance").strip()
    if mode and mode not in ("fast", "balance", "detail"):
        errors.append(
            f"[UNIDOCK] search_mode='{mode}' ungueltig "
            f"(fast|balance|detail oder leer)"
        )
    scoring = d.get("UNIDOCK", "scoring", fallback="vina").strip()
    if scoring not in ("vina", "vinardo", "ad4"):
        errors.append(f"[UNIDOCK] scoring='{scoring}' ungueltig")
    if scoring == "ad4":
        warnings.append(
            "[UNIDOCK] scoring=ad4 erwartet vorberechnete AutoGrid-Maps "
            "statt --receptor"
        )

    chunk = d.getint("CHUNK", "chunk_size", fallback=5000)
    batch = d.getint("UNIDOCK", "batch_size", fallback=1000)
    if batch > chunk:
        warnings.append(
            f"[UNIDOCK] batch_size ({batch}) > [CHUNK] chunk_size ({chunk}) – "
            f"die Batch-Groesse wird nie erreicht"
        )

    # ── Rescoring Plausibilitaet ──────────────────────────────────────
    r = parsers["rescore"]
    if r.getboolean("RESCORE", "enabled", fallback=True):
        # Mindestens eine Scoring-Funktion muss aktiv sein
        score_flags = {
            "vina":              r.getboolean("RESCORE", "vina_enabled", fallback=True),
            "vinardo":           r.getboolean("RESCORE", "vinardo_enabled", fallback=False),
            "ad4":               r.getboolean("RESCORE", "ad4_enabled", fallback=False),
            "cnnaffinity":       r.getboolean("RESCORE", "cnnaffinity_enabled", fallback=False),
            "cnnscore":          r.getboolean("RESCORE", "cnnscore_enabled", fallback=False),
            "deltalinf9xgb":     r.getboolean("RESCORE", "deltalinf9xgb_enabled", fallback=False),
            "dense":             r.getboolean("RESCORE", "dense_enabled", fallback=False),
        }
        active_scores = [k for k, v in score_flags.items() if v]
        if not active_scores:
            errors.append(
                "[RESCORE] keine einzige Scoring-Funktion aktiv – "
                "das ECR haette nichts zu rangieren"
            )
        else:
            print(f"  OK      Scores aktiv: {', '.join(active_scores)}")

        # Fehlbeschriftung: score_vina enthaelt die Funktion, mit der
        # GEDOCKT wurde – nicht zwingend Vina.
        dock_scoring = d.get("UNIDOCK", "scoring", fallback="vina").strip()
        if score_flags["vina"] and dock_scoring != "vina":
            warnings.append(
                f"[RESCORE] vina_enabled=true, aber docking.ini hat "
                f"scoring={dock_scoring}. Die Spalte 'score_vina' enthaelt "
                f"dann {dock_scoring}-Werte, nicht Vina."
            )

        # Empirische Ueberrepraesentation im Konsens
        empirical = [k for k in ("vina", "vinardo", "ad4") if score_flags[k]]
        learned   = [k for k in ("cnnaffinity", "cnnscore",
                                 "deltalinf9xgb", "dense") if score_flags[k]]
        if len(empirical) >= 2 and learned:
            eq_weights = all(
                r.getfloat("RESCORE", f"w_{k}", fallback=0.0) == 0.0
                for k in empirical + learned
            )
            if eq_weights:
                warnings.append(
                    f"[RESCORE] {len(empirical)} empirische Funktionen "
                    f"({', '.join(empirical)}) bei Gleichgewichtung – sie "
                    f"korrelieren stark und uebergewichten damit die "
                    f"empirische Sicht gegenueber {', '.join(learned)}"
                )

        gnina_needed = any(
            score_flags[k] for k in
            ("vinardo", "ad4", "cnnaffinity", "cnnscore", "dense")
        )
        gbin = r.get("RESCORE", "gnina_binary", fallback="").strip()
        if gnina_needed and not gbin:
            warnings.append(
                "[RESCORE] gnina wird gebraucht, gnina_binary ist leer – "
                "Autodetect via PATH muss im Container greifen"
            )
        weights = [
            r.getfloat("RESCORE", f"w_{k}", fallback=0.0)
            for k in ("vina", "vinardo", "ad4", "cnnaffinity", "cnnscore",
                      "deltalinf9xgb", "dense_cnnaffinity", "dense_cnnscore")
        ]
        if any(w < 0 for w in weights):
            errors.append("[RESCORE] negative ECR-Gewichte sind nicht erlaubt")

    if r.getboolean("REFINEMENT", "enabled", fallback=False):
        tf = r.getfloat("REFINEMENT", "top_fraction", fallback=0.15)
        if not 0.0 < tf <= 1.0:
            errors.append(
                f"[REFINEMENT] top_fraction={tf} liegt ausserhalb (0, 1]"
            )
        rm = r.get("REFINEMENT", "refinement_mode", fallback="local_only")
        if rm not in ("local_only", "minimize", "autobox"):
            errors.append(f"[REFINEMENT] refinement_mode='{rm}' ungueltig")

    # ── Ausgabe ───────────────────────────────────────────────────────
    for w in warnings:
        print(f"  WARNUNG {w}")
    for e in errors:
        print(f"  FEHLER  {e}")

    if errors:
        print(f"\n{len(errors)} Fehler – Pipeline nicht starten.")
        return 1
    print(f"\nKonfiguration konsistent"
          f"{f' ({len(warnings)} Warnung(en))' if warnings else ''}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
