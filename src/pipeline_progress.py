#!/usr/bin/env python3
"""
pipeline_progress.py
====================
Ermittelt, wieviel Docking-Arbeit noch offen ist. Laeuft auf dem HOST.

    python3 src/pipeline_progress.py config/docking.ini

Ausgabe (eine Zeile je Target plus Summenzeile) und Exit-Code:

    0   es ist noch Arbeit offen        -> Slurm-Kette fortsetzen
    1   alles fertig                    -> Kette beenden
    2   Konfigurations- oder Datenfehler-> Kette beenden, Ticket

Der Exit-Code ist die eigentliche Schnittstelle. Ohne diese Pruefung
wuerde eine afterany-Kette nach dem letzten sinnvollen Lauf weiter Jobs
einreichen, bis das Kontingent aufgebraucht ist.

Gezaehlt wird gegen dieselbe Groesse, die auch der Restart-Orchestrator
benutzt: vorhandene <ligand>_docked.pdbqt. Damit stimmen Abbruchbedingung
und Wiederaufnahme per Konstruktion ueberein, statt zwei unabhaengige
Vorstellungen von "fertig" zu haben.

--json gibt dasselbe maschinenlesbar aus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_common import load_ini, parse_target_config, require  # noqa: E402


def count_ligands(pdbqt_dir: Path, subdir: str | None) -> int:
    root = pdbqt_dir
    if subdir and (pdbqt_dir / subdir).is_dir():
        root = pdbqt_dir / subdir
    return sum(1 for _ in root.rglob("*.pdbqt"))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    ini_path = Path(args[0]) if args else Path("config/docking.ini")

    if not ini_path.is_file():
        print(f"FEHLER: INI nicht gefunden: {ini_path}", file=sys.stderr)
        return 2

    try:
        p = load_ini(ini_path)
        pdbqt_dir  = Path(require(p, "PATHS", "pdbqt_dir"))
        target_dir = Path(require(p, "PATHS", "target_dir"))
        results_dir = Path(require(p, "PATHS", "results_dir"))
        targets, warnings = parse_target_config(
            target_dir / "config.txt", target_dir
        )
    except Exception as exc:                       # noqa: BLE001
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    if not targets:
        print("FEHLER: Keine gueltigen Targets in config.txt", file=sys.stderr)
        return 2

    rows = []
    total_expected = total_done = 0

    for t in targets:
        expected = count_ligands(pdbqt_dir, t.ligand_subdir)
        tdir = results_dir / t.name
        done = (sum(1 for _ in tdir.glob("*_docked.pdbqt"))
                if tdir.is_dir() else 0)
        # Fehlgeschlagene Liganden zaehlen als erledigt: sie werden auch
        # beim naechsten Lauf scheitern und wuerden die Kette sonst
        # endlos weiterlaufen lassen.
        failed = (sum(1 for _ in tdir.glob("*_ERROR.log"))
                  if tdir.is_dir() else 0)
        handled = min(done + failed, expected)
        rows.append({
            "target": t.name, "expected": expected, "done": done,
            "failed": failed, "remaining": max(0, expected - handled),
        })
        total_expected += expected
        total_done     += handled

    remaining = max(0, total_expected - total_done)

    if as_json:
        print(json.dumps({
            "targets": rows, "expected": total_expected,
            "handled": total_done, "remaining": remaining,
        }, indent=2))
    else:
        for w in warnings:
            print(f"  {w}")
        print(f"  {'Target':<24}{'erwartet':>12}{'fertig':>10}"
              f"{'fehler':>9}{'offen':>10}")
        print("  " + "-" * 65)
        for r in rows:
            print(f"  {r['target']:<24}{r['expected']:>12,}{r['done']:>10,}"
                  f"{r['failed']:>9,}{r['remaining']:>10,}")
        print("  " + "-" * 65)
        pct = 100.0 * total_done / total_expected if total_expected else 0.0
        print(f"  {'GESAMT':<24}{total_expected:>12,}{total_done:>10,}"
              f"{'':>9}{remaining:>10,}   ({pct:.1f} % erledigt)")

    return 0 if remaining > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
