#!/usr/bin/env python3
"""
check_ligands.py
================
Prueft die von sdf_to_pdbqt.py erzeugte PDBQT-Bibliothek, bevor das
Docking startet. Laeuft auf dem HOST.

    python3 check_ligands.py data/PDBQT

Geprueft wird:
  1. Layout       – flach oder in Unterordnern (0000/, 0001/, ...)
  2. Anzahl       – wieviele PDBQTs insgesamt
  3. Kollisionen  – doppelte Dateinamen ueber Unterordner hinweg.
                    Kritisch, weil die Docking-Ergebnisse FLACH in
                    RESULTS/<target>/<stem>_docked.pdbqt landen: zwei
                    Liganden mit gleichem Stem ueberschreiben sich dort
                    gegenseitig, ohne Fehlermeldung.
  4. Leere Dateien – abgebrochene Konvertierungen

Exit-Code 1 bei Kollisionen oder leerer Bibliothek.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Aufruf: python3 check_ligands.py <pdbqt_dir>")
        return 1

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"  FEHLER  {root} ist kein Verzeichnis")
        return 1

    flat_count = len(list(root.glob("*.pdbqt")))
    all_files  = sorted(root.rglob("*.pdbqt"))
    total      = len(all_files)

    if total == 0:
        print(f"  FEHLER  Keine PDBQTs unter {root}")
        return 1

    subdirs = sorted({f.parent for f in all_files if f.parent != root})
    if subdirs:
        print(f"  Layout  {len(subdirs)} Unterordner "
              f"({subdirs[0].name} ... {subdirs[-1].name})"
              f"{f', dazu {flat_count} flach' if flat_count else ''}")
        print("          → Docking-Code muss rekursiv suchen "
              "(find_ligand_files in pipeline_common.py)")
    else:
        print(f"  Layout  flach, {total} Dateien in {root}")
        if total > 100_000:
            print("  WARNUNG Sehr viele Dateien in einem Ordner – "
                  "--flat weglassen entlastet das Dateisystem")

    print(f"  Anzahl  {total} PDBQTs")

    # ── Kollisionen ───────────────────────────────────────────────────
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for f in all_files:
        by_stem[f.stem].append(f)
    collisions = {k: v for k, v in by_stem.items() if len(v) > 1}

    if collisions:
        print(f"  FEHLER  {len(collisions)} doppelte Ligandennamen "
              f"({sum(len(v) for v in collisions.values())} Dateien betroffen)")
        for stem, paths in list(collisions.items())[:5]:
            print(f"          '{stem}': {', '.join(str(p) for p in paths[:3])}"
                  f"{' ...' if len(paths) > 3 else ''}")
        if len(collisions) > 5:
            print(f"          ... und {len(collisions) - 5} weitere")
        print("          Ursache: sdf_to_pdbqt.py leitet den Dateinamen aus")
        print("          dem SDF-Titel ab (safe_mol_name). Nicht eindeutige")
        print("          Titel → gleicher Dateiname. In RESULTS/ ueberschreiben")
        print("          sich diese Liganden gegenseitig.")
    else:
        print("  Namen   alle eindeutig")

    # ── Leere Dateien ─────────────────────────────────────────────────
    empty = [f for f in all_files if f.stat().st_size == 0]
    if empty:
        print(f"  WARNUNG {len(empty)} leere PDBQTs "
              f"(z.B. {empty[0]}) – abgebrochene Konvertierungen")

    if collisions:
        print("\nKollisionen beheben, bevor gedockt wird.")
        return 1
    print("\nBibliothek in Ordnung.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
