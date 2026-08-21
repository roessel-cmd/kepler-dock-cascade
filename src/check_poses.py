#!/usr/bin/env python3
"""
check_poses.py — findet abgeschnittene _docked.pdbqt vor einem Neustart.

    python3 src/check_poses.py config/docking.ini                # nur pruefen
    python3 src/check_poses.py config/docking.ini --since 6      # letzte 6 h
    python3 src/check_poses.py config/docking.ini --since 6 --delete
    python3 src/check_poses.py config/docking.ini --quarantine data/BROKEN

Exit: 0 = alles heil (oder repariert), 1 = Defekte gefunden, 2 = Fehler.

Warum: restart_orchestrator.find_completed_ligands() akzeptiert jede
_docked.pdbqt mit st_size > 0. Ein Absturz mitten im Schreiben hinterlaesst
eine nicht-leere, aber unvollstaendige Datei — der Ligand gilt damit als
erledigt und wird nie neu gedockt. Auffallen wuerde das erst Stunden
spaeter, wenn gnina die Pose nicht parsen kann.

Geprueft wird ohne die Datei ganz zu lesen: Kopf fuer das REMARK, Ende fuer
das abschliessende ENDMDL. Das genuegt, weil Abschneiden immer das Ende
trifft.

Laeuft auf dem HOST, ohne Container, nur Standardbibliothek.
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HEAD_BYTES = 4096
TAIL_BYTES = 1024

OK        = "ok"
EMPTY     = "leer"
NO_REMARK = "ohne REMARK VINA RESULT"
TRUNCATED = "kein abschliessendes ENDMDL"
UNREADABLE = "nicht lesbar"


def inspect(path_str: str) -> tuple[str, str]:
    """(Pfad, Befund). Liest nur Kopf und Ende der Datei."""
    path = Path(path_str)
    try:
        size = path.stat().st_size
        if size == 0:
            return path_str, EMPTY

        with open(path, "rb") as fh:
            head = fh.read(min(HEAD_BYTES, size))
            if size > TAIL_BYTES:
                fh.seek(-TAIL_BYTES, os.SEEK_END)
                tail = fh.read()
            else:
                tail = head

        if b"REMARK VINA RESULT" not in head:
            return path_str, NO_REMARK

        lines = [ln.strip() for ln in tail.split(b"\n")]
        last  = next((ln for ln in reversed(lines) if ln), b"")
        if last != b"ENDMDL":
            return path_str, TRUNCATED

        return path_str, OK
    except OSError:
        return path_str, UNREADABLE


def collect(results_dir: Path, targets: list[str],
            since_hours: float | None) -> list[str]:
    cutoff = time.time() - since_hours * 3600 if since_hours else None
    out: list[str] = []
    for name in targets:
        for f in (results_dir / name).glob("*_docked.pdbqt"):
            # Nur die Dateien um den Absturzzeitpunkt herum pruefen; alles
            # aeltere wurde von einem geordnet beendeten Lauf geschrieben.
            if cutoff is not None and f.stat().st_mtime < cutoff:
                continue
            out.append(str(f))
    return out


def read_config(ini_path: Path) -> tuple[Path, Path]:
    if not ini_path.is_file():
        raise FileNotFoundError(f"INI nicht gefunden: {ini_path}")
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read(ini_path, encoding="utf-8")
    try:
        return (Path(p.get("PATHS", "results_dir")).expanduser(),
                Path(p.get("PATHS", "target_dir")).expanduser())
    except (configparser.NoSectionError, configparser.NoOptionError) as exc:
        raise KeyError(f"[PATHS] unvollstaendig in {ini_path}: {exc}")


def target_names(results_dir: Path, target_dir: Path,
                 only: list[str]) -> tuple[list[str], list[str]]:
    if only:
        return only, []
    notes: list[str] = []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pipeline_common import parse_target_config  # noqa: PLC0415
        targets, warns = parse_target_config(target_dir / "config.txt", target_dir)
        names = [t.name for t in targets]
        if names:
            return names, warns
        notes.append("config.txt lieferte keine Targets – RESULTS/ wird durchsucht.")
    except Exception as exc:                       # noqa: BLE001
        notes.append(f"config.txt nicht auswertbar ({exc}) – RESULTS/ wird durchsucht.")
    names = [d.name for d in sorted(results_dir.iterdir())
             if d.is_dir() and next(d.glob("*_docked.pdbqt"), None) is not None]
    return names, notes


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Abgeschnittene Posen vor einem Docking-Neustart finden")
    ap.add_argument("ini", nargs="?", type=Path, default=Path("config/docking.ini"))
    ap.add_argument("--since", type=float, metavar="STUNDEN",
                    help="Nur Dateien juenger als N Stunden pruefen")
    ap.add_argument("--target", action="append", default=[],
                    help="Nur dieses Target (mehrfach moeglich)")
    ap.add_argument("--delete", action="store_true",
                    help="Defekte loeschen, damit der Restart sie neu dockt")
    ap.add_argument("--quarantine", type=Path, metavar="DIR",
                    help="Defekte dorthin verschieben statt loeschen")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--list", action="store_true",
                    help="Jeden Defekt einzeln ausgeben")
    args = ap.parse_args()

    if args.delete and args.quarantine:
        print("FEHLER: --delete und --quarantine schliessen sich aus",
              file=sys.stderr)
        return 2

    try:
        results_dir, target_dir = read_config(args.ini)
    except (OSError, KeyError, configparser.Error) as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    if not results_dir.is_dir():
        print(f"FEHLER: results_dir existiert nicht: {results_dir}", file=sys.stderr)
        return 2

    names, notes = target_names(results_dir, target_dir, args.target)
    for n in notes:
        print(f"  {n}")
    if not names:
        print("FEHLER: Keine Targets ermittelbar", file=sys.stderr)
        return 2

    files = collect(results_dir, names, args.since)
    if not files:
        scope = f" (juenger als {args.since} h)" if args.since else ""
        print(f"  Keine Posen zu pruefen{scope}.")
        return 0

    print(f"  Pruefe {len(files):,} Posen aus {len(names)} Target(s) "
          f"mit {args.workers} Prozessen ...")

    t0 = time.time()
    broken: dict[str, list[str]] = {}
    n_ok = 0
    with Pool(args.workers) as pool:
        for path_str, verdict in pool.imap_unordered(inspect, files,
                                                     chunksize=512):
            if verdict == OK:
                n_ok += 1
            else:
                broken.setdefault(verdict, []).append(path_str)
    dt = time.time() - t0

    n_broken = sum(len(v) for v in broken.values())
    print(f"  {n_ok:,} in Ordnung, {n_broken:,} defekt  ({dt:.1f} s)")

    if not n_broken:
        return 0

    for verdict, paths in sorted(broken.items()):
        print(f"    {len(paths):>8,}  {verdict}")
        if args.list:
            for p in sorted(paths):
                print(f"              {p}")

    if not (args.delete or args.quarantine):
        print("\n  Nichts geaendert. Zum Entfernen --delete oder "
              "--quarantine DIR angeben,")
        print("  danach das Docking neu starten – die Liganden werden dann "
              "erneut gedockt.")
        return 1

    if args.quarantine:
        args.quarantine.mkdir(parents=True, exist_ok=True)

    moved = failed = 0
    for paths in broken.values():
        for p in paths:
            try:
                if args.quarantine:
                    # Zielname mit Target-Praefix, sonst kollidieren
                    # gleichnamige Liganden aus zwei Targets.
                    src = Path(p)
                    shutil.move(p, args.quarantine / f"{src.parent.name}__{src.name}")
                else:
                    Path(p).unlink()
                moved += 1
            except OSError as exc:
                print(f"  WARNUNG: {p}: {exc}", file=sys.stderr)
                failed += 1

    verb = "verschoben" if args.quarantine else "geloescht"
    print(f"\n  {moved:,} Datei(en) {verb}"
          + (f", {failed:,} fehlgeschlagen" if failed else ""))
    print("  Docking jetzt neu starten:")
    print("    RUN_CONVERSION=false RUN_RESCORING=false ./pipeline_start.sh --restart")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
