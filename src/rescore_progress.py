#!/usr/bin/env python3
"""
rescore_progress.py — Fortschrittszaehler fuer das Rescoring.

    python3 src/rescore_progress.py config/rescore.ini [--json] [--target NAME]

Exit-Codes wie pipeline_progress.py: 0 = offen, 1 = fertig, 2 = Fehler.
Einheit ist der Block (RESULTS/<target>/.rescore_partial/scores_NNNNN.csv),
nicht der Ligand — das Rescoring legt keine Datei pro Ligand an.

Gelesen wird config/rescore.ini. Nicht docking.ini (keine [RESCORE]-Sektion)
und nicht src/pipeline_config.ini (existiert nur im Container).
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import sys
from pathlib import Path

PARTIAL_DIR = ".rescore_partial"


def read_config(ini_path: Path) -> tuple[Path, Path, int, bool]:
    if not ini_path.is_file():
        raise FileNotFoundError(f"INI nicht gefunden: {ini_path}")

    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read(ini_path, encoding="utf-8")

    if not p.has_section("RESCORE"):
        raise KeyError(f"Sektion [RESCORE] fehlt in {ini_path} – "
                       f"ist das versehentlich docking.ini?")
    try:
        results_dir = Path(p.get("PATHS", "results_dir")).expanduser()
    except (configparser.NoSectionError, configparser.NoOptionError):
        raise KeyError(f"Pflichtparameter '[PATHS] results_dir' fehlt in {ini_path}")

    return (results_dir,
            Path(p.get("PATHS", "target_dir", fallback="./TARGET")).expanduser(),
            p.getint("RESCORE", "rescore_block_size", fallback=0),
            p.getboolean("RESCORE", "enabled", fallback=True))


def target_names(results_dir: Path, target_dir: Path,
                 only: list[str] | None) -> tuple[list[str], list[str]]:
    """Targetliste aus TARGET/config.txt, damit beide Stufen dieselbe zaehlen."""
    notes: list[str] = []
    if only:
        return only, notes

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pipeline_common import parse_target_config  # noqa: PLC0415
        targets, warns = parse_target_config(target_dir / "config.txt", target_dir)
        names = [t.name for t in targets]
        if names:
            notes.extend(warns)
            return names, notes
        notes.append("config.txt lieferte keine Targets – RESULTS/ wird durchsucht.")
    except Exception as exc:                       # noqa: BLE001
        notes.append(f"config.txt nicht auswertbar ({exc}) – RESULTS/ wird durchsucht.")

    out = [d.name for d in sorted(results_dir.iterdir())
           if d.is_dir() and next(d.glob("*_docked.pdbqt"), None) is not None]
    return out, notes


def partial_indices(tdir: Path) -> set[int]:
    pdir = tdir / PARTIAL_DIR
    if not pdir.is_dir():
        return set()
    idx = set()
    for f in pdir.glob("scores_*.csv"):
        if f.stat().st_size <= 0:
            continue
        stem = f.stem.split("_", 1)[1]
        if stem.isdigit():
            idx.add(int(stem))
    return idx


def unique_ligands(path: Path, limit: int = 2_000_000) -> int:
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seen.add(row["ligand"])
            if len(seen) > limit:
                break
    return len(seen)


def check_block_size(tdir: Path, idx: int, expected: int) -> str | None:
    """
    Blockindizes zeigen auf Positionen in der sortierten Dateiliste. Wurde
    rescore_block_size zwischen zwei Laeufen geaendert, meinen alte Dateien
    einen anderen Bereich und der gezaehlte Fortschritt waere erfunden.
    Eine _docked.pdbqt = ein Ligand, also muss die Zahl verschiedener
    Liganden der Blocklaenge entsprechen.
    """
    path = tdir / PARTIAL_DIR / f"scores_{idx:05d}.csv"
    try:
        found = unique_ligands(path)
    except (OSError, ValueError, KeyError) as exc:
        return f"{path.name} unlesbar ({exc})"
    if found != expected:
        return (f"{path.name} enthaelt {found:,} Liganden, erwartet {expected:,}. "
                f"Wurde rescore_block_size geaendert? Dann {tdir/PARTIAL_DIR} "
                f"loeschen und neu scoren.")
    return None


def scan_target(results_dir: Path, name: str, block_size: int) -> dict:
    tdir  = results_dir / name
    files = sorted(tdir.glob("*_docked.pdbqt"))
    n_files = len(files)

    final = tdir / f"rescoring_ligands_{name}.csv"
    if final.is_file() and final.stat().st_size > 0:
        return dict(target=name, poses=n_files, blocks=0, done=0,
                    remaining=0, complete=True, note="fertig", warn=None)

    if n_files == 0:
        return dict(target=name, poses=0, blocks=0, done=0,
                    remaining=0, complete=True, note="keine Posen", warn=None)

    # block_size = 0: alles in einem Durchgang, kein Zwischenstand.
    size    = block_size if block_size > 0 else n_files
    blocks  = math.ceil(n_files / size)
    have    = partial_indices(tdir)

    stale = {i for i in have if i >= blocks}
    warn  = None
    if stale:
        warn = (f"{len(stale)} Blockdatei(en) ausserhalb des gueltigen Bereichs "
                f"(Index >= {blocks}) – Rest eines Laufs mit anderer Blockgroesse.")

    valid = sorted(i for i in have if i < blocks)
    if valid and block_size > 0:
        first    = valid[0]
        expected = min(size, n_files - first * size)
        problem  = check_block_size(tdir, first, expected)
        if problem:
            return dict(target=name, poses=n_files, blocks=blocks, done=0,
                        remaining=blocks, complete=False, note="INKONSISTENT",
                        warn=problem, fatal=True)

    done = len(valid)
    return dict(target=name, poses=n_files, blocks=blocks, done=done,
                remaining=blocks - done, complete=(done >= blocks),
                note="" if done < blocks else "Bloecke fertig, Ranking offen",
                warn=warn)


def print_table(rows: list[dict]) -> None:
    print(f"  {'Target':<26}{'Posen':>12}{'Bloecke':>10}{'fertig':>9}{'offen':>9}")
    print("  " + "-" * 65)
    for r in rows:
        print(f"  {r['target']:<26}{r['poses']:>12,}{r['blocks']:>10,}"
              f"{r['done']:>9,}{r['remaining']:>9,}"
              + (f"   {r['note']}" if r['note'] else ""))
    print("  " + "-" * 65)

    tb = sum(r["blocks"] for r in rows)
    td = sum(r["done"] for r in rows)
    tr = sum(r["remaining"] for r in rows)
    pct = 100.0 * td / tb if tb else 100.0
    print(f"  {'GESAMT':<26}{sum(r['poses'] for r in rows):>12,}{tb:>10,}"
          f"{td:>9,}{tr:>9,}   ({pct:.1f} % erledigt)")

    for r in rows:
        if r.get("warn"):
            print(f"  WARNUNG [{r['target']}]: {r['warn']}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rescoring-Fortschritt in Bloecken zaehlen")
    ap.add_argument("ini", nargs="?", type=Path,
                    default=Path("config/rescore.ini"),
                    help="Rescoring-INI (default config/rescore.ini)")
    ap.add_argument("--json", action="store_true",
                    help="Maschinenlesbar auf stdout, sonst nichts")
    ap.add_argument("--target", action="append", default=[],
                    help="Nur dieses Target (mehrfach moeglich)")
    args = ap.parse_args()

    try:
        results_dir, target_dir, block_size, enabled = read_config(args.ini)
    except (OSError, KeyError, configparser.Error) as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    if not results_dir.is_dir():
        print(f"FEHLER: results_dir existiert nicht: {results_dir}", file=sys.stderr)
        return 2

    if not enabled:
        if args.json:
            print(json.dumps({"remaining": 0, "reason": "rescoring disabled"}))
        else:
            print("Rescoring in der INI deaktiviert ([RESCORE] enabled = false).")
        return 1

    try:
        names, notes = target_names(results_dir, target_dir, args.target or None)
    except OSError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    if notes and not args.json:
        for n in notes:
            print(f"  {n}")

    if not names:
        print(f"FEHLER: Keine Targets ermittelbar "
              f"(weder {target_dir/'config.txt'} noch {results_dir})",
              file=sys.stderr)
        return 2

    missing = [n for n in names if not (results_dir / n).is_dir()]
    if missing:
        print(f"FEHLER: Kein RESULTS-Verzeichnis fuer: {', '.join(missing)}. "
              f"Ist das Docking dieser Targets gelaufen?", file=sys.stderr)
        return 2

    rows = [scan_target(results_dir, n, block_size) for n in names]

    if any(r.get("fatal") for r in rows):
        for r in rows:
            if r.get("fatal"):
                print(f"FEHLER [{r['target']}]: {r['warn']}", file=sys.stderr)
        return 2

    remaining = sum(r["remaining"] for r in rows)

    if args.json:
        print(json.dumps({
            "remaining":  remaining,
            "unit":       "blocks",
            "block_size": block_size,
            "total":      sum(r["blocks"] for r in rows),
            "done":       sum(r["done"] for r in rows),
            "targets":    [{k: v for k, v in r.items() if k != "fatal"}
                           for r in rows],
        }))
        return 0 if remaining > 0 else 1

    print_table(rows)
    if block_size <= 0:
        print("\n  HINWEIS: rescore_block_size = 0 – keine Zwischenstaende. Ein an "
              "der Walltime\n  abgebrochenes Target faengt von vorn an, und die "
              "Kette sieht keinen Teilfortschritt.", file=sys.stderr)

    return 0 if remaining > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
