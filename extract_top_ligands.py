#!/usr/bin/env python3
"""
extract_top_ligands.py — Top-N Liganden pro Target aus einem Docking-Archiv extrahieren.

Verwendung:
    python3 extract_top_ligands.py ARCHIV_ODER_ORDNER [--target NAME] [--top 100] [--out extracted]

ARCHIV_ODER_ORDNER: Pfad zu <jobname>.tar.gz oder zum entpackten <jobname>-Ordner.
Erwartet in rescoring_ligands_<target>.csv die Spalten ecr_rank, ligand, ecr_score, best_pose.

Legt je Target einen Ordner an mit
  * 00_target.pdbqt          - die Rezeptorstruktur
  * <rank>_<basename>.pdbqt  - die Top-N Posen (Default-Benennung, siehe --name-mode)
  * selection.csv            - rank, ligand, file, ecr_score, best_pose + alle Score-Spalten

Zusaetzliche Score-Spalten werden, falls vorhanden, aus Top250_<target>.csv
dazugemischt (Join ueber den Ligandennamen).  MANIFEST.txt/config werden in den
Zielordner kopiert, damit die Auswertung autark ist.

Im Anschluss wird automatisch der Word-Report erzeugt (dock_report.py muss im
selben Verzeichnis liegen; abschaltbar mit --no-report).
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tarfile
from pathlib import Path

ID_HINTS = ("ligand", "name", "id", "molecule", "mol")
EXTRA_CSV_PATTERNS = ("Top250_{target}.csv", "Top*_{target}.csv")
MANIFEST_NAMES = ("MANIFEST.txt", "config", "config.ini")

# Spalten, die in selection.csv immer zuerst stehen
BASE_FIELDS = ["rank", "ligand", "file", "ecr_score", "best_pose"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def find_ranking_csv(job_dir: Path, target: str) -> Path:
    """Massgeblich ist ausschliesslich rescoring_ligands_<target>.csv."""
    csv_path = job_dir / f"rescoring_ligands_{target}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"rescoring_ligands_{target}.csv fehlt in {job_dir}.")
    return csv_path


def load_ranked_rows(csv_path: Path, top_n: int) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def score(row: dict) -> float:
        try:
            return -float(row.get("ecr_score") or 0.0)
        except ValueError:
            return 0.0

    rows.sort(key=score)
    return rows[:top_n]


def load_extra_scores(job_dir: Path, target: str) -> dict[str, dict]:
    """
    Zusatzspalten (score_vina_best, score_dense_cnnaffinity_best, ...) aus
    Top250_<target>.csv laden.  Join-Key ist der normalisierte Ligandenname.
    """
    path = None
    for pattern in EXTRA_CSV_PATTERNS:
        hits = sorted(job_dir.glob(pattern.format(target=target)))
        if hits:
            path = hits[0]
            break
    if path is None:
        return {}

    out: dict[str, dict] = {}
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            id_col = next(
                (f for f in fields
                 if any(h in (f or "").lower() for h in ID_HINTS)), None)
            if id_col is None:
                return {}
            for row in reader:
                key = norm(str(row.get(id_col, "")).replace("_docked", ""))
                if key:
                    out[key] = row
    except OSError as exc:
        print(f"  [WARNUNG] {path.name} nicht lesbar: {exc}", file=sys.stderr)
        return {}

    print(f"  [{target}] Zusatzspalten aus {path.name} ({len(out)} Zeilen)")
    return out


def guess_ligand_id(row: dict, results_dir: Path) -> str | None:
    """Fallback falls die Spalte 'ligand' fehlt: Werte gegen vorhandene *_docked.pdbqt pruefen."""
    keys = sorted(row.keys(), key=lambda k: not any(h in k.lower() for h in ID_HINTS))
    for key in keys:
        val = (row.get(key) or "").strip()
        if not val:
            continue
        matches = list(results_dir.glob(f"{val}*_docked.pdbqt"))
        if matches:
            return matches[0].name.removesuffix("_docked.pdbqt")
    return None


def target_filename(rank: int, ligand: str, target: str, name_mode: str,
                    basename: str | None) -> str:
    """
    Dateiname der extrahierten Pose.

    name_mode
      target  0125_eS4_7b7d_p0.pdbqt          (Default, <basename> = Pocket)
      ligand  0125_mol_0022884.pdbqt          (nur die interne Molekuel-ID)
      full    0125_<ligand>_docked.pdbqt      (altes Verhalten)
    """
    if name_mode == "full":
        return f"{rank:04d}_{ligand}_docked.pdbqt"
    if name_mode == "ligand":
        short = ligand.split("_mol_")[-1] if "_mol_" in ligand else ligand
        return f"{rank:04d}_mol_{short}.pdbqt" if "_mol_" in ligand \
            else f"{rank:04d}_{ligand}.pdbqt"
    return f"{rank:04d}_{basename or target}.pdbqt"


def copy_manifest(job_dir: Path, out_dir: Path) -> None:
    """MANIFEST.txt / config in den Zielordner kopieren (einmal pro Lauf)."""
    for name in MANIFEST_NAMES:
        src = job_dir / name
        if src.is_file():
            dst = out_dir / name
            if not dst.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  {name} -> {dst}")
            return
        if src.is_dir():                       # 'config' kann ein Ordner sein
            for cand in sorted(src.glob("*.txt")) + sorted(src.glob("*.ini")):
                dst = out_dir / cand.name
                if not dst.exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cand, dst)
                    print(f"  {cand.name} -> {dst}")
            return


def process_target(job_dir: Path, target: str, top_n: int, out_dir: Path,
                   name_mode: str = "target",
                   basename: str | None = None) -> None:
    csv_path = find_ranking_csv(job_dir, target)
    rows = load_ranked_rows(csv_path, top_n)
    extra = load_extra_scores(job_dir, target)

    results_dir = job_dir / "RESULTS" / target
    target_pdbqt = job_dir / "TARGET" / f"{target}.pdbqt"
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"{results_dir} fehlt - wurde das Archiv mit --no-poses erzeugt?"
        )
    if not target_pdbqt.is_file():
        raise FileNotFoundError(f"{target_pdbqt} fehlt.")

    dest = out_dir / target
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_pdbqt, dest / "00_target.pdbqt")

    selection_rows: list[dict] = []
    extra_fields: list[str] = []
    n_ok = 0
    for i, row in enumerate(rows, start=1):
        ligand = (row.get("ligand") or "").strip() or guess_ligand_id(row, results_dir)
        if not ligand:
            print(f"  [WARNUNG] {target} Zeile {i}: keine Liganden-ID gefunden, uebersprungen.", file=sys.stderr)
            continue
        src = results_dir / f"{ligand}_docked.pdbqt"
        if not src.is_file():
            print(f"  [WARNUNG] {target}/{ligand}: {src.name} nicht in RESULTS gefunden, uebersprungen.",
                  file=sys.stderr)
            continue

        try:
            rank = int(row.get("ecr_rank") or i)
        except ValueError:
            rank = i
        try:
            best_pose = int(row.get("best_pose") or 1)
        except ValueError:
            best_pose = 1

        fname = target_filename(rank, ligand, target, name_mode, basename)
        shutil.copy2(src, dest / fname)

        merged = {k: v for k, v in row.items() if k not in BASE_FIELDS}
        for k, v in extra.get(norm(ligand), {}).items():
            if k not in merged or not str(merged.get(k, "")).strip():
                merged[k] = v
        for k in merged:
            if k not in extra_fields:
                extra_fields.append(k)

        selection_rows.append({
            "rank": rank, "ligand": ligand, "file": fname,
            "ecr_score": row.get("ecr_score", ""), "best_pose": best_pose,
            **merged,
        })
        n_ok += 1

    fieldnames = BASE_FIELDS + [f for f in extra_fields if f not in BASE_FIELDS]
    with (dest / "selection.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(selection_rows)

    print(f"[{target}] {n_ok}/{len(rows)} Posen extrahiert -> {dest}")


def resolve_job_dir(archive_arg: str, out_dir: Path) -> Path:
    src = Path(archive_arg)
    if src.is_file() and src.name.endswith(".tar.gz"):
        unpack_dir = out_dir / "_unpacked"
        unpack_dir.mkdir(parents=True, exist_ok=True)
        print(f"Entpacke {src} nach {unpack_dir} ...")
        with tarfile.open(src, "r:gz") as tf:
            tf.extractall(unpack_dir)
        return next(p for p in unpack_dir.iterdir() if p.is_dir())  # <jobname>/ ist der einzige Top-Ordner
    if src.is_dir():
        return src
    sys.exit(f"FEHLER: {src} ist weder ein .tar.gz noch ein Ordner.")


def discover_targets(job_dir: Path) -> list[str]:
    targets = sorted({
        p.name.removeprefix("rescoring_ligands_").removesuffix(".csv")
        for p in job_dir.glob("rescoring_ligands_*.csv")
    })
    if not targets:
        sys.exit(f"FEHLER: keine rescoring_ligands_*.csv in {job_dir} gefunden.")
    return targets


def make_report(out_dir: Path, report_path: Path | None,
                also_csv: bool, also_xlsx: bool, top: int | None) -> None:
    """dock_report.py aufrufen (liegt idealerweise neben diesem Skript)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import dock_report
    except ImportError:
        print("\n[WARNUNG] dock_report.py nicht gefunden - Report uebersprungen.\n"
              "          Skript in denselben Ordner legen oder --no-report nutzen.",
              file=sys.stderr)
        return

    print("\n--- Report ---")
    try:
        dock_report.build_report(
            base=out_dir,
            out=report_path or (out_dir / "docking_report.docx"),
            top=top, also_csv=also_csv, also_xlsx=also_xlsx,
        )
    except SystemExit as exc:                  # z.B. python-docx fehlt
        print(f"[WARNUNG] Report nicht erzeugt: {exc}", file=sys.stderr)
    except Exception as exc:                   # noqa: BLE001
        print(f"[WARNUNG] Report fehlgeschlagen: {exc}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("archive", help="Pfad zu <jobname>.tar.gz oder zum entpackten <jobname>-Ordner")
    ap.add_argument("--target", help="Nur dieses Target verarbeiten (Default: alle gefundenen)")
    ap.add_argument("--top", type=int, default=100, help="Anzahl der Top-Liganden (Default: 100)")
    ap.add_argument("--out", default="extracted", help="Zielordner (Default: ./extracted)")
    ap.add_argument("--name-mode", choices=("target", "ligand", "full"),
                    default="target",
                    help="Benennung der Posen: target = <rank>_<pocket>.pdbqt "
                         "(Default), ligand = <rank>_mol_<id>.pdbqt, "
                         "full = altes Schema mit vollem Ligandennamen")
    ap.add_argument("--basename", default=None,
                    help="Pocket-Name fuer --name-mode target "
                         "(Default: Target-Name)")
    ap.add_argument("--no-report", action="store_true",
                    help="keinen Word-Report erzeugen")
    ap.add_argument("--report-out", type=Path, default=None,
                    help="Pfad der Word-Datei (Default: <out>/docking_report.docx)")
    ap.add_argument("--report-top", type=int, default=None,
                    help="nur die besten N Eintraege in den Report schreiben")
    ap.add_argument("--also-csv", action="store_true",
                    help="Sammel-CSV zusaetzlich zum Report")
    ap.add_argument("--also-xlsx", action="store_true",
                    help="XLSX-Mappe zusaetzlich zum Report")
    args = ap.parse_args()

    out_dir = Path(args.out)
    job_dir = resolve_job_dir(args.archive, out_dir)
    targets = [args.target] if args.target else discover_targets(job_dir)

    copy_manifest(job_dir, out_dir)

    ok = 0
    for target in targets:
        try:
            process_target(job_dir, target, args.top, out_dir,
                           args.name_mode, args.basename)
            ok += 1
        except FileNotFoundError as e:
            print(f"[{target}] FEHLER: {e}", file=sys.stderr)

    if ok and not args.no_report:
        make_report(out_dir, args.report_out, args.also_csv, args.also_xlsx,
                    args.report_top)


if __name__ == "__main__":
    main()
