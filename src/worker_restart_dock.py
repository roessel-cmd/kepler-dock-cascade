#!/usr/bin/env python3
"""
worker_restart.py
=================
Läuft INNERHALB des Apptainer-Containers.
Wird vom restart_orchestrator pro GPU gestartet.

Umgebungsvariablen:
  CUDA_VISIBLE_DEVICES  → welche physische GPU
  WORKER_TARGET         → Target-Name
  WORKER_GPU_ID         → GPU Index

Arbeitsverzeichnis: /workspace
GPU-Docking laeuft ueber unidock_engine.dock_batch_unidock().
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from unidock_engine import dock_batch_unidock
from pipeline_common import find_ligand_files

from docking_config import (
    PIPELINE_CONFIG_FILE,
    DockingConfig as Config,
    TargetConfig,
    parse_target_config,
    setup_logging,
)

# Rescoring und Refinement laufen in Stufe 3 (rescoring-gpu.sif) und werden
# von hier NICHT mehr aufgerufen. Frueher haengte der Restart-Worker das
# Rescoring direkt an das Docking an; mit der Aufteilung in drei Container
# waeren die Module hier gar nicht vorhanden.

WORKER_TARGET      = os.environ.get("WORKER_TARGET", "")
WORKER_GPU_ID      = int(os.environ.get("WORKER_GPU_ID", "0"))
WORKER_LIGAND_LIST = os.environ.get("WORKER_LIGAND_LIST", "")
WORKER_CHUNK_ID    = os.environ.get("WORKER_CHUNK_ID", "")
WORKER_PERSISTENT  = os.environ.get("WORKER_PERSISTENT", "") == "1"
WORKER_JOB_DIR     = os.environ.get("WORKER_JOB_DIR", "")


# ======================================================================
# FERTIGE LIGANDEN ERMITTELN
# ======================================================================

def find_completed_from_pdbqt(target_results_dir: Path) -> set[str]:
    completed = set()
    if not target_results_dir.exists():
        return completed
    for f in target_results_dir.glob("*_docked.pdbqt"):
        if f.stat().st_size > 0:
            completed.add(f.stem.removesuffix("_docked"))
    return completed


def find_completed_from_log(log_file: Path, target_name: str) -> dict[str, float | None]:
    completed: dict[str, float | None] = {}
    if not log_file.exists():
        return completed

    this_target = re.compile(
        r"===\s+TARGET\s+\d+/\d+:\s+" + re.escape(target_name) + r"\s+==="
    )
    any_target  = re.compile(r"===\s+TARGET\s+\d+/\d+:")
    # Unterstützt altes und neues Log-Format
    success_pat = re.compile(
        r"\[\w[\w\s]*\]\s+\[\d+/\d+\]\s+(\S+)\s+([-\d.]+)\s+kcal/mol"
    )
    in_block = False

    with open(log_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if this_target.search(line):
                in_block = True
                continue
            if in_block and "===" in line:
                if any_target.search(line) and not this_target.search(line):
                    in_block = False
                elif "PIPELINE" in line or "RESTART" in line:
                    in_block = False
            if not in_block:
                continue
            m = success_pat.search(line)
            if m and "FEHLER" not in line:
                try:
                    completed[m.group(1)] = float(m.group(2))
                except ValueError:
                    completed[m.group(1)] = None

    return completed


def find_completed_ligands(
    target_name: str,
    target_results_dir: Path,
    log_file: Path,
    logger: logging.Logger,
) -> tuple[set[str], dict[str, float | None]]:
    pdbqt_done = find_completed_from_pdbqt(target_results_dir)
    log_done   = find_completed_from_log(log_file, target_name)
    all_done   = pdbqt_done | set(log_done.keys())
    energies   = {name: log_done.get(name) for name in all_done}

    logger.debug(
        "  [%s] PDBQT: %d | Log: %d | Union: %d",
        target_name, len(pdbqt_done), len(log_done), len(all_done),
    )
    return all_done, energies


# ======================================================================
# DOCKING
# ======================================================================

# Das GPU-Docking laeuft ueber unidock_engine.dock_batch_unidock().
# Der frueher hier stehende Ein-Ligand-pro-Prozess-Aufruf von Vina-GPU+
# wurde entfernt: er hat pro Ligand OpenCL-Context, Kernel-Build und
# Grid-Map-Berechnung neu bezahlt und die GPU bei ~20 % Auslastung gehalten.


# ======================================================================
# CSV
# ======================================================================

def load_existing_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    results = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["success"] = row["success"].strip().lower() in ("true", "1", "yes")
            try:
                row["best_energy_kcal_mol"] = (
                    float(row["best_energy_kcal_mol"])
                    if row["best_energy_kcal_mol"] not in ("", "None", "null")
                    else None
                )
            except (ValueError, KeyError):
                row["best_energy_kcal_mol"] = None
            results.append(row)
    return results


def write_results_csv(results, target_results_dir, target_name) -> Path:
    csv_path   = target_results_dir / f"docking_results_{target_name}.csv"
    successful = sorted(
        [r for r in results if r["success"]],
        key=lambda r: (r["best_energy_kcal_mol"] is None,
                       r["best_energy_kcal_mol"] or 0.0),
    )
    failed = [r for r in results if not r["success"]]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ligand", "success", "best_energy_kcal_mol", "error"]
        )
        writer.writeheader()
        writer.writerows(successful + failed)
    return csv_path


# ======================================================================
# MAIN
# ======================================================================

def main():
    # Arbeitsverzeichnis auf /workspace setzen
    os.chdir("/workspace")

    try:
        cfg = Config.from_ini()
    except (FileNotFoundError, KeyError) as exc:
        print(f"FEHLER Konfiguration: {exc}", file=sys.stderr)
        sys.exit(1)

    for d in cfg.all_dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger        = setup_logging(cfg.log_dir, logger_name="docking_restart",
                                   log_filename="restart.log")
    session_start = datetime.now()
    log_file      = cfg.log_dir / "pipeline.log"

    # ── Persistent Worker Modus ───────────────────────────────────
    if WORKER_PERSISTENT:
        logger.info("=== PERSISTENT RESTART WORKER GPU %d ===", WORKER_GPU_ID)
        logger.info("  Job-Verzeichnis: %s", WORKER_JOB_DIR)
        _run_persistent_loop(cfg, logger)
        return

    if not WORKER_TARGET:
        logger.error("WORKER_TARGET nicht gesetzt – Abbruch.")
        sys.exit(1)

    logger.info("=== RESTART WORKER GPU %d: Target '%s' ===",
                WORKER_GPU_ID, WORKER_TARGET)
    logger.info("  CUDA_VISIBLE_DEVICES: %s",
                os.environ.get("CUDA_VISIBLE_DEVICES", "nicht gesetzt"))

    # Target laden
    try:
        targets, warnings = parse_target_config(
            cfg.target_config_file, cfg.target_dir
        )
    except Exception as exc:
        logger.error("config.txt Fehler: %s", exc)
        sys.exit(1)

    for w in warnings:
        logger.warning(w)

    target = next((t for t in targets if t.name == WORKER_TARGET), None)
    if target is None:
        logger.error("Target '%s' nicht in config.txt.", WORKER_TARGET)
        sys.exit(1)

    logger.info("  Rezeptor : %s", target.pdbqt_path.name)
    logger.info("  CENTER   : %s", target.center)
    logger.info("  BOX_SIZE : %s", target.box_size)

    # Liganden bestimmen
    chunk_mode = False
    if WORKER_LIGAND_LIST:
        ligand_list_path = Path(WORKER_LIGAND_LIST)
        if ligand_list_path.exists():
            all_ligands = [
                Path(line.strip()) for line in
                ligand_list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            chunk_mode = True
            logger.info("  Chunk-Modus: %d Liganden aus %s",
                        len(all_ligands), WORKER_LIGAND_LIST)
        else:
            logger.error("WORKER_LIGAND_LIST %s nicht gefunden!",
                         WORKER_LIGAND_LIST)
            sys.exit(1)
    else:
        # Rekursiv – sdf_to_pdbqt.py legt Unterordner 0000/, 0001/, ... an
        all_ligands, _lig_warnings = find_ligand_files(
            cfg.pdbqt_dir, target.ligand_subdir
        )
        for _w in _lig_warnings:
            logger.warning(_w)

    # Fertige Liganden ermitteln
    target_results_dir = cfg.results_dir / target.name
    target_results_dir.mkdir(parents=True, exist_ok=True)

    completed_names, completed_energies = find_completed_ligands(
        target.name, target_results_dir, log_file, logger
    )

    remaining = [f for f in all_ligands if f.stem not in completed_names]
    skipped   = len(all_ligands) - len(remaining)

    logger.info(
        "  Gesamt: %d | Fertig: %d | Ausstehend: %d",
        len(all_ligands), skipped, len(remaining),
    )

    if not remaining:
        logger.info("  Alle Liganden bereits abgeschlossen ✓")
        # CSV prüfen und ggf. ergänzen
        csv_path = target_results_dir / f"docking_results_{target.name}.csv"
        existing = load_existing_csv(csv_path)
        existing_names = {r["ligand"] for r in existing}
        missing = completed_names - existing_names
        if missing:
            for name in missing:
                existing.append({
                    "ligand": name, "success": True,
                    "best_energy_kcal_mol": completed_energies.get(name),
                    "error": None,
                })
            write_results_csv(existing, target_results_dir, target.name)
            logger.info("  CSV ergänzt: %d fehlende Einträge hinzugefügt.", len(missing))
    else:
        # Vorhandene CSV laden
        csv_path = target_results_dir / f"docking_results_{target.name}.csv"
        existing_results  = load_existing_csv(csv_path)
        existing_by_name  = {r["ligand"]: r for r in existing_results}

        for name in completed_names:
            if name not in existing_by_name:
                existing_by_name[name] = {
                    "ligand": name, "success": True,
                    "best_energy_kcal_mol": completed_energies.get(name),
                    "error": None,
                }

        # Docking der verbleibenden Liganden
        total_remaining = len(remaining)

        logger.info(
            "  Restart-Docking GPU %d (Uni-Dock): %d ausstehend | "
            "%d/%d bereits fertig",
            WORKER_GPU_ID, total_remaining, skipped, len(all_ligands),
        )

        new_results = dock_batch_unidock(
            ligand_files=remaining,
            target=target,
            cfg=cfg,
            logger=logger,
        )

        for r in new_results:
            if not r["success"]:
                (target_results_dir / f"{r['ligand']}_ERROR.log").write_text(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} – {r['error']}\n"
                )

        # Ergebnisse zusammenführen und CSV schreiben
        for r in new_results:
            existing_by_name[r["ligand"]] = r

        all_results = list(existing_by_name.values())
        if chunk_mode and WORKER_CHUNK_ID:
            csv_out = write_results_csv(
                all_results, target_results_dir,
                f"{target.name}_{WORKER_CHUNK_ID}")
        else:
            csv_out = write_results_csv(all_results, target_results_dir, target.name)
        successful  = sum(1 for r in all_results if r["success"])

        logger.info(
            "  RESTART ABGESCHLOSSEN – OK: %d/%d | CSV: %s",
            successful, len(all_results), csv_out,
        )

        top_hits = sorted(
            [r for r in all_results if r["success"] and
             r["best_energy_kcal_mol"] is not None],
            key=lambda r: r["best_energy_kcal_mol"],
        )[:10]
        if top_hits:
            logger.info("  --- TOP 10 HITS ---")
            for rank, r in enumerate(top_hits, 1):
                logger.info("  #%-3d %-25s %.2f kcal/mol",
                            rank, r["ligand"], r["best_energy_kcal_mol"])

    duration = datetime.now() - session_start
    logger.info("  Laufzeit: %s", str(duration).split(".")[0])


# ======================================================================
# PERSISTENT WORKER LOOP
# ======================================================================

def _run_persistent_loop(cfg, logger):
    """
    Pollt Job-Verzeichnis auf .job-Dateien.
    Format: TARGET=<name>, CHUNK_FILE=<path>, CHUNK_ID=<id>
    Nach Docking → .done-Datei mit Exit-Code.
    SHUTDOWN-Sentinel beendet die Schleife.
    """
    import time as _time

    job_dir = Path(WORKER_JOB_DIR)
    if not job_dir.exists():
        logger.error("Job-Verzeichnis %s existiert nicht!", WORKER_JOB_DIR)
        sys.exit(1)

    # Targets einmalig laden
    targets, warnings = parse_target_config(
        cfg.target_config_file, cfg.target_dir
    )
    for w in warnings:
        logger.warning(w)
    targets_by_name = {t.name: t for t in targets}

    logger.info("  Bereit – warte auf Jobs in %s", WORKER_JOB_DIR)

    while True:
        # SHUTDOWN prüfen
        if (job_dir / "SHUTDOWN").exists():
            logger.info("  SHUTDOWN empfangen – beende Worker")
            break

        # Neue .job-Dateien suchen
        job_files = sorted(job_dir.glob("*.job"))
        if not job_files:
            _time.sleep(0.3)
            continue

        # Erstes Job-File verarbeiten
        job_file = job_files[0]
        job_data = {}
        try:
            for line in job_file.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    job_data[k.strip()] = v.strip()
        except Exception as exc:
            logger.error("Job-Datei %s nicht lesbar: %s", job_file, exc)
            job_file.unlink(missing_ok=True)
            continue

        target_name = job_data.get("TARGET", "")
        chunk_file  = job_data.get("CHUNK_FILE", "")
        chunk_id    = job_data.get("CHUNK_ID", "")

        # Job-Datei sofort löschen
        job_file.unlink(missing_ok=True)

        if not target_name or not chunk_file:
            logger.error("Ungültiger Job: %s", job_data)
            if chunk_id:
                (job_dir / f"{chunk_id}.done").write_text("1", encoding="utf-8")
            continue

        target = targets_by_name.get(target_name)
        if target is None:
            logger.error("Target '%s' unbekannt", target_name)
            if chunk_id:
                (job_dir / f"{chunk_id}.done").write_text("1", encoding="utf-8")
            continue

        # Liganden aus Chunk-Datei lesen
        chunk_path = Path(chunk_file)
        if not chunk_path.exists():
            logger.error("Chunk-Datei %s nicht gefunden", chunk_file)
            if chunk_id:
                (job_dir / f"{chunk_id}.done").write_text("1", encoding="utf-8")
            continue

        ligand_files = [
            Path(line.strip()) for line in
            chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        logger.info("=== JOB: %s | %s | %d Liganden ===",
                     chunk_id, target_name, len(ligand_files))

        # Docking
        rc = 0
        try:
            target_results_dir = cfg.results_dir / target.name
            target_results_dir.mkdir(parents=True, exist_ok=True)

            new_results = dock_batch_unidock(
                ligand_files=ligand_files,
                target=target,
                cfg=cfg,
                logger=logger,
            )

            for r in new_results:
                if not r["success"]:
                    (target_results_dir / f"{r['ligand']}_ERROR.log").write_text(
                        f"{datetime.now():%Y-%m-%d %H:%M:%S} – {r['error']}\n"
                    )

            successful = sum(1 for r in new_results if r["success"])
            logger.info("  DOCKING ABGESCHLOSSEN – OK: %d | FEHLER: %d",
                        successful, len(new_results) - successful)

            # CSV schreiben
            if new_results:
                csv_path = write_results_csv(
                    new_results, target_results_dir,
                    f"{target.name}_{chunk_id}")
                logger.info("  CSV: %s", csv_path)

        except Exception as exc:
            logger.error("  Job %s fehlgeschlagen: %s", chunk_id, exc,
                         exc_info=True)
            rc = 1

        # .done schreiben
        (job_dir / f"{chunk_id}.done").write_text(str(rc), encoding="utf-8")

    logger.info("=== PERSISTENT RESTART WORKER BEENDET ===")


if __name__ == "__main__":
    main()
