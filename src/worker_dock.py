#!/usr/bin/env python3
"""
worker_dock.py
==============
Stufe 2: DOCKING. Laeuft INNERHALB des Uni-Dock-Containers.
Wird vom Orchestrator pro GPU gestartet.

Loest worker_gpu.py ab. Entfallen sind:
  - PREPARE_ONLY (Schritte 1-3)  → eigener Container: sdf_to_pdbqt.sif
  - RESCORE_ONLY                 → eigener Container: rescoring-gpu.sif
  - der CPU-Docking-Fallback     → Docking laeuft ausschliesslich auf GPU
  - split_sdf/convert_sdf_to_pdb/convert_ligand und deren Importe
    (RDKit, OpenBabel, AutoDockTools, vina)

Dadurch braucht dieser Worker im Container nur noch: Uni-Dock.

Umgebungsvariablen (vom Orchestrator gesetzt):
  CUDA_VISIBLE_DEVICES  → welche physische GPU
  WORKER_TARGET         → Target-Name aus config.txt
  WORKER_GPU_ID         → GPU-Index (fuer Logging)
  WORKER_LIGAND_LIST    → optional: Chunk-Datei mit Ligandenpfaden
  WORKER_CHUNK_ID       → optional: Chunk-Bezeichner fuer den CSV-Namen
  WORKER_PERSISTENT     → 1 = Job-Verzeichnis pollen statt einmalig laufen
  WORKER_JOB_DIR        → Job-Verzeichnis im Persistent-Modus
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from unidock_engine import dock_batch_unidock

from docking_config import (
    DockingConfig,
    parse_target_config,
    setup_logging,
)
from pipeline_common import find_ligand_files


# ======================================================================
# MODUS
# ======================================================================

WORKER_TARGET      = os.environ.get("WORKER_TARGET", "")
WORKER_GPU_ID      = int(os.environ.get("WORKER_GPU_ID", "0"))
WORKER_LIGAND_LIST = os.environ.get("WORKER_LIGAND_LIST", "")
WORKER_CHUNK_ID    = os.environ.get("WORKER_CHUNK_ID", "")
WORKER_PERSISTENT  = os.environ.get("WORKER_PERSISTENT", "") == "1"
WORKER_JOB_DIR     = os.environ.get("WORKER_JOB_DIR", "")


# ======================================================================
# CSV
# ======================================================================

def write_results_csv(results, target_results_dir: Path, target_name: str) -> Path:
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


def log_top_hits(results, logger, n: int = 10) -> None:
    top = sorted(
        [r for r in results if r["success"] and r["best_energy_kcal_mol"] is not None],
        key=lambda r: r["best_energy_kcal_mol"],
    )[:n]
    if not top:
        return
    logger.info("  --- TOP %d HITS ---", len(top))
    for rank, r in enumerate(top, 1):
        logger.info("  #%-3d %-25s %.2f kcal/mol",
                    rank, r["ligand"], r["best_energy_kcal_mol"])


# ======================================================================
# LIGANDEN BESTIMMEN
# ======================================================================

def resolve_ligands(cfg, target, logger) -> tuple[list[Path], bool]:
    """
    Rueckgabe: (Ligandenliste, chunk_mode)

    Chunk-Modus hat Vorrang: der Orchestrator hat die Liste bereits
    zusammengestellt und als Datei uebergeben.
    """
    if WORKER_LIGAND_LIST:
        list_path = Path(WORKER_LIGAND_LIST)
        if not list_path.exists():
            logger.error("WORKER_LIGAND_LIST %s nicht gefunden!", WORKER_LIGAND_LIST)
            sys.exit(1)
        ligands = [
            Path(line.strip())
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info("  Chunk-Modus: %d Liganden aus %s",
                    len(ligands), WORKER_LIGAND_LIST)
        return ligands, True

    ligands, warnings = find_ligand_files(cfg.pdbqt_dir, target.ligand_subdir)
    for w in warnings:
        logger.warning(w)
    return ligands, False


# ======================================================================
# EINMAL-MODUS: EIN TARGET AUF EINER GPU
# ======================================================================

def run_single(cfg, logger) -> None:
    if not WORKER_TARGET:
        logger.error("WORKER_TARGET nicht gesetzt – Abbruch.")
        sys.exit(1)

    logger.info("=== WORKER GPU %d: Target '%s' ===", WORKER_GPU_ID, WORKER_TARGET)
    logger.info("  CUDA_VISIBLE_DEVICES: %s",
                os.environ.get("CUDA_VISIBLE_DEVICES", "nicht gesetzt"))

    try:
        targets, warnings = parse_target_config(
            cfg.target_config_file, cfg.target_dir
        )
    except Exception as exc:                       # noqa: BLE001
        logger.error("config.txt Fehler: %s", exc)
        sys.exit(1)

    for w in warnings:
        logger.warning(w)

    target = next((t for t in targets if t.name == WORKER_TARGET), None)
    if target is None:
        logger.error("Target '%s' nicht in config.txt gefunden.", WORKER_TARGET)
        sys.exit(1)

    logger.info("  Rezeptor : %s", target.pdbqt_path.name)
    logger.info("  CENTER   : %s", target.center)
    logger.info("  BOX_SIZE : %s", target.box_size)

    ligand_files, chunk_mode = resolve_ligands(cfg, target, logger)
    if not ligand_files:
        logger.error("Keine Liganden-PDBQTs gefunden.")
        sys.exit(1)

    started = datetime.now()
    results = dock_batch_unidock(ligand_files, target, cfg, logger)

    target_results_dir = cfg.results_dir / target.name
    for r in results:
        if not r["success"]:
            (target_results_dir / f"{r['ligand']}_ERROR.log").write_text(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} – {r['error']}\n"
            )

    successful = sum(1 for r in results if r["success"])
    logger.info("  DOCKING ABGESCHLOSSEN – OK: %d | FEHLER: %d",
                successful, len(results) - successful)

    if results:
        name = (f"{target.name}_{WORKER_CHUNK_ID}"
                if chunk_mode and WORKER_CHUNK_ID else target.name)
        logger.info("  CSV: %s", write_results_csv(results, target_results_dir, name))

    log_top_hits(results, logger)
    logger.info("  Laufzeit: %s", str(datetime.now() - started).split(".")[0])


# ======================================================================
# PERSISTENT-MODUS
# ======================================================================

def run_persistent(cfg, logger) -> None:
    """
    Pollt ein Job-Verzeichnis auf .job-Dateien:
        TARGET=<name>
        CHUNK_FILE=<pfad>
        CHUNK_ID=<id>
    Schreibt nach jedem Job <chunk_id>.done mit dem Exit-Code.
    SHUTDOWN-Sentinel beendet die Schleife.
    """
    job_dir = Path(WORKER_JOB_DIR)
    if not job_dir.exists():
        logger.error("Job-Verzeichnis %s existiert nicht!", WORKER_JOB_DIR)
        sys.exit(1)

    targets, warnings = parse_target_config(cfg.target_config_file, cfg.target_dir)
    for w in warnings:
        logger.warning(w)
    targets_by_name = {t.name: t for t in targets}

    logger.info("=== PERSISTENT WORKER GPU %d ===", WORKER_GPU_ID)
    logger.info("  Bereit – warte auf Jobs in %s", WORKER_JOB_DIR)

    while True:
        if (job_dir / "SHUTDOWN").exists():
            logger.info("  SHUTDOWN empfangen – beende Worker")
            break

        job_files = sorted(job_dir.glob("*.job"))
        if not job_files:
            time.sleep(0.3)
            continue

        job_file = job_files[0]
        job_data: dict[str, str] = {}
        try:
            for line in job_file.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    job_data[k.strip()] = v.strip()
        except Exception as exc:                   # noqa: BLE001
            logger.error("Job-Datei %s nicht lesbar: %s", job_file, exc)
            job_file.unlink(missing_ok=True)
            continue

        target_name = job_data.get("TARGET", "")
        chunk_file  = job_data.get("CHUNK_FILE", "")
        chunk_id    = job_data.get("CHUNK_ID", "")

        # Sofort loeschen, damit kein anderer Prozess den Job sieht
        job_file.unlink(missing_ok=True)

        def fail(msg: str) -> None:
            logger.error(msg)
            if chunk_id:
                (job_dir / f"{chunk_id}.done").write_text("1", encoding="utf-8")

        if not target_name or not chunk_file:
            fail(f"Ungueltiger Job: {job_data}")
            continue

        target = targets_by_name.get(target_name)
        if target is None:
            fail(f"Target '{target_name}' unbekannt")
            continue

        chunk_path = Path(chunk_file)
        if not chunk_path.exists():
            fail(f"Chunk-Datei {chunk_file} nicht gefunden")
            continue

        ligand_files = [
            Path(line.strip())
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        logger.info("=== JOB: %s | %s | %d Liganden ===",
                    chunk_id, target_name, len(ligand_files))

        rc = 0
        try:
            results = dock_batch_unidock(ligand_files, target, cfg, logger)

            target_results_dir = cfg.results_dir / target.name
            target_results_dir.mkdir(parents=True, exist_ok=True)
            for r in results:
                if not r["success"]:
                    (target_results_dir / f"{r['ligand']}_ERROR.log").write_text(
                        f"{datetime.now():%Y-%m-%d %H:%M:%S} – {r['error']}\n"
                    )

            successful = sum(1 for r in results if r["success"])
            logger.info("  DOCKING ABGESCHLOSSEN – OK: %d | FEHLER: %d",
                        successful, len(results) - successful)

            if results:
                csv_path = write_results_csv(
                    results, target_results_dir, f"{target.name}_{chunk_id}"
                )
                logger.info("  CSV: %s", csv_path)
        except Exception as exc:                   # noqa: BLE001
            logger.error("  Job %s fehlgeschlagen: %s", chunk_id, exc, exc_info=True)
            rc = 1

        (job_dir / f"{chunk_id}.done").write_text(str(rc), encoding="utf-8")

    logger.info("=== PERSISTENT WORKER BEENDET ===")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    # Relative Pfade aus der INI (./data/..., ./RESULTS) gegen /workspace
    # aufloesen – dort bindet der Orchestrator den Projektordner ein.
    os.chdir("/workspace")

    try:
        cfg = DockingConfig.from_ini()
    except (FileNotFoundError, KeyError) as exc:
        print(f"FEHLER Konfiguration: {exc}", file=sys.stderr)
        sys.exit(1)

    for d in cfg.all_dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg.log_dir)

    if WORKER_PERSISTENT:
        run_persistent(cfg, logger)
    else:
        run_single(cfg, logger)


if __name__ == "__main__":
    main()
