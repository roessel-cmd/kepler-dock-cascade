#!/usr/bin/env python3
"""
worker_rescore.py
=================
Stufe 3: RESCORING. Laeuft INNERHALB von rescoring-gpu.sif.
Wird vom Orchestrator pro GPU und Target gestartet.

Ersetzt den frueheren RESCORE_ONLY-Modus von worker_gpu.py, der mit der
Aufteilung in drei Container entfallen ist.

Warum es diesen Worker ueberhaupt braucht:
docking_rescore.py hat zwar ein eigenes main(), das aber ALLE Targets
nacheinander abarbeitet. Fuer die Target-Parallelisierung ueber mehrere
GPUs braucht der Orchestrator einen Einstiegspunkt, der genau EIN Target
bearbeitet – ausgewaehlt ueber WORKER_TARGET.

(gnina_refinement.py wertet WORKER_TARGET in seinem eigenen main() bereits
aus und wird vom Orchestrator direkt aufgerufen – dafuer ist hier nichts
noetig.)

Umgebungsvariablen (vom Orchestrator gesetzt):
  CUDA_VISIBLE_DEVICES  → welche physische GPU
  WORKER_TARGET         → Target-Name; leer = alle Targets
  WORKER_GPU_ID         → GPU-Index (fuer Logging)

Konfiguration: /app/pipeline_config.ini (= config/rescore.ini auf dem Host)
"""

from __future__ import annotations

import configparser
import os
import sys
from datetime import datetime
from pathlib import Path

from docking_rescore import (
    PIPELINE_CONFIG_FILE,
    RescoringConfig,
    _parse_target_config_standalone,
    _setup_logger,
    log_top10_ecr,
    rescore_target,
    rescore_target_blocked,
)

WORKER_TARGET = os.environ.get("WORKER_TARGET", "")
WORKER_GPU_ID = os.environ.get("WORKER_GPU_ID", "0")


def _require(p: configparser.ConfigParser, section: str, key: str) -> str:
    try:
        return p.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        raise KeyError(f"Pflichtparameter '[{section}] {key}' fehlt.")


def main() -> None:
    os.chdir("/workspace")

    ini = PIPELINE_CONFIG_FILE
    if not ini.exists():
        print(f"FEHLER: INI nicht gefunden: {ini}", file=sys.stderr)
        sys.exit(1)

    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read(ini, encoding="utf-8")

    try:
        target_dir  = Path(_require(p, "PATHS", "target_dir"))
        results_dir = Path(_require(p, "PATHS", "results_dir"))
        log_dir     = Path(_require(p, "PATHS", "log_dir"))
        cfg = RescoringConfig.from_ini(ini)
    except Exception as exc:                       # noqa: BLE001
        print(f"FEHLER Konfiguration: {exc}", file=sys.stderr)
        sys.exit(1)

    if not cfg.enabled:
        print("Rescoring deaktiviert ([RESCORE] enabled=false).")
        sys.exit(0)

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(log_dir)

    logger.info("=== RESCORING WORKER GPU %s ===", WORKER_GPU_ID)
    logger.info("  CUDA_VISIBLE_DEVICES: %s",
                os.environ.get("CUDA_VISIBLE_DEVICES", "nicht gesetzt"))
    logger.info("  Aktiv: Vina=ja | CNNaffinity=%s | CNNscore=%s | "
                "dLinF9XGB=%s | Dense=%s",
                cfg.cnnaffinity_enabled, cfg.cnnscore_enabled,
                cfg.deltalinf9xgb_enabled, cfg.dense_enabled)

    try:
        targets, warnings = _parse_target_config_standalone(
            target_dir / "config.txt", target_dir
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("config.txt Fehler: %s", exc)
        sys.exit(1)

    for w in warnings:
        logger.warning(w)

    if WORKER_TARGET:
        targets = [t for t in targets if t.name == WORKER_TARGET]
        if not targets:
            logger.error("Target '%s' nicht in config.txt gefunden.",
                         WORKER_TARGET)
            sys.exit(1)
        logger.info("  Nur Target '%s' (GPU %s)", WORKER_TARGET, WORKER_GPU_ID)

    if not targets:
        logger.error("Keine gueltigen Targets – Abbruch.")
        sys.exit(1)

    failures = 0
    for target in targets:
        started = datetime.now()
        logger.info("=== TARGET: %s ===", target.name)

        target_results_dir = results_dir / target.name
        if not target_results_dir.exists():
            logger.warning("  Kein RESULTS-Verzeichnis fuer '%s' – "
                           "uebersprungen.", target.name)
            continue

        try:
            # Blockmodus, wenn rescore_block_size > 0: scort in Bloecken und
            # sichert nach jedem einen Zwischenstand. Nur so ueberlebt ein
            # Target den Abbruch an der Walltime, ohne von vorn zu beginnen.
            if cfg.rescore_block_size > 0:
                results = rescore_target_blocked(
                    target=target,
                    target_results_dir=target_results_dir,
                    rescore_cfg=cfg,
                    logger=logger,
                )
            else:
                results = rescore_target(
                    target=target,
                    target_results_dir=target_results_dir,
                    rescore_cfg=cfg,
                    logger=logger,
                )
            log_top10_ecr(results, target.name, logger)
        except Exception as exc:                   # noqa: BLE001
            logger.error("  Rescoring '%s' fehlgeschlagen: %s",
                         target.name, exc, exc_info=True)
            failures += 1
            continue

        logger.info("  Laufzeit: %s",
                    str(datetime.now() - started).split(".")[0])

    logger.info("=== RESCORING WORKER BEENDET ===")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
