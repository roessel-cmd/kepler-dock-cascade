#!/usr/bin/env python3
"""
unidock_engine.py
=================
Batch-Docking-Engine auf Basis von Uni-Dock (dptech-corp/Uni-Dock).

Ersetzt den frueheren Ein-Ligand-pro-Prozess-Aufruf von Vina-GPU+.
Kernidee: EIN unidock-Prozess dockt einen ganzen Chunk. Rezeptor-Parsing,
Grid-Maps und GPU-Context werden ueber alle Liganden des Chunks amortisiert,
und Uni-Dock batcht die Liganden intern in einen gemeinsamen CUDA-Launch.

Wird importiert von worker_gpu.py und worker_restart.py.

Aufrufkonvention (identisch zur alten run_docking_for_target-Signatur):

    from unidock_engine import dock_batch_unidock
    results = dock_batch_unidock(ligand_files, target, cfg, logger)

Rueckgabe: Liste von Dicts mit den Keys
    ligand, success, best_energy_kcal_mol, error
– also exakt das Format, das write_results_csv() und merge_chunk_results()
bereits erwarten. Downstream (Rescoring, Refinement, ECR) bleibt unveraendert.

Posen werden als <ligand>_docked.pdbqt in results_dir/<target> abgelegt –
derselbe Dateiname wie bisher, damit docking_rescore.py nichts merkt.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path


# ======================================================================
# OUTPUT-PARSING
# ======================================================================

_VINA_RESULT_RE = re.compile(r"^REMARK\s+VINA\s+RESULT:\s*([-+]?\d+\.?\d*)")

# Fehlermuster, bei denen ein Retry mit halbierter Batch-Groesse sinnvoll ist
_OOM_PATTERNS = (
    "out of memory",
    "bad_alloc",
    "cudaerrormemoryallocation",
    "cuda error",
    "insufficient",
)


def parse_best_energy(pdbqt_path: Path) -> float | None:
    """Liest die beste Bindungsenergie aus einer Uni-Dock Output-PDBQT."""
    try:
        with open(pdbqt_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = _VINA_RESULT_RE.match(line)
                if match:
                    return float(match.group(1))
                # Ab MODEL 2 kommen nur noch schlechtere Posen
                if line.startswith("MODEL") and line.strip() != "MODEL 1":
                    break
    except (OSError, ValueError):
        return None
    return None


# ======================================================================
# KOMMANDO-AUFBAU
# ======================================================================

def build_unidock_cmd(
    receptor_path: Path,
    index_file: Path,
    out_dir: Path,
    center: list[float],
    box_size: list[float],
    cfg,
) -> list[str]:
    """
    Baut den unidock-Aufruf. --ligand_index statt --gpu_batch, weil die
    Chunk-Dateien des Orchestrators genau dieses Format haben (ein Pfad
    pro Zeile) und wir so nicht gegen das ARGV-Limit laufen.
    """
    cmd = [
        str(cfg.unidock_binary),
        "--receptor",     str(receptor_path.resolve()),
        "--ligand_index", str(index_file.resolve()),
        "--dir",          str(out_dir.resolve()),
        "--center_x",     str(center[0]),
        "--center_y",     str(center[1]),
        "--center_z",     str(center[2]),
        "--size_x",       str(box_size[0]),
        "--size_y",       str(box_size[1]),
        "--size_z",       str(box_size[2]),
        "--scoring",      str(cfg.unidock_scoring),
        "--num_modes",    str(cfg.num_modes),
        "--energy_range", str(cfg.energy_range),
        "--verbosity",    "1",
    ]

    # search_mode ist die empfohlene Kurzform fuer exhaustiveness + max_step.
    # Leerstring => explizite Werte verwenden.
    if cfg.unidock_search_mode:
        cmd += ["--search_mode", str(cfg.unidock_search_mode)]
    else:
        cmd += ["--exhaustiveness", str(cfg.exhaustiveness)]
        if cfg.unidock_max_step > 0:
            cmd += ["--max_step", str(cfg.unidock_max_step)]

    if cfg.unidock_max_gpu_memory > 0:
        cmd += ["--max_gpu_memory", str(cfg.unidock_max_gpu_memory)]

    if cfg.unidock_refine_step > 0:
        cmd += ["--refine_step", str(cfg.unidock_refine_step)]

    if cfg.unidock_seed != 0:
        cmd += ["--seed", str(cfg.unidock_seed)]

    return cmd


# ======================================================================
# EIN SUB-BATCH
# ======================================================================

def _run_sub_batch(
    ligands: list[Path],
    target,
    cfg,
    logger,
    target_results_dir: Path,
    depth: int = 0,
) -> list[dict]:
    """
    Dockt eine Teilmenge in EINEM unidock-Prozess.
    Bei GPU-OOM wird rekursiv halbiert (max. 4 Ebenen tief).
    """
    if not ligands:
        return []

    tmp_root = target_results_dir / ".unidock_tmp"
    tmp_dir  = tmp_root / uuid.uuid4().hex[:12]
    tmp_dir.mkdir(parents=True, exist_ok=True)

    index_file = tmp_dir / "ligands.idx"
    index_file.write_text(
        "\n".join(str(p) for p in ligands) + "\n", encoding="utf-8"
    )

    cmd = build_unidock_cmd(
        receptor_path=target.pdbqt_path,
        index_file=index_file,
        out_dir=tmp_dir,
        center=target.center,
        box_size=target.box_size,
        cfg=cfg,
    )

    started = time.time()
    stderr_tail = ""
    crashed     = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.unidock_timeout if cfg.unidock_timeout > 0 else None,
        )
        if proc.returncode != 0:
            crashed     = True
            stderr_tail = (proc.stderr or proc.stdout or "").strip()[-500:]
    except subprocess.TimeoutExpired:
        crashed     = True
        stderr_tail = f"TIMEOUT nach {cfg.unidock_timeout}s"
    except Exception as exc:                      # noqa: BLE001
        crashed     = True
        stderr_tail = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - started

    # ── OOM-Retry: Batch halbieren ────────────────────────────────────
    if crashed and len(ligands) > 1 and depth < 4 and \
            any(pat in stderr_tail.lower() for pat in _OOM_PATTERNS):
        logger.warning(
            "  Uni-Dock OOM bei %d Liganden – halbiere (Ebene %d)",
            len(ligands), depth + 1,
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        mid = len(ligands) // 2
        return (
            _run_sub_batch(ligands[:mid], target, cfg, logger,
                           target_results_dir, depth + 1)
            + _run_sub_batch(ligands[mid:], target, cfg, logger,
                             target_results_dir, depth + 1)
        )

    # ── Ergebnisse einsammeln ─────────────────────────────────────────
    # Uni-Dock schreibt <stem>_out.pdbqt nach --dir. Wir mappen defensiv:
    # exakter Name zuerst, sonst per Praefix.
    produced = {p.name: p for p in tmp_dir.glob("*.pdbqt")}
    results: list[dict] = []
    recovered = 0

    for lig in ligands:
        stem = lig.stem
        out  = produced.get(f"{stem}_out.pdbqt") or produced.get(f"{stem}.pdbqt")
        if out is None:
            for name, path in produced.items():
                if name.startswith(stem):
                    out = path
                    break

        if out is None or not out.exists():
            results.append({
                "ligand": stem,
                "success": False,
                "best_energy_kcal_mol": None,
                "error": stderr_tail or "keine Uni-Dock Ausgabe erzeugt",
            })
            continue

        energy = parse_best_energy(out)
        if energy is None:
            results.append({
                "ligand": stem,
                "success": False,
                "best_energy_kcal_mol": None,
                "error": "REMARK VINA RESULT nicht parsebar",
            })
            continue

        # Pose unter dem von der Pipeline erwarteten Namen ablegen
        final = target_results_dir / f"{stem}_docked.pdbqt"
        try:
            shutil.move(str(out), str(final))
        except OSError as exc:
            results.append({
                "ligand": stem,
                "success": False,
                "best_energy_kcal_mol": energy,
                "error": f"Pose nicht verschiebbar: {exc}",
            })
            continue

        recovered += 1
        results.append({
            "ligand": stem,
            "success": True,
            "best_energy_kcal_mol": energy,
            "error": None,
        })

    shutil.rmtree(tmp_dir, ignore_errors=True)

    per_lig = elapsed / max(1, len(ligands))
    logger.info(
        "  Sub-Batch: %d/%d OK | %.1fs gesamt | %.2fs/Ligand",
        recovered, len(ligands), elapsed, per_lig,
    )
    if crashed and recovered == 0:
        logger.error("  Uni-Dock Fehler: %s", stderr_tail[:300])

    return results


# ======================================================================
# OEFFENTLICHE API
# ======================================================================

def dock_batch_unidock(ligand_files, target, cfg, logger) -> list[dict]:
    """
    Dockt alle uebergebenen Liganden gegen ein Target.
    Zerlegt in Sub-Batches von cfg.unidock_batch_size.
    """
    target_results_dir = cfg.results_dir / target.name
    target_results_dir.mkdir(parents=True, exist_ok=True)

    ligand_files = [Path(p) for p in ligand_files]
    total        = len(ligand_files)
    batch_size   = max(1, cfg.unidock_batch_size)

    mode_desc = (cfg.unidock_search_mode
                 if cfg.unidock_search_mode
                 else f"exhaustiveness={cfg.exhaustiveness}")
    logger.info(
        "  Uni-Dock: %d Liganden | Batch %d | %s | scoring=%s",
        total, batch_size, mode_desc, cfg.unidock_scoring,
    )

    results: list[dict] = []
    started = time.time()

    for offset in range(0, total, batch_size):
        sub = ligand_files[offset:offset + batch_size]
        logger.info(
            "  Batch %d–%d von %d",
            offset + 1, min(offset + batch_size, total), total,
        )
        results.extend(
            _run_sub_batch(sub, target, cfg, logger, target_results_dir)
        )

    # Aufraeumen: leeres .unidock_tmp entfernen
    tmp_root = target_results_dir / ".unidock_tmp"
    if tmp_root.exists() and not any(tmp_root.iterdir()):
        tmp_root.rmdir()

    elapsed    = time.time() - started
    successful = sum(1 for r in results if r["success"])
    rate       = successful / elapsed * 3600 if elapsed > 0 else 0.0
    logger.info(
        "  Uni-Dock fertig: %d/%d OK | %.1fs | ~%.0f Liganden/h",
        successful, total, elapsed, rate,
    )

    return results
