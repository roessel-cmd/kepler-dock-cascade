#!/usr/bin/env python3
"""
orchestrator.py
===============
Läuft AUSSERHALB des Apptainer-Containers auf dem Host.
Benötigt nur Python 3.8+ ohne externe Abhängigkeiten.

Ablauf:
  1. Schritte 1-3 (SDF→PDB→PDBQT) in einem Vorbereitungs-Container
  2. Targets einlesen, Liganden pro Target in Chunks aufteilen
  3. Pro GPU EINEN langlebigen Container starten (Persistent Worker)
  4. Chunks als Job-Dateien ins Dateisystem schreiben
     → Worker pollt, dockt, markiert als fertig
  5. Idle GPUs helfen bei verbleibenden Chunks anderer Targets
  6. SHUTDOWN-Sentinel beendet Worker → Chunk-CSVs mergen → Rescoring

Verwendung:
  python3 orchestrator.py [--config pipeline_config.ini]
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import re
import subprocess
import sys
import threading

from pipeline_common import find_ligand_files
import time
from datetime import datetime
from pathlib import Path


# ======================================================================
# KONFIGURATION
# ======================================================================

def load_ini(ini_path):
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read(ini_path, encoding="utf-8")
    return p

def get(p, s, k, fallback=None): return p.get(s, k, fallback=fallback)
def getint(p, s, k, fallback=0): return p.getint(s, k, fallback=fallback)
def getbool(p, s, k, fallback=False): return p.getboolean(s, k, fallback=fallback)


# ======================================================================
# GPU ERKENNUNG
# ======================================================================

def compute_cap_to_arch(cap):
    cap = cap.replace(".", "")
    return {"80": "ampere", "86": "ampere", "87": "ampere",
            "89": "ada", "90": "hopper", "90a": "hopper",
            "100": "blackwell", "100a": "blackwell",
            "120": "blackwell", "120a": "blackwell"}.get(cap, "")

def detect_gpus(project_dir):
    project_dir = project_dir.resolve()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3: continue
        idx, name, cap = int(parts[0]), parts[1], parts[2].replace(".", "")
        arch = compute_cap_to_arch(cap)
        kd = project_dir / "kernels" / arch if arch else None
        k1 = kd / "Kernel1_Opt.bin" if kd else None
        k2 = kd / "Kernel2_Opt.bin" if kd else None
        gpus.append({
            "index": idx, "name": name, "cap": cap, "arch": arch,
            "kernel_dir": kd,
            "kernel_k1": k1 if (k1 and k1.exists()) else None,
            "kernel_k2": k2 if (k2 and k2.exists()) else None,
        })
    return gpus


# ======================================================================
# LOGGING
# ======================================================================

_log_lock = threading.Lock()

def log(msg, tag="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    with _log_lock:
        print(f"{ts} [ORCH {tag:8s}] {msg}", flush=True)


# ======================================================================
# CONTAINER AUFRUFE
# ======================================================================

# ======================================================================
# PARALLEL GPU RESCORING
# ======================================================================

def _run_rescore_worker(gpu, sif, project_dir, ini_path,
                        target_name, results):
    idx = gpu["index"]
    log(f"GPU {idx}: Rescoring '{target_name}'")

    # Conda-Environment: rescoring-gpu.sif → rescore_env, sonst docking_env
    conda_env = "rescore_env" if "rescoring" in str(sif) else "docking_env"

    cmd = [
        "apptainer", "exec", "--nv",
        "--pwd", "/workspace",
        "--bind", f"{project_dir}:/workspace",
        "--bind", f"{ini_path}:/app/pipeline_config.ini",
        "--env", "ORCHESTRATOR_RESCORE_ONLY=1",
        "--env", f"CUDA_VISIBLE_DEVICES={idx}",
        "--env", f"WORKER_TARGET={target_name}",
        "--env", f"WORKER_GPU_ID={idx}",
    ]
    # Lokale Quellen binden (Entwicklermodus)
    for pyfile in ["worker_rescore.py", "docking_rescore.py",
                    "gnina_gpu_worker.py", "gnina_refinement.py",
                    "linf9xgb_scorer.py", "ecr.py"]:
        local = project_dir / "src" / pyfile
        if local.exists():
            cmd += ["--bind", f"{local}:/app/{pyfile}"]
    cmd += [str(sif), "bash", "-c",
        f"source /opt/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {conda_env} && python /app/worker_rescore.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log(line.rstrip(), tag=f"RS-G{idx}")
    proc.wait()
    rc = proc.returncode
    results[target_name] = rc
    tag = "OK" if rc == 0 else "ERROR"
    log(f"GPU {idx}: Rescoring '{target_name}' "
        f"{'✓' if rc == 0 else f'Fehler (Exit {rc})'}", tag=tag)

def _rescore_worker_with_queue(gpu, first_target, tq, ql,
                               sif, project_dir, ini_path, results):
    target = first_target
    while target is not None:
        _run_rescore_worker(gpu, sif, project_dir, ini_path,
                            target["name"], results)
        with ql:
            target = tq.pop(0) if tq else None

def run_rescoring(sif, project_dir, ini_path, targets=None, gpus=None):
    log("=== RESCORING STARTEN ===")
    if gpus and targets:
        log(f"  Parallel: {len(targets)} Target(s) auf {len(gpus)} GPU(s)")
        tq, ql, res, threads = list(targets), threading.Lock(), {}, []
        for gpu in gpus:
            if not tq: break
            with ql:
                t = tq.pop(0) if tq else None
            if t is None: break
            th = threading.Thread(
                target=_rescore_worker_with_queue,
                args=(gpu, t, tq, ql, sif, project_dir, ini_path, res),
                daemon=True)
            th.start(); threads.append(th)
        for th in threads: th.join()
        ok  = sum(1 for r in res.values() if r == 0)
        err = sum(1 for r in res.values() if r != 0)
        log(f"=== RESCORING FERTIG – OK: {ok} | Fehler: {err} ===")
        return 1 if err > 0 else 0
    # Fallback
    log("  Fallback: einzelner Container")
    conda_env = "rescore_env" if "rescoring" in str(sif) else "docking_env"
    cmd = [
        "apptainer", "exec", "--nv", "--pwd", "/workspace",
        "--bind", f"{project_dir}:/workspace",
        "--bind", f"{ini_path}:/app/pipeline_config.ini",
        "--env", "ORCHESTRATOR_RESCORE_ONLY=1",
    ]
    for pyfile in ["worker_rescore.py", "docking_rescore.py",
                    "gnina_gpu_worker.py", "gnina_refinement.py",
                    "linf9xgb_scorer.py", "ecr.py"]:
        local = project_dir / "src" / pyfile
        if local.exists():
            cmd += ["--bind", f"{local}:/app/{pyfile}"]
    cmd += [str(sif), "bash", "-c",
        f"source /opt/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {conda_env} && python /app/worker_rescore.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log(line.rstrip(), tag="RESCORE")
    proc.wait()
    return proc.returncode


# ======================================================================
# PARALLEL GPU REFINEMENT
# ======================================================================

def _run_refine_worker(gpu, sif, project_dir, ini_path,
                       target_name, results):
    idx = gpu["index"]
    log(f"GPU {idx}: Refinement '{target_name}'")

    conda_env = "rescore_env" if "rescoring" in str(sif) else "docking_env"

    cmd = [
        "apptainer", "exec", "--nv",
        "--pwd", "/workspace",
        "--bind", f"{project_dir}:/workspace",
        "--bind", f"{ini_path}:/app/pipeline_config.ini",
        "--env", f"CUDA_VISIBLE_DEVICES={idx}",
        "--env", f"WORKER_TARGET={target_name}",
        "--env", f"WORKER_GPU_ID={idx}",
    ]
    # Lokale Quellen binden (Entwicklermodus)
    for pyfile in ["gnina_refinement.py", "docking_rescore.py",
                    "gnina_gpu_worker.py", "linf9xgb_scorer.py"]:
        local = project_dir / "src" / pyfile
        if local.exists():
            cmd += ["--bind", f"{local}:/app/{pyfile}"]
    cmd += [str(sif), "bash", "-c",
        f"source /opt/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {conda_env} && python /app/gnina_refinement.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log(line.rstrip(), tag=f"RF-G{idx}")
    proc.wait()
    rc = proc.returncode
    results[target_name] = rc
    tag = "OK" if rc == 0 else "ERROR"
    log(f"GPU {idx}: Refinement '{target_name}' "
        f"{'✓' if rc == 0 else f'Fehler (Exit {rc})'}", tag=tag)

def _refine_worker_with_queue(gpu, first_target, tq, ql,
                              sif, project_dir, ini_path, results):
    target = first_target
    while target is not None:
        _run_refine_worker(gpu, sif, project_dir, ini_path,
                           target["name"], results)
        with ql:
            target = tq.pop(0) if tq else None

def run_refinement(sif, project_dir, ini_path, targets=None, gpus=None):
    log("=== REFINEMENT STARTEN ===")
    if gpus and targets:
        log(f"  Parallel: {len(targets)} Target(s) auf {len(gpus)} GPU(s)")
        tq, ql, res, threads = list(targets), threading.Lock(), {}, []
        for gpu in gpus:
            if not tq: break
            with ql:
                t = tq.pop(0) if tq else None
            if t is None: break
            th = threading.Thread(
                target=_refine_worker_with_queue,
                args=(gpu, t, tq, ql, sif, project_dir, ini_path, res),
                daemon=True)
            th.start(); threads.append(th)
        for th in threads: th.join()
        ok  = sum(1 for r in res.values() if r == 0)
        err = sum(1 for r in res.values() if r != 0)
        log(f"=== REFINEMENT FERTIG – OK: {ok} | Fehler: {err} ===")
        return 1 if err > 0 else 0
    # Fallback: einzelner Container
    log("  Fallback: einzelner Container")
    conda_env = "rescore_env" if "rescoring" in str(sif) else "docking_env"
    cmd = [
        "apptainer", "exec", "--nv", "--pwd", "/workspace",
        "--bind", f"{project_dir}:/workspace",
        "--bind", f"{ini_path}:/app/pipeline_config.ini",
    ]
    for pyfile in ["gnina_refinement.py", "docking_rescore.py",
                    "gnina_gpu_worker.py", "linf9xgb_scorer.py"]:
        local = project_dir / "src" / pyfile
        if local.exists():
            cmd += ["--bind", f"{local}:/app/{pyfile}"]
    cmd += [str(sif), "bash", "-c",
        f"source /opt/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {conda_env} && python /app/gnina_refinement.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log(line.rstrip(), tag="REFINE")
    proc.wait()
    return proc.returncode


# ======================================================================
# PERSISTENT WORKER – Container starten & stoppen
# ======================================================================

def start_persistent_worker(gpu, sif, project_dir, ini_path, job_dir):
    """
    Startet EINEN langlebigen Container für eine GPU.
    Der Worker darin pollt job_dir auf neue Job-Dateien.
    Gibt das Popen-Objekt zurück.
    """
    idx = gpu["index"]
    k1  = gpu["kernel_k1"]
    k2  = gpu["kernel_k2"]

    cmd = [
        "apptainer", "exec", "--nv",
        "--pwd", "/workspace",
        "--bind", f"{project_dir}:/workspace",
        "--bind", f"{ini_path}:/app/pipeline_config.ini",
    ]
    if k1 and k2:
        cmd += [
            "--bind", f"{k1}:/opt/vina-opencl/Kernel1_Opt.bin",
            "--bind", f"{k2}:/opt/vina-opencl/Kernel2_Opt.bin",
        ]
    for _name in ("worker_dock.py", "unidock_engine.py",
                  "docking_config.py", "pipeline_common.py"):
        _src = project_dir / "src" / _name
        if _src.exists():
            cmd += ["--bind", f"{_src}:/app/{_name}"]

    # Container-Pfad für job_dir: /workspace/data/LOG/jobs/gpu{N}
    container_job_dir = f"/workspace/{job_dir.relative_to(project_dir)}"

    cmd += [
        "--env", f"CUDA_VISIBLE_DEVICES={idx}",
        "--env", f"WORKER_GPU_ID={idx}",
        "--env", "WORKER_PERSISTENT=1",
        "--env", f"WORKER_JOB_DIR={container_job_dir}",
        str(sif), "bash", "-c",
        "source /opt/miniconda3/etc/profile.d/conda.sh && "
        "conda activate docking_env && python /app/worker_dock.py",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    log(f"GPU {idx} ({gpu['name']}): Persistent Worker gestartet "
        f"(PID {proc.pid})", tag="START")

    return proc


def stream_worker_output(proc, gpu_idx):
    """Liest stdout des Persistent Workers und gibt es an den Logger weiter."""
    tag = f"GPU{gpu_idx}"
    for line in proc.stdout:
        log(line.rstrip(), tag=tag)
    proc.wait()


def shutdown_worker(job_dir):
    """Schreibt SHUTDOWN-Sentinel → Worker beendet sich."""
    (job_dir / "SHUTDOWN").write_text("", encoding="utf-8")


# ======================================================================
# CHUNK-INFRASTRUKTUR
# ======================================================================

class LigandChunkQueue:
    """Thread-sichere Queue. get_next_chunk() ist atomar."""

    def __init__(self, target_name, ligand_files, chunk_size=200):
        self.target_name = target_name
        self._lock = threading.Lock()
        self.chunks = [
            ligand_files[i:i + chunk_size]
            for i in range(0, len(ligand_files), chunk_size)
        ]
        self._next_idx = 0
        self.total = len(self.chunks)

    def get_next_chunk(self):
        with self._lock:
            if self._next_idx >= self.total:
                return None
            idx = self._next_idx
            self._next_idx += 1
            return idx, self.chunks[idx]

    @property
    def remaining(self):
        with self._lock:
            return self.total - self._next_idx


def _get_ligand_files(target, pdbqt_dir):
    """Rekursive Ligandensuche – sdf_to_pdbqt.py legt 0000/, 0001/, ... an."""
    files, warnings = find_ligand_files(
        pdbqt_dir, target.get("ligand_subdir")
    )
    for w in warnings:
        log(w, tag="WARN")
    return files


def write_chunk_file(project_dir, target_name, gpu_idx, chunk_idx,
                     chunk_files):
    """Schreibt Chunk-Datei mit Container-Pfaden."""
    path = (project_dir / "data" / "LOG" /
            f"chunk_gpu{gpu_idx}_{target_name}_{chunk_idx}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for lig in chunk_files:
            rel = lig.relative_to(project_dir)
            f.write(f"/workspace/{rel}\n")
    return path


def merge_chunk_results(target_name, results_dir):
    target_dir = results_dir / target_name
    chunk_files = sorted(
        target_dir.glob(f"docking_results_{target_name}_chunk*.csv"))
    if not chunk_files:
        return None
    merged_path = target_dir / f"docking_results_{target_name}.csv"
    by_ligand = {}
    for cf in chunk_files:
        with open(cf, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                lig = row.get("ligand", "")
                if lig:
                    by_ligand[lig] = row
    if not by_ligand:
        return None
    all_rows = list(by_ligand.values())
    def sort_key(r):
        try: return (0, float(r.get("best_energy_kcal_mol", "")))
        except (ValueError, TypeError): return (1, 0.0)
    all_rows.sort(key=sort_key)
    fieldnames = ["ligand", "success", "best_energy_kcal_mol", "error"]
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    log(f"  Merge '{target_name}': {len(chunk_files)} CSVs → "
        f"{len(all_rows)} Liganden")
    return merged_path


# ======================================================================
# TARGET-KONFIGURATION
# ======================================================================

def parse_targets(config_txt, target_dir):
    if not config_txt.exists(): return []
    targets, current = [], {}
    for raw in config_txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            if current and "name" in current:
                if (target_dir / f"{current['name']}.pdbqt").exists():
                    targets.append(current)
                current = {}
            continue
        if line.upper().startswith("CENTER"):
            m = re.search(r"\[([^\]]+)\]", line)
            if m: current["center"] = [float(x) for x in m.group(1).split(",")]
        elif line.upper().startswith("BOX_SIZE"):
            m = re.search(r"\[([^\]]+)\]", line)
            if m: current["box_size"] = [float(x) for x in m.group(1).split(",")]
        elif "=" in line and line.upper().startswith("LIGAND_SUBDIR"):
            current["ligand_subdir"] = line.split("=", 1)[1].strip()
        elif re.match(r"^[\w\-]+$", line):
            if current and "name" in current:
                if (target_dir / f"{current['name']}.pdbqt").exists():
                    targets.append(current)
            current = {"name": line}
    if current and "name" in current:
        if (target_dir / f"{current['name']}.pdbqt").exists():
            targets.append(current)
    return targets


# ======================================================================
# GPU THREAD – Chunks über Persistent Worker dispatchen
# ======================================================================

def _gpu_chunk_worker(
    gpu, first_target, target_queue, queue_lock,
    chunk_queues, project_dir, job_dir, results,
):
    """
    Orchestrator-Thread pro GPU.
    Schreibt Job-Dateien und wartet auf .done — der Container
    läuft dauerhaft im Hintergrund.
    """
    gpu_idx = gpu["index"]
    target  = first_target

    while True:
        if target is not None:
            _process_target_chunks(
                gpu, target, chunk_queues, project_dir, job_dir, results,
            )

        with queue_lock:
            target = target_queue.pop(0) if target_queue else None
        if target is not None:
            continue

        helped = _help_any_target(
            gpu, chunk_queues, project_dir, job_dir, results,
        )
        if helped:
            with queue_lock:
                target = target_queue.pop(0) if target_queue else None
            if target is not None:
                continue
            continue

        break

    log(f"GPU {gpu_idx}: Alle Arbeit erledigt", tag="DONE")


def _dispatch_chunk(gpu, target_name, chunk_idx, chunk_files,
                    project_dir, job_dir, results):
    """Schreibt einen Chunk-Job und wartet auf Abschluss."""
    gpu_idx  = gpu["index"]
    chunk_id = f"chunk_gpu{gpu_idx}_{target_name}_{chunk_idx}"

    # Chunk-Datei mit Ligand-Pfaden schreiben
    chunk_file = write_chunk_file(
        project_dir, target_name, gpu_idx, chunk_idx, chunk_files,
    )

    # Job-Datei schreiben → Worker sieht sie sofort
    job_file = job_dir / f"{chunk_id}.job"
    job_file.write_text(
        f"TARGET={target_name}\n"
        f"CHUNK_FILE=/workspace/{chunk_file.relative_to(project_dir)}\n"
        f"CHUNK_ID={chunk_id}\n",
        encoding="utf-8",
    )

    # Auf Abschluss warten
    done_file = job_dir / f"{chunk_id}.done"
    while not done_file.exists():
        time.sleep(0.5)

    try:
        rc = int(done_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        rc = 0

    results[f"{target_name}_chunk{chunk_idx}"] = rc
    return rc


def _process_target_chunks(gpu, target, chunk_queues,
                           project_dir, job_dir, results):
    gpu_idx  = gpu["index"]
    tgt_name = target["name"]
    cq = chunk_queues.get(tgt_name)
    if not cq:
        log(f"GPU {gpu_idx}: Keine Chunks für '{tgt_name}'", tag="WARN")
        return

    log(f"GPU {gpu_idx}: Starte Target '{tgt_name}' "
        f"({cq.total} Chunks)", tag="START")

    while True:
        chunk_data = cq.get_next_chunk()
        if chunk_data is None:
            break
        chunk_idx, chunk_files = chunk_data

        log(f"GPU {gpu_idx}: {tgt_name} Chunk "
            f"{chunk_idx + 1}/{cq.total} ({len(chunk_files)} Lig.)")

        _dispatch_chunk(gpu, tgt_name, chunk_idx, chunk_files,
                        project_dir, job_dir, results)

    log(f"GPU {gpu_idx}: Target '{tgt_name}' fertig ✓", tag="OK")


def _help_any_target(gpu, chunk_queues, project_dir, job_dir,
                     results):
    gpu_idx = gpu["index"]
    helped = False
    for tgt_name, cq in chunk_queues.items():
        while True:
            chunk_data = cq.get_next_chunk()
            if chunk_data is None:
                break
            helped = True
            chunk_idx, chunk_files = chunk_data
            log(f"GPU {gpu_idx}: Hilft bei '{tgt_name}' Chunk "
                f"{chunk_idx + 1}/{cq.total}", tag="HELP")
            _dispatch_chunk(gpu, tgt_name, chunk_idx, chunk_files,
                            project_dir, job_dir, results)
    return helped


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Docking Pipeline Orchestrator")
    parser.add_argument("--config", default="pipeline_config.ini")
    parser.add_argument("--project", default=".")
    parser.add_argument("--stage", choices=["dock", "rescore"], default="dock",
                        help="dock = nur Docking, rescore = nur Rescoring "
                             "(+ Refinement). Gesteuert von pipeline_start.sh.")
    parser.add_argument("--sif", default="",
                        help="Container-Image; leer = automatisch nach Stage.")
    args = parser.parse_args()
    STAGE = args.stage

    project_dir = Path(args.project).resolve()
    ini_path    = project_dir / args.config
    if not ini_path.exists():
        print(f"FEHLER: {ini_path} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    cfg = load_ini(ini_path)
    session_start = datetime.now()

    log("=" * 60)
    log(f"  DOCKING ORCHESTRATOR (Persistent Worker)")
    log(f"  Gestartet: {session_start:%Y-%m-%d %H:%M:%S}")
    log(f"  Projekt:   {project_dir}")
    log("=" * 60)

    if args.sif:
        sif = Path(args.sif)
        if not sif.is_absolute():
            sif = project_dir / sif
    elif STAGE == "dock":
        sif = project_dir / "unidock-gpu.sif"
    else:
        sif = project_dir / "rescoring-gpu.sif"

    if not sif.exists():
        log(f"FEHLER: Container nicht gefunden: {sif}", tag="ERROR")
        sys.exit(1)

    sif_rescore = sif   # In der Stage-Architektur ist es derselbe Container
    log(f"  Stage:     {STAGE}")
    log(f"  Container: {sif.name}")
    log(f"  INI:       {ini_path.name}")

    # ── GPUs ──────────────────────────────────────────────────────
    num_gpus_cfg   = getint(cfg, "GPU", "num_gpus", fallback=1)
    cuda_device_id = getint(cfg, "GPU", "cuda_device_id", fallback=0)
    # Kein CPU-Pfad mehr – Docking laeuft ausschliesslich auf GPU.
    use_gpu        = True

    if use_gpu:
        all_gpus = detect_gpus(project_dir)
        if num_gpus_cfg == 1:
            # Single-GPU-Modus: die per cuda_device_id konfigurierte GPU verwenden
            gpus = [g for g in all_gpus if g["index"] == cuda_device_id]
            if not gpus:
                log(f"FEHLER: GPU {cuda_device_id} (cuda_device_id) nicht gefunden! "
                    f"Verfügbar: {[g['index'] for g in all_gpus]}", tag="ERROR")
                sys.exit(1)
        else:
            # Multi-GPU-Modus: erste N GPUs (cuda_device_id wird ignoriert)
            gpus = all_gpus[:num_gpus_cfg]
        log(f"GPUs: {len(all_gpus)} erkannt | {len(gpus)} verwendet")
        for g in gpus:
            ks = f"Kernel: {g['arch']}" if g["kernel_k1"] else "FEHLT"
            log(f"  GPU {g['index']}: {g['name']} (sm_{g['cap']}) | {ks}")
        if not gpus:
            log("FEHLER: Keine GPUs!", tag="ERROR"); sys.exit(1)
    else:
        gpus = []

    # ── Targets ───────────────────────────────────────────────────
    target_dir = project_dir / get(cfg, "PATHS", "target_dir",
                                   "./TARGET").lstrip("./")
    targets = parse_targets(target_dir / "config.txt", target_dir)
    if not targets:
        log("FEHLER: Keine Targets.", tag="ERROR"); sys.exit(1)
    log(f"Targets: {len(targets)} – "
        f"{', '.join(t['name'] for t in targets)}")

    # Die Ligandenaufbereitung ist Stufe 1 (sdf_to_pdbqt.sif) und wird
    # von pipeline_start.sh gestartet, nicht mehr von hier.
    rescore_only = (STAGE == "rescore")

    # ── Config ────────────────────────────────────────────────────
    chunk_size  = getint(cfg, "CHUNK", "chunk_size", fallback=200)
    pdbqt_dir   = project_dir / get(cfg, "PATHS", "pdbqt_dir",
                                    "./data/PDBQT").lstrip("./")
    results_dir = project_dir / get(cfg, "PATHS", "results_dir",
                                    "./RESULTS").lstrip("./")

    # ══════════════════════════════════════════════════════════════
    #  DOCKING
    # ══════════════════════════════════════════════════════════════
    if STAGE == "dock" and use_gpu:
        log("=" * 60)
        log(f"=== DOCKING: {len(targets)} Targets | {len(gpus)} GPUs | "
            f"Chunks à {chunk_size} ===")
        log("=" * 60)

        # ── Chunk-Queues vorab erstellen ──────────────────────────
        chunk_queues = {}
        for target in targets:
            lf = _get_ligand_files(target, pdbqt_dir)
            if not lf:
                log(f"  WARNUNG: 0 Liganden für '{target['name']}'", tag="WARN")
                continue
            cq = LigandChunkQueue(target["name"], lf, chunk_size)
            chunk_queues[target["name"]] = cq
            log(f"  {target['name']}: {len(lf)} Lig. → {cq.total} Chunks")
            # Alte Chunk-CSVs löschen
            tr = results_dir / target["name"]
            if tr.exists():
                for old in tr.glob(f"docking_results_{target['name']}_chunk*.csv"):
                    old.unlink()

        # ── Job-Verzeichnisse erstellen ───────────────────────────
        jobs_base = project_dir / "data" / "LOG" / "jobs"
        gpu_job_dirs = {}
        for gpu in gpus:
            jd = jobs_base / f"gpu{gpu['index']}"
            # Aufräumen von vorherigen Runs
            if jd.exists():
                for f in jd.iterdir():
                    f.unlink()
            jd.mkdir(parents=True, exist_ok=True)
            gpu_job_dirs[gpu["index"]] = jd

        # ── Persistent Worker starten (1 Container pro GPU) ───────
        worker_procs   = {}
        output_threads = {}
        for gpu in gpus:
            jd = gpu_job_dirs[gpu["index"]]
            proc = start_persistent_worker(gpu, sif, project_dir, ini_path, jd)
            worker_procs[gpu["index"]] = proc
            # Stdout in separatem Thread lesen
            ot = threading.Thread(
                target=stream_worker_output,
                args=(proc, gpu["index"]),
                daemon=True,
                name=f"stdout-gpu{gpu['index']}",
            )
            ot.start()
            output_threads[gpu["index"]] = ot

        # Kurz warten bis Worker bereit sind
        time.sleep(2)

        # ── Dispatch-Threads starten ──────────────────────────────
        target_queue = [t for t in targets if t["name"] in chunk_queues]
        queue_lock   = threading.Lock()
        results      = {}
        dispatch_threads = []

        for gpu in gpus:
            with queue_lock:
                target = target_queue.pop(0) if target_queue else None
            jd = gpu_job_dirs[gpu["index"]]
            t = threading.Thread(
                target=_gpu_chunk_worker,
                args=(gpu, target, target_queue, queue_lock,
                      chunk_queues, project_dir, jd, results),
                daemon=True,
                name=f"dispatch-gpu{gpu['index']}",
            )
            t.start()
            dispatch_threads.append(t)

        # Warten bis alle Chunks dispatched & fertig
        for t in dispatch_threads:
            t.join()

        # ── Worker herunterfahren ─────────────────────────────────
        for gpu in gpus:
            jd = gpu_job_dirs[gpu["index"]]
            shutdown_worker(jd)
            log(f"GPU {gpu['index']}: SHUTDOWN gesendet", tag="STOP")

        # Auf Worker-Prozesse warten
        for gpu_idx, proc in worker_procs.items():
            proc.wait()
            log(f"GPU {gpu_idx}: Container beendet (Exit {proc.returncode})",
                tag="STOP")

        # ── Ergebnisse ────────────────────────────────────────────
        log("=" * 60)
        log("=== ALLE DOCKING-JOBS ABGESCHLOSSEN ===")
        ok  = sum(1 for rc in results.values() if rc == 0)
        err = sum(1 for rc in results.values() if rc != 0)
        log(f"  Erfolgreich: {ok} | Fehler: {err}")

        for target in targets:
            merge_chunk_results(target["name"], results_dir)
        log("=" * 60)

    # ── Rescoring-GPU-Konfiguration bestimmen ─────────────────────
    rescore_enabled = (STAGE == "rescore"
                       and getbool(cfg, "RESCORE", "enabled", fallback=True))
    if rescore_enabled:
        rescore_num_gpus = getint(cfg, "RESCORE", "rescore_num_gpus", fallback=0)
        rescore_gpu_id   = getint(cfg, "RESCORE", "rescore_cuda_device_id", fallback=0)

        if rescore_num_gpus > 0 and use_gpu:
            # Explizite Rescoring-GPU-Konfiguration
            all_gpus_rs = detect_gpus(project_dir)
            if rescore_num_gpus == 1:
                rescore_gpus = [g for g in all_gpus_rs if g["index"] == rescore_gpu_id]
                if not rescore_gpus:
                    log(f"FEHLER: Rescoring-GPU {rescore_gpu_id} nicht gefunden! "
                        f"Verfügbar: {[g['index'] for g in all_gpus_rs]}", tag="ERROR")
                    rescore_gpus = gpus  # Fallback auf Docking-GPUs
                else:
                    log(f"Rescoring-GPUs: {len(rescore_gpus)} "
                        f"(separat konfiguriert: GPU {rescore_gpu_id})")
            else:
                rescore_gpus = all_gpus_rs[:rescore_num_gpus]
                log(f"Rescoring-GPUs: {len(rescore_gpus)} "
                    f"(separat konfiguriert: erste {rescore_num_gpus})")
        else:
            # Default: gleiche GPUs wie Docking
            rescore_gpus = gpus

    # ── Rescoring ─────────────────────────────────────────────────
    if rescore_enabled:
        rc = run_rescoring(sif_rescore, project_dir, ini_path,
                           targets=targets if use_gpu else None,
                           gpus=rescore_gpus if use_gpu else None)
        if rc != 0:
            log(f"Rescoring fehlgeschlagen (Exit {rc})", tag="WARN")

    # ── Refinement ────────────────────────────────────────────────
    refine_enabled = getbool(cfg, "REFINEMENT", "enabled", fallback=False)
    if refine_enabled and rescore_enabled:
        # Refinement nutzt die gleichen GPUs wie Rescoring
        # (Target-Level Parallelisierung, identisch zum Rescoring)
        refine_gpus = rescore_gpus if rescore_enabled and use_gpu else gpus
        rc = run_refinement(sif_rescore, project_dir, ini_path,
                            targets=targets if use_gpu else None,
                            gpus=refine_gpus if use_gpu else None)
        if rc != 0:
            log(f"Refinement fehlgeschlagen (Exit {rc})", tag="WARN")
    elif refine_enabled and not rescore_enabled:
        log("WARNUNG: Refinement aktiviert aber Rescoring deaktiviert – "
            "Refinement braucht ECR-Ergebnisse. Uebersprungen.", tag="WARN")

    total = datetime.now() - session_start
    log("=" * 60)
    log(f"=== PIPELINE ABGESCHLOSSEN | "
        f"Laufzeit: {str(total).split('.')[0]} ===")
    log("=" * 60)


if __name__ == "__main__":
    main()
