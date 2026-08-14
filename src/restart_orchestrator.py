#!/usr/bin/env python3
"""
restart_orchestrator.py – Persistent Worker Version
Crash-Recovery: nur fehlende Liganden werden gechunkt und gedockt.
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


# ── Hilfsfunktionen (identisch mit orchestrator.py) ──────────────────

def load_ini(p):
    c = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    c.read(p, encoding="utf-8"); return c
def get(p, s, k, fallback=None): return p.get(s, k, fallback=fallback)
def getint(p, s, k, fallback=0): return p.getint(s, k, fallback=fallback)
def getbool(p, s, k, fallback=False): return p.getboolean(s, k, fallback=fallback)

def compute_cap_to_arch(cap):
    cap = cap.replace(".", "")
    return {"80":"ampere","86":"ampere","87":"ampere","89":"ada",
            "90":"hopper","90a":"hopper","100":"blackwell",
            "100a":"blackwell","120":"blackwell","120a":"blackwell"}.get(cap,"")

def detect_gpus(project_dir):
    project_dir = project_dir.resolve()
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=index,name,compute_cap",
                            "--format=csv,noheader"],
                           capture_output=True,text=True,timeout=10)
    except FileNotFoundError: return []
    gpus = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts)<3: continue
        idx,name,cap = int(parts[0]),parts[1],parts[2].replace(".","")
        arch = compute_cap_to_arch(cap)
        kd = project_dir/"kernels"/arch if arch else None
        k1 = kd/"Kernel1_Opt.bin" if kd else None
        k2 = kd/"Kernel2_Opt.bin" if kd else None
        gpus.append({"index":idx,"name":name,"cap":cap,"arch":arch,
                     "kernel_dir":kd,
                     "kernel_k1":k1 if (k1 and k1.exists()) else None,
                     "kernel_k2":k2 if (k2 and k2.exists()) else None})
    return gpus

_log_lock = threading.Lock()
def log(msg, tag="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    with _log_lock:
        print(f"{ts} [REST {tag:8s}] {msg}", flush=True)

def parse_targets(config_txt, target_dir):
    if not config_txt.exists(): return []
    targets, current = [], {}
    for raw in config_txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            if current and "name" in current:
                if (target_dir/f"{current['name']}.pdbqt").exists():
                    targets.append(current)
                current = {}
            continue
        if line.upper().startswith("CENTER"):
            m = re.search(r"\[([^\]]+)\]",line)
            if m: current["center"]=[float(x) for x in m.group(1).split(",")]
        elif line.upper().startswith("BOX_SIZE"):
            m = re.search(r"\[([^\]]+)\]",line)
            if m: current["box_size"]=[float(x) for x in m.group(1).split(",")]
        elif "=" in line and line.upper().startswith("LIGAND_SUBDIR"):
            current["ligand_subdir"]=line.split("=",1)[1].strip()
        elif re.match(r"^[\w\-]+$",line):
            if current and "name" in current:
                if (target_dir/f"{current['name']}.pdbqt").exists():
                    targets.append(current)
            current = {"name":line}
    if current and "name" in current:
        if (target_dir/f"{current['name']}.pdbqt").exists():
            targets.append(current)
    return targets


# ── Completed Detection ──────────────────────────────────────────────

def find_completed_ligands(target_results_dir):
    completed = set()
    if target_results_dir.exists():
        for f in target_results_dir.glob("*_docked.pdbqt"):
            if f.stat().st_size > 0:
                stem = f.stem
                if stem.endswith("_docked"):
                    stem = stem[:-7]
                completed.add(stem)
    return completed


# ── Chunk-Infrastruktur ──────────────────────────────────────────────

class LigandChunkQueue:
    def __init__(self, target_name, ligand_files, chunk_size=200):
        self.target_name = target_name
        self._lock = threading.Lock()
        self.chunks = [ligand_files[i:i+chunk_size]
                       for i in range(0,len(ligand_files),chunk_size)]
        self._next_idx = 0
        self.total = len(self.chunks)
    def get_next_chunk(self):
        with self._lock:
            if self._next_idx >= self.total: return None
            idx = self._next_idx; self._next_idx += 1
            return idx, self.chunks[idx]
    @property
    def remaining(self):
        with self._lock: return self.total - self._next_idx

def _get_ligand_files(target, pdbqt_dir):
    """Rekursive Ligandensuche – sdf_to_pdbqt.py legt 0000/, 0001/, ... an."""
    files, warnings = find_ligand_files(pdbqt_dir, target.get("ligand_subdir"))
    for w in warnings:
        log(w, tag="WARN")
    return files

def write_chunk_file(project_dir, target_name, gpu_idx, chunk_idx, chunk_files):
    path = project_dir/"data"/"LOG"/f"chunk_gpu{gpu_idx}_{target_name}_{chunk_idx}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for lig in chunk_files:
            f.write(f"/workspace/{lig.relative_to(project_dir)}\n")
    return path

def merge_chunk_results(target_name, results_dir):
    td = results_dir/target_name
    cfs = sorted(td.glob(f"docking_results_{target_name}_chunk*.csv"))
    if not cfs: return None
    mp = td/f"docking_results_{target_name}.csv"
    by_lig = {}
    if mp.exists():
        try:
            with open(mp,encoding="utf-8",newline="") as f:
                for row in csv.DictReader(f):
                    l = row.get("ligand","")
                    if l: by_lig[l]=row
        except Exception: pass
    for cf in cfs:
        with open(cf,encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                l = row.get("ligand","")
                if l: by_lig[l]=row
    if not by_lig: return None
    rows = list(by_lig.values())
    def sk(r):
        try: return (0,float(r.get("best_energy_kcal_mol","")))
        except: return (1,0.0)
    rows.sort(key=sk)
    with open(mp,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=["ligand","success","best_energy_kcal_mol","error"])
        w.writeheader(); w.writerows(rows)
    log(f"  Merge '{target_name}': {len(cfs)} CSVs → {len(rows)} Lig.")
    return mp


# ── Persistent Worker ────────────────────────────────────────────────

def start_persistent_worker(gpu, sif, project_dir, ini_path, job_dir):
    idx = gpu["index"]
    cmd = ["apptainer","exec","--nv","--pwd","/workspace",
           "--bind",f"{project_dir}:/workspace",
           "--bind",f"{ini_path}:/app/pipeline_config.ini"]
    # Lokale Quellen binden (Entwicklermodus) – im Image liegen sie ohnehin
    for _name in ("worker_restart_dock.py", "unidock_engine.py",
                  "docking_config.py", "pipeline_common.py"):
        _src = project_dir / "src" / _name
        if _src.exists():
            cmd += ["--bind", f"{_src}:/app/{_name}"]
    cjd = f"/workspace/{job_dir.relative_to(project_dir)}"
    cmd += ["--env",f"CUDA_VISIBLE_DEVICES={idx}",
            "--env",f"WORKER_GPU_ID={idx}",
            "--env","WORKER_PERSISTENT=1",
            "--env",f"WORKER_JOB_DIR={cjd}",
            str(sif),"bash","-c",
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate docking_env && python /app/worker_restart_dock.py"]
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True,bufsize=1)
    log(f"GPU {idx}: Persistent Worker gestartet (PID {proc.pid})", tag="START")
    return proc

def stream_worker_output(proc, gpu_idx):
    for line in proc.stdout:
        log(line.rstrip(), tag=f"GPU{gpu_idx}")
    proc.wait()

def shutdown_worker(job_dir):
    (job_dir/"SHUTDOWN").write_text("", encoding="utf-8")


# ── Rescoring ────────────────────────────────────────────────────────

def _run_rescore_worker(gpu,sif,project_dir,ini_path,target_name,results):
    idx=gpu["index"]
    log(f"GPU {idx}: Rescoring '{target_name}'")
    conda_env="rescore_env" if "rescoring" in str(sif) else "docking_env"
    cmd=["apptainer","exec","--nv","--pwd","/workspace",
         "--bind",f"{project_dir}:/workspace",
         "--bind",f"{ini_path}:/app/pipeline_config.ini",
         "--env","ORCHESTRATOR_RESCORE_ONLY=1",
         "--env",f"CUDA_VISIBLE_DEVICES={idx}",
         "--env",f"WORKER_TARGET={target_name}",
         "--env",f"WORKER_GPU_ID={idx}"]
    for pyfile in ["worker_rescore.py","docking_rescore.py",
                    "gnina_gpu_worker.py",
                    "gnina_refinement.py","linf9xgb_scorer.py","ecr.py"]:
        local=project_dir/"src"/pyfile
        if local.exists(): cmd+=["--bind",f"{local}:/app/{pyfile}"]
    cmd+=[str(sif),"bash","-c",
          f"source /opt/miniconda3/etc/profile.d/conda.sh && "
          f"conda activate {conda_env} && python /app/worker_rescore.py"]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout: log(line.rstrip(),tag=f"RS-G{idx}")
    proc.wait(); rc=proc.returncode; results[target_name]=rc
    log(f"GPU {idx}: Rescoring '{target_name}' {'✓' if rc==0 else f'Fehler ({rc})'}",
        tag="OK" if rc==0 else "ERROR")

def _rescore_queue(gpu,first,tq,ql,sif,pd,ip,res):
    t=first
    while t is not None:
        _run_rescore_worker(gpu,sif,pd,ip,t["name"],res)
        with ql: t=tq.pop(0) if tq else None

def run_rescoring(sif,pd,ip,targets=None,gpus=None):
    log("=== RESCORING ===")
    if gpus and targets:
        tq,ql,res,ths=list(targets),threading.Lock(),{},[]
        for gpu in gpus:
            if not tq: break
            with ql: t=tq.pop(0) if tq else None
            if t is None: break
            th=threading.Thread(target=_rescore_queue,args=(gpu,t,tq,ql,sif,pd,ip,res),daemon=True)
            th.start(); ths.append(th)
        for th in ths: th.join()
        ok=sum(1 for r in res.values() if r==0)
        log(f"=== RESCORING FERTIG – OK: {ok} | Fehler: {len(res)-ok} ===")
        return 1 if any(r!=0 for r in res.values()) else 0
    return 0


# ── Refinement ───────────────────────────────────────────────────────

def _run_refine_worker(gpu,sif,project_dir,ini_path,target_name,results):
    idx=gpu["index"]
    log(f"GPU {idx}: Refinement '{target_name}'")
    conda_env="rescore_env" if "rescoring" in str(sif) else "docking_env"
    cmd=["apptainer","exec","--nv","--pwd","/workspace",
         "--bind",f"{project_dir}:/workspace",
         "--bind",f"{ini_path}:/app/pipeline_config.ini",
         "--env",f"CUDA_VISIBLE_DEVICES={idx}",
         "--env",f"WORKER_TARGET={target_name}",
         "--env",f"WORKER_GPU_ID={idx}"]
    for pyfile in ["gnina_refinement.py","docking_rescore.py",
                    "gnina_gpu_worker.py",
                    "gnina_refinement.py"]:
        local=project_dir/"src"/pyfile
        if local.exists(): cmd+=["--bind",f"{local}:/app/{pyfile}"]
    cmd+=[str(sif),"bash","-c",
          f"source /opt/miniconda3/etc/profile.d/conda.sh && "
          f"conda activate {conda_env} && python /app/gnina_refinement.py"]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout: log(line.rstrip(),tag=f"RF-G{idx}")
    proc.wait(); rc=proc.returncode; results[target_name]=rc
    log(f"GPU {idx}: Refinement '{target_name}' {'✓' if rc==0 else f'Fehler ({rc})'}",
        tag="OK" if rc==0 else "ERROR")

def _refine_queue(gpu,first,tq,ql,sif,pd,ip,res):
    t=first
    while t is not None:
        _run_refine_worker(gpu,sif,pd,ip,t["name"],res)
        with ql: t=tq.pop(0) if tq else None

def run_refinement(sif,pd,ip,targets=None,gpus=None):
    log("=== REFINEMENT ===")
    if gpus and targets:
        tq,ql,res,ths=list(targets),threading.Lock(),{},[]
        for gpu in gpus:
            if not tq: break
            with ql: t=tq.pop(0) if tq else None
            if t is None: break
            th=threading.Thread(target=_refine_queue,args=(gpu,t,tq,ql,sif,pd,ip,res),daemon=True)
            th.start(); ths.append(th)
        for th in ths: th.join()
        ok=sum(1 for r in res.values() if r==0)
        log(f"=== REFINEMENT FERTIG – OK: {ok} | Fehler: {len(res)-ok} ===")
        return 1 if any(r!=0 for r in res.values()) else 0
    return 0


# ── GPU Dispatch Thread ──────────────────────────────────────────────

def _gpu_chunk_worker(gpu,first_target,target_queue,queue_lock,
                      chunk_queues,project_dir,job_dir,results):
    gpu_idx=gpu["index"]; target=first_target
    while True:
        if target is not None:
            _process_target_chunks(gpu,target,chunk_queues,project_dir,job_dir,results)
        with queue_lock:
            target=target_queue.pop(0) if target_queue else None
        if target is not None: continue
        helped=_help_any_target(gpu,chunk_queues,project_dir,job_dir,results)
        if helped:
            with queue_lock:
                target=target_queue.pop(0) if target_queue else None
            if target is not None: continue
            continue
        break
    log(f"GPU {gpu_idx}: Fertig",tag="DONE")

def _dispatch_chunk(gpu,tgt_name,chunk_idx,chunk_files,project_dir,job_dir,results):
    gpu_idx=gpu["index"]
    chunk_id=f"chunk_gpu{gpu_idx}_{tgt_name}_{chunk_idx}"
    chunk_file=write_chunk_file(project_dir,tgt_name,gpu_idx,chunk_idx,chunk_files)
    jf=job_dir/f"{chunk_id}.job"
    jf.write_text(f"TARGET={tgt_name}\nCHUNK_FILE=/workspace/{chunk_file.relative_to(project_dir)}\nCHUNK_ID={chunk_id}\n",encoding="utf-8")
    df=job_dir/f"{chunk_id}.done"
    while not df.exists(): time.sleep(0.5)
    try: rc=int(df.read_text(encoding="utf-8").strip())
    except: rc=0
    results[f"{tgt_name}_chunk{chunk_idx}"]=rc

def _process_target_chunks(gpu,target,chunk_queues,project_dir,job_dir,results):
    gpu_idx=gpu["index"]; tgt_name=target["name"]
    cq=chunk_queues.get(tgt_name)
    if not cq: return
    log(f"GPU {gpu_idx}: Starte '{tgt_name}' ({cq.total} Chunks)",tag="START")
    while True:
        cd=cq.get_next_chunk()
        if cd is None: break
        ci,cf=cd
        log(f"GPU {gpu_idx}: {tgt_name} Chunk {ci+1}/{cq.total} ({len(cf)} Lig.)")
        _dispatch_chunk(gpu,tgt_name,ci,cf,project_dir,job_dir,results)
    log(f"GPU {gpu_idx}: '{tgt_name}' fertig ✓",tag="OK")

def _help_any_target(gpu,chunk_queues,project_dir,job_dir,results):
    gpu_idx=gpu["index"]; helped=False
    for tn,cq in chunk_queues.items():
        while True:
            cd=cq.get_next_chunk()
            if cd is None: break
            helped=True; ci,cf=cd
            log(f"GPU {gpu_idx}: Hilft bei '{tn}' Chunk {ci+1}/{cq.total}",tag="HELP")
            _dispatch_chunk(gpu,tn,ci,cf,project_dir,job_dir,results)
    return helped


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    parser=argparse.ArgumentParser(description="Restart Orchestrator")
    parser.add_argument("--config",default="pipeline_config.ini")
    parser.add_argument("--project",default=".")
    parser.add_argument("--stage",choices=["dock","rescore"],default="dock",
                        help="dock = nur Docking fortsetzen, rescore = nur "
                             "Rescoring. Gesetzt von pipeline_start.sh.")
    parser.add_argument("--sif",default="",
                        help="Container-Image; leer = automatisch nach Stage.")
    args=parser.parse_args()
    STAGE=args.stage
    project_dir=Path(args.project).resolve()
    ini_path=project_dir/args.config
    if not ini_path.exists():
        print(f"FEHLER: {ini_path}",file=sys.stderr); sys.exit(1)
    cfg=load_ini(ini_path); session_start=datetime.now()
    log("="*60)
    log(f"  RESTART ORCHESTRATOR (Persistent Worker)")
    log(f"  {session_start:%Y-%m-%d %H:%M:%S}")
    log("="*60)

    if args.sif:
        sif=Path(args.sif)
        if not sif.is_absolute():
            sif=project_dir/sif
    elif STAGE=="dock":
        sif=project_dir/"unidock-gpu.sif"
    else:
        sif=project_dir/"rescoring-gpu.sif"
    if not sif.exists():
        log(f"FEHLER: Container nicht gefunden: {sif}",tag="ERROR"); sys.exit(1)

    sif_rescore=sif
    log(f"  Stage:     {STAGE}")
    log(f"  Container: {sif.name}")
    log(f"  INI:       {ini_path.name}")

    num_gpus_cfg   = getint(cfg, "GPU", "num_gpus", fallback=1)
    cuda_device_id = getint(cfg, "GPU", "cuda_device_id", fallback=0)
    # Kein CPU-Pfad mehr – Docking laeuft ausschliesslich auf GPU.
    use_gpu        = True

    if use_gpu:
        all_gpus = detect_gpus(project_dir)
        if num_gpus_cfg == 1:
            gpus = [g for g in all_gpus if g["index"] == cuda_device_id]
            if not gpus:
                log(f"FEHLER: GPU {cuda_device_id} (cuda_device_id) nicht gefunden! "
                    f"Verfügbar: {[g['index'] for g in all_gpus]}", tag="ERROR")
                sys.exit(1)
        else:
            gpus = all_gpus[:num_gpus_cfg]
    else:
        gpus = []
    for g in gpus:
        log(f"  GPU {g['index']}: {g['name']} | {'Kernel: '+g['arch'] if g['kernel_k1'] else 'FEHLT'}")

    target_dir=project_dir/get(cfg,"PATHS","target_dir","./TARGET").lstrip("./")
    results_dir=project_dir/get(cfg,"PATHS","results_dir","./RESULTS").lstrip("./")
    pdbqt_dir=project_dir/get(cfg,"PATHS","pdbqt_dir","./data/PDBQT").lstrip("./")
    chunk_size=getint(cfg,"CHUNK","chunk_size",fallback=200)

    targets=parse_targets(target_dir/"config.txt",target_dir)
    if not targets: log("Keine Targets",tag="ERROR"); sys.exit(1)

    # ── Fortschritt + Chunk-Queues ────────────────────────────────
    # Nur in Stage "dock"; in Stage "rescore" bleibt die Liste leer und
    # der Docking-Block wird uebersprungen.
    pending_targets=[]; chunk_queues={}
    for target in (targets if STAGE=="dock" else []):
        tn=target["name"]
        completed=find_completed_ligands(results_dir/tn)
        all_lig=_get_ligand_files(target,pdbqt_dir)
        remaining=[f for f in all_lig if f.stem not in completed]
        if not remaining:
            log(f"  '{tn}': alle {len(all_lig)} fertig ✓"); continue
        log(f"  '{tn}': {len(all_lig)-len(remaining)}/{len(all_lig)} fertig | {len(remaining)} ausstehend")
        pending_targets.append(target)
        cq=LigandChunkQueue(tn,remaining,chunk_size)
        chunk_queues[tn]=cq
        log(f"    → {cq.total} Chunks à {chunk_size}")
        tr=results_dir/tn
        if tr.exists():
            for old in tr.glob(f"docking_results_{tn}_chunk*.csv"): old.unlink()

    if not pending_targets:
        log("=== Alles fertig – nur Rescoring ===")
    else:
        log("="*60)
        log(f"=== RESTART: {len(pending_targets)} Target(s) | {len(gpus)} GPUs ===")
        log("="*60)

        jobs_base=project_dir/"data"/"LOG"/"jobs"
        gpu_job_dirs={}
        for gpu in gpus:
            jd=jobs_base/f"gpu{gpu['index']}"
            if jd.exists():
                for f in jd.iterdir(): f.unlink()
            jd.mkdir(parents=True,exist_ok=True)
            gpu_job_dirs[gpu["index"]]=jd

        worker_procs={}; output_threads={}
        for gpu in gpus:
            jd=gpu_job_dirs[gpu["index"]]
            proc=start_persistent_worker(gpu,sif,project_dir,ini_path,jd)
            worker_procs[gpu["index"]]=proc
            ot=threading.Thread(target=stream_worker_output,args=(proc,gpu["index"]),
                                daemon=True)
            ot.start(); output_threads[gpu["index"]]=ot

        time.sleep(2)

        target_queue=list(pending_targets)
        queue_lock=threading.Lock(); results={}; threads=[]
        for gpu in gpus:
            with queue_lock: target=target_queue.pop(0) if target_queue else None
            jd=gpu_job_dirs[gpu["index"]]
            t=threading.Thread(target=_gpu_chunk_worker,
                               args=(gpu,target,target_queue,queue_lock,
                                     chunk_queues,project_dir,jd,results),
                               daemon=True)
            t.start(); threads.append(t)
        for t in threads: t.join()

        for gpu in gpus:
            shutdown_worker(gpu_job_dirs[gpu["index"]])
        for gi,proc in worker_procs.items():
            proc.wait()
            log(f"GPU {gi}: Container beendet (Exit {proc.returncode})",tag="STOP")

        ok=sum(1 for r in results.values() if r==0)
        log(f"=== RESTART FERTIG – OK: {ok} | Fehler: {len(results)-ok} ===")
        for t in pending_targets: merge_chunk_results(t["name"],results_dir)

    # ── Rescoring-GPU-Konfiguration bestimmen ─────────────────────
    rescore_enabled = (STAGE=="rescore"
                       and getbool(cfg,"RESCORE","enabled",fallback=True))
    if rescore_enabled:
        rescore_num_gpus = getint(cfg, "RESCORE", "rescore_num_gpus", fallback=0)
        rescore_gpu_id   = getint(cfg, "RESCORE", "rescore_cuda_device_id", fallback=0)

        if rescore_num_gpus > 0 and use_gpu:
            all_gpus_rs = detect_gpus(project_dir)
            if rescore_num_gpus == 1:
                rescore_gpus = [g for g in all_gpus_rs if g["index"] == rescore_gpu_id]
                if not rescore_gpus:
                    log(f"FEHLER: Rescoring-GPU {rescore_gpu_id} nicht gefunden! "
                        f"Verfügbar: {[g['index'] for g in all_gpus_rs]}", tag="ERROR")
                    rescore_gpus = gpus
                else:
                    log(f"Rescoring-GPUs: {len(rescore_gpus)} "
                        f"(separat konfiguriert: GPU {rescore_gpu_id})")
            else:
                rescore_gpus = all_gpus_rs[:rescore_num_gpus]
                log(f"Rescoring-GPUs: {len(rescore_gpus)} "
                    f"(separat konfiguriert: erste {rescore_num_gpus})")
        else:
            rescore_gpus = gpus

    # ── Rescoring ─────────────────────────────────────────────────
    if rescore_enabled:
        rc = run_rescoring(sif_rescore, project_dir, ini_path,
                           targets=targets, gpus=rescore_gpus)
        if rc != 0:
            log(f"Rescoring fehlgeschlagen (Exit {rc})", tag="WARN")

    # ── Refinement ────────────────────────────────────────────────
    refine_enabled = getbool(cfg,"REFINEMENT","enabled",fallback=False)
    if refine_enabled and rescore_enabled:
        refine_gpus = rescore_gpus if rescore_enabled and use_gpu else gpus
        run_refinement(sif_rescore,project_dir,ini_path,targets=targets,gpus=refine_gpus)
    elif refine_enabled and not rescore_enabled:
        log("WARNUNG: Refinement aktiviert aber Rescoring deaktiviert – "
            "Refinement braucht ECR-Ergebnisse. Uebersprungen.", tag="WARN")

    log("="*60)
    log(f"=== ABGESCHLOSSEN | {str(datetime.now()-session_start).split('.')[0]} ===")
    log("="*60)

if __name__=="__main__":
    main()
