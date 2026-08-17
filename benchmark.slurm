#!/bin/bash
# benchmark.slurm — Messreihe Docking-Durchsatz (4x H100, ein Job)
#   sbatch benchmark.slurm
#   sbatch --export=ALL,BENCH_OUT=benchmark_batch,BENCH_N_LIGANDS=40000,BENCH_RUNS="4:balance:250:250 4:balance:1000:1000 4:balance:4000:4000" benchmark.slurm

#SBATCH --job-name=kdc-bench
#SBATCH -N 1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --threads-per-core=1
#SBATCH --time=08:00:00
#SBATCH --mail-user=thomas.roessel@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/bench-%j.out

# Vier GPUs anfordern, auch fuer die 1- und 2-GPU-Laeufe: der Orchestrator
# begrenzt sich intern ueber num_gpus. Nur so werden alle Konfigurationen
# auf identischer Hardware gemessen.

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "$PROJECT_DIR/benchmark.sh" ]; then
    echo "FEHLER: $PROJECT_DIR ist nicht die Projektwurzel." >&2
    exit 2
fi
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "Benchmark — Job ${SLURM_JOB_ID:-?} auf $(hostname -s)"
log "Partition ${SLURM_JOB_PARTITION:-?} | QoS ${SLURM_JOB_QOS:-?} | GPUs ${SLURM_GPUS_ON_NODE:-?}"

ENVFILE="logs/bench-${SLURM_JOB_ID:-manual}-environment.txt"
{
    echo "=== Knoten ==="
    hostname -f
    echo "Partition: ${SLURM_JOB_PARTITION:-?}  QoS: ${SLURM_JOB_QOS:-?}"
    echo
    echo "=== CPU ==="
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)"
    echo
    echo "=== GPUs ==="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null
    echo
    echo "=== Topologie ==="
    nvidia-smi topo -m 2>/dev/null
    echo
    echo "=== Ablage ==="
    df -h . 2>/dev/null | tail -1
    echo
    echo "=== Container ==="
    apptainer --version
    for sif in unidock-gpu.sif rescoring-gpu.sif sdf_to_pdbqt.sif; do
        [ -f "$sif" ] && printf '%-22s %s\n' "$sif" "$(stat -c '%y  %s bytes' "$sif" | cut -d. -f1)"
    done
    echo
    echo "=== Uni-Dock ==="
    apptainer exec --nv unidock-gpu.sif bash -lc \
        'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --version' 2>&1 | head -5
    echo
    echo "=== Git ==="
    git rev-parse HEAD 2>/dev/null
    git status --short 2>/dev/null | head -20
} > "$ENVFILE" 2>&1
log "Umgebung protokolliert: $ENVFILE"

# Vorabtest: laeuft Uni-Dock auf diesem Knoten (Compute Capability)?
if ! apptainer exec --nv unidock-gpu.sif bash -lc \
     'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --help' \
     > /dev/null 2>&1; then
    log "FEHLER: unidock laeuft auf diesem Knoten nicht (Compute Capability?)."
    exit 1
fi
log "Uni-Dock startet."

./benchmark.sh
RC=$?
log "Benchmark beendet, Exit $RC"

SUMMARY="${BENCH_OUT:-benchmark}/summary.csv"
if [ -f "$SUMMARY" ]; then
    log "Ergebnis ($SUMMARY):"
    column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
fi

exit $RC
