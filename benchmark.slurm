#!/bin/bash
# ============================================================================
# benchmark.slurm
# Fuehrt die Benchmark-Matrix als einzelnen Slurm-Job aus.
#
#     sbatch benchmark.slurm
#
# Bewusst OHNE Kette: die Matrix ist so bemessen, dass sie in eine Walltime
# passt. Falls nicht, ist der Messsatz (BENCH_N_LIGANDS in benchmark.sh) zu
# gross gewaehlt – eine ueber zwei Jobs verteilte Messreihe waere ohnehin
# nicht vergleichbar, weil der zweite Job auf anderer Hardware landen kann.
# ============================================================================

#SBATCH --job-name=kepler-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/bench-%j.out

# --exclusive ist hier keine Bequemlichkeit, sondern Voraussetzung: teilen
# sich andere Jobs den Knoten, misst du deren Last mit. Fuer eine Messreihe,
# die in eine Veroeffentlichung soll, ist das nicht brauchbar.

set -uo pipefail

# Projektverzeichnis. SLURM_SUBMIT_DIR ist das Verzeichnis, aus dem sbatch
# aufgerufen wurde – nicht der Ort des Skripts. Slurm fuehrt eine Kopie aus
# dem Spool-Verzeichnis aus, $0 und BASH_SOURCE zeigen also NICHT ins Repo.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ ! -f "$PROJECT_DIR/pipeline_start.sh" ]; then
    echo "FEHLER: $PROJECT_DIR ist nicht die Projektwurzel." >&2
    echo "        sbatch aus dem Repo-Hauptverzeichnis aufrufen:" >&2
    echo "          cd ~/kepler-dock-cascade && sbatch $(basename "$0")" >&2
    exit 2
fi
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "════════════════════════════════════════════════════════════"
log " Docking Benchmark"
log " Job-ID : ${SLURM_JOB_ID:-?}"
log " Knoten : $(hostname -s)"
log "════════════════════════════════════════════════════════════"

# ── Hardware protokollieren ────────────────────────────────────────────────
# Gehoert zu jeder Messreihe dazu: ohne diese Angaben ist eine Durchsatzzahl
# nicht interpretierbar und in einem Methodenteil wertlos.
{
    echo "=== Knoten ==="
    hostname -f
    echo
    echo "=== CPU ==="
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)"
    echo
    echo "=== GPUs ==="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
               --format=csv 2>/dev/null
    echo
    echo "=== GPU-Topologie ==="
    nvidia-smi topo -m 2>/dev/null
    echo
    echo "=== Container ==="
    apptainer --version
    for sif in unidock-gpu.sif rescoring-gpu.sif sdf_to_pdbqt.sif; do
        [ -f "$sif" ] && printf '%-22s %s\n' "$sif" \
            "$(stat -c '%y  %s bytes' "$sif" | cut -d. -f1)"
    done
    echo
    echo "=== Uni-Dock ==="
    apptainer exec --nv unidock-gpu.sif bash -lc \
        'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --version' \
        2>&1 | head -5
    echo
    echo "=== Git ==="
    git rev-parse HEAD 2>/dev/null
    git status --short 2>/dev/null | head -20
} > "logs/bench-${SLURM_JOB_ID:-manual}-environment.txt" 2>&1

log "Umgebung protokolliert: logs/bench-${SLURM_JOB_ID:-manual}-environment.txt"

# ── sm_90-Vorabtest ────────────────────────────────────────────────────────
# Schlaegt der fehl, ist die ganze Messreihe hinfaellig. Lieber hier
# abbrechen als nach Stunden mit leeren CSVs dastehen.
if ! apptainer exec --nv unidock-gpu.sif bash -lc \
     'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --help' \
     > /dev/null 2>&1; then
    log "FEHLER: unidock laeuft auf diesem Knoten nicht (Compute Capability?)."
    log "        Siehe README, Abschnitt Installation."
    exit 1
fi
log "Uni-Dock startet auf diesem Knoten."

# ── Messreihe ──────────────────────────────────────────────────────────────
log "Starte Benchmark-Matrix ..."
./benchmark.sh
RC=$?

log "Benchmark beendet mit Exit-Code $RC"
[ -f benchmark/summary.csv ] && {
    log "Ergebnis:"
    column -t -s, benchmark/summary.csv 2>/dev/null || cat benchmark/summary.csv
}

exit $RC
