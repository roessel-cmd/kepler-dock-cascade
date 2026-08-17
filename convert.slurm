#!/bin/bash
# convert.slurm — Stufe 1: SDF/SMILES → PDBQT (CPU)
#   sbatch convert.slurm

#SBATCH --job-name=kdc-convert
#SBATCH -N 1
#SBATCH -p zen4_0768
#SBATCH --qos idle_zen4_0768
#SBATCH --threads-per-core=1
#SBATCH --time=02:00:00
#SBATCH --mail-user=thomas.roessel@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/convert-%j.out

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "$PROJECT_DIR/pipeline_start.sh" ]; then
    echo "FEHLER: $PROJECT_DIR ist nicht die Projektwurzel." >&2
    exit 2
fi
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# SLURM_CPUS_PER_TASK ist im Standardmodus nicht zuverlaessig gesetzt.
CORES=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
WORKERS=$(( CORES > 2 ? CORES - 1 : 1 ))

log "Ligandenaufbereitung — Job ${SLURM_JOB_ID:-?} auf $(hostname -s)"
log "Partition ${SLURM_JOB_PARTITION:-?} | QoS ${SLURM_JOB_QOS:-?}"
log "$CORES Kerne → $WORKERS Worker"
log "Ablage: $(df -h . 2>/dev/null | tail -1 | awk '{print $6"  "$4" frei"}')"

RUN_CONVERSION=true \
RUN_DOCKING=false \
RUN_RESCORING=false \
CHECK_LIGANDS=false \
CONV_WORKERS="$WORKERS" \
    ./pipeline_start.sh
RC=$?

log "Konvertierung beendet, Exit $RC"

if [ $RC -eq 0 ]; then
    PDBQT_DIR=$(sed -n '/^\[PATHS\]/,/^\[/p' config/docking.ini \
        | grep -E '^[[:space:]]*pdbqt_dir[[:space:]]*=' \
        | sed 's/[^=]*=//; s/[#;].*//; s/[[:space:]]//g')
    log "Pruefe Bibliothek in $PDBQT_DIR"
    python3 src/check_ligands.py "$PDBQT_DIR"
    CHECK_RC=$?
    if [ $CHECK_RC -ne 0 ]; then
        log "check_ligands.py meldet Probleme (Exit $CHECK_RC) — vor dem Docking beheben."
        exit $CHECK_RC
    fi
    log "Bibliothek in Ordnung. Naechster Schritt: sbatch benchmark.slurm"
fi

exit $RC
