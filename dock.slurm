#!/bin/bash
# dock.slurm — Stufe 2: Docking, reicht sich selbst nach (72h-Limit)
#   sbatch dock.slurm
#   sbatch --export=ALL,CHAIN_MAX=2 dock.slurm     # kurze Kette zum Testen
# Kette anhalten: squeue -u $USER, dann scancel auf den wartenden Folgejob.

#SBATCH --job-name=kdc-dock
#SBATCH -N 1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --threads-per-core=1
#SBATCH --time=23:30:00
#SBATCH --mail-user=thomas.roessel@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/dock-%j.out
#SBATCH --signal=B:USR1@600

# 71:30 statt 72:00 laesst Raum fuer den geordneten Abbruch.
# B:USR1@600 sendet 10 min vor Ablauf ein Signal an DIESE Shell (ohne B:
# ginge es an die Tasks).

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "$PROJECT_DIR/pipeline_start.sh" ]; then
    echo "FEHLER: $PROJECT_DIR ist nicht die Projektwurzel." >&2
    exit 2
fi
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

CHAIN_MAX=${CHAIN_MAX:-20}
CHAIN_INDEX=${CHAIN_INDEX:-1}
DOCK_INI="config/docking.ini"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "Docking — Kettenglied $CHAIN_INDEX von max. $CHAIN_MAX"
log "Job ${SLURM_JOB_ID:-?} auf $(hostname -s) | GPUs ${SLURM_GPUS_ON_NODE:-?}"

python3 src/pipeline_progress.py "$DOCK_INI"
case $? in
    1) log "Nichts mehr offen — Kette endet."; exit 0 ;;
    2) log "FEHLER: Fortschritt nicht ermittelbar — Kette endet."; exit 2 ;;
esac

REMAINING_BEFORE=$(python3 src/pipeline_progress.py "$DOCK_INI" --json \
                   | python3 -c 'import json,sys; print(json.load(sys.stdin)["remaining"])')
log "Offen vor dem Lauf: $REMAINING_BEFORE"

# Folgejob VORAB einreichen: mit afterany entsteht er auch dann, wenn dieser
# Job an der Walltime stirbt. Pfad explizit, nicht "$0" — das zeigt auf die
# Spool-Kopie, die Slurm nach Jobende aufraeumt.
NEXT_JOB=""
if [ "$CHAIN_INDEX" -lt "$CHAIN_MAX" ]; then
    NEXT_JOB=$(sbatch --parsable \
                      --dependency=afterany:"$SLURM_JOB_ID" \
                      --export=ALL,CHAIN_INDEX=$(( CHAIN_INDEX + 1 )),CHAIN_MAX="$CHAIN_MAX" \
                      "$PROJECT_DIR/dock.slurm" 2>/dev/null)
    [ -n "$NEXT_JOB" ] && log "Folgejob: $NEXT_JOB" || log "WARNUNG: Folgejob nicht eingereicht."
else
    log "Kettenlaenge erreicht — kein Folgejob."
fi

PIPELINE_PID=""
TERMINATING=0

on_usr1() {
    TERMINATING=1
    log "USR1 empfangen — beende geordnet."
    if [ -n "$PIPELINE_PID" ]; then
        kill -TERM -"$PIPELINE_PID" 2>/dev/null || kill -TERM "$PIPELINE_PID" 2>/dev/null
        for _ in $(seq 1 100); do
            kill -0 "$PIPELINE_PID" 2>/dev/null || break
            sleep 5
        done
        kill -KILL -"$PIPELINE_PID" 2>/dev/null || true
    fi
}
trap on_usr1 USR1

STAGE_ENV=(
    RUN_CONVERSION=false
    RUN_DOCKING=true
    RUN_RESCORING="${RUN_RESCORING:-false}"
    CHECK_LIGANDS=false
)

log "Starte Pipeline"
# setsid --wait: eigene Prozessgruppe, $! zeigt auf deren Leader. Ohne --wait
# forkt setsid und wait kehrt zurueck, waehrend die Pipeline noch laeuft.
if setsid --wait true 2>/dev/null; then
    setsid --wait env "${STAGE_ENV[@]}" ./pipeline_start.sh --restart &
else
    env "${STAGE_ENV[@]}" ./pipeline_start.sh --restart &
fi
PIPELINE_PID=$!

wait "$PIPELINE_PID"
RC=$?
while kill -0 "$PIPELINE_PID" 2>/dev/null; do sleep 5; done
log "Pipeline beendet, Exit $RC (terminating=$TERMINATING)"

python3 src/pipeline_progress.py "$DOCK_INI"
REMAINING_AFTER=$(python3 src/pipeline_progress.py "$DOCK_INI" --json \
                  | python3 -c 'import json,sys; print(json.load(sys.stdin)["remaining"])' 2>/dev/null || echo -1)
log "Offen nach dem Lauf: $REMAINING_AFTER (vorher $REMAINING_BEFORE)"

CANCEL_REASON=""
if [ "$REMAINING_AFTER" = "0" ]; then
    CANCEL_REASON="alles abgearbeitet"
elif [ "$TERMINATING" -eq 0 ] && [ "$REMAINING_AFTER" = "$REMAINING_BEFORE" ]; then
    CANCEL_REASON="kein Fortschritt (Exit $RC)"
fi

if [ -n "$CANCEL_REASON" ] && [ -n "$NEXT_JOB" ]; then
    log "Kette beendet: $CANCEL_REASON"
    scancel "$NEXT_JOB" 2>/dev/null && log "Folgejob $NEXT_JOB storniert."
fi

exit 0
