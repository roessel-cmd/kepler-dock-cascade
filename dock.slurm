#!/bin/bash
# dock.slurm — Stufe 2/3: Docking oder Rescoring, reicht sich selbst nach
#   sbatch dock.slurm                                          # Docking
#   sbatch --export=ALL,RUN_DOCKING=false,RUN_RESCORING=true dock.slurm
#   sbatch --export=ALL,CHAIN_MAX=2 dock.slurm                 # kurze Testkette
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

# 23:30 statt 24:00 laesst Raum fuer den geordneten Abbruch.
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

DOCK_INI="${DOCK_INI:-config/docking.ini}"
RESCORE_INI="${RESCORE_INI:-config/rescore.ini}"

RUN_CONVERSION=${RUN_CONVERSION:-false}
RUN_DOCKING=${RUN_DOCKING:-true}
RUN_RESCORING=${RUN_RESCORING:-false}
CHECK_LIGANDS=${CHECK_LIGANDS:-false}

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Docking zaehlt Liganden, Rescoring zaehlt Bloecke. Mit pipeline_progress.py
# meldete ein reiner Rescoring-Lauf nach fertigem Docking "0 offen" und die
# Kette endete, bevor Stufe 3 beginnt (siehe dock-73991.out).
if [ "$RUN_RESCORING" = "true" ] && [ "$RUN_DOCKING" != "true" ]; then
    STAGE="Rescoring"
    PROGRESS_CMD=(python3 src/rescore_progress.py "$RESCORE_INI")
    UNIT="Bloecke"
    MIN_PROGRESS=${MIN_PROGRESS:-1}
else
    STAGE="Docking"
    PROGRESS_CMD=(python3 src/pipeline_progress.py "$DOCK_INI")
    UNIT="Liganden"
    MIN_PROGRESS=${MIN_PROGRESS:-1000}
fi

# "-1" bei jedem Fehler: eine leere Variable wuerde jeden spaeteren Vergleich
# still fehlschlagen lassen und den Abbruchwaechter deaktivieren.
remaining_count() {
    "${PROGRESS_CMD[@]}" --json 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["remaining"])' 2>/dev/null \
        || echo -1
}

is_count() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
        *)           return 0 ;;
    esac
}

log "$STAGE — Kettenglied $CHAIN_INDEX von max. $CHAIN_MAX"
log "Job ${SLURM_JOB_ID:-?} auf $(hostname -s) | GPUs ${SLURM_GPUS_ON_NODE:-?}"
log "Stufen: conversion=$RUN_CONVERSION docking=$RUN_DOCKING rescoring=$RUN_RESCORING"

"${PROGRESS_CMD[@]}"
case $? in
    1) log "Nichts mehr offen ($STAGE) — Kette endet."; exit 0 ;;
    2) log "FEHLER: Fortschritt nicht ermittelbar — Kette endet."; exit 2 ;;
esac

REMAINING_BEFORE=$(remaining_count)
if ! is_count "$REMAINING_BEFORE"; then
    # Vor dem Einreichen des Folgejobs: ohne Startwert liefe die Kette blind.
    log "FEHLER: Offene $UNIT vor dem Lauf nicht ermittelbar — Kette endet."
    exit 2
fi
log "Offen vor dem Lauf: $REMAINING_BEFORE $UNIT (Mindestfortschritt: $MIN_PROGRESS)"

# Folgejob VORAB einreichen: mit afterany entsteht er auch dann, wenn dieser
# Job an der Walltime stirbt. Pfad explizit, nicht "$0" — das zeigt auf die
# Spool-Kopie, die Slurm nach Jobende aufraeumt.
NEXT_JOB=""
if [ "$CHAIN_INDEX" -lt "$CHAIN_MAX" ]; then
    NEXT_JOB=$(sbatch --parsable \
                      --dependency=afterany:"$SLURM_JOB_ID" \
                      --export=ALL,CHAIN_INDEX=$(( CHAIN_INDEX + 1 )),CHAIN_MAX="$CHAIN_MAX",RUN_CONVERSION="$RUN_CONVERSION",RUN_DOCKING="$RUN_DOCKING",RUN_RESCORING="$RUN_RESCORING",CHECK_LIGANDS="$CHECK_LIGANDS" \
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
    RUN_CONVERSION="$RUN_CONVERSION"
    RUN_DOCKING="$RUN_DOCKING"
    RUN_RESCORING="$RUN_RESCORING"
    CHECK_LIGANDS="$CHECK_LIGANDS"
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

"${PROGRESS_CMD[@]}"
REMAINING_AFTER=$(remaining_count)
log "Offen nach dem Lauf: $REMAINING_AFTER $UNIT (vorher $REMAINING_BEFORE)"

CANCEL_REASON=""
if ! is_count "$REMAINING_AFTER"; then
    CANCEL_REASON="Fortschritt nicht ermittelbar (Exit $RC)"
elif [ "$REMAINING_AFTER" -eq 0 ]; then
    CANCEL_REASON="alles abgearbeitet"
else
    PROGRESS=$(( REMAINING_BEFORE - REMAINING_AFTER ))
    if [ "$TERMINATING" -eq 1 ]; then
        # Dauert eine Einheit laenger als die Walltime — beim Rescoring ein zu
        # gross gewaehltes rescore_block_size —, sichert kein Lauf je etwas.
        if [ "$PROGRESS" -le 0 ]; then
            CANCEL_REASON="Walltime erreicht, aber kein $UNIT gesichert — Einheit zu gross?"
        else
            log "Fortschritt: $PROGRESS $UNIT (Walltime-Abbruch)"
        fi
    elif [ "$PROGRESS" -lt "$MIN_PROGRESS" ]; then
        CANCEL_REASON="zu wenig Fortschritt: $PROGRESS statt mindestens $MIN_PROGRESS $UNIT (Exit $RC)"
    else
        log "Fortschritt: $PROGRESS $UNIT"
    fi
fi

if [ -n "$CANCEL_REASON" ] && [ -n "$NEXT_JOB" ]; then
    log "Kette beendet: $CANCEL_REASON"
    scancel "$NEXT_JOB" 2>/dev/null && log "Folgejob $NEXT_JOB storniert."
fi

# Exit-Code durchreichen, damit --mail-type=FAIL greift. Die Kette haengt an
# afterany und ist davon unberuehrt.
if [ "$TERMINATING" -eq 1 ]; then
    log "Walltime-Abbruch — als Erfolg gewertet, Folgejob macht weiter."
    exit 0
fi
exit "$RC"
