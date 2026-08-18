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

# Mindestfortschritt pro Kettenglied. Bleibt ein Lauf darunter, ohne dass er
# an der Walltime abgebrochen wurde, endet die Kette. Frueher galt schon ein
# einziger Ligand als Fortschritt – bei einem Fehler, der 99 % der Ergebnisse
# still verwirft, waeren so alle CHAIN_MAX Glieder durchgelaufen.
MIN_PROGRESS=${MIN_PROGRESS:-1000}

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Zahl der offenen Liganden. Gibt bei jedem Fehler "-1" aus, damit der
# Aufrufer den Unterschied zwischen "0 offen" und "nicht ermittelbar" sieht;
# eine leere Variable wuerde jeden spaeteren Vergleich still fehlschlagen
# lassen und damit den Abbruchwaechter deaktivieren.
remaining_count() {
    python3 src/pipeline_progress.py "$DOCK_INI" --json 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["remaining"])' 2>/dev/null \
        || echo -1
}

is_count() {   # nur nichtnegative Ganzzahlen sind gueltig ("-1" also nicht)
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
        *)           return 0 ;;
    esac
}

log "Docking — Kettenglied $CHAIN_INDEX von max. $CHAIN_MAX"
log "Job ${SLURM_JOB_ID:-?} auf $(hostname -s) | GPUs ${SLURM_GPUS_ON_NODE:-?}"

python3 src/pipeline_progress.py "$DOCK_INI"
case $? in
    1) log "Nichts mehr offen — Kette endet."; exit 0 ;;
    2) log "FEHLER: Fortschritt nicht ermittelbar — Kette endet."; exit 2 ;;
esac

REMAINING_BEFORE=$(remaining_count)
if ! is_count "$REMAINING_BEFORE"; then
    # Bewusst vor dem Einreichen des Folgejobs: ohne belastbaren Startwert
    # kann der Waechter unten nicht arbeiten, und die Kette liefe blind.
    log "FEHLER: Offene Liganden vor dem Lauf nicht ermittelbar — Kette endet."
    exit 2
fi
log "Offen vor dem Lauf: $REMAINING_BEFORE (Mindestfortschritt: $MIN_PROGRESS)"

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
REMAINING_AFTER=$(remaining_count)
log "Offen nach dem Lauf: $REMAINING_AFTER (vorher $REMAINING_BEFORE)"

CANCEL_REASON=""
if ! is_count "$REMAINING_AFTER"; then
    # Frueher lief die Kette hier weiter: -1 ist weder 0 noch gleich
    # REMAINING_BEFORE, also griff kein Abbruchgrund.
    CANCEL_REASON="Fortschritt nicht ermittelbar (Exit $RC)"
elif [ "$REMAINING_AFTER" -eq 0 ]; then
    CANCEL_REASON="alles abgearbeitet"
elif [ "$TERMINATING" -eq 0 ]; then
    PROGRESS=$(( REMAINING_BEFORE - REMAINING_AFTER ))
    if [ "$PROGRESS" -lt "$MIN_PROGRESS" ]; then
        CANCEL_REASON="zu wenig Fortschritt: $PROGRESS statt mindestens $MIN_PROGRESS (Exit $RC)"
    else
        log "Fortschritt: $PROGRESS Liganden"
    fi
fi

if [ -n "$CANCEL_REASON" ] && [ -n "$NEXT_JOB" ]; then
    log "Kette beendet: $CANCEL_REASON"
    scancel "$NEXT_JOB" 2>/dev/null && log "Folgejob $NEXT_JOB storniert."
fi

# Exit-Code der Pipeline durchreichen. Frueher stand hier 'exit 0', wodurch
# der Job immer als erfolgreich galt und --mail-type=FAIL nie ausgeloest hat.
# Die Kette haengt an afterany und ist davon unberuehrt.
if [ "$TERMINATING" -eq 1 ]; then
    log "Walltime-Abbruch — als Erfolg gewertet, Folgejob macht weiter."
    exit 0
fi
exit "$RC"
