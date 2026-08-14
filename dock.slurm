#!/bin/bash
# ============================================================================
# dock.slurm
# Produktivlauf der Docking-Pipeline auf einem HPC-Cluster mit Walltime-Grenze.
#
# Reicht sich selbst als Folgejob ein, bis die Bibliothek abgearbeitet ist.
# Die Kette bricht ab, wenn nichts mehr offen ist, wenn ein Lauf keinen
# Fortschritt erzielt hat, oder wenn die maximale Kettenlaenge erreicht ist.
#
#     sbatch dock.slurm
#
# Fortsetzen einer unterbrochenen Kette: einfach erneut absetzen. Der
# Fortschritt steht im Dateisystem (RESULTS/<target>/*_docked.pdbqt), nicht
# im Job, also findet der naechste Lauf von selbst die richtige Stelle.
#
# VORAUSSETZUNG: Die Liganden liegen bereits als PDBQT vor. Die
# Konvertierung laeuft als eigener CPU-Job (convert.slurm) und ist hier
# abgeschaltet – sie braucht keine GPU, und jedes Kettenglied wuerde sonst
# die Ligandenbilanz ueber Millionen Dateien neu berechnen, nur um
# festzustellen, dass nichts zu tun ist.
# ============================================================================

#SBATCH --job-name=kepler-dock
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --signal=B:USR1@600

# --signal=B:USR1@600 laesst Slurm 10 Minuten vor Ablauf der Walltime ein
# USR1 an die Batch-Shell schicken (das B: ist wesentlich – ohne geht das
# Signal an die Tasks, nicht an dieses Skript). Wir nutzen die Zeit, um die
# laufenden Container geordnet zu beenden, statt mitten im Schreiben einer
# PDBQT erschlagen zu werden.

set -uo pipefail

# ════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

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

CHAIN_MAX=${CHAIN_MAX:-20}        # Obergrenze fuer die Kettenlaenge
CHAIN_INDEX=${CHAIN_INDEX:-1}     # wird beim Resubmit hochgezaehlt
DOCK_INI="config/docking.ini"

# ════════════════════════════════════════════════════════════════════════════

cd "$PROJECT_DIR" || exit 1
mkdir -p logs

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "════════════════════════════════════════════════════════════"
log " Docking Pipeline – Kettenglied $CHAIN_INDEX von max. $CHAIN_MAX"
log " Job-ID      : ${SLURM_JOB_ID:-?}"
log " Knoten      : $(hostname -s)"
log " GPUs        : ${SLURM_GPUS_ON_NODE:-?}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
log " Projekt     : $PROJECT_DIR"
log "════════════════════════════════════════════════════════════"

# ── Fortschritt VOR dem Lauf ───────────────────────────────────────────────
python3 src/pipeline_progress.py "$DOCK_INI"
PROGRESS_RC=$?

case $PROGRESS_RC in
    1) log "Nichts mehr offen – Kette endet hier."; exit 0 ;;
    2) log "FEHLER: Fortschritt nicht ermittelbar. Kette wird NICHT fortgesetzt."
       exit 2 ;;
esac

REMAINING_BEFORE=$(python3 src/pipeline_progress.py "$DOCK_INI" --json \
                   | python3 -c 'import json,sys; print(json.load(sys.stdin)["remaining"])')
log "Offen vor diesem Lauf: $REMAINING_BEFORE Liganden"

# ── Folgejob VORAB einreichen ──────────────────────────────────────────────
# Bewusst vor der Rechenarbeit, mit afterany: so entsteht das naechste
# Kettenglied auch dann, wenn dieser Job an der Walltime stirbt, abstuerzt
# oder vom Scheduler abgeraeumt wird. Ein Resubmit am Skriptende wuerde in
# genau diesen Faellen nie erreicht.
NEXT_JOB=""
if [ "$CHAIN_INDEX" -lt "$CHAIN_MAX" ]; then
    # Ausdruecklich der Pfad im Repo, nicht "$0": das zeigt auf die
    # Spool-Kopie, die Slurm nach Jobende aufraeumt. Die Kette wuerde
    # sonst irgendwann ins Leere greifen.
    NEXT_JOB=$(sbatch --parsable \
                      --dependency=afterany:"$SLURM_JOB_ID" \
                      --export=ALL,CHAIN_INDEX=$(( CHAIN_INDEX + 1 )),CHAIN_MAX="$CHAIN_MAX" \
                      "$PROJECT_DIR/dock.slurm" 2>/dev/null)
    if [ -n "$NEXT_JOB" ]; then
        log "Folgejob eingereicht: $NEXT_JOB (Glied $(( CHAIN_INDEX + 1 )))"
        echo "$NEXT_JOB" > "logs/next_job_${SLURM_JOB_ID}.id"
    else
        log "WARNUNG: Folgejob konnte nicht eingereicht werden."
    fi
else
    log "Kettenlaenge $CHAIN_MAX erreicht – kein Folgejob."
fi

# ── Geordneter Abbruch bei USR1 ────────────────────────────────────────────
PIPELINE_PID=""
TERMINATING=0

on_usr1() {
    TERMINATING=1
    log "USR1 empfangen (Walltime naht) – beende geordnet."
    if [ -n "$PIPELINE_PID" ]; then
        # SIGTERM an die Prozessgruppe: Orchestrator, Dispatch-Threads und
        # die Apptainer-Worker. Der aktuell laufende Sub-Batch geht dabei
        # verloren; fertige Posen liegen bereits als *_docked.pdbqt auf der
        # Platte und werden vom naechsten Kettenglied uebersprungen. Der
        # verlorene Batch wird schlicht neu gedockt – hoechstens
        # batch_size Liganden, also Minuten, nicht Stunden.
        kill -TERM -"$PIPELINE_PID" 2>/dev/null || kill -TERM "$PIPELINE_PID" 2>/dev/null
        for _ in $(seq 1 120); do
            kill -0 "$PIPELINE_PID" 2>/dev/null || break
            sleep 5
        done
        kill -KILL -"$PIPELINE_PID" 2>/dev/null || true
    fi
    log "Geordnet beendet."
}
trap on_usr1 USR1

# ── Lauf ───────────────────────────────────────────────────────────────────
# Nur Docking. CHECK_LIGANDS ebenfalls aus: die Pruefung laeuft ueber die
# gesamte Bibliothek und gehoert einmal ans Ende der Konvertierung
# (convert.slurm erledigt das), nicht vor jedes Kettenglied.
STAGE_ENV=(
    RUN_CONVERSION=false
    RUN_DOCKING=true
    RUN_RESCORING="${RUN_RESCORING:-false}"
    CHECK_LIGANDS=false
)

log "Starte Pipeline ..."
# setsid --wait: eigene Prozessgruppe UND $! zeigt auf deren Leader.
# Ohne --wait forkt setsid, $! zeigt auf einen sofort endenden Prozess,
# und 'wait' kehrt zurueck waehrend die Pipeline noch laeuft – der
# USR1-Handler haette dann nichts mehr zu signalisieren.
# Die eigene Prozessgruppe ist noetig, damit 'kill -TERM -PGID' den
# Orchestrator UND die Apptainer-Worker erreicht, nicht nur die Shell.
if setsid --wait true 2>/dev/null; then
    setsid --wait env "${STAGE_ENV[@]}" ./pipeline_start.sh --restart &
else
    log "WARNUNG: setsid --wait nicht verfuegbar, Signalweiterleitung eingeschraenkt."
    env "${STAGE_ENV[@]}" ./pipeline_start.sh --restart &
fi
PIPELINE_PID=$!

# 'wait' kehrt bei einem Signal frueh zurueck; deshalb in der Schleife,
# bis der Prozess wirklich weg ist.
wait "$PIPELINE_PID"
RC=$?
while kill -0 "$PIPELINE_PID" 2>/dev/null; do sleep 5; done

log "Pipeline beendet mit Exit-Code $RC (terminating=$TERMINATING)"

# ── Fortschritt NACH dem Lauf ──────────────────────────────────────────────
python3 src/pipeline_progress.py "$DOCK_INI"
REMAINING_AFTER=$(python3 src/pipeline_progress.py "$DOCK_INI" --json \
                  | python3 -c 'import json,sys; print(json.load(sys.stdin)["remaining"])' \
                  2>/dev/null || echo -1)

log "Offen nach diesem Lauf: $REMAINING_AFTER (vorher: $REMAINING_BEFORE)"

# ── Kette abbrechen, wenn sie nichts mehr bringt ───────────────────────────
# Zwei Faelle, in denen das Weiterlaufen nur Rechenzeit verbrennt:
#   fertig       – nichts mehr offen
#   kein Fortschritt – gleich viele offene Liganden wie vorher, und der Lauf
#                  wurde nicht durch die Walltime unterbrochen. Dann steckt
#                  etwas fest und der naechste Job wuerde dasselbe tun.
CANCEL_REASON=""
if [ "$REMAINING_AFTER" = "0" ]; then
    CANCEL_REASON="alle Liganden abgearbeitet"
elif [ "$TERMINATING" -eq 0 ] && [ "$REMAINING_AFTER" = "$REMAINING_BEFORE" ]; then
    CANCEL_REASON="kein Fortschritt in diesem Lauf (Exit $RC)"
fi

if [ -n "$CANCEL_REASON" ] && [ -n "$NEXT_JOB" ]; then
    log "Kette wird beendet: $CANCEL_REASON"
    scancel "$NEXT_JOB" 2>/dev/null && log "Folgejob $NEXT_JOB storniert."
fi

log "Kettenglied $CHAIN_INDEX abgeschlossen."
exit 0
