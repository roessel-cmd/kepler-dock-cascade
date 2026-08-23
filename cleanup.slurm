#!/bin/bash
# cleanup.slurm — loescht verschobene Verzeichnisse.
#   sbatch cleanup.slurm RESULTS.old_20260823_181602 data/LOG.old_20260823_181602
#
# Wird normalerweise von archive_run.sh --purge eingereicht, nachdem das
# Archiv verifiziert und die Verzeichnisse verschoben wurden.

#SBATCH --job-name=kdc-cleanup
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p zen4_0768
#SBATCH --qos zen4_0768
#SBATCH --mem=4GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/cleanup-%j.out

# Eigener Job statt Hintergrundprozess: aus einem Slurm-Job heraus
# ueberlebt ein nohup-rm das Jobende nicht, Slurm raeumt die Prozesse der
# Job-Step ab. Die .old-Verzeichnisse blieben dann liegen.
# Ein Task genuegt, rm ist I/O-lastige Metadaten-Arbeit auf einem Kern.

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

[ $# -gt 0 ] || { echo "FEHLER: keine Verzeichnisse angegeben." >&2; exit 2; }

log "Aufraeumen — Job ${SLURM_JOB_ID:-?} auf $(hostname -s)"
log "Verzeichnisse: $*"

RC=0
for d in "$@"; do
    if [ ! -e "$d" ]; then
        log "  $d existiert nicht – uebersprungen."
        continue
    fi

    # Geloescht wird nur, was wie ein verschobenes Verzeichnis aussieht –
    # ein durchgereichtes "RESULTS" oder "/tmp" waere sonst fatal.
    case "$(basename "$d")" in
        *.old_[0-9]*) ;;
        *) log "  ABGELEHNT: '$d' passt nicht auf *.old_<zeitstempel>."
           RC=1; continue ;;
    esac

    n=$(find "$d" -type f 2>/dev/null | wc -l)
    log "  $d: $n Datei(en), loesche ..."
    t0=$(date +%s)
    if rm -r "$d"; then
        log "  $d entfernt nach $(( ($(date +%s) - t0) / 60 )) min"
    else
        log "  FEHLER beim Loeschen von $d"
        RC=1
    fi
done

log "Fertig, Exit $RC"
exit "$RC"
