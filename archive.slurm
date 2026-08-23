#!/bin/bash
# archive.slurm — abgeschlossenen Lauf archivieren, auf der CPU-Partition.
#   sbatch archive.slurm dock-76523
#   sbatch archive.slurm 76523 --purge
#   sbatch archive.slurm dock-76523 --no-poses --top 500
#
# Alles nach dem Skriptnamen wird an src/archive_run.sh durchgereicht.
# Kein nohup noetig – der Job laeuft ohnehin im Hintergrund und ueberlebt
# das Abmelden.

#SBATCH --job-name=kdc-archive
#SBATCH -N 1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --mail-user=thomas.roessel@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/archive-%j.out

# Ganze Node: Tarren und Komprimieren sind CPU- und I/O-Arbeit, keine GPU.
# Bei 2x AMD 9654 stehen damit 192 Kerne fuer pigz bereit.
#
# Acht Stunden sind grosszuegig. Mit pigz auf 192 Kernen ist auch ein
# Target mit 2,29 Mio. Posen (~57 GB) in unter einer Stunde durch; ohne
# pigz laeuft gzip einfaedig und braucht ein Vielfaches. Die Verifikation
# entpackt das Archiv noch einmal vollstaendig, das kommt hinzu.

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "$PROJECT_DIR/pipeline_start.sh" ]; then
    echo "FEHLER: $PROJECT_DIR ist nicht die Projektwurzel." >&2
    exit 2
fi
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

[ -x src/archive_run.sh ] || {
    echo "FEHLER: src/archive_run.sh fehlt oder ist nicht ausfuehrbar." >&2
    echo "        chmod +x src/archive_run.sh" >&2
    exit 2
}

[ $# -gt 0 ] || {
    echo "FEHLER: Jobname fehlt.  sbatch archive.slurm dock-76523 [--purge]" >&2
    exit 2
}

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "Archivierung — Job ${SLURM_JOB_ID:-?} auf $(hostname -s)"
log "Kerne      : ${SLURM_CPUS_ON_NODE:-?}"
# Ohne srun ist SLURM_CPUS_PER_TASK nicht gesetzt oder zu klein
# (Job 77159 meldete 22 statt 176). Node-Kerne sind hier die richtige Zahl.
export ARCHIVE_THREADS="${ARCHIVE_THREADS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"
log "pigz-Threads: $ARCHIVE_THREADS"
log "Argumente  : $*"

if command -v pigz >/dev/null 2>&1; then
    log "pigz       : $(pigz --version 2>&1 | head -1)"
else
    log "pigz       : NICHT vorhanden – gzip laeuft einfaedig."
    log "             Bei grossen Targets kann das Stunden dauern."
    log "             Abhilfe: --no-poses, oder pigz ins Modulsystem laden."
fi

# Nur df – das ist ein Systemaufruf. Ein 'du' ueber RESULTS liest die
# Metadaten von Millionen Dateien und brauchte in Job 77159 zehn Minuten,
# bevor ueberhaupt etwas passierte. archive_run.sh erhebt die Groessen
# selbst, dort aber nur wenn sie gebraucht werden.
FREE_KB=$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {print $4}')
log "Frei       : $(( FREE_KB / 1024 / 1024 )) GB"

log "── Start ──"
START=$(date +%s)

# Direkt aufrufen, NICHT ueber srun: auf der exklusiven Node startete
# srun das Skript einmal pro Task (Job 77159: tasks 0-3). Vier Prozesse
# schrieben in dasselbe Staging-Verzeichnis und dasselbe Archiv – das
# Ergebnis war unlesbar. Die Parallelitaet steckt in pigz, nicht hier.
./src/archive_run.sh "$@"
RC=$?

log "── Ende ── Exit $RC nach $(( ($(date +%s) - START) / 60 )) min"

if [ "$RC" -eq 0 ]; then
    log "Archive im Verzeichnis:"
    ls -lh archive/*.tar.gz 2>/dev/null | tail -5
fi

exit "$RC"
