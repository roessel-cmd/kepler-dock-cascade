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
#SBATCH -p zen4_0768
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
log "Argumente  : $*"

if command -v pigz >/dev/null 2>&1; then
    log "pigz       : $(pigz --version 2>&1 | head -1)"
else
    log "pigz       : NICHT vorhanden – gzip laeuft einfaedig."
    log "             Bei grossen Targets kann das Stunden dauern."
    log "             Abhilfe: --no-poses, oder pigz ins Modulsystem laden."
fi

# Freien Platz pruefen, bevor stundenlang gearbeitet wird. Ein Archiv, das
# an der vollen Platte scheitert, kostet die gesamte Laufzeit – und mit
# --purge wuerde es zwar nichts loeschen (die Verifikation faengt das ab),
# aber der Job waere umsonst gewesen.
NEED_KB=$(du -sk RESULTS data/LOG 2>/dev/null | awk '{s+=$1} END {print s}')
FREE_KB=$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {print $4}')
log "Rohdaten   : $(( NEED_KB / 1024 / 1024 )) GB"
log "Frei       : $(( FREE_KB / 1024 / 1024 )) GB"
if [ "$FREE_KB" -lt "$(( NEED_KB / 2 ))" ]; then
    log "WARNUNG: Weniger als die halbe Rohdatengroesse frei. PDBQT komprimiert"
    log "         zwar gut (oft auf ein Fuenftel), aber knapp wird es trotzdem."
fi

log "── Start ──"
START=$(date +%s)

srun --cpu-bind=none ./src/archive_run.sh "$@"
RC=$?

log "── Ende ── Exit $RC nach $(( ($(date +%s) - START) / 60 )) min"

if [ "$RC" -eq 0 ]; then
    log "Archive im Verzeichnis:"
    ls -lh archive/*.tar.gz 2>/dev/null | tail -5
fi

exit "$RC"
