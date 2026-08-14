#!/bin/bash
# ============================================================================
# convert.slurm
# Stufe 1 (SDF → PDBQT) als eigener CPU-Job.
#
#     sbatch convert.slurm
#
# Warum getrennt von dock.slurm:
#
#   1. Die Konvertierung braucht KEINE GPU. Sie in einem Job mitlaufen zu
#      lassen, der vier H100 belegt, verschwendet genau die knappe
#      Ressource – bei einer Millionenbibliothek ueber Stunden.
#
#   2. Sie ist EINMALIG. Die Docking-Kette reicht sich ueber die Walltime
#      hinweg immer wieder selbst ein; jedes Kettenglied wuerde die
#      Ligandenbilanz neu berechnen und dabei ueber Millionen Dateien
#      laufen, nur um festzustellen, dass nichts zu tun ist.
#
# Wiederaufnahme: einfach erneut absetzen. Der Konverter ueberspringt per
# --skip-existing alles Fertige und setzt dort fort, wo er stehengeblieben
# ist – die Molekuelindizes bleiben dabei stabil, die Unterordner-
# Aufteilung verschiebt sich also nicht.
# ============================================================================

#SBATCH --job-name=kepler-convert
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/convert-%j.out

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

# Worker an die tatsaechlich zugeteilten Kerne anpassen, statt die 15 aus
# der Voreinstellung zu nehmen. Einen Kern fuer den Producer freilassen:
# er streamt die SDF und wuerde sonst mit einem Worker um Rechenzeit
# konkurrieren.
CORES=${SLURM_CPUS_PER_TASK:-15}
WORKERS=$(( CORES > 2 ? CORES - 1 : 1 ))

log "════════════════════════════════════════════════════════════"
log " Ligandenaufbereitung (Stufe 1)"
log " Job-ID  : ${SLURM_JOB_ID:-?}"
log " Knoten  : $(hostname -s)"
log " Kerne   : $CORES  →  $WORKERS Worker"
log "════════════════════════════════════════════════════════════"

RUN_CONVERSION=true \
RUN_DOCKING=false \
RUN_RESCORING=false \
CHECK_LIGANDS=false \
CONV_WORKERS="$WORKERS" \
    ./pipeline_start.sh
RC=$?

log "Konvertierung beendet mit Exit-Code $RC"

# Bibliothek pruefen, wenn die Konvertierung durch ist. Bewusst erst hier
# und nicht in pipeline_start.sh: der Check laeuft ueber alle erzeugten
# Dateien und lohnt nur einmal am Ende, nicht vor jeder Stufe.
if [ $RC -eq 0 ]; then
    log "Pruefe erzeugte Bibliothek ..."
    python3 src/check_ligands.py "$(sed -n '/^\[PATHS\]/,/^\[/p' config/docking.ini \
        | grep -E '^\s*pdbqt_dir\s*=' | sed 's/[^=]*=//; s/[#;].*//; s/[[:space:]]//g')"
    CHECK_RC=$?
    if [ $CHECK_RC -ne 0 ]; then
        log "WARNUNG: check_ligands.py meldet Probleme (Exit $CHECK_RC)."
        log "         Vor dem Docking beheben – Namenskollisionen fuehren"
        log "         sonst zu stillem Ueberschreiben von Ergebnissen."
        exit $CHECK_RC
    fi
    log "Bibliothek in Ordnung. Naechster Schritt: sbatch dock.slurm"
fi

exit $RC
