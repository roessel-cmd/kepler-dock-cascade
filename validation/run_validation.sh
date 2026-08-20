#!/usr/bin/env bash
# =============================================================================
# run_validation.sh
# Führt enrichment_analysis.py + ecr_cross_validation.py für alle Experimente
# sequenziell im Apptainer-Container aus.
#
# Die Liste der Experimente wird automatisch aus ecr_cross_validation.py gezogen
# und gegen enrichment_analysis.py abgeglichen – so kann es keinen Drift
# zwischen Bash und Python mehr geben.
#
# Usage:
#   bash ~/run_validation.sh                        # alle Experimente
#   bash ~/run_validation.sh one_per_family         # nur ein bestimmtes Experiment
#   bash ~/run_validation.sh --list                 # nur die verfügbaren Experimente anzeigen
# =============================================================================

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
SIF="${HOME}/validation.sif"
PY_SCRIPTS_DIR="${HOME}/scripts"
DATA_DIR="/home/roessel/gpu8.0"

# ── Farben ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()    { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════════════${NC}";
            echo -e "${CYAN}${BOLD}  $*${NC}";
            echo -e "${CYAN}${BOLD}══════════════════════════════════════════════${NC}"; }
expstep() { echo -e "\n${MAGENTA}${BOLD}╔══════════════════════════════════════════════╗${NC}";
            echo -e "${MAGENTA}${BOLD}║  Experiment: $*${NC}";
            echo -e "${MAGENTA}${BOLD}╚══════════════════════════════════════════════╝${NC}"; }

# ── Startzeit ────────────────────────────────────────────────────────────────
GLOBAL_START=$(date +%s)
info "Analyse gestartet: $(date '+%Y-%m-%d %H:%M:%S')"

# ── Checks ───────────────────────────────────────────────────────────────────
command -v apptainer >/dev/null 2>&1 || error "apptainer nicht gefunden."
[ -f "${SIF}" ]            || error "Container nicht gefunden: ${SIF}"
[ -d "${PY_SCRIPTS_DIR}" ] || error "Skript-Verzeichnis nicht gefunden: ${PY_SCRIPTS_DIR}"
[ -d "${DATA_DIR}" ]       || error "Daten-Verzeichnis nicht gefunden: ${DATA_DIR}"

# Beide Python-Skripte müssen existieren, bevor wir die Experiment-Liste ziehen
[ -f "${PY_SCRIPTS_DIR}/ecr_cross_validation.py" ] || \
    error "Skript nicht gefunden: ${PY_SCRIPTS_DIR}/ecr_cross_validation.py"
[ -f "${PY_SCRIPTS_DIR}/enrichment_analysis.py" ] || \
    error "Skript nicht gefunden: ${PY_SCRIPTS_DIR}/enrichment_analysis.py"

info "Container:   ${SIF}"
info "Skripte:     ${PY_SCRIPTS_DIR}  →  /scripts"
info "Daten:       ${DATA_DIR}  →  /data"

# ── Experiment-Liste automatisch aus den Python-Skripten ziehen ──────────────
# Wir laden EXPERIMENTS aus beiden Skripten und prüfen, dass sie identisch
# sind. So fällt es sofort auf, wenn man eines der beiden ändert und das
# andere vergisst.
info "Lese verfügbare Experimente aus den Python-Skripten ..."

# Beide Module drucken beim Import ihre Konfiguration auf stdout
# ([cross_validate] Experiment: ... usw.). Ohne Praefix landen diese Zeilen
# als Pseudo-Experimente im Array, --list zeigt zehn statt vier Eintrage und
# der Lauf ueber alle Experimente bricht beim ersten davon ab. Deshalb wird
# jede echte Zeile mit EXPERIMENT: markiert und unten herausgefiltert.
PY_EXTRACT='
import sys
sys.path.insert(0, "/scripts")
from ecr_cross_validation import EXPERIMENTS as CV_EXP, SCORES as CV_SCORES
from enrichment_analysis import (EXPERIMENTS as AN_EXP,
                                    SCORES_COMPONENTS_ALL as AN_SCORES)

cv_keys = list(CV_EXP.keys())
an_keys = list(AN_EXP.keys())
problems = []

if set(cv_keys) != set(an_keys):
    only_cv = sorted(set(cv_keys) - set(an_keys))
    only_an = sorted(set(an_keys) - set(cv_keys))
    problems.append("EXPERIMENTS-Dicts stimmen nicht ueberein.")
    if only_cv:
        problems.append(f"  Nur in ecr_cross_validation.py:        {only_cv}")
    if only_an:
        problems.append(f"  Nur in enrichment_analysis.py: {only_an}")

# Auch die Score-Zusammensetzung je Experiment muss uebereinstimmen,
# sonst laufen Analyse und Kreuzvalidierung auf verschiedenen Ensembles.
for k in sorted(set(cv_keys) & set(an_keys)):
    if list(CV_EXP[k]) != list(AN_EXP[k]):
        problems.append(f"Experiment {k!r} enthaelt verschiedene Scores:")
        problems.append(f"  cross_validate:  {list(CV_EXP[k])}")
        problems.append(f"  analyze:         {list(AN_EXP[k])}")

# Und die Richtungen (higher_is_better) — ein Unterschied hier dreht die
# ECR in einem der beiden Skripte um, ohne dass es auffaellt.
cv_dir = {c: h for c, _l, h, _e in CV_SCORES}
an_dir = {c: h for c, _l, h, _e in AN_SCORES}
for col in sorted(set(cv_dir) & set(an_dir)):
    if cv_dir[col] != an_dir[col]:
        problems.append(
            f"higher_is_better fuer {col!r} unterschiedlich: "
            f"cross_validate={cv_dir[col]}, analyze={an_dir[col]}")
for col in sorted(set(cv_dir) ^ set(an_dir)):
    problems.append(f"Score {col!r} nur in einem der beiden Kataloge.")

if problems:
    sys.stderr.write("\n".join(problems) + "\n")
    sys.exit(2)

# Reihenfolge aus ecr_cross_validation.py beibehalten (dort definiert)
for k in cv_keys:
    print("EXPERIMENT:" + k)
'

# mapfile liest jede Zeile in ein Array-Element. stderr laeuft durch, damit
# Python-Fehlermeldungen sichtbar bleiben; aus stdout wird nur das
# herausgefiltert, was der Extraktor als Experiment markiert hat.
EXTRACT_OUT=$(
    apptainer exec \
        --bind "${PY_SCRIPTS_DIR}:/scripts" \
        "${SIF}" \
        python3 -c "${PY_EXTRACT}"
) || error "Konnte Experiment-Liste nicht aus den Python-Skripten extrahieren."

mapfile -t ALL_EXPERIMENTS < <(printf '%s\n' "${EXTRACT_OUT}" | sed -n 's/^EXPERIMENT://p')

if [ ${#ALL_EXPERIMENTS[@]} -eq 0 ]; then
    error "Experiment-Liste ist leer – stimmt der Pfad zu den Python-Skripten?"
fi

info "Gefunden: ${#ALL_EXPERIMENTS[@]} Experimente"
for exp in "${ALL_EXPERIMENTS[@]}"; do
    echo "    • ${exp}"
done

# ── --list: nur anzeigen, dann beenden ───────────────────────────────────────
if [ $# -ge 1 ] && [ "$1" = "--list" ]; then
    info "Nur Auflistung angefordert – Ende."
    exit 0
fi

# ── Welche Experimente laufen? ───────────────────────────────────────────────
if [ $# -ge 1 ]; then
    # Einzelnes Experiment per Argument – prüfen ob es existiert
    REQUESTED="$1"
    FOUND=0
    for exp in "${ALL_EXPERIMENTS[@]}"; do
        if [ "${exp}" = "${REQUESTED}" ]; then
            FOUND=1
            break
        fi
    done
    if [ ${FOUND} -eq 0 ]; then
        error "Unbekanntes Experiment: '${REQUESTED}'. Verfügbar: ${ALL_EXPERIMENTS[*]}"
    fi
    EXPERIMENTS=("${REQUESTED}")
    info "Modus: einzelnes Experiment → ${REQUESTED}"
else
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
    info "Modus: alle ${#ALL_EXPERIMENTS[@]} Experimente"
fi

# ── Hilfsfunktion ────────────────────────────────────────────────────────────
run_script() {
    local step_num="$1"
    local script_name="$2"
    local exp_name="$3"
    local total_steps="$4"

    step "Schritt ${step_num}/${total_steps}: ${script_name}  [${exp_name}]"

    [ -f "${PY_SCRIPTS_DIR}/${script_name}" ] || \
        error "Skript nicht gefunden: ${PY_SCRIPTS_DIR}/${script_name}"

    local t_start
    t_start=$(date +%s)

    # set -e + lokales exit_code: temporär abschalten, damit wir die Laufzeit
    # auch im Fehlerfall berechnen und eine saubere Meldung ausgeben können.
    set +e
    EXPERIMENT_NAME="${exp_name}" \
    apptainer exec \
        --bind "${PY_SCRIPTS_DIR}:/scripts" \
        --bind "${DATA_DIR}:/data" \
        "${SIF}" \
        python3 "/scripts/${script_name}"
    local exit_code=$?
    set -e

    local elapsed=$(( $(date +%s) - t_start ))

    [ ${exit_code} -eq 0 ] || \
        error "Schritt ${step_num} FEHLGESCHLAGEN (Exit-Code: ${exit_code}) nach ${elapsed}s"
    info "Schritt ${step_num} erfolgreich in ${elapsed}s."
}

# ── Haupt-Loop über alle Experimente ─────────────────────────────────────────
N_EXP=${#EXPERIMENTS[@]}
EXP_INDEX=0

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_INDEX=$(( EXP_INDEX + 1 ))
    EXP_START=$(date +%s)

    expstep "${EXP}  (${EXP_INDEX}/${N_EXP})"

    run_script 1 "enrichment_analysis.py" "${EXP}" 2
    run_script 2 "ecr_cross_validation.py" "${EXP}" 2

    EXP_ELAPSED=$(( $(date +%s) - EXP_START ))
    info "Experiment '${EXP}' abgeschlossen in $(( EXP_ELAPSED / 60 ))m $(( EXP_ELAPSED % 60 ))s."
done

# ── Gesamtlaufzeit ────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - GLOBAL_START ))

echo ""
info "================================================================"
info " Alle ${N_EXP} Experimente erfolgreich abgeschlossen!"
info " Gesamtlaufzeit: $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
info " Beendet:        $(date '+%Y-%m-%d %H:%M:%S')"
info "================================================================"
