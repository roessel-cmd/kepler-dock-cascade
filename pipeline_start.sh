#!/bin/bash
# ============================================================================
# pipeline_start.sh
# Steuert alle drei Pipeline-Stufen sequentiell.
#
#   Stufe 1  CONVERSION   sdf_to_pdbqt.sif    SDF → PDBQT (RDKit + Meeko, CPU)
#   Stufe 2  DOCKING      unidock-gpu.sif     Uni-Dock, Multi-GPU
#   Stufe 3  RESCORING    rescoring-gpu.sif   gnina / ECR / Refinement
#
# Jede Stufe laesst sich einzeln an- und abschalten. Die Stufen laufen
# nacheinander; schlaegt eine fehl, bricht das Skript ab (ausser
# CONTINUE_ON_ERROR=true).
#
# Aufruf:
#     ./pipeline_start.sh                 # Schalter unten verwenden
#     ./pipeline_start.sh --dry-run       # nur zeigen was passieren wuerde
#     ./pipeline_start.sh --restart       # Docking-Stufe im Restart-Modus
# ============================================================================

set -uo pipefail

# ════════════════════════════════════════════════════════════════════════════
#  SCHALTER
# ════════════════════════════════════════════════════════════════════════════

RUN_CONVERSION=true      # Stufe 1: SDF → PDBQT
RUN_DOCKING=false         # Stufe 2: Uni-Dock
RUN_RESCORING=false       # Stufe 3: Rescoring + Refinement

# Vorabpruefungen (schnell, verhindern stundenlange Fehllaeufe)
CHECK_CONFIG=true        # docking.ini gegen rescore.ini pruefen
CHECK_LIGANDS=true       # PDBQT-Bibliothek pruefen (Layout, Namenskollisionen)

# Bei Fehler einer Stufe trotzdem weitermachen
CONTINUE_ON_ERROR=false


# ════════════════════════════════════════════════════════════════════════════
#  STUFE 1 – CONVERSION
# ════════════════════════════════════════════════════════════════════════════

CONV_SIF="sdf_to_pdbqt.sif"
CONV_LIB_DIR="LIB"             # Eingangs-SDFs
CONV_OUT_DIR="data/PDBQT"      # muss [PATHS] pdbqt_dir aus docking.ini entsprechen
CONV_WORKERS=15
CONV_TIMEOUT=120               # Sekunden pro Molekuel
CONV_UFF_ITERS=800
CONV_FLAT=false                # true = alle PDBQTs in einen Ordner
                               # false = Unterordner 0000/, 0001/ (empfohlen)

# Bei mehreren SDFs in LIB bekommt jede ihren eigenen Unterordner unter
# CONV_OUT_DIR. Das verhindert, dass die 0000/-Zaehlung zweier Dateien
# kollidiert, und die rekursive Ligandensuche im Docking findet sie trotzdem.
CONV_SUBDIR_PER_SDF=true

# Wiederaufnahme: bereits konvertierte Molekuele ueberspringen.
# Der Konverter prueft das pro Molekuel anhand des Ziel-PDBQT, nicht pauschal
# pro SDF. Ein abgebrochener Lauf wird damit genau dort fortgesetzt, wo er
# stehengeblieben ist.
CONV_RESUME=true

# Nur ueberspringen wenn die SDF nachweislich VOLLSTAENDIG konvertiert ist,
# also  konvertiert + fehlgeschlagen >= Molekuele im SDF.  false = immer
# durchlaufen (der Konverter ueberspringt Fertiges dann selbst).
CONV_SKIP_IF_COMPLETE=true


# ════════════════════════════════════════════════════════════════════════════
#  STUFE 2 – DOCKING
# ════════════════════════════════════════════════════════════════════════════

DOCK_SIF="unidock-gpu.sif"
DOCK_INI="config/docking.ini"
DOCK_RESTART=false             # true = restart_orchestrator statt orchestrator
                               # (--restart auf der Kommandozeile setzt das auch)


# ════════════════════════════════════════════════════════════════════════════
#  STUFE 3 – RESCORING
# ════════════════════════════════════════════════════════════════════════════

RESCORE_SIF="rescoring-gpu.sif"
RESCORE_INI="config/rescore.ini"


# ════════════════════════════════════════════════════════════════════════════
#  AB HIER: KEINE KONFIGURATION MEHR
# ════════════════════════════════════════════════════════════════════════════

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/data/LOG"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --restart) DOCK_RESTART=true ;;
        --help|-h)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unbekannte Option: $arg" >&2; exit 2 ;;
    esac
done

# ── Ausgabe-Helfer ──────────────────────────────────────────────────────────

C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
C_RED=$'\033[31m';  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'

say()  { printf '%s[%s]%s %s\n' "$C_BLUE" "$(date '+%H:%M:%S')" "$C_RESET" "$*"; }
ok()   { printf '%s[%s] ✓%s %s\n' "$C_GREEN" "$(date '+%H:%M:%S')" "$C_RESET" "$*"; }
warn() { printf '%s[%s] !%s %s\n' "$C_YELLOW" "$(date '+%H:%M:%S')" "$C_RESET" "$*"; }
err()  { printf '%s[%s] ✗%s %s\n' "$C_RED" "$(date '+%H:%M:%S')" "$C_RESET" "$*" >&2; }

banner() {
    printf '\n%s%s%s\n' "$C_BOLD" "════════════════════════════════════════════════════════════" "$C_RESET"
    printf '%s  %s%s\n' "$C_BOLD" "$*" "$C_RESET"
    printf '%s%s%s\n\n' "$C_BOLD" "════════════════════════════════════════════════════════════" "$C_RESET"
}

fmt_duration() {
    local s=$1
    printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

run() {
    if [ "$DRY_RUN" = true ]; then
        printf '   %s(dry-run)%s %s\n' "$C_YELLOW" "$C_RESET" "$*"
        return 0
    fi
    "$@"
}

# ── Voraussetzungen ─────────────────────────────────────────────────────────

require_file() {
    if [ ! -f "$1" ]; then
        err "Nicht gefunden: $1${2:+ ($2)}"
        return 1
    fi
    return 0
}

if ! command -v apptainer >/dev/null 2>&1; then
    err "apptainer nicht im PATH."
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Stufen-Ausfuehrung mit Zeitmessung ──────────────────────────────────────

declare -A STAGE_STATUS
declare -A STAGE_TIME
PIPELINE_START=$(date +%s)

finish_stage() {   # $1 = Name, $2 = Exit-Code, $3 = Startzeit
    local name=$1 rc=$2 t0=$3
    local dt=$(( $(date +%s) - t0 ))
    STAGE_TIME[$name]=$dt
    if [ "$rc" -eq 0 ]; then
        STAGE_STATUS[$name]="OK"
        ok "$name abgeschlossen ($(fmt_duration $dt))"
    else
        STAGE_STATUS[$name]="FEHLER ($rc)"
        err "$name fehlgeschlagen mit Exit-Code $rc ($(fmt_duration $dt))"
        if [ "$CONTINUE_ON_ERROR" != true ]; then
            summary
            exit "$rc"
        fi
        warn "CONTINUE_ON_ERROR=true – mache weiter"
    fi
}

summary() {
    local total=$(( $(date +%s) - PIPELINE_START ))
    banner "ZUSAMMENFASSUNG"
    for stage in CONVERSION DOCKING RESCORING; do
        local status="${STAGE_STATUS[$stage]:-uebersprungen}"
        local t="${STAGE_TIME[$stage]:-}"
        if [ -n "$t" ]; then
            printf '  %-12s %-14s %s\n' "$stage" "$status" "$(fmt_duration "$t")"
        else
            printf '  %-12s %s\n' "$stage" "$status"
        fi
    done
    printf '\n  %-12s %s\n\n' "GESAMT" "$(fmt_duration $total)"
}

banner "DOCKING PIPELINE"
say "Projekt : $PROJECT_DIR"
say "Stufen  : Conversion=$RUN_CONVERSION | Docking=$RUN_DOCKING | Rescoring=$RUN_RESCORING"
[ "$DRY_RUN" = true ] && warn "DRY-RUN – es wird nichts ausgefuehrt"


# ════════════════════════════════════════════════════════════════════════════
#  VORABPRUEFUNGEN
# ════════════════════════════════════════════════════════════════════════════

if [ "$CHECK_CONFIG" = true ] && [ "$DRY_RUN" != true ]; then
    if [ -f "$PROJECT_DIR/src/check_config.py" ]; then
        say "Pruefe INI-Konsistenz ..."
        if ! python3 "$PROJECT_DIR/src/check_config.py" "$PROJECT_DIR/config"; then
            err "Konfiguration inkonsistent – Abbruch."
            exit 1
        fi
    else
        warn "src/check_config.py nicht gefunden – Pruefung uebersprungen"
    fi
fi


# ════════════════════════════════════════════════════════════════════════════
#  STUFE 1 – CONVERSION
# ════════════════════════════════════════════════════════════════════════════

if [ "$RUN_CONVERSION" = true ]; then
    banner "STUFE 1 – CONVERSION (SDF → PDBQT)"
    T0=$(date +%s)
    RC=0

    require_file "$PROJECT_DIR/$CONV_SIF" "Container fuer Stufe 1" || exit 1

    shopt -s nullglob
    SDF_FILES=("$PROJECT_DIR/$CONV_LIB_DIR"/*.sdf)
    shopt -u nullglob

    if [ ${#SDF_FILES[@]} -eq 0 ]; then
        err "Keine .sdf-Dateien in $PROJECT_DIR/$CONV_LIB_DIR"
        RC=1
    else
        say "${#SDF_FILES[@]} SDF-Datei(en) gefunden"
        mkdir -p "$PROJECT_DIR/$CONV_OUT_DIR"

        for sdf in "${SDF_FILES[@]}"; do
            stem="$(basename "$sdf" .sdf)"

            if [ "$CONV_SUBDIR_PER_SDF" = true ] && [ ${#SDF_FILES[@]} -gt 1 ]; then
                out_sub="$CONV_OUT_DIR/$stem"
            else
                out_sub="$CONV_OUT_DIR"
            fi
            mkdir -p "$PROJECT_DIR/$out_sub"

            # ── Bilanz: wieviel ist von dieser SDF schon erledigt? ──
            n_done=$(find "$PROJECT_DIR/$out_sub" -name '*.pdbqt' -type f 2>/dev/null | wc -l)
            n_fail=$(find "$LOG_DIR" -name '*_convert_error.log' -type f 2>/dev/null | wc -l)
            # "$$$$" trennt Molekuele im SDF; bei .gz entsprechend dekomprimiert
            case "$sdf" in
                *.gz) n_total=$(zgrep -c '^\$\$\$\$' "$sdf" 2>/dev/null || echo 0) ;;
                *)    n_total=$(grep  -c '^\$\$\$\$' "$sdf" 2>/dev/null || echo 0) ;;
            esac
            n_handled=$(( n_done + n_fail ))

            say "$stem: $n_total Molekuele | $n_done konvertiert | $n_fail fehlgeschlagen | $(( n_total - n_handled )) offen"

            if [ "$CONV_SKIP_IF_COMPLETE" = true ] && [ "$n_total" -gt 0 ] \
               && [ "$n_handled" -ge "$n_total" ]; then
                warn "$stem: vollstaendig – uebersprungen"
                continue
            fi

            if [ "$n_done" -gt 0 ]; then
                say "Setze $stem fort → $out_sub"
            else
                say "Konvertiere $stem → $out_sub"
            fi

            CONV_ARGS=(
                --sdf-file "/data/sdf/$(basename "$sdf")"
                --out-dir  "/data/pdbqt"
                --log-dir  "/data/log"
                --workers  "$CONV_WORKERS"
                --timeout  "$CONV_TIMEOUT"
                --uff-iters "$CONV_UFF_ITERS"
            )
            [ "$CONV_FLAT" = true ]   && CONV_ARGS+=(--flat)
            [ "$CONV_RESUME" = true ] && CONV_ARGS+=(--skip-existing)

            run apptainer run \
                --bind "$PROJECT_DIR/$CONV_LIB_DIR:/data/sdf" \
                --bind "$PROJECT_DIR/$out_sub:/data/pdbqt" \
                --bind "$LOG_DIR:/data/log" \
                "$PROJECT_DIR/$CONV_SIF" \
                "${CONV_ARGS[@]}"
            rc=$?
            [ $rc -ne 0 ] && RC=$rc
        done
    fi

    finish_stage CONVERSION "$RC" "$T0"
else
    say "STUFE 1 – CONVERSION uebersprungen (RUN_CONVERSION=false)"
fi


# ── Ligandenpruefung: nach Stufe 1, vor Stufe 2 ─────────────────────────────

if [ "$RUN_DOCKING" = true ] && [ "$CHECK_LIGANDS" = true ] && [ "$DRY_RUN" != true ]; then
    if [ -f "$PROJECT_DIR/src/check_ligands.py" ]; then
        say "Pruefe Liganden-Bibliothek ..."
        if ! python3 "$PROJECT_DIR/src/check_ligands.py" "$PROJECT_DIR/$CONV_OUT_DIR"; then
            err "Ligandenbibliothek fehlerhaft – Docking wuerde Ergebnisse "
            err "ueberschreiben. Abbruch."
            exit 1
        fi
    else
        warn "src/check_ligands.py nicht gefunden – Pruefung uebersprungen"
    fi
fi


# ════════════════════════════════════════════════════════════════════════════
#  STUFE 2 – DOCKING
# ════════════════════════════════════════════════════════════════════════════

if [ "$RUN_DOCKING" = true ]; then
    if [ "$DOCK_RESTART" = true ]; then
        banner "STUFE 2 – DOCKING (Uni-Dock, RESTART)"
        ORCH="src/restart_orchestrator.py"
    else
        banner "STUFE 2 – DOCKING (Uni-Dock)"
        ORCH="src/orchestrator.py"
    fi
    T0=$(date +%s)

    require_file "$PROJECT_DIR/$DOCK_SIF" "Container fuer Stufe 2" || exit 1
    require_file "$PROJECT_DIR/$DOCK_INI" "INI fuer Stufe 2"       || exit 1
    require_file "$PROJECT_DIR/$ORCH"     "Orchestrator"           || exit 1

    # Log rotieren – nur beim Docking, das ist der lange Lauf
    LOG_FILE="$LOG_DIR/pipeline.log"
    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ] && [ "$DRY_RUN" != true ]; then
        ARCHIVE="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"
        mv "$LOG_FILE" "$ARCHIVE"
        say "Alter Log archiviert: $(basename "$ARCHIVE")"
    fi

    run python3 "$PROJECT_DIR/$ORCH" \
        --project "$PROJECT_DIR" \
        --config  "$DOCK_INI" \
        --stage   dock \
        --sif     "$DOCK_SIF"
    finish_stage DOCKING $? "$T0"
else
    say "STUFE 2 – DOCKING uebersprungen (RUN_DOCKING=false)"
fi


# ════════════════════════════════════════════════════════════════════════════
#  STUFE 3 – RESCORING
# ════════════════════════════════════════════════════════════════════════════

if [ "$RUN_RESCORING" = true ]; then
    banner "STUFE 3 – RESCORING + REFINEMENT"
    T0=$(date +%s)

    require_file "$PROJECT_DIR/$RESCORE_SIF" "Container fuer Stufe 3" || exit 1
    require_file "$PROJECT_DIR/$RESCORE_INI" "INI fuer Stufe 3"       || exit 1

    run python3 "$PROJECT_DIR/src/orchestrator.py" \
        --project "$PROJECT_DIR" \
        --config  "$RESCORE_INI" \
        --stage   rescore \
        --sif     "$RESCORE_SIF"
    finish_stage RESCORING $? "$T0"
else
    say "STUFE 3 – RESCORING uebersprungen (RUN_RESCORING=false)"
fi


# ════════════════════════════════════════════════════════════════════════════

summary

FAILED=0
for stage in "${!STAGE_STATUS[@]}"; do
    [[ "${STAGE_STATUS[$stage]}" == FEHLER* ]] && FAILED=1
done

if [ $FAILED -ne 0 ]; then
    warn "Docking fortsetzen: ./pipeline_start.sh --restart"
    exit 1
fi
exit 0
