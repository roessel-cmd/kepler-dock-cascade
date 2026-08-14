#!/bin/bash
# ============================================================================
# benchmark.sh
# Misst Docking-Durchsatz ueber GPU-Anzahl, search_mode und batch_size.
#
# Fuehrt eine Matrix von Laeufen aus, jeder mit identischer Ligandenmenge,
# und schreibt am Ende eine CSV mit Durchsatz, Speedup, Parallel-Effizienz,
# GPU-Auslastung und Leistungsaufnahme.
#
#     ./benchmark.sh              # Matrix laut Konfiguration unten
#     ./benchmark.sh --dry-run    # nur zeigen, was ausgefuehrt wuerde
# ============================================================================

set -uo pipefail

# ════════════════════════════════════════════════════════════════════════════
#  MESSMATRIX
# ════════════════════════════════════════════════════════════════════════════
# Ein Eintrag pro Lauf:  "<gpus>:<search_mode>:<batch_size>[:<chunk_size>]"

RUNS=(
    "1:fast:1000"
    "1:balance:1000"
    "1:detail:1000"
    "2:fast:1000"
    "2:balance:1000"
    "4:fast:1000"
    "4:balance:1000"
    "4:detail:1000"
)

# Als EIGENE Messreihe fahren, nicht in dieselbe Matrix mischen: die
# GPU-Skalierung beantwortet "wie gut verteilt der Orchestrator", der
# Batch-Sweep "wie gut saettigt die Engine". Dazu oben RUNS auskommentieren
# und stattdessen:
# RUNS=("4:balance:250" "4:balance:500" "4:balance:1000"
#       "4:balance:2000" "4:balance:4000" "4:balance:8000")


# ════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

BENCH_SIF="unidock-gpu.sif"
BENCH_INI_TEMPLATE="config/docking.ini"   # Basis; GPU/Modus/Batch werden ersetzt
BENCH_OUT="benchmark"                     # Ergebnisverzeichnis (Logs + CSV)

BENCH_N_LIGANDS=20000


BENCH_WARMUP=true
BENCH_WARMUP_LIGANDS=2000

# Abtastintervall fuer nvidia-smi in Sekunden. 0 = keine GPU-Telemetrie.
BENCH_SAMPLE_INTERVAL=5
BENCH_SEED=42


# ════════════════════════════════════════════════════════════════════════════
#  AB HIER: KEINE KONFIGURATION MEHR
# ════════════════════════════════════════════════════════════════════════════

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'

say()  { printf '%s[%s]%s %s\n' "$C_BLUE" "$(date '+%H:%M:%S')" "$C_RESET" "$*"; }
ok()   { printf '%s[%s] ✓%s %s\n' "$C_GREEN" "$(date '+%H:%M:%S')" "$C_RESET" "$*"; }
warn() { printf '%s[%s] !%s %s\n' "$C_YELLOW" "$(date '+%H:%M:%S')" "$C_RESET" "$*"; }
err()  { printf '%s[%s] ✗%s %s\n' "$C_RED" "$(date '+%H:%M:%S')" "$C_RESET" "$*" >&2; }
banner() {
    printf '\n%s%s%s\n  %s%s%s\n%s%s%s\n\n' \
        "$C_BOLD" "════════════════════════════════════════════════════════════" "$C_RESET" \
        "$C_BOLD" "$*" "$C_RESET" \
        "$C_BOLD" "════════════════════════════════════════════════════════════" "$C_RESET"
}

ini_get() {   # $1 = Sektion, $2 = Key
    sed -n "/^\[$1\]/,/^\[/p" "$PROJECT_DIR/$BENCH_INI_TEMPLATE" \
        | grep -E "^\s*$2\s*=" | head -1 \
        | sed 's/[^=]*=//; s/[#;].*//; s/^[[:space:]]*//; s/[[:space:]]*$//'
}

PDBQT_DIR="$(ini_get PATHS pdbqt_dir)"
RESULTS_DIR="$(ini_get PATHS results_dir)"
LOG_DIR="$(ini_get PATHS log_dir)"
PDBQT_DIR="${PDBQT_DIR#./}"; RESULTS_DIR="${RESULTS_DIR#./}"; LOG_DIR="${LOG_DIR#./}"

banner "DOCKING BENCHMARK"
say "Projekt   : $PROJECT_DIR"
say "Liganden  : $PROJECT_DIR/$PDBQT_DIR"
say "Laeufe    : ${#RUNS[@]}"
[ "$DRY_RUN" = true ] && warn "DRY-RUN"

for f in "$BENCH_SIF" "$BENCH_INI_TEMPLATE" "src/orchestrator.py" "TARGET/config.txt"; do
    [ -e "$PROJECT_DIR/$f" ] || { err "Fehlt: $f"; exit 1; }
done

mkdir -p "$PROJECT_DIR/$BENCH_OUT"
SUMMARY="$PROJECT_DIR/$BENCH_OUT/summary.csv"

# ── Ligandenausschnitt: fester Satz fuer alle Laeufe ────────────────────────
# Symlinks statt Kopien – die Bibliothek kann Millionen Dateien umfassen.
BENCH_LIG_DIR="$PROJECT_DIR/$BENCH_OUT/ligands"
prepare_ligands() {   # $1 = Anzahl, $2 = Zielverzeichnis
    local n="$1" dst="$2"
    rm -rf "$dst"; mkdir -p "$dst"
    if [ "$n" -eq 0 ]; then
        ln -s "$PROJECT_DIR/$PDBQT_DIR" "$dst/all"
    else
        find "$PROJECT_DIR/$PDBQT_DIR" -name '*.pdbqt' -type f 2>/dev/null \
            | sort | head -n "$n" \
            | while read -r f; do ln -s "$f" "$dst/"; done
    fi
    find "$dst" -name '*.pdbqt' 2>/dev/null | wc -l
}

# ── GPU-Telemetrie ─────────────────────────────────────────────────────────
start_sampling() {   # $1 = Ausgabedatei
    [ "$BENCH_SAMPLE_INTERVAL" -eq 0 ] && return 0
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
               --format=csv,noheader,nounits \
               -l "$BENCH_SAMPLE_INTERVAL" > "$1" 2>/dev/null &
    echo $!
}

stop_sampling() {   # $1 = PID
    [ -z "${1:-}" ] && return 0
    kill "$1" 2>/dev/null; wait "$1" 2>/dev/null
}

summarise_samples() {   # $1 = Datei -> "mittlere_util mittlere_leistung max_speicher"
    [ -s "${1:-/dev/null}" ] || { echo "  "; return; }
    awk -F', *' '{u+=$2; m=($3>m?$3:m); p+=$4; n++}
                 END {if(n) printf "%.1f %.0f %.0f", u/n, p/n, m; else printf "  "}' "$1"
}

# ── Ein Lauf ───────────────────────────────────────────────────────────────
run_one() {   # $1 = Nummer, $2 = gpus, $3 = mode, $4 = batch, $5 = ligandenzahl,
              # $6 = chunk_size (leer = batch * 5)
    local num="$1" gpus="$2" mode="$3" batch="$4" nlig="$5"
    local chunk="${6:-}"
    [ -z "$chunk" ] && chunk=$(( batch * 5 ))
    local label="run_${num}_${gpus}xGPU_${mode}_b${batch}"
    local ini="$PROJECT_DIR/$BENCH_OUT/${label}.ini"
    local log="$PROJECT_DIR/$BENCH_OUT/${label}.log"
    local smi="$PROJECT_DIR/$BENCH_OUT/${label}.gpu.csv"

    banner "LAUF $num — ${gpus} GPU | $mode | batch $batch | chunk $chunk | $nlig Liganden"

    # INI aus der Vorlage ableiten
    sed -E "s|^([[:space:]]*num_gpus[[:space:]]*=).*|\1 $gpus|;
            s|^([[:space:]]*search_mode[[:space:]]*=).*|\1 $mode|;
            s|^([[:space:]]*batch_size[[:space:]]*=).*|\1 $batch|;
            s|^([[:space:]]*chunk_size[[:space:]]*=).*|\1 $chunk|;
            s|^([[:space:]]*seed[[:space:]]*=).*|\1 $BENCH_SEED|;
            s|^([[:space:]]*pdbqt_dir[[:space:]]*=).*|\1 ./$BENCH_OUT/ligands|" \
        "$PROJECT_DIR/$BENCH_INI_TEMPLATE" > "$ini"

    if [ "$DRY_RUN" = true ]; then
        printf '   (dry-run) python3 src/orchestrator.py --config %s --stage dock --sif %s\n' \
            "${ini#$PROJECT_DIR/}" "$BENCH_SIF"
        return 0
    fi

    # RESULTS leeren: sonst zaehlen Posen und Chunk-CSVs des Vorlaufs mit,
    # und die Zeit fuer bereits vorhandene Ergebnisse faellt weg.
    rm -rf "$PROJECT_DIR/$RESULTS_DIR"
    mkdir -p "$PROJECT_DIR/$RESULTS_DIR"
    rm -f "$PROJECT_DIR/$LOG_DIR/pipeline.log"

    local smi_pid; smi_pid=$(start_sampling "$smi")
    local t0; t0=$(date +%s)

    python3 "$PROJECT_DIR/src/orchestrator.py" \
        --project "$PROJECT_DIR" \
        --config  "${ini#$PROJECT_DIR/}" \
        --stage   dock \
        --sif     "$BENCH_SIF" > "$log" 2>&1
    local rc=$?

    local dt=$(( $(date +%s) - t0 ))
    stop_sampling "$smi_pid"

    # Erfolgreiche Liganden aus den Chunk-CSVs zaehlen (nicht aus dem Log:
    # der Durchsatz soll auf tatsaechlich gedockten Molekuelen beruhen)
    local n_ok
    n_ok=$(find "$PROJECT_DIR/$RESULTS_DIR" -name 'docking_results_*.csv' \
           -exec awk -F, 'NR>1 && $2=="True"' {} + 2>/dev/null | wc -l)

    local lph=0
    [ "$dt" -gt 0 ] && lph=$(awk -v n="$n_ok" -v t="$dt" 'BEGIN{printf "%.0f", n/t*3600}')

    read -r util power mem <<< "$(summarise_samples "$smi")"

    if [ $rc -eq 0 ]; then
        ok "$label: ${n_ok} Liganden in ${dt}s → ${lph} L/h | GPU ${util:-?}% | ${power:-?} W"
    else
        err "$label: Exit $rc nach ${dt}s (siehe $log)"
    fi

    echo "$num,$gpus,$mode,$batch,$chunk,$nlig,$n_ok,$dt,$lph,${util:-},${power:-},${mem:-},$rc" \
        >> "$SUMMARY"
}

# ── Aufwaermlauf ───────────────────────────────────────────────────────────
if [ "$BENCH_WARMUP" = true ] && [ "$DRY_RUN" != true ]; then
    banner "AUFWAERMLAUF (wird nicht gewertet)"
    n=$(prepare_ligands "$BENCH_WARMUP_LIGANDS" "$BENCH_LIG_DIR")
    say "$n Liganden"
    SUMMARY_BAK="$SUMMARY"; SUMMARY="/dev/null"
    run_one 0 1 fast 1000 "$n"
    SUMMARY="$SUMMARY_BAK"
fi

# ── Messreihe ──────────────────────────────────────────────────────────────
echo "run,gpus,mode,batch_size,chunk_size,ligands_in,ligands_ok,wall_s,ligands_per_h,gpu_util_pct,power_w,mem_mib_max,exit" \
    > "$SUMMARY"

N_LIG=0
if [ "$DRY_RUN" != true ]; then
    N_LIG=$(prepare_ligands "$BENCH_N_LIGANDS" "$BENCH_LIG_DIR")
    say "Messsatz: $N_LIG Liganden (identisch fuer alle Laeufe)"
fi

i=1
for spec in "${RUNS[@]}"; do
    IFS=: read -r g m b c <<< "$spec"
    run_one "$i" "$g" "$m" "$b" "$N_LIG" "${c:-}"
    i=$(( i + 1 ))
done

# ── Auswertung ─────────────────────────────────────────────────────────────
[ "$DRY_RUN" = true ] && exit 0

banner "ERGEBNIS"
awk -F, 'NR==1 {next}
{
    key = $3 "_" $4
    if (!( key in base ) || $2+0 < basegpu[key]) { base[key]=$9; basegpu[key]=$2+0 }
    rows[NR]=$0
}
END {
    printf "  %-4s %-5s %-9s %-7s %10s %8s %9s %8s %7s %7s\n",
           "Run","GPUs","Modus","Batch","Liganden","Zeit_s","L/h","Speedup","Eff_%","GPU_%"
    printf "  %s\n", "---------------------------------------------------------------------------------------"
    for (r in rows) {
        split(rows[r], f, ",")
        key = f[3] "_" f[4]
        sp  = (base[key] > 0) ? f[9]/base[key] : 0
        eff = (f[2] > 0 && basegpu[key] > 0) ? 100*sp/(f[2]/basegpu[key]) : 0
        printf "  %-4s %-5s %-9s %-7s %10s %8s %9s %7.2fx %6.0f%% %6s%%\n",
               f[1], f[2], f[3], f[4], f[7], f[8], f[9], sp, eff, f[10]
    }
}' "$SUMMARY" | sort -k1,1n

echo
say "CSV: $SUMMARY"
say "Logs: $PROJECT_DIR/$BENCH_OUT/run_*.log"
echo
echo "  Speedup und Effizienz beziehen sich jeweils auf den Lauf mit der"
echo "  kleinsten GPU-Zahl bei gleichem Modus und gleicher Batch-Groesse."
echo "  Effizienz unter 100 % heisst: die zusaetzlichen GPUs bringen weniger"
echo "  als linear – typische Ursachen sind Dispatch-Overhead, zu grosse"
echo "  Chunks am Ende des Laufs oder ein zu kleiner Messsatz."
