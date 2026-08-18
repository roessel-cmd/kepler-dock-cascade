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
#
# Matrix, Ausgabeverzeichnis und Messsatz lassen sich per Umgebungsvariable
# ueberschreiben, ohne das Skript zu kopieren:
#
#     BENCH_RUNS="4:balance:250:250 4:balance:500:500" \
#     BENCH_OUT=benchmark_batch BENCH_N_LIGANDS=60000 ./benchmark.sh
#
# WICHTIG bei einer zweiten Messreihe: BENCH_OUT aendern. Sonst
# ueberschreibt sie summary.csv und die Logs der ersten.
#
# Voraussetzung: die Liganden liegen bereits als PDBQT vor (Stufe 1 ist
# nicht Teil der Messung), und TARGET/config.txt enthaelt mindestens ein
# Target mit vorhandenem Rezeptor-PDBQT.
#
# ── Aenderungen gegenueber der Erstfassung ─────────────────────────────────
#   * nvidia-smi tastet nur noch die tatsaechlich genutzten GPUs ab
#     (BENCH_SMI_IDS). Vorher wurden bei einem 1-GPU-Lauf auf einem
#     4-GPU-Knoten drei idle Karten mitgemittelt -> Auslastung ~25 %.
#   * Aufwaermlauf nutzt die maximale GPU-Zahl der Matrix, nicht fix 1.
#   * BUGFIX prepare_ligands: 'find -L' bei BENCH_N_LIGANDS=0
#     (find folgt Symlinks nicht -> Zaehlung lieferte immer 0).
#   * BUGFIX Ligandenzaehlung: FNR>1 statt NR>1 (NR ueberspringt nur den
#     Header der ersten CSV).
#   * Auswertung: Kopfzeile laeuft nicht mehr durch 'sort'.
# ============================================================================

set -uo pipefail

# ════════════════════════════════════════════════════════════════════════════
#  MESSMATRIX
# ════════════════════════════════════════════════════════════════════════════
# Ein Eintrag pro Lauf:  "<gpus>:<search_mode>:<batch_size>[:<chunk_size>]"
# Reihenfolge ist die Ausfuehrungsreihenfolge.
#
# chunk_size ist optional. Ohne Angabe wird batch_size * 5 gesetzt, damit
# jeder Chunk in genau fuenf gleich grosse Sub-Batches zerfaellt. Das ist
# beim Batch-Sweep entscheidend: bei festem chunk_size=5000 ergaebe
# batch_size=4000 die Aufteilung 4000+1000, und der 1000er-Rest laeuft bei
# geringerer Sättigung. batch=4000 saehe dadurch schlechter aus als es ist.

# Per Umgebungsvariable ueberschreibbar (Eintraege durch Leerzeichen getrennt)
if [ -n "${BENCH_RUNS:-}" ]; then
    read -r -a RUNS <<< "$BENCH_RUNS"
else
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
fi

# ACHTUNG zur Laufzeit: 'detail' kostet ein Vielfaches von 'balance'. Der
# 1-GPU-detail-Lauf ist damit der mit Abstand laengste der Matrix und
# bestimmt, wie lange das Ganze dauert. Er ist trotzdem drin, weil ohne ihn
# der Speedup fuer 4:detail keine Basis haette. Wenn die Zeit knapp ist:
# diese eine Zeile streichen – 4:detail wird dann zu seiner eigenen Basis
# und zeigt Speedup 1.00x, der Durchsatzwert bleibt aber gueltig.
#
# Besser als streichen: 'detail' als EIGENE Messreihe mit kleinerem
# Ligandensatz fahren. Die Speedup-Basis wird unten je (Modus, Batch)
# gebildet, Modi sind also ohnehin voneinander unabhaengig:
#     BENCH_OUT=bench_fb     BENCH_N_LIGANDS=200000 BENCH_RUNS="1:fast:... "
#     BENCH_OUT=bench_detail BENCH_N_LIGANDS=40000  BENCH_RUNS="1:detail:..."
# Nur die absoluten L/h sind zwischen den Reihen nicht vergleichbar – das
# waren sie zwischen Modi aber ohnehin nie.

# ── Batch-Sweep ────────────────────────────────────────────────────────────
# Die eigentliche Stellschraube fuer die SM-Auslastung: batch_size bestimmt,
# wieviele Liganden Uni-Dock in einen gemeinsamen CUDA-Launch packt. Der
# Standardwert 1000 ist gesetzt, nicht gemessen.
#
# REIHENFOLGE: erst Batch-Sweep auf EINER Karte, dann die Skalierungsmatrix
# mit dem gefundenen Batch. Andersherum misst man die Skalierung eines
# falsch parametrierten Systems – und die sieht sogar gut aus, weil eine
# unterausgelastete Karte fast linear skaliert.
#
# Als EIGENE Messreihe fahren, nicht in dieselbe Matrix mischen: die
# GPU-Skalierung beantwortet "wie gut verteilt der Orchestrator", der
# Batch-Sweep "wie gut saettigt die Engine". Dazu oben RUNS auskommentieren
# und stattdessen:
# RUNS=("1:balance:250" "1:balance:500" "1:balance:1000"
#       "1:balance:2000" "1:balance:4000" "1:balance:8000")
# Auf H100 (80 GB) ruhig bis 16000/32000 verlaengern, bis der Durchsatz
# plateaut oder OOM kommt. Beides ist ein Ergebnis.


# ════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

BENCH_SIF="unidock-gpu.sif"
BENCH_INI_TEMPLATE="config/docking.ini"   # Basis; GPU/Modus/Batch werden ersetzt
BENCH_OUT="${BENCH_OUT:-benchmark}"       # Ergebnisverzeichnis (Logs + CSV)

# Ligandenmenge pro Lauf. 0 = alle aus pdbqt_dir verwenden.
# Ein fester Ausschnitt macht die Laeufe vergleichbar UND kurz genug, dass
# die ganze Matrix in vertretbarer Zeit durchlaeuft. Als Richtwert: bei
# 47.000 Liganden/h auf einer Karte dauert ein 1-GPU-Lauf mit 20.000
# Liganden gut 25 Minuten – ein detail-Lauf entsprechend laenger.
#
# Bevor du die ganze Matrix startest: einmal mit 2000 durchlaufen lassen.
# Das dauert wenige Minuten und zeigt, ob Auswertung, GPU-Telemetrie und
# CSV stimmen. Danach hochsetzen und die eigentliche Messung fahren.
#
# AUF SCHNELLER HARDWARE NEU KALIBRIEREN. t0 wird vor dem Python-Start
# gesetzt, d.h. Container-Start, GPU-Init und Ligandenscan stecken in
# wall_s. Sockel einmalig messen:
#     BENCH_OUT=bench_overhead BENCH_N_LIGANDS=100 BENCH_WARMUP=false \
#     BENCH_RUNS="4:fast:1000" ./benchmark.sh
# Danach den Messsatz so waehlen, dass der SCHNELLSTE Lauf der Matrix
# mindestens das 10- bis 20-fache dieses Sockels dauert.
BENCH_N_LIGANDS="${BENCH_N_LIGANDS:-15000}"

# Ein Aufwaermlauf vor der Messung. Der erste Lauf zahlt Page-Cache,
# GPU-Initialisierung und Container-Start; ohne Aufwaermen sieht der
# erste Matrixeintrag systematisch schlechter aus als er ist.
# Er laeuft mit der maximalen GPU-Zahl der Matrix (siehe unten), sonst
# zahlt der erste Mehr-GPU-Lauf die Init der uebrigen Karten.
BENCH_WARMUP="${BENCH_WARMUP:-true}"
BENCH_WARMUP_LIGANDS=2000

# Abtastintervall fuer nvidia-smi in Sekunden. 0 = keine GPU-Telemetrie.
BENCH_SAMPLE_INTERVAL=5

# Welche GPUs nvidia-smi abtastet. Leer = automatisch 0..(gpus-1) je Lauf.
# Das ist richtig, solange der Orchestrator bei num_gpus=N die Karten 0..N-1
# nimmt. Wird per CUDA_VISIBLE_DEVICES eine andere Auswahl erzwungen (etwa
# CUDA_VISIBLE_DEVICES=1 auf einer Workstation mit ungleichen Karten), muss
# hier die PHYSISCHE Index-Liste stehen, die nvidia-smi sieht:
#     BENCH_SMI_IDS=1 CUDA_VISIBLE_DEVICES=1 ./benchmark.sh
BENCH_SMI_IDS="${BENCH_SMI_IDS:-}"

# Hinweis zur Interpretation: utilization.gpu ist der Zeitanteil, in dem
# IRGENDEIN Kernel resident war – nicht die SM-Belegung. Auf H100 steht dort
# 100 %, auch wenn die SMs zu 15 % ausgelastet sind. Als Saettigungsindikator
# taugt die Spalte nicht; dafuer ist allein die Durchsatzkurve ueber
# batch_size aussagekraeftig (bzw. DCGM_FI_PROF_SM_OCCUPANCY, falls da).

# Seed fuer reproduzierbare Laeufe. 0 = zufaellig (nicht empfohlen fuer
# eine Messreihe, weil Sampling-Unterschiede in den Durchsatz eingehen).
BENCH_SEED="${BENCH_SEED:-42}"


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

# Maximale GPU-Zahl der Matrix (fuer den Aufwaermlauf)
BENCH_MAX_GPUS=1
for spec in "${RUNS[@]}"; do
    g="${spec%%:*}"
    case "$g" in
        ''|*[!0-9]*) continue ;;
    esac
    [ "$g" -gt "$BENCH_MAX_GPUS" ] && BENCH_MAX_GPUS="$g"
done

banner "DOCKING BENCHMARK"
say "Projekt   : $PROJECT_DIR"
say "Liganden  : $PROJECT_DIR/$PDBQT_DIR"
say "Laeufe    : ${#RUNS[@]}  (max ${BENCH_MAX_GPUS} GPU)"
[ -n "$BENCH_SMI_IDS" ] && say "smi-IDs   : $BENCH_SMI_IDS (fest)"
[ "$DRY_RUN" = true ] && warn "DRY-RUN"

for f in "$BENCH_SIF" "$BENCH_INI_TEMPLATE" "src/orchestrator.py" "TARGET/config.txt"; do
    [ -e "$PROJECT_DIR/$f" ] || { err "Fehlt: $f"; exit 1; }
done

# Fremde Last auf dem Knoten verfaelscht sowohl Durchsatz als auch Telemetrie
if [ "$DRY_RUN" != true ] && command -v nvidia-smi >/dev/null 2>&1; then
    foreign=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$foreign" -gt 0 ] && warn "$foreign fremde Compute-Prozesse auf den GPUs – Knoten nicht exklusiv?"
fi

mkdir -p "$PROJECT_DIR/$BENCH_OUT"
SUMMARY="$PROJECT_DIR/$BENCH_OUT/summary.csv"

# ── Ligandenausschnitt: fester Satz fuer alle Laeufe ────────────────────────
# Symlinks statt Kopien – die Bibliothek kann Millionen Dateien umfassen.
# Auf Lustre/GPFS ist das ein Risiko: vier GPUs ziehen dann hunderttausende
# kleine PDBQT ueber das Netzwerk-FS, und gemessen wird die Metadaten-
# Performance, nicht die Engine. Gegenprobe: Messsatz nach $TMPDIR (node-
# lokales NVMe) kopieren und einen Lauf wiederholen. Dafuer 'ln -s' unten
# durch 'cp' ersetzen.
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
    # -L: ohne das folgt find dem Verzeichnis-Symlink im n=0-Fall nicht und
    # liefert 0. Bei n=0 und Millionen Dateien dauert das Zaehlen entsprechend.
    find -L "$dst" -name '*.pdbqt' 2>/dev/null | wc -l
}

# ── GPU-Telemetrie ─────────────────────────────────────────────────────────
start_sampling() {   # $1 = Ausgabedatei, $2 = GPU-Indexliste (leer = alle)
    [ "$BENCH_SAMPLE_INTERVAL" -eq 0 ] && return 0
    local ids="${2:-}"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
               --format=csv,noheader,nounits \
               ${ids:+-i "$ids"} \
               -l "$BENCH_SAMPLE_INTERVAL" > "$1" 2>"${1%.csv}.err" &
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

    # Nur die tatsaechlich genutzten Karten abtasten. Ohne diese Einschraenkung
    # mitteln idle Karten die Auslastung herunter (1 von 4 GPUs -> ~25 %), was
    # in der Ergebnistabelle wie ein Skalierungsbefund aussieht und keiner ist.
    local ids="$BENCH_SMI_IDS"
    if [ -z "$ids" ]; then
        ids=$(seq -s, 0 $(( gpus - 1 )))
    fi

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
        printf '   (dry-run) nvidia-smi -i %s\n' "$ids"
        return 0
    fi

    # RESULTS leeren: sonst zaehlen Posen und Chunk-CSVs des Vorlaufs mit,
    # und die Zeit fuer bereits vorhandene Ergebnisse faellt weg.
    rm -rf "$PROJECT_DIR/$RESULTS_DIR"
    mkdir -p "$PROJECT_DIR/$RESULTS_DIR"
    rm -f "$PROJECT_DIR/$LOG_DIR/pipeline.log"

    local smi_pid; smi_pid=$(start_sampling "$smi" "$ids")
    local t0; t0=$(date +%s)

    python3 "$PROJECT_DIR/src/orchestrator.py" \
        --project "$PROJECT_DIR" \
        --config  "${ini#$PROJECT_DIR/}" \
        --stage   dock \
        --sif     "$BENCH_SIF" > "$log" 2>&1
    local rc=$?

    local dt=$(( $(date +%s) - t0 ))
    stop_sampling "$smi_pid"

    # Telemetrie still gescheitert? Dann sind util/power/mem leer, und man
    # soll wissen warum, statt Leerzeichen in der CSV zu interpretieren.
    if [ "$BENCH_SAMPLE_INTERVAL" -ne 0 ] && [ ! -s "$smi" ]; then
        warn "Keine GPU-Telemetrie (siehe ${smi%.csv}.err)"
    fi

    # Erfolgreiche Liganden aus den Chunk-CSVs zaehlen (nicht aus dem Log:
    # der Durchsatz soll auf tatsaechlich gedockten Molekuelen beruhen).
    # FNR statt NR: NR ueberspringt nur den Header der ERSTEN Datei.
    local n_ok
    n_ok=$(find "$PROJECT_DIR/$RESULTS_DIR" -name 'docking_results_*.csv' \
           -exec awk -F, 'FNR>1 && $2=="True"' {} + 2>/dev/null | wc -l)

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
    say "$n Liganden auf ${BENCH_MAX_GPUS} GPU"
    SUMMARY_BAK="$SUMMARY"; SUMMARY="/dev/null"
    run_one 0 "$BENCH_MAX_GPUS" fast 1000 "$n"
    SUMMARY="$SUMMARY_BAK"
fi

# ── Messreihe ──────────────────────────────────────────────────────────────
echo "run,gpus,mode,batch_size,chunk_size,ligands_in,ligands_ok,wall_s,ligands_per_h,gpu_util_pct,power_w,mem_mib_max,exit" \
    > "$SUMMARY"

N_LIG=0
if [ "$DRY_RUN" != true ]; then
    N_LIG=$(prepare_ligands "$BENCH_N_LIGANDS" "$BENCH_LIG_DIR")
    say "Messsatz: $N_LIG Liganden (identisch fuer alle Laeufe)"
    [ "$N_LIG" -eq 0 ] && { err "Keine Liganden gefunden – pdbqt_dir pruefen"; exit 1; }
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

# Kopf ausserhalb von awk, sonst sortiert 'sort' die Trennlinie ueber die
# Ueberschrift.
printf "  %-4s %-5s %-9s %-7s %10s %8s %9s %8s %7s %7s\n" \
       "Run" "GPUs" "Modus" "Batch" "Liganden" "Zeit_s" "L/h" "Speedup" "Eff_%" "GPU_%"
printf "  %s\n" "---------------------------------------------------------------------------------------"

awk -F, 'NR==1 {next}
{
    key = $3 "_" $4
    if (!( key in base ) || $2+0 < basegpu[key]) { base[key]=$9; basegpu[key]=$2+0 }
    rows[NR]=$0
}
END {
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
echo
echo "  Vorsicht bei ungleichen Karten (Workstation): dort sind Speedup und"
echo "  Effizienz bedeutungslos, weil die schnellere Karte auf die langsamere"
echo "  wartet. Nur Batch-Sweep und Einzelkarten-Durchsatz sind dort gueltig."
