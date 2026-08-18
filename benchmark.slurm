#!/bin/bash
# benchmark.slurm — Messreihe Docking-Durchsatz (4x H100)
#
#   sbatch benchmark.slurm
#
# Faehrt BEIDE Messreihen nacheinander in einem Job:
#
#   Reihe 1  Batch-Sweep auf EINER Karte  -> "wie gut saettigt die Engine"
#   Reihe 2  GPU-Skalierung               -> "wie gut verteilt der Orchestrator"
#
# Die Reihenfolge ist nicht beliebig: Reihe 2 laeuft mit dem batch_size, den
# Reihe 1 gewonnen hat. Umgekehrt wuerde man die Skalierung eines falsch
# parametrierten Systems messen – und die sieht sogar gut aus, weil eine
# unterausgelastete Karte fast linear skaliert.
#
# Den Gewinner liest der Job selbst aus bench_batch/summary.csv. Manuell
# vorgeben (Reihe 1 laeuft dann trotzdem, nur ohne Einfluss):
#     sbatch --export=ALL,SCALE_BATCH=4000 benchmark.slurm
#
# Nur eine Reihe fahren:
#     sbatch --export=ALL,RUN_SCALE=false benchmark.slurm
#     sbatch --export=ALL,RUN_SWEEP=false,SCALE_BATCH=4000 benchmark.slurm

#SBATCH --job-name=kdc-bench
#SBATCH -N 1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --threads-per-core=1
#SBATCH --time=23:30:00
#SBATCH --mail-user=thomas.roessel@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/bench-%j.out

# Vier GPUs anfordern, auch fuer die 1- und 2-GPU-Laeufe: der Orchestrator
# begrenzt sich intern ueber num_gpus. Nur so werden alle Konfigurationen
# auf identischer Hardware gemessen.
#
# Zur Walltime: Reihe 1 dauert rund 1,8 h, Reihe 2 rund 5 h (hochgerechnet
# aus 36.120 L/h je Karte bei balance, gemessen in Job 83248). Zusammen
# ca. 7 h – 23:30 laesst Raum, falls der gewonnene Batch die Laeufe
# deutlich verkuerzt oder verlaengert.

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "$PROJECT_DIR/benchmark.sh" ]; then
    echo "FEHLER: $PROJECT_DIR ist nicht die Projektwurzel." >&2
    exit 2
fi
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

# Beim Herunterladen oder Kopieren geht das x-Bit leicht verloren. Das faellt
# sonst erst auf, wenn die GPUs schon allokiert sind (Exit 126).
if [ ! -x ./benchmark.sh ]; then
    chmod +x ./benchmark.sh 2>/dev/null
    if [ ! -x ./benchmark.sh ]; then
        echo "FEHLER: ./benchmark.sh ist nicht ausfuehrbar und chmod schlug fehl." >&2
        exit 2
    fi
    echo "HINWEIS: x-Bit auf benchmark.sh fehlte, wurde gesetzt."
fi

log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
rule() { printf '\n════════════════════════════════════════════════════════════\n  %s\n════════════════════════════════════════════════════════════\n\n' "$*"; }

# ════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

RUN_SWEEP="${RUN_SWEEP:-true}"
RUN_SCALE="${RUN_SCALE:-true}"

# ── Reihe 1: Batch-Sweep ───────────────────────────────────────────────────
# 16.000 teilt sich glatt in 16/8/4/2 Chunks, es bleibt kein Rest bei
# geringerer Saettigung uebrig. Eine Karte genuegt: Saettigung misst man
# nicht mit vieren. Dauer rund 27 min je Punkt.
SWEEP_OUT="${SWEEP_OUT:-bench_batch}"
SWEEP_N="${SWEEP_N:-16000}"
SWEEP_RUNS="${SWEEP_RUNS:-1:balance:1000:1000 1:balance:2000:2000 1:balance:4000:4000 1:balance:8000:8000}"

# ── Reihe 2: GPU-Skalierung ────────────────────────────────────────────────
# batch_size kommt aus Reihe 1, chunk_size wird gleich batch_size gesetzt.
SCALE_OUT="${SCALE_OUT:-bench_scale}"
SCALE_N="${SCALE_N:-100000}"
SCALE_GPUS="${SCALE_GPUS:-1 2 4}"
SCALE_MODE="${SCALE_MODE:-balance}"
SCALE_BATCH="${SCALE_BATCH:-}"      # leer = aus Reihe 1 ermitteln

# Wieviel Prozent Durchsatz man bereit ist aufzugeben, um den KLEINEREN
# Batch zu nehmen. Verhindert, dass fuer 1 % Mehrleistung ein Batch gewaehlt
# wird, der im Produktionslauf am Speicher scheitert.
PLATEAU_TOL="${PLATEAU_TOL:-2}"

# Mindestzahl Chunks je GPU, sonst stehen am Ende Karten leer.
BENCH_MIN_CHUNKS_PER_GPU="${BENCH_MIN_CHUNKS_PER_GPU:-4}"

export BENCH_SEED="${BENCH_SEED:-42}"

# ════════════════════════════════════════════════════════════════════════════
#  AB HIER: KEINE KONFIGURATION MEHR
# ════════════════════════════════════════════════════════════════════════════

log "Benchmark — Job ${SLURM_JOB_ID:-?} auf $(hostname -s)"
log "Partition ${SLURM_JOB_PARTITION:-?} | QoS ${SLURM_JOB_QOS:-?} | GPUs ${SLURM_GPUS_ON_NODE:-?}"

FOREIGN=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$FOREIGN" -gt 0 ]; then
    log "WARNUNG: $FOREIGN fremde Compute-Prozesse auf den GPUs."
    log "         Durchsatz UND Telemetrie sind dann nicht belastbar."
fi

# ── Umgebung protokollieren ────────────────────────────────────────────────
ENVFILE="logs/bench-${SLURM_JOB_ID:-manual}-environment.txt"
{
    echo "=== Knoten ==="
    hostname -f
    echo "Partition: ${SLURM_JOB_PARTITION:-?}  QoS: ${SLURM_JOB_QOS:-?}"
    echo
    echo "=== CPU ==="
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)"
    echo
    echo "=== GPUs ==="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null
    echo
    echo "=== Taktgrenzen und Drosselung (vor der Messung) ==="
    # Wird waehrend der Messreihe 'SW Thermal Slowdown' oder 'HW Power Brake'
    # aktiv, sind alle L/h-Werte nach unten verzerrt. Hier der Ausgangszustand
    # zum Vergleich mit dem Nachher-Block am Jobende.
    nvidia-smi --query-gpu=index,clocks.max.sm,clocks.sm,temperature.gpu,power.limit \
               --format=csv 2>/dev/null
    nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -A8 "Clocks Event Reasons" | head -40
    echo
    echo "=== Topologie ==="
    nvidia-smi topo -m 2>/dev/null
    echo
    echo "=== Ablage ==="
    df -h . 2>/dev/null | tail -1
    echo
    echo "=== Container ==="
    apptainer --version
    for sif in unidock-gpu.sif rescoring-gpu.sif sdf_to_pdbqt.sif; do
        [ -f "$sif" ] && printf '%-22s %s\n' "$sif" "$(stat -c '%y  %s bytes' "$sif" | cut -d. -f1)"
    done
    echo
    echo "=== Uni-Dock ==="
    apptainer exec --nv unidock-gpu.sif bash -lc \
        'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --version' 2>&1 | head -5
    echo
    echo "=== Git ==="
    git rev-parse HEAD 2>/dev/null
    git status --short 2>/dev/null | head -20
} > "$ENVFILE" 2>&1
log "Umgebung protokolliert: $ENVFILE"

# ── Laeuft Uni-Dock auf diesem Knoten? ─────────────────────────────────────
if ! apptainer exec --nv unidock-gpu.sif bash -lc \
     'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --help' \
     > /dev/null 2>&1; then
    log "FEHLER: unidock laeuft auf diesem Knoten nicht (Compute Capability?)."
    exit 1
fi

# ── Chunk-Granularitaet pruefen ────────────────────────────────────────────
# Der teuerste Fehler der letzten Messreihe: 15.000 Liganden bei chunk_size
# 5000 ergeben 3 Chunks. Auf 4 GPUs bleibt eine Karte den ganzen Lauf leer,
# und die Effizienz deckelt bei 75 % — unabhaengig davon, wie gut der
# Orchestrator wirklich verteilt.
#
# Der Dry-Run kennt die tatsaechliche Matrix, deshalb wird sie hier
# ausgelesen statt nachgebaut.
check_granularity() {
    local out lines
    out=$(./benchmark.sh --dry-run 2>&1 | sed 's/\x1b\[[0-9;]*m//g')

    # Ohne diese Pruefung meldet die Funktion "in Ordnung", wenn der Dry-Run
    # gar nicht erst laeuft (fehlendes x-Bit, fehlende Datei, Abbruch in der
    # Vorpruefung von benchmark.sh): awk bekommt dann keine Zeile, END setzt
    # kein bad, und der Exit-Code ist 0. Genau so ist Job 86171 mit
    # "Granularitaet in Ordnung" in ein 'Permission denied' gelaufen.
    lines=$(printf '%s\n' "$out" | grep -cE '^[[:space:]]*LAUF')
    if [ "$lines" -eq 0 ]; then
        log "FEHLER: Dry-Run lieferte keine Laufliste. Ausgabe von benchmark.sh:"
        printf '%s\n' "$out" | head -20 | sed 's/^/      /'
        return 2
    fi

    printf '%s\n' "$out" | grep -E '^[[:space:]]*LAUF' \
    | awk -v n="$BENCH_N_LIGANDS" -v ratio="$BENCH_MIN_CHUNKS_PER_GPU" '
    {
        for (i = 1; i <= NF; i++) {
            if ($i == "GPU")   g = $(i-1)
            if ($i == "batch") b = $(i+1)
            if ($i == "chunk") c = $(i+1)
        }
        if (n <= 0 || c <= 0) {
            printf "  Lauf %-3s %s GPU  batch %-6s chunk %-6s  (Messsatz unbekannt)\n", $2, g, b, c
            next
        }
        chunks = int((n + c - 1) / c)
        rounds = int((chunks + g - 1) / g)
        sp = chunks / rounds
        # Nur bei mehreren GPUs relevant: auf einer Karte laufen die Chunks
        # ohnehin nacheinander, die Granularitaet aendert nichts am Ergebnis.
        coarse = (g > 1 && chunks < ratio * g)
        if (coarse) bad++
        printf "  Lauf %-3s %s GPU  batch %-6s chunk %-6s -> %3d Chunks | max %.2fx (%3.0f%%)%s\n",
               $2, g, b, c, chunks, sp, 100*sp/g, (coarse ? "   << zu grob" : "")
    }
    END { exit (bad > 0 ? 1 : 0) }'
}

# ── Eine Messreihe fahren ──────────────────────────────────────────────────
run_series() {   # $1 = BENCH_OUT, $2 = Ligandenzahl, $3 = RUNS
    export BENCH_OUT="$1" BENCH_N_LIGANDS="$2" BENCH_RUNS="$3"

    log "Messreihe : $BENCH_OUT | $BENCH_N_LIGANDS Liganden"
    log "Laeufe    : $BENCH_RUNS"

    if [ -f "$BENCH_OUT/summary.csv" ] && [ "${BENCH_FORCE:-0}" != "1" ]; then
        log "FEHLER: $BENCH_OUT/summary.csv existiert bereits."
        log "        Verzeichnis aendern oder mit BENCH_FORCE=1 ueberschreiben."
        return 2
    fi

    log "Pruefe Chunk-Granularitaet ..."
    check_granularity
    local gran=$?
    if [ "$gran" -eq 2 ]; then
        log "        Der Dry-Run selbst ist gescheitert — nicht die Granularitaet."
        log "        Haeufig: benchmark.sh nicht ausfuehrbar, fehlende .sif-Datei"
        log "        oder fehlendes TARGET/config.txt."
        return 4
    elif [ "$gran" -ne 0 ]; then
        log "FEHLER: Messsatz zu klein fuer die Zahl der GPUs."
        log "        Abhilfe: Ligandenzahl erhoehen (>= $BENCH_MIN_CHUNKS_PER_GPU Chunks je GPU)"
        log "        oder chunk_size im 4. Feld verkleinern."
        log "        Bewusst trotzdem messen: BENCH_MIN_CHUNKS_PER_GPU=1"
        return 3
    fi
    log "Granularitaet in Ordnung."

    ./benchmark.sh
    local rc=$?
    log "Messreihe $BENCH_OUT beendet, Exit $rc"
    return $rc
}

# ── Gewinner-Batch aus einer summary.csv lesen ─────────────────────────────
# Nicht einfach das Maximum: liegen mehrere Batches innerhalb von
# PLATEAU_TOL Prozent, gewinnt der KLEINSTE. Der spart Speicher, ohne
# nennenswert Durchsatz zu kosten. Gescheiterte Laeufe (etwa OOM bei zu
# grossem Batch) fallen ueber die exit-Spalte heraus.
pick_batch() {   # $1 = summary.csv
    awk -F, -v tolpct="$PLATEAU_TOL" '
        NR == 1     { next }
        $13+0 != 0  { next }
        $9+0  <= 0  { next }
        { lph[$4] = $9+0; if ($9+0 > max) max = $9+0 }
        END {
            if (max <= 0) exit 1
            tol = max * (1 - tolpct/100)
            best = ""
            for (b in lph)
                if (lph[b] >= tol)
                    if (best == "" || b+0 < best+0) best = b
            print best
        }' "$1"
}

show_summary() {   # $1 = Verzeichnis
    local f="$1/summary.csv"
    [ -f "$f" ] || return 0
    log "Ergebnis ($f):"
    column -t -s, "$f" 2>/dev/null || cat "$f"
}

# ════════════════════════════════════════════════════════════════════════════
#  REIHE 1 — BATCH-SWEEP
# ════════════════════════════════════════════════════════════════════════════
SWEEP_RC=0
if [ "$RUN_SWEEP" = "true" ]; then
    rule "REIHE 1 — BATCH-SWEEP (1 GPU)"
    run_series "$SWEEP_OUT" "$SWEEP_N" "$SWEEP_RUNS"
    SWEEP_RC=$?
    show_summary "$SWEEP_OUT"
else
    log "Reihe 1 uebersprungen (RUN_SWEEP=false)."
fi

# ════════════════════════════════════════════════════════════════════════════
#  BATCH FUER REIHE 2 BESTIMMEN
# ════════════════════════════════════════════════════════════════════════════
if [ "$RUN_SCALE" = "true" ] && [ -z "$SCALE_BATCH" ]; then
    if [ -f "$SWEEP_OUT/summary.csv" ]; then
        SCALE_BATCH=$(pick_batch "$SWEEP_OUT/summary.csv")
        if [ -n "$SCALE_BATCH" ]; then
            log "Gewaehlter batch_size fuer Reihe 2: $SCALE_BATCH"
            log "  (kleinster Batch innerhalb von ${PLATEAU_TOL} % des Maximums)"
        else
            log "FEHLER: kein brauchbarer Lauf in $SWEEP_OUT/summary.csv."
            log "        Reihe 2 wird uebersprungen. Manuell: SCALE_BATCH=<n>"
            RUN_SCALE=false
        fi
    else
        log "FEHLER: $SWEEP_OUT/summary.csv fehlt — Reihe 2 wird uebersprungen."
        RUN_SCALE=false
    fi
fi

# ════════════════════════════════════════════════════════════════════════════
#  REIHE 2 — GPU-SKALIERUNG
# ════════════════════════════════════════════════════════════════════════════
SCALE_RC=0
if [ "$RUN_SCALE" = "true" ]; then
    SCALE_RUNS=""
    for g in $SCALE_GPUS; do
        SCALE_RUNS="$SCALE_RUNS $g:$SCALE_MODE:$SCALE_BATCH:$SCALE_BATCH"
    done
    SCALE_RUNS="${SCALE_RUNS# }"

    rule "REIHE 2 — GPU-SKALIERUNG (batch $SCALE_BATCH)"
    run_series "$SCALE_OUT" "$SCALE_N" "$SCALE_RUNS"
    SCALE_RC=$?
    show_summary "$SCALE_OUT"
else
    log "Reihe 2 uebersprungen."
fi

# ── Drosselung nachher ─────────────────────────────────────────────────────
{
    echo
    echo "=== Taktgrenzen und Drosselung (nach der Messung) ==="
    nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv 2>/dev/null
    nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -A8 "Clocks Event Reasons" | head -40
} >> "$ENVFILE" 2>&1

rule "FERTIG"
log "Reihe 1 (Batch-Sweep) : Exit $SWEEP_RC  -> $SWEEP_OUT/summary.csv"
log "Reihe 2 (Skalierung)  : Exit $SCALE_RC  -> $SCALE_OUT/summary.csv"
[ -n "$SCALE_BATCH" ] && log "Verwendeter batch_size : $SCALE_BATCH"
echo
echo "  Zur Interpretation:"
echo "  - gpu_util_pct ist der Zeitanteil mit aktivem Kernel, NICHT die"
echo "    SM-Belegung. Saettigung zeigt allein die Durchsatzkurve ueber"
echo "    batch_size in Reihe 1."
echo "  - mem_mib_max gegen die 80 GB halten: bleibt viel frei, ist"
echo "    batch_size noch nicht ausgereizt."
echo "  - power_w deutlich unter 350 W je aktiver Karte heisst ebenfalls:"
echo "    da geht mehr."

[ "$SWEEP_RC" -ne 0 ] && exit "$SWEEP_RC"
exit "$SCALE_RC"
