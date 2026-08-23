#!/bin/bash
# archive_run.sh — abgeschlossenen Lauf archivieren und aufraeumen.
#
#   nohup ./src/archive_run.sh dock-76523 > ~/archive.log 2>&1 &
#   nohup ./src/archive_run.sh 76523 --purge > ~/archive.log 2>&1 &
#   ./src/archive_run.sh dock-76523 --no-poses --dry-run
#
# Baut <jobname>.tar.gz mit:
#   <jobname>/TARGET/*.pdbqt, config.txt
#   <jobname>/LOG/                      (aus data/LOG)
#   <jobname>/slurm/                    (die .out-Dateien des Jobs)
#   <jobname>/RESULTS/                  (Posen + CSVs, oder nur CSVs)
#   <jobname>/rescoring_ligands_*.csv   (je Target, direkt greifbar)
#   <jobname>/Top250_*.csv              (je Target, aus dem Ranking)
#   <jobname>/MANIFEST.txt              (was drin ist, mit Pruefsummen)
#
# Mit --purge werden RESULTS/ und data/LOG/ danach verschoben und im
# Hintergrund geloescht – aber NUR, wenn das Archiv verifiziert wurde.

set -uo pipefail

# ── Argumente ─────────────────────────────────────────────────────────
JOB=""
PURGE=false
DRY=false
WITH_POSES=true
ALL_LOGS=false
TOP_N=250
OUT_DIR=""

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Optionen:
  --purge         RESULTS/ und data/LOG/ nach der Verifikation loeschen
  --no-poses      nur CSVs aus RESULTS/, keine _docked.pdbqt
  --top N         Groesse der Top-Liste (default 250)
  --all-logs      auch Konvertierungs-Logs aus Stufe 1 mitnehmen
  --out DIR       Zielverzeichnis fuer das Archiv (default ./archive)
  --dry-run       nur zeigen, was passieren wuerde
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --purge)    PURGE=true ;;
        --no-poses) WITH_POSES=false ;;
        --all-logs) ALL_LOGS=true ;;
        --top)      TOP_N="$2"; shift ;;
        --out)      OUT_DIR="$2"; shift ;;
        --dry-run)  DRY=true ;;
        -h|--help)  usage; exit 0 ;;
        -*)         echo "Unbekannte Option: $1" >&2; exit 2 ;;
        *)          JOB="$1" ;;
    esac
    shift
done

[ -n "$JOB" ] || { usage; exit 2; }

# ── Projektwurzel ─────────────────────────────────────────────────────
PROJECT="${PROJECT_DIR:-$(pwd)}"
[ -f "$PROJECT/pipeline_start.sh" ] || {
    echo "FEHLER: $PROJECT ist nicht die Projektwurzel." >&2; exit 2; }
cd "$PROJECT" || exit 1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { printf '[%s] FEHLER: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 1; }

# ── Jobnamen aufloesen ────────────────────────────────────────────────
# Akzeptiert "dock-76523", "76523" und "logs/dock-76523.out".
JOB="$(basename "$JOB")"
JOB="${JOB%.out}"
if [[ "$JOB" =~ ^[0-9]+$ ]]; then
    match=$(find logs -maxdepth 1 -name "*-${JOB}.out" 2>/dev/null | head -1)
    [ -n "$match" ] || die "Kein Log zu Job-ID $JOB unter logs/ gefunden."
    JOB="$(basename "$match" .out)"
fi
JOBID="${JOB##*-}"

log "Job        : $JOB (ID $JOBID)"

OUT_DIR="${OUT_DIR:-$PROJECT/archive}"
ARCHIVE="$OUT_DIR/${JOB}.tar.gz"
STAGE="$OUT_DIR/.stage_${JOB}"

[ -e "$ARCHIVE" ] && die "$ARCHIVE existiert bereits."

# Sperre gegen Parallellaeufe: sie wuerden in dasselbe Staging-
# Verzeichnis und dasselbe Archiv schreiben. mkdir ist atomar.
mkdir -p "$OUT_DIR" 2>/dev/null
LOCK="$OUT_DIR/.lock_${JOB}"
if ! mkdir "$LOCK" 2>/dev/null; then
    die "Ein Lauf fuer $JOB ist bereits aktiv ($LOCK). Falls das ein Rest
       eines abgebrochenen Laufs ist: rmdir '$LOCK'"
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# ── Bestandsaufnahme ──────────────────────────────────────────────────
[ -d RESULTS ] || die "RESULTS/ fehlt."
[ -d TARGET ]  || die "TARGET/ fehlt."

mapfile -t TARGETS < <(find RESULTS -mindepth 1 -maxdepth 1 -type d \
                       -not -name '_probe_*' -printf '%f\n' | sort)
[ ${#TARGETS[@]} -gt 0 ] || die "Keine Targets in RESULTS/."

log "Targets    : ${TARGETS[*]}"

# Zaehlen kostet bei Millionen Dateien Minuten. Nur erheben, wenn die
# Zahl gebraucht wird.
if [ "$WITH_POSES" = true ]; then
    log "Zaehle Posen (dauert bei Millionen Dateien einige Minuten) ..."
    N_POSEN=$(find RESULTS -name '*_docked.pdbqt' | wc -l)
    SZ_RESULTS=$(du -sh RESULTS 2>/dev/null | cut -f1)
    log "RESULTS    : $SZ_RESULTS ($N_POSEN Posendateien)"
    if [ "$N_POSEN" -gt 500000 ]; then
        log "HINWEIS: $N_POSEN Posendateien werden mitarchiviert. Das dauert"
        log "         lange und erzeugt ein sehr grosses Archiv. Mit --no-poses"
        log "         landen nur die CSVs im Tarball."
    fi
else
    N_POSEN=0
    log "RESULTS    : nur CSVs (--no-poses), Posen werden nicht gezaehlt"
fi

# Kompressor: pigz nutzt alle Kerne, gzip nur einen.
# Threads: was Slurm zugeteilt hat, nicht was die Maschine hat. pigz
# wuerde sonst alle Kerne der Node belegen, auch fremde.
THREADS="${ARCHIVE_THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}}}"
if command -v pigz >/dev/null 2>&1; then
    COMPRESS="pigz -p $THREADS"
    log "Kompressor : pigz mit $THREADS Threads"
else
    COMPRESS="gzip"
    log "Kompressor : gzip (einfaedig – pigz waere deutlich schneller)"
fi

if [ "$DRY" = true ]; then
    log "--dry-run: hier waere Schluss. Archiv waere $ARCHIVE"
    exit 0
fi

# ── Staging: die kleinen Dinge ────────────────────────────────────────
rm -rf "$STAGE"
mkdir -p "$STAGE/$JOB"/{TARGET,slurm}

log "── Sammle Metadaten ──"

if [ -f TARGET/config.txt ]; then
    cp -a TARGET/config.txt "$STAGE/$JOB/TARGET/" \
        || die "TARGET/config.txt nicht kopierbar."
else
    log "WARNUNG: TARGET/config.txt fehlt."
fi
for t in "${TARGETS[@]}"; do
    if [ -f "TARGET/${t}.pdbqt" ]; then
        cp -a "TARGET/${t}.pdbqt" "$STAGE/$JOB/TARGET/"
    else
        log "WARNUNG: TARGET/${t}.pdbqt fehlt."
    fi
done

# Slurm-Logs: das benannte plus alles mit derselben ID.
found_logs=0
for f in logs/*"${JOBID}"*.out logs/*"${JOBID}"*.err; do
    [ -f "$f" ] && { cp -a "$f" "$STAGE/$JOB/slurm/"; found_logs=$((found_logs+1)); }
done
log "Slurm-Logs : $found_logs Datei(en)"

# Nicht kopieren, sondern spaeter direkt in den Tarball schreiben.
# Stufe 1 hinterlaesst pro fehlgeschlagenem Molekuel eine
# *_convert_error.log; die gehoeren nicht zu diesem Lauf.
LOG_EXCLUDE=()
if [ -d data/LOG ]; then
    n_all=$(find data/LOG -type f | wc -l)
    n_conv=$(find data/LOG -type f \( -name '*_convert_error.log' \
                                     -o -name 'conversion.log' \) | wc -l)
    if [ "$ALL_LOGS" = true ]; then
        log "Worker-Logs: $n_all Datei(en), inkl. $n_conv aus der Konvertierung"
    else
        LOG_EXCLUDE=(--exclude=*_convert_error.log --exclude=conversion.log)
        log "Worker-Logs: $(( n_all - n_conv )) Datei(en)"
        [ "$n_conv" -gt 0 ] && \
            log "             $n_conv Konvertierungs-Logs ausgelassen (--all-logs nimmt sie mit)"
    fi
else
    log "WARNUNG: data/LOG fehlt."
fi

# ── Ranking-CSVs und Top-Listen ───────────────────────────────────────
log "── Ranking ──"
n_rank=0
for t in "${TARGETS[@]}"; do
    src="RESULTS/$t/rescoring_ligands_${t}.csv"
    if [ ! -s "$src" ]; then
        log "Kein Ranking fuer '$t' – Rescoring noch nicht gelaufen."
        continue
    fi
    cp -a "$src" "$STAGE/$JOB/"
    n_rank=$((n_rank+1))

    # Explizit nach ecr_score sortieren, nicht auf ecr_rank verlassen.
    python3 - "$src" "$STAGE/$JOB/Top${TOP_N}_${t}.csv" "$TOP_N" <<'PY'
import csv, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    sys.exit(0)
def key(r):
    try:    return -float(r.get("ecr_score") or 0.0)
    except ValueError: return 0.0
rows.sort(key=key)
with open(dst, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows[:n])
print(f"  Top{n}: {min(n, len(rows))} von {len(rows):,} Liganden")
PY
done
if [ "$n_rank" -eq 0 ]; then
    log "Kein Rescoring vorhanden – Archiv enthaelt nur Docking-Ergebnisse."
else
    log "Rankings   : $n_rank Target(s)"
fi

# ── Manifest ──────────────────────────────────────────────────────────
{
    echo "Lauf:        $JOB"
    echo "Archiviert:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Projekt:     $PROJECT"
    echo "Host:        $(hostname -s)"
    echo "Targets:     ${TARGETS[*]}"
    echo "Posen:       $N_POSEN"
    echo "Posen im Archiv: $WITH_POSES"
    echo "Rankings:    $n_rank von ${#TARGETS[@]} Target(s)"
    [ "$n_rank" -eq 0 ] && echo "Hinweis:     Rescoring war zum Zeitpunkt der Archivierung nicht gelaufen."
    echo "Konvertierungs-Logs: $ALL_LOGS"
    echo
    echo "── Konfiguration ──"
    for ini in config/*.ini; do
        [ -f "$ini" ] && { echo "--- $ini ---"; grep -v '^\s*#' "$ini" | grep -v '^\s*$'; echo; }
    done
} > "$STAGE/$JOB/MANIFEST.txt"

mkdir -p "$STAGE/$JOB/config"
cp -a config/*.ini "$STAGE/$JOB/config/" 2>/dev/null || true

# ── Archiv bauen ──────────────────────────────────────────────────────
log "── Archiv bauen ──"
mkdir -p "$OUT_DIR"
START=$(date +%s)

# RESULTS wird nicht kopiert, sondern per --transform direkt aus dem
# Original unter <jobname>/RESULTS einsortiert.
if [ "$WITH_POSES" = true ]; then
    RESULTS_ARGS=(-C "$PROJECT" RESULTS)
else
    # Dateiliste statt Argumenten, sonst wird die Kommandozeile zu lang.
    find RESULTS -maxdepth 2 -name '*.csv' -not -path '*/.rescore_partial/*' \
        > "$STAGE/results_files.txt"
    log "Ohne Posen : $(wc -l < "$STAGE/results_files.txt") CSV-Datei(en)"
    RESULTS_ARGS=(-C "$PROJECT" -T "$STAGE/results_files.txt")
fi

LOG_ARGS=()
[ -d data/LOG ] && LOG_ARGS=(-C "$PROJECT/data" LOG)

tar --use-compress-program="$COMPRESS" \
    --transform "s|^RESULTS|$JOB/RESULTS|" \
    --transform "s|^LOG|$JOB/LOG|" \
    "${LOG_EXCLUDE[@]}" \
    -cf "$ARCHIVE" \
    -C "$STAGE" "$JOB" \
    "${LOG_ARGS[@]}" \
    "${RESULTS_ARGS[@]}" \
    || die "tar fehlgeschlagen."

DT=$(( $(date +%s) - START ))
SZ=$(du -sh "$ARCHIVE" | cut -f1)
log "Archiv     : $ARCHIVE ($SZ, ${DT}s)"

# ── Verifikation ──────────────────────────────────────────────────────
# Ohne diesen Schritt wuerde --purge Daten loeschen, deren Archiv
# moeglicherweise abgeschnitten ist (volle Platte, Timeout, Signal).
log "── Verifikation ──"
# Genau einmal auflisten: jeder Durchlauf entpackt das ganze Archiv.
LIST="$STAGE/archive_list.txt"
tar --use-compress-program="$COMPRESS" -tf "$ARCHIVE" > "$LIST" \
    || die "Archiv nicht lesbar – NICHTS wird geloescht."
N_ENTRIES=$(wc -l < "$LIST")
log "Eintraege  : $N_ENTRIES"

# Gegen die Datei pruefen, nicht per Pipe: grep -q loest sonst SIGPIPE
# aus und die Pruefung scheitert unter pipefail trotz vorhandener Datei.
for must in "$JOB/MANIFEST.txt" "$JOB/TARGET/config.txt"; do
    grep -qxF "$must" "$LIST" || die "$must fehlt im Archiv."
done
log "Archiv verifiziert."

rm -rf "$STAGE"

# ── Aufraeumen ────────────────────────────────────────────────────────
if [ "$PURGE" != true ]; then
    log "Fertig. Zum Aufraeumen erneut mit --purge aufrufen."
    exit 0
fi

log "── Aufraeumen ──"
STAMP=$(date '+%Y%m%d_%H%M%S')

# Erst verschieben: sofort fertig, die Pipeline kann wieder schreiben.
MOVED=()
for d in RESULTS data/LOG; do
    [ -d "$d" ] || continue
    old="${d}.old_${STAMP}"
    mv "$d" "$old" || { log "WARNUNG: $d nicht verschiebbar."; continue; }
    mkdir -p "$d"
    log "$d -> $old"
    MOVED+=("$old")
done

if [ ${#MOVED[@]} -eq 0 ]; then
    log "Nichts zu loeschen."
    log "Fertig. Archiv: $ARCHIVE"
    exit 0
fi

# Eigener Job statt Hintergrundprozess: aus einem Slurm-Job heraus
# ueberlebt ein nohup-rm das Jobende nicht. Keine --dependency noetig,
# diese Stelle wird nur nach erfolgreicher Verifikation erreicht.
if command -v sbatch >/dev/null 2>&1 && [ -f "$PROJECT/cleanup.slurm" ]; then
    CLEAN_JOB=$(sbatch --parsable "$PROJECT/cleanup.slurm" "${MOVED[@]}" 2>&1)
    if [[ "$CLEAN_JOB" =~ ^[0-9]+$ ]]; then
        log "Loeschjob   : $CLEAN_JOB  (${MOVED[*]})"
    else
        log "WARNUNG: Loeschjob nicht eingereicht: $CLEAN_JOB"
        log "         Von Hand:  sbatch cleanup.slurm ${MOVED[*]}"
    fi
else
    # Von der Login-Shell aus ueberlebt nohup.
    log "Kein sbatch/cleanup.slurm – loesche im Hintergrund."
    mkdir -p logs
    for old in "${MOVED[@]}"; do
        # Nach logs/, sonst passt die Logdatei selbst auf *.old_*.
        rmlog="logs/rm_$(basename "$old").log"
        nohup rm -r "$old" > "$rmlog" 2>&1 &
        log "  rm -r $old (PID $!, Log: $rmlog)"
    done
    log "Fortschritt: ps -u \$USER -o pid,etime,cmd | grep '[r]m -r'"
fi

log "Fertig. Archiv: $ARCHIVE"
exit 0
