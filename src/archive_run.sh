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
TOP_N=250
OUT_DIR=""

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Optionen:
  --purge         RESULTS/ und data/LOG/ nach der Verifikation loeschen
  --no-poses      nur CSVs aus RESULTS/, keine _docked.pdbqt
  --top N         Groesse der Top-Liste (default 250)
  --out DIR       Zielverzeichnis fuer das Archiv (default ./archive)
  --dry-run       nur zeigen, was passieren wuerde
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --purge)    PURGE=true ;;
        --no-poses) WITH_POSES=false ;;
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

# ── Bestandsaufnahme ──────────────────────────────────────────────────
[ -d RESULTS ] || die "RESULTS/ fehlt."
[ -d TARGET ]  || die "TARGET/ fehlt."

mapfile -t TARGETS < <(find RESULTS -mindepth 1 -maxdepth 1 -type d \
                       -not -name '_probe_*' -printf '%f\n' | sort)
[ ${#TARGETS[@]} -gt 0 ] || die "Keine Targets in RESULTS/."

log "Targets    : ${TARGETS[*]}"

N_POSEN=$(find RESULTS -name '*_docked.pdbqt' | wc -l)
SZ_RESULTS=$(du -sh RESULTS 2>/dev/null | cut -f1)
SZ_LOG=$(du -sh data/LOG 2>/dev/null | cut -f1)
log "RESULTS    : $SZ_RESULTS ($N_POSEN Posendateien)"
log "data/LOG   : ${SZ_LOG:-fehlt}"

if [ "$WITH_POSES" = true ] && [ "$N_POSEN" -gt 500000 ]; then
    log "HINWEIS: $N_POSEN Posendateien werden mitarchiviert. Das dauert"
    log "         lange und erzeugt ein sehr grosses Archiv. Mit --no-poses"
    log "         landen nur die CSVs im Tarball."
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

cp -a TARGET/config.txt "$STAGE/$JOB/TARGET/" 2>/dev/null \
    || log "WARNUNG: TARGET/config.txt fehlt."
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

# Worker-Logs aus data/LOG
if [ -d data/LOG ]; then
    cp -a data/LOG "$STAGE/$JOB/LOG"
    log "Worker-Logs: $(find "$STAGE/$JOB/LOG" -type f | wc -l) Datei(en)"
else
    log "WARNUNG: data/LOG fehlt."
fi

# ── Ranking-CSVs und Top-Listen ───────────────────────────────────────
log "── Ranking ──"
n_rank=0
for t in "${TARGETS[@]}"; do
    src="RESULTS/$t/rescoring_ligands_${t}.csv"
    if [ ! -s "$src" ]; then
        log "WARNUNG: $src fehlt – Rescoring fuer '$t' nicht abgeschlossen?"
        continue
    fi
    cp -a "$src" "$STAGE/$JOB/"
    n_rank=$((n_rank+1))

    # Top-N: die Datei ist bereits nach ecr_rank sortiert, aber darauf
    # verlassen wir uns nicht – lieber explizit nach ecr_score sortieren.
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
[ "$n_rank" -gt 0 ] || log "WARNUNG: Kein einziges Ranking gefunden."

# ── Manifest ──────────────────────────────────────────────────────────
{
    echo "Lauf:        $JOB"
    echo "Archiviert:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Projekt:     $PROJECT"
    echo "Host:        $(hostname -s)"
    echo "Targets:     ${TARGETS[*]}"
    echo "Posen:       $N_POSEN"
    echo "Posen im Archiv: $WITH_POSES"
    echo
    echo "── Konfiguration ──"
    for ini in config/*.ini; do
        [ -f "$ini" ] && { echo "--- $ini ---"; grep -v '^\s*#' "$ini" | grep -v '^\s*$'; echo; }
    done
} > "$STAGE/$JOB/MANIFEST.txt"

# Konfigurationsdateien selbst mit hinein – der Kommentarapparat ist Teil
# der Dokumentation des Laufs.
mkdir -p "$STAGE/$JOB/config"
cp -a config/*.ini "$STAGE/$JOB/config/" 2>/dev/null || true

# ── Archiv bauen ──────────────────────────────────────────────────────
log "── Archiv bauen ──"
mkdir -p "$OUT_DIR"
START=$(date +%s)

# RESULTS wird NICHT kopiert, sondern direkt aus dem Original in den
# Tarball geschrieben und dabei unter <jobname>/RESULTS einsortiert.
if [ "$WITH_POSES" = true ]; then
    RESULTS_ARGS=(-C "$PROJECT" RESULTS)
else
    # Nur CSVs: Dateiliste vorbereiten, sonst wird die Kommandozeile zu lang.
    find RESULTS -maxdepth 2 -name '*.csv' -not -path '*/.rescore_partial/*' \
        > "$STAGE/results_files.txt"
    log "Ohne Posen : $(wc -l < "$STAGE/results_files.txt") CSV-Datei(en)"
    RESULTS_ARGS=(-C "$PROJECT" -T "$STAGE/results_files.txt")
fi

tar --use-compress-program="$COMPRESS" \
    --transform "s|^RESULTS|$JOB/RESULTS|" \
    -cf "$ARCHIVE" \
    -C "$STAGE" "$JOB" \
    "${RESULTS_ARGS[@]}" \
    || die "tar fehlgeschlagen."

DT=$(( $(date +%s) - START ))
SZ=$(du -sh "$ARCHIVE" | cut -f1)
log "Archiv     : $ARCHIVE ($SZ, ${DT}s)"

# ── Verifikation ──────────────────────────────────────────────────────
# Ohne diesen Schritt wuerde --purge Daten loeschen, deren Archiv
# moeglicherweise abgeschnitten ist (volle Platte, Timeout, Signal).
log "── Verifikation ──"
# Genau EINMAL auflisten: bei einem 50-GB-Archiv ist jeder Durchlauf ein
# vollstaendiges Entpacken. Das Ergebnis wandert in eine Datei, gegen die
# dann geprueft wird.
LIST="$STAGE/archive_list.txt"
tar --use-compress-program="$COMPRESS" -tf "$ARCHIVE" > "$LIST" \
    || die "Archiv nicht lesbar – NICHTS wird geloescht."
N_ENTRIES=$(wc -l < "$LIST")
log "Eintraege  : $N_ENTRIES"

# grep -q beendet sich beim ersten Treffer; in einer Pipeline mit tar
# haette das unter 'set -o pipefail' SIGPIPE ausgeloest und die Pruefung
# waere fehlgeschlagen, obwohl die Datei vorhanden ist.
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

# Erst verschieben, dann im Hintergrund loeschen: das Umbenennen ist
# sofort fertig, die Pipeline kann direkt wieder schreiben, und das
# eigentliche Loeschen von Millionen Dateien laeuft nebenher.
for d in RESULTS data/LOG; do
    [ -d "$d" ] || continue
    old="${d}.old_${STAMP}"
    mv "$d" "$old" || { log "WARNUNG: $d nicht verschiebbar."; continue; }
    mkdir -p "$d"
    log "$d -> $old (wird im Hintergrund geloescht)"
    setsid nohup rm -rf "$old" > /dev/null 2>&1 &
done

log "Fertig. Archiv: $ARCHIVE"
log "Loeschvorgaenge laufen im Hintergrund – mit 'du -sh *.old_*' pruefbar."
exit 0
