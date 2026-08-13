#!/bin/bash
# ============================================================================
# setup_repo.sh
# Baut aus dem flachen Datei-Bundle die Repository-Struktur.
#
# Die Dateien liegen nach dem Download alle in einem Ordner. README,
# pipeline_start.sh und die Container-Builds erwarten aber src/, config/,
# docs/ und build/. Dieses Skript sortiert einmalig ein.
#
#     bash setup_repo.sh              # im Ordner mit den Dateien ausfuehren
#     bash setup_repo.sh /ziel/pfad   # oder in ein leeres Zielverzeichnis
#
# Idempotent: mehrfaches Ausfuehren schadet nicht.
# ============================================================================

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-$SRC}"

echo "Quelle : $SRC"
echo "Ziel   : $DST"
echo

mkdir -p "$DST"/{src,config,docs,build,LIB,TARGET,data/LOG,data/PDBQT,RESULTS}

move() {   # $1 = Datei, $2 = Zielordner
    if [ -f "$SRC/$1" ]; then
        mkdir -p "$DST/$2"
        [ "$SRC/$1" = "$DST/$2/$1" ] || cp "$SRC/$1" "$DST/$2/$1"
        echo "  $2/$1"
    else
        echo "  FEHLT: $1" >&2
    fi
}

echo "── src/ ─────────────────────────────────────────────"
for f in pipeline_common.py docking_config.py unidock_engine.py \
         worker_dock.py worker_restart_dock.py worker_rescore.py \
         orchestrator.py restart_orchestrator.py \
         docking_rescore.py gnina_refinement.py gnina_gpu_worker.py \
         linf9xgb_scorer.py sdf_to_pdbqt.py \
         check_config.py check_ligands.py; do
    move "$f" src
done

echo
echo "── config/ ──────────────────────────────────────────"
for f in docking.ini rescore.ini; do move "$f" config; done

echo
echo "── build/ ───────────────────────────────────────────"
# featureSASA.py und prepare_betaAtoms.py sind gepatchte Fremdquellen und
# werden im %post des Rescoring-Containers an ihren Zielort kopiert.
for f in unidock-gpu.def rescoring-gpu.def sdf_to_pdbqt.def \
         featureSASA.py prepare_betaAtoms.py; do
    move "$f" build
done

echo
echo "── docs/ ────────────────────────────────────────────"
for f in pipeline.svg pipeline-dark.svg pipeline.png pipeline-dark.png \
         make_diagram.py README_PIPELINE.md MANIFEST.md; do
    move "$f" docs
done

echo
echo "── Wurzel ───────────────────────────────────────────"
for f in README.md THIRD_PARTY.md CITATION.cff .gitignore .gitattributes \
         pipeline_start.sh setup_repo.sh; do
    move "$f" .
done
chmod +x "$DST/pipeline_start.sh" "$DST/setup_repo.sh" 2>/dev/null || true

# Leere Verzeichnisse fuer git sichtbar halten
for d in LIB TARGET data/LOG data/PDBQT RESULTS; do
    touch "$DST/$d/.gitkeep"
done

echo
echo "════════════════════════════════════════════════════"
echo " Struktur angelegt. Naechste Schritte:"
echo "════════════════════════════════════════════════════"
cat <<'NEXT'

  1. Platzhalter ersetzen
       README.md      Abschnitte License und Contact
       CITATION.cff   authors, license, repository-code

  2. gnina-Binary nach build/ legen
       https://github.com/gnina/gnina/releases

  3. Rezeptoren eintragen
       TARGET/config.txt  +  TARGET/<name>.pdbqt

  4. Git initialisieren
       git init
       git config core.autocrlf false        # nur unter Windows noetig
       git add .
       git commit -m "Initial commit: three-stage GPU docking pipeline"
       git branch -M main
       git remote add origin git@github.com:roessel-cmd/kepler-dock-cascade.git
       git push -u origin main

  5. Ausfuehrbar-Bit sichern (Windows setzt es nicht)
       git update-index --chmod=+x pipeline_start.sh setup_repo.sh
       git commit -m "Mark scripts executable" && git push

  Nicht mitversioniert (siehe .gitignore): *.sif, build/gnina,
  LIB/, data/, RESULTS/ und TARGET/*.pdbqt

NEXT
