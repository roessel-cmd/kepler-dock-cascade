# MANIFEST – alle Dateien, Ziele, Reihenfolge

Vollstaendige Liste dessen, was fuer einen Start gebraucht wird.
Legende: **N** = neu, **P** = gepatcht, **U** = unveraendert von dir.

---

## Zielstruktur

```
projekt/
├── pipeline_start.sh                    N   ← alle Schalter
├── README.md / MANIFEST.md              N
│
├── docs/
│   ├── pipeline.svg                     N   ← im README eingebunden
│   ├── pipeline-dark.svg                N   ← GitHub Dark Mode
│   ├── pipeline.png                     N   ← 2x, Fallback fuer PowerPoint
│   ├── pipeline-dark.png                N
│   └── make_diagram.py                  N   ← erzeugt alle vier neu
│
├── sdf_to_pdbqt.sif                         Build aus sdf_to_pdbqt.def
├── unidock-gpu.sif                          Build aus unidock-gpu.def
├── rescoring-gpu.sif                        Build aus rescoring-gpu.def
│
├── config/
│   ├── docking.ini                      N
│   └── rescore.ini                      N
│
├── src/
│   │  ── geteilt ────────────────────────────────────────────
│   ├── pipeline_common.py               N
│   │
│   │  ── Stufe 1 (Conversion) ───────────────────────────────
│   ├── sdf_to_pdbqt.py                  U
│   │
│   │  ── Stufe 2 (Docking) ──────────────────────────────────
│   ├── docking_config.py                P
│   ├── unidock_engine.py                N
│   ├── worker_dock.py                   N   (ersetzt worker_gpu.py)
│   ├── worker_restart_dock.py           P   (ersetzt worker_restart.py)
│   ├── orchestrator.py                  P
│   ├── restart_orchestrator.py          P
│   │
│   │  ── Stufe 3 (Rescoring) ────────────────────────────────
│   ├── worker_rescore.py                N
│   ├── docking_rescore.py               U
│   ├── gnina_refinement.py              U
│   ├── gnina_gpu_worker.py              U
│   ├── linf9xgb_scorer.py               U
│   │
│   │  ── Pruefwerkzeuge (Host) ──────────────────────────────
│   ├── check_config.py                  N
│   └── check_ligands.py                 N
│
├── build/                                   nur zum Bauen der Container
│   ├── sdf_to_pdbqt.def                 U
│   ├── unidock-gpu.def                  P
│   ├── rescoring-gpu.def                P
│   ├── gnina                                Binary, selbst herunterladen
│   ├── featureSASA.py                   U   (gepatcht, wird in %post kopiert)
│   └── prepare_betaAtoms.py             U   (gepatcht, wird in %post kopiert)
│
├── LIB/                                     Eingangs-SDFs
├── TARGET/                                  Rezeptor-PDBQTs + config.txt
├── data/{PDBQT,LOG}/
└── RESULTS/
```

**Ja – die Rescoring-Module gehoeren nach `src/`.** Der Orchestrator bindet
sie von dort in den Rescoring-Container (Entwicklermodus), gleichzeitig
liegen sie ueber `%files` im Image. `src/` ist damit die einzige Quelle der
Wahrheit; Aenderungen wirken sofort, ohne Rebuild.

---

## Container bauen

Jeder Build braucht seine Dateien im **Build-Verzeichnis** (Apptainer
`%files` loest relativ zum Aufrufort auf):

```bash
cd build/

# Stufe 1 – braucht nur sdf_to_pdbqt.py
cp ../src/sdf_to_pdbqt.py .
apptainer build ../sdf_to_pdbqt.sif sdf_to_pdbqt.def

# Stufe 2 – braucht die vier Docking-Module
cp ../src/{pipeline_common.py,docking_config.py,unidock_engine.py} .
cp ../src/{worker_dock.py,worker_restart_dock.py} .
apptainer build ../unidock-gpu.sif unidock-gpu.def

# Stufe 3 – braucht die Rescoring-Module + gnina + die zwei Patch-Dateien
cp ../src/{worker_rescore.py,docking_rescore.py,gnina_refinement.py} .
cp ../src/{gnina_gpu_worker.py,linf9xgb_scorer.py} .
# featureSASA.py, prepare_betaAtoms.py und gnina liegen bereits hier
apptainer build ../rescoring-gpu.sif rescoring-gpu.def
```

Nach dem Build von Stufe 2 **einmal auf einer H100 pruefen**, ob das
conda-forge-Binary sm_90 mitbringt:

```bash
apptainer exec --nv unidock-gpu.sif bash -c \
  'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --help'
```

Bei `no kernel image is available for execution on the device` fehlt sm_90 –
dann Uni-Dock im `%post` aus dem Quellcode bauen mit
`CMAKE_CUDA_ARCHITECTURES 90`.

---

## Erststart

```bash
python3 src/check_config.py config/        # INIs konsistent?
./pipeline_start.sh --dry-run              # zeigt jeden Aufruf
./pipeline_start.sh                        # los
```

---

## Was ersatzlos entfaellt

Diese Dateien sind durch den Umbau tot und koennen weg:

| Datei | Grund |
|---|---|
| `worker_gpu.py` | → `worker_dock.py` (Docking) + `worker_rescore.py` (Rescoring) |
| `worker_restart.py` | → `worker_restart_dock.py` |
| `docking_pipeline.py` | Standalone-Pfad, ersetzt durch Orchestrator + Stages |
| `docking_restart.py` | dito |
| `docking-gpu.def` | → `unidock-gpu.def` |
| `patch_kernels.py` | Vina-GPU-OpenCL-Patches, mit Uni-Dock gegenstandslos |
| `kernels/{ampere,ada,hopper,blackwell}/` | Kernel-Bins, dito |
| `start_pipeline.sh`, `start_restart.sh` | → `pipeline_start.sh` |
| `pipeline_config.ini` (monolithisch) | → `config/docking.ini` + `config/rescore.ini` |

`docking_pipeline.py` und `docking_restart.py` erst loeschen, wenn du
sicher bist, dass du die Standalone-Pfade nicht mehr brauchst – sie
enthalten noch die alte Vina-GPU-Logik und wurden nicht auf Uni-Dock
umgestellt.

---

## Nachtrag: vier Fehler, die die Pruefung gefunden hat

1. **`restart_orchestrator.py` band `worker_restart.py`** – den alten Namen.
   Der Restart-Lauf waere mit „file not found" gestorben. Jetzt
   `worker_restart_dock.py`, plus `unidock_engine.py`, `docking_config.py`
   und `pipeline_common.py` im Entwicklermodus mitgebunden.

2. **`restart_orchestrator.py` kannte `--stage` und `--sif` nicht**, obwohl
   `pipeline_start.sh --restart` sie uebergibt → „unrecognized arguments".
   Beide Argumente ergaenzt, Container-Aufloesung wie im Orchestrator.

3. **`worker_restart_dock.py` importierte `rescore_target` und
   `log_top10_ecr` aus `docking_config`** – die das schlanke Modul nicht mehr
   re-exportiert. ImportError direkt beim Start. Der angehaengte
   Rescoring-Block ist entfernt; Rescoring ist Stufe 3.

4. **`rescore.ini` hatte keine `[GPU]`-Sektion.** Der Orchestrator liest
   `num_gpus` auch in Stage `rescore`; ohne die Sektion griff der Fallback 1
   und das Rescoring haette auf einer statt vier GPUs gelaufen.

Ausserdem entfernt: `use_gpu` wurde noch aus der INI gelesen, stand dort aber
nicht mehr. Da es keinen CPU-Pfad mehr gibt, ist es jetzt im Code auf `True`
gesetzt statt konfigurierbar – sonst haette ein `use_gpu = false` die Pipeline
still nichts tun lassen.

## Nachtrag: ein Bug, den ich beim Zusammenstellen gefunden habe

Der Orchestrator rief fuer das Rescoring bisher
`/app/worker_gpu.py` mit `ORCHESTRATOR_RESCORE_ONLY=1` auf. Diesen Modus
gab es nur im alten Sammel-Worker, den wir aufgeloest haben – die
Rescoring-Stufe waere also mit „file not found" gestorben.

Deshalb `worker_rescore.py`: ein duenner Einstiegspunkt, der genau ein
Target bearbeitet (ausgewaehlt ueber `WORKER_TARGET`) und `rescore_target()`
aus `docking_rescore.py` aufruft. Noetig, weil dessen eigenes `main()`
immer **alle** Targets nacheinander abarbeitet und damit die
Target-Parallelisierung ueber mehrere GPUs verhindert.

`gnina_refinement.py` braucht so einen Wrapper nicht – es wertet
`WORKER_TARGET` in seinem `main()` bereits selbst aus und wird direkt
aufgerufen.

Ausserdem aus den Rescoring-Binds entfernt: `docking_config.py`. Es wird
dort nicht gebraucht und wuerde ohne `pipeline_common.py` im Image sogar
einen ImportError werfen.
