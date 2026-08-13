# Pipeline – Aufbau und Betrieb

Drei Stufen, drei Container, ein Startskript.

```
Stufe 1  CONVERSION   sdf_to_pdbqt.sif    SDF → PDBQT      CPU     CLI-Parameter
Stufe 2  DOCKING      unidock-gpu.sif     Uni-Dock         GPU     config/docking.ini
Stufe 3  RESCORING    rescoring-gpu.sif   gnina/ECR        GPU     config/rescore.ini
```

---

## Verzeichnislayout

```
projekt/
├── pipeline_start.sh          ← alle Schalter stehen hier oben
├── sdf_to_pdbqt.sif
├── unidock-gpu.sif
├── rescoring-gpu.sif
├── config/
│   ├── docking.ini
│   └── rescore.ini
├── src/
│   ├── pipeline_common.py     ← geteilt: TargetConfig, Ligandensuche, Logging
│   ├── docking_config.py
│   ├── unidock_engine.py
│   ├── worker_dock.py
│   ├── worker_restart_dock.py
│   ├── orchestrator.py
│   ├── restart_orchestrator.py
│   ├── check_config.py
│   └── check_ligands.py
├── LIB/                       ← Eingangs-SDFs
├── TARGET/                    ← Rezeptor-PDBQTs + config.txt
├── data/
│   ├── PDBQT/                 ← Stufe 1 → Stufe 2
│   └── LOG/
└── RESULTS/                   ← Stufe 2 → Stufe 3
```

---

## Bedienung

Alle Schalter stehen im Kopf von `pipeline_start.sh`:

```bash
RUN_CONVERSION=true
RUN_DOCKING=true
RUN_RESCORING=true

CHECK_CONFIG=true       # INI-Konsistenz vorab pruefen
CHECK_LIGANDS=true      # PDBQT-Bibliothek vorab pruefen
CONTINUE_ON_ERROR=false # bei Fehler abbrechen statt weitermachen
```

```bash
./pipeline_start.sh              # Stufen laut Schaltern
./pipeline_start.sh --dry-run    # nur zeigen, was ausgefuehrt wuerde
./pipeline_start.sh --restart    # Docking-Stufe im Restart-Modus
```

Typische Kombinationen: nur `RUN_CONVERSION=true` fuer eine neue Bibliothek,
danach nur `RUN_DOCKING=true`, und bei geaenderten ECR-Gewichten nur
`RUN_RESCORING=true` – ohne neu zu docken.

---

## Vorabpruefungen

Beide laufen automatisch, wenn die Schalter gesetzt sind, und sind auch
einzeln aufrufbar:

```bash
python3 src/check_config.py config/
python3 src/check_ligands.py data/PDBQT
```

`check_config.py` prueft, dass `results_dir` und `target_dir` in
`docking.ini` und `rescore.ini` uebereinstimmen, plus Plausibilitaeten
(`search_mode`, `scoring`, `batch_size` vs. `chunk_size`, ECR-Gewichte,
`top_fraction`).

`check_ligands.py` prueft Layout, Anzahl, leere Dateien und – am
wichtigsten – **doppelte Ligandennamen**. `sdf_to_pdbqt.py` leitet den
Dateinamen aus dem SDF-Titel ab; zwei Molekuele mit gleichem Titel
bekommen denselben Namen und wuerden sich in `RESULTS/` gegenseitig
ueberschreiben. Der Check bricht die Pipeline in dem Fall ab, bevor
Rechenzeit verbrannt wird.

---

## Stufe 1 im Detail

`pipeline_start.sh` findet alle `*.sdf` in `LIB/` und ruft den Container
pro Datei auf. Bei mehreren SDFs bekommt jede ihren eigenen Unterordner
unter `data/PDBQT/` (`CONV_SUBDIR_PER_SDF=true`) – sonst wuerden sich die
`0000/`-Zaehlungen zweier Laeufe ueberlagern. Die Docking-Stufe sucht
rekursiv und findet sie trotzdem.

`CONV_SKIP_IF_EXISTS=true` ueberspringt eine SDF, wenn im Zielordner
schon PDBQTs liegen – praktisch beim Wiederanlauf.

Wichtig: `CONV_OUT_DIR` muss `[PATHS] pdbqt_dir` in `docking.ini`
entsprechen. Das prueft kein Skript automatisch, weil Stufe 1 keine INI
liest.

---

## Stufe 2 und 3

Beide laufen ueber denselben `orchestrator.py`, unterschieden durch
`--stage`:

```bash
python3 src/orchestrator.py --project . --config config/docking.ini \
                            --stage dock    --sif unidock-gpu.sif
python3 src/orchestrator.py --project . --config config/rescore.ini \
                            --stage rescore --sif rescoring-gpu.sif
```

`--stage dock` fuehrt nur Chunking, Worker-Dispatch und CSV-Merge aus;
`--stage rescore` nur Rescoring und Refinement. Der frueher eingebaute
Aufruf der Ligandenaufbereitung ist entfallen – das ist jetzt Stufe 1.
Ebenso der CPU-Modus.

Die `[FLAGS] rescore_only`-Option wird nicht mehr gebraucht: die Stufe
entscheidet das Shell-Skript.

---

## Restart

Schlaegt Stufe 2 fehl, meldet das Skript am Ende:

```
./pipeline_start.sh --restart
```

Das setzt `DOCK_RESTART=true` und nutzt `restart_orchestrator.py`, der
fertige Liganden anhand vorhandener `*_docked.pdbqt` ueberspringt.
Setz dabei `RUN_CONVERSION=false`, sonst laeuft Stufe 1 nochmal an
(sie ueberspringt sich zwar selbst, kostet aber Zeit beim Scannen).

---

## Was noch offen ist

Der Rescoring-Container `rescoring-gpu.def` ist unveraendert – dort liegen
`docking_rescore.py`, `gnina_refinement.py`, `linf9xgb_scorer.py` und die
gnina/MSMS/AlphaSpace-Umgebung. Er importiert `docking_config` nicht mehr,
weil die Kopplung aufgeloest wurde; ein Rebuild ist nur noetig, wenn du
`pipeline_common.py` auch dort verwenden willst (aktuell braucht er es
nicht – beide Module bringen ihre eigene Target-Parserfunktion mit).

Das ist Redundanz, die man beizeiten aufloesen sollte:
`_parse_target_config_standalone()` in `docking_rescore.py` macht
dasselbe wie `parse_target_config()` in `pipeline_common.py`.
