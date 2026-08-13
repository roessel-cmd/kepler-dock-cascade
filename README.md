<!--
  VOR DEM OEFFENTLICHSCHALTEN NOCH OFFEN:
    1. Abschnitt License → Lizenz waehlen, siehe THIRD_PARTY.md
    2. Abschnitt Contact → Name, Institution (optional)
    3. CITATION.cff → authors, license
    4. Abschnitt Validation → nach dem DUD-E-Lauf mit Zahlen ersetzen
-->

# kepler-dock-cascade

![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)
![GPU](https://img.shields.io/badge/GPU-compute%20capability%20%E2%89%A5%207.0-76B900)
![Containers](https://img.shields.io/badge/containers-Apptainer-1D4ED8)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)

**A three-stage, multi-GPU pipeline for high-throughput structure-based virtual screening.**

Ligand preparation, GPU docking, and consensus rescoring run as three independent
containers, each with its own configuration and its own entry point. Stages can be
run individually or chained, and the docking stage resumes from where it stopped
after a crash or a wall-clock limit.

Built around [Uni-Dock](https://github.com/dptech-corp/Uni-Dock) for docking and
[gnina](https://github.com/gnina/gnina) for CNN-based rescoring.

---

## Overview

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/pipeline-dark.svg">
  <img alt="Three-stage pipeline: preparation, docking, rescoring" src="docs/pipeline.svg" width="100%">
</picture>

Each stage is a separate Apptainer image with a disjoint dependency set. The
preparation container has no CUDA, the docking container has no scoring stack,
and the rescoring container has no ligand-preparation tooling.

---

## Design notes

**Direct SDF → PDBQT conversion.** Going through PDB discards explicit bond
orders; downstream tools then reconstruct connectivity geometrically, which
misassigns bonds for unfavourable conformers. RDKit reads bond orders from the
SDF and Meeko writes PDBQT natively, so the information is never lost.

**Batched docking.** Ligands are submitted to Uni-Dock in batches rather than one
process per ligand. Receptor parsing, grid-map construction, and GPU context
setup are amortised across the batch instead of being repaid for every molecule.
On out-of-memory the batch is halved and retried recursively, so a single
oversized ligand does not cost the whole chunk.

**Work-stealing across GPUs.** Targets are assigned to GPUs, but a GPU that
finishes its target early pulls chunks from other targets' queues instead of
idling. Two chunk sizes are tuned independently: the orchestrator's dispatch
unit and Uni-Dock's per-process batch.

**Filesystem as state.** Completion is derived from the presence of
`*_docked.pdbqt` files and `.done` markers, not from a database or an in-memory
job table. A restart re-derives what is left to do by looking at the disk, which
makes it robust to any kind of abrupt termination.

**Robustness against container process hangs.** Apptainer containers have no
PID 1 init to reap children. Under parallel workers, `SIGCHLD` can be lost and
`os.system` blocks indefinitely in `wait()`. The MSMS and MGLTools call sites are
patched to use `subprocess` with timeouts, `start_new_session`, and process-group
kills, and the container build verifies via AST inspection that no `os.system`
call survives.

---

## Requirements

- Linux with NVIDIA GPUs, compute capability ≥ 7.0
- NVIDIA driver with CUDA 12.x support
- [Apptainer](https://apptainer.org/) ≥ 1.2
- Python 3.10+ on the host (for the orchestrator and pre-flight checks)

Verified on 4 × NVIDIA H100 (sm_90).

---

## Installation

```bash
git clone https://github.com/roessel-cmd/kepler-dock-cascade.git
cd kepler-dock-cascade
```

The `gnina` binary is not distributed here. Download it from the
[gnina releases](https://github.com/gnina/gnina/releases) and place it in
`build/` before building the rescoring image.

Build the three containers:

```bash
cd build/

cp ../src/sdf_to_pdbqt.py .
apptainer build ../sdf_to_pdbqt.sif sdf_to_pdbqt.def

cp ../src/{pipeline_common.py,docking_config.py,unidock_engine.py} .
cp ../src/{worker_dock.py,worker_restart_dock.py} .
apptainer build ../unidock-gpu.sif unidock-gpu.def

cp ../src/{worker_rescore.py,docking_rescore.py,gnina_refinement.py} .
cp ../src/{gnina_gpu_worker.py,linf9xgb_scorer.py} .
apptainer build ../rescoring-gpu.sif rescoring-gpu.def
```

Verify that the Uni-Dock build supports your GPU architecture — this is the one
check worth doing before anything else:

```bash
apptainer exec --nv unidock-gpu.sif bash -c \
  'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --help'
```

`no kernel image is available for execution on the device` means the packaged
binary lacks your compute capability. Build Uni-Dock from source in `%post` with
`CMAKE_CUDA_ARCHITECTURES` set accordingly.

---

## Usage

Define your receptors in `TARGET/config.txt`:

```
BRD4
CENTER [12.5, -8.3, 22.1]
BOX_SIZE [25.0, 25.0, 25.0]

CDK2
CENTER [4.1, 15.7, -3.2]
BOX_SIZE [22.0, 22.0, 22.0]
LIGAND_SUBDIR = cdk2_dude
```

Each block needs a `<name>.pdbqt` in `TARGET/`. `LIGAND_SUBDIR` restricts a
target to its own ligand subset, which is what the DUD-E validation runs use.

Place your library as one or more `.sdf` files in `LIB/`, then:

```bash
python3 src/check_config.py config/    # configuration consistency
./pipeline_start.sh --dry-run          # show every command without running it
./pipeline_start.sh
```

Stages are toggled at the top of `pipeline_start.sh`:

```bash
RUN_CONVERSION=true
RUN_DOCKING=true
RUN_RESCORING=true
```

Re-ranking with different consensus weights costs one flag and no re-docking:
set `RUN_CONVERSION=false`, `RUN_DOCKING=false`, adjust the `w_*` values in
`config/rescore.ini`, and run again.

To resume an interrupted docking run:

```bash
./pipeline_start.sh --restart
```

---

## Configuration

Two INI files, one per GPU stage. Stage 1 is configured through CLI arguments in
`pipeline_start.sh`.

`config/docking.ini` — search effort, batch size, GPU count:

```ini
[UNIDOCK]
search_mode = balance    # fast | balance | detail
batch_size  = 1000       # ligands per Uni-Dock process
scoring     = vina       # vina | vinardo | ad4

[CHUNK]
chunk_size  = 5000       # orchestrator dispatch unit
```

`config/rescore.ini` — which scoring functions contribute and how they are
weighted in the consensus:

```ini
[RESCORE]
# empirical (Vina family)
vina_enabled          = true    # read from the pose file, free
vinardo_enabled       = false   # one extra gnina --score_only pass
ad4_enabled           = false   # one extra gnina --score_only pass
# learned
cnnaffinity_enabled   = true
cnnscore_enabled      = true
deltalinf9xgb_enabled = false

sigma_fraction        = 4.0
w_vina                = 0.0     # all zero -> equal weights over active scores
```

Every scoring function is individually switchable and individually weighted.
Weights normalise over the *active* functions only, so disabling one does not
silently reweight the others. The Vina-family functions are strongly correlated
with each other, so enabling all three under equal weighting gives the empirical
view three votes against one each for the learned scores. `check_config.py`
warns about this.

`check_config.py` verifies that the handover paths (`results_dir`, `target_dir`)
agree between the two files and that value ranges are sane. It runs automatically
before each pipeline start.

---

## Consensus ranking

Scores from different functions are combined by exponential consensus ranking
(Palacio-Rodríguez et al., *Sci Rep* **9**, 5142, 2019):

```
ECR_j(r) = exp(-r / σ)          r = rank of the pose under score j
P(pose)  = Σ_j ECR_j(r_j)       σ = N / sigma_fraction
P(ligand) = max over its poses
```

Scores where larger is better (CNNaffinity, CNNscore, ΔLin_F9XGB) are inverted
before ranking. Weights normalise over the active functions only, so disabling a
score does not silently reweight the rest.

---

## Pre-flight checks

Two host-side scripts, both wired into `pipeline_start.sh`:

```bash
python3 src/check_config.py  config/
python3 src/check_ligands.py data/PDBQT
```

`check_ligands.py` catches the failure mode that is easiest to miss: the
converter derives filenames from SDF titles, so two molecules sharing a title
produce the same filename. Docking results are written flat per target, meaning
such ligands overwrite each other silently and the run finishes with a plausible
but incomplete result table. The check aborts the pipeline before any GPU time
is spent.

---

## Validation

**Results from this pipeline have not yet been benchmarked against a reference
set.** Before using it for production screening, run a DUD-E target through it
and compare enrichment (EF1%, ROC AUC) against published values for the same
target and scoring function. Absolute binding energies are not comparable across
docking engines; ranking behaviour is what matters.

This section will be updated once those numbers exist.

---

## Known limitations

- Uni-Dock rejects ligands exceeding its internal atom and torsion limits. These
  are reported as failures in the per-target CSV rather than silently dropped,
  but they do bias the screened set toward smaller, less flexible molecules.
- Stage 1 keeps only the largest fragment of multi-component entries, which is
  the right behaviour for salts but is applied without reporting.
- `parse_target_config` exists in two implementations — `pipeline_common.py` and
  a standalone copy inside `docking_rescore.py`. Changes to the `config.txt`
  format must currently be made in both.
- No automated test suite.

---

## Repository layout

```
├── pipeline_start.sh          Stage toggles and orchestration
├── config/                    docking.ini, rescore.ini
├── src/                       All Python modules (single source of truth)
├── build/                     Container definitions and build inputs
├── LIB/                       Input SDF libraries (not tracked)
├── TARGET/                    Receptor PDBQTs and config.txt
├── data/                      Intermediate PDBQTs and logs (not tracked)
└── RESULTS/                   Poses, docking CSVs, rescoring output (not tracked)
```

`src/` is bind-mounted into the containers at runtime as well as baked in at
build time, so edits take effect without a rebuild.

---

## Third-party components

This pipeline orchestrates several external tools. Please cite them when you
publish results:

- **Uni-Dock** — Yu, Cai, Wang, Bo, Zhu & Zheng, *J Chem Theory Comput* (2023)
- **AutoDock Vina** — Eberhardt, Santos-Martins, Tillack & Forli, *J Chem Inf Model* **61**, 3891 (2021)
- **gnina** — McNutt et al., *J Cheminform* **13**, 43 (2021)
- **Meeko / RDKit** — ligand preparation
- **ΔLin_F9XGB** — Yang & Zhang, *J Chem Inf Model* (2022)
- **ECR** — Palacio-Rodríguez et al., *Sci Rep* **9**, 5142 (2019)

`build/featureSASA.py` and `build/prepare_betaAtoms.py` are modified versions of
files from the delta_LinF9_XGB project, redistributed under that project's
license. See `THIRD_PARTY.md`.

---

## License

Not yet chosen. `build/featureSASA.py` and `build/prepare_betaAtoms.py` are
modified copies of third-party source, and the upstream license constrains what
this repository can be released under. See [THIRD_PARTY.md](THIRD_PARTY.md)
before picking one.

---

## Contact

Questions and bug reports: please open an issue.

<!-- Optional: name, affiliation, email -->
