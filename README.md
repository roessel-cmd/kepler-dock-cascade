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

Multi-GPU pipeline for structure-based virtual screening.

Ligand preparation, docking and consensus rescoring run as three separate
Apptainer containers, each with its own configuration and entry point. Stages
can be run individually or in sequence. The docking stage resumes from where it
stopped after a crash or a wall-clock limit.

Docking uses [Uni-Dock](https://github.com/dptech-corp/Uni-Dock), rescoring uses
[gnina](https://github.com/gnina/gnina).

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

**Direct SDF to PDBQT conversion.** PDB has no bond records, so the common
SDF → PDB → PDBQT route loses bond orders and the receiving tool reconstructs
connectivity from interatomic distances. That misassigns bonds for unfavourable
conformers. RDKit reads bond orders from the SDF, Meeko writes PDBQT without the
PDB step.

**Batched docking.** Ligands go to Uni-Dock in batches, not one process per
ligand. Receptor parsing, grid-map construction and GPU context setup happen
once per batch instead of once per molecule. On out-of-memory the batch is
halved and retried, so one oversized ligand does not cost the whole chunk.

**Work-stealing across GPUs.** Targets are assigned to GPUs. A GPU that finishes
its target early takes chunks from other targets' queues rather than idling.

**Filesystem as state.** Progress is derived from `*_docked.pdbqt` files and
`.done` markers, not from a database or an in-memory job table. A restart
recomputes the remaining work from what is on disk, so any form of abrupt
termination is recoverable.

**Hang protection in the rescoring container.** Apptainer containers have no
PID 1 init to reap children. With parallel workers, `SIGCHLD` can be lost and
`os.system` then blocks in `wait()` indefinitely. The MSMS and MGLTools call
sites use `subprocess` with timeouts, `start_new_session` and process-group
kills instead. The container build checks by AST inspection that no `os.system`
call remains.

---

## Requirements

- Linux with NVIDIA GPUs, compute capability ≥ 7.0
- NVIDIA driver with CUDA 12.x support
- [Apptainer](https://apptainer.org/) ≥ 1.2
- Python 3.10+ on the host (for the orchestrator and pre-flight checks)

Verified on 4 × NVIDIA H100 (sm_90) @[MUSICA](https://docs.vsc.ac.at/systems/musica.html) ASC-Supercomputer

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

# Stage 1
cp ../src/sdf_to_pdbqt.py .
apptainer build ../sdf_to_pdbqt.sif sdf_to_pdbqt.def

# Stage 2
cp ../src/{pipeline_common.py,docking_config.py,unidock_engine.py} .
cp ../src/{worker_dock.py,worker_restart_dock.py} .
apptainer build ../unidock-gpu.sif unidock-gpu.def

# Stage 3 — gnina must already be in build/
cp ../src/{worker_rescore.py,docking_rescore.py,gnina_refinement.py} .
cp ../src/{gnina_gpu_worker.py,linf9xgb_scorer.py,ecr.py} .
apptainer build ../rescoring-gpu.sif rescoring-gpu.def
```

Apptainer resolves `%files` relative to the working directory, so the modules
have to be copied into `build/` before each build. Rather than tracking that by
hand, check it:

```bash
cd build/
for d in sdf_to_pdbqt.def unidock-gpu.def rescoring-gpu.def; do
  echo "--- $d"
  sed -n '/^%files/,/^%[a-z]/p' "$d" | grep -oE '^\s+\S+' | tr -d ' ' | \
  while read -r f; do [ -e "$f" ] && echo "  ok    $f" || echo "  FEHLT $f"; done
done
```

Every entry has to resolve before you start a build. A missing file aborts the
build after the base image has already been pulled, which on a slow connection
costs more time than the check.

### Rebuilding after changes

Only the container whose modules changed needs rebuilding. `--force` overwrites
the existing image:

| Changed | Rebuild |
|---|---|
| `sdf_to_pdbqt.py` | `sdf_to_pdbqt.sif` |
| `pipeline_common.py`, `docking_config.py`, `unidock_engine.py`, `worker_dock.py`, `worker_restart_dock.py` | `unidock-gpu.sif` |
| `docking_rescore.py`, `ecr.py`, `worker_rescore.py`, `gnina_refinement.py`, `gnina_gpu_worker.py`, `linf9xgb_scorer.py` | `rescoring-gpu.sif` |
| `orchestrator.py`, `restart_orchestrator.py`, `pipeline_start.sh`, `check_*.py`, `rescore_rank.py` | nothing — these run on the host |

For iterating on a module you can skip the rebuild entirely: the orchestrators
bind-mount `src/` into the containers at run time, so an edited file takes
effect on the next run. The rebuild is what makes the change permanent in the
image, which matters for reproducibility and for anyone who clones the
repository.

Verify that the Uni-Dock build supports your GPU architecture before anything
else:

```bash
apptainer exec --nv unidock-gpu.sif bash -c \
  'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate docking_env && unidock --help'
```

If this reports `no kernel image is available for execution on the device`, the
packaged binary does not cover your compute capability. Build Uni-Dock from
source in `%post` with `CMAKE_CUDA_ARCHITECTURES` set accordingly.

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

Re-ranking with different consensus weights needs no re-docking: set
`RUN_CONVERSION=false` and `RUN_DOCKING=false`, adjust the `w_*` values in
`config/rescore.ini`, run again.

To resume an interrupted docking run:

```bash
./pipeline_start.sh --restart
```

---

## Configuration

Three configuration surfaces: the switches at the top of `pipeline_start.sh`,
`config/docking.ini`, and `config/rescore.ini`. Stage 1 has no INI; it is
configured through the `CONV_*` variables in the shell script.

Every option is listed below with its default. Options absent from an INI fall
back to the value shown, except where marked required.

### pipeline_start.sh

| Variable | Default | Effect |
|---|---|---|
| `RUN_CONVERSION` | `true` | Run stage 1 |
| `RUN_DOCKING` | `true` | Run stage 2 |
| `RUN_RESCORING` | `true` | Run stage 3 |
| `CHECK_CONFIG` | `true` | Run `check_config.py` before stage 1 |
| `CHECK_LIGANDS` | `true` | Run `check_ligands.py` between stage 1 and 2 |
| `CONTINUE_ON_ERROR` | `false` | Continue after a failed stage instead of aborting |
| `CONV_SIF` | `sdf_to_pdbqt.sif` | Stage 1 container |
| `CONV_LIB_DIR` | `LIB` | Directory scanned for input files |
| `CONV_INPUT_TYPES` | `sdf` | `sdf`, `smiles`, or `all`. Which file types are collected from `LIB/` |
| `CONV_SMILES_COL` | *(empty)* | SMILES column, header name or 0-based index. Empty means autodetect |
| `CONV_NAME_COL` | *(empty)* | Name column, same convention |
| `CONV_OUT_DIR` | `data/PDBQT` | PDBQT output. Must match `[PATHS] pdbqt_dir` |
| `CONV_WORKERS` | `15` | Parallel conversion processes |
| `CONV_TIMEOUT` | `120` | Seconds per molecule before SIGALRM |
| `CONV_UFF_ITERS` | `800` | UFF optimisation steps per molecule |
| `CONV_FLAT` | `false` | `true` writes all PDBQTs into one directory |
| `CONV_SUBDIR_PER_SDF` | `true` | One output subdirectory per input SDF |
| `CONV_RESUME` | `true` | Pass `--skip-existing` to the converter: molecules whose PDBQT already exists are skipped individually |
| `CONV_SKIP_IF_COMPLETE` | `true` | Skip an SDF entirely once converted + failed ≥ molecules in the file |
| `DOCK_SIF` | `unidock-gpu.sif` | Stage 2 container |
| `DOCK_INI` | `config/docking.ini` | Stage 2 configuration |
| `DOCK_RESTART` | `false` | Use `restart_orchestrator.py`; also set by `--restart` |
| `RESCORE_SIF` | `rescoring-gpu.sif` | Stage 3 container |
| `RESCORE_INI` | `config/rescore.ini` | Stage 3 configuration |

Command line: `--dry-run` prints every command without executing it, `--restart`
sets `DOCK_RESTART=true`, `--help` prints the header.

### config/docking.ini

**`[PATHS]`.** All four are required. A missing key fails at startup with
`KeyError`. Relative paths resolve against the project directory.

| Key | Purpose |
|---|---|
| `pdbqt_dir` | Ligand PDBQTs from stage 1. Searched recursively |
| `target_dir` | Receptor PDBQTs and `config.txt` |
| `results_dir` | Poses and per-chunk CSVs. Handover to stage 3 |
| `log_dir` | `pipeline.log`, chunk files, job directories |

**`[GPU]`**

| Key | Default | Effect |
|---|---|---|
| `num_gpus` | `1` | Number of GPUs used. The first N reported by `nvidia-smi` |
| `cuda_device_id` | `0` | Which GPU to use when `num_gpus = 1`. Ignored otherwise |

**`[UNIDOCK]`.** These map directly to Uni-Dock CLI flags.

| Key | Default | Effect |
|---|---|---|
| `binary` | `unidock` | Path or command name of the Uni-Dock executable |
| `search_mode` | `balance` | `fast`, `balance`, or `detail`. Empty string enables `exhaustiveness` and `max_step` |
| `exhaustiveness` | `384` | Monte Carlo runs. Only read when `search_mode` is empty |
| `max_step` | `0` | Steps per run. Only read when `search_mode` is empty; `0` omits the flag |
| `num_modes` | `9` | Poses written per ligand |
| `energy_range` | `3` | kcal/mol window below the best pose |
| `scoring` | `vina` | `vina`, `vinardo`, or `ad4`. `ad4` requires precomputed AutoGrid maps |
| `batch_size` | `1000` | Ligands per Uni-Dock process. Main lever for GPU utilisation |
| `max_gpu_memory` | `0` | MB cap. `0` means no cap. Set to about 75% of VRAM if OOM occurs |
| `timeout` | `7200` | Seconds per sub-batch. `0` disables the timeout |
| `refine_step` | `0` | Uni-Dock refinement steps. `0` omits the flag |
| `seed` | `0` | Random seed. `0` means non-deterministic |

**`[CHUNK]`**

| Key | Default | Effect |
|---|---|---|
| `chunk_size` | `5000` | Ligands per dispatch unit: one job file, one chunk CSV |

`chunk_size` and `batch_size` are different units. A chunk of 5000 runs as five
sub-batches of 1000 inside one worker. Larger chunks reduce dispatch overhead;
smaller chunks distribute the tail of a target more evenly across GPUs.

### config/rescore.ini

**`[PATHS]`.** `target_dir`, `results_dir` and `log_dir`, all required.
`results_dir` and `target_dir` must match the values in `docking.ini`;
`check_config.py` verifies this.

**`[GPU]`.** Same two keys as in `docking.ini`. The orchestrator reads them in
the rescore stage too. Without this section `num_gpus` falls back to 1 and
rescoring runs on a single GPU.

**`[RESCORE]`, scoring functions.** Each is switched on and weighted separately. At least one must be active.

| Key | Default | Score | Direction |
|---|---|---|---|
| `vina_enabled` | `true` | From `REMARK VINA RESULT` in the pose file. No extra cost | smaller better |
| `vinardo_enabled` | `false` | `gnina --score_only --scoring vinardo`. One extra pass per ligand | smaller better |
| `ad4_enabled` | `false` | `gnina --score_only --scoring ad4_scoring`. One extra pass per ligand | smaller better |
| `cnnaffinity_enabled` | `false` | gnina CNNaffinity, predicted pK | larger better |
| `cnnscore_enabled` | `false` | gnina CNNscore, pose quality in [0,1] | larger better |
| `deltalinf9xgb_enabled` | `false` | Lin_F9 + XGBoost, separate conda env | larger better |
| `dense_enabled` | `false` | Second CNN model, adds `dense_cnnaffinity` and `dense_cnnscore` | larger better |

`vina_enabled` reads the value the docking used. With `[UNIDOCK] scoring = vinardo`
the column `score_vina` holds Vinardo values; `check_config.py` warns about this.

**`[RESCORE]`, consensus weights.** All zero means equal weighting over the
active functions. Otherwise weights normalise to 1 over the active set.

| Key | Default |
|---|---|
| `w_vina` | `0.0` |
| `w_vinardo` | `0.0` |
| `w_ad4` | `0.0` |
| `w_cnnaffinity` | `0.0` |
| `w_cnnscore` | `0.0` |
| `w_deltalinf9xgb` | `0.0` |
| `w_dense_cnnaffinity` | `0.0` |
| `w_dense_cnnscore` | `0.0` |

**`[RESCORE]`, execution.**

| Key | Default | Effect |
|---|---|---|
| `enabled` | `true` | Master switch for the whole stage |
| `sigma_fraction` | `4.0` | Decay parameter of the consensus. Larger means sharper decay |
| `cnn_model` | `crossdock_default2018_ensemble` | gnina CNN model |
| `dense_model` | `dense_ensemble` | Second model, used when `dense_enabled` |
| `gnina_binary` | *(empty)* | Path to gnina. Empty falls back to autodetection via PATH |
| `gnina_use_gpu` | `true` | `false` forces `--no_gpu` |
| `n_jobs` | `1` | joblib workers for the CLI path. `-1` uses all cores |
| `rescore_batch_size` | `0` | Poses per GPU batch. `0` derives it from available VRAM |
| `rescore_block_size` | `0` | Ligands per resumable block. `0` disables blocking. See below |
| `cluster_poses` | `false` | RMSD clustering before scoring, reduces the number of poses |
| `cluster_rmsd_cutoff` | `2.0` | Ångström threshold for clustering |
| `deltalinf9xgb_n_workers` | `1` | Scoring workers for ΔLin_F9XGB |
| `deltalinf9xgb_prep_workers` | `0` | MOL2 preparation workers. `0` mirrors `n_workers` |
| `rescore_num_gpus` | `0` | GPUs for rescoring. `0` uses the `[GPU]` setting |
| `rescore_cuda_device_id` | `0` | Which GPU when `rescore_num_gpus = 1` |

**`[REFINEMENT]`.** Runs after rescoring on the best-ranked ligands.

| Key | Default | Effect |
|---|---|---|
| `enabled` | `false` | Master switch |
| `top_fraction` | `0.15` | Fraction of the ECR ranking to refine |
| `refinement_mode` | `local_only` | `local_only`, `minimize`, or `autobox` |
| `autobox_extend` | `4.0` | Ångström padding, only for `refinement_mode = autobox` |
| `cnn_model` | `crossdock_default2018_ensemble` | gnina CNN model |
| `exhaustiveness` | `0` | `0` means local optimisation without a new search |
| `num_modes` | `1` | Poses per refined ligand |
| `use_gpu` | `true` | GPU for the refinement runs |
| `gnina_binary` | *(empty)* | Path to gnina, autodetected when empty |

### Target definition

`TARGET/config.txt` is not an INI. One block per receptor, blocks separated by
blank lines:

```
BRD4
CENTER [12.5, -8.3, 22.1]
BOX_SIZE [25.0, 25.0, 25.0]

CDK2
CENTER [4.1, 15.7, -3.2]
BOX_SIZE [22.0, 22.0, 22.0]
LIGAND_SUBDIR = cdk2_dude
```

| Field | Required | Meaning |
|---|---|---|
| name | yes | First line of the block. Needs a matching `<name>.pdbqt` in `target_dir` |
| `CENTER` | yes | Box centre in Ångström |
| `BOX_SIZE` | yes | Box dimensions in Ångström |
| `LIGAND_SUBDIR` | no | Restricts this target to a subdirectory of `pdbqt_dir` |

Lines starting with `#` are comments. A receptor whose PDBQT is missing produces
a warning and is skipped rather than aborting the run.

## Consensus ranking

Scores from different functions are combined by **exponential consensus ranking**
(Palacio-Rodríguez et al., *Sci Rep* **9**, 5142, 2019). ECR ranks poses per scoring function and
combines the ranks. Scores on incompatible scales, such as a Vina energy in
kcal/mol and a CNN classification score in $[0,1]$, can therefore be combined
without normalisation.

### Step 1 — Direction correction

Every score is normalised so that **smaller is better** before ranking. Functions
that report "larger is better" are negated:

| Score | Raw output | Direction | Stored as |
|---|---|---|---|
| `vina` | kcal/mol | smaller better | $s$ |
| `vinardo` | kcal/mol | smaller better | $s$ |
| `ad4` | kcal/mol | smaller better | $s$ |
| `cnnaffinity` | predicted pK | larger better | $-s$ |
| `cnnscore` | $[0,1]$ | larger better | $-s$ |
| `deltalinf9xgb` | predicted pK$_d$ | larger better | $-s$ |
| `dense_cnnaffinity` | predicted pK | larger better | $-s$ |
| `dense_cnnscore` | $[0,1]$ | larger better | $-s$ |

### Step 2 — Per-score ranking

For each active scoring function $j$, all poses of the target that carry a valid
value are sorted ascending and assigned ranks $r_j \in \{1, 2, \dots\}$, so rank 1
is the best pose under that function. Poses without a value for $j$ receive no
rank and contribute nothing to that term.

### Step 3 — Exponential weighting

$$\mathrm{ECR}_j(\text{pose}) = e^{-r_j / \sigma}, \qquad \sigma = \max\left(\frac{N}{f}, 1\right)$$

with $N$ the total number of poses of the target and $f$ the configurable
`sigma_fraction` (default 4.0). The contribution decays exponentially with rank. A function that places a pose
near the top contributes close to its full weight, one that places it in the
tail contributes close to nothing. Larger $f$ gives smaller $\sigma$ and a
sharper decay, so only the leading ranks carry weight.

### Step 4 — Weighted sum per pose

$$P(\text{pose}) = \sum_{j \in A} w_j \cdot \mathrm{ECR}_j(\text{pose})$$

where $A$ is the set of active functions that actually produced data. Weights are
normalised over $A$ only:

$$w_j = \frac{w_j^{\text{ini}}}{\sum_{k \in A} w_k^{\text{ini}}}
\qquad\text{or}\qquad w_j = \frac{1}{|A|} \;\text{ if all } w^{\text{ini}} = 0$$

Disabling a function therefore does not silently reweight the others, and a
function that was enabled but produced no values at all is dropped from $A$ with
a warning rather than counted as zero.

### Step 5 — Aggregation to ligands

$$P(\text{ligand}) = \max_{\text{pose} \in \text{ligand}} P(\text{pose})$$

The ligand inherits the score of its best pose; that pose index is reported as
`best_pose`. Ligands are sorted by $P$ descending, so **larger ECR is better**,
opposite to the underlying energies.

### Output

| File | Content |
|---|---|
| `rescoring_poses_<target>.csv` | every pose with all raw scores, ranks $r_j$, terms $\mathrm{ECR}_j$ and $P(\text{pose})$ |
| `rescoring_ligands_<target>.csv` | ligand ranking by $P(\text{ligand})$ — the main result |

The pose-level CSV contains all intermediate values, so $P$ can be recomputed
from the columns and the weighting can be changed offline without re-running any
scoring.

### Worked example

Three ligands, two poses each, two active functions at equal weight
($N = 6$, $f = 4$, therefore $\sigma = 1.5$):

| Pose | $s_\text{vina}$ | $s_\text{cnn}$ | $r_\text{vina}$ | $r_\text{cnn}$ | $\mathrm{ECR}_\text{vina}$ | $\mathrm{ECR}_\text{cnn}$ | $P$ |
|---|---|---|---|---|---|---|---|
| ligA/1 | −9.0 | −0.90 | 2 | 2 | 0.2636 | 0.2636 | **0.2636** |
| ligA/2 | −8.0 | −0.70 | 4 | 3 | 0.0695 | 0.1353 | 0.1024 |
| ligB/1 | −9.5 | −0.40 | 1 | 4 | 0.5134 | 0.0695 | **0.2915** |
| ligB/2 | −7.0 | — | 5 | — | 0.0357 | 0.0000 | 0.0178 |
| ligC/1 | −8.5 | −0.95 | 3 | 1 | 0.1353 | 0.5134 | **0.3244** |
| ligC/2 | −6.0 | −0.10 | 6 | 5 | 0.0183 | 0.0357 | 0.0270 |

Final ranking: **ligC** (0.3244) > **ligB** (0.2915) > **ligA** (0.2636).

ligB/1 has the best Vina energy in the set but ranks fourth on CNN, so it loses
to ligC/1, which is third on Vina and first on CNN. A Vina-only ranking would
have placed ligB first. ligB/2 shows the handling of missing values: the absent
CNN score contributes zero to that term, which penalises the pose rather than
treating it as neutral.

### Deviations from the original formulation

Two points where this implementation differs from the paper. Both are
deliberate, but they matter when comparing against published numbers.

- **Ranking happens at pose level, not ligand level.** All poses of all ligands
  of a target compete in one ranking, and the per-ligand value is taken as the
  maximum afterwards. $\sigma$ is therefore derived from the pose count, not the
  ligand count. With `num_modes = 9` that makes $\sigma$ about nine times larger
  than a ligand-level formulation would give, which flattens the decay.
- **Ties are broken arbitrarily** by sort order rather than receiving a shared
  rank. With continuous scores exact ties are rare; with `cnnscore` saturating
  at 1.0 they are not impossible.

---

## Pre-flight checks

Two host-side scripts, both wired into `pipeline_start.sh`:

```bash
python3 src/check_config.py  config/
python3 src/check_ligands.py data/PDBQT
```

`check_ligands.py` covers a failure mode that produces no error message: the
converter derives filenames from SDF titles, so two molecules with the same
title get the same filename. Docking results are written flat per target, so
those ligands overwrite each other and the run ends with a plausible but
incomplete result table. The check aborts before any GPU time is spent.

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
- `parse_target_config` exists twice: in `pipeline_common.py` and as a standalone
  copy inside `docking_rescore.py`. Changes to the `config.txt` format have to be
  made in both.
- No automated test suite.

---

## Repository layout

```
├── pipeline_start.sh          Stage toggles, runs the three stages
├── benchmark.sh               Throughput measurements (GPU scaling, batch sweep)
│
├── config/
│   ├── docking.ini            Stage 2
│   └── rescore.ini            Stage 3
│
├── src/
│   ├── pipeline_common.py     Target parsing, ligand discovery, logging
│   ├── ecr.py                 Consensus ranking, standard library only
│   │
│   ├── sdf_to_pdbqt.py        Stage 1
│   │
│   ├── docking_config.py      Stage 2 config
│   ├── unidock_engine.py      Batched Uni-Dock calls
│   ├── worker_dock.py         Stage 2 worker (in-container)
│   ├── worker_restart_dock.py Stage 2 worker, restart mode
│   ├── orchestrator.py        Chunk dispatch across GPUs (host)
│   ├── restart_orchestrator.py
│   │
│   ├── worker_rescore.py      Stage 3 worker (in-container)
│   ├── docking_rescore.py     Scoring and consensus
│   ├── gnina_refinement.py    Refinement of top-ranked ligands
│   ├── gnina_gpu_worker.py    gninatorch GPU backend
│   ├── linf9xgb_scorer.py     ΔLin_F9XGB backend
│   │
│   ├── check_config.py        Pre-flight: INI consistency
│   ├── check_ligands.py       Pre-flight: library layout and name collisions
│   └── rescore_rank.py        Re-rank from stored scores, no container needed
│
├── build/
│   ├── *.def                  Container definitions
│   ├── gnina                  Binary, download separately (not tracked)
│   ├── featureSASA.py         Patched third-party source
│   └── prepare_betaAtoms.py   Patched third-party source
│
├── docs/
│   ├── pipeline.svg           Diagram used in this README (plus dark variant)
│   ├── make_diagram.py        Regenerates all diagram files
│   └── *.md                   German operating notes
├── LIB/                       Input libraries (not tracked)
├── TARGET/                    Receptor PDBQTs and config.txt
├── data/                      Intermediate PDBQTs and logs (not tracked)
└── RESULTS/                   Poses, CSVs, rescoring output (not tracked)
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
