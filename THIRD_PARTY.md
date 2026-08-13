# Third-Party Components

This pipeline is an orchestration layer. The scientific work is done by external
tools, and two files in this repository are modified copies of third-party source.
Both facts constrain how this repository can be licensed and redistributed.

**Before publishing, verify each license below against the upstream repository.**
The notes here reflect the state at the time of writing and are not legal advice.

---

## Modified third-party source in this repository

These are **derivative works**, not just dependencies. Redistributing them means
complying with the upstream license, including its terms on modification notices.

| File | Origin | Nature of change |
|---|---|---|
| `build/featureSASA.py` | delta_LinF9_XGB (author: Cheng Wang) | `os.system` → `subprocess.run` with timeouts; `mkdir`/`cp`/`cat`/`sed` replaced by native Python; per-call temp directory |
| `build/prepare_betaAtoms.py` | delta_LinF9_XGB | `os.system` → `Popen` with `start_new_session` and process-group kill on timeout; `ADT` path adjusted for the container layout |

Both changes are functional only — identical tool invocations with identical
arguments, differing solely in the invocation mechanism. Output for
non-hanging inputs is bit-identical to the original. The modification notices
are retained as comments at the top of each file.

**Action required:** confirm the delta_LinF9_XGB license permits redistribution
of modified files, and whether it imposes conditions on the license of this
repository. If it is copyleft, that determines your choice; if it is permissive,
retain the copyright notice and state the modifications — which the file headers
already do.

---

## Tools invoked but not redistributed

Installed inside the container images at build time from their official sources.
Not included in this repository.

| Tool | Role | License (verify) | Reference |
|---|---|---|---|
| Uni-Dock | Docking engine (stage 2) | Apache 2.0 since 2025-03-10 | Yu et al., *J Chem Theory Comput* (2023) |
| AutoDock Vina | Scoring function underlying Uni-Dock | Apache 2.0 | Eberhardt et al., *J Chem Inf Model* **61**, 3891 (2021) |
| gnina | CNN rescoring and refinement (stage 3) | verify upstream | McNutt et al., *J Cheminform* **13**, 43 (2021) |
| RDKit | Conformer generation, chemistry toolkit | BSD 3-Clause | rdkit.org |
| Meeko | PDBQT writer | LGPL 2.1 | github.com/forlilab/Meeko |
| Open Babel | Format handling | GPL 2.0 | openbabel.org |
| MSMS | Solvent-accessible surface (ΔLin_F9XGB) | **academic use only** | Sanner et al. (1996) |
| MGLTools / AutoDockTools | Receptor preparation (ΔLin_F9XGB) | LGPL | ccsb.scripps.edu |
| AlphaSpace2 | Beta-atom pockets (ΔLin_F9XGB) | verify upstream | — |
| XGBoost | ΔLin_F9XGB model inference | Apache 2.0 | — |

**MSMS deserves particular attention.** It is distributed for academic use and
its terms do not cover commercial use. It is pulled in by the ΔLin_F9XGB scoring
path (`deltalinf9xgb_enabled = true` in `rescore.ini`). If that matters for your
setting, leave that scoring function disabled — the pipeline runs without it.

---

## The gnina binary

`build/gnina` is deliberately not tracked in git. Users download it from the
[gnina releases page](https://github.com/gnina/gnina/releases). This avoids
redistributing a large third-party binary under unclear terms and keeps the
repository small.

---

## Citation

If results from this pipeline appear in a publication, cite the underlying
methods rather than only this repository — the docking and scoring are theirs:

- **Uni-Dock:** Yu, Y., Cai, C., Wang, J., Bo, Z., Zhu, Z., & Zheng, H. (2023).
  Uni-Dock: GPU-Accelerated Docking Enables Ultralarge Virtual Screening.
  *Journal of Chemical Theory and Computation*.
- **AutoDock Vina 1.2:** Eberhardt, J., Santos-Martins, D., Tillack, A. F., &
  Forli, S. (2021). *J Chem Inf Model* **61**(8), 3891–3898.
- **gnina:** McNutt, A. T., et al. (2021). GNINA 1.0: molecular docking with deep
  learning. *Journal of Cheminformatics* **13**, 43.
- **ΔLin_F9XGB:** Yang, C., & Zhang, Y. (2022). *Journal of Chemical Information
  and Modeling*.
- **Exponential Consensus Ranking:** Palacio-Rodríguez, K., Lans, I.,
  Cavasotto, C. N., & Cossio, P. (2019). *Scientific Reports* **9**, 5142.
