"""
docking_config.py
=================
Konfiguration der Stufe 2: DOCKING (Uni-Dock, GPU).

Liest ausschliesslich die Sektionen [PATHS], [GPU], [UNIDOCK] und [CHUNK].
Kennt weder Rescoring noch Ligandenaufbereitung.

Gegenueber der frueheren Fassung entfallen:
  - der Import von docking_rescore (die Rescoring-Stufe ist eigenstaendig;
    dieser Import hat den Docking-Container gezwungen, docking_rescore.py
    und eine [RESCORE]-Sektion mitzuschleppen, obwohl er nie rescored)
  - das Pflichtfeld rescore_cfg
  - [CPU], vina_cores_per_job, max_parallel_docking_jobs, preparation_cores
    (kein CPU-Fallback mehr – Docking laeuft ausschliesslich auf GPU)
  - [VINA] (exhaustiveness/num_modes/energy_range sind Uni-Dock-Flags und
    stehen jetzt in [UNIDOCK])
  - [CONVERSION] und die force_*-Flags (gehoeren zur Conversion-Stufe)
  - lib_dir, sdf_out_dir, pdb_out_dir (nur die Conversion-Stufe braucht sie)

Gemeinsame Bausteine kommen aus pipeline_common.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline_common import (  # noqa: F401  (Re-Export fuer die Worker)
    PIPELINE_CONFIG_FILE,
    TargetConfig,
    load_ini,
    parse_target_config,
    require,
    setup_logging,
)


@dataclass
class DockingConfig:
    """
    Parameter der Docking-Stufe. Nie manuell instanziieren –
    immer ueber DockingConfig.from_ini() laden.
    """

    # --- Verzeichnisse ---
    log_dir:     Path
    pdbqt_dir:   Path
    target_dir:  Path
    results_dir: Path

    # --- GPU ---
    cuda_device_id: int
    num_gpus:       int

    # --- Uni-Dock ---
    unidock_binary:         Path
    unidock_search_mode:    str    # fast | balance | detail | "" = explizit
    exhaustiveness:         int    # nur wenn search_mode leer
    unidock_max_step:       int    # nur wenn search_mode leer; 0 = weglassen
    num_modes:              int
    energy_range:           int
    unidock_scoring:        str    # vina | vinardo | ad4
    unidock_batch_size:     int    # Liganden pro unidock-Prozess
    unidock_max_gpu_memory: int    # MB, 0 = kein Limit
    unidock_timeout:        int    # Sekunden pro Sub-Batch, 0 = kein Timeout
    unidock_refine_step:    int    # 0 = Uni-Dock Default
    unidock_seed:           int    # 0 = kein fester Seed

    # --- Chunking ---
    chunk_size: int

    # ------------------------------------------------------------------
    @property
    def target_config_file(self) -> Path:
        return self.target_dir / "config.txt"

    @property
    def all_dirs(self) -> list[Path]:
        return [self.log_dir, self.pdbqt_dir,
                self.target_dir, self.results_dir]

    # ------------------------------------------------------------------
    @classmethod
    def from_ini(cls, ini_path: Path = PIPELINE_CONFIG_FILE) -> "DockingConfig":
        p = load_ini(ini_path)

        return cls(
            log_dir=Path(require(p, "PATHS", "log_dir")),
            pdbqt_dir=Path(require(p, "PATHS", "pdbqt_dir")),
            target_dir=Path(require(p, "PATHS", "target_dir")),
            results_dir=Path(require(p, "PATHS", "results_dir")),

            cuda_device_id=p.getint("GPU", "cuda_device_id", fallback=0),
            num_gpus=p.getint("GPU", "num_gpus", fallback=1),

            unidock_binary=Path(
                p.get("UNIDOCK", "binary", fallback="unidock")
            ).expanduser(),
            unidock_search_mode=p.get(
                "UNIDOCK", "search_mode", fallback="balance"
            ).strip(),
            exhaustiveness=p.getint("UNIDOCK", "exhaustiveness", fallback=384),
            unidock_max_step=p.getint("UNIDOCK", "max_step", fallback=0),
            num_modes=p.getint("UNIDOCK", "num_modes", fallback=9),
            energy_range=p.getint("UNIDOCK", "energy_range", fallback=3),
            unidock_scoring=p.get(
                "UNIDOCK", "scoring", fallback="vina"
            ).strip(),
            unidock_batch_size=p.getint("UNIDOCK", "batch_size", fallback=1000),
            unidock_max_gpu_memory=p.getint(
                "UNIDOCK", "max_gpu_memory", fallback=0
            ),
            unidock_timeout=p.getint("UNIDOCK", "timeout", fallback=7200),
            unidock_refine_step=p.getint("UNIDOCK", "refine_step", fallback=0),
            unidock_seed=p.getint("UNIDOCK", "seed", fallback=0),

            chunk_size=p.getint("CHUNK", "chunk_size", fallback=5000),
        )


# Rueckwaertskompatibler Alias: bestehender Code, der `Config` importiert,
# laeuft unveraendert weiter.
Config = DockingConfig
