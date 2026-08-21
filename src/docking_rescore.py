"""
docking_rescore.py
==================
Vereinheitlichtes Rescoring-Modul fuer die Docking-Pipeline.

Dieses Modul wird von docking_config.py importiert und kann auch als
eigenstaendiges Skript ausgefuehrt werden.

Ersetzt die bisherigen getrennten Module:
  - docking_rescore.py       (nur Vina + CNNaffinity)
  - docking_rescore_gnina.py (Vina + CNNaffinity + CNNscore)

Alle Scoring-Funktionen sind jetzt optional ueber pipeline_config.ini steuerbar.

Rescoring-Uebersicht
--------------------
Fuer jeden Liganden (alle Posen aus der _docked.pdbqt) werden bis zu drei
Scoring-Funktionen angewendet (alle aus einem einzigen gnina-Aufruf):

  Score 1 - Vina        : Energie aus REMARK VINA RESULT (kcal/mol)
                          Richtung: kleiner = besser
                          → keine Transformation noetig

  Score 2 - CNNaffinity : gnina CLI (vorhergesagter pK-Wert)
                          Richtung: groesser = bessere Bindung
                          → INVERTIEREN fuer ECR (kleiner = besser)
                          Aktivierung: [RESCORE] cnnaffinity_enabled = true

  Score 3 - CNNscore    : gnina CLI (crossdock_default2018_ensemble)
                          Richtung: groesser = besser (0-1, Klassifikation)
                          → INVERTIEREN fuer ECR (kleiner = besser)
                          Aktivierung: [RESCORE] cnnscore_enabled = true

  Score 4 - ΔLin_F9XGB  : Lin_F9 + XGBoost Δ-Learning (Yang & Zhang 2022)
                          Richtung: groesser = besser (vorhergesagter pKd)
                          → INVERTIEREN fuer ECR (kleiner = besser)
                          Eigenes Conda-Env (linf9xgb_env) im Container.
                          Aktivierung: [RESCORE] deltalinf9xgb_enabled = true

Exponential Consensus Ranking (ECR)
------------------------------------
Referenz: Palacio-Rodriguez et al., Sci Rep 9, 5142 (2019).
          https://doi.org/10.1038/s41598-019-41594-3

Formel Pose-Ebene:
  ecr_j(r)   = exp( -r / sigma )       r    = Rang der Pose bzgl. Score j
  P(pose)    = Summe_j ecr_j(r_pose_j) sigma = N / sigma_fraction

Aggregation auf Liganden-Ebene:
  P(ligand)  = max( P(pose) )  ueber alle Posen des Liganden

Finales Ranking: Liganden absteigend nach P(ligand) sortieren.
Groesserer ECR-Score = besser.

Ausgabedateien (je Target)
--------------------------
./RESULTS/<target>/rescoring_poses_<target>.csv
    Alle Posen mit allen Scores, Raengen und ECR-Beitraegen.

./RESULTS/<target>/rescoring_ligands_<target>.csv
    Liganden-Rangliste sortiert nach ECR (Hauptergebnis).

Konfiguration in pipeline_config.ini
-------------------------------------
[RESCORE]
enabled              = true
sigma_fraction       = 4
cnnaffinity_enabled  = true
cnnscore_enabled     = true
cnn_model            = crossdock_default2018_ensemble
gnina_binary         = /usr/local/bin/gnina
gnina_use_gpu        = true
n_jobs               = -1

Standalone-Aufruf
-----------------
  python docking_rescore.py

Setzt voraus dass docking_pipeline.py bereits ausgefuehrt wurde und
_docked.pdbqt-Dateien in ./RESULTS/<target>/ vorhanden sind.
"""

from __future__ import annotations

import configparser
import csv
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from joblib import Parallel, delayed

import ecr as ecr_mod
import numpy as np

# ---------------------------------------------------------------------------
# ODDT: nicht mehr verwendet (ersetzt durch CNNaffinity aus gnina)
# ---------------------------------------------------------------------------
_ODDT_OK = False

# ---------------------------------------------------------------------------
# gninatorch GPU Worker: auto-detect
# ---------------------------------------------------------------------------
try:
    from gnina_gpu_worker import GninaGPUScorer, GNINATORCH_OK
except ImportError:
    GNINATORCH_OK = False

# ---------------------------------------------------------------------------
# gnina CLI: auto-detect Binary
# ---------------------------------------------------------------------------

def _find_gnina_binary() -> Optional[str]:
    """Sucht das gnina Binary in PATH und bekannten Pfaden."""
    found = shutil.which("gnina")
    if found:
        return found
    for candidate in [
        Path.home() / "gnina",
        Path.home() / "bin" / "gnina",
        Path("/usr/local/bin/gnina"),
        Path("/opt/gnina/gnina"),
        Path(__file__).parent / "gnina",
    ]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None

_GNINA_BIN = _find_gnina_binary()
_GNINA_OK  = _GNINA_BIN is not None


def _resolve_gnina_binary(cfg_binary: str) -> Optional[str]:
    """Gibt konfiguriertes Binary zurueck wenn gesetzt, sonst auto-detected."""
    if cfg_binary and Path(cfg_binary).is_file():
        return cfg_binary
    return _GNINA_BIN


# ======================================================================
# KONFIGURATION
# ======================================================================

PIPELINE_CONFIG_FILE = Path(__file__).parent / "pipeline_config.ini"


@dataclass
class RescoringConfig:
    """
    Rescoring-Parameter aus [RESCORE] in pipeline_config.ini.
    Alle Felder sind optional und haben sinnvolle Standardwerte.

    Scoring-Funktionen (alle aus einem gnina-Aufruf):
      - Vina:         Energie aus PDBQT-Header (immer aktiv)
      - CNNaffinity:  vorhergesagter pK-Wert (Bindungsaffinitaet)
      - CNNscore:     Pose-Qualitaetsklassifikation (0-1)
    """
    enabled:          bool  = True
    sigma_fraction:   float = 4.0
    # Empirische Funktionen (Vina-Familie). vina kommt gratis aus dem
    # PDBQT-Header der Pose; vinardo und ad4 kosten je einen zusaetzlichen
    # gnina --score_only Durchlauf ueber dieselben Posen.
    vina_enabled:        bool = True    # aus REMARK VINA RESULT
    vinardo_enabled:     bool = False   # gnina --scoring vinardo
    ad4_enabled:         bool = False   # gnina --scoring ad4_scoring
    cnnaffinity_enabled: bool = False   # CNNaffinity opt-in
    cnnscore_enabled:    bool = False   # CNNscore opt-in
    deltalinf9xgb_enabled: bool = False  # ΔLin_F9XGB opt-in
    deltalinf9xgb_n_workers: int = 1    # parallele Worker fuer ΔLin_F9XGB-Scoring
    deltalinf9xgb_prep_workers: int = 0 # parallele Worker fuer MOL2-Vorbereitung
                                        # 0 = automatisch = deltalinf9xgb_n_workers
    cnn_model:        str   = 'crossdock_default2018_ensemble'
    gnina_binary:     str   = ''
    gnina_use_gpu:    bool  = True
    gnina_gpu_id:     Optional[int] = None   # GPU-Index fuer gnina (CUDA_VISIBLE_DEVICES)
    n_jobs:           int   = 1
    cluster_poses:       bool  = False  # Pose-Clustering vor Rescoring
    cluster_rmsd_cutoff: float = 2.0    # RMSD-Schwelle in Angstrom
    rescore_batch_size:  int   = 0      # 0 = auto (VRAM-basiert)
    min_block_coverage:  float = 0.99   # Mindestabdeckung je Block
    # Blockgroesse fuer wiederaufnahmefaehiges Rescoring.
    # 0 = aus: alles in einem Durchgang wie bisher, kein Zwischenstand.
    rescore_block_size:  int   = 0
    # Dense-Ensemble (zweites CNN-Modell)
    dense_enabled:       bool  = False
    dense_model:         str   = 'dense_ensemble'
    # ECR-Gewichte (normiert, Summe = 1.0)
    # Default: gleichgewichtet (0.0 = automatisch 1/K)
    w_vina:        float = 0.0
    w_vinardo:     float = 0.0
    w_ad4:         float = 0.0
    w_cnnaffinity: float = 0.0
    w_cnnscore:    float = 0.0
    w_deltalinf9xgb: float = 0.0
    w_dense_cnnaffinity: float = 0.0
    w_dense_cnnscore:    float = 0.0

    @classmethod
    def from_ini(cls, ini_path: Path = PIPELINE_CONFIG_FILE) -> "RescoringConfig":
        """Laedt [RESCORE]-Block aus INI. Gibt Defaults wenn Sektion fehlt."""
        p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        p.read(ini_path, encoding="utf-8")
        s = "RESCORE"
        gnina_bin_raw = p.get(s, "gnina_binary", fallback="").strip()
        gnina_bin = str(Path(gnina_bin_raw).expanduser()) if gnina_bin_raw else ""
        return cls(
            enabled          = p.getboolean(s, "enabled",          fallback=True),
            sigma_fraction   = p.getfloat(  s, "sigma_fraction",   fallback=4.0),
            vina_enabled     = p.getboolean(s, "vina_enabled",     fallback=True),
            vinardo_enabled  = p.getboolean(s, "vinardo_enabled",  fallback=False),
            ad4_enabled      = p.getboolean(s, "ad4_enabled",      fallback=False),
            cnnaffinity_enabled = p.getboolean(s, "cnnaffinity_enabled", fallback=False),
            cnnscore_enabled    = p.getboolean(s, "cnnscore_enabled",    fallback=False),
            deltalinf9xgb_enabled = p.getboolean(s, "deltalinf9xgb_enabled", fallback=False),
            deltalinf9xgb_n_workers = p.getint(s, "deltalinf9xgb_n_workers", fallback=1),
            deltalinf9xgb_prep_workers = p.getint(s, "deltalinf9xgb_prep_workers", fallback=0),
            cnn_model        = p.get(       s, "cnn_model",        fallback="crossdock_default2018_ensemble"),
            gnina_binary     = gnina_bin,
            gnina_use_gpu    = p.getboolean(s, "gnina_use_gpu",    fallback=True),
            gnina_gpu_id     = None,   # Wird zur Laufzeit gesetzt (Orchestrator/ENV)
            n_jobs           = p.getint(    s, "n_jobs",           fallback=1),
            cluster_poses       = p.getboolean(s, "cluster_poses",       fallback=False),
            cluster_rmsd_cutoff = p.getfloat(  s, "cluster_rmsd_cutoff", fallback=2.0),
            rescore_batch_size  = p.getint(    s, "rescore_batch_size",  fallback=0),
            min_block_coverage  = p.getfloat(  s, "min_block_coverage",  fallback=0.99),
            rescore_block_size  = p.getint(    s, "rescore_block_size",  fallback=0),
            dense_enabled       = p.getboolean(s, "dense_enabled",       fallback=False),
            dense_model         = p.get(       s, "dense_model",         fallback="dense_ensemble"),
            w_vina        = p.getfloat(s, "w_vina",        fallback=0.0),
            w_vinardo     = p.getfloat(s, "w_vinardo",     fallback=0.0),
            w_ad4         = p.getfloat(s, "w_ad4",         fallback=0.0),
            w_cnnaffinity = p.getfloat(s, "w_cnnaffinity", fallback=0.0),
            w_cnnscore    = p.getfloat(s, "w_cnnscore",    fallback=0.0),
            w_deltalinf9xgb = p.getfloat(s, "w_deltalinf9xgb", fallback=0.0),
            w_dense_cnnaffinity = p.getfloat(s, "w_dense_cnnaffinity", fallback=0.0),
            w_dense_cnnscore    = p.getfloat(s, "w_dense_cnnscore",    fallback=0.0),
        )

    @property
    def gnina_needed(self) -> bool:
        """Gibt True zurueck wenn mindestens eine gnina-Funktion aktiv ist."""
        return (self.cnnaffinity_enabled or self.cnnscore_enabled
                or self.dense_enabled or self.empirical_extra_needed)

    @property
    def empirical_extra_needed(self) -> bool:
        """True wenn eine empirische Funktion ausser Vina aktiv ist.

        Vina kommt aus dem PDBQT-Header und kostet nichts. Vinardo und AD4
        erfordern je einen eigenen gnina --score_only Durchlauf, weil gnina
        pro Aufruf genau eine --scoring-Funktion auswertet.
        """
        return self.vinardo_enabled or self.ad4_enabled

    @property
    def primary_cnn_needed(self) -> bool:
        """True, wenn ein Ausgang des Primaermodells ins Ranking eingeht."""
        return self.cnnaffinity_enabled or self.cnnscore_enabled

    @property
    def dense_cnn_needed(self) -> bool:
        """True, wenn das Dense-Modell gebraucht wird.

        Eigene Property, damit das Rechnen derselben Bedingung folgt wie
        das Zaehlen. Frueher entschied ueber das Rechnen allein, ob der
        Scorer geladen werden konnte, ueber das Zaehlen dagegen das Flag –
        mit dense_enabled=true und beiden crossdock-Flags auf false liefen
        deshalb zwei Netze statt einem.
        """
        return self.dense_enabled

    @property
    def extra_empirical_functions(self) -> list[tuple[str, str]]:
        """[(ECR-Key, gnina --scoring Name)] fuer die Extra-Durchlaeufe."""
        out = []
        if self.vinardo_enabled:
            out.append(("vinardo", "vinardo"))
        if self.ad4_enabled:
            out.append(("ad4", "ad4_scoring"))
        return out

    def get_ecr_weights(self, active_scores: list[str]) -> dict[str, float]:
        """
        Gibt normierte ECR-Gewichte fuer die aktiven Scoring-Funktionen zurueck.

        Wenn alle Gewichte 0.0 sind (Default): gleichgewichtet (1/K).
        Sonst: normiert auf Summe = 1.0, nur fuer aktive Scores.
        """
        weight_map = {
            "vina":              self.w_vina,
            "vinardo":           self.w_vinardo,
            "ad4":               self.w_ad4,
            "cnnaffinity":       self.w_cnnaffinity,
            "cnnscore":          self.w_cnnscore,
            "deltalinf9xgb":     self.w_deltalinf9xgb,
            "dense_cnnaffinity": self.w_dense_cnnaffinity,
            "dense_cnnscore":    self.w_dense_cnnscore,
        }
        active_weights = {k: weight_map.get(k, 0.0) for k in active_scores}
        w_sum = sum(active_weights.values())

        if w_sum < 1e-9:
            # Alle 0 → gleichgewichtet
            n = len(active_scores)
            return {k: 1.0 / n for k in active_scores}

        # Normieren
        return {k: v / w_sum for k, v in active_weights.items()}


@dataclass
class _TargetInfo:
    """Minimale Target-Informationen fuer das Rescoring-Modul."""
    name:       str
    pdbqt_path: Path
    center:     list
    box_size:   list


# ======================================================================
# DATENKLASSEN
# ======================================================================

@dataclass
class PoseResult:
    """
    Alle Scores einer einzelnen Docking-Pose.

    Konvention nach Richtungskorrektur (immer kleiner = besser):
      score_vina              : native kcal/mol          → unveraendert
      score_vinardo           : Vinardo kcal/mol         → unveraendert
      score_ad4               : AD4 kcal/mol             → unveraendert
      score_cnnaffinity       : -pKd (CNNaffinity)       → invertiert
      score_cnnscore          : -CNNscore (0-1)          → invertiert
      score_deltalinf9xgb     : -pKd (ΔLin_F9XGB)        → invertiert
      score_dense_cnnaffinity : -pKd (dense ensemble)    → invertiert
      score_dense_cnnscore    : -CNNscore (dense ensemble)→ invertiert

    None = Score konnte nicht berechnet werden.
    """
    ligand:        str
    pose:          int            # 1-basiert

    score_vina:              Optional[float] = None
    score_vinardo:           Optional[float] = None
    score_ad4:               Optional[float] = None
    score_cnnaffinity:       Optional[float] = None
    score_cnnscore:          Optional[float] = None
    score_deltalinf9xgb:     Optional[float] = None
    score_dense_cnnaffinity: Optional[float] = None
    score_dense_cnnscore:    Optional[float] = None

    # ECR-Werte (werden durch _compute_ecr befuellt)
    rank_vina:               Optional[int] = None
    rank_vinardo:            Optional[int] = None
    rank_ad4:                Optional[int] = None
    rank_cnnaffinity:        Optional[int] = None
    rank_cnnscore:           Optional[int] = None
    rank_deltalinf9xgb:      Optional[int] = None
    rank_dense_cnnaffinity:  Optional[int] = None
    rank_dense_cnnscore:     Optional[int] = None
    ecr_vina:               float = 0.0
    ecr_vinardo:            float = 0.0
    ecr_ad4:                float = 0.0
    ecr_cnnaffinity:        float = 0.0
    ecr_cnnscore:           float = 0.0
    ecr_deltalinf9xgb:      float = 0.0
    ecr_dense_cnnaffinity:  float = 0.0
    ecr_dense_cnnscore:     float = 0.0
    ecr_total:              float = 0.0


@dataclass
class LigandResult:
    """Aggregierter ECR-Score fuer einen Liganden."""
    ligand:          str
    best_pose:       int
    ecr_score:       float
    ecr_rank:        int   = 0
    score_vina_best:              Optional[float] = None
    score_vinardo_best:           Optional[float] = None
    score_ad4_best:               Optional[float] = None
    score_cnnaffinity_best:       Optional[float] = None
    score_cnnscore_best:          Optional[float] = None
    score_deltalinf9xgb_best:     Optional[float] = None
    score_dense_cnnaffinity_best: Optional[float] = None
    score_dense_cnnscore_best:    Optional[float] = None


# ======================================================================
# SCHRITT 1: VINA-SCORES AUS PDBQT PARSEN
# ======================================================================

def _parse_vina_pdbqt(pdbqt_path: Path) -> list[tuple[int, float]]:
    """
    Liest alle Posen und Vina-Scores aus einer _docked.pdbqt-Datei.

    Vina schreibt pro Pose eine Zeile der Form:
      REMARK VINA RESULT:    -7.520      0.000      0.000

    Rueckgabe: [(pose_1basiert, energie_kcal_mol), ...]
    """
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return []

    poses:   list[tuple[int, float]] = []
    counter  = 0
    pattern  = re.compile(r"REMARK VINA RESULT:\s+([-\d.]+)")

    with open(pdbqt_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                counter += 1
                try:
                    poses.append((counter, float(m.group(1))))
                except ValueError:
                    pass

    return poses


# ======================================================================
# SCHRITT 1b: POSE-CLUSTERING (optional, vor Rescoring)
# ======================================================================


def _parse_pdbqt_heavy_atoms(pdbqt_path: Path) -> list[np.ndarray]:
    """
    Liest die Schwer-Atom-Koordinaten (nicht H) fuer jede Pose aus
    einer Multi-Pose _docked.pdbqt-Datei.

    Vina-PDBQT-Format:
      MODEL 1
      ATOM      1  C1  LIG ...  x  y  z  ...
      ...
      ENDMDL
      MODEL 2
      ...

    HETATM-Zeilen werden ebenfalls beruecksichtigt.
    Wasserstoffatome (Element H, HD, HS) werden uebersprungen.

    Rueckgabe: [np.array([[x,y,z], ...]), ...] pro Pose.
    Leere Liste wenn Datei fehlt oder keine Posen gefunden.
    """
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return []

    all_poses: list[list[list[float]]] = []
    current_coords: list[list[float]] = []
    in_model = False
    hydrogen_elements = {"H", "HD", "HS"}

    with open(pdbqt_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                in_model = True
                current_coords = []
                continue
            if line.startswith("ENDMDL"):
                if current_coords:
                    all_poses.append(current_coords)
                current_coords = []
                in_model = False
                continue
            if line.startswith(("ATOM", "HETATM")):
                # Element-Typ aus Spalte 77-78 (PDBQT) oder Atomname
                # In PDBQT steht der AutoDock-Atomtyp ab Spalte 77
                element = line[77:79].strip() if len(line) > 78 else ""
                if not element:
                    # Fallback: Atomname (Spalte 12-16), erstes Non-Digit
                    atom_name = line[12:16].strip()
                    element = "".join(c for c in atom_name if c.isalpha())
                if element in hydrogen_elements:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current_coords.append([x, y, z])
                except (ValueError, IndexError):
                    pass

    # Falls keine MODEL/ENDMDL-Records: gesamte Datei als eine Pose
    if not all_poses and current_coords:
        all_poses.append(current_coords)

    return [np.array(coords) for coords in all_poses if coords]


def _compute_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Berechnet den RMSD zwischen zwei Koordinaten-Arrays.

    Keine Superposition noetig, da alle Docking-Posen bereits im
    Rezeptor-Koordinatensystem liegen.

    Bei unterschiedlicher Atomanzahl (z.B. durch fehlende Atome in
    einer Pose) wird auf die kuerzere Laenge getrimmt.
    """
    n = min(len(coords_a), len(coords_b))
    if n == 0:
        return float("inf")
    diff = coords_a[:n] - coords_b[:n]
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _cluster_poses_leader(
    vina_poses:  list[tuple[int, float]],
    all_coords:  list[np.ndarray],
    rmsd_cutoff: float,
) -> list[int]:
    """
    Leader-Algorithmus: Clustert Posen nach RMSD.

    Posen sind nach Vina-Score sortiert (beste zuerst, da Vina das so
    ausgibt). Die erste Pose wird immer Cluster-Leader. Jede weitere
    Pose wird dem ersten Leader zugeordnet, dessen RMSD < cutoff ist.
    Wenn kein Leader passt, wird die Pose selbst ein neuer Leader.

    Rueckgabe: Liste der 1-basierten Pose-Indizes der Cluster-Leader
    (= Repraesentanten mit bestem Vina-Score pro Cluster).
    """
    if not all_coords or not vina_poses:
        return [p[0] for p in vina_poses]  # Alle behalten

    # Posen nach Vina-Score sortieren (kleinster/bester zuerst)
    sorted_poses = sorted(
        zip(vina_poses, all_coords),
        key=lambda x: x[0][1]  # Vina-Score
    )

    leaders: list[tuple[int, np.ndarray]] = []  # (pose_idx, coords)

    for (pose_idx, _vina_score), coords in sorted_poses:
        is_new_leader = True
        for _leader_idx, leader_coords in leaders:
            if _compute_rmsd(coords, leader_coords) < rmsd_cutoff:
                is_new_leader = False
                break
        if is_new_leader:
            leaders.append((pose_idx, coords))

    return [idx for idx, _ in leaders]


def _write_filtered_pdbqt(
    pdbqt_path:     Path,
    keep_poses:     set[int],
    output_path:    Path,
) -> None:
    """
    Schreibt eine gefilterte PDBQT-Datei, die nur die angegebenen
    Posen (1-basiert) enthaelt. MODEL/ENDMDL-Records bleiben erhalten.
    """
    current_model = 0
    writing = False

    with open(pdbqt_path, encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("MODEL"):
                current_model += 1
                writing = current_model in keep_poses
                if writing:
                    fout.write(line)
                continue
            if line.startswith("ENDMDL"):
                if writing:
                    fout.write(line)
                writing = False
                continue
            if writing or current_model == 0:
                # current_model == 0: kein MODEL-Record (Einzelpose)
                fout.write(line)


def _apply_pose_clustering(
    docked_path: Path,
    vina_poses:  list[tuple[int, float]],
    rmsd_cutoff: float,
    logger:      Optional[logging.Logger] = None,
) -> tuple[Path, list[tuple[int, float]]]:
    """
    Fuehrt Pose-Clustering auf einer _docked.pdbqt-Datei durch.

    Ablauf:
      1. Schwer-Atom-Koordinaten aller Posen lesen
      2. Leader-Clustering nach RMSD
      3. Gefilterte PDBQT in temp-Datei schreiben
      4. Vina-Poses-Liste auf Cluster-Repraesentanten reduzieren

    Rueckgabe: (gefilterte_pdbqt_path, gefilterte_vina_poses)
    Bei Fehler oder nur 1 Pose: Original zurueckgeben.
    """
    if len(vina_poses) <= 1:
        return docked_path, vina_poses

    all_coords = _parse_pdbqt_heavy_atoms(docked_path)

    # Wenn Koordinaten-Parsing fehlschlaegt: alle Posen behalten
    if len(all_coords) != len(vina_poses):
        if logger:
            logger.debug("    Clustering: Koordinaten-Mismatch (%d coords vs %d poses) "
                         "fuer %s – ueberspringe.", len(all_coords), len(vina_poses),
                         docked_path.name)
        return docked_path, vina_poses

    # Clustering
    leader_indices = _cluster_poses_leader(vina_poses, all_coords, rmsd_cutoff)
    keep_set = set(leader_indices)

    if len(keep_set) == len(vina_poses):
        # Alle Posen sind Leader → kein Filtering noetig
        return docked_path, vina_poses

    # Gefilterte PDBQT schreiben
    import tempfile
    fd, tmp_path = tempfile.mkstemp(
        suffix=".pdbqt",
        prefix=f"{docked_path.stem}_clustered_"
    )
    os.close(fd)  # Dateihandle schliessen, _write_filtered_pdbqt oeffnet selbst
    filtered = Path(tmp_path)
    _write_filtered_pdbqt(docked_path, keep_set, filtered)

    # Vina-Poses filtern + neu nummerieren fuer konsistente Zuordnung
    filtered_vina = [(new_idx, score)
                     for new_idx, (old_idx, score)
                     in enumerate(
                         ((idx, s) for idx, s in vina_poses if idx in keep_set),
                         start=1
                     )]

    if logger:
        logger.debug("    Clustering: %s – %d→%d Posen (cutoff=%.1f Å)",
                     docked_path.name, len(vina_poses), len(filtered_vina),
                     rmsd_cutoff)

    return filtered, filtered_vina


# ======================================================================
# SCHRITT 2: GNINA CNN SCORING (BATCH – ein Aufruf pro Ligand)
# ======================================================================

def _extract_poses_from_pdbqt(lines: list[str]) -> list[list[str]]:
    """
    Splittet eine Multi-Pose PDBQT (Vina-GPU+ Format mit MODEL/ENDMDL)
    in einzelne Posen auf.

    gnina 1.3.2 akzeptiert im --score_only Modus nur Einzelposen:
      - KEIN MODEL, KEIN ENDMDL  (→ "Unknown or inappropriate tag")
      - TORSDOF muss vorhanden bleiben (→ sonst "Missing TORSDOF")

    Jede zurueckgegebene Pose enthaelt:
      - Alle REMARK/ROOT/HETATM/ATOM/BRANCH/ENDBRANCH/ENDROOT-Zeilen
      - TORSDOF N  (von gnina benoetigt, wird behalten)
      - KEIN MODEL, KEIN ENDMDL

    Falls keine MODEL-Records vorhanden: gesamte Datei als eine Pose.
    """
    poses: list[list[str]] = []
    current: list[str] = []
    in_model = False

    for line in lines:
        if line.startswith("MODEL"):
            in_model = True
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                poses.append(current)
            current = []
            in_model = False
            continue
        if in_model:
            current.append(line)   # TORSDOF-Zeilen werden mitgenommen

    # Keine MODEL/ENDMDL-Records: gesamte Datei als eine Pose
    if not poses and current:
        poses.append(current)

    return poses


def _parse_gnina_single_pose_output(output: str) -> Optional[dict[str, Optional[float]]]:
    """
    Parst den gnina --score_only Output fuer eine einzelne Pose.

    gnina gibt fuer Einzelposen aus:
      Affinity: -6.48158 (kcal/mol)
      CNNscore: 0.51952
      CNNaffinity: 6.39816
      CNNvariance: 0.02164
    """
    cnnscore    = None
    cnnaffinity = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("CNNscore:"):
            try:
                cnnscore = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("CNNaffinity:"):
            try:
                cnnaffinity = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    if cnnscore is not None and 0.0 <= cnnscore <= 1.0:
        return {"cnnscore": cnnscore, "cnnaffinity": cnnaffinity}
    return None


def _parse_gnina_batch_output(output: str, n_poses: int) -> list[Optional[dict[str, Optional[float]]]]:
    """
    Parst den gnina --score_only Output fuer mehrere Liganden (Batch-Modus).

    gnina gibt bei --score_only mit --cnn_scoring rescore pro Ligand aus:
      Affinity: -6.48158 (kcal/mol)
      CNNscore: 0.51952
      CNNaffinity: 6.39816
      CNNvariance: 0.02164

    Mehrere Liganden werden nacheinander ausgegeben, getrennt durch die
    jeweiligen Scoring-Header (## Name ...) oder Affinity-Zeilen.

    Rueckgabe: Liste mit einem Dict pro Pose (oder None bei Parse-Fehler).
    """
    results: list[Optional[dict[str, Optional[float]]]] = []
    current_cnnscore:    Optional[float] = None
    current_cnnaffinity: Optional[float] = None
    in_pose = False

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Affinity:"):
            # Neue Pose beginnt – vorherige abschliessen
            if in_pose:
                if current_cnnscore is not None and 0.0 <= current_cnnscore <= 1.0:
                    results.append({"cnnscore": current_cnnscore, "cnnaffinity": current_cnnaffinity})
                else:
                    results.append(None)
            in_pose = True
            current_cnnscore = None
            current_cnnaffinity = None
        elif line.startswith("CNNscore:"):
            try:
                current_cnnscore = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("CNNaffinity:"):
            try:
                current_cnnaffinity = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    # Letzte Pose abschliessen
    if in_pose:
        if current_cnnscore is not None and 0.0 <= current_cnnscore <= 1.0:
            results.append({"cnnscore": current_cnnscore, "cnnaffinity": current_cnnaffinity})
        else:
            results.append(None)

    return results


def _parse_gnina_affinity_batch(output: str) -> list[Optional[float]]:
    """
    Parst NUR die Affinity-Zeilen eines gnina --score_only Batch-Laufs.

    Wird fuer die empirischen Extra-Durchlaeufe (Vinardo, AD4) gebraucht:
    dort ist CNN abgeschaltet, es gibt also keine CNNscore-Zeile, an der
    sich _parse_gnina_batch_output orientieren koennte.

    gnina gibt pro Ligand aus:
      Affinity: -6.48158 (kcal/mol)
    oder bei minimierten Laeufen zwei Werte (Score, RMSD-Term) – wir
    nehmen den ersten.

    Rueckgabe: eine Liste in Eingabereihenfolge, None wo nicht parsebar.
    """
    values: list[Optional[float]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("Affinity:"):
            continue
        try:
            rest = line.split(":", 1)[1].strip()
            values.append(float(rest.split()[0]))
        except (ValueError, IndexError):
            values.append(None)
    return values


def _score_gnina_empirical(
    docked_pdbqt:  Path,
    protein_pdbqt: Path,
    scoring_fn:    str,
    gnina_binary:  Optional[str] = None,
    gpu_id:        Optional[int] = None,
) -> dict[int, Optional[float]]:
    """
    Bewertet alle Posen eines Liganden mit EINER empirischen gnina-Funktion.

    gnina wertet pro Aufruf genau eine --scoring-Funktion aus, deshalb ein
    Durchlauf je aktivierter Funktion. CNN ist hier explizit abgeschaltet
    (--cnn_scoring none), das macht den Durchlauf billig und CPU-bound.

    scoring_fn: gnina-Name, z.B. "vinardo" oder "ad4_scoring".

    Rueckgabe: {pose_1basiert: kcal/mol oder None}
    """
    bin_path = (gnina_binary if gnina_binary and Path(gnina_binary).is_file()
                else _GNINA_BIN)
    if not bin_path or not docked_pdbqt.exists() or not protein_pdbqt.exists():
        return {}

    import tempfile

    _tmp_dir: Optional[str] = None
    try:
        with open(docked_pdbqt, encoding="utf-8", errors="replace") as fin:
            lines_raw = fin.readlines()

        poses = _extract_poses_from_pdbqt(lines_raw)
        if not poses:
            return {}

        env = os.environ.copy()
        if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        _tmp_dir = tempfile.mkdtemp(
            prefix=f"gnina_{scoring_fn}_{docked_pdbqt.stem}_"
        )
        tmp_files: list[Path] = []
        for pose_idx, pose_lines in enumerate(poses, start=1):
            tmp_file = Path(_tmp_dir) / f"pose_{pose_idx}.pdbqt"
            with open(tmp_file, "w", encoding="utf-8") as fout:
                for l in pose_lines:
                    fout.write(l)
            tmp_files.append(tmp_file)

        cmd = [bin_path, "--score_only", "-r", str(protein_pdbqt)]
        for tmp_file in tmp_files:
            cmd.extend(["-l", str(tmp_file)])
        cmd.extend(["--scoring", scoring_fn, "--cnn_scoring", "none"])

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30 + 5 * len(poses), env=env,
        )
        if result.returncode != 0:
            return {}

        values = _parse_gnina_affinity_batch(result.stdout)
        return {
            idx: values[idx - 1] if idx - 1 < len(values) else None
            for idx in range(1, len(poses) + 1)
        }

    except (subprocess.TimeoutExpired, OSError, ValueError):
        return {}
    finally:
        if _tmp_dir:
            shutil.rmtree(_tmp_dir, ignore_errors=True)


def _score_gnina_ligand(
    docked_pdbqt:  Path,
    protein_pdbqt: Path,
    cnn_model:     str,
    use_gpu:       bool = True,
    gnina_binary:  Optional[str] = None,
    gpu_id:        Optional[int] = None,
) -> dict[int, dict[str, Optional[float]]]:
    """
    Berechnet CNNscore und CNNaffinity fuer ALLE Posen eines Liganden.

    Batch-Modus: Alle Posen werden als einzelne Temp-Dateien geschrieben
    und in EINEM gnina-Aufruf mit multiplen -l Flags gescoret.

    gnina 1.3.2 Kompatibilitaet (verifiziert per Quelltextanalyse):
      Der PDBQT-Ligand-Parser erkennt nur: REMARK, ROOT, ENDROOT, BRANCH,
      ENDBRANCH, ATOM, HETATM, TORSDOF, TER, WARNING, USER.
      MODEL/ENDMDL → "Unknown or inappropriate tag".
      → Jede Pose muss als separate Datei vorliegen.
      → Multiple -l Flags erlauben Batch-Scoring in einem Prozessaufruf.

    Fallback: Wenn der Batch-Modus fehlschlaegt, werden die Posen
    einzeln gescoret (wie der bisherige Single-Pose-Workaround).

    Parameter gpu_id: wenn gesetzt und CUDA_VISIBLE_DEVICES NICHT bereits
    in der Umgebung steht, wird CUDA_VISIBLE_DEVICES auf diesen Index
    gesetzt. Wenn CUDA_VISIBLE_DEVICES bereits gesetzt ist (z.B. durch
    den Orchestrator/Container), wird es NICHT ueberschrieben, da CUDA
    GPUs remappt (GPU 1 wird zu Device 0 im Container).

    Rueckgabe: {pose_1basiert: {"cnnscore": float|None, "cnnaffinity": float|None}}
    Leeres Dict bei Fehler oder fehlendem gnina Binary.
    """
    bin_path = gnina_binary if gnina_binary and Path(gnina_binary).is_file() else _GNINA_BIN
    if not bin_path or not docked_pdbqt.exists() or not protein_pdbqt.exists():
        return {}

    results: dict[int, dict[str, Optional[float]]] = {}
    _tmp_files: list[Path] = []
    _tmp_dir: Optional[str] = None

    try:
        import tempfile

        with open(docked_pdbqt, encoding="utf-8", errors="replace") as fin:
            lines_raw = fin.readlines()

        # Posen extrahieren (ohne MODEL/ENDMDL, TORSDOF bleibt)
        poses = _extract_poses_from_pdbqt(lines_raw)
        if not poses:
            return {}

        # GPU-Umgebung vorbereiten
        env = os.environ.copy()
        if gpu_id is not None and use_gpu:
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Alle Posen als einzelne Temp-Dateien schreiben
        _tmp_dir = tempfile.mkdtemp(prefix=f"gnina_{docked_pdbqt.stem}_")
        for pose_idx, pose_lines in enumerate(poses, start=1):
            tmp_file = Path(_tmp_dir) / f"pose_{pose_idx}.pdbqt"
            with open(tmp_file, "w", encoding="utf-8") as fout:
                for l in pose_lines:
                    fout.write(l)
            _tmp_files.append(tmp_file)

        # ── Batch-Modus: ein gnina-Aufruf mit multiplen -l Flags ──
        cmd = [
            bin_path,
            "--score_only",
            "-r", str(protein_pdbqt),
        ]
        for tmp_file in _tmp_files:
            cmd.extend(["-l", str(tmp_file)])
        cmd.extend([
            "--cnn", cnn_model,
            "--cnn_scoring", "rescore",
        ])
        if not use_gpu:
            cmd.append("--no_gpu")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30 + 15 * len(poses),  # skaliert mit Posenanzahl
                env=env,
            )
            output = result.stdout + result.stderr

            if result.returncode == 0:
                # Batch-Parsing
                parsed_list = _parse_gnina_batch_output(output, len(poses))
                for pose_idx, parsed in enumerate(parsed_list, start=1):
                    if parsed is not None:
                        results[pose_idx] = parsed

                # Erfolgreich? Pruefen ob alle Posen gescored wurden
                if len(results) == len(poses):
                    return results

                # Teilweise erfolgreich – fehlende Posen einzeln nachscoren
                missing = [i for i in range(1, len(poses) + 1) if i not in results]
                if missing:
                    _log = logging.getLogger("docking_rescore")
                    _log.debug("    Batch-Scoring: %d/%d Posen erfolgreich, "
                               "%d fehlende werden einzeln nachgescored (%s)",
                               len(results), len(poses), len(missing),
                               docked_pdbqt.name)
                    for pose_idx in missing:
                        single = _score_gnina_single_pose(
                            _tmp_files[pose_idx - 1], protein_pdbqt,
                            cnn_model, use_gpu, bin_path, env,
                        )
                        if single is not None:
                            results[pose_idx] = single
                return results

        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        # ── Fallback: einzelne Posen scoren (alter Workaround) ──────
        _log = logging.getLogger("docking_rescore")
        _log.debug("    Batch-Scoring fehlgeschlagen fuer %s – "
                    "Fallback auf Einzel-Scoring", docked_pdbqt.name)

        for pose_idx, tmp_file in enumerate(_tmp_files, start=1):
            single = _score_gnina_single_pose(
                tmp_file, protein_pdbqt, cnn_model, use_gpu, bin_path, env,
            )
            if single is not None:
                results[pose_idx] = single

    except Exception:
        pass
    finally:
        # Temp-Verzeichnis komplett aufraeumen
        if _tmp_dir is not None:
            try:
                shutil.rmtree(_tmp_dir, ignore_errors=True)
            except Exception:
                pass

    return results


def _score_gnina_single_pose(
    pose_file:     Path,
    protein_pdbqt: Path,
    cnn_model:     str,
    use_gpu:       bool,
    bin_path:      str,
    env:           dict,
) -> Optional[dict[str, Optional[float]]]:
    """
    Scored eine einzelne Pose-Datei mit gnina (Fallback-Methode).

    Wird nur aufgerufen wenn der Batch-Modus fehlschlaegt oder
    einzelne Posen im Batch-Output fehlen.
    """
    cmd = [
        bin_path,
        "--score_only",
        "-r", str(protein_pdbqt),
        "-l", str(pose_file),
        "--cnn", cnn_model,
        "--cnn_scoring", "rescore",
    ]
    if not use_gpu:
        cmd.append("--no_gpu")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, env=env,
        )
        output = result.stdout + result.stderr
        return _parse_gnina_single_pose_output(output)
    except (subprocess.TimeoutExpired, Exception):
        return None


# ======================================================================
# SCHRITT 4: ECR-BERECHNUNG (Pose-Ebene)
# ======================================================================

def _compute_ecr(
    poses:          list[PoseResult],
    sigma_fraction: float,
    active_scores:  list[str],
    weights:        Optional[dict[str, float]] = None,
) -> list[PoseResult]:
    """
    Duenner Wrapper um ecr.compute_ecr().

    Die eigentliche Rechnung liegt in ecr.py, damit sie ohne numpy, joblib
    und PyTorch auskommt und auch auf dem Host laufen kann (rescore_rank.py
    rankt damit ohne Container neu). Die Signatur bleibt unveraendert,
    bestehende Aufrufe funktionieren weiter.
    """
    return ecr_mod.compute_ecr(poses, sigma_fraction, active_scores, weights)


# ======================================================================
# SCHRITT 5: AGGREGATION AUF LIGANDENEBENE
# ======================================================================

def _aggregate_ligands(poses: list[PoseResult]) -> list[LigandResult]:
    """
    P(ligand) = max(ecr_total) ueber alle Posen des Liganden.

    Rueckgabe: nach ecr_score absteigend sortiert, mit Rang.
    """
    by_lig: dict[str, list[PoseResult]] = {}
    for p in poses:
        by_lig.setdefault(p.ligand, []).append(p)

    results: list[LigandResult] = []
    for name, ps in by_lig.items():
        best = max(ps, key=lambda p: p.ecr_total)
        results.append(LigandResult(
            ligand                        = name,
            best_pose                     = best.pose,
            ecr_score                     = best.ecr_total,
            score_vina_best               = best.score_vina,
            score_vinardo_best            = best.score_vinardo,
            score_ad4_best                = best.score_ad4,
            score_cnnaffinity_best        = best.score_cnnaffinity,
            score_cnnscore_best           = best.score_cnnscore,
            score_deltalinf9xgb_best      = best.score_deltalinf9xgb,
            score_dense_cnnaffinity_best  = best.score_dense_cnnaffinity,
            score_dense_cnnscore_best     = best.score_dense_cnnscore,
        ))

    results.sort(key=lambda r: r.ecr_score, reverse=True)
    for i, lr in enumerate(results, 1):
        lr.ecr_rank = i

    return results


# ======================================================================
# HILFSFUNKTION: _score_one_ligand() fuer parallele Ausfuehrung
# ======================================================================

def _score_empirical_one_ligand(
    docked:        Path,
    protein_pdbqt: Path,
    functions:     list,
    gnina_binary:  Optional[str] = None,
    gpu_id:        Optional[int] = None,
) -> dict:
    """
    Worker fuer den nachgezogenen empirischen Durchlauf.

    Rueckgabe: {(ligand_name, pose_idx): {ecr_key: wert|None}}
    Der Schluessel ist bewusst (Ligand, Pose), damit das Ergebnis ohne
    Reihenfolgeannahme in die bestehenden PoseResults gemischt werden kann.
    """
    ligand_name = docked.stem.removesuffix("_docked")
    out: dict = {}
    for ecr_key, gnina_name in functions:
        score_map = _score_gnina_empirical(
            docked, protein_pdbqt, gnina_name, gnina_binary, gpu_id,
        )
        for pose_idx, value in score_map.items():
            out.setdefault((ligand_name, pose_idx), {})[ecr_key] = value
    return out


def _score_one_ligand(
    docked:        Path,
    protein_pdbqt: Path,
    do_cnn:        bool,
    cnn_model:     str,
    gnina_use_gpu: bool = True,
    gnina_binary:  Optional[str] = None,
    do_cluster:    bool = False,
    cluster_rmsd:  float = 2.0,
    gpu_id:        Optional[int] = None,
    extra_empirical: Optional[list] = None,
) -> list:
    """
    Worker-Funktion fuer paralleles Scoring eines Liganden.
    - Pose-Clustering: optional, vor dem Scoring (reduziert Scoring-Aufwand)
    - CNNscore + CNNaffinity: aus einem gnina-Aufruf pro Ligand
    - extra_empirical: [(ecr_key, gnina_scoring_name)] – je Eintrag ein
      zusaetzlicher gnina --score_only Durchlauf (Vinardo, AD4).
      Vina selbst steht bereits im PDBQT-Header und kostet nichts.
    """
    import logging
    _log = logging.getLogger("docking_rescore")

    ligand_name = docked.stem.removesuffix("_docked")
    vina_poses  = _parse_vina_pdbqt(docked)
    if not vina_poses:
        return []

    # ── Pose-Clustering (optional) ────────────────────────────────────
    score_path = docked
    temp_file  = None
    if do_cluster and len(vina_poses) > 1:
        score_path, vina_poses = _apply_pose_clustering(
            docked, vina_poses, cluster_rmsd, _log
        )
        if score_path != docked:
            temp_file = score_path

    # ── CNNscore + CNNaffinity via gnina CLI ──────────────────────────
    gnina_map: dict[int, dict[str, Optional[float]]] = {}
    if do_cnn and _GNINA_OK:
        gnina_map = _score_gnina_ligand(
            score_path, protein_pdbqt, cnn_model, gnina_use_gpu, gnina_binary,
            gpu_id,
        )

    # ── Empirische Extra-Funktionen (Vinardo, AD4) ────────────────────
    empirical_maps: dict = {}
    if extra_empirical and _GNINA_OK:
        for ecr_key, gnina_name in extra_empirical:
            empirical_maps[ecr_key] = _score_gnina_empirical(
                score_path, protein_pdbqt, gnina_name, gnina_binary, gpu_id,
            )

    # ── Temp-Datei aufraeumen ──────────────────────────────────────────
    if temp_file is not None:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass

    # ── Ergebnisse zusammenfuehren ─────────────────────────────────────
    poses = []
    for pose_idx, vina_score in vina_poses:
        gnina_entry = gnina_map.get(pose_idx, {})
        cnn_raw = gnina_entry.get("cnnscore")
        aff_raw = gnina_entry.get("cnnaffinity")
        poses.append(PoseResult(
            ligand            = ligand_name,
            pose              = pose_idx,
            score_vina        = vina_score,
            # Empirische Scores sind bereits richtungsrichtig
            # (kleiner = besser), daher keine Invertierung.
            score_vinardo     = empirical_maps.get("vinardo", {}).get(pose_idx),
            score_ad4         = empirical_maps.get("ad4", {}).get(pose_idx),
            score_cnnaffinity = (-aff_raw) if aff_raw is not None else None,
            score_cnnscore    = (-cnn_raw) if cnn_raw is not None else None,
        ))
    return poses


# ======================================================================
# ΔLin_F9XGB SCORING (neu)
# ======================================================================

def _resolve_obabel_binary() -> str:
    """
    Findet das obabel-CLI-Binary.

    Priorisierung:
      1. Umgebungsvariable OBABEL_BIN (falls explizit gesetzt)
      2. obabel aus dem linf9xgb_env (kommt mit openbabel=3.1.0)
      3. Fallback: 'obabel' im PATH (klappt nur wenn System-Paket installiert)
    """
    explicit = os.environ.get("OBABEL_BIN")
    if explicit and Path(explicit).exists():
        return explicit
    candidate = "/opt/miniconda3/envs/linf9xgb_env/bin/obabel"
    if Path(candidate).exists():
        return candidate
    return "obabel"  # PATH-Fallback


_OBABEL_BIN = _resolve_obabel_binary()


def _convert_pdbqt_receptor_to_pdb(
    receptor_pdbqt: Path,
    out_dir:        Path,
    logger:         logging.Logger,
) -> Optional[Path]:
    """
    Konvertiert den Rezeptor PDBQT → PDB fuer ΔLin_F9XGB.
    Pro Target nur einmal aufrufen (Ergebnis im out_dir cachen).
    """
    out_pdb = out_dir / f"{receptor_pdbqt.stem}.pdb"
    if out_pdb.exists() and out_pdb.stat().st_size > 0:
        return out_pdb

    try:
        proc = subprocess.run(
            [_OBABEL_BIN, str(receptor_pdbqt), "-O", str(out_pdb)],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0 or not out_pdb.exists():
            logger.warning(
                "  obabel PDBQT→PDB fehlgeschlagen fuer %s: %s",
                receptor_pdbqt.name, proc.stderr[:200],
            )
            return None
        return out_pdb
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("  obabel-Fehler bei Receptor-Konvertierung: %s", exc)
        return None


def _extract_pose_to_mol2(
    docked_pdbqt: Path,
    pose_idx:     int,    # 1-basiert
    out_mol2:     Path,
) -> bool:
    """
    Extrahiert Pose `pose_idx` aus der Multi-Pose _docked.pdbqt-Datei
    und schreibt sie als MOL2 fuer ΔLin_F9XGB.

    Rueckgabe: True bei Erfolg, False sonst.

    Hinweis: Fuer Batch-Vorbereitung mehrerer Posen eines Liganden ist
    `_prepare_ligand_mol2s_worker` deutlich effizienter, weil dort das
    PDBQT nur einmal gelesen wird.
    """
    # Pose isolieren (eigenes PDBQT in Temp-Datei)
    try:
        with open(docked_pdbqt, encoding="utf-8", errors="replace") as fh:
            lines_raw = fh.readlines()
    except OSError:
        return False

    pose_blocks = _extract_poses_from_pdbqt(lines_raw)
    if pose_idx < 1 or pose_idx > len(pose_blocks):
        return False

    # Single-Pose PDBQT in Temp-Datei
    tmp_pdbqt = out_mol2.with_suffix(".pdbqt")
    try:
        with open(tmp_pdbqt, "w", encoding="utf-8") as fh:
            fh.writelines(pose_blocks[pose_idx - 1])

        # Konvertierung via obabel
        proc = subprocess.run(
            [_OBABEL_BIN, str(tmp_pdbqt), "-O", str(out_mol2)],
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode == 0 and out_mol2.exists() and out_mol2.stat().st_size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    finally:
        try:
            tmp_pdbqt.unlink(missing_ok=True)
        except Exception:
            pass


# ----------------------------------------------------------------------
# Parallele Pose-Vorbereitung (ProcessPoolExecutor)
# ----------------------------------------------------------------------
#
# `_prepare_ligand_mol2s_worker` ist BEWUSST modulglobal definiert, damit
# `concurrent.futures.ProcessPoolExecutor` sie picklen kann (Closures
# innerhalb von `_score_target_with_deltalinf9xgb` waeren nicht picklebar).
#
# Pro Liganden wird das _docked.pdbqt nur EINMAL gelesen und gesplittet.
# Anschliessend wird pro angeforderter Pose ein obabel-Subprocess gestartet.
# Das vermeidet das wiederholte Re-Parsen der Multi-Pose-PDBQT bei
# Liganden mit vielen Posen.
#
# Rueckgabeformat: list[tuple[(ligand, pose_idx), Optional[Path]]]
#   - Path bei Erfolg
#   - None bei Fehlschlag (Pose-Index ungueltig, obabel-Crash, IO-Fehler)
#
# Logger wird absichtlich NICHT uebergeben (Logger sind nicht zuverlaessig
# picklebar). Fehler werden im Hauptprozess anhand der None-Eintraege
# aggregiert geloggt.

def _prepare_ligand_mol2s_worker(
    ligand:       str,
    docked_pdbqt: str,         # str statt Path: einfacher zu picklen
    pose_indices: list[int],   # 1-basiert
    out_dir:      str,
    obabel_bin:   str,
) -> list[tuple[tuple[str, int], Optional[str]]]:
    """
    Modulglobaler Worker fuer ProcessPoolExecutor.

    Liest das _docked.pdbqt EINMAL und konvertiert alle angeforderten
    Posen zu MOL2. Rueckgabe ist immer eine vollstaendige Liste fuer
    alle `pose_indices` (auch fehlgeschlagene → None), damit der
    Hauptprozess Fehler zaehlen kann.

    Args:
        ligand:       Ligand-Name (fuer Key-Tupel)
        docked_pdbqt: Pfad zur Multi-Pose _docked.pdbqt
        pose_indices: 1-basierte Pose-Indizes
        out_dir:      Ausgabeverzeichnis fuer MOL2-Dateien
        obabel_bin:   Pfad zum obabel-Binary
    """
    # Robustheit: ungueltige Eingaben → alle als None markieren
    results: list[tuple[tuple[str, int], Optional[str]]] = []

    try:
        with open(docked_pdbqt, encoding="utf-8", errors="replace") as fh:
            lines_raw = fh.readlines()
    except OSError:
        return [((ligand, p), None) for p in pose_indices]

    pose_blocks = _extract_poses_from_pdbqt(lines_raw)
    n_blocks = len(pose_blocks)

    out_dir_path = Path(out_dir)
    for pidx in pose_indices:
        if pidx < 1 or pidx > n_blocks:
            results.append(((ligand, pidx), None))
            continue

        mol2_path = out_dir_path / f"{ligand}_pose{pidx}.mol2"
        tmp_pdbqt = mol2_path.with_suffix(".pdbqt")

        try:
            with open(tmp_pdbqt, "w", encoding="utf-8") as fh:
                fh.writelines(pose_blocks[pidx - 1])

            proc = subprocess.run(
                [obabel_bin, str(tmp_pdbqt), "-O", str(mol2_path)],
                capture_output=True, text=True, timeout=30,
            )
            ok = (
                proc.returncode == 0
                and mol2_path.exists()
                and mol2_path.stat().st_size > 0
            )
            results.append(((ligand, pidx), str(mol2_path) if ok else None))
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            results.append(((ligand, pidx), None))
        finally:
            try:
                tmp_pdbqt.unlink(missing_ok=True)
            except Exception:
                pass

    return results


def _score_target_with_deltalinf9xgb(
    all_poses:      list[PoseResult],
    docked_files:   list[Path],
    target:         "_TargetInfo | object",
    logger:         logging.Logger,
    n_workers:      int = 1,
    prep_workers:   int = 0,
) -> int:
    """
    Scored alle Posen mit ΔLin_F9XGB ueber einen Worker-Pool.
    Schreibt das Ergebnis (invertiert: -pKd, kleiner=besser) in
    pose.score_deltalinf9xgb.

    Architektur:
      1. Receptor PDBQT → PDB einmal konvertieren
      2. Pro Liganden: PDBQT EINMAL lesen, alle Posen → MOL2
         (parallelisiert ueber ProcessPoolExecutor mit `prep_workers`)
      3. Pool von n_workers Subprocesses scoret alle Posen parallel
      4. Score invertieren und in PoseResult schreiben
      5. Pool und tmp_root nach Target shutdown

    Args:
        n_workers:    Anzahl paralleler ΔLin_F9XGB-Subprocesses (Default 1).
        prep_workers: Anzahl paralleler Worker fuer MOL2-Vorbereitung.
                      0 (Default) = automatisch = max(1, n_workers).
                      Jeder Worker startet pro Pose einen obabel-Subprocess,
                      die effektive Subprocess-Last ist also `prep_workers`
                      gleichzeitig. Bei mehreren Targets parallel
                      (Multi-GPU-Orchestrator) sollte `prep_workers` niedriger
                      gesetzt werden, damit nicht alle Targets gleichzeitig
                      die CPU saturieren (siehe pipeline_config.ini).

    Rueckgabe: Anzahl erfolgreich gescoreter Posen.
    """
    try:
        import linf9xgb_scorer  # type: ignore
    except ImportError:
        logger.warning(
            "  [%s] ΔLin_F9XGB aktiviert, aber linf9xgb_scorer nicht importierbar – "
            "Score wird uebersprungen.",
            target.name,
        )
        return 0

    if not linf9xgb_scorer.is_available():
        logger.warning(
            "  [%s] ΔLin_F9XGB-Toolkit nicht installiert (LINF9XGB_DIR fehlt) – "
            "Score wird uebersprungen.",
            target.name,
        )
        return 0

    # Receptor einmal konvertieren, MOL2-Dateien fuer alle Posen vorbereiten
    import tempfile
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tmp_root = Path(tempfile.mkdtemp(prefix="linf9xgb_", dir="/tmp"))
    try:
        receptor_pdb = _convert_pdbqt_receptor_to_pdb(
            Path(target.pdbqt_path), tmp_root, logger,
        )
        if receptor_pdb is None:
            logger.warning(
                "  [%s] Receptor-Konvertierung fehlgeschlagen – "
                "ΔLin_F9XGB wird uebersprungen.",
                target.name,
            )
            return 0

        # Pose-Index Lookup: ligand_name → docked_file
        docked_by_lig = {
            p.stem.removesuffix("_docked"): p for p in docked_files
        }

        # --- Phase 1: alle Posen als MOL2 vorbereiten (parallelisiert) ---
        # Pro Liganden wird das _docked.pdbqt EINMAL gelesen und alle
        # angeforderten Posen in einem Rutsch konvertiert. Das vermeidet
        # das wiederholte Re-Parsen bei Liganden mit vielen Posen.
        #
        # Parallelisierung ueber ProcessPoolExecutor: jeder Worker bearbeitet
        # einen kompletten Liganden. ProcessPool (nicht ThreadPool), weil
        # obabel als externer Subprocess laeuft und wir keine Probleme mit
        # dem GIL bei vielen kleinen Disk-I/O Operationen haben wollen.
        n_total = len(all_poses)
        t_prep = time.monotonic()

        # Map (ligand, pose_idx) → PoseResult fuer Backref nach Score
        # Wird erst NACH erfolgreicher MOL2-Konvertierung gefuellt, damit
        # er nur Schluessel enthaelt fuer die auch ein Job existiert.
        pose_by_key: dict = {}
        # Posen pro Liganden gruppieren (deterministische Reihenfolge)
        ligand_pose_map: dict[str, list[int]] = {}
        # Zwischenmap zum spaeteren Befuellen von pose_by_key
        pose_lookup: dict[tuple[str, int], PoseResult] = {}
        n_no_docked = 0

        for pose in all_poses:
            if pose.ligand not in docked_by_lig:
                n_no_docked += 1
                continue
            ligand_pose_map.setdefault(pose.ligand, []).append(pose.pose)
            pose_lookup[(pose.ligand, pose.pose)] = pose

        n_ligands = len(ligand_pose_map)
        if n_ligands == 0:
            logger.warning(
                "  [%s] ΔLin_F9XGB: keine Liganden mit _docked.pdbqt gefunden "
                "(%d Posen ohne Docking-File).",
                target.name, n_no_docked,
            )
            return 0

        # Effektive Worker-Anzahl bestimmen
        eff_prep_workers = prep_workers if prep_workers > 0 else max(1, n_workers)
        eff_prep_workers = max(1, min(eff_prep_workers, n_ligands))

        logger.info(
            "  [%s] ΔLin_F9XGB Phase 1: %d Posen ueber %d Liganden "
            "mit %d Prep-Workern vorbereiten…",
            target.name, n_total - n_no_docked, n_ligands, eff_prep_workers,
        )

        # Job-Liste fuer den Scorer-Pool
        jobs: list = []
        n_prep_failed = n_no_docked  # Posen ohne Docking-File zaehlen mit
        n_done_ligands = 0
        # Progress-Logging: alle ~5% der Liganden
        log_every = max(1, n_ligands // 20)
        # `as_completed`-Reihenfolge ist NICHT die Submit-Reihenfolge;
        # `n_done_ligands` darf deshalb nur hier im Hauptthread inkrementiert
        # werden – nicht in einer Worker-Closure.

        with ProcessPoolExecutor(max_workers=eff_prep_workers) as pool:
            # Submit: pro Liganden ein Future
            future_to_ligand = {}
            for ligand, pose_indices in ligand_pose_map.items():
                docked = docked_by_lig[ligand]
                fut = pool.submit(
                    _prepare_ligand_mol2s_worker,
                    ligand,
                    str(docked),
                    pose_indices,
                    str(tmp_root),
                    _OBABEL_BIN,
                )
                future_to_ligand[fut] = ligand

            # Collect: Worker-Crash-resilient
            for fut in as_completed(future_to_ligand):
                ligand = future_to_ligand[fut]
                try:
                    ligand_results = fut.result()
                except Exception as exc:
                    # Worker-Tod oder unerwarteter Fehler – ALLE Posen
                    # dieses Liganden als fehlgeschlagen markieren
                    logger.warning(
                        "  [%s] ΔLin_F9XGB Prep-Worker fuer Ligand '%s' "
                        "abgestuerzt: %s", target.name, ligand, exc,
                    )
                    n_prep_failed += len(ligand_pose_map[ligand])
                    n_done_ligands += 1
                    continue

                for (lig, pidx), mol2_str in ligand_results:
                    if mol2_str is None:
                        n_prep_failed += 1
                        continue
                    pose_obj = pose_lookup.get((lig, pidx))
                    if pose_obj is None:
                        # Unerwartet: Worker liefert Key, den wir nicht
                        # angefragt haben → defensiv ignorieren
                        continue
                    pose_by_key[(lig, pidx)] = pose_obj
                    jobs.append(linf9xgb_scorer.ScoringJob(
                        key=(lig, pidx),
                        protein_pdb=receptor_pdb,
                        ligand_mol2=Path(mol2_str),
                    ))

                n_done_ligands += 1
                if n_done_ligands % log_every == 0 or n_done_ligands == n_ligands:
                    elapsed = time.monotonic() - t_prep
                    rate = n_done_ligands / elapsed if elapsed > 0 else 0.0
                    eta = (n_ligands - n_done_ligands) / rate if rate > 0 else 0.0
                    logger.info(
                        "  [%s] ΔLin_F9XGB Phase 1: %d/%d Liganden | "
                        "%.1fs verstrichen | %.1f Lig/s | ETA: %.0fs",
                        target.name, n_done_ligands, n_ligands,
                        elapsed, rate, eta,
                    )

        prep_elapsed = time.monotonic() - t_prep
        logger.info(
            "  [%s] ΔLin_F9XGB: %d MOL2-Dateien vorbereitet "
            "(%d Vorbereitungs-Fehler, %.1fs, %.1f Posen/s)",
            target.name, len(jobs), n_prep_failed, prep_elapsed,
            len(jobs) / prep_elapsed if prep_elapsed > 0 else 0.0,
        )

        if not jobs:
            return 0

        # --- Phase 2: paralleles Scoring ---
        log_every = max(50, len(jobs) // 20)
        t_score = time.monotonic()

        def progress(done, total, key, score):
            # Wird thread-safe vom Pool aufgerufen
            if done % log_every == 0 or done == total:
                elapsed = time.monotonic() - t_score
                eta = (elapsed / done) * (total - done) if done > 0 else 0.0
                logger.info(
                    "  [%s] ΔLin_F9XGB: %d/%d Posen | %.1fs verstrichen | ETA: %.0fs",
                    target.name, done, total, elapsed, eta,
                )

        results = linf9xgb_scorer.score_poses_batch(
            jobs,
            n_workers=max(1, n_workers),
            progress_callback=progress,
        )

        # --- Phase 3: Ergebnisse in PoseResult eintragen ---
        n_scored = 0
        n_score_failed = 0
        for key, pkd in results.items():
            pose = pose_by_key.get(key)
            if pose is None:
                continue
            if pkd is not None:
                # Invertieren: groesser=besser → kleiner=besser fuer ECR
                pose.score_deltalinf9xgb = -pkd
                n_scored += 1
            else:
                n_score_failed += 1

        score_elapsed = time.monotonic() - t_score
        logger.info(
            "  [%s] ΔLin_F9XGB fertig: %d/%d Posen gescored "
            "(%d Vorbereitung, %d Score, %.1fs gesamt)",
            target.name, n_scored, n_total,
            n_prep_failed, n_score_failed,
            prep_elapsed + score_elapsed,
        )

        # Pool zwischen Targets beenden (Speicher-Hygiene, naechster
        # Target startet seinen eigenen Pool in ~3s)
        # NB: shutdown ist auch im finally-Block (zur Absicherung gegen
        # Exceptions vor diesem Punkt), hier aber explizit fuer den
        # Normalfall, weil die Pool-Shutdown-Latenz nicht in den
        # tmp_root-Cleanup-Pfad einfliessen sollte.
        try:
            linf9xgb_scorer.shutdown()
        except Exception:
            pass

        return n_scored
    finally:
        # tmp_root mit ALLEN MOL2-Dateien auf einmal aufraeumen
        try:
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        # Falls der Score-Pool durch eine Exception oben nicht beendet
        # wurde: Sicherheitsnetz. linf9xgb_scorer.shutdown ist idempotent.
        try:
            import linf9xgb_scorer  # type: ignore
            linf9xgb_scorer.shutdown()
        except Exception:
            pass


# ======================================================================
# HAUPTFUNKTION: rescore_target()
# ======================================================================

# ======================================================================
# RANKING + AUSGABE (aus rescore_target herausgeloest)
# ======================================================================

def _rank_and_write(
    all_poses:          list[PoseResult],
    active:             list[str],
    target,
    target_results_dir: Path,
    rescore_cfg:        RescoringConfig,
    logger:             logging.Logger,
) -> list[LigandResult]:
    """
    Zweite Haelfte des Rescorings: ECR ueber ALLE Posen, Aggregation,
    CSV-Ausgabe.

    Herausgeloest, weil sie im Blockmodus erst laufen darf, wenn saemtliche
    Bloecke gescort sind – die Raenge werden ueber den gesamten Posensatz
    gebildet, ein blockweises Ranking waere ein anderes Verfahren.
    Ausserdem ist dieser Teil billig und rein rechnerisch, laesst sich also
    beliebig oft mit anderen Gewichten wiederholen (rescore_rank.py).
    """
    # --- Nur Scores in ECR einbeziehen fuer die Daten vorhanden sind ---
    ATTR_MAP = {
        "vina":              "score_vina",
        "vinardo":           "score_vinardo",
        "ad4":               "score_ad4",
        "cnnaffinity":       "score_cnnaffinity",
        "cnnscore":          "score_cnnscore",
        "deltalinf9xgb":     "score_deltalinf9xgb",
        "dense_cnnaffinity": "score_dense_cnnaffinity",
        "dense_cnnscore":    "score_dense_cnnscore",
    }
    scores_with_data = [
        sc for sc in active
        if any(getattr(p, ATTR_MAP[sc]) is not None for p in all_poses)
    ]

    excluded = set(active) - set(scores_with_data)
    if excluded:
        logger.warning(
            "  [%s] Folgende Scores haben keine Daten und werden vom "
            "ECR ausgeschlossen: %s",
            target.name, ", ".join(sorted(excluded)),
        )

    if not scores_with_data:
        logger.error("  [%s] Keine Score-Daten fuer ECR vorhanden – Abbruch.",
                     target.name)
        return []

    sigma = len(all_poses) / rescore_cfg.sigma_fraction
    ecr_weights = rescore_cfg.get_ecr_weights(scores_with_data)
    w_str = ", ".join(f"{k}={v:.2f}" for k, v in ecr_weights.items())
    logger.info(
        "  [%s] ECR: %d Posen | %d Scores (%s) | sigma = %.2f | weights: %s",
        target.name, len(all_poses), len(scores_with_data),
        ", ".join(scores_with_data), sigma, w_str,
    )

    # --- ECR berechnen ---
    all_poses = _compute_ecr(all_poses, rescore_cfg.sigma_fraction,
                             scores_with_data, ecr_weights)

    # --- Aggregation ---
    ligand_results = _aggregate_ligands(all_poses)

    # --- CSVs schreiben ---
    pose_csv   = _write_pose_csv(all_poses, target_results_dir, target.name)
    ligand_csv = _write_ligand_csv(ligand_results, target_results_dir, target.name)
    logger.info("  [%s] Pose-CSV:   %s", target.name, pose_csv.name)
    logger.info("  [%s] Ligand-CSV: %s", target.name, ligand_csv.name)

    return ligand_results


def rescore_target(
    target:             "_TargetInfo | object",
    target_results_dir: Path,
    rescore_cfg:        RescoringConfig,
    logger:             logging.Logger,
    files:              Optional[list[Path]] = None,
    poses_only:         bool = False,
) -> list[LigandResult]:
    """
    Fuehrt das vollstaendige Rescoring fuer einen Target durch.

    Ablauf:
      1.  _docked.pdbqt-Dateien einlesen
      2.  Vina-Scores aus PDBQT-Headern
      3.  CNNaffinity + CNNscore via gnina CLI (optional, ein Aufruf pro Ligand)
      4.  Richtungskorrektur: CNNaffinity + CNNscore invertieren
      5.  ECR auf Pose-Ebene berechnen
      6.  Aggregation → P(ligand) = max(ECR)
      7.  CSVs schreiben (Pose-Level + Liganden-Rangliste)

    Rueckgabe: LigandResult-Liste, absteigend nach ECR-Score.
    """
    # files: explizite Dateiliste statt aller Posen des Targets. Wird vom
    # Blockmodus benutzt, um einen Ausschnitt zu scoren.
    docked_files = (sorted(files) if files is not None
                    else sorted(target_results_dir.glob("*_docked.pdbqt")))

    if not docked_files:
        logger.warning(
            "  [%s] Keine _docked.pdbqt-Dateien – Rescoring uebersprungen.",
            target.name,
        )
        return []

    logger.info(
        "  [%s] Starte Rescoring: %d Liganden",
        target.name, len(docked_files),
    )

    # --- Aktive Scoring-Funktionen bestimmen ---
    active: list[str] = []
    if rescore_cfg.vina_enabled:
        active.append("vina")
    if rescore_cfg.vinardo_enabled:
        active.append("vinardo")
    if rescore_cfg.ad4_enabled:
        active.append("ad4")
    if rescore_cfg.cnnaffinity_enabled:
        active.append("cnnaffinity")
    if rescore_cfg.cnnscore_enabled:
        active.append("cnnscore")
    if rescore_cfg.deltalinf9xgb_enabled:
        active.append("deltalinf9xgb")
    if rescore_cfg.dense_enabled:
        active.append("dense_cnnaffinity")
        active.append("dense_cnnscore")

    # --- Scoring-Backend bestimmen ---
    # Prioritaet: gninatorch GPU > gnina CLI
    # gninatorch: persistent GPU model, ~5-10x schneller
    # gnina CLI:  Subprocess pro Ligand, CPU-bound (Fallback)
    use_gninatorch = (
        GNINATORCH_OK
        and rescore_cfg.gnina_needed
        and rescore_cfg.gnina_use_gpu
    )

    if rescore_cfg.gnina_needed and not use_gninatorch and not _GNINA_OK:
        logger.warning(
            "  gnina nicht verfuegbar (weder gninatorch noch CLI Binary) – "
            "CNNscore/CNNaffinity werden uebersprungen."
            " (pip install gninatorch oder wget gnina Binary)"
        )

    if rescore_cfg.cluster_poses:
        logger.info("  [%s] Pose-Clustering aktiv: RMSD-Cutoff=%.1f Å",
                    target.name, rescore_cfg.cluster_rmsd_cutoff)

    # --- Scoring-Backend loggen ---
    if rescore_cfg.gnina_needed:
        if use_gninatorch:
            # GPU-ID bestimmen: Im Container ist durch CUDA_VISIBLE_DEVICES
            # immer nur eine GPU sichtbar → Device 0 verwenden.
            # Nur wenn KEINE Container-Isolation: gnina_gpu_id nutzen.
            _env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
            if _env_gpu is not None:
                # Container/Orchestrator hat GPU bereits gepinnt
                _torch_gpu_id = 0
                gpu_info = f"GPU via Container-Env (CUDA_VISIBLE_DEVICES={_env_gpu}) → torch device 0"
            elif rescore_cfg.gnina_gpu_id is not None:
                _torch_gpu_id = rescore_cfg.gnina_gpu_id
                gpu_info = f"GPU {_torch_gpu_id} (explizit)"
            else:
                _torch_gpu_id = 0
                gpu_info = "GPU 0 (default)"
            logger.info("  [%s] gninatorch GPU-Rescoring aktiv: %s (%s)",
                        target.name, rescore_cfg.cnn_model, gpu_info)
        else:
            _effective_gnina = _resolve_gnina_binary(rescore_cfg.gnina_binary)
            if _effective_gnina:
                _env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
                if rescore_cfg.gnina_gpu_id is not None:
                    gpu_info = f"GPU {rescore_cfg.gnina_gpu_id} (explizit)"
                elif _env_gpu is not None:
                    gpu_info = f"GPU via Container-Env (CUDA_VISIBLE_DEVICES={_env_gpu})"
                else:
                    gpu_info = "alle GPUs (kein Pinning)"
                logger.info("  [%s] gnina CLI Rescoring (Fallback): %s (Binary: %s, %s)",
                            target.name, rescore_cfg.cnn_model, _effective_gnina,
                            gpu_info)
            else:
                logger.warning("  [%s] gnina Binary nicht gefunden – "
                               "CNNscore/CNNaffinity deaktiviert.", target.name)

    # --- Alle Liganden / Posen scoren ---
    n_total      = len(docked_files)
    t_loop_start = datetime.now()

    def _log_progress(n_done):
        if n_done % 1000 == 0 or n_done == n_total:
            from datetime import timedelta
            elapsed   = datetime.now() - t_loop_start
            pct       = n_done / n_total * 100
            eta_s     = (elapsed.total_seconds() / n_done) * (n_total - n_done) if n_done > 0 else 0
            eta_str   = str(timedelta(seconds=int(eta_s))) if eta_s > 0 else "--:--:--"
            logger.info(
                "  [%s] Fortschritt: %d/%d Liganden (%.1f%%) | vergangen: %s | ETA: %s",
                target.name, n_done, n_total, pct,
                str(elapsed).split(".")[0], eta_str,
            )

    # ==================================================================
    # PFAD 1: gninatorch GPU (persistent model, sequentiell)
    # ==================================================================
    # GPU ist der Bottleneck, nicht CPU → kein joblib noetig.
    # Modell wird einmal geladen und fuer alle Liganden wiederverwendet.
    # Fallback auf CLI wenn Scorer-Init oder einzelner Ligand fehlschlaegt.
    # ==================================================================

    all_poses: list[PoseResult] = []
    _scoring_done = False

    if use_gninatorch and rescore_cfg.gnina_needed:
        logger.info(
            "  [%s] Starte GPU-Scoring (gninatorch): %d Liganden",
            target.name, n_total,
        )

        _temp_files: list[Path] = []
        _gpu_scorer: Optional[GninaGPUScorer] = None
        _dense_scorer: Optional[GninaGPUScorer] = None
        _gpu_failures = 0

        try:
            if rescore_cfg.primary_cnn_needed:
                _gpu_scorer = GninaGPUScorer(
                    gpu_id=_torch_gpu_id,
                    cnn_model=rescore_cfg.cnn_model,
                )
            else:
                logger.info("  [%s] Primaermodell uebersprungen "
                            "(cnnaffinity/cnnscore aus).", target.name)
                _gpu_scorer = None
        except Exception as exc:
            logger.warning(
                "  [%s] GninaGPUScorer Init fehlgeschlagen: %s – "
                "Fallback auf gnina CLI.",
                target.name, exc,
            )
            _gpu_scorer = None

        # Dense-Scorer laden (optionales zweites Modell)
        # Nicht mehr an _gpu_scorer gekoppelt: sonst faellt Dense mit aus,
        # sobald das Primaermodell abgeschaltet ist – genau der Fall
        # "nur Dense-Affinity".
        if rescore_cfg.dense_cnn_needed:
            try:
                _dense_scorer = GninaGPUScorer(
                    gpu_id=_torch_gpu_id,
                    cnn_model=rescore_cfg.dense_model,
                )
                logger.info("  [%s] Dense-Scorer geladen: %s",
                            target.name, rescore_cfg.dense_model)
            except Exception as exc:
                logger.warning(
                    "  [%s] Dense-Scorer Init fehlgeschlagen: %s – "
                    "Dense-Scores werden uebersprungen.",
                    target.name, exc,
                )
                _dense_scorer = None

        if _gpu_scorer is not None or _dense_scorer is not None:
            try:
                # ── Modus-Auswahl: Multi-Ligand-Batch vs. Einzel-Ligand ──
                # Multi-Ligand-Batching ist deutlich schneller, aber nur moeglich
                # wenn kein Pose-Clustering aktiv ist (Clustering erzeugt
                # individuelle Temp-Dateien pro Ligand).

                if not rescore_cfg.cluster_poses:
                    # ══════════════════════════════════════════════════════
                    # PFAD 1a: Multi-Ligand-Batch (schnell, ohne Clustering)
                    # ══════════════════════════════════════════════════════
                    # Vina-Scores vorab fuer alle Liganden parsen
                    vina_data: dict[str, list[tuple[int, float]]] = {}
                    valid_docked: list[Path] = []
                    for docked in docked_files:
                        ligand_name = docked.stem.removesuffix("_docked")
                        vp = _parse_vina_pdbqt(docked)
                        if vp:
                            vina_data[ligand_name] = vp
                            valid_docked.append(docked)

                    if valid_docked:
                        # Batch-Scoring: Primaermodell
                        batch_results = (
                            _gpu_scorer.score_ligands_batch(
                                target.pdbqt_path,
                                valid_docked,
                                max_poses_per_batch=rescore_cfg.rescore_batch_size,
                            )
                            if _gpu_scorer is not None else {}
                        )

                        # Batch-Scoring: Dense-Modell (wenn aktiv)
                        dense_batch_results: dict = {}
                        if _dense_scorer is not None:
                            logger.info("  [%s] Dense-Scoring: %d Liganden",
                                        target.name, len(valid_docked))
                            dense_batch_results = _dense_scorer.score_ligands_batch(
                                target.pdbqt_path,
                                valid_docked,
                                max_poses_per_batch=rescore_cfg.rescore_batch_size,
                            )

                        # Ergebnisse zusammenbauen
                        _lig_count = 0
                        # Ohne Primaermodell ist batch_results leer – dann
                        # muss die Iteration ueber die Dense-Ergebnisse gehen,
                        # sonst wird still gar nichts zusammengebaut.
                        _source = batch_results or dense_batch_results
                        for ligand_name in _source:
                            gnina_map = batch_results.get(ligand_name, {})
                            _lig_count += 1
                            vina_poses = vina_data.get(ligand_name, [])
                            dense_map = dense_batch_results.get(ligand_name, {})

                            # CLI-Fallback fuer leere GPU-Ergebnisse.
                            # Nur wenn das Primaermodell ueberhaupt laufen
                            # sollte: ist es abgeschaltet, ist gnina_map fuer
                            # JEDEN Liganden leer und der Fallback wuerde
                            # einen gnina-Prozess pro Ligand starten – fuer
                            # Werte, die gar nicht gebraucht werden.
                            if (_gpu_scorer is not None
                                    and not gnina_map and _GNINA_OK):
                                _gpu_failures += 1
                                docked_path = next(
                                    (d for d in valid_docked
                                     if d.stem.removesuffix("_docked") == ligand_name),
                                    None
                                )
                                if docked_path:
                                    gnina_map = _score_gnina_ligand(
                                        docked_path, target.pdbqt_path,
                                        rescore_cfg.cnn_model,
                                        rescore_cfg.gnina_use_gpu,
                                        rescore_cfg.gnina_binary,
                                        rescore_cfg.gnina_gpu_id,
                                    )

                            for pose_idx, vina_score in vina_poses:
                                gnina_entry = gnina_map.get(pose_idx, {})
                                dense_entry = dense_map.get(pose_idx, {})
                                cnn_raw = gnina_entry.get("cnnscore")
                                aff_raw = gnina_entry.get("cnnaffinity")
                                d_cnn_raw = dense_entry.get("cnnscore")
                                d_aff_raw = dense_entry.get("cnnaffinity")
                                all_poses.append(PoseResult(
                                    ligand                  = ligand_name,
                                    pose                    = pose_idx,
                                    score_vina              = vina_score,
                                    score_cnnaffinity       = (-aff_raw) if aff_raw is not None else None,
                                    score_cnnscore          = (-cnn_raw) if cnn_raw is not None else None,
                                    score_dense_cnnaffinity = (-d_aff_raw) if d_aff_raw is not None else None,
                                    score_dense_cnnscore    = (-d_cnn_raw) if d_cnn_raw is not None else None,
                                ))
                            _log_progress(_lig_count)

                    _scoring_done = True

                else:
                    # ══════════════════════════════════════════════════════
                    # PFAD 1b: Einzel-Ligand (mit Clustering)
                    # ══════════════════════════════════════════════════════
                    for lig_idx, docked in enumerate(docked_files, 1):
                        ligand_name = docked.stem.removesuffix("_docked")
                        vina_poses  = _parse_vina_pdbqt(docked)
                        if not vina_poses:
                            logger.debug("    Keine Vina-Scores in '%s' – ueberspringe.",
                                         docked.name)
                            continue

                        # ── Pose-Clustering ──
                        score_path = docked
                        if len(vina_poses) > 1:
                            score_path, vina_poses = _apply_pose_clustering(
                                docked, vina_poses, rescore_cfg.cluster_rmsd_cutoff, logger
                            )
                            if score_path != docked:
                                _temp_files.append(score_path)

                        # ── gninatorch GPU Rescoring (Primaermodell) ──
                        # _gpu_scorer ist None, wenn cnnaffinity und cnnscore
                        # beide aus sind – dann liefe hier ein AttributeError.
                        gnina_map: dict = {}
                        if _gpu_scorer is not None:
                            gnina_map = _gpu_scorer.score_ligand(
                                target.pdbqt_path, score_path,
                            )

                        # ── Dense-Scoring (wenn aktiv) ──
                        dense_map: dict = {}
                        if _dense_scorer is not None:
                            dense_map = _dense_scorer.score_ligand(
                                target.pdbqt_path, score_path,
                            )

                        # Fallback auf CLI wenn GPU-Scoring fehlschlaegt.
                        # Nur bei aktivem Primaermodell – sonst waere
                        # gnina_map immer leer und der Fallback liefe fuer
                        # jeden Liganden als eigener Prozess.
                        if (_gpu_scorer is not None
                                and not gnina_map and _GNINA_OK):
                            _gpu_failures += 1
                            gnina_map = _score_gnina_ligand(
                                score_path, target.pdbqt_path,
                                rescore_cfg.cnn_model, rescore_cfg.gnina_use_gpu,
                                rescore_cfg.gnina_binary,
                                rescore_cfg.gnina_gpu_id,
                            )

                        for pose_idx, vina_score in vina_poses:
                            gnina_entry = gnina_map.get(pose_idx, {})
                            dense_entry = dense_map.get(pose_idx, {})
                            cnn_raw = gnina_entry.get("cnnscore")
                            aff_raw = gnina_entry.get("cnnaffinity")
                            d_cnn_raw = dense_entry.get("cnnscore")
                            d_aff_raw = dense_entry.get("cnnaffinity")
                            all_poses.append(PoseResult(
                                ligand                  = ligand_name,
                                pose                    = pose_idx,
                                score_vina              = vina_score,
                                score_cnnaffinity       = (-aff_raw) if aff_raw is not None else None,
                                score_cnnscore          = (-cnn_raw) if cnn_raw is not None else None,
                                score_dense_cnnaffinity = (-d_aff_raw) if d_aff_raw is not None else None,
                                score_dense_cnnscore    = (-d_cnn_raw) if d_cnn_raw is not None else None,
                            ))
                        _log_progress(lig_idx)

                    _scoring_done = True

            finally:
                if _gpu_scorer is not None:
                    _gpu_scorer.close()
                if _dense_scorer is not None:
                    _dense_scorer.close()

            # Temp-Dateien aufraeumen
            for tf in _temp_files:
                try:
                    tf.unlink(missing_ok=True)
                except Exception:
                    pass

            if _gpu_failures > 0:
                logger.info("  [%s] GPU-Scoring: %d Liganden via CLI-Fallback rescored.",
                            target.name, _gpu_failures)

    # ==================================================================
    # CLI-FALLBACK (nur wenn gninatorch nicht verfuegbar oder Init
    # fehlgeschlagen – alter gnina-Subprocess-Modus)
    # ==================================================================

    if not _scoring_done:
        all_poses = []  # Reset falls GPU-Pfad teilweise gelaufen ist

        # n_jobs fuer CLI: parallel lohnt sich hier (CPU-bound)
        n_jobs = rescore_cfg.n_jobs
        logger.info(
            "  [%s] CLI-Fallback: n_jobs=%s | %d Liganden",
            target.name, "alle Cores" if n_jobs == -1 else str(n_jobs),
            n_total,
        )

        cli_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_score_one_ligand)(
                docked,
                target.pdbqt_path,
                rescore_cfg.gnina_needed,
                rescore_cfg.cnn_model,
                rescore_cfg.gnina_use_gpu,
                rescore_cfg.gnina_binary,
                rescore_cfg.cluster_poses,
                rescore_cfg.cluster_rmsd_cutoff,
                rescore_cfg.gnina_gpu_id,
                rescore_cfg.extra_empirical_functions,
            )
            for docked in docked_files
        )
        for poses in cli_results:
            all_poses.extend(poses)
        _log_progress(n_total)

    if not all_poses:
        logger.warning("  [%s] Keine Pose-Daten – Rescoring leer.", target.name)
        return []

    # --- Empirische Extra-Funktionen nachziehen (Vinardo, AD4) --------
    # Der gninatorch-GPU-Pfad liefert nur CNN-Scores und umgeht den
    # CLI-Fallback. Falls Vinardo/AD4 aktiv sind und dort noch keine Werte
    # stehen, werden sie hier nachgereicht – ein gnina-Durchlauf je Funktion
    # und Ligand, CPU-bound und ueber n_jobs parallelisiert.
    if rescore_cfg.empirical_extra_needed:
        missing = [
            (key, gnina_name)
            for key, gnina_name in rescore_cfg.extra_empirical_functions
            if all(getattr(p, f"score_{key}") is None for p in all_poses)
        ]
        if missing and _GNINA_OK:
            logger.info(
                "  [%s] Empirisches Rescoring nachziehen: %s (%d Liganden)",
                target.name, ", ".join(k for k, _ in missing), len(docked_files),
            )
            emp_results = Parallel(n_jobs=rescore_cfg.n_jobs, verbose=0)(
                delayed(_score_empirical_one_ligand)(
                    docked, target.pdbqt_path, missing,
                    rescore_cfg.gnina_binary, rescore_cfg.gnina_gpu_id,
                )
                for docked in docked_files
            )
            merged = {}
            for entry in emp_results:
                merged.update(entry)
            n_set = 0
            for pose in all_poses:
                vals = merged.get((pose.ligand, pose.pose))
                if not vals:
                    continue
                for key, value in vals.items():
                    if value is not None:
                        setattr(pose, f"score_{key}", value)
                        n_set += 1
            logger.info("  [%s] Empirisch gesetzt: %d Werte", target.name, n_set)
        elif missing:
            logger.warning(
                "  [%s] Vinardo/AD4 aktiviert, aber kein gnina Binary – "
                "Scores werden vom ECR ausgeschlossen.", target.name,
            )

    # --- ΔLin_F9XGB Scoring (separates Conda-Env, Worker-Pool) ---
    if rescore_cfg.deltalinf9xgb_enabled:
        eff_prep = (
            rescore_cfg.deltalinf9xgb_prep_workers
            if rescore_cfg.deltalinf9xgb_prep_workers > 0
            else rescore_cfg.deltalinf9xgb_n_workers
        )
        logger.info(
            "  [%s] ΔLin_F9XGB: starte Scoring fuer %d Posen "
            "(Score-Pool n=%d, Prep-Pool n=%d)…",
            target.name, len(all_poses),
            rescore_cfg.deltalinf9xgb_n_workers, eff_prep,
        )
        n_dlf9 = _score_target_with_deltalinf9xgb(
            all_poses, docked_files, target, logger,
            n_workers=rescore_cfg.deltalinf9xgb_n_workers,
            prep_workers=rescore_cfg.deltalinf9xgb_prep_workers,
        )
        if n_dlf9 == 0:
            logger.warning(
                "  [%s] ΔLin_F9XGB hat keine Posen scoren koennen – "
                "Score wird aus ECR ausgeschlossen.",
                target.name,
            )

    # Blockmodus: nur die Rohscores zurueckgeben, Ranking macht der
    # Aufrufer ueber alle Bloecke hinweg (siehe rescore_target_blocked).
    if poses_only:
        return all_poses

    return _rank_and_write(all_poses, active, target, target_results_dir,
                           rescore_cfg, logger)




# ======================================================================
# BLOCKMODUS: SCORING IN TEILSCHRITTEN MIT ZWISCHENSTAENDEN
# ======================================================================

_PARTIAL_DIR  = ".rescore_partial"
_PARTIAL_COLS = (["ligand", "pose"]
                 + [ecr_mod.FIELDS[k][0] for k in ecr_mod.ALL_KEYS])


def _partial_path(target_results_dir: Path, block_idx: int) -> Path:
    return (target_results_dir / _PARTIAL_DIR
            / f"scores_{block_idx:05d}.csv")


def _active_keys_for_coverage(rescore_cfg) -> list[str]:
    """ECR-Keys, die vollstaendig vorliegen muessen.

    Nur die tatsaechlich berechneten. vina kommt aus dem PDBQT-Header und
    fehlt hoechstens, wenn die Pose kaputt ist – das faengt check_poses.py
    ab und ist kein Batch-Problem, deshalb hier ausgeklammert.
    """
    keys: list[str] = []
    if rescore_cfg.cnnaffinity_enabled:
        keys.append("cnnaffinity")
    if rescore_cfg.cnnscore_enabled:
        keys.append("cnnscore")
    if rescore_cfg.dense_enabled:
        keys.append("dense_cnnaffinity")
    if rescore_cfg.vinardo_enabled:
        keys.append("vinardo")
    if rescore_cfg.ad4_enabled:
        keys.append("ad4")
    if rescore_cfg.deltalinf9xgb_enabled:
        keys.append("deltalinf9xgb")
    return keys


def _write_partial(poses: list[PoseResult], path: Path) -> None:
    """
    Rohscores eines Blocks sichern – atomar ueber eine .tmp-Datei.

    Ohne das Umbenennen koennte ein Abbruch mitten im Schreiben eine
    halbe CSV hinterlassen, die beim naechsten Lauf als vollstaendig
    gelesen wuerde. Geschrieben werden nur Ligand, Pose und die Rohscores:
    Raenge und ECR-Terme haengen vom Gesamtsatz ab und werden ohnehin neu
    berechnet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_PARTIAL_COLS)
        w.writeheader()
        for p in poses:
            row = {"ligand": p.ligand, "pose": p.pose}
            for key in ecr_mod.ALL_KEYS:
                attr = ecr_mod.FIELDS[key][0]
                val = getattr(p, attr, None)
                row[attr] = "" if val is None else f"{val:.6f}"
            w.writerow(row)
    tmp.replace(path)


def _read_partial(path: Path) -> list[PoseResult]:
    """Liest einen gesicherten Block zurueck in PoseResult-Objekte."""
    poses: list[PoseResult] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pr = PoseResult(ligand=row["ligand"], pose=int(row["pose"]))
            for key in ecr_mod.ALL_KEYS:
                attr = ecr_mod.FIELDS[key][0]
                raw = (row.get(attr) or "").strip()
                if raw:
                    setattr(pr, attr, float(raw))
            poses.append(pr)
    return poses


def rescore_target_blocked(
    target:             "_TargetInfo | object",
    target_results_dir: Path,
    rescore_cfg:        RescoringConfig,
    logger:             logging.Logger,
) -> list[LigandResult]:
    """
    Rescoring in Bloecken mit Zwischenstaenden – wiederaufnahmefaehig.

    Warum es das gibt: rescore_target() haelt alle Posen im Speicher und
    schreibt erst am Ende. Ein an der Walltime abgebrochener Job verliert
    damit die gesamte Arbeit des Targets. Hier wird die Ligandenliste in
    Bloecke geschnitten, jeder Block komplett gescort und sofort gesichert.
    Ein Neustart ueberspringt alles Gesicherte.

    Das Ranking laeuft unveraendert ueber den GESAMTEN Posensatz, erst
    nachdem alle Bloecke vorliegen – die ECR-Raenge sind global, ein
    blockweises Ranking waere ein anderes Verfahren mit anderen Ergebnissen.

    Preis: pro Block wird das gninatorch-Modell neu geladen und der
    ΔLin_F9XGB-Worker-Pool neu gestartet. Bei Bloecken ab etwa 1000
    Liganden faellt das nicht ins Gewicht, bei 100 schon.
    """
    all_files = sorted(target_results_dir.glob("*_docked.pdbqt"))
    if not all_files:
        logger.warning("  [%s] Keine Posen gefunden.", target.name)
        return []

    size = max(1, rescore_cfg.rescore_block_size)
    blocks = [all_files[i:i + size] for i in range(0, len(all_files), size)]

    logger.info(
        "  [%s] Blockmodus: %d Posen-Dateien in %d Bloecken à %d",
        target.name, len(all_files), len(blocks), size,
    )

    all_poses: list[PoseResult] = []
    n_restored = 0

    for idx, block in enumerate(blocks):
        part = _partial_path(target_results_dir, idx)

        if part.exists() and part.stat().st_size > 0:
            try:
                restored = _read_partial(part)
                all_poses.extend(restored)
                n_restored += 1
                logger.info("  [%s] Block %d/%d: %d Posen aus %s uebernommen",
                            target.name, idx + 1, len(blocks),
                            len(restored), part.name)
                continue
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("  [%s] Block %d unlesbar (%s) – wird neu gescort.",
                               target.name, idx + 1, exc)

        logger.info("  [%s] Block %d/%d: %d Liganden werden gescort",
                    target.name, idx + 1, len(blocks), len(block))
        poses = rescore_target(
            target=target,
            target_results_dir=target_results_dir,
            rescore_cfg=rescore_cfg,
            logger=logger,
            files=block,
            poses_only=True,
        )
        if not poses:
            logger.warning("  [%s] Block %d lieferte keine Posen.",
                           target.name, idx + 1)
            continue

        # ── Abdeckungspruefung vor dem Sichern ──
        # Ein Block, dem Scores fehlen, darf NICHT als erledigt gesichert
        # werden: der Blockmodus ueberspringt ihn beim Neustart und der
        # Verlust waere dauerhaft. Am 21.08.2026 wurde so ein Block mit
        # 17 % Abdeckung gesichert (CUDA-OOM verwarf ganze Batches still).
        # Posen ohne Wert bekommen in ecr.py keinen Rang und tragen 0 bei –
        # sie werden also benachteiligt, nicht neutral behandelt.
        _cov_problem = None
        for _key in _active_keys_for_coverage(rescore_cfg):
            _attr = ecr_mod.FIELDS[_key][0]
            _have = sum(1 for _p in poses
                        if getattr(_p, _attr, None) is not None)
            _frac = _have / len(poses)
            if _frac < rescore_cfg.min_block_coverage:
                _cov_problem = (_key, _have, len(poses), _frac)
                break

        if _cov_problem:
            _key, _have, _tot, _frac = _cov_problem
            logger.error(
                "  [%s] Block %d NICHT gesichert: nur %d von %d Posen haben "
                "%s (%.1f %%, Minimum %.0f %%). Wahrscheinlich CUDA-OOM – "
                "rescore_batch_size verkleinern.",
                target.name, idx + 1, _have, _tot, _key,
                100 * _frac, 100 * rescore_cfg.min_block_coverage)
            raise RuntimeError(
                f"Block {idx + 1} von {target.name} unvollstaendig: "
                f"{_frac:.1%} Abdeckung bei '{_key}'. Abbruch, damit der "
                f"Block beim naechsten Lauf neu gescort wird."
            )

        _write_partial(poses, part)
        all_poses.extend(poses)
        logger.info("  [%s] Block %d gesichert: %s (%d Posen)",
                    target.name, idx + 1, part.name, len(poses))

    if n_restored:
        logger.info("  [%s] %d von %d Bloecken stammten aus einem frueheren Lauf.",
                    target.name, n_restored, len(blocks))

    if not all_poses:
        logger.error("  [%s] Keine Posen – Rescoring leer.", target.name)
        return []

    # Aktive Scores wie im Einbahn-Pfad bestimmen
    active: list[str] = []
    if rescore_cfg.vina_enabled:          active.append("vina")
    if rescore_cfg.vinardo_enabled:       active.append("vinardo")
    if rescore_cfg.ad4_enabled:           active.append("ad4")
    if rescore_cfg.cnnaffinity_enabled:   active.append("cnnaffinity")
    if rescore_cfg.cnnscore_enabled:      active.append("cnnscore")
    if rescore_cfg.deltalinf9xgb_enabled: active.append("deltalinf9xgb")
    if rescore_cfg.dense_enabled:
        active += ["dense_cnnaffinity", "dense_cnnscore"]

    return _rank_and_write(all_poses, active, target, target_results_dir,
                           rescore_cfg, logger)


# ======================================================================
# LOG-AUSGABE: TOP 10
# ======================================================================

def log_top10_ecr(
    results:     list[LigandResult],
    target_name: str,
    logger:      logging.Logger,
) -> None:
    """Gibt die Top-10 Liganden nach ECR-Score ins Log aus."""
    top10 = results[:10]
    if not top10:
        return

    logger.info("  --- ECR TOP 10 fuer %s ---", target_name)
    logger.info(
        "  %-4s  %-25s  %-10s  %-6s  %s",
        "Rang", "Ligand", "ECR-Score", "Pose", "Vina [kcal/mol]",
    )
    logger.info("  " + "-" * 60)
    for lr in top10:
        vina = f"{lr.score_vina_best:+.2f}" if lr.score_vina_best is not None else "     N/A"
        logger.info(
            "  #%-3d  %-25s  %-10.4f  %-6d  %s",
            lr.ecr_rank, lr.ligand, lr.ecr_score, lr.best_pose, vina,
        )


# ======================================================================
# CSV-HILFSFUNKTIONEN
# ======================================================================

def _f(v: Optional[float], n: int = 4) -> str:
    """Formatiert optionalen Float fuer CSV; None → leerer String."""
    return "" if v is None else f"{v:.{n}f}"


def _write_pose_csv(
    poses: list[PoseResult], outdir: Path, target_name: str
) -> Path:
    """
    Schreibt die Pose-Level-Ergebnisse.
    Dateiname: rescoring_poses_<target>.csv
    """
    path = outdir / f"rescoring_poses_{target_name}.csv"
    fields = [
        "ligand", "pose",
        "score_vina",  "score_vinardo",  "score_ad4",
        "score_cnnaffinity",  "score_cnnscore",
        "score_deltalinf9xgb",
        "score_dense_cnnaffinity", "score_dense_cnnscore",
        "rank_vina",   "rank_vinardo",   "rank_ad4",
        "rank_cnnaffinity",   "rank_cnnscore",
        "rank_deltalinf9xgb",
        "rank_dense_cnnaffinity", "rank_dense_cnnscore",
        "ecr_vina",    "ecr_vinardo",    "ecr_ad4",
        "ecr_cnnaffinity",    "ecr_cnnscore",
        "ecr_deltalinf9xgb",
        "ecr_dense_cnnaffinity", "ecr_dense_cnnscore",
        "ecr_total",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in sorted(poses, key=lambda x: (x.ligand, x.pose)):
            w.writerow({
                "ligand":                 p.ligand,
                "pose":                   p.pose,
                "score_vina":             _f(p.score_vina),
                "score_vinardo":          _f(p.score_vinardo),
                "score_ad4":              _f(p.score_ad4),
                "score_cnnaffinity":      _f(p.score_cnnaffinity),
                "score_cnnscore":         _f(p.score_cnnscore),
                "score_deltalinf9xgb":    _f(p.score_deltalinf9xgb),
                "score_dense_cnnaffinity":_f(p.score_dense_cnnaffinity),
                "score_dense_cnnscore":   _f(p.score_dense_cnnscore),
                "rank_vina":             p.rank_vina              if p.rank_vina              is not None else "",
                "rank_vinardo":          p.rank_vinardo           if p.rank_vinardo           is not None else "",
                "rank_ad4":              p.rank_ad4               if p.rank_ad4               is not None else "",
                "rank_cnnaffinity":      p.rank_cnnaffinity       if p.rank_cnnaffinity       is not None else "",
                "rank_cnnscore":         p.rank_cnnscore          if p.rank_cnnscore          is not None else "",
                "rank_deltalinf9xgb":    p.rank_deltalinf9xgb     if p.rank_deltalinf9xgb     is not None else "",
                "rank_dense_cnnaffinity":p.rank_dense_cnnaffinity if p.rank_dense_cnnaffinity is not None else "",
                "rank_dense_cnnscore":   p.rank_dense_cnnscore    if p.rank_dense_cnnscore    is not None else "",
                "ecr_vina":              f"{p.ecr_vina:.6f}",
                "ecr_vinardo":           f"{p.ecr_vinardo:.6f}",
                "ecr_ad4":               f"{p.ecr_ad4:.6f}",
                "ecr_cnnaffinity":       f"{p.ecr_cnnaffinity:.6f}",
                "ecr_cnnscore":          f"{p.ecr_cnnscore:.6f}",
                "ecr_deltalinf9xgb":     f"{p.ecr_deltalinf9xgb:.6f}",
                "ecr_dense_cnnaffinity": f"{p.ecr_dense_cnnaffinity:.6f}",
                "ecr_dense_cnnscore":    f"{p.ecr_dense_cnnscore:.6f}",
                "ecr_total":             f"{p.ecr_total:.6f}",
            })
    return path


def _write_ligand_csv(
    ligands: list[LigandResult], outdir: Path, target_name: str
) -> Path:
    """
    Schreibt die Liganden-Rangliste (Hauptergebnis).
    Dateiname: rescoring_ligands_<target>.csv
    Sortierung: absteigend nach ECR-Score (bester Ligand zuerst).
    """
    path   = outdir / f"rescoring_ligands_{target_name}.csv"
    fields = ["ecr_rank", "ligand", "ecr_score", "best_pose",
              "score_vina_best", "score_vinardo_best", "score_ad4_best",
              "score_cnnaffinity_best", "score_cnnscore_best",
              "score_deltalinf9xgb_best",
              "score_dense_cnnaffinity_best", "score_dense_cnnscore_best",
              "active"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lr in ligands:
            if lr.ligand.endswith("_a"):
                active = 1
            elif lr.ligand.endswith("_d"):
                active = 0
            else:
                active = ""
            w.writerow({
                "ecr_rank":               lr.ecr_rank,
                "ligand":                 lr.ligand,
                "ecr_score":              f"{lr.ecr_score:.6f}",
                "best_pose":              lr.best_pose,
                "score_vina_best":        _f(lr.score_vina_best),
                "score_vinardo_best":     _f(lr.score_vinardo_best),
                "score_ad4_best":         _f(lr.score_ad4_best),
                "score_cnnaffinity_best": _f(lr.score_cnnaffinity_best),
                "score_cnnscore_best":    _f(lr.score_cnnscore_best),
                "score_deltalinf9xgb_best":    _f(lr.score_deltalinf9xgb_best),
                "score_dense_cnnaffinity_best": _f(lr.score_dense_cnnaffinity_best),
                "score_dense_cnnscore_best":    _f(lr.score_dense_cnnscore_best),
                "active":                 active,
            })
    return path


# ======================================================================
# STANDALONE-HILFSFUNKTIONEN
# ======================================================================

def _parse_target_config_standalone(
    config_file: Path, target_dir: Path
) -> tuple[list[_TargetInfo], list[str]]:
    """
    Minimaler Target-Parser fuer den Standalone-Betrieb.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"config.txt nicht gefunden: {config_file}")

    targets:  list[_TargetInfo] = []
    warnings: list[str]         = []
    current:  dict              = {}

    def flush(cur, lineno):
        if not cur:
            return None
        missing = [k for k in ("name", "center", "box_size") if k not in cur]
        if missing:
            raise ValueError(f"Block nahe Zeile {lineno} unvollstaendig: {missing}")
        pdbqt = target_dir / f"{cur['name']}.pdbqt"
        if not pdbqt.exists():
            warnings.append(
                f"WARNUNG: PDBQT fuer '{cur['name']}' fehlt – uebersprungen."
            )
            return None
        return _TargetInfo(cur["name"], pdbqt, cur["center"], cur["box_size"])

    lines = config_file.read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            if current:
                r = flush(current, lineno)
                if r:
                    targets.append(r)
                current = {}
            continue
        if line.upper().startswith("CENTER"):
            m = re.search(r"\[([^\]]+)\]", line)
            if not m:
                raise ValueError(f"Zeile {lineno}: Ungueltiges CENTER-Format")
            current["center"] = [float(x) for x in m.group(1).split(",")]
            continue
        if line.upper().startswith("BOX_SIZE"):
            m = re.search(r"\[([^\]]+)\]", line)
            if not m:
                raise ValueError(f"Zeile {lineno}: Ungueltiges BOX_SIZE-Format")
            current["box_size"] = [float(x) for x in m.group(1).split(",")]
            continue
        if re.match(r"^[\w\-]+$", line):
            if "name" in current:
                r = flush(current, lineno)
                if r:
                    targets.append(r)
                current = {}
            current["name"] = line
        else:
            raise ValueError(f"Zeile {lineno}: Unbekanntes Format: '{line}'")

    if current:
        r = flush(current, len(lines))
        if r:
            targets.append(r)

    return targets, warnings


def _setup_logger(log_dir: Path) -> logging.Logger:
    """Logger fuer Standalone-Betrieb. Schreibt rescore.log + stdout."""
    logger = logging.getLogger("docking_rescore")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / "rescore.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ======================================================================
# STANDALONE-MAIN
# ======================================================================

def main() -> None:
    """
    Standalone-Rescoring aller bereits gedockten Liganden.

    Voraussetzung: docking_pipeline.py wurde ausgefuehrt und
    _docked.pdbqt-Dateien liegen in ./RESULTS/<target>/.
    """
    ini = PIPELINE_CONFIG_FILE
    if not ini.exists():
        print(f"FEHLER: pipeline_config.ini nicht gefunden: {ini}", file=sys.stderr)
        sys.exit(1)

    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read(ini, encoding="utf-8")

    def req(s, k):
        try:
            return p.get(s, k)
        except (configparser.NoSectionError, configparser.NoOptionError):
            raise KeyError(f"Pflichtparameter '[{s}] {k}' fehlt.")

    try:
        paths = {
            "target_dir":  Path(req("PATHS", "target_dir")),
            "results_dir": Path(req("PATHS", "results_dir")),
            "log_dir":     Path(req("PATHS", "log_dir")),
        }
        cfg = RescoringConfig.from_ini(ini)
    except (KeyError, Exception) as exc:
        print(f"FEHLER Konfiguration: {exc}", file=sys.stderr)
        sys.exit(1)

    if not cfg.enabled:
        print("Rescoring deaktiviert (pipeline_config.ini: [RESCORE] enabled=false).")
        sys.exit(0)

    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(paths["log_dir"])

    logger.info("=== RESCORING GESTARTET: %s ===",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("gnina verfuegbar: %s", _GNINA_OK)
    logger.info("Aktive Module: Vina=ja | CNNaffinity=%s | CNNscore=%s",
                cfg.cnnaffinity_enabled, cfg.cnnscore_enabled)

    try:
        targets, warnings = _parse_target_config_standalone(
            paths["target_dir"] / "config.txt", paths["target_dir"]
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Fehler beim Lesen der config.txt: %s", exc)
        sys.exit(1)

    for w in warnings:
        logger.warning(w)

    if not targets:
        logger.error("Keine gueltigen Targets – Abbruch.")
        sys.exit(1)

    logger.info("%d Target(s): %s", len(targets), ", ".join(t.name for t in targets))

    for idx, target in enumerate(targets, 1):
        t0 = datetime.now()
        logger.info("=== TARGET %d/%d: %s ===", idx, len(targets), target.name)

        tdir = paths["results_dir"] / target.name
        if not tdir.exists():
            logger.warning("  Kein RESULTS-Verzeichnis fuer '%s' – uebersprungen.",
                           target.name)
            continue

        try:
            results = rescore_target(target, tdir, cfg, logger)
        except Exception as exc:
            logger.error("  Fehler bei '%s': %s", target.name, exc, exc_info=True)
            continue

        log_top10_ecr(results, target.name, logger)
        logger.info("  Laufzeit: %s", str(datetime.now() - t0).split(".")[0])

    logger.info("=== RESCORING ABGESCHLOSSEN ===")


if __name__ == "__main__":
    main()
