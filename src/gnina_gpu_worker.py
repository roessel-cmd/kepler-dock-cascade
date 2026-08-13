"""
gnina_gpu_worker.py
===================
GPU-beschleunigtes CNN-Rescoring mit gninatorch + molgrid.

Ersetzt den gnina-CLI-Subprocess-Pfad in docking_rescore.py:
  - Modell wird EINMAL geladen und auf der GPU gehalten
  - Voxelisierung ueber molgrid.ExampleProvider + GridMaker (GPU)
  - Forward-Pass ueber PyTorch (GPU)
  - Ergebnis: {pose_idx: {"cnnscore": float, "cnnaffinity": float}}

Architektur:
  GninaGPUScorer:
    __init__(gpu_id, cnn_model)  → Modell laden, GPU pinnen
    score_ligand(protein_pdbqt, docked_pdbqt)  → alle Posen scoren
    close()  → Ressourcen freigeben

Wichtig:
  - gninatorch gibt log_CNNscore zurueck, nicht CNNscore!
    → CNNscore = exp(log_CNNscore)
  - Ensemble (crossdock_default2018_ensemble) gibt 3 Werte:
    log_CNNscore, CNNaffinity, CNNvariance
  - molgrid.ExampleProvider liest .types-Dateien (Textformat):
    <receptor_path> <ligand_path>
  - PDBQT wird von OpenBabel/molgrid direkt unterstuetzt

Fallback:
  Wenn gninatorch/molgrid nicht importierbar → GNINATORCH_OK = False
  → docking_rescore.py faellt auf CLI-Modus zurueck

Validierung:
  Scores muessen mit gnina CLI uebereinstimmen (Floating-Point-Toleranz).
  Referenz: rescoring_ligands_hxk4.csv (gnina CLI crossdock_default2018_ensemble)
"""

from __future__ import annotations

import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

_log = logging.getLogger("gnina_gpu_worker")

# ======================================================================
# IMPORT-CHECK: gninatorch + molgrid + torch
# ======================================================================

GNINATORCH_OK = False
_import_error_msg = ""

try:
    import torch
    import molgrid
    import numpy as np
    from gninatorch.gnina import setup_gnina_model
    GNINATORCH_OK = True
except ImportError as exc:
    _import_error_msg = str(exc)
except Exception as exc:
    _import_error_msg = f"Unerwarteter Fehler beim Import: {exc}"


def get_import_status() -> tuple[bool, str]:
    """Gibt (verfuegbar, fehlermeldung) zurueck."""
    return GNINATORCH_OK, _import_error_msg


# ======================================================================
# HILFSFUNKTION: PDBQT in Einzelposen aufsplitten
# ======================================================================

def _split_pdbqt_poses(docked_pdbqt: Path) -> list[Path]:
    """
    Splittet eine Multi-Pose PDBQT-Datei in einzelne Temp-Dateien.

    Jede Pose wird OHNE MODEL/ENDMDL geschrieben (molgrid/OpenBabel
    erwartet Einzelmolekuele). TORSDOF bleibt erhalten.

    Rueckgabe: Liste von Temp-Pfaden (muessen vom Aufrufer aufgeraeumt werden).
    """
    poses: list[list[str]] = []
    current: list[str] = []
    all_lines: list[str] = []
    in_model = False

    with open(docked_pdbqt, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            all_lines.append(line)
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
                current.append(line)

    # Keine MODEL/ENDMDL-Records: gesamte Datei als eine Pose
    if not poses and all_lines:
        poses.append(all_lines)

    if not poses:
        return []

    # Temp-Dateien schreiben
    tmp_dir = tempfile.mkdtemp(prefix=f"gnina_gpu_{docked_pdbqt.stem}_")
    tmp_files: list[Path] = []
    for idx, lines in enumerate(poses, start=1):
        tmp_path = Path(tmp_dir) / f"pose_{idx}.pdbqt"
        with open(tmp_path, "w", encoding="utf-8") as fout:
            fout.writelines(lines)
        tmp_files.append(tmp_path)

    return tmp_files


# ======================================================================
# GNINA GPU SCORER
# ======================================================================

class GninaGPUScorer:
    """
    Persistenter GPU-Scorer mit gninatorch.

    Laedt das CNN-Modell einmal auf die GPU und scored dann
    beliebig viele Protein-Ligand-Paare ohne Neuinitialisierung.

    Usage:
        scorer = GninaGPUScorer(gpu_id=0, cnn_model="crossdock_default2018_ensemble")
        results = scorer.score_ligand(protein_pdbqt, docked_pdbqt)
        # results = {1: {"cnnscore": 0.95, "cnnaffinity": 7.2}, ...}
        scorer.close()
    """

    def __init__(
        self,
        gpu_id: int = 0,
        cnn_model: str = "crossdock_default2018_ensemble",
    ):
        if not GNINATORCH_OK:
            raise RuntimeError(
                f"gninatorch nicht verfuegbar: {_import_error_msg}"
            )

        self.gpu_id = gpu_id
        self.cnn_model = cnn_model
        self.device = f"cuda:{gpu_id}"

        # GPU pinnen
        torch.cuda.set_device(gpu_id)

        # Modell laden
        # setup_gnina_model() gibt (model, is_ensemble) zurueck
        _log.info("GninaGPUScorer: Lade Modell '%s' auf GPU %d ...",
                   cnn_model, gpu_id)
        self.model, self.is_ensemble = setup_gnina_model(cnn_model)
        self.model = self.model.to(self.device)
        self.model.eval()

        # GridMaker – Defaults: resolution=0.5, dimension=23.5 (48 grid points)
        # Das entspricht den gnina-Defaults
        self.gmaker = molgrid.GridMaker()

        # dims werden beim ersten score_ligand() Aufruf bestimmt,
        # da sie von der ExampleProvider-Konfiguration abhaengen
        self.dims = None

        _log.info("GninaGPUScorer: Modell geladen auf GPU %d.", gpu_id)

    def _score_batch(
        self,
        protein_pdbqt: Path,
        pose_files: list[Path],
    ) -> dict[int, dict[str, Optional[float]]]:
        """
        Gemeinsame Scoring-Logik fuer eine Liste von Pose-Dateien.

        Ablauf:
          1. .types-Datei schreiben (pro Pose eine Zeile)
          2. molgrid.ExampleProvider → Batch laden
          3. GridMaker → Voxelisierung auf GPU
          4. Forward-Pass → log_CNNscore, CNNaffinity, CNNvariance
          5. CNNscore = exp(log_CNNscore)

        Rueckgabe: {pose_1basiert: {"cnnscore": float, "cnnaffinity": float}}
        """
        if not pose_files:
            return {}

        tmp_dir: Optional[str] = None

        try:
            # .types-Datei in einem Temp-Verzeichnis
            tmp_dir = tempfile.mkdtemp(prefix="gnina_gpu_types_")
            types_path = Path(tmp_dir) / "scoring.types"

            with open(types_path, "w", encoding="utf-8") as tf:
                for pose_file in pose_files:
                    tf.write(f"{protein_pdbqt.resolve()} {pose_file.resolve()}\n")

            batch_size = len(pose_files)

            # ExampleProvider mit GNINA-Typers
            eprov = molgrid.ExampleProvider(
                molgrid.defaultGninaReceptorTyper,
                molgrid.defaultGninaLigandTyper,
                shuffle=False,
                default_batch_size=batch_size,
            )
            eprov.populate(str(types_path))

            # dims lazy init: beim ersten Aufruf aus dem Provider ableiten
            if self.dims is None:
                self.dims = self.gmaker.grid_dimensions(eprov.num_types())
                _log.info("GninaGPUScorer: Grid dims=%s (num_types=%d)",
                          self.dims, eprov.num_types())

            # Batch laden
            batch = eprov.next_batch(batch_size)

            # Grid-Tensor auf GPU allozieren
            tensor_shape = (batch_size,) + self.dims
            input_tensor = torch.zeros(
                tensor_shape, dtype=torch.float32, device=self.device
            )

            # Voxelisierung: GridMaker.forward(ExampleVec, Tensor, float, bool)
            # Positional-Args fuer C++-Bindings (Keywords nicht immer unterstuetzt)
            self.gmaker.forward(batch, input_tensor, 0.0, False)

            # Forward-Pass
            with torch.no_grad():
                output = self.model(input_tensor)
                # Ensemble-Output (crossdock_default2018_ensemble):
                #   output[0]: shape (batch_size, 2) – [log_CNNscore, CNNaffinity] pro Pose
                #   output[1]: shape (batch_size,)   – CNNaffinity (Ensemble-Mittelwert)
                #   output[2]: shape (batch_size,)   – CNNvariance
                # Einzelmodell-Output:
                #   output[0]: shape (batch_size,)   – log_CNNscore
                #   output[1]: shape (batch_size,)   – CNNaffinity

            # ── Output-Tensors auspacken (Shape-basiert, modellunabhaengig) ──
            # Ensemble (crossdock_default2018_ensemble):
            #   output[0]: (batch, 2) – [log_CNNscore, raw_affinity]
            #   output[1]: (batch,)   – CNNaffinity (Ensemble-Mittelwert, pKd)
            #   output[2]: (batch,)   – CNNvariance
            # Einzelmodell (crossdock_default2018, dense, general_default2018):
            #   output[0]: (batch,)   – log_CNNscore
            #   output[1]: (batch,)   – CNNaffinity
            out0 = output[0].cpu().numpy()

            if out0.ndim == 2:
                # Ensemble: log_CNNscore aus Spalte 0, Affinity aus output[1]
                log_cnnscore_np = out0[:, 0]
                cnnaffinity_np  = output[1].cpu().numpy()
            else:
                # Einzelmodell: beide 1D
                log_cnnscore_np = out0
                cnnaffinity_np  = output[1].cpu().numpy()

            # Ergebnisse zusammenbauen
            results: dict[int, dict[str, Optional[float]]] = {}
            for pose_idx in range(batch_size):
                log_cs = float(log_cnnscore_np[pose_idx])
                aff = float(cnnaffinity_np[pose_idx])

                # CNNscore = exp(log_CNNscore)
                cnnscore = float(np.exp(log_cs))
                # Clampen auf [0, 1]
                cnnscore = max(0.0, min(1.0, cnnscore))

                results[pose_idx + 1] = {
                    "cnnscore": cnnscore,
                    "cnnaffinity": aff,
                }

            return results

        finally:
            if tmp_dir is not None:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    def score_ligand(
        self,
        protein_pdbqt: Path,
        docked_pdbqt: Path,
    ) -> dict[int, dict[str, Optional[float]]]:
        """
        Scored alle Posen eines Liganden.

        Splittet die Multi-Pose PDBQT in Einzelposen und scored den Batch.

        Rueckgabe: {pose_1basiert: {"cnnscore": float, "cnnaffinity": float}}
        """
        if not protein_pdbqt.exists() or not docked_pdbqt.exists():
            return {}

        tmp_files: list[Path] = []
        tmp_dir: Optional[str] = None

        try:
            # Posen splitten
            tmp_files = _split_pdbqt_poses(docked_pdbqt)
            if not tmp_files:
                return {}
            tmp_dir = str(tmp_files[0].parent)

            return self._score_batch(protein_pdbqt, tmp_files)

        except Exception as exc:
            _log.warning("GninaGPUScorer.score_ligand fehlgeschlagen fuer %s: %s",
                         docked_pdbqt.name, exc)
            return {}

        finally:
            # Temp-Verzeichnis der gesplitteten Posen aufraeumen
            if tmp_dir is not None:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    def score_ligand_from_poses(
        self,
        protein_pdbqt: Path,
        pose_files: list[Path],
    ) -> dict[int, dict[str, Optional[float]]]:
        """
        Scored bereits aufgesplittete Pose-Dateien (fuer Integration
        mit dem bestehenden Clustering-Workflow).

        Rueckgabe: {pose_1basiert: {"cnnscore": float, "cnnaffinity": float}}
        """
        if not protein_pdbqt.exists() or not pose_files:
            return {}

        try:
            return self._score_batch(protein_pdbqt, pose_files)
        except Exception as exc:
            _log.warning("GninaGPUScorer.score_ligand_from_poses fehlgeschlagen: %s",
                         exc)
            return {}

    def score_ligands_batch(
        self,
        protein_pdbqt: Path,
        docked_pdbqts: list[Path],
        max_poses_per_batch: int = 0,
    ) -> dict[str, dict[int, dict[str, Optional[float]]]]:
        """
        Scored mehrere Liganden in einem GPU-Batch.

        Sammelt alle Posen aller Liganden, fuehrt sie in grossen Batches
        durch den Forward-Pass und ordnet die Ergebnisse zurueck.

        Dies ist deutlich schneller als score_ligand() in einer Schleife,
        da der GPU-Overhead (Voxelisierung, Kernel-Launch) ueber viele
        Posen amortisiert wird.

        Parameter:
          protein_pdbqt:       Pfad zum Rezeptor-PDBQT
          docked_pdbqts:       Liste der _docked.pdbqt-Dateien
          max_poses_per_batch: Maximale Posen pro GPU-Batch.
                               0 = auto (basierend auf verfuegbarem VRAM).
                               Empfohlene Werte:
                                 RTX 3070 (8 GB):   100
                                 RTX 4090 (24 GB):  500
                                 H100 (80 GB):     2000

        Rueckgabe:
          {ligand_stem: {pose_1basiert: {"cnnscore": float, "cnnaffinity": float}}}
          Leeres dict fuer Liganden die fehlgeschlagen sind.
        """
        if not protein_pdbqt.exists() or not docked_pdbqts:
            return {}

        # ── Auto-Bestimmung der Batch-Size ──
        if max_poses_per_batch <= 0:
            try:
                free_mem, _total_mem = torch.cuda.mem_get_info(self.gpu_id)
                # ~12 MB pro Pose (48^3 * 28 channels * 4 bytes + Overhead)
                # Konservativ: 50% des freien Speichers nutzen
                mem_per_pose = 12 * 1024 * 1024
                max_poses_per_batch = max(10, int(free_mem * 0.5 / mem_per_pose))
                _log.info("GninaGPUScorer: Auto batch_size=%d (%.1f GB frei)",
                          max_poses_per_batch, free_mem / 1e9)
            except Exception:
                max_poses_per_batch = 100  # Sicherer Default

        # ── Phase 1: Alle Posen splitten und Index-Mapping aufbauen ──
        all_pose_files: list[Path] = []
        all_tmp_dirs: list[str] = []
        # Mapping: (start_idx, end_idx, ligand_stem)
        ligand_ranges: list[tuple[int, int, str]] = []

        for docked in docked_pdbqts:
            if not docked.exists():
                continue
            stem = docked.stem.removesuffix("_docked")
            try:
                pose_files = _split_pdbqt_poses(docked)
                if not pose_files:
                    continue
                all_tmp_dirs.append(str(pose_files[0].parent))
                start = len(all_pose_files)
                all_pose_files.extend(pose_files)
                end = len(all_pose_files)
                ligand_ranges.append((start, end, stem))
            except Exception as exc:
                _log.warning("score_ligands_batch: Split fehlgeschlagen "
                             "fuer %s: %s", docked.name, exc)

        if not all_pose_files:
            return {}

        _log.info("GninaGPUScorer: Multi-Ligand-Batch: %d Liganden, "
                   "%d Posen gesamt, batch_size=%d",
                   len(ligand_ranges), len(all_pose_files), max_poses_per_batch)

        # ── Phase 2: In Batches scoren ──
        all_results_flat: list[dict[str, Optional[float]]] = [
            {} for _ in range(len(all_pose_files))
        ]

        try:
            for batch_start in range(0, len(all_pose_files), max_poses_per_batch):
                batch_end = min(batch_start + max_poses_per_batch, len(all_pose_files))
                batch_files = all_pose_files[batch_start:batch_end]

                try:
                    batch_results = self._score_batch(protein_pdbqt, batch_files)
                    # _score_batch gibt 1-basierte Indizes zurueck
                    for local_idx, scores in batch_results.items():
                        global_idx = batch_start + local_idx - 1
                        all_results_flat[global_idx] = scores
                except Exception as exc:
                    _log.warning("score_ligands_batch: Batch %d-%d fehlgeschlagen: %s",
                                 batch_start, batch_end, exc)

            # ── Phase 3: Ergebnisse auf Liganden zurückverteilen ──
            results: dict[str, dict[int, dict[str, Optional[float]]]] = {}
            for start, end, stem in ligand_ranges:
                lig_results: dict[int, dict[str, Optional[float]]] = {}
                for i in range(start, end):
                    pose_idx = i - start + 1  # 1-basiert
                    if all_results_flat[i]:
                        lig_results[pose_idx] = all_results_flat[i]
                results[stem] = lig_results

            return results

        finally:
            # Alle Temp-Verzeichnisse aufraeumen
            for td in all_tmp_dirs:
                try:
                    shutil.rmtree(td, ignore_errors=True)
                except Exception:
                    pass

    def close(self):
        """Gibt GPU-Ressourcen frei."""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log.info("GninaGPUScorer: GPU %d Ressourcen freigegeben.", self.gpu_id)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
