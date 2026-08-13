"""
gnina_refinement.py
====================
GPU-beschleunigtes Refinement-Redocking der besten ECR-Liganden.

Dieses Modul nimmt die Top-N% Liganden aus dem ECR-Ranking und
fuehrt ein Refinement-Redocking mit dem GNINA CLI Binary durch.
GNINA optimiert die Posen unter Verwendung des CNN-Scoring-Modells
via lokalen Gradient-Descent (--local_only oder --minimize).

Ablauf
------
1. ECR-Liganden-CSV lesen → Top 15% (konfigurierbar) auswaehlen
2. Fuer jeden selektierten Liganden:
   a) Beste Docking-Pose aus _docked.pdbqt extrahieren
   b) GNINA refinement: --local_only / --minimize
      → CNN-gestuetztes Gradient-Descent auf der Pose
   c) Neue Scores (CNNscore, CNNaffinity, Vina) einsammeln
3. Re-Ranking nach gewichteter Kombination (CNN + Vina)
4. Ergebnis-CSV schreiben:
   ./RESULTS/<target>/refinement_<target>.csv

Multi-GPU-Parallelisierung
--------------------------
Refinement wird auf TARGET-Ebene parallelisiert – identisch zum
Rescoring im Orchestrator:
  - Orchestrator verteilt Targets auf GPUs via WORKER_TARGET + CUDA_VISIBLE_DEVICES
  - Jeder Worker-Container fuehrt refine_target() fuer sein Target aus
  - Innerhalb eines Targets laeuft das Refinement sequentiell auf einer GPU
    (gnina nutzt die GPU intern, externe Parallelisierung waere kontraproduktiv)

Integration in die Pipeline
----------------------------
1. worker_gpu.py:  Nach dem Rescoring refine_target() aufrufen
2. orchestrator.py: run_refinement() analog zu run_rescoring() einfuegen
3. docking_pipeline.py: Nach rescore_target() das Refinement einfuegen

Standalone-Aufruf
-----------------
  python gnina_refinement.py

Setzt voraus dass das Rescoring bereits ausgefuehrt wurde und
rescoring_ligands_<target>.csv in ./RESULTS/<target>/ vorhanden ist.

Konfiguration (pipeline_config.ini)
-------------------------------------
[REFINEMENT]
enabled             = true
top_fraction        = 0.15
refinement_mode     = local_only    # local_only | minimize | autobox
cnn_model           = crossdock_default2018_ensemble
autobox_extend      = 4.0
gnina_binary        = /usr/local/bin/gnina
use_gpu             = true
exhaustiveness      = 0
num_modes           = 1
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
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Konfigurationspfad (gleich wie restliche Pipeline)
# ---------------------------------------------------------------------------
PIPELINE_CONFIG_FILE = Path(__file__).parent / "pipeline_config.ini"

# ---------------------------------------------------------------------------
# gnina CLI: auto-detect
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


# ======================================================================
# KONFIGURATION
# ======================================================================

@dataclass
class RefinementConfig:
    """
    Refinement-Parameter aus [REFINEMENT] in pipeline_config.ini.
    """
    enabled:          bool  = False
    top_fraction:     float = 0.15      # Top 15%
    refinement_mode:  str   = "local_only"  # local_only | minimize | autobox
    cnn_model:        str   = "crossdock_default2018_ensemble"
    autobox_extend:   float = 4.0       # Angstrom fuer autobox
    use_gpu:          bool  = True
    gnina_binary:     str   = ""
    gpu_id:           Optional[int] = None
    exhaustiveness:   int   = 0         # 0 = nur lokale Optimierung
    num_modes:        int   = 1

    @classmethod
    def from_ini(cls, ini_path: Path = PIPELINE_CONFIG_FILE) -> "RefinementConfig":
        """Laedt [REFINEMENT]-Block aus INI. Gibt Defaults wenn Sektion fehlt."""
        p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        p.read(ini_path, encoding="utf-8")
        s = "REFINEMENT"

        if not p.has_section(s):
            return cls()

        gnina_bin_raw = p.get(s, "gnina_binary", fallback="").strip()
        gnina_bin = str(Path(gnina_bin_raw).expanduser()) if gnina_bin_raw else ""

        return cls(
            enabled         = p.getboolean(s, "enabled",         fallback=False),
            top_fraction    = p.getfloat(  s, "top_fraction",    fallback=0.15),
            refinement_mode = p.get(       s, "refinement_mode", fallback="local_only"),
            cnn_model       = p.get(       s, "cnn_model",
                                    fallback="crossdock_default2018_ensemble"),
            autobox_extend  = p.getfloat(  s, "autobox_extend",  fallback=4.0),
            use_gpu         = p.getboolean(s, "use_gpu",         fallback=True),
            gnina_binary    = gnina_bin,
            gpu_id          = None,  # Wird zur Laufzeit gesetzt
            exhaustiveness  = p.getint(    s, "exhaustiveness",  fallback=0),
            num_modes       = p.getint(    s, "num_modes",       fallback=1),
        )


# ======================================================================
# DATENKLASSEN
# ======================================================================

@dataclass
class RefinementResult:
    """Ergebnis des Refinements fuer einen einzelnen Liganden."""
    ligand:            str
    ecr_rank:          int          # Original-ECR-Rang
    ecr_score:         float        # Original-ECR-Score

    # Refinement-Scores (nach lokaler Optimierung)
    refined_vina:        Optional[float] = None   # kcal/mol
    refined_cnnscore:    Optional[float] = None   # 0-1
    refined_cnnaffinity: Optional[float] = None   # pKd

    # Original-Scores (Vergleich)
    original_vina:       Optional[float] = None
    original_cnnscore:   Optional[float] = None
    original_cnnaffinity:Optional[float] = None

    # Refinement-Metadaten
    refinement_mode:   str  = ""
    refined_pose_file: Optional[Path] = None

    # Ranking nach Refinement
    refined_rank:      int  = 0
    combined_score:    float = 0.0

    @property
    def delta_vina(self) -> Optional[float]:
        """Aenderung Vina-Score (negativ = Verbesserung)."""
        if self.refined_vina is not None and self.original_vina is not None:
            return self.refined_vina - self.original_vina
        return None

    @property
    def delta_cnnaffinity(self) -> Optional[float]:
        """Aenderung CNNaffinity (positiv = Verbesserung)."""
        if self.refined_cnnaffinity is not None and self.original_cnnaffinity is not None:
            return self.refined_cnnaffinity - self.original_cnnaffinity
        return None


# ======================================================================
# SCHRITT 1: TOP-N% LIGANDEN AUS ECR-CSV LESEN
# ======================================================================

def select_top_ligands(
    ligand_csv: Path,
    top_fraction: float,
    logger: logging.Logger,
) -> list[dict]:
    """
    Liest die ECR-Liganden-CSV und waehlt die besten N% aus.

    Erwartet Spalten: ecr_rank, ligand, ecr_score, best_pose,
                      score_vina_best, score_cnnaffinity_best, score_cnnscore_best

    Die CSV ist bereits nach ECR-Score absteigend sortiert (Rang 1 = bester).

    Rueckgabe: Liste von Dicts mit den Liganden-Informationen.
    """
    if not ligand_csv.exists():
        logger.error("ECR-CSV nicht gefunden: %s", ligand_csv)
        return []

    ligands = []
    with open(ligand_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ligands.append(row)

    if not ligands:
        return []

    n_select = max(1, int(math.ceil(len(ligands) * top_fraction)))
    selected = ligands[:n_select]

    logger.info("  Refinement: %d/%d Liganden selektiert (Top %.0f%%)",
                n_select, len(ligands), top_fraction * 100)

    return selected


# ======================================================================
# SCHRITT 2: BESTE POSE EXTRAHIEREN
# ======================================================================

def _extract_best_pose(
    docked_pdbqt: Path,
    best_pose: int,
) -> Optional[Path]:
    """
    Extrahiert eine bestimmte Pose (1-basiert) aus einer Multi-Pose PDBQT.

    Schreibt die Pose OHNE MODEL/ENDMDL in eine Temp-Datei
    (gnina erwartet Einzelmolekuele im --local_only / --minimize Modus).

    Rueckgabe: Pfad zur Temp-Datei oder None bei Fehler.
    """
    if not docked_pdbqt.exists():
        return None

    current_model = 0
    pose_lines: list[str] = []
    found = False

    with open(docked_pdbqt, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                current_model += 1
                if current_model == best_pose:
                    found = True
                continue
            if line.startswith("ENDMDL"):
                if found:
                    break
                continue
            if found:
                pose_lines.append(line)

    # Keine MODEL/ENDMDL: gesamte Datei als Pose 1
    if not found and best_pose == 1 and current_model == 0:
        with open(docked_pdbqt, encoding="utf-8", errors="replace") as fh:
            pose_lines = fh.readlines()

    if not pose_lines:
        return None

    tmp_dir = tempfile.mkdtemp(prefix=f"refine_{docked_pdbqt.stem}_")
    tmp_path = Path(tmp_dir) / f"pose_{best_pose}.pdbqt"
    with open(tmp_path, "w", encoding="utf-8") as fout:
        fout.writelines(pose_lines)

    return tmp_path


# ======================================================================
# SCHRITT 3: GNINA REFINEMENT (CLI)
# ======================================================================

def _parse_gnina_refinement_output(
    output: str,
) -> Optional[dict[str, Optional[float]]]:
    """
    Parst den gnina --local_only / --minimize Output.

    gnina gibt nach dem Refinement aus (score_only-Format):
      Affinity: -7.234 (kcal/mol)
      CNNscore: 0.82145
      CNNaffinity: 7.51234
      CNNvariance: 0.01234

    Oder im Tabellen-Format (bei --out mit Posen):
      mode |   affinity | intramol | CNN    |  CNN
           | (kcal/mol) |          | score  | affinity
      -----+------------+----------+--------+---------
         1       -7.234      0.000   0.8215     7.512
    """
    vina_score  = None
    cnnscore    = None
    cnnaffinity = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Affinity:"):
            try:
                vina_score = float(line.split(":")[1].split("(")[0].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("CNNscore:"):
            try:
                cnnscore = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("CNNaffinity:"):
            try:
                cnnaffinity = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    # Tabellen-Format parsen wenn Einzel-Format nichts ergeben hat
    if vina_score is None and cnnscore is None:
        for line in output.splitlines():
            match = re.match(
                r"^\s*1\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                line.strip(),
            )
            if match:
                try:
                    vina_score  = float(match.group(1))
                    cnnscore    = float(match.group(3))
                    cnnaffinity = float(match.group(4))
                except ValueError:
                    pass

    if cnnscore is not None or cnnaffinity is not None:
        return {
            "vina": vina_score,
            "cnnscore": cnnscore,
            "cnnaffinity": cnnaffinity,
        }
    return None


def refine_ligand_gnina(
    pose_pdbqt:     Path,
    protein_pdbqt:  Path,
    center:         list[float],
    box_size:       list[float],
    output_pdbqt:   Path,
    cfg:            RefinementConfig,
) -> Optional[dict[str, Optional[float]]]:
    """
    Fuehrt GNINA Refinement fuer eine einzelne Pose durch.

    Modi:
      local_only : --local_only → lokale Minimierung der Startpose
                   (schnellster Modus, nur Gradient-Descent)
      minimize   : --minimize → staerkere Minimierung mit Line-Search
      autobox    : --autobox_ligand → Box automatisch um Ligand
                   (--autobox_extend steuert den Rand in Angstrom)

    GPU-Zuweisung: CUDA_VISIBLE_DEVICES wird vom Orchestrator/Worker
    auf Container-Ebene gesetzt und hier NICHT ueberschrieben.

    Rueckgabe: {vina, cnnscore, cnnaffinity} oder None bei Fehler.
    """
    bin_path = (cfg.gnina_binary
                if cfg.gnina_binary and Path(cfg.gnina_binary).is_file()
                else _GNINA_BIN)
    if not bin_path:
        return None

    cmd = [
        bin_path,
        "-r", str(protein_pdbqt),
        "-l", str(pose_pdbqt),
        "--cnn", cfg.cnn_model,
    ]

    # Refinement-Modus
    if cfg.refinement_mode == "autobox":
        cmd.extend([
            "--autobox_ligand", str(pose_pdbqt),
            "--autobox_extend", str(cfg.autobox_extend),
            "--local_only",
        ])
    else:
        cmd.extend([
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x",   str(box_size[0]),
            "--size_y",   str(box_size[1]),
            "--size_z",   str(box_size[2]),
        ])
        if cfg.refinement_mode == "minimize":
            cmd.append("--minimize")
        else:
            cmd.append("--local_only")

    cmd.extend([
        "--cnn_scoring", "rescore",
        "--num_modes", str(cfg.num_modes),
        "--out", str(output_pdbqt),
    ])

    if cfg.exhaustiveness > 0:
        cmd.extend(["--exhaustiveness", str(cfg.exhaustiveness)])

    if not cfg.use_gpu:
        cmd.append("--no_gpu")

    env = os.environ.copy()
    if cfg.gpu_id is not None and cfg.use_gpu:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        output = result.stdout + result.stderr
        return _parse_gnina_refinement_output(output)

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ======================================================================
# SCHRITT 4: RANKING NACH REFINEMENT
# ======================================================================

def _compute_combined_score(result: RefinementResult) -> float:
    """
    Berechnet einen gewichteten Combined-Score fuer das Ranking.

    Formel: combined = 0.4 * norm_vina + 0.3 * cnnscore + 0.3 * norm_aff

    Alle Scores werden auf [0, 1] normalisiert, hoeher = besser:
      - Vina: invertiert (negativer = besser → wird zu 0-1)
      - CNNscore: direkt (0-1)
      - CNNaffinity: normalisiert auf [0, 1] via x / 10 (capped)

    Bei fehlenden Scores wird das Gewicht auf die vorhandenen verteilt.
    """
    components: list[tuple[float, float]] = []  # (weight, value)

    if result.refined_cnnscore is not None:
        components.append((0.3, result.refined_cnnscore))

    if result.refined_cnnaffinity is not None:
        norm_aff = max(0.0, min(1.0, result.refined_cnnaffinity / 10.0))
        components.append((0.3, norm_aff))

    if result.refined_vina is not None:
        norm_vina = max(0.0, min(1.0, -result.refined_vina / 12.0))
        components.append((0.4, norm_vina))

    if not components:
        return 0.0

    w_sum = sum(w for w, _ in components)
    return sum((w / w_sum) * v for w, v in components)


def rank_refinement_results(
    results: list[RefinementResult],
) -> list[RefinementResult]:
    """
    Berechnet Combined-Score und rankt die Ergebnisse.
    Absteigend nach combined_score (hoeher = besser).
    """
    for r in results:
        r.combined_score = _compute_combined_score(r)

    results.sort(key=lambda r: r.combined_score, reverse=True)
    for i, r in enumerate(results, 1):
        r.refined_rank = i

    return results


# ======================================================================
# SCHRITT 5: CSV SCHREIBEN
# ======================================================================

def _f(v: Optional[float], n: int = 4) -> str:
    """Formatiert optionalen Float; None → leerer String."""
    return "" if v is None else f"{v:.{n}f}"


def write_refinement_csv(
    results:     list[RefinementResult],
    outdir:      Path,
    target_name: str,
) -> Path:
    """
    Schreibt die Refinement-Ergebnisse.
    Dateiname: refinement_<target>.csv
    """
    path = outdir / f"refinement_{target_name}.csv"
    fields = [
        "refined_rank", "ligand", "ecr_rank", "ecr_score",
        "combined_score",
        "refined_vina", "refined_cnnscore", "refined_cnnaffinity",
        "original_vina", "original_cnnscore", "original_cnnaffinity",
        "delta_vina", "delta_cnnaffinity",
        "refinement_mode",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "refined_rank":        r.refined_rank,
                "ligand":              r.ligand,
                "ecr_rank":            r.ecr_rank,
                "ecr_score":           _f(r.ecr_score, 6),
                "combined_score":      _f(r.combined_score, 6),
                "refined_vina":        _f(r.refined_vina),
                "refined_cnnscore":    _f(r.refined_cnnscore),
                "refined_cnnaffinity": _f(r.refined_cnnaffinity),
                "original_vina":       _f(r.original_vina),
                "original_cnnscore":   _f(r.original_cnnscore),
                "original_cnnaffinity":_f(r.original_cnnaffinity),
                "delta_vina":          _f(r.delta_vina),
                "delta_cnnaffinity":   _f(r.delta_cnnaffinity),
                "refinement_mode":     r.refinement_mode,
            })

    return path


def write_refined_poses(
    results:     list[RefinementResult],
    target_name: str,
    outdir:      Path,
    logger:      logging.Logger,
) -> Optional[Path]:
    """
    Sammelt alle verfeinerten Posen in einer kombinierten PDBQT.
    Dateiname: refined_poses_<target>.pdbqt
    """
    out_path = outdir / f"refined_poses_{target_name}.pdbqt"
    count = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in results:
            if r.refined_pose_file and r.refined_pose_file.exists():
                fout.write(f"MODEL     {r.refined_rank}\n")
                fout.write(f"REMARK  LIGAND: {r.ligand}\n")
                fout.write(f"REMARK  ECR_RANK: {r.ecr_rank}\n")
                fout.write(f"REMARK  REFINED_RANK: {r.refined_rank}\n")
                if r.refined_vina is not None:
                    fout.write(f"REMARK VINA RESULT:    {r.refined_vina:.3f}"
                               f"      0.000      0.000\n")
                with open(r.refined_pose_file, encoding="utf-8",
                          errors="replace") as fin:
                    for line in fin:
                        if not line.startswith(("MODEL", "ENDMDL")):
                            fout.write(line)
                fout.write("ENDMDL\n")
                count += 1

    if count == 0:
        out_path.unlink(missing_ok=True)
        return None

    logger.info("  [%s] Verfeinerte Posen: %s (%d Liganden)",
                target_name, out_path.name, count)
    return out_path


# ======================================================================
# LOG-AUSGABE: TOP 10 REFINEMENT
# ======================================================================

def log_top10_refinement(
    results:     list[RefinementResult],
    target_name: str,
    logger:      logging.Logger,
) -> None:
    """Gibt die Top-10 Liganden nach Refinement-Score ins Log aus."""
    top10 = results[:10]
    if not top10:
        return

    logger.info("  --- REFINEMENT TOP 10 fuer %s ---", target_name)
    logger.info(
        "  %-4s  %-25s  %-10s  %-8s  %-10s  %-10s  %s",
        "Rang", "Ligand", "Combined", "ECR-Rang",
        "CNN-Score", "CNN-Aff", "Vina [kcal/mol]",
    )
    logger.info("  " + "-" * 85)
    for r in top10:
        vina = f"{r.refined_vina:+.2f}" if r.refined_vina is not None else "  N/A"
        cs   = f"{r.refined_cnnscore:.4f}" if r.refined_cnnscore is not None else "N/A"
        ca   = f"{r.refined_cnnaffinity:.2f}" if r.refined_cnnaffinity is not None else "N/A"
        logger.info(
            "  #%-3d  %-25s  %-10.4f  %-8d  %-10s  %-10s  %s",
            r.refined_rank, r.ligand, r.combined_score,
            r.ecr_rank, cs, ca, vina,
        )


# ======================================================================
# HAUPTFUNKTION: refine_target()
# ======================================================================

def refine_target(
    target:              object,       # TargetConfig oder _TargetInfo
    target_results_dir:  Path,
    refinement_cfg:      RefinementConfig,
    logger:              logging.Logger,
) -> list[RefinementResult]:
    """
    Fuehrt das Refinement fuer einen Target auf einer GPU durch.

    Multi-GPU-Parallelisierung erfolgt auf TARGET-Ebene:
    Der Orchestrator/Worker verteilt Targets auf GPUs,
    und ruft refine_target() fuer jedes Target einzeln auf.
    Innerhalb dieses Aufrufs laeuft alles sequentiell auf der
    zugewiesenen GPU (gnina nutzt die GPU intern).

    Ablauf:
      1. ECR-CSV lesen → Top N% selektieren
      2. Pro Ligand: beste Pose extrahieren → GNINA Refinement
      3. Ranking nach Combined-Score
      4. CSVs + kombinierte Posen-Datei schreiben

    Rueckgabe: RefinementResult-Liste, absteigend nach Combined-Score.
    """
    # --- gnina pruefen ---
    gnina_bin = (
        refinement_cfg.gnina_binary
        if refinement_cfg.gnina_binary and Path(refinement_cfg.gnina_binary).is_file()
        else _GNINA_BIN
    )
    if not gnina_bin:
        logger.error(
            "  [%s] gnina Binary nicht gefunden – Refinement nicht moeglich. "
            "Bitte gnina_binary in [REFINEMENT] setzen oder gnina in PATH.",
            target.name,
        )
        return []

    # --- ECR-CSV finden ---
    ligand_csv = target_results_dir / f"rescoring_ligands_{target.name}.csv"
    if not ligand_csv.exists():
        logger.warning(
            "  [%s] ECR-CSV nicht gefunden (%s) – Refinement uebersprungen.",
            target.name, ligand_csv.name,
        )
        return []

    # --- Top-N% selektieren ---
    selected = select_top_ligands(
        ligand_csv, refinement_cfg.top_fraction, logger,
    )
    if not selected:
        logger.warning(
            "  [%s] Keine Liganden fuer Refinement selektiert.", target.name,
        )
        return []

    # --- GPU-Info loggen ---
    _env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _env_gpu is not None:
        gpu_info = f"GPU via Container-Env (CUDA_VISIBLE_DEVICES={_env_gpu})"
    elif refinement_cfg.gpu_id is not None:
        gpu_info = f"GPU {refinement_cfg.gpu_id} (explizit)"
    else:
        gpu_info = "GPU 0 (default)"

    logger.info(
        "  [%s] Starte Refinement: %d Liganden | Modus: %s | %s | %s",
        target.name, len(selected), refinement_cfg.refinement_mode,
        refinement_cfg.cnn_model, gpu_info,
    )

    # --- Refinement-Verzeichnis ---
    refine_dir = target_results_dir / "refinement"
    refine_dir.mkdir(parents=True, exist_ok=True)

    # --- Refinement pro Ligand (sequentiell auf einer GPU) ---
    results: list[RefinementResult] = []
    tmp_dirs_to_clean: list[str] = []
    n_total  = len(selected)
    n_ok     = 0
    n_failed = 0
    t_start  = datetime.now()

    for lig_idx, lig_info in enumerate(selected, 1):
        ligand_name = lig_info["ligand"]
        ecr_rank    = int(lig_info.get("ecr_rank", 0))
        ecr_score   = float(lig_info.get("ecr_score", 0.0))
        best_pose   = int(lig_info.get("best_pose", 1))

        # --- Original-Scores aus CSV (fuer Vergleich) ---
        orig_vina = None
        orig_cnnaffinity = None
        orig_cnnscore = None
        try:
            v = lig_info.get("score_vina_best", "")
            orig_vina = float(v) if v else None
        except ValueError:
            pass
        try:
            v = lig_info.get("score_cnnaffinity_best", "")
            # In der ECR-CSV ist CNNaffinity INVERTIERT (-pKd)
            orig_cnnaffinity = -float(v) if v else None
        except ValueError:
            pass
        try:
            v = lig_info.get("score_cnnscore_best", "")
            # In der ECR-CSV ist CNNscore INVERTIERT
            orig_cnnscore = -float(v) if v else None
        except ValueError:
            pass

        # --- Docked-PDBQT finden ---
        docked_path = target_results_dir / f"{ligand_name}_docked.pdbqt"
        if not docked_path.exists():
            logger.debug("  [%s] Docked-PDBQT fehlt: %s",
                         target.name, docked_path.name)
            n_failed += 1
            continue

        # --- Beste Pose extrahieren ---
        pose_file = _extract_best_pose(docked_path, best_pose)
        if pose_file is None:
            logger.debug("  [%s] Pose %d nicht extrahierbar: %s",
                         target.name, best_pose, docked_path.name)
            n_failed += 1
            continue

        tmp_dirs_to_clean.append(str(pose_file.parent))

        # --- Output-Pfad ---
        refined_pdbqt = refine_dir / f"{ligand_name}_refined.pdbqt"

        # --- GNINA Refinement ---
        scores = refine_ligand_gnina(
            pose_pdbqt=pose_file,
            protein_pdbqt=target.pdbqt_path,
            center=target.center,
            box_size=target.box_size,
            output_pdbqt=refined_pdbqt,
            cfg=refinement_cfg,
        )

        # --- Ergebnis zusammenbauen ---
        rr = RefinementResult(
            ligand=ligand_name,
            ecr_rank=ecr_rank,
            ecr_score=ecr_score,
            original_vina=orig_vina,
            original_cnnscore=orig_cnnscore,
            original_cnnaffinity=orig_cnnaffinity,
            refinement_mode=refinement_cfg.refinement_mode,
        )

        if scores is not None:
            rr.refined_vina        = scores.get("vina")
            rr.refined_cnnscore    = scores.get("cnnscore")
            rr.refined_cnnaffinity = scores.get("cnnaffinity")
            if refined_pdbqt.exists():
                rr.refined_pose_file = refined_pdbqt
            n_ok += 1
        else:
            n_failed += 1

        results.append(rr)

        # --- Fortschritt loggen ---
        if lig_idx % 100 == 0 or lig_idx == n_total:
            elapsed = datetime.now() - t_start
            pct = lig_idx / n_total * 100
            eta_s = ((elapsed.total_seconds() / lig_idx)
                     * (n_total - lig_idx)) if lig_idx > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_s))) if eta_s > 0 else "--:--:--"
            logger.info(
                "  [%s] Refinement: %d/%d (%.1f%%) | OK: %d | Fehler: %d "
                "| vergangen: %s | ETA: %s",
                target.name, lig_idx, n_total, pct, n_ok, n_failed,
                str(elapsed).split(".")[0], eta_str,
            )

    # --- Temp-Verzeichnisse aufraeumen ---
    for td in tmp_dirs_to_clean:
        try:
            shutil.rmtree(td, ignore_errors=True)
        except Exception:
            pass

    if not results:
        logger.warning("  [%s] Keine Refinement-Ergebnisse.", target.name)
        return []

    # --- Ranking ---
    results = rank_refinement_results(results)

    # --- CSVs schreiben ---
    csv_path = write_refinement_csv(results, target_results_dir, target.name)
    logger.info("  [%s] Refinement-CSV: %s", target.name, csv_path.name)

    # --- Kombinierte Posen-Datei ---
    write_refined_poses(results, target.name, target_results_dir, logger)

    # --- Statistik ---
    n_improved_vina = sum(
        1 for r in results
        if r.delta_vina is not None and r.delta_vina < -0.1
    )
    n_improved_aff = sum(
        1 for r in results
        if r.delta_cnnaffinity is not None and r.delta_cnnaffinity > 0.1
    )
    logger.info(
        "  [%s] Refinement: %d/%d erfolgreich | "
        "Vina verbessert: %d | CNNaffinity verbessert: %d",
        target.name, n_ok, len(results),
        n_improved_vina, n_improved_aff,
    )

    elapsed_total = datetime.now() - t_start
    logger.info("  [%s] Refinement Laufzeit: %s",
                target.name, str(elapsed_total).split(".")[0])

    return results


# ======================================================================
# STANDALONE-MAIN
# ======================================================================

def _setup_logger(log_dir: Path) -> logging.Logger:
    """Logger fuer Standalone-Betrieb."""
    logger = logging.getLogger("gnina_refinement")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / "refinement.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _parse_target_config_standalone(
    config_file: Path, target_dir: Path,
) -> tuple[list, list[str]]:
    """
    Minimaler Target-Parser fuer Standalone-Betrieb.
    Identische Logik wie in docking_rescore.py.
    """
    from docking_rescore import _TargetInfo

    if not config_file.exists():
        raise FileNotFoundError(f"config.txt nicht gefunden: {config_file}")

    targets  = []
    warnings = []
    current: dict = {}

    def flush(cur, lineno):
        if not cur:
            return None
        missing = [k for k in ("name", "center", "box_size") if k not in cur]
        if missing:
            raise ValueError(
                f"Block nahe Zeile {lineno} unvollstaendig: {missing}"
            )
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
                raise ValueError(
                    f"Zeile {lineno}: Ungueltiges BOX_SIZE-Format"
                )
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


def main() -> None:
    """
    Standalone-Refinement aller bereits rescored Liganden.

    Voraussetzung: Rescoring wurde ausgefuehrt und
    rescoring_ligands_<target>.csv existiert in ./RESULTS/<target>/.

    Unterstuetzt WORKER_TARGET fuer den Orchestrator-Modus:
    Wenn gesetzt wird nur dieses eine Target auf der zugewiesenen
    GPU bearbeitet (identisch zum Rescoring-Worker).
    """
    ini = PIPELINE_CONFIG_FILE
    if not ini.exists():
        print(f"FEHLER: pipeline_config.ini nicht gefunden: {ini}",
              file=sys.stderr)
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
        cfg = RefinementConfig.from_ini(ini)
    except (KeyError, Exception) as exc:
        print(f"FEHLER Konfiguration: {exc}", file=sys.stderr)
        sys.exit(1)

    if not cfg.enabled:
        print("Refinement deaktiviert "
              "(pipeline_config.ini: [REFINEMENT] enabled=false).")
        sys.exit(0)

    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(paths["log_dir"])

    logger.info("=== REFINEMENT GESTARTET: %s ===",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("gnina Binary: %s", _GNINA_BIN or "NICHT GEFUNDEN")
    logger.info("Modus: %s | Top-Fraction: %.0f%% | Modell: %s",
                cfg.refinement_mode, cfg.top_fraction * 100, cfg.cnn_model)

    try:
        targets, warnings = _parse_target_config_standalone(
            paths["target_dir"] / "config.txt", paths["target_dir"],
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Fehler beim Lesen der config.txt: %s", exc)
        sys.exit(1)

    for w in warnings:
        logger.warning(w)

    if not targets:
        logger.error("Keine gueltigen Targets – Abbruch.")
        sys.exit(1)

    # Orchestrator-Modus: nur ein Target bearbeiten
    worker_target = os.environ.get("WORKER_TARGET", "")
    if worker_target:
        targets = [t for t in targets if t.name == worker_target]
        if not targets:
            logger.error("Target '%s' nicht in config.txt.", worker_target)
            sys.exit(1)
        logger.info("Worker-Modus: nur Target '%s'", worker_target)

    logger.info("%d Target(s): %s",
                len(targets), ", ".join(t.name for t in targets))

    for idx, target in enumerate(targets, 1):
        t0 = datetime.now()
        logger.info("=== TARGET %d/%d: %s ===", idx, len(targets), target.name)

        tdir = paths["results_dir"] / target.name
        if not tdir.exists():
            logger.warning(
                "  Kein RESULTS-Verzeichnis fuer '%s' – uebersprungen.",
                target.name,
            )
            continue

        try:
            results = refine_target(target, tdir, cfg, logger)
        except Exception as exc:
            logger.error("  Fehler bei '%s': %s",
                         target.name, exc, exc_info=True)
            continue

        log_top10_refinement(results, target.name, logger)
        logger.info("  Laufzeit: %s", str(datetime.now() - t0).split(".")[0])

    logger.info("=== REFINEMENT ABGESCHLOSSEN ===")


if __name__ == "__main__":
    main()
