"""
pipeline_common.py
==================
Gemeinsame Basis fuer alle drei Pipeline-Stufen (Conversion, Docking,
Rescoring). Liegt in jedem der drei Container unter /app/.

Enthaelt bewusst NUR das, was wirklich alle Stufen brauchen:

  - PIPELINE_CONFIG_FILE  : kanonischer INI-Pfad (/app/pipeline_config.ini)
  - require()             : Pflichtparameter aus INI lesen
  - load_ini()            : ConfigParser mit den Pipeline-Konventionen
  - TargetConfig          : Rezeptor-Definition
  - parse_target_config() : Parser fuer ./TARGET/config.txt
  - setup_logging()       : Logger mit Konsole + Datei-Handler

Keine Stufe importiert die Config einer anderen Stufe. Wer
Docking-Parameter braucht, importiert docking_config; wer
Rescoring-Parameter braucht, importiert docking_rescore. Dadurch
funktioniert jeder Container ohne die Module der anderen Stufen.

ZUR INI-KONVENTION
------------------
Jeder Container sieht seine INI unter demselben Pfad
(/app/pipeline_config.ini), bekommt aber per --bind eine andere
Host-Datei:

    convert.ini   → Conversion-Container
    docking.ini   → Docking-Container
    rescore.ini   → Rescoring-Container

So bleibt der Code stufenunabhaengig und es gibt keine Sektionen,
die eine Stufe liest aber nicht braucht.
"""

from __future__ import annotations

import configparser
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ======================================================================
# INI-BASIS
# ======================================================================

PIPELINE_CONFIG_FILE = Path(__file__).parent / "pipeline_config.ini"


def load_ini(ini_path: Path = PIPELINE_CONFIG_FILE) -> configparser.ConfigParser:
    """ConfigParser mit den Pipeline-Konventionen (Inline-Kommentare)."""
    if not ini_path.exists():
        raise FileNotFoundError(
            f"INI nicht gefunden: {ini_path}\n"
            f"Der Orchestrator bindet die stufenspezifische INI nach "
            f"/app/pipeline_config.ini – fehlt der --bind, tritt dieser "
            f"Fehler auf."
        )
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read(ini_path, encoding="utf-8")
    return p


def require(parser: configparser.ConfigParser, section: str, key: str) -> str:
    """Liest einen Pflicht-Parameter – KeyError wenn fehlend."""
    try:
        return parser.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        raise KeyError(
            f"Pflichtparameter '[{section}] {key}' fehlt in der INI"
        )


# ======================================================================
# TARGET-KONFIGURATION
# ======================================================================

@dataclass
class TargetConfig:
    """Enthaelt alle Docking-Parameter fuer einen einzelnen Rezeptor."""
    name:           str
    pdbqt_path:     Path
    center:         list[float]
    box_size:       list[float]
    ligand_subdir:  str | None = None   # target-spezifische Liganden (DUD-E)


def parse_target_config(
    config_file: Path,
    target_dir:  Path,
) -> tuple[list[TargetConfig], list[str]]:
    if not config_file.exists():
        raise FileNotFoundError(
            f"Target-Konfigurationsdatei nicht gefunden: {config_file}\n"
            f"Bitte config.txt im Ordner {target_dir} anlegen."
        )

    targets:  list[TargetConfig] = []
    warnings: list[str]          = []
    current:  dict               = {}

    def flush_block(cur: dict, line_hint: int) -> TargetConfig | None:
        if not cur:
            return None
        missing = [k for k in ("name", "center", "box_size") if k not in cur]
        if missing:
            raise ValueError(
                f"Unvollstaendiger Target-Block nahe Zeile {line_hint}: "
                f"Fehlende Felder: {missing}"
            )
        pdbqt_path = target_dir / f"{cur['name']}.pdbqt"
        if not pdbqt_path.exists():
            warnings.append(
                f"WARNUNG: PDBQT nicht gefunden fuer '{cur['name']}' "
                f"(erwartet: {pdbqt_path}) – Target wird uebersprungen."
            )
            return None
        return TargetConfig(
            name=cur["name"],
            pdbqt_path=pdbqt_path,
            center=cur["center"],
            box_size=cur["box_size"],
            ligand_subdir=cur.get("ligand_subdir", None),
        )

    lines = config_file.read_text(encoding="utf-8").splitlines()

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if line.startswith("#"):
            continue

        if not line:
            if current:
                result = flush_block(current, lineno)
                if result:
                    targets.append(result)
                current = {}
            continue

        if line.upper().startswith("CENTER"):
            match = re.search(r"\[([^\]]+)\]", line)
            if not match:
                raise ValueError(
                    f"Zeile {lineno}: Ungueltiges CENTER-Format: '{line}'"
                )
            current["center"] = [float(x) for x in match.group(1).split(",")]
            continue

        if line.upper().startswith("BOX_SIZE"):
            match = re.search(r"\[([^\]]+)\]", line)
            if not match:
                raise ValueError(
                    f"Zeile {lineno}: Ungueltiges BOX_SIZE-Format: '{line}'"
                )
            current["box_size"] = [float(x) for x in match.group(1).split(",")]
            continue

        if "=" in line and line.upper().startswith("LIGAND_SUBDIR"):
            current["ligand_subdir"] = line.split("=", 1)[1].strip()
            continue

        if re.match(r"^[\w\-]+$", line):
            if "name" in current:
                result = flush_block(current, lineno)
                if result:
                    targets.append(result)
                current = {}
            current["name"] = line
        else:
            raise ValueError(f"Zeile {lineno}: Unbekanntes Format: '{line}'")

    if current:
        result = flush_block(current, len(lines))
        if result:
            targets.append(result)

    return targets, warnings


# ======================================================================
# LIGANDEN-SUCHE
# ======================================================================

def find_ligand_files(
    pdbqt_dir:     Path,
    ligand_subdir: str | None = None,
) -> tuple[list[Path], list[str]]:
    """
    Sucht Liganden-PDBQTs REKURSIV.

    Grund: sdf_to_pdbqt.py verteilt die Ausgabe standardmaessig auf
    Unterordner 0000/, 0001/, ... (je 10.000 Dateien, gegen inode-Engpaesse).
    Ein flaches pdbqt_dir.glob("*.pdbqt") findet in diesem Layout GENAU NULL
    Liganden – ohne Fehlermeldung, die Pipeline meldet nur "keine Liganden".

    ligand_subdir (DUD-E-Validierung) wird weiterhin unterstuetzt und
    ebenfalls rekursiv durchsucht; ist er leer, wird auf pdbqt_dir
    zurueckgefallen.

    Rueckgabe: (sortierte Dateiliste, Warnungen)
    """
    warnings: list[str] = []

    search_root = pdbqt_dir
    if ligand_subdir:
        candidate = pdbqt_dir / ligand_subdir
        if candidate.is_dir():
            search_root = candidate
        else:
            warnings.append(
                f"WARNUNG: ligand_subdir '{ligand_subdir}' nicht gefunden "
                f"({candidate}) – nutze {pdbqt_dir}."
            )

    files = sorted(search_root.rglob("*.pdbqt"))

    if not files:
        warnings.append(f"WARNUNG: Keine PDBQTs unter {search_root} gefunden.")
        return files, warnings

    # Kollisionspruefung: die Ergebnisse landen spaeter FLACH in
    # RESULTS/<target>/<stem>_docked.pdbqt. Zwei Liganden mit gleichem
    # Stem in verschiedenen Unterordnern wuerden sich dort gegenseitig
    # ueberschreiben – ein stiller Datenverlust.
    seen: dict[str, Path] = {}
    duplicates: list[tuple[Path, Path]] = []
    for f in files:
        first = seen.get(f.stem)
        if first is None:
            seen[f.stem] = f
        else:
            duplicates.append((first, f))

    if duplicates:
        warnings.append(
            f"WARNUNG: {len(duplicates)} doppelte Liganden-Namen gefunden – "
            f"die Ergebnisse wuerden sich in RESULTS/ ueberschreiben. "
            f"Beispiel: {duplicates[0][0]} vs {duplicates[0][1]}"
        )

    return files, warnings


# ======================================================================
# LOGGING
# ======================================================================

def setup_logging(
    log_dir:      Path,
    logger_name:  str = "docking_pipeline",
    log_filename: str = "pipeline.log",
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / log_filename, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
