"""
sdf_to_pdbqt.py
================
Standalone-Konverter: Sammel-SDF → PDBQT direkt via RDKit + Meeko.

Hintergrund
-----------
Der Umweg SDF → PDB → PDBQT verliert Bindungsordnungs-Informationen
(PDB hat keine echten Bond-Records). Tools wie AutoDockTools rekonstruieren
die Bindungen anschliessend rein geometrisch aus Atomabstaenden – was bei
ungluecklichen Konformationen zu Fehlern fuehrt (z.B. 5 Bindungen wo nur
4 sein sollten).

Dieses Skript geht stattdessen direkt:
    SDF (gestreamt)  →  (RDKit: H-Add + ggf. ETKDG + UFF-Opt)  →  (Meeko: PDBQT)

RDKit kennt aus dem SDF die Bindungsordnungen explizit, und Meeko schreibt
PDBQT nativ ohne PDB-Zwischenschritt.

Architektur fuer GROSSE Sammel-SDFs (Millionen Molekuele)
---------------------------------------------------------
- Producer:  liest die SDF zeilenweise und splittet an "$$$$"-Trennern
             (RAM-konstant, unabhaengig von Dateigroesse)
- Workers:   N parallele Prozesse, holen sich Eintraege aus einer Queue,
             parsen MolBlock mit RDKit, schreiben PDBQT mit Meeko
- Timeout:   SIGALRM nach --timeout Sekunden pro Molekuel (Linux)
- Output:    PDBQTs verteilt auf Unterordner pdbqt/0000/, 0001/, ...
             je 10.000 Dateien (vermeidet inode-Engpaesse). --flat fuer einen Ordner.

Aufruf
------
    python sdf_to_pdbqt.py \\
        --sdf-file  /pfad/zur/library.sdf \\
        --out-dir   /pfad/zum/PDBQT_NEW \\
        --log-dir   /pfad/zum/LOG \\
        --workers   15 \\
        --timeout   120
"""

from __future__ import annotations

import argparse
import gzip
import logging
import multiprocessing as mp
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Empty


# ======================================================================
# KONSTANTEN
# ======================================================================

QUEUE_MAXSIZE       = 1000     # Producer blockiert wenn Queue voll → Backpressure
LOG_PROGRESS_EVERY  = 100      # Fortschritt alle N erledigte Molekuele loggen
FILES_PER_SUBDIR    = 10000    # Unterordner-Aufteilung im PDBQT-Output
SENTINEL            = None     # Signalisiert Worker: keine Jobs mehr


# ======================================================================
# HILFSFUNKTIONEN
# ======================================================================

def make_output_path(out_dir: Path, mol_name: str, mol_index: int,
                     flat: bool) -> Path:
    """Bestimmt den Ziel-PDBQT-Pfad inklusive Unterordner-Aufteilung."""
    if flat:
        return out_dir / f"{mol_name}.pdbqt"
    subdir_idx = mol_index // FILES_PER_SUBDIR
    return out_dir / f"{subdir_idx:04d}" / f"{mol_name}.pdbqt"


def safe_mol_name(rdkit_name: str, fallback_index: int) -> str:
    """Erzeugt einen dateisystemtauglichen Molekülnamen."""
    if rdkit_name:
        # Whitespace und problematische Zeichen ersetzen
        cleaned = "".join(
            c if (c.isalnum() or c in "._-") else "_"
            for c in rdkit_name.strip()
        )
        if cleaned:
            return cleaned[:60]  # Längen-Cap fuer Dateisysteme
    return f"mol_{fallback_index:07d}"


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ======================================================================
# TIMEOUT VIA SIGALRM (im Worker, nicht im Main!)
# ======================================================================

class ConversionTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ConversionTimeout("SIGALRM Timeout")


# ======================================================================
# WORKER-PROZESS
# ======================================================================

def worker_loop(worker_id: int,
                job_queue: mp.Queue,
                result_queue: mp.Queue,
                out_dir_str: str,
                log_dir_str: str,
                uff_iters: int,
                timeout_s: int,
                flat: bool) -> None:
    """
    Holt Jobs aus der Queue bis SENTINEL, konvertiert, meldet Ergebnis.

    Job:    (mol_index, payload, mol_name, fmt)   fmt ∈ {"sdf", "smiles"}
    Result: (mol_index, mol_name, status, msg)
            status ∈ {"OK", "ERROR", "TIMEOUT"}
    """
    # Lokale Imports — vermeidet teure Imports im Main-Prozess
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    RDLogger.DisableLog("rdApp.*")

    out_dir = Path(out_dir_str)
    log_dir = Path(log_dir_str)

    # SIGALRM-Handler nur im Worker registrieren
    signal.signal(signal.SIGALRM, _alarm_handler)

    # Profiling-Akkumulator: kumulative Zeit pro Stufe ueber 100 Molekuele
    prof = {"parse": 0.0, "salt": 0.0, "addh": 0.0, "embed": 0.0,
            "uff": 0.0, "meeko": 0.0, "write": 0.0}
    prof_count = 0

    while True:
        try:
            job = job_queue.get(timeout=30)
        except Empty:
            # Producer eventuell noch nicht so weit — weiterwarten.
            # Falls Main den Prozess killt, kommt das ueber SIGTERM.
            continue

        if job is SENTINEL:
            break

        mol_index, payload, mol_name, fmt = job
        out_path = make_output_path(out_dir, mol_name, mol_index, flat)

        signal.alarm(timeout_s)  # Stopuhr starten
        try:
            t0 = time.perf_counter()
            if fmt == "smiles":
                mol = Chem.MolFromSmiles(payload)
                if mol is None:
                    raise ValueError(f"SMILES nicht parsebar: {payload[:80]}")
            else:
                mol = Chem.MolFromMolBlock(payload, removeHs=False,
                                           sanitize=True)
                if mol is None:
                    raise ValueError("MolBlock konnte nicht geparst werden")
            t_parse = time.perf_counter()

            # ── SALT-STRIPPING ─────────────────────────────────────────
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(fragments) > 1:
                mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
                Chem.SanitizeMol(mol)
            t_salt = time.perf_counter()

            mol = Chem.AddHs(mol, addCoords=True)
            t_addh = time.perf_counter()

            # ── 3D-EMBEDDING (zweistufig: schnell → robust) ────────────
            conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            needs_embed = conf is None or all(
                abs(conf.GetAtomPosition(i).z) < 1e-6
                for i in range(mol.GetNumAtoms())
            )
            if needs_embed:
                params_fast = AllChem.ETKDGv3()
                params_fast.randomSeed = 42

                if AllChem.EmbedMolecule(mol, params_fast) != 0:
                    params_robust = AllChem.ETKDGv3()
                    params_robust.randomSeed = 0xC0FFEE
                    params_robust.maxIterations = 50
                    params_robust.useRandomCoords = True

                    if AllChem.EmbedMolecule(mol, params_robust) != 0:
                        raise ValueError("3D-Embedding fehlgeschlagen (beide Stufen)")
            t_embed = time.perf_counter()

            AllChem.UFFOptimizeMolecule(mol, maxIters=uff_iters)
            t_uff = time.perf_counter()

            # Makrozyklen starr behandeln statt sie aufzubrechen.
            # Setzt Meeko >= 0.7 voraus (bis 0.6 hiess der Parameter
            # macrocycle_opening=False). Die Version ist in
            # sdf_to_pdbqt.def auf 0.7.1 gepinnt; eine aeltere Meeko
            # scheitert hier sofort mit TypeError statt still anders zu
            # rechnen, was das gewuenschte Verhalten ist.
            prep = MoleculePreparation(rigid_macrocycles=True)
            setups = prep.prepare(mol)
            if not setups:
                raise ValueError("Meeko: kein MoleculeSetup")

            pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
            if not is_ok:
                raise ValueError(f"Meeko-Writer: {err}")
            t_meeko = time.perf_counter()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(pdbqt_str, encoding="utf-8")
            t_write = time.perf_counter()

            signal.alarm(0)  # Stopuhr aus

            # ── Profiling akkumulieren ─────────────────────────────────
            prof["parse"] += t_parse - t0
            prof["salt"]  += t_salt  - t_parse
            prof["addh"]  += t_addh  - t_salt
            prof["embed"] += t_embed - t_addh
            prof["uff"]   += t_uff   - t_embed
            prof["meeko"] += t_meeko - t_uff
            prof["write"] += t_write - t_meeko
            prof_count += 1

            if prof_count >= 100:
                total = sum(prof.values())
                if total > 0:
                    msg = (
                        f"W{worker_id} Profil (100 mol, {total:.1f}s): "
                        f"parse={prof['parse']*1000/100:.1f}ms "
                        f"salt={prof['salt']*1000/100:.1f}ms "
                        f"addH={prof['addh']*1000/100:.1f}ms "
                        f"embed={prof['embed']*1000/100:.1f}ms "
                        f"uff={prof['uff']*1000/100:.1f}ms "
                        f"meeko={prof['meeko']*1000/100:.1f}ms "
                        f"write={prof['write']*1000/100:.1f}ms"
                    )
                    # Profil als spezielles Result einschleusen
                    result_queue.put((-1, f"PROFILE_W{worker_id}", "INFO", msg))
                for k in prof: prof[k] = 0.0
                prof_count = 0

            result_queue.put((mol_index, mol_name, "OK", ""))

        except ConversionTimeout:
            signal.alarm(0)
            (log_dir / f"{mol_name}_convert_error.log").write_text(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} – TIMEOUT nach {timeout_s}s\n",
                encoding="utf-8",
            )
            result_queue.put((mol_index, mol_name, "TIMEOUT", f"nach {timeout_s}s"))

        except Exception as exc:
            signal.alarm(0)
            msg = f"{type(exc).__name__}: {exc}"
            (log_dir / f"{mol_name}_convert_error.log").write_text(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} – {msg}\n",
                encoding="utf-8",
            )
            result_queue.put((mol_index, mol_name, "ERROR", msg))


# ======================================================================
# EINGABEFORMATE
# ======================================================================
# Standard bleibt SDF. SMILES-Listen sind eine Alternative, kein Ersatz.
#
# Warum SMILES unproblematisch ist: die Bindungsordnung IST die Notation.
# Es gibt nichts zu rekonstruieren. Da ChEMBL-SDFs ohnehin 2D sind und die
# 3D-Struktur hier per ETKDG erzeugt wird, ist der Weg identisch – nur ohne
# den MolBlock-Parse davor.
#
# PDB wird bewusst NICHT unterstuetzt: das Format kennt keine
# Bindungsordnungen, RDKit muesste sie aus Atomabstaenden raten. Genau der
# Informationsverlust, wegen dem diese Pipeline den Umweg ueber PDB meidet.

SMILES_HEADER_ALIASES = ("smiles", "canonical_smiles", "smi", "structure")
NAME_HEADER_ALIASES   = ("name", "id", "chembl_id", "molecule_chembl_id",
                         "title", "compound", "compound_id", "identifier")


def detect_input_format(path: Path) -> str:
    """Erkennt das Eingabeformat an der Dateiendung."""
    suffixes = [s.lower() for s in path.suffixes]
    if ".gz" in suffixes:
        suffixes = [s for s in suffixes if s != ".gz"]
    ext = suffixes[-1] if suffixes else ""
    if ext in (".sdf", ".sd", ".mol"):
        return "sdf"
    if ext in (".smi", ".smiles", ".csv", ".tsv", ".txt"):
        return "smiles"
    if ext == ".pdb":
        raise ValueError(
            "PDB wird nicht unterstuetzt: das Format enthaelt keine "
            "Bindungsordnungen. Konvertiere zuerst nach SDF oder SMILES "
            "mit einem Werkzeug, das die Konnektivitaet kennt."
        )
    raise ValueError(
        f"Unbekannte Dateiendung '{ext}'. Erlaubt: .sdf, .smi, .csv, .tsv, "
        f"jeweils optional .gz. Mit --input-format laesst es sich erzwingen."
    )


def _sniff_delimiter(line: str, path: Path) -> str | None:
    """
    Trennzeichen bestimmen. None bedeutet: an beliebigem Whitespace teilen.

    Nur bei .csv/.tsv wird ein festes Trennzeichen verwendet, weil dort leere
    Felder zwischen zwei Trennern bedeutungstragend sind. Bei .smi/.txt wird
    immer an Whitespace geteilt: solche Dateien mischen in der Praxis Tabs und
    Leerzeichen zeilenweise, ein einmal erkannter Tab wuerde die naechste
    Zeile mit Leerzeichen dann falsch aufteilen.
    """
    ext = [x.lower() for x in path.suffixes if x.lower() != ".gz"]
    ext = ext[-1] if ext else ""
    if ext not in (".csv", ".tsv"):
        return None
    for delim in ("\t", ";", ","):
        if delim in line:
            return delim
    return None


def _resolve_columns(header_fields: list[str], path: Path,
                     smiles_col: str, name_col: str) -> tuple[int, int, bool]:
    """
    Bestimmt (smiles_index, name_index, hat_kopfzeile).

    Reihenfolge der Entscheidung:
      1. Explizite --smiles-col / --name-col (Zahl oder Spaltenname)
      2. Kopfzeile mit bekanntem Spaltennamen
      3. Endung: .csv/.tsv -> name,smiles   |   .smi -> smiles name

    Die dritte Regel folgt den jeweiligen Konventionen: das .smi-Format hat
    SMILES traditionell zuerst, eine selbstgeschriebene CSV nennt ueblicherweise
    erst den Namen.
    """
    lowered = [f.strip().lower() for f in header_fields]

    def find(explicit: str, aliases: tuple) -> int | None:
        if explicit:
            if explicit.isdigit():
                return int(explicit)
            if explicit.strip().lower() in lowered:
                return lowered.index(explicit.strip().lower())
            raise ValueError(f"Spalte '{explicit}' nicht in der Kopfzeile: "
                             f"{header_fields}")
        for alias in aliases:
            if alias in lowered:
                return lowered.index(alias)
        return None

    s_idx = find(smiles_col, SMILES_HEADER_ALIASES)
    n_idx = find(name_col,   NAME_HEADER_ALIASES)

    has_header = (s_idx is not None or n_idx is not None) and not (
        smiles_col.isdigit() if smiles_col else False
    )

    if s_idx is None or n_idx is None:
        ext = [x.lower() for x in path.suffixes if x.lower() != ".gz"]
        ext = ext[-1] if ext else ""
        if ext in (".csv", ".tsv"):
            s_idx = 1 if s_idx is None else s_idx
            n_idx = 0 if n_idx is None else n_idx
        else:                       # .smi-Konvention
            s_idx = 0 if s_idx is None else s_idx
            n_idx = 1 if n_idx is None else n_idx
        has_header = False

    return s_idx, n_idx, has_header


def iter_smiles(path: Path, smiles_col: str = "", name_col: str = ""):
    """
    Liest eine SMILES-Liste und liefert (smiles, name) je Zeile.

    Unterstuetzt CSV/TSV mit oder ohne Kopfzeile sowie das .smi-Format mit
    Whitespace-Trennung. Leere Zeilen und Kommentarzeilen (#) werden
    uebersprungen.
    """
    opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="replace")) \
        if str(path).endswith(".gz") else \
        (lambda p: open(p, "r", encoding="utf-8", errors="replace"))

    with opener(path) as fh:
        first = None
        for line in fh:
            if line.strip() and not line.lstrip().startswith("#"):
                first = line.rstrip("\n")
                break
        if first is None:
            return

        delim  = _sniff_delimiter(first, path)
        fields = first.split(delim) if delim else first.split()
        s_idx, n_idx, has_header = _resolve_columns(
            fields, path, smiles_col, name_col
        )

        if not has_header:
            row = [f.strip() for f in fields]
            if len(row) > s_idx and row[s_idx]:
                yield row[s_idx], (row[n_idx] if len(row) > n_idx else "")

        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = [f.strip() for f in (line.split(delim) if delim else line.split())]
            if len(row) <= s_idx or not row[s_idx]:
                continue
            yield row[s_idx], (row[n_idx] if len(row) > n_idx else "")


# ======================================================================
# PRODUCER – streamt Sammel-SDF zeilenweise
# ======================================================================

def _iter_sdf_blocks(sdf_path: Path):
    """Liefert (mol_block, roher_name) je Eintrag der Sammel-SDF."""
    opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="replace")) \
        if str(sdf_path).endswith(".gz") else \
        (lambda p: open(p, "r", encoding="utf-8", errors="replace"))
    with opener(sdf_path) as fh:
        block_lines: list[str] = []
        for raw_line in fh:
            block_lines.append(raw_line)
            if raw_line.startswith("$$$$"):
                # SDF-Standard: Name ist die 1. Zeile des Blocks
                first_line = block_lines[0].strip() if block_lines else ""
                yield "".join(block_lines), first_line
                block_lines = []


def producer_loop(sdf_file_str: str,
                  job_queue: mp.Queue,
                  num_workers: int,
                  control_queue: mp.Queue,
                  out_dir_str: str = "",
                  flat: bool = False,
                  skip_existing: bool = False,
                  input_format: str = "sdf",
                  smiles_col: str = "",
                  name_col: str = "",
                  log_dir_str: str = "") -> None:
    """
    Liest die Eingabe und sendet (index, payload, name, format) in die
    job_queue. Schliesst mit num_workers Sentinels.

    Zwei Eingabeformate:
      sdf     – Sammel-SDF, an '$$$$' gesplittet. Payload ist der ROHE
                MolBlock; geparst wird im Worker, damit ein kaputter Block
                dort sauber als ERROR landet und der Producer stabil bleibt.
      smiles  – CSV/TSV/.smi-Liste. Payload ist der SMILES-String.

    Duplikatpruefung: Der Dateiname entsteht aus dem Molekuelnamen, und die
    Docking-Ergebnisse landen spaeter flach in einem Verzeichnis pro Target.
    Zwei Eintraege mit gleichem Namen wuerden sich dort ueberschreiben, ohne
    Fehlermeldung. Deshalb wird jeder Name nur einmal durchgelassen und die
    Duplikate in duplicate_names.log geschrieben.
    """
    sdf_path = Path(sdf_file_str)
    out_dir  = Path(out_dir_str) if out_dir_str else None
    sent      = 0    # Index in der Eingabe – bestimmt den Unterordner
    queued    = 0    # tatsaechlich in die Queue gelegt
    skipped   = 0    # bereits konvertiert
    duplicate = 0    # Name schon vergeben
    unnamed   = 0    # kein Name in der Eingabe, Fallback mol_XXXXXXX

    seen_names: set[str] = set()
    dup_log = (Path(log_dir_str) / "duplicate_names.log") if log_dir_str else None
    dup_fh = None

    if input_format == "smiles":
        source = ((smi, raw_name)
                  for smi, raw_name in iter_smiles(sdf_path, smiles_col, name_col))
    else:
        source = _iter_sdf_blocks(sdf_path)

    try:
        for payload, raw_name in source:
            if not raw_name.strip():
                unnamed += 1
            mol_name = safe_mol_name(raw_name, sent)

            # Namenskollision: zweiter Eintrag wird verworfen, nicht
            # ueberschrieben. Der Index laeuft weiter (siehe unten).
            if mol_name in seen_names:
                duplicate += 1
                if dup_log is not None:
                    if dup_fh is None:
                        dup_fh = open(dup_log, "w", encoding="utf-8")
                    dup_fh.write(f"{sent}\t{mol_name}\n")
                sent += 1
                continue
            seen_names.add(mol_name)

            # Wiederaufnahme: liegt das PDBQT schon, ueberspringen.
            # Der Index sent laeuft trotzdem weiter, damit sich die
            # Unterordner-Aufteilung nicht verschiebt – sonst landete
            # dasselbe Molekuel beim naechsten Lauf woanders.
            if skip_existing and out_dir is not None:
                out_path = make_output_path(out_dir, mol_name, sent, flat)
                if out_path.exists() and out_path.stat().st_size > 0:
                    sent += 1
                    skipped += 1
                    continue

            job_queue.put((sent, payload, mol_name, input_format))
            sent += 1
            queued += 1
    finally:
        if dup_fh is not None:
            dup_fh.close()

    parts = [f"{sent:,} Eintraege gelesen", f"{queued:,} zu erledigen"]
    if skipped:
        parts.insert(1, f"{skipped:,} bereits konvertiert")
    if duplicate:
        parts.append(f"{duplicate:,} Namensduplikate verworfen"
                     + (f" -> {dup_log}" if dup_log else ""))
    if unnamed:
        parts.append(f"{unnamed:,} ohne Namen (Fallback mol_XXXXXXX)")
    print("[Producer] " + " | ".join(parts), flush=True)

    # Sentinel pro Worker
    for _ in range(num_workers):
        job_queue.put(SENTINEL)

    control_queue.put(("TOTAL", sent))


# ======================================================================
# LOGGING
# ======================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sdf_to_pdbqt")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "conversion.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ======================================================================
# MAIN
# ======================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Streaming SDF → PDBQT Konverter (RDKit + Meeko, kein PDB-Umweg)."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sdf-file", type=Path,
                     help="Pfad zur Sammel-SDF (kann viele Molekuele enthalten).")
    src.add_argument("--input", type=Path, dest="input_file",
                     help="Eingabedatei beliebigen unterstuetzten Formats: "
                          ".sdf, .smi, .csv, .tsv, jeweils optional .gz. "
                          "Gleichwertig zu --sdf-file, nur formatneutral benannt.")
    ap.add_argument("--input-format", choices=["auto", "sdf", "smiles"],
                    default="auto",
                    help="Eingabeformat. 'auto' erkennt es an der Endung "
                         "(default). PDB wird nicht unterstuetzt: das Format "
                         "kennt keine Bindungsordnungen.")
    ap.add_argument("--smiles-col", default="",
                    help="Spalte mit dem SMILES: Name aus der Kopfzeile oder "
                         "0-basierter Index. Ohne Angabe automatisch erkannt "
                         "(Kopfzeile, sonst .csv=Spalte 1 / .smi=Spalte 0). "
                         "Ein numerischer Index bedeutet: Datei ohne Kopfzeile.")
    ap.add_argument("--name-col", default="",
                    help="Spalte mit dem Molekuelnamen, analog zu --smiles-col "
                         "(.csv=Spalte 0 / .smi=Spalte 1).")
    ap.add_argument("--out-dir",  required=True, type=Path,
                    help="Ausgabe-Verzeichnis fuer PDBQTs.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Molekuele ueberspringen, deren PDBQT bereits "
                         "vorhanden und nicht leer ist. Fuer die Wiederaufnahme "
                         "eines abgebrochenen Laufs.")
    ap.add_argument("--log-dir",  required=True, type=Path,
                    help="Log-Verzeichnis (conversion.log + Per-Molekuel Errors).")
    ap.add_argument("--workers", type=int, default=15,
                    help="Anzahl paralleler Worker (default 15).")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Hartes Timeout pro Molekuel in Sekunden (default 120).")
    ap.add_argument("--uff-iters", type=int, default=800,
                    help="UFF-Iterationen (default 800).")
    ap.add_argument("--flat", action="store_true",
                    help="Alle PDBQTs in einen Ordner (sonst aufgeteilt in 0000/, 0001/, ...).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    in_path = args.input_file if args.input_file is not None else args.sdf_file
    if not in_path.is_file():
        print(f"FEHLER: Eingabedatei nicht gefunden: {in_path}", file=sys.stderr)
        return 1

    if args.input_format == "auto":
        try:
            in_format = detect_input_format(in_path)
        except ValueError as exc:
            print(f"FEHLER: {exc}", file=sys.stderr)
            return 1
    else:
        in_format = args.input_format

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_dir)

    logger.info("=== %s → PDBQT KONVERTIERUNG (Streaming, RDKit + Meeko) ===",
                "SMILES" if in_format == "smiles" else "SDF")
    logger.info("  Eingabe          : %s", in_path)
    logger.info("  Eingabeformat    : %s", in_format)
    logger.info("  PDBQT-Output     : %s", args.out_dir)
    logger.info("  Log-Verzeichnis  : %s", args.log_dir)
    logger.info("  Worker           : %d", args.workers)
    logger.info("  Timeout/Molekül  : %ds", args.timeout)
    logger.info("  Output-Layout    : %s",
                "flat" if args.flat else f"Unterordner à {FILES_PER_SUBDIR} Dateien")

    ctx = mp.get_context("spawn")  # spawn ist robuster als fork bei großen Libs
    job_queue:     mp.Queue = ctx.Queue(maxsize=QUEUE_MAXSIZE)
    result_queue:  mp.Queue = ctx.Queue()
    control_queue: mp.Queue = ctx.Queue()

    t_start = time.time()

    # Producer starten
    producer = ctx.Process(
        target=producer_loop,
        args=(str(in_path), job_queue, args.workers, control_queue,
              str(args.out_dir), args.flat, args.skip_existing,
              in_format, args.smiles_col, args.name_col, str(args.log_dir)),
        name="producer",
        daemon=True,
    )
    producer.start()

    # Worker starten
    workers = []
    for wid in range(args.workers):
        p = ctx.Process(
            target=worker_loop,
            args=(wid, job_queue, result_queue,
                  str(args.out_dir), str(args.log_dir),
                  args.uff_iters, args.timeout, args.flat),
            name=f"worker-{wid}",
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Result-Drain im Main-Prozess
    ok = errors = timeouts = 0
    total_known: int | None = None
    finished = 0

    while True:
        # Total-Count entgegennehmen, falls Producer fertig ist
        try:
            while True:
                tag, value = control_queue.get_nowait()
                if tag == "TOTAL":
                    total_known = value
                    logger.info(
                        "  Producer fertig: %d Molekuele eingespeist.", total_known
                    )
        except Empty:
            pass

        # Ergebnis abholen
        try:
            mol_index, mol_name, status, msg = result_queue.get(timeout=1.0)
        except Empty:
            # Wenn Producer fertig (Total bekannt) und alle drin sind: Schluss
            if total_known is not None and finished >= total_known:
                break
            # Sonst: leben noch Worker?
            if not any(w.is_alive() for w in workers):
                break
            continue

        finished += 1
        if status == "OK":
            ok += 1
        elif status == "INFO":
            # Profiling-Nachricht aus Worker — nicht als finished zählen
            finished -= 1
            logger.info("  %s", msg)
            continue
        elif status == "TIMEOUT":
            timeouts += 1
            logger.warning("  [%d] %-30s TIMEOUT (%s)", mol_index, mol_name, msg)
        else:
            errors += 1
            logger.warning("  [%d] %-30s FEHLER: %s", mol_index, mol_name, msg)

        if finished % LOG_PROGRESS_EVERY == 0:
            elapsed = time.time() - t_start
            rate = finished / max(elapsed, 1e-6)
            if total_known:
                remaining = max(0, total_known - finished)
                eta_s = remaining / max(rate, 1e-6)
                logger.info(
                    "  Fortschritt: %d / %d  |  %.1f mol/s  |  OK=%d ERR=%d TMO=%d  |  ETA %s",
                    finished, total_known, rate, ok, errors, timeouts,
                    _fmt_duration(eta_s),
                )
            else:
                logger.info(
                    "  Fortschritt: %d (Total unbekannt)  |  %.1f mol/s  |  OK=%d ERR=%d TMO=%d",
                    finished, rate, ok, errors, timeouts,
                )

    # Aufraeumen
    producer.join(timeout=5)
    for w in workers:
        w.join(timeout=5)

    duration = time.time() - t_start
    logger.info("=== ABGESCHLOSSEN | OK: %d | FEHLER: %d | TIMEOUTS: %d ===",
                ok, errors, timeouts)
    logger.info("  Gesamtlaufzeit: %s (%.1f mol/s)",
                _fmt_duration(duration), finished / max(duration, 1e-6))

    return 0 if (errors == 0 and timeouts == 0) else 2


if __name__ == "__main__":
    sys.exit(main())

