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

    Job:    (mol_index, mol_block_str, mol_name)
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

        mol_index, mol_block, mol_name = job
        out_path = make_output_path(out_dir, mol_name, mol_index, flat)

        signal.alarm(timeout_s)  # Stopuhr starten
        try:
            t0 = time.perf_counter()
            mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=True)
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

            prep = MoleculePreparation(macrocycle_opening=False)
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
# PRODUCER – streamt Sammel-SDF zeilenweise
# ======================================================================

def producer_loop(sdf_file_str: str,
                  job_queue: mp.Queue,
                  num_workers: int,
                  control_queue: mp.Queue) -> None:
    """
    Liest die Sammel-SDF zeilenweise und splittet an '$$$$'-Trennern.
    Sendet (index, mol_block, mol_name) in die job_queue.
    Schliesst mit num_workers Sentinels und meldet Total-Count.

    Bewusst kein ForwardSDMolSupplier: wir wollen den ROHEN MolBlock an
    die Worker geben — RDKit parsen erfolgt dort. So kommt jeder kaputte
    Block im Worker an, wird sauber als ERROR gemeldet, und der Producer
    bleibt stabil.
    """
    sdf_path = Path(sdf_file_str)
    sent = 0
    with open(sdf_path, "r", encoding="utf-8", errors="replace") as fh:
        block_lines: list[str] = []
        for raw_line in fh:
            block_lines.append(raw_line)
            if raw_line.startswith("$$$$"):
                mol_block = "".join(block_lines)
                # SDF-Standard: Name ist die 1. Zeile des Blocks
                first_line = block_lines[0].strip() if block_lines else ""
                mol_name = safe_mol_name(first_line, sent)
                job_queue.put((sent, mol_block, mol_name))
                sent += 1
                block_lines = []

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
    ap.add_argument("--sdf-file", required=True, type=Path,
                    help="Pfad zur Sammel-SDF (kann viele Molekuele enthalten).")
    ap.add_argument("--out-dir",  required=True, type=Path,
                    help="Ausgabe-Verzeichnis fuer PDBQTs.")
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

    if not args.sdf_file.is_file():
        print(f"FEHLER: SDF-Datei nicht gefunden: {args.sdf_file}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_dir)

    logger.info("=== SDF → PDBQT KONVERTIERUNG (Streaming, RDKit + Meeko) ===")
    logger.info("  Sammel-SDF       : %s", args.sdf_file)
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
        args=(str(args.sdf_file), job_queue, args.workers, control_queue),
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

