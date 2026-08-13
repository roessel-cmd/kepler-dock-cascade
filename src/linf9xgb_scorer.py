"""
linf9xgb_scorer.py
==================
Wrapper fuer ΔLin_F9XGB (Yang & Zhang, J. Chem. Inf. Model. 2022, 62).

ΔLin_F9XGB ist ein Δ-Learning-Scoring: Lin_F9 (lineare empirische SF) plus
einer XGBoost-Korrektur, die auf 92 physikalisch motivierten Features
trainiert wurde. Output ist ein vorhergesagter pKd-Wert
(groesser = bessere Bindung).

Das Toolkit lebt in einem separaten Conda-Environment (linf9xgb_env)
weil es Python 3.7 + alte rdkit/scipy Versionen erwartet.

Architektur
-----------
Worker-Pool mit persistenten Subprocesses:

  - N Worker, jeder ein eigener Subprocess in linf9xgb_env
  - Jeder Worker laedt sein XGBoost-Modell einmal beim Start (~2-3s)
  - Caller verteilt Jobs ueber einen ThreadPoolExecutor an freie Worker
  - stdin/stdout-JSON-Protokoll pro Worker
  - Bei Worker-Tod: Job-Ergebnis ist None, naechster Job geht an
    anderen Worker

Public API
----------
  score_poses_batch(jobs, n_workers)   # bevorzugt: parallel
  score_pose(protein, ligand)          # fallback: sequenziell (ein Worker)
  is_available()
  shutdown()
  configure(n_workers)

CSV-Konvention
--------------
ΔLin_F9XGB liefert pKd (groesser = besser).
Die Invertierung (-pKd) erfolgt im Aufrufer (docking_rescore.py).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional


# ======================================================================
# KONFIGURATION
# ======================================================================

LINF9XGB_DIR  = Path(os.environ.get("LINF9XGB_DIR",  "/opt/delta_LinF9_XGB"))
LINF9XGB_ENV  = os.environ.get("LINF9XGB_ENV",       "linf9xgb_env")
CONDA_DIR     = Path(os.environ.get("CONDA_DIR",     "/opt/miniconda3"))
RUNXGB_SCRIPT = LINF9XGB_DIR / "script" / "runXGB.py"
WORKER_SCRIPT = Path("/tmp/linf9xgb_worker.py")

WORKER_STARTUP_TIMEOUT = 60.0
WORKER_REQUEST_TIMEOUT = 180.0

_logger = logging.getLogger("linf9xgb_scorer")

# Logger-Propagation explizit aktivieren damit Meldungen an den
# Pipeline-Root-Logger (mit FileHandler auf pipeline.log) durchkommen.
# Falls der Root-Logger noch keinen Handler hat (z.B. bei Standalone-Tests),
# verhindert NullHandler die "No handler"-Warnung.
_logger.propagate = True
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())
# Sicherstellen dass WARNING/INFO durchkommen (kein Filter auf hoeherem Level)
_logger.setLevel(logging.DEBUG)


def _attach_pipeline_handlers():
    """
    Sucht den Pipeline-Logger (in der Pipeline meist 'docking_pipeline')
    und kopiert dessen Handler an unseren Logger. Damit landen unsere
    Log-Meldungen sicher in pipeline.log, auch wenn die Logger-Hierarchie
    keine Propagation zum Root erlaubt.
    Idempotent: keine Doppel-Handler.
    """
    candidates = ["docking_pipeline", "pipeline", "rescore"]
    existing_targets = {id(h) for h in _logger.handlers}
    for name in candidates:
        parent = logging.getLogger(name)
        for h in parent.handlers:
            if id(h) not in existing_targets:
                _logger.addHandler(h)
                existing_targets.add(id(h))


# Beim ersten score_*-Aufruf einmalig die Pipeline-Handler ziehen.
# Das passiert spaet (zur Aufruf-Zeit), wenn die Pipeline ihren Logger
# bereits aufgesetzt hat.
_handlers_attached = False


# ======================================================================
# PERSISTENTER WORKER (laeuft im linf9xgb_env)
# ======================================================================
# Wird beim ersten Aufruf nach /tmp geschrieben. Protokoll:
#   IN  (eine Zeile JSON):  {"protein": "/abs/path.pdb", "ligand": "/abs/path.mol2"}
#   OUT (eine Zeile JSON):  {"ok": true,  "score": 7.42}
#                       /  {"ok": false, "error": "..."}
#   IN  "SHUTDOWN\n" → Worker beendet sich

_WORKER_SOURCE = r'''
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

LINF9XGB_DIR = Path(os.environ.get("LINF9XGB_DIR", "/opt/delta_LinF9_XGB"))
RUNXGB       = LINF9XGB_DIR / "script" / "runXGB.py"
sys.path.insert(0, str(LINF9XGB_DIR / "script"))

# Worker-spezifisches CWD: pro Subprocess ein eigenes Verzeichnis.
# Das ist KRITISCH weil prepare_betaAtoms.py Hilfsdateien
# (aa2ar.pdbqt, aa2ar_noh.pdb, aa2ar_noh.pdbqt, tmp/) im CWD anlegt
# und am Ende loescht. Bei gemeinsamem CWD ueberschreiben/loeschen
# parallele Worker gegenseitig ihre Dateien → Race Conditions.
WORKER_CWD = Path(tempfile.mkdtemp(prefix="linf9xgb_worker_"))


def _cleanup_cwd_residue():
    """Loescht residual files im Worker-CWD zwischen Aufrufen.
    Ein gescheiterter runXGB-Aufruf laesst sonst Files liegen die
    den naechsten Lauf irritieren (z.B. tmp/-Ordner)."""
    for entry in WORKER_CWD.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        except Exception:
            pass


def _score_via_cli(protein_pdb: str, ligand_mol2: str) -> float:
    # Vor jedem Aufruf alle Reste vom vorherigen Lauf wegraeumen
    _cleanup_cwd_residue()
    proc = subprocess.run(
        ["python", str(RUNXGB), protein_pdb, ligand_mol2],
        capture_output=True, text=True, timeout=180,
        cwd=str(WORKER_CWD),
    )
    if proc.returncode != 0:
        # Vollen stderr durchreichen damit echte Fehler sichtbar werden
        # (Python-Tracebacks brauchen mehr als 200 Zeichen)
        err = proc.stderr.strip() or proc.stdout.strip() or "<kein output>"
        raise RuntimeError(
            f"runXGB.py rc={proc.returncode}: {err[:1500]}"
        )
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("XGB"):
            parts = s.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    continue
    raise RuntimeError(f"Konnte XGB-Score nicht parsen: {proc.stdout[:200]!r}")


def main():
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            if raw == "SHUTDOWN":
                sys.stdout.write(json.dumps({"shutdown": True}) + "\n")
                sys.stdout.flush()
                return 0
            try:
                req = json.loads(raw)
                score = _score_via_cli(req["protein"], req["ligand"])
                sys.stdout.write(json.dumps({"ok": True, "score": score}) + "\n")
            except Exception as exc:
                sys.stdout.write(json.dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(limit=3),
                }) + "\n")
            sys.stdout.flush()
    finally:
        # Worker-CWD wegraeumen wenn Worker stirbt (egal warum)
        try:
            shutil.rmtree(WORKER_CWD, ignore_errors=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# ======================================================================
# DATENKLASSEN
# ======================================================================

@dataclass
class ScoringJob:
    """Ein Auftrag fuer den Pool: Protein + Ligand → Score."""
    key:          tuple    # beliebiger Identifier (zB (ligand, pose_idx))
    protein_pdb:  Path
    ligand_mol2:  Path


# ======================================================================
# EINZELNER WORKER
# ======================================================================

class _Worker:
    """Ein persistenter Subprocess im linf9xgb_env. Thread-safe via Lock."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._ready = False
        self._dead = False
        self._n_errors = 0   # Score-Fehler-Counter

    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Subprocess starten + auf Ready-Signal warten. True bei Erfolg."""
        if not WORKER_SCRIPT.exists():
            try:
                WORKER_SCRIPT.write_text(_WORKER_SOURCE, encoding="utf-8")
            except OSError as exc:
                _logger.error("[w%d] Worker-Script-Fehler: %s",
                              self.worker_id, exc)
                self._dead = True
                return False

        if not RUNXGB_SCRIPT.exists():
            _logger.error("[w%d] runXGB.py nicht gefunden: %s",
                          self.worker_id, RUNXGB_SCRIPT)
            self._dead = True
            return False

        cmd = [
            "bash", "-c",
            f"source {CONDA_DIR}/etc/profile.d/conda.sh && "
            f"conda activate {LINF9XGB_ENV} && "
            f"exec python {WORKER_SCRIPT}",
        ]
        env = os.environ.copy()
        env["LINF9XGB_DIR"] = str(LINF9XGB_DIR)

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            _logger.error("[w%d] Subprocess-Start: %s", self.worker_id, exc)
            self._dead = True
            return False

        # Ready-Signal abwarten
        t0 = time.monotonic()
        while time.monotonic() - t0 < WORKER_STARTUP_TIMEOUT:
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() if self._proc.stderr else "")[:300]
                _logger.error(
                    "[w%d] beendet sich vor Ready (rc=%s): %s",
                    self.worker_id, self._proc.returncode, err,
                )
                self._dead = True
                return False
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("ready"):
                self._ready = True
                _logger.debug("[w%d] bereit (%.1fs)",
                              self.worker_id, time.monotonic() - t0)
                return True
        _logger.error("[w%d] Startup-Timeout (%.0fs)",
                      self.worker_id, WORKER_STARTUP_TIMEOUT)
        self._dead = True
        return False

    # ------------------------------------------------------------------
    @property
    def is_alive(self) -> bool:
        if self._dead:
            return False
        if self._proc is None or self._proc.poll() is not None:
            self._dead = True
            return False
        return self._ready

    # ------------------------------------------------------------------
    def score(self, protein_pdb: Path, ligand_mol2: Path) -> Optional[float]:
        """Worker exklusiv fuer eine Pose. Rueckgabe: pKd oder None."""
        with self._lock:
            if not self.is_alive:
                return None

            assert self._proc is not None and self._proc.stdin is not None
            req = json.dumps({
                "protein": str(protein_pdb),
                "ligand":  str(ligand_mol2),
            })
            try:
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._dead = True
                return None

            line = self._proc.stdout.readline()
            if not line:
                self._dead = True
                return None
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                return None
            if not resp.get("ok"):
                self._n_errors += 1
                err_msg = resp.get("error", "?")
                # Erste 3 Fehler pro Worker auf WARNING-Level damit sie
                # im Standard-Log sichtbar sind. Danach DEBUG um Spam
                # zu vermeiden.
                if self._n_errors <= 3:
                    _logger.warning(
                        "[w%d] Score-Fehler #%d (Pose: %s): %s",
                        self.worker_id, self._n_errors,
                        Path(ligand_mol2).name, err_msg[:600],
                    )
                    if self._n_errors == 3:
                        _logger.warning(
                            "[w%d] Weitere Fehler werden nur im DEBUG-Level "
                            "geloggt. Setze Logger auf DEBUG fuer alle.",
                            self.worker_id,
                        )
                else:
                    _logger.debug("[w%d] Score-Fehler #%d: %s",
                                  self.worker_id, self._n_errors, err_msg)
                return None
            return float(resp["score"])

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                self._dead = True
                self._ready = False
                return
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.write("SHUTDOWN\n")
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
            self._dead = True
            self._ready = False


# ======================================================================
# WORKER-POOL
# ======================================================================

class _WorkerPool:
    """
    Pool von N Workern. Jobs werden ueber ThreadPoolExecutor verteilt.
    Worker werden lazy gestartet (parallel) sobald der erste Job kommt.
    """

    def __init__(self, n_workers: int):
        self.n_workers = max(1, n_workers)
        self._workers: list[_Worker] = []
        self._available: Queue = Queue()
        self._started = False
        self._start_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _ensure_started(self) -> int:
        """Startet alle Worker parallel. Rueckgabe: aktive Worker-Anzahl."""
        with self._start_lock:
            if self._started:
                return sum(1 for w in self._workers if w.is_alive)

            _logger.info("Starte ΔLin_F9XGB-Worker-Pool (n=%d)…",
                         self.n_workers)
            t0 = time.monotonic()
            self._workers = [_Worker(i) for i in range(self.n_workers)]

            # Parallel starten - sonst dauert n=8 ungefaehr 8×3s sequenziell
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {ex.submit(w.start): w for w in self._workers}
                for fut in futures:
                    try:
                        ok = fut.result(timeout=WORKER_STARTUP_TIMEOUT + 5)
                    except Exception as exc:
                        _logger.error("Worker-Start Exception: %s", exc)
                        ok = False
                    if ok:
                        self._available.put(futures[fut])

            n_alive = self._available.qsize()
            elapsed = time.monotonic() - t0
            if n_alive == self.n_workers:
                _logger.info("  Pool bereit: %d/%d Worker (%.1fs)",
                             n_alive, self.n_workers, elapsed)
            else:
                _logger.warning("  Pool teilweise bereit: %d/%d Worker (%.1fs)",
                                n_alive, self.n_workers, elapsed)
            self._started = True
            return n_alive

    # ------------------------------------------------------------------
    def _checkout(self, timeout: Optional[float] = None) -> Optional[_Worker]:
        """
        Holt einen freien Worker aus der Queue.

        Da der ThreadPoolExecutor max_workers=n_alive verwendet, fragen nie
        mehr Threads gleichzeitig _checkout an als Worker existieren - es
        kann also nie zu echtem Hunger kommen, solange mindestens ein
        Worker lebt.

        Schutz vor Deadlock: wenn alle Worker tot sind und keiner mehr
        in die Queue zurueckkommen kann, geben wir None zurueck statt
        unendlich zu warten. Dazu pollen wir mit kurzen Timeouts.
        """
        if timeout is not None:
            try:
                return self._available.get(timeout=timeout)
            except Empty:
                return None

        # Default-Pfad: pollen alle 2s, abbrechen wenn alle Worker tot
        while True:
            try:
                return self._available.get(timeout=2.0)
            except Empty:
                if not any(w.is_alive for w in self._workers):
                    _logger.error("_checkout: alle Worker tot, breche ab")
                    return None
                # sonst weiter warten

    def _checkin(self, worker: _Worker) -> None:
        if worker.is_alive:
            self._available.put(worker)
        # tote Worker kommen nicht zurueck

    # ------------------------------------------------------------------
    def _score_one(self, job: ScoringJob) -> tuple:
        worker = self._checkout()
        if worker is None:
            return (job.key, None)
        try:
            score = worker.score(job.protein_pdb, job.ligand_mol2)
            return (job.key, score)
        finally:
            self._checkin(worker)

    # ------------------------------------------------------------------
    def score_batch(
        self,
        jobs: list[ScoringJob],
        progress_callback: Optional[Callable[[int, int, tuple, Optional[float]], None]] = None,
    ) -> dict:
        """
        Scoret eine Liste von Jobs parallel.

        Args:
            jobs: Liste von ScoringJob
            progress_callback: optional, (done, total, key, score)

        Returns: dict {job.key: pKd or None}
        """
        if not jobs:
            return {}

        n_alive = self._ensure_started()
        if n_alive == 0:
            _logger.error("Pool hat 0 aktive Worker - alle Scores None")
            return {job.key: None for job in jobs}

        results: dict = {}
        n_total = len(jobs)
        n_done = 0
        results_lock = threading.Lock()

        # max_workers = n_alive: mehr Threads als Worker waeren nutzlos
        # (alle wuerden auf _checkout warten). Wir nehmen as_completed
        # damit progress_callback in tatsaechlicher Completion-Reihenfolge
        # gefeuert wird (statt in Submit-Reihenfolge mit kuenstlichen Bursts).
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=n_alive) as ex:
            future_to_job = {
                ex.submit(self._score_one, job): job for job in jobs
            }
            for fut in as_completed(future_to_job):
                try:
                    key, score = fut.result()
                except Exception as exc:
                    job = future_to_job[fut]
                    _logger.debug("Job-Exception fuer %s: %s", job.key, exc)
                    key, score = job.key, None

                with results_lock:
                    results[key] = score
                    n_done += 1
                    if progress_callback:
                        try:
                            progress_callback(n_done, n_total, key, score)
                        except Exception:
                            pass

        return results

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if not self._started or not self._workers:
            return
        _logger.info("Beende ΔLin_F9XGB-Worker-Pool (n=%d)…",
                     len(self._workers))
        with ThreadPoolExecutor(max_workers=len(self._workers)) as ex:
            for w in self._workers:
                ex.submit(w.shutdown)
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except Empty:
                break
        self._workers = []
        self._started = False


# ======================================================================
# MODUL-LEVEL POOL-VERWALTUNG
# ======================================================================

_pool: Optional[_WorkerPool] = None
_pool_lock = threading.Lock()
_default_n_workers = 1


def configure(n_workers: int) -> None:
    """
    Setzt die Worker-Anzahl fuer den naechsten Pool. Wirkt erst beim
    naechsten _get_pool()-Aufruf, also nach shutdown() oder vor dem
    ersten score_*-Aufruf.
    """
    global _default_n_workers
    if n_workers < 1:
        n_workers = 1
    _default_n_workers = n_workers


def _get_pool() -> _WorkerPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _WorkerPool(_default_n_workers)
        return _pool


# ======================================================================
# OEFFENTLICHE API
# ======================================================================

def is_available() -> bool:
    """True wenn das ΔLin_F9XGB-Toolkit installiert ist."""
    return RUNXGB_SCRIPT.exists()


def score_pose(protein_pdb: Path, ligand_mol2: Path) -> Optional[float]:
    """
    Sequenzieller Einzel-Score (Pool wird mit n=1 verwendet).
    Bevorzugt fuer Batches: score_poses_batch().
    """
    global _handlers_attached
    if not _handlers_attached:
        _attach_pipeline_handlers()
        _handlers_attached = True
    pool = _get_pool()
    job = ScoringJob(key=("single",), protein_pdb=protein_pdb,
                    ligand_mol2=ligand_mol2)
    res = pool.score_batch([job])
    return res.get(("single",))


def score_poses_batch(
    jobs: list[ScoringJob],
    n_workers: int = 0,
    progress_callback: Optional[Callable[[int, int, tuple, Optional[float]], None]] = None,
) -> dict:
    """
    Scoret viele Posen parallel.

    Args:
        jobs: Liste von ScoringJob
        n_workers: Pool-Groesse.
                   > 0: explizite Groesse, Pool wird neu erstellt falls
                        bestehender Pool eine andere Groesse hat.
                   = 0: aktuellen configure()-Wert verwenden (Default 1).
                        Bestehender Pool wird wiederverwendet.
        progress_callback: (done, total, key, score), wird in
                           Completion-Reihenfolge gefeuert.

    Returns: dict {job.key: pKd or None}
    """
    global _pool, _handlers_attached
    # Beim ersten Aufruf Pipeline-Handler attachen damit Logs sichtbar werden
    if not _handlers_attached:
        _attach_pipeline_handlers()
        _handlers_attached = True

    if n_workers > 0:
        # Pool-Groesse pruefen und ggf. neu erstellen - alles im Lock
        with _pool_lock:
            configure(n_workers)
            if _pool is not None and _pool.n_workers != n_workers:
                _pool.shutdown()
                _pool = None
            if _pool is None:
                _pool = _WorkerPool(n_workers)
            pool = _pool
    else:
        pool = _get_pool()

    return pool.score_batch(jobs, progress_callback=progress_callback)


def shutdown() -> None:
    """Pool beenden (z.B. zwischen Targets fuer Speicher-Hygiene)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown()
            _pool = None
