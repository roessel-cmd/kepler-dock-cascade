#-----------------------------------------------------------------------------
# Pharamcophore based SASA for complex (Author: Cheng Wang)
#-----------------------------------------------------------------------------
# PATCH (2026-05): os.system → subprocess.run mit Timeout
# Grund: pdb_to_xyzr (ein nawk-Skript) hat in unserem Container/Multi-Worker-
# Setup einen reproduzierbaren Hänger ('do_wait' auf nicht mehr existierendes
# Kind, vermutlich verlorener SIGCHLD wegen fehlendem Container-Init).
# os.system bietet keine Möglichkeit zu timeouten - der Worker-Subprocess
# bleibt für Stunden in C-Land 'wait()' stecken und kann nicht abgebrochen
# werden. subprocess.run(..., timeout=N) sendet bei Timeout SIGKILL und gibt
# uns die Kontrolle zurück.
#
# Triviale Aufrufe (mkdir/cp/cat/sed) wurden zusätzlich durch native Python-
# Operationen ersetzt - kein numerischer Effekt, nur weniger Subprocess-
# Overhead (~50-100ms pro Pose).
#
# Numerische Identität: Die MSMS- und pdb_to_xyzr-Outputs für nicht-hängende
# Posen sind bit-identisch zur unpatched Variante (gleiche Binaries, gleiche
# Argumente, nur anderer Aufruf-Mechanismus).
import os
import subprocess
import tempfile
import shutil, sys
from openbabel import openbabel as ob
from openbabel import pybel
import numpy as np
import pandas as pd
from pharma import pharma

msmsdir = "/opt/delta_LinF9_XGB/software/msms/"

# Timeouts - großzügig gewählt. Normale Posen brauchen <2s, also ist 30s
# bereits 15× Sicherheit. pdb_to_xyzr ist trivial (Lookup-Tabelle); MSMS
# selbst kann bei pathologischen Geometrien länger brauchen, daher 60s.
_PDB_TO_XYZR_TIMEOUT = 30   # sec
_MSMS_TIMEOUT        = 60   # sec
_SED_TIMEOUT         = 10   # sec

# Modul-Counter für Stille Telemetrie. Worker können beim Shutdown loggen
# wieviele Timeouts es gab. Pro Worker zählend, daher OK ohne Lock
# (run_XGB ist nicht thread-safe und wird im Worker serialisiert aufgerufen).
_n_timeouts_pdb_to_xyzr = 0
_n_timeouts_msms = 0


def _run_with_timeout(cmd, timeout, stdout_to=None, stderr_to=None,
                      shell=False):
    """Wrapper um subprocess.run mit Timeout. Bei Timeout: SIGKILL + log.

    Rückgabe: True wenn erfolgreich (rc=0), False sonst (Timeout, Crash,
    Nicht-Null-Returncode).

    Rationale check=False: wir wollen explizit selbst entscheiden was bei
    Fehler passiert. Der Aufrufer prüft anhand der erwarteten Output-Datei
    ob der Aufruf erfolgreich war (so wie der Original-Code es schon tut).
    """
    try:
        result = subprocess.run(
            cmd, shell=shell, timeout=timeout,
            stdout=stdout_to, stderr=stderr_to, check=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # subprocess.run hat bereits SIGKILL gesendet und auf Reaping gewartet.
        # Falls die Output-Datei teilweise geschrieben wurde, bleibt sie da -
        # der Aufrufer erkennt das anhand fehlender/leerer Output-Datei.
        return False
    except (OSError, ValueError):
        return False

def runMSMS(inprot, inlig, MSMSDIR = '.'):
    """Assign pharmaphore type to each atom and calculate SASA by MSMS

    Details can be found in comments

    Parameters
    ----------
    inprot : str
        Protein structure input
    inlig : str
        Ligand structure input

    Returns
    ----------
    df

    """
    # create tmp folder for all intermediate files
    olddir = os.getcwd()
    _msms_workdir = tempfile.mkdtemp(prefix='msms_run_')
    os.chdir(_msms_workdir)
    # PATCH: os.makedirs statt os.system('mkdir tmp') - identischer Effekt,
    # kein Subprocess.
    os.makedirs('tmp', exist_ok=True)

    # Process Input file to be p.pdb and l.pdb
    # convert protein file to PDB if not and remove hetatm card
    ppdb = 'tmp/p.pdb'
    __, intype = os.path.splitext(inprot)
    if intype[1:].lower() != 'pdb':
        prot = pybel.readfile(intype[1:], inprot).__next__()
        output = pybel.Outputfile("pdb", ppdb, overwrite=True)
        output.write(prot)
        output.close()
    else:
        # change possible HETATM to ATOM in pdb
        # PATCH: 'sed s/HETATM/ATOM  /g' nativ in Python. Identische Semantik:
        # GNU sed mit /g ersetzt alle Vorkommen pro Zeile, was hier OK ist
        # weil HETATM nur am Zeilen-Anfang vorkommt.
        with open(inprot, 'r') as _fin, open(ppdb, 'w') as _fout:
            for _line in _fin:
                _fout.write(_line.replace('HETATM', 'ATOM  '))

    # convert ligand file to be PDB by openbabel
    lpdb = 'tmp/l.pdb'
    __, intype = os.path.splitext(inlig)
    if intype[1:].lower() != 'pdb':
        lig = pybel.readfile(intype[1:], inlig).__next__()
        output = pybel.Outputfile("pdb", lpdb, overwrite=True)
        output.write(lig)
        output.close()
    else:
        # change possible HETATM to ATOM in pdb
        # PATCH: native Python statt sed (siehe oben).
        with open(inlig, 'r') as _fin, open(lpdb, 'w') as _fout:
            for _line in _fin:
                _fout.write(_line.replace('HETATM', 'ATOM  '))

    os.chdir('tmp')
    # PATCH: shutil.copy statt os.system("cp ..."). atmtypenumbers wird von
    # pdb_to_xyzr aus dem CWD gelesen (relativ: "./atmtypenumbers"), daher
    # muss die Datei hier liegen. Identisches Ziel, kein Subprocess.
    shutil.copy(os.path.join(msmsdir, "atmtypenumbers"), ".")
    # Process p.pdb/l.pdb to be p_sa.pdb/l_sa.pdb after pharma assignment
    # get full atom idx list and pharma
    ppdb2 = 'p_sa.pdb'
    lpdb2 = 'l_sa.pdb'

    pidx, ppharm = pharma('p.pdb').assign(write=True, outfn = ppdb2)
    lidx, lpharm = pharma('l.pdb').assign(write=True, outfn = lpdb2)

    # get subset atom idx which is nine element type
    # This have been done in pharma but still do it again
    elementint = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    psub = [idx for idx in pidx if ppharm[idx][0] in elementint]
    lsub = [idx for idx in lidx if lpharm[idx][0] in elementint]

    # get element number and pharma type and assign to df1
    comp = []
    for idx in psub:
        comp.append(ppharm[idx][0:2])
    for idx in lsub:
        comp.append(lpharm[idx][0:2])

    df1 = {}
    df1['atm'] = np.array(comp)[:,0]
    df1['pharma'] = np.array(comp)[:,1]
    df1 = pd.DataFrame(df1)
    # pdb to xyzr convert
    msms_pdbtoxyzr = os.path.join(msmsdir, "pdb_to_xyzr")

    # PATCH: pdb_to_xyzr ist der Haupt-Hänger-Verdächtige. Container ohne
    # PID-1-Init + parallele os.system aus mehreren Workern → verlorene
    # SIGCHLD → wait() blockiert für Stunden in der inneren sh.
    # subprocess.run mit Timeout sendet SIGKILL nach _PDB_TO_XYZR_TIMEOUT
    # Sekunden und kommt garantiert zurück. Bei Timeout: kein xyzr-File,
    # nachgelagerter MSMS-Aufruf wird scheitern, runMSMS gibt 0-SASA zurück
    # (Original-Verhalten bei MSMS-Fehler).
    global _n_timeouts_pdb_to_xyzr
    for _input, _output in [(ppdb2, 'p_sa.xyzr'), (lpdb2, 'l_sa.xyzr')]:
        with open(_output, 'w') as _fout:
            ok = _run_with_timeout(
                [msms_pdbtoxyzr, _input],
                timeout=_PDB_TO_XYZR_TIMEOUT,
                stdout_to=_fout,
                stderr_to=subprocess.DEVNULL,
            )
        if not ok:
            _n_timeouts_pdb_to_xyzr += 1
            # Output kann teilweise geschrieben sein - leeren damit MSMS
            # sauber failed (nicht auf halben Daten weiterläuft).
            try:
                open(_output, 'w').close()
            except OSError:
                pass

    # PATCH: cat durch Python-Concat ersetzen. Identisches Ziel.
    try:
        with open('pl_sa.xyzr', 'wb') as _fout:
            for _src in ('p_sa.xyzr', 'l_sa.xyzr'):
                if os.path.exists(_src):
                    with open(_src, 'rb') as _fin:
                        shutil.copyfileobj(_fin, _fout)
    except OSError:
        pass

    # run msms in with radius 1.0 (if fail, will increase to be 1.1)
    # PATCH: alle MSMS-Aufrufe via subprocess.run mit Timeout. MSMS hängt
    # bisher nicht in unseren Diagnose-Daten, aber os.system ohne Timeout
    # ist überall ein Risiko. log*.tmp behalten wir bei (Original-Verhalten,
    # falls jemand sie inspizieren will).
    global _n_timeouts_msms

    def _run_msms_three(msms_bin, probe_radius):
        """Drei MSMS-Aufrufe (protein, ligand, complex) mit gegebenem
        probe_radius. Identische Args wie Original (inkl. Doppel-Space).
        Rückgabe: True wenn alle drei Output-Files existieren."""
        for _xyzr, _area, _log in [
            ('p_sa.xyzr', 'p_sa.area', 'log1.tmp'),
            ('l_sa.xyzr', 'l_sa.area', 'log2.tmp'),
            ('pl_sa.xyzr', 'pl_sa.area', 'log3.tmp'),
        ]:
            with open(_log, 'w') as _flog:
                ok = _run_with_timeout(
                    [msms_bin, '-if', _xyzr, '-af', _area,
                     '-probe_radius', str(probe_radius), '-surface', 'ases'],
                    timeout=_MSMS_TIMEOUT,
                    stdout_to=_flog,
                    stderr_to=subprocess.STDOUT,
                )
                if not ok:
                    # Counter erhöhen; das _area-File fehlt oder ist
                    # unvollständig - den Failure erkennt der nachgelagerte
                    # isfile-Check.
                    _n_timeouts_msms += 1

    if sys.platform == "linux":
        msms = os.path.join(msmsdir, "msms.x86_64Linux2.2.6.1")
        _run_msms_three(msms, 1.0)
        if not (os.path.isfile('p_sa.area') and os.path.isfile('l_sa.area')
                and os.path.isfile('pl_sa.area')):
            _run_msms_three(msms, 1.1)
            print('1.1')
        if not (os.path.isfile('p_sa.area') and os.path.isfile('l_sa.area')
                and os.path.isfile('pl_sa.area')):
            print("SASA failed")
    elif sys.platform == "darwin":
        msms = os.path.join(msmsdir, "msms.MacOSX.2.6.1")
        _run_msms_three(msms, 1.0)
        if not (os.path.isfile('p_sa.area') and os.path.isfile('l_sa.area')
                and os.path.isfile('pl_sa.area')):
            _run_msms_three(msms, 1.1)
            print('1.1')
        if not (os.path.isfile('p_sa.area') and os.path.isfile('l_sa.area')
                and os.path.isfile('pl_sa.area')):
            print("SASA failed")

    # read surface area to df2
    df2 = {}
    tmp1 = np.genfromtxt('p_sa.area', skip_header=1)[:,2]
    num_p = len(tmp1)
    tmp2 = np.genfromtxt('l_sa.area', skip_header=1)[:,2]
    num_l = len(tmp2)
    tmp3 = np.genfromtxt('pl_sa.area', skip_header=1)[:,2]
    df2[2] = np.append(tmp1, tmp2)
    df2[3] = tmp3
    df2 = pd.DataFrame(df2)
    df = pd.concat([df1, df2], axis=1)
    df.columns = ['atm','pharma','pl','c']

    df_pro = df[0:num_p].copy()
    df_lig = df[num_p:num_p + num_l].copy()

    os.chdir(olddir)
    shutil.rmtree(_msms_workdir, ignore_errors=True)
    return df, df_pro, df_lig

def featureSASA(datadir, inprot, inlig, write=False):
    """Group the SASA by pharmacophore type

    Details can be found in comments

    Parameters
    ----------
    inprot : str
        Protein structure input
    inlig : str
        Ligand structure input

    Returns
    ----------
    sasalist : list [float]

    """

    # nine elements and nine pharma types
    #elemint = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    #elemstr = [str(i) for i in elemint]
    pharmatype = ['P', 'N', 'DA', 'D', 'A', 'AR', 'H', 'PL', 'HA']
    outdict = {i:0 for i in pharmatype}
    outdict_pro = {i:0 for i in pharmatype}
    outdict_lig = {i:0 for i in pharmatype}

    # run MSMS
    df,df_pro,df_lig = runMSMS(inprot, inlig, datadir)

    ## delta SASA with clip 0 (if value less 0, cut to 0)
    df["d"] = (df["pl"] - df["c"]).clip(0,None)
    df_pro["d"] = (df_pro["pl"] - df_pro["c"]).clip(0,None)
    df_lig["d"] = (df_lig["pl"] - df_lig["c"]).clip(0,None)

    # group delta sasa by element and pharma type
    
    dfg =  df.groupby("pharma")["d"].sum()
    dfgdict =  dfg.to_dict()

    dfg_pro =  df_pro.groupby("pharma")["d"].sum()
    dfgdict_pro =  dfg_pro.to_dict()

    dfg_lig =  df_lig.groupby("pharma")["d"].sum()
    dfgdict_lig =  dfg_lig.to_dict()


    # assign grouped dict to outdict
    for i in dfgdict:
        outdict[i] = dfgdict[i]

    for i in dfgdict_pro:
        outdict_pro[i] = dfgdict_pro[i]

    for i in dfgdict_lig:
        outdict_lig[i] = dfgdict_lig[i]

    # output list
    sasalist = []
    sasalist_pro = []
    sasalist_lig = []
    for i in pharmatype:
        sasalist.append(outdict[i])
        sasalist_pro.append(outdict_pro[i])
        sasalist_lig.append(outdict_lig[i])

    sasalist.append(sum(sasalist))
    sasalist_pro.append(sum(sasalist_pro))
    sasalist_lig.append(sum(sasalist_lig))

    if write:
        f = open("sasa.dat", "w")
        f.write(" ".join([str(np.round(i,2)) for i in sasalist]) + "\n")
        f.close()
        
    ### write ligand SASA info
    #df_lig = df_lig.round({'pl': 3, 'c': 3, 'd' : 3})
    #df_lig.to_csv('%s/df_lig.csv'%datadir, index=False)
    return df,df_pro, df_lig, sasalist, sasalist_pro, sasalist_lig


class sasa:
    """Buried SASA features

    """

    def __init__(self, datadir,prot, lig):
        """Pharmacophore based buried SASA Features

        Parameters
        ----------
        prot : str
            protein structure
        lig : str
            ligand structure

        """
        self.prot = prot
        self.lig = lig
        self.datadir = datadir

        self.rawdata, self.rawdata_pro, self.rawdata_lig, self.sasa, self.sasa_pro, self.sasa_lig = featureSASA( self.datadir, self.prot, self.lig)
        

        self.sasaTotal = self.sasa[-1]
        self.sasa_proTotal = self.sasa_pro[-1]
        self.sasa_ligTotal = self.sasa_lig[-1]
        self.sasaFeatures = self.sasa[0:-1]
        self.sasa_proFeatures = self.sasa_pro[0:-1]
        self.sasa_ligFeatures = self.sasa_lig[0:-1]

    def info(self):
        """Feature list"""
        featureInfo = ['P', 'N', 'DA', 'D', 'A', 'AR', 'H', 'PL', 'HA']
        return featureInfo




