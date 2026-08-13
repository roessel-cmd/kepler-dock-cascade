import sys, os
import subprocess
import numpy as np
import pandas as pd
import alphaspace2 as al
import mdtraj
 
ADT = '/opt/mgltools/bin/pythonsh /opt/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'

# PATCH (2026-05): os.system → subprocess.run mit Timeout
# Grund: prepare_receptor4.py via pythonsh kann im Apptainer-Container
# (kein PID-1-Init) durch verlorene SIGCHLD in /bin/sh dauerhaft im
# wait() haengen. Selber Mechanismus wie zuvor in featureSASA.py.
# Identische Tool-Aufrufe, identische Argumente - nur subprocess.run mit
# Timeout statt os.system. Numerisch 1:1 zur Original-Implementierung.
_PREPARE_RECEPTOR_TIMEOUT = 120   # sec (MGLTools ist langsam)
_n_timeouts_prepare_receptor = 0


def _run_with_timeout(cmd, timeout, shell=False):
    """Wrapper um Popen mit Timeout und Process-Group-Kill.

    WICHTIG: subprocess.run(timeout=...) sendet bei Timeout nur SIGKILL
    an den direkten Subprocess. Bei shell=True ist das nur /bin/sh, nicht
    die Enkel (pythonsh, MGLTools-Python). Die wuerden zu Waisen und
    weiterlaufen → derselbe Haenger wie vorher.

    Loesung: start_new_session=True macht den Subprocess zum Process Group
    Leader. Bei Timeout senden wir os.killpg(SIGKILL) an die ganze Gruppe -
    das erwischt /bin/sh, pythonsh und alle MGLTools-Python-Prozesse
    auf einmal.

    Rueckgabe: True wenn rc=0, False sonst (Timeout, Crash, rc!=0).
    """
    import signal
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, shell=shell,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
            return rc == 0
        except subprocess.TimeoutExpired:
            # Process Group sterben lassen
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            # Reaping: wait() bis Kernel den Subprocess aufgeraeumt hat.
            # Ohne das bleibt er als Zombie und wir leaken.
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return False
    except (OSError, ValueError):
        return False

def Protein_pdbqt(PDB, PDBQT, ADT):
    '''
    generate the protein pdbqt file.
    '''
    cmd = ADT + " -r " + PDB + " -o " + PDBQT + " -U nphs"
    # PATCH: subprocess.run mit Timeout statt os.system. ADT enthaelt
    # bereits einen Pfad mit Leerzeichen (pythonsh + script.py), daher
    # shell=True - die Shell kuemmert sich um's Splitten. Bei Timeout
    # werden pythonsh + Kindprozesse via Process-Group-Kill beendet.
    global _n_timeouts_prepare_receptor
    ok = _run_with_timeout(cmd, timeout=_PREPARE_RECEPTOR_TIMEOUT, shell=True)
    if not ok:
        _n_timeouts_prepare_receptor += 1
        # Output kann teilweise geschrieben sein - leeren damit der Aufrufer
        # den Failure am fehlenden/leeren PDBQT-File erkennt.
        # (mdtraj.load wuerde sonst auf halben Daten weiterlaufen.)
        try:
            if os.path.exists(PDBQT):
                os.remove(PDBQT)
        except OSError:
            pass

def Strip_h(input_file,output_file):
    '''
    input_file and output_file need to be in pdb or pdbqt format 
    '''
    inputlines = open(input_file,'r').readlines()
    output = open(output_file,'w')
    for line in inputlines:
        if not 'H' in line[12:14]:
            output.write(line)
    output.close()

def Write_betaAtoms(ss, outfile):
    '''
    ss is the input AlphaSpace object, outfile is the output pdb file. 
    '''
    betaAtoms = open(outfile,'w')
    count = 1
    c = 1
    for p in ss.pockets:
        for betaAtom in p.betas:
            count = count+1
            coord = betaAtom.centroid
            ASpace = '%.1f'%betaAtom.space
            Score = '%.1f'%betaAtom.score
            atomtype = betaAtom.best_probe_type
            x, y, z  = '%.3f'%coord[-3], '%.3f'%coord[-2], '%.3f'%coord[-1]
            line = 'ATOM  ' + str(count).rjust(5) + str(atomtype).upper().rjust(5) + ' BAC' + str(c).rjust(5) + '     ' + str(x).rjust(8) + str(y).rjust(8) + str(z).rjust(8) + ' ' + str(ASpace).rjust(5) + ' ' + str(Score).rjust(5) + '           %s\n'%atomtype
            betaAtoms.write(line)
    betaAtoms.close()
    
def Prepare_beta(pdb, outfile, ADT=ADT):
    pdbqt = pdb[:-4]+'.pdbqt'
    Protein_pdbqt(pdb, pdbqt, ADT)

    pdb_noh = pdb[:-4]+'_noh.pdb'
    pdbqt_noh = pdb[:-4]+'_noh.pdbqt'

    Strip_h(pdb, pdb_noh)
    Strip_h(pdbqt, pdbqt_noh)

    prot = mdtraj.load(pdb_noh)
    al.annotateVinaAtomTypes(pdbqt=pdbqt_noh, receptor=prot)
    ss = al.Snapshot()
    ss.run(prot)
    Write_betaAtoms(ss, outfile)
    # PATCH: native Python statt os.system('rm ...'). Schneller, kein
    # Subprocess, identischer Effekt.
    for _f in (pdb_noh, pdbqt_noh):
        try:
            os.remove(_f)
        except OSError:
            pass
    return pdbqt

def main():
    args = sys.argv[1:]
    if not args:
        print ('usage: python prepare_betaAtoms.py pro.pdb outfile')

        sys.exit(1)

    elif sys.argv[1] == '--help':
        print ('usage: python prepare_betaAtoms.py pro.pdb outfile')

        sys.exit(1)

    elif len(args) == 2 and sys.argv[1].endswith('.pdb'):
        pdb = sys.argv[1]
        outfile = sys.argv[2]
        pdbqt = Prepare_beta(pdb, outfile)
        
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
