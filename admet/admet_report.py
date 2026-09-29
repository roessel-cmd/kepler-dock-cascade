#!/usr/bin/env python3
"""
ADMET-AI Pipeline mit PDF-Report fuer pocket-basierte SMILES-Listen.

INPUT-FORMAT
------------
Eine einfache Textdatei (UTF-8):

    #POCKET_1
    #rank, Smiles

    1, C=C(NC(=O)...
    2, Cc1ccc(...

    #POCKET_2
    #rank, Smiles

    1, CC(=O)Oc1ccccc1C(=O)O

Regeln:
  * Eine Zeile, die mit '#' beginnt und das Wort POCKET enthaelt, eroeffnet eine
    neue Pocket. Alles dahinter ist der Bezeichner; Trennung durch Unterstrich,
    Leerzeichen, Doppelpunkt oder Bindestrich ist gleichwertig. '#POCKET_1',
    '#POCKET 1' und '#pocket-1' ergeben alle 'POCKET_1'.
  * Andere '#'-Zeilen sind Kommentare. Enthalten sie 'rank' oder 'smiles', gelten
    sie als Spaltenkopf und werden stillschweigend uebergangen; jede sonstige
    '#'-Zeile wird gemeldet, damit ein verschriebener Pocket-Kopf nicht
    unbemerkt bleibt.
  * Datenzeilen lauten '<rang>, <smiles>'. Weitere Spalten (etwa ein
    Docking-Score) werden ignoriert -- SMILES enthalten nie ein Komma.
  * Fehlt das Komma, werden ersatzweise Tabulator oder Semikolon als
    Trennzeichen akzeptiert.
  * Leerzeilen und umgebende Leerzeichen sind unerheblich. Ist der Rang keine
    Zahl, wird fortlaufend nummeriert und gewarnt.
  * Identische SMILES in mehreren Pockets werden nur einmal gerechnet.

berechnet mit ADMET-AI die ADMET-Properties des Kernsets und erzeugt:

    <output>/admet_report.pdf      Hauptausgabe: Deckblatt, je Pocket eine Uebersicht
                                   mit DrugBank-Scatterplot, je Molekuel eine Detailseite
                                   mit Struktur, Spider-Plot und Parametertabelle
    <output>/admet_all.csv         vollstaendige Rohdaten (Absicherung, da eine
                                   Neuberechnung teuer ist) -- abschaltbar mit --no-csv
    <output>/failed_smiles.txt     SMILES, die RDKit nicht parsen konnte

Das Modell wird genau einmal geladen; alle eindeutigen SMILES werden in einem Batch
praediziert und anschliessend per Merge auf die Pocket/Rank-Tabelle verteilt.

Installation:
    pip install admet-ai reportlab

Beispiel:
    python admet_report.py --input pockets.txt --output-dir results
    python admet_report.py --input pockets.txt --detail-top 10
    python admet_report.py --input pockets.txt --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, bevor pyplot irgendwo importiert wird

import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# Start aus der IDE
# Beim Start ueber die Kommandozeile wird dieser Block vollstaendig ignoriert.
# --------------------------------------------------------------------------- #
IDE_CONFIG = {
    "input": r"C:\Users\thoma\OneDrive\Desktop\ADMET\admet_smiles.txt",          # Pfad zum Input-Textfile
    "output_dir": r"C:\Users\thoma\OneDrive\Desktop\ADMET",         # Ausgabeverzeichnis
    "detail_top": None,              # None = Detailseiten fuer alle Molekuele, sonst z.B. 10
    "x_property": "Human Intestinal Absorption",
    "y_property": "Clinical Toxicity",
    "max_molecule_num": None,        # z.B. 10, um Molekuele im Scatterplot zu nummerieren
    "no_scatter": False,             # True = keine DrugBank-Scatterplots im Report
    "atc_code": "",                # z.B. "N" fuer ZNS-Wirkstoffe als Referenz
    "no_physchem": False,            # True = physikochemische Properties weglassen
    "num_workers": None,             # None = 0 ohne GPU, 8 mit GPU
    "no_csv": True,                 # True = keine Rohdaten-CSV schreiben
    "dry_run": False,                # True = nur Parsing pruefen, ohne ADMET-AI
    "list_atc": False,               # True = gueltige ATC-Bezeichner ausgeben und beenden
}

# Hinweis zu "atc_code": ADMET-AI erwartet einen Klartext-Namen in Kleinschreibung
# ("dermatologicals", "antineoplastic and immunomodulating agents", ...), NICHT den
# list_atc = True beziehungsweise --list-atc auf der Kommandozeile.

# --------------------------------------------------------------------------- #
# Kernset
#   key -> (Anzeigename, Einheit, Richtung, n_TDC, Metrik, Bestwert)
#   Richtung: "up" hoeher guenstig | "down" hoeher riskant | "ctx" kontextabhaengig
#   Bestwert = bester veroeffentlichter TDC-Leaderboard-Wert (obere Schranke,
#   nicht die Leistung von ADMET-AI selbst). Belege im Quellendokument.
# --------------------------------------------------------------------------- #
CORE_GROUPS: dict[str, list[tuple]] = {
    # Rohe Deskriptoren bleiben bewusst ungefaerbt: ein hohes Molekulargewicht ist
    # als solches kein Mangel, sondern erst im Licht der Lipinski-/Veber-Regeln --
    # und die stehen bereits in der Flags-Spalte. Gefaerbt werden nur QED und
    # Lipinski, weil beide von sich aus Bewertungsmasse sind.
    "Physicochemical — computed (RDKit); rule violations listed above": [
        ("molecular_weight", "Molecular weight", "Da", "ctx", None, None, None),
        ("logP", "logP (Crippen)", "–", "ctx", None, None, None),
        ("tpsa", "TPSA", "A<super>2</super>", "ctx", None, None, None),
        ("hydrogen_bond_donors", "H-bond donors", "count", "ctx", None, None, None),
        ("QED", "QED (drug-likeness)", "0–1", "up", None, None, None),
        ("Lipinski", "Lipinski rules passed", "0–4", "up", None, None, None),
    ],
    "Absorption & Solubility": [
        ("HIA_Hou", "Intestinal absorption", "Score 0–1", "up", 578, "AUROC", "0.993"),
        ("Bioavailability_Ma", "Oral bioavailability", "Score 0–1", "up", 640, "AUROC", "0.942"),
        ("Caco2_Wang", "Caco-2 permeability", "log cm/s", "up", 906, "MAE", "0.256"),
        ("Solubility_AqSolDB", "Aqueous solubility", "log mol/L", "up", 9982, "MAE", "0.741"),
        ("Lipophilicity_AstraZeneca", "Lipophilicity (logD 7.4)", "logD", "ctx", 4200, "MAE", "0.456"),
    ],
    "Distribution": [
        ("BBB_Martins", "Blood-brain barrier", "Score 0–1", "ctx", 1975, "AUROC", "0.924"),
    ],
    "Metabolism — CYP inhibition": [
        ("CYP3A4_Veith", "CYP3A4 inhibition", "Score 0–1", "down", 12328, "AUPRC", "0.916"),
        ("CYP2D6_Veith", "CYP2D6 inhibition", "Score 0–1", "down", 13130, "AUPRC", "0.790"),
        ("CYP2C9_Veith", "CYP2C9 inhibition", "Score 0–1", "down", 12092, "AUPRC", "0.859"),
        ("cyp_inhibition_max", "Max of the three <i>(aggregated)</i>", "Score 0–1", "down",
         None, None, None),
    ],
    "Toxicity": [
        ("hERG", "hERG blockade", "Score 0–1", "down", 648, "AUROC", "0.880"),
        ("AMES", "AMES mutagenicity", "Score 0–1", "down", 7255, "AUROC", "0.871"),
        ("DILI", "Liver injury (DILI)", "Score 0–1", "down", 475, "AUROC", "0.956"),
        ("ClinTox", "Clinical toxicity", "Score 0–1", "down", 1484, "—", "no BM"),
        ("LD50_Zhu", "Acute toxicity LD50", "log(1/(mol/kg))", "down", 7385, "MAE", "0.552"),
        ("tox21_max", "Tox21 max <i>(12 assays, aggr.)</i>", "Score 0–1", "down",
         6000, "—", "no BM"),
        ("tox21_n_above_0.5", "Tox21 count &gt; 0.5 <i>(aggr.)</i>", "count", "down",
         6000, "—", "no BM"),
    ],
    "Outlier indicator — not quantitatively reliable": [
        ("Half_Life_Obach", "Half-life", "h", "ctx", 667, "Spearman", "0.576"),
    ],
}

TOX21_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
CYP_INHIBITION_COLUMNS = ["CYP3A4_Veith", "CYP2D6_Veith", "CYP2C9_Veith"]

# Spalten, die plot_radial_summary zwingend braucht (ohne Perzentil-Suffix)
RADIAL_PROPERTIES = ["BBB_Martins", "ClinTox", "Solubility_AqSolDB",
                     "Bioavailability_Ma", "hERG"]

# Richtungspfeil; in Uebersicht und Detailtabelle identisch verwendet
ARROW = {"up": "↑", "down": "↓", "ctx": "~"}

# Harte Physchem-Filter (Lipinski + Veber, orale Verfuegbarkeit)
PHYSCHEM_RULES = [
    ("molecular_weight", 500, "MW &gt; 500"),
    ("logP", 5, "logP &gt; 5"),
    ("hydrogen_bond_donors", 5, "HBD &gt; 5"),
    ("hydrogen_bond_acceptors", 10, "HBA &gt; 10"),
    ("tpsa", 140, "TPSA &gt; 140"),
]

# Uebersichtstabelle: (Spaltenkopf, Property-Key, Art)
OVERVIEW_COLUMNS = [
    ("MW", "molecular_weight", "value"),
    ("QED", "QED", "pct"),
    ("Lip.", "Lipinski", "pct"),
    ("HIA", "HIA_Hou", "pct"),
    ("Sol.", "Solubility_AqSolDB", "pct"),
    ("hERG", "hERG", "pct"),
    ("DILI", "DILI", "pct"),
    ("Tox21", "tox21_max", "pct"),
]

# Pocket-Header: alles nach dem Wort POCKET wird als Bezeichner uebernommen,
# egal ob mit Unterstrich, Leerzeichen, Doppelpunkt oder Bindestrich abgetrennt.
POCKET_RE = re.compile(r"^#\s*pocket[\s_:.-]*(.*)$", flags=re.IGNORECASE)
# Spaltenkopf wie "#rank, Smiles" -- wird stillschweigend uebergangen.
COLUMN_HEADER_RE = re.compile(r"^#.*\b(rank|smiles)\b", flags=re.IGNORECASE)
# Ersatztrennzeichen, falls eine Datenzeile kein Komma enthaelt.
FALLBACK_SEPARATORS = ["\t", ";"]

# --------------------------------------------------------------------------- #
# Stil
# --------------------------------------------------------------------------- #
INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#d4d4d4")
BAND = colors.HexColor("#f2f2f0")
HEADBG = colors.HexColor("#e4e4e1")
GROUPBG = colors.HexColor("#ecebe8")
GOOD, WARN, RISK = (colors.HexColor("#d9ead3"), colors.HexColor("#fff2cc"),
                    colors.HexColor("#f8d7da"))
GOOD_T, WARN_T, RISK_T = (colors.HexColor("#2e5c1f"), colors.HexColor("#7a5c00"),
                          colors.HexColor("#8b1d28"))
GOOD_B, WARN_B, RISK_B = (colors.HexColor("#6aa84f"), colors.HexColor("#e0b000"),
                          colors.HexColor("#cc4b56"))
BAR_BG = colors.HexColor("#dedede")

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=19,
                    leading=22, textColor=INK, alignment=TA_LEFT, spaceAfter=1)
SUBT = ParagraphStyle("SUBT", parent=_ss["Normal"], fontName="Helvetica", fontSize=9.5,
                      leading=13, textColor=MUTED, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=12.5,
                    leading=15, textColor=INK, spaceBefore=8, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=8.5,
                    leading=11, textColor=MUTED, spaceAfter=2)
BODY = ParagraphStyle("BODY", parent=_ss["Normal"], fontName="Helvetica", fontSize=8.6,
                      leading=11.6, textColor=INK)
SMALL = ParagraphStyle("SMALL", parent=_ss["Normal"], fontName="Helvetica", fontSize=7.3,
                       leading=9.4, textColor=MUTED)
MONO = ParagraphStyle("MONO", parent=_ss["Normal"], fontName="Courier", fontSize=6.3,
                      leading=8.0, textColor=INK)
CELL = ParagraphStyle("CELL", parent=_ss["Normal"], fontName="Helvetica", fontSize=7.2,
                      leading=8.8, textColor=INK)
RCELL = ParagraphStyle("RCELL", parent=CELL, alignment=2)
HCELL = ParagraphStyle("HCELL", parent=_ss["Normal"], fontName="Helvetica-Bold",
                       fontSize=7.0, leading=8.0, textColor=INK)
HCELLC = ParagraphStyle("HCELLC", parent=HCELL, alignment=1)


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def parse_pocket_file(path: Path) -> list[dict]:
    """Parst das Pocket-Textfile in eine Liste von Eintraegen."""
    entries: list[dict] = []
    current_pocket: str | None = None

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                match = POCKET_RE.match(line)
                if match:
                    # sanitize() faellt bei leerer Eingabe auf "unnamed" zurueck -- was
                    # fuer Dateinamen richtig ist, hier aber aus '#POCKET' ein
                    # 'POCKET_unnamed' gemacht und den Zweig unten unerreichbar gelassen hat.
                    raw_label = match.group(1).strip()
                    label = sanitize(raw_label) if raw_label else ""
                    current_pocket = f"POCKET_{label}" if label else "POCKET"
                elif not COLUMN_HEADER_RE.match(line):
                    print(f"  Warning: line {lineno} is NOT recognized as a pocket header "
                          f"({line[:40]!r}). A pocket header must contain the word POCKET. "
                          f"Subsequent molecules stay in "
                          f"'{current_pocket or 'UNASSIGNED'}'.", file=sys.stderr)
                continue

            # Trennzeichen bestimmen: Komma, ersatzweise Tab oder Semikolon
            separator = "," if "," in line else next(
                (s for s in FALLBACK_SEPARATORS if s in line), None)
            if separator is None:
                print(f"  Warning: line {lineno} has no separator, skipped.",
                      file=sys.stderr)
                continue

            # SMILES enthalten nie ein Komma -> vollstaendig splitten ist sicher und
            # laesst zusaetzliche Spalten (z. B. Docking-Scores) unbeachtet.
            parts = [p.strip() for p in line.split(separator)]
            rank_str, smiles = parts[0], parts[1] if len(parts) > 1 else ""
            if not smiles:
                print(f"  Warning: line {lineno} has no SMILES, skipped.", file=sys.stderr)
                continue

            if current_pocket is None:
                print(f"  Warning: line {lineno} appears before the first #POCKET header.",
                      file=sys.stderr)
            try:
                rank_num = int(rank_str)
            except ValueError:
                rank_num = len(entries) + 1
                print(f"  Warning: line {lineno} has no numeric rank "
                      f"('{rank_str}') -> using {rank_num} instead.", file=sys.stderr)

            entries.append({"pocket": current_pocket or "UNASSIGNED", "rank": rank_num,
                            "smiles": smiles, "line": lineno})
    return entries


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "unnamed"


def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def run_predictions(entries, include_physchem, atc_code, num_workers):
    """Laedt das Modell einmal und praediziert alle eindeutigen SMILES."""
    from admet_ai import ADMETModel

    unique_smiles = list(dict.fromkeys(e["smiles"] for e in entries))
    print(f"{len(entries)} entries, of which {len(unique_smiles)} unique SMILES "
          f"({len(entries) - len(unique_smiles)} duplicates are not computed twice).")

    print("Loading ADMET-AI model (once) ...")
    model = ADMETModel(include_physchem=include_physchem, atc_code=atc_code,
                       num_workers=num_workers)

    print(f"Predicting {len(unique_smiles)} molecules ...")
    preds = model.predict(smiles=unique_smiles)
    preds.index.name = "smiles"
    preds = preds.reset_index()

    valid = set(preds["smiles"])
    failed = [e for e in entries if e["smiles"] not in valid]

    merged = pd.DataFrame(entries).merge(preds, on="smiles", how="inner")
    merged = merged.sort_values(["pocket", "rank"], kind="stable").reset_index(drop=True)
    return model, merged, failed


def atc_codes_available() -> list[str]:
    """Liest die gültigen ATC-Bezeichner aus der DrugBank-Referenz von ADMET-AI.

    Es sind Klartext-Namen in Kleinschreibung ('dermatologicals', 'antiinfectives
    for systemic use', ...), keine Buchstabencodes wie 'D'.
    """
    from admet_ai.drugbank import get_drugbank_unique_atc_codes
    return list(get_drugbank_unique_atc_codes())


def resolve_atc_code(atc_code: str | None) -> str | None:
    """Normalisiert den ATC-Bezeichner und prüft ihn, BEVOR das Modell geladen wird.

    ADMET-AI legt seine ATC-Zuordnung in Kleinschreibung an, vergleicht die Eingabe
    aber ohne sie zu normalisieren -- 'DERMATOLOGICALS' schlägt deshalb fehl, obwohl
    der Bezeichner existiert. Hier wird das abgefangen und ein unbekannter Wert mit
    Vorschlägen gemeldet, statt mitten im Modellaufbau abzustürzen.
    """
    if atc_code is None:
        return None

    normalized = atc_code.strip().lower()
    # Leerer String und reine Leerzeichen gelten wie None: kein Filter.
    if not normalized:
        return None

    try:
        valid = atc_codes_available()
    except Exception as exc:                                    # noqa: BLE001
        print(f"  Warning: ATC identifier not verifiable ({exc}). Using "
              f"'{normalized}' unchecked.", file=sys.stderr)
        return normalized

    if normalized in valid:
        if normalized != atc_code:
            print(f"  ATC identifier '{atc_code}' normalized to '{normalized}'.")
        return normalized

    import difflib
    # Teilstring-Suche erst ab drei Zeichen, sonst trifft etwa "d" fast jeden Namen
    contains = [c for c in valid if normalized in c] if len(normalized) >= 3 else []
    similar = difflib.get_close_matches(normalized, valid, n=5, cutoff=0.6)
    suggestions = list(dict.fromkeys(contains + similar))[:8]

    print(f"\nUnknown ATC identifier: '{atc_code}'", file=sys.stderr)
    print("ADMET-AI expects a lowercase plain-text name, not a letter code "
          "such as 'D'.", file=sys.stderr)
    if suggestions:
        print("Did you mean one of these?", file=sys.stderr)
        for s in suggestions:
            print(f"    {s}", file=sys.stderr)
    else:
        print("No similar identifier found.", file=sys.stderr)
    print(f"\nTo list all {len(valid)} valid identifiers:  "
          f"python admet_report.py --list-atc\n", file=sys.stderr)
    raise SystemExit(1)


def percentile_suffix(columns) -> str | None:
    for column in columns:
        index = column.find("drugbank")
        if index != -1:
            return column[index:]
    return None


def add_aggregates(df: pd.DataFrame, suffix: str | None) -> pd.DataFrame:
    """Ergaenzt Aggregat-, Flag- und Strukturspalten."""
    df = df.copy()

    def agg_max(cols, name):
        present = [c for c in cols if c in df.columns]
        if not present:
            return
        df[name] = df[present].max(axis=1)
        if suffix:
            # Perzentil des Endpunkts, der den Maximalwert liefert.
            # Als Series aufgebaut, damit die Zuordnung ueber den Index laeuft
            # und nicht ueber die Position.
            argmax = df[present].idxmax(axis=1)
            df[f"{name}_{suffix}"] = pd.Series(
                {i: (df.at[i, f"{col}_{suffix}"]
                     if isinstance(col, str) and f"{col}_{suffix}" in df.columns else None)
                 for i, col in argmax.items()})

    agg_max(TOX21_COLUMNS, "tox21_max")
    agg_max(CYP_INHIBITION_COLUMNS, "cyp_inhibition_max")

    present_tox21 = [c for c in TOX21_COLUMNS if c in df.columns]
    if present_tox21:
        df["tox21_n_above_0.5"] = (df[present_tox21] > 0.5).sum(axis=1)
        if suffix:
            df[f"tox21_n_above_0.5_{suffix}"] = None

    # Physchem-Flags
    flags, counts, charges, formulas = [], [], [], []
    for _, row in df.iterrows():
        violated = [label for key, limit, label in PHYSCHEM_RULES
                    if key in df.columns and pd.notna(row.get(key)) and row[key] > limit]
        flags.append(" · ".join(violated))
        counts.append(len(violated))
        mol = Chem.MolFromSmiles(row["smiles"])
        charges.append(Chem.GetFormalCharge(mol) if mol else None)
        formulas.append(rdMolDescriptors.CalcMolFormula(mol) if mol else "")
    df["physchem_flags"] = flags
    df["n_physchem_violations"] = counts
    df["formal_charge"] = charges
    df["molecular_formula"] = formulas
    return df


# --------------------------------------------------------------------------- #
# Grafiken
# --------------------------------------------------------------------------- #
def structure_png(smiles: str, path: Path, size=(460, 330)) -> Path | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())
    return path


def radial_png(row, suffix: str, path: Path) -> Path | None:
    """Nutzt plot_radial_summary aus ADMET-AI, damit der Plot identisch bleibt."""
    from admet_ai.plot import plot_radial_summary
    try:
        data = plot_radial_summary(property_id_to_percentile=row.to_dict(),
                                   percentile_suffix=suffix, image_type="png")
    except KeyError as exc:
        print(f"  Radar plot skipped (missing column {exc}).", file=sys.stderr)
        return None
    path.write_bytes(data)
    return path


def scatter_png(model, group: pd.DataFrame, x_prop: str, y_prop: str,
                max_num: int | None, path: Path) -> Path | None:
    """Nutzt plot_drugbank_reference aus ADMET-AI."""
    from admet_ai.plot import plot_drugbank_reference
    if model.drugbank is None:
        return None
    try:
        data = plot_drugbank_reference(
            preds_df=group, drugbank_df=model.drugbank_atc_filtered,
            x_property_name=x_prop, y_property_name=y_prop,
            max_molecule_num=max_num, image_type="png")
    except KeyError as exc:
        print(f"  Scatter plot skipped (unknown property {exc}).", file=sys.stderr)
        return None
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- #
# PDF-Bausteine
# --------------------------------------------------------------------------- #
def shade(direction: str, pct):
    if pct is None or pd.isna(pct) or direction == "ctx":
        return None, INK
    p = pct if direction == "up" else 100 - pct
    return (GOOD, GOOD_T) if p >= 60 else (WARN, WARN_T) if p >= 25 else (RISK, RISK_T)


# Schwellen fuer Spalten, die in der Uebersicht den Absolutwert zeigen:
# (Warnschwelle, Risikoschwelle) -- oberhalb der Risikoschwelle rot.
#
# Absichtlich leer: rohe Deskriptoren wie das Molekulargewicht werden nicht
# eingefaerbt, weil ihre Bewertung ueber die Physchem-Flags laeuft. Wer eine
# Wertspalte doch ampeln will, traegt sie hier ein, z. B.
#     VALUE_THRESHOLDS = {"molecular_weight": (350, 500)}
VALUE_THRESHOLDS: dict[str, tuple[float, float]] = {}


def shade_by_threshold(key: str, value):
    """Ampel anhand des angezeigten Absolutwerts statt des Perzentils."""
    limits = VALUE_THRESHOLDS.get(key)
    if limits is None or value is None or pd.isna(value):
        return None, INK
    warn_at, risk_at = limits
    value = float(value)
    return (RISK, RISK_T) if value > risk_at else (WARN, WARN_T) if value > warn_at else (GOOD, GOOD_T)


def bar(pct, direction, width=56, height=4.6) -> Drawing:
    d = Drawing(width, height + 1)
    d.add(Rect(0, 1, width, height, fillColor=BAR_BG, strokeColor=None))
    if pct is not None and not pd.isna(pct):
        fill, _ = shade(direction, pct)
        col = (GOOD_B if fill is GOOD else WARN_B if fill is WARN
               else RISK_B if fill is RISK else colors.HexColor("#9a9a9a"))
        d.add(Rect(0, 1, max(width * float(pct) / 100, 0.8), height,
                   fillColor=col, strokeColor=None))
    return d


def fmt_value(v) -> str:
    """Formatiert Zahlen. Ganzzahlige Werte unter 100 werden ohne Nachkommastellen
    ausgegeben -- auch wenn sie als numpy-Typ vorliegen (HBD, Tox21-Anzahl)."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pd.isna(v):
        return "—"
    if v.is_integer() and abs(v) < 1000:
        return str(int(v))
    return f"{v:.2f}" if abs(v) >= 100 else f"{v:.3f}" if abs(v) >= 1 else f"{v:.4f}"


def fmt_int(n) -> str:
    return f"{n:,}" if n else "—"


def fmt_pct(p) -> str:
    return "—" if p is None or pd.isna(p) else f"{float(p):.1f}"


def cover_page(args, pockets, stats, suffix, story):
    story.append(Paragraph("ADMET Profile Report", H1))
    story.append(Paragraph("Core set evaluation · ADMET-AI (Chemprop / TDC)", SUBT))

    failed_text = f"{stats['failed']} invalid SMILES" if stats["failed"] else "none"
    # Ist ein ATC-Filter aktiv, beziehen sich ALLE Perzentile auf die gefilterte
    # Teilmenge -- der Referenzsatz sagt das jetzt selbst, statt es dem Leser
    # zu ueberlassen, es aus der ATC-Zeile zu erschliessen.
    reference = ("DrugBank approved (ATC-filtered)" if args.atc_code
                 else "DrugBank approved")
    meta = Table([
        ["Input file", args.input.name, "Created", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Pockets", str(len(pockets)),
         "Entries", f"{stats['valid']} evaluated, {stats['unique']} unique"],
        ["Reference set", reference, "Discarded", failed_text],
        ["ATC filter", args.atc_code or "none", "Detail pages",
         "all" if args.detail_top is None else f"top {args.detail_top} per pocket"],
    ], colWidths=[30 * mm, 52 * mm, 34 * mm, 52 * mm])
    meta.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.2),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8.2),
        ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 8.2),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED), ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE), ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta)
    story.append(Spacer(1, 3))
    # Macht nachpruefbar, gegen welche Spaltenmenge die Perzentile gerechnet sind.
    story.append(Paragraph(
        "All percentiles in this report refer to the column set "
        f"<font face='Courier'>{esc(suffix)}</font>."
        if suffix else
        "No DrugBank percentiles available — color coding and percentile columns are empty.",
        SMALL))
    story.append(Spacer(1, 8))

    story.append(Paragraph("How to read this", H2))
    rows = [
        ("Color code", "Refers to the <b>percentile against approved drugs</b>, "
                       "already direction-corrected: green = favorable, yellow = attention, red = "
                       "unfavorable. Raw descriptors such as molecular weight, TPSA or H-bond donors "
                       "are deliberately left <b>uncolored</b>."),
        ("Direction", "↑ higher is favorable · ↓ higher means risk · ~ context-dependent "
                      "or purely descriptive; assessment depends on target profile and route of "
                      "administration."),
        ("Score 0–1", "Classification endpoints return an <b>uncalibrated model score</b> between "
                      "0 and 1. It ranks molecules against each other but must not be read as a "
                      "probability — a score of 0.7 does not mean a 70 % chance."),
        ("n (TDC)", "Number of compounds in the training set of the respective endpoint."),
        ("Best value", "Best published value on the TDC leaderboard — the <b>upper bound of what "
                       "is currently achievable</b>, not the performance of ADMET-AI itself. "
                       "<i>no BM</i> = no benchmark available in the TDC ADMET Benchmark Group; "
                       "the prediction is unvalidated there. It is set in grey, not red, because "
                       "it describes the model, not the molecule. References in the separate "
                       "sources document."),
    ]
    leg = Table([[Paragraph(f"<b>{k}</b>", BODY), Paragraph(v, BODY)] for k, v in rows],
                colWidths=[24 * mm, 144 * mm])
    leg.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("TOPPADDING", (0, 0), (-1, -1), 3),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(leg)


def overview_section(pocket, group, suffix, scatter, story, detail_top):
    story.append(Paragraph(f"{pocket} — Overview", H2))

    # Kopfzeile zweizeilig: Name + Richtungspfeil oben, Einheit darunter. Damit ist
    # auf einen Blick klar, welche Spalte ein Perzentil (%) und welche einen
    # Absolutwert (Da) zeigt, und warum hERG 92 rot und HIA 92 gruen ist.
    header = [Paragraph("Rank", HCELL), Paragraph("Formula", HCELL)]
    for name, key, kind in OVERVIEW_COLUMNS:
        unit = "Da" if kind == "value" else "%"
        header.append(Paragraph(
            f"{esc(name)} {ARROW[DIRECTION.get(key, 'ctx')]}<br/>"
            f"<font size='6' color='#6b6b6b'>{unit}</font>", HCELLC))
    header.append(Paragraph("Flags<br/><font size='6' color='#6b6b6b'>n</font>", HCELLC))
    rows, cell_styles = [header], []
    for i, (_, row) in enumerate(group.iterrows(), start=1):
        line = [str(int(row["rank"])), row.get("molecular_formula", "") or "—"]
        for _, key, kind in OVERVIEW_COLUMNS:
            if kind == "value":
                line.append(fmt_value(row.get(key)))
            else:
                pct = row.get(f"{key}_{suffix}") if suffix else None
                line.append(fmt_pct(pct))
        line.append(str(int(row.get("n_physchem_violations", 0))))
        rows.append(line)

        # Ampel je Zelle. Wert-Spalten werden am angezeigten Absolutwert eingefaerbt,
        # damit Zahl und Farbe sich auf dasselbe beziehen; Perzentil-Spalten am Perzentil.
        for j, (_, key, kind) in enumerate(OVERVIEW_COLUMNS, start=2):
            if kind == "value":
                bg, tc = shade_by_threshold(key, row.get(key))
            else:
                bg, tc = shade(DIRECTION.get(key, "ctx"), row.get(f"{key}_{suffix}")
                               if suffix else None)
            if bg:
                cell_styles += [("BACKGROUND", (j, i), (j, i), bg),
                                ("TEXTCOLOR", (j, i), (j, i), tc)]
        nviol = int(row.get("n_physchem_violations", 0))
        if nviol:
            bg, tc = (RISK, RISK_T) if nviol >= 2 else (WARN, WARN_T)
            cell_styles += [("BACKGROUND", (len(header) - 1, i), (len(header) - 1, i), bg),
                            ("TEXTCOLOR", (len(header) - 1, i), (len(header) - 1, i), tc)]

    widths = [11 * mm, 34 * mm] + [15 * mm] * len(OVERVIEW_COLUMNS) + [12 * mm]
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONT", (0, 1), (-1, -1), "Helvetica", 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        # Schmales Padding in den Zahlenspalten, damit die Kopfzeile einzeilig bleibt
        ("LEFTPADDING", (2, 0), (-1, -1), 2), ("RIGHTPADDING", (2, 0), (-1, -1), 2),
    ] + cell_styles))
    story.append(table)
    story.append(Spacer(1, 3))
    note = ("Percentile columns (%) are already direction-corrected in their coloring, the "
            "printed number is not: a high hERG percentile is red, a high HIA percentile green. "
            "<b>0 or 100 means the value lies outside the DrugBank reference range</b> — only the "
            "raw value on the detail page is meaningful then.")
    if detail_top is not None:
        note += f" Detail pages follow for the {detail_top} top-ranked molecules."
    story.append(Paragraph(note, SMALL))

    if scatter is not None:
        story.append(Spacer(1, 8))
        story.append(Image(str(scatter), width=110 * mm, height=110 * mm))
        story.append(Paragraph("DrugBank reference: input molecules (red stars) against the "
                               "approved drugs. Numbering follows rank.", SMALL))


def detail_page(row, pocket, suffix, struct, radial, story):
    story.append(Paragraph(f"{pocket} · Rank {int(row['rank'])}", H3))
    story.append(Paragraph("Detail profile", H1))
    story.append(Spacer(1, 4))

    left = Image(str(struct), width=84 * mm, height=60 * mm) if struct else Paragraph("", BODY)
    right = Image(str(radial), width=62 * mm, height=62 * mm) if radial else Paragraph("", BODY)
    head = Table([[left, right]], colWidths=[94 * mm, 74 * mm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (0, 0), "MIDDLE"),
                              ("VALIGN", (1, 0), (1, 0), "TOP"),
                              ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(head)
    if radial:
        story.append(Paragraph("Radar plot: DrugBank percentiles, direction-corrected — further "
                               "out is always better.", SMALL))
    story.append(Spacer(1, 5))

    charge = row.get("formal_charge")
    idrow = Table([[
        Paragraph(f"<b>Formula</b>  {esc(row.get('molecular_formula', '—'))}", SMALL),
        Paragraph(f"<b>MW</b>  {fmt_value(row.get('molecular_weight'))} Da", SMALL),
        Paragraph(f"<b>Charge</b>  {'—' if charge is None else int(charge)}", SMALL),
        Paragraph(f"<b>Physchem violations</b>  {int(row.get('n_physchem_violations', 0))} "
                  f"of {len(PHYSCHEM_RULES)}", SMALL),
    ]], colWidths=[62 * mm, 30 * mm, 24 * mm, 52 * mm])
    idrow.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND),
                               ("TOPPADDING", (0, 0), (-1, -1), 4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                               ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
    story.append(idrow)

    notes = []
    if row.get("physchem_flags"):
        notes.append(f"Violations: {row['physchem_flags']}")
    if charge is not None and abs(int(charge)) >= 2:
        notes.append(f"Formal charge {int(charge)} — the Crippen logP is unreliable for "
                     f"multiply charged species.")
    if notes:
        story.append(Spacer(1, 2))
        for n in notes:
            story.append(Paragraph(n, SMALL))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>SMILES</b>", SMALL))
    story.append(Paragraph(esc(row["smiles"]), MONO))
    story.append(Spacer(1, 6))

    # Die Balkenspalte bleibt ohne Kopf: sie zeigt nichts anderes als die
    # Perzentilspalte daneben. "Position" las sich wie ein Rang im Pocket.
    data = [["Property", "Value", "Unit", "Pct.", "", "n (TDC)", "Best value"]]
    style = [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.3),
             ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
             ("ALIGN", (1, 1), (3, -1), "RIGHT"), ("ALIGN", (5, 1), (5, -1), "RIGHT"),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.7), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    r = 1
    for group_name, items in CORE_GROUPS.items():
        data.append([group_name, "", "", "", "", "", ""])
        style += [("SPAN", (0, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), GROUPBG),
                  ("FONT", (0, r), (-1, r), "Helvetica-Bold", 7.0),
                  ("TEXTCOLOR", (0, r), (-1, r), MUTED), ("TOPPADDING", (0, r), (-1, r), 3.6)]
        r += 1
        for key, label, unit, direction, n, metric, best in items:
            if key not in row.index:
                continue
            pct = row.get(f"{key}_{suffix}") if suffix else None
            arrow = ARROW[direction]
            if best is None:
                bm = "—"
            elif best == "no BM":
                # Grau kursiv statt rot: die Aussage betrifft das Modell, nicht das
                # Molekuel, und darf nicht wie eine Ampel-Warnung gelesen werden.
                bm = f"<i><font color='#6b6b6b'>{best}</font></i>"
            else:
                bm = f"{metric} {best}"
            data.append([Paragraph(f"{label} &nbsp;{arrow}", CELL), fmt_value(row.get(key)),
                         Paragraph(unit, RCELL), fmt_pct(pct), bar(pct, direction),
                         fmt_int(n), Paragraph(bm, RCELL)])
            bg, tc = shade(direction, pct)
            if bg:
                style += [("BACKGROUND", (3, r), (3, r), bg), ("TEXTCOLOR", (3, r), (3, r), tc)]
            style += [("FONT", (1, r), (5, r), "Helvetica", 7.2),
                      ("TEXTCOLOR", (5, r), (5, r), MUTED),
                      ("LINEBELOW", (0, r), (-1, r), 0.25, RULE)]
            r += 1

    table = Table(data, colWidths=[52 * mm, 18 * mm, 21 * mm, 12 * mm, 21 * mm, 17 * mm, 27 * mm],
                  repeatRows=1)
    table.setStyle(TableStyle(style))
    story.append(table)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Percentiles refer to approved DrugBank drugs across all routes of administration. At "
        "percentile 0 or 100 the value lies outside the reference range — only the raw value is "
        "meaningful then. For aggregated rows (max of several endpoints) the percentile shown is "
        "that of the single endpoint supplying the maximum, not a percentile of the aggregate. "
        "Best values apply to molecules within the training domain.",
        SMALL))


# Richtungs-Lookup fuer die Uebersichtstabelle
DIRECTION = {key: direction
             for items in CORE_GROUPS.values()
             for key, _, _, direction, *_ in items}


def build_report(model, merged, suffix, args, out_path: Path, tmpdir: Path,
                 stats: dict) -> None:
    doc = BaseDocTemplate(str(out_path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                          topMargin=15 * mm, bottomMargin=16 * mm,
                          title="ADMET Profile Report", author="ADMET-AI Pipeline")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def furniture(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 10 * mm, "ADMET-AI Profile Report")
        canvas.drawRightString(A4[0] - doc.leftMargin, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, 13 * mm, A4[0] - doc.leftMargin, 13 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=furniture)])

    pockets = list(dict.fromkeys(merged["pocket"]))
    story: list = []
    cover_page(args, pockets, stats, suffix, story)

    for pocket in pockets:
        group = merged[merged["pocket"] == pocket].sort_values("rank", kind="stable")
        if group.empty:
            continue
        story.append(PageBreak())

        scatter = None
        if not args.no_scatter:
            scatter = scatter_png(model, group, args.x_property, args.y_property,
                                  args.max_molecule_num,
                                  tmpdir / f"scatter_{sanitize(pocket)}.png")

        detail = group if args.detail_top is None else group.head(args.detail_top)
        overview_section(pocket, group, suffix, scatter, story, args.detail_top)

        for _, row in tqdm(detail.iterrows(), total=len(detail),
                           desc=f"Detail pages {pocket}"):
            story.append(PageBreak())
            stem = f"{sanitize(pocket)}_r{int(row['rank']):03d}"
            struct = structure_png(row["smiles"], tmpdir / f"struct_{stem}.png")
            radial = (radial_png(row, suffix, tmpdir / f"radial_{stem}.png")
                      if suffix else None)
            detail_page(row, pocket, suffix, struct, radial, story)

    doc.build(story)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADMET-AI predictions and PDF report for pocket-based SMILES lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", "-i", type=Path, default=None,
                   help="Input text file with pockets. If omitted, IDE_CONFIG applies.")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("results"),
                   help="Output directory.")
    p.add_argument("--detail-top", type=int, default=None,
                   help="Detail pages only for the N top-ranked molecules per pocket. "
                        "If omitted, for all.")
    p.add_argument("--x-property", default="Human Intestinal Absorption",
                   help="x axis of the DrugBank scatter plot (plain name).")
    p.add_argument("--y-property", default="Clinical Toxicity",
                   help="y axis of the DrugBank scatter plot (plain name).")
    p.add_argument("--max-molecule-num", type=int, default=None,
                   help="Number the first N molecules in the scatter plot.")
    p.add_argument("--no-scatter", action="store_true",
                   help="Do not include DrugBank scatter plots in the report.")
    p.add_argument("--atc-code", default=None,
                   help="ATC identifier for filtering the DrugBank reference, e.g. "
                        "'dermatologicals'. Plain-text name, not a letter code. "
                        "Case is normalized.")
    p.add_argument("--list-atc", action="store_true",
                   help="Print all valid ATC identifiers and exit.")
    p.add_argument("--no-physchem", action="store_true",
                   help="Do not compute physicochemical properties (RDKit).")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers. Default: 0 without GPU, 8 with GPU.")
    p.add_argument("--no-csv", action="store_true",
                   help="Do not write the raw-data CSV (PDF only).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only parse the input file and print an overview.")

    args = p.parse_args()

    # Ohne Kommandozeilenargumente (Start aus der IDE) gilt IDE_CONFIG.
    if len(sys.argv) == 1:
        print("No command line arguments — using IDE_CONFIG from the file header.")
        for key, value in IDE_CONFIG.items():
            if key in ("input", "output_dir") and value is not None:
                value = Path(value).expanduser()
            setattr(args, key, value)

    # --list-atc ist eine reine Auskunft und braucht kein Input-File
    if args.input is None and not args.list_atc:
        p.error("No input file given: either set --input or fill in "
                "IDE_CONFIG['input'] in the file header.")
    return args


def main() -> int:
    args = parse_args()

    if args.list_atc:
        try:
            codes = atc_codes_available()
        except Exception as exc:                                # noqa: BLE001
            print(f"ATC identifiers not readable: {exc}", file=sys.stderr)
            return 1
        print(f"{len(codes)} valid ATC identifiers in the DrugBank reference:\n")
        for code in codes:
            print(f"  {code}")
        return 0

    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Parsing {args.input} ...")
    entries = parse_pocket_file(args.input)
    if not entries:
        print("No SMILES found — please check the input format.", file=sys.stderr)
        return 1

    pockets = sorted({e["pocket"] for e in entries})
    print(f"  {len(entries)} entries in {len(pockets)} pockets "
          f"({len({e['smiles'] for e in entries})} unique SMILES)")

    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "parsed_input.csv"
        pd.DataFrame(entries).to_csv(path, index=False)
        print(f"  Dry run: parsed entries -> {path}")
        for pocket in pockets:
            print(f"    {pocket}: {sum(1 for e in entries if e['pocket'] == pocket)} molecules")
        return 0

    # ATC-Bezeichner normalisieren und prüfen, bevor das Modell geladen wird
    args.atc_code = resolve_atc_code(args.atc_code)

    if args.no_physchem:
        print("  Note: --no-physchem active — molecular weight, logP, TPSA, QED and the "
              "Lipinski check are missing from the report, as are the physchem flags.",
              file=sys.stderr)

    model, merged, failed = run_predictions(entries, not args.no_physchem,
                                            args.atc_code, args.num_workers)
    if merged.empty:
        print("Not a single molecule could be evaluated — please check the SMILES.",
              file=sys.stderr)
        return 1

    suffix = percentile_suffix(merged.columns)
    if suffix is None:
        print("  Warning: no DrugBank percentiles found — color coding, radar plots and "
              "scatter plots are omitted.", file=sys.stderr)
    else:
        missing = [p for p in RADIAL_PROPERTIES if f"{p}_{suffix}" not in merged.columns]
        if missing:
            print(f"  Warning: missing for the radar plot: {', '.join(missing)}.",
                  file=sys.stderr)

    merged = add_aggregates(merged, suffix)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_csv:
        csv_path = args.output_dir / "admet_all.csv"
        merged.drop(columns=["line"]).to_csv(csv_path, index=False)
        print(f"  Raw data saved: {csv_path}")

    failed_path = args.output_dir / "failed_smiles.txt"
    with open(failed_path, "w", encoding="utf-8") as fh:
        for e in failed:
            fh.write(f"{e['pocket']}\t{e['rank']}\tline {e['line']}\t{e['smiles']}\n")
    if failed:
        print(f"  {len(failed)} invalid SMILES -> {failed_path}")

    # Pockets, deren Moleküle alle verworfen wurden, tauchen im Report nicht auf
    lost = sorted({e["pocket"] for e in failed} - set(merged["pocket"]))
    if lost:
        print(f"  Note: no evaluable molecule, therefore not in the report: "
              f"{', '.join(lost)}", file=sys.stderr)

    stats = {"valid": len(merged), "unique": merged["smiles"].nunique(),
             "failed": len(failed)}

    report_path = args.output_dir / "admet_report.pdf"
    tmpdir = Path(tempfile.mkdtemp(prefix="admet_report_"))
    try:
        print("Generating PDF report ...")
        build_report(model, merged, suffix, args, report_path, tmpdir, stats)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    size_kb = report_path.stat().st_size / 1024
    print(f"Done. Report: {report_path} ({size_kb:,.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
