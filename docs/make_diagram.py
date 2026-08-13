#!/usr/bin/env python3
"""
make_diagram.py
===============
Erzeugt das Pipeline-Diagramm als SVG – je eine Fassung fuer helles und
dunkles Theme, layoutgleich, weil aus derselben Quelle generiert.

    python3 make_diagram.py docs/

Warum SVG: skaliert verlustfrei (PowerPoint importiert es seit 2016 als
Vektor), rendert nativ in GitHub-Markdown und bleibt als Text diffbar.

Bewusst verzichtet auf: Filter, Gradienten, eingebettete Schriften.
PowerPoint rastert bei Filtern gerne und ersetzt Schriften stillschweigend –
flache Flaechen und ein Systemschrift-Stack sind der robustere Weg.
"""

from __future__ import annotations

import sys
from pathlib import Path

W, H = 960, 740

FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,"
        "'Helvetica Neue',Arial,sans-serif")
MONO = ("ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,"
        "'Liberation Mono',monospace")

# ── Themes ────────────────────────────────────────────────────────────
LIGHT = {
    "name":      "light",
    "text":      "#0F172A",
    "muted":     "#475569",
    "faint":     "#64748B",
    "arrow":     "#94A3B8",
    "chip_bg":   "#F1F5F9",
    "chip_line": "#CBD5E1",
    "stages": [
        {"fill": "#F0FDFA", "line": "#5EEAD4", "accent": "#0D9488", "badge_fg": "#FFFFFF"},
        {"fill": "#EFF6FF", "line": "#93C5FD", "accent": "#2563EB", "badge_fg": "#FFFFFF"},
        {"fill": "#FAF5FF", "line": "#D8B4FE", "accent": "#7C3AED", "badge_fg": "#FFFFFF"},
    ],
}

DARK = {
    "name":      "dark",
    "text":      "#F1F5F9",
    "muted":     "#94A3B8",
    "faint":     "#64748B",
    "arrow":     "#475569",
    "chip_bg":   "#1E293B",
    "chip_line": "#334155",
    "stages": [
        {"fill": "#0B2F2C", "line": "#115E59", "accent": "#2DD4BF", "badge_fg": "#042F2E"},
        {"fill": "#14243F", "line": "#1E3A8A", "accent": "#60A5FA", "badge_fg": "#0B1B33"},
        {"fill": "#231242", "line": "#5B21B6", "accent": "#A78BFA", "badge_fg": "#1B0B33"},
    ],
}

# ── Inhalt ────────────────────────────────────────────────────────────
STAGES = [
    {
        "n": "1", "title": "PREPARATION", "container": "sdf_to_pdbqt.sif",
        "badge": "CPU",
        "lines": ["RDKit + Meeko, parallel process pool",
                  "SDF → PDBQT directly, no PDB intermediate"],
    },
    {
        "n": "2", "title": "DOCKING", "container": "unidock-gpu.sif",
        "badge": "GPU",
        "lines": ["Uni-Dock, one persistent worker per GPU",
                  "Batched ligands, work-stealing chunk queue"],
    },
    {
        "n": "3", "title": "RESCORING", "container": "rescoring-gpu.sif",
        "badge": "GPU",
        "lines": ["gnina CNNaffinity · CNNscore · ΔLin_F9XGB",
                  "Exponential consensus ranking, refinement"],
    },
]

FLOWS = ["data/PDBQT/", "RESULTS/<target>/*_docked.pdbqt"]

CARD_X, CARD_W, CARD_H = 60, 840, 116
CARD_Y = [128, 320, 512]
RAIL_X = 110          # senkrechte Flusslinie, mittig unter dem Badge


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build(theme: dict) -> str:
    t = theme
    o: list[str] = []
    add = o.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" '
        f'aria-label="Three-stage virtual screening pipeline: preparation, '
        f'docking, rescoring">')
    add(f'<title>dock-cascade pipeline ({t["name"]})</title>')

    # Pfeilspitze
    add('<defs>')
    add(f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>')
    add('</defs>')

    add(f'<g font-family="{FONT}">')

    # ── Eingang ───────────────────────────────────────────────────────
    add(f'<rect x="{CARD_X}" y="52" width="200" height="34" rx="17" '
        f'fill="{t["chip_bg"]}" stroke="{t["chip_line"]}"/>')
    add(f'<text x="{CARD_X + 20}" y="74" font-family="{MONO}" font-size="14" '
        f'fill="{t["muted"]}">LIB/*.sdf</text>')
    add(f'<text x="{CARD_X + 224}" y="74" font-size="13" fill="{t["faint"]}">'
        f'input library</text>')

    # Linie Eingang → Stufe 1
    add(f'<line x1="{RAIL_X}" y1="86" x2="{RAIL_X}" y2="{CARD_Y[0] - 6}" '
        f'stroke="{t["arrow"]}" stroke-width="2" marker-end="url(#ah)"/>')

    # ── Stufen ────────────────────────────────────────────────────────
    for i, st in enumerate(STAGES):
        s, y = t["stages"][i], CARD_Y[i]

        add(f'<rect x="{CARD_X}" y="{y}" width="{CARD_W}" height="{CARD_H}" '
            f'rx="16" fill="{s["fill"]}" stroke="{s["line"]}" stroke-width="1.5"/>')
        # Akzentkante links
        add(f'<path d="M{CARD_X+1} {y+16} a15,15 0 0 1 15,-15 h6 v{CARD_H-2} '
            f'h-6 a15,15 0 0 1 -15,-15 z" fill="{s["accent"]}"/>')

        # Nummernbadge
        add(f'<rect x="{RAIL_X - 22}" y="{y + 26}" width="44" height="44" '
            f'rx="12" fill="{s["accent"]}"/>')
        add(f'<text x="{RAIL_X}" y="{y + 56}" text-anchor="middle" '
            f'font-size="22" font-weight="700" fill="{s["badge_fg"]}">'
            f'{st["n"]}</text>')

        # Titel + Container
        add(f'<text x="{CARD_X + 92}" y="{y + 47}" font-size="19" '
            f'font-weight="600" fill="{t["text"]}" letter-spacing="0.4">'
            f'{st["title"]}</text>')
        add(f'<text x="{CARD_X + 92}" y="{y + 72}" font-family="{MONO}" '
            f'font-size="13" fill="{t["faint"]}">{st["container"]}</text>')

        # CPU/GPU-Pille
        pw = 52 if st["badge"] == "CPU" else 52
        add(f'<rect x="{CARD_X + 92}" y="{y + 84}" width="{pw}" height="22" '
            f'rx="11" fill="none" stroke="{s["accent"]}" stroke-width="1.2"/>')
        add(f'<text x="{CARD_X + 92 + pw/2}" y="{y + 99}" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="{s["accent"]}" '
            f'letter-spacing="0.6">{st["badge"]}</text>')

        # Beschreibung
        add(f'<line x1="{CARD_X + 320}" y1="{y + 28}" x2="{CARD_X + 320}" '
            f'y2="{y + CARD_H - 28}" stroke="{s["line"]}" stroke-width="1.5"/>')
        for j, line in enumerate(st["lines"]):
            add(f'<text x="{CARD_X + 348}" y="{y + 52 + j * 26}" font-size="15" '
                f'fill="{t["muted"]}">{esc(line)}</text>')

    # ── Fluss zwischen den Stufen ─────────────────────────────────────
    for i, label in enumerate(FLOWS):
        y0 = CARD_Y[i] + CARD_H
        y1 = CARD_Y[i + 1]
        add(f'<line x1="{RAIL_X}" y1="{y0}" x2="{RAIL_X}" y2="{y1 - 6}" '
            f'stroke="{t["arrow"]}" stroke-width="2" marker-end="url(#ah)"/>')
        add(f'<text x="{RAIL_X + 22}" y="{(y0 + y1) / 2 + 5}" '
            f'font-family="{MONO}" font-size="13" fill="{t["faint"]}">'
            f'{esc(label)}</text>')

    # ── Ausgang ───────────────────────────────────────────────────────
    y_out = CARD_Y[2] + CARD_H
    add(f'<line x1="{RAIL_X}" y1="{y_out}" x2="{RAIL_X}" y2="{y_out + 34}" '
        f'stroke="{t["arrow"]}" stroke-width="2" marker-end="url(#ah)"/>')
    add(f'<rect x="{CARD_X}" y="{y_out + 42}" width="420" height="34" rx="17" '
        f'fill="{t["chip_bg"]}" stroke="{t["chip_line"]}"/>')
    add(f'<text x="{CARD_X + 20}" y="{y_out + 64}" font-family="{MONO}" '
        f'font-size="13" fill="{t["muted"]}">'
        f'RESULTS/&lt;target&gt;/rescoring_ligands_&lt;target&gt;.csv</text>')
    add(f'<text x="{CARD_X + 444}" y="{y_out + 64}" font-size="13" '
        f'fill="{t["faint"]}">ranked hit list</text>')

    add('</g>')
    add('</svg>')
    return "\n".join(o)


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    outdir.mkdir(parents=True, exist_ok=True)
    for theme, name in ((LIGHT, "pipeline.svg"), (DARK, "pipeline-dark.svg")):
        path = outdir / name
        path.write_text(build(theme), encoding="utf-8")
        print(f"  {path}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
