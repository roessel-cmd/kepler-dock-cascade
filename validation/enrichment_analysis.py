"""
enrichment_analysis.py
=========================
Validation analysis for DUD-E docking results (gnina rescoring).

Protokoll: Pose 1 aus dem Vina-Ranking, danach Re-Ranking nach ECR bzw.
den Einzelscores. Die Pose wird bewusst NICHT nach ECR gewaehlt — das
waere eine Auswahl auf derselben Groesse, die anschliessend bewertet wird.
ecr_cross_validation.py verfaehrt hier anders (Best-Pose nach ECR); die
sigma_fraction aus der Kreuzvalidierung ist deshalb nicht 1:1 auf dieses
Skript uebertragbar.

Pro Target: ROC/AUC, Enrichment Factors, Score-Verteilungen, PR/AUPRC,
Cohen's d und KS, Score-Korrelationen. Ueber alle Targets zusaetzlich
Vergleichs- und Orthogonalitaetsplots.

Experiment ueber die Umgebungsvariable EXPERIMENT_NAME waehlen (siehe
EXPERIMENTS unten), sonst greift der Default.

Usage:
    python3 enrichment_analysis.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("/home/roessel/gpu8.0/RESULTS")

TARGETS = [
    "aa2ar", "abl1", "ace", "aces", "csf1r", "cxcr4",
    "dpp4", "hxk4", "inha", "kith", "mapk2", "mcr",
    "mk01", "nram", "pa2ga", "pnph", "pur2", "pygm",
    "reni", "rock1", "rxra", "src", "tgfr1", "tryb1",
    "urok", "wee1", "xiap"
]

TARGETS = [
    (target, BASE_DIR / target / f"rescoring_poses_{target}.csv")
    for target in TARGETS
]

ANALYSIS_DIR_BASE = Path("/home/roessel/gpu8.0/RESULTS/analysis")
EF_FRACTIONS = [0.01, 0.05, 0.10]
DPI          = 800
N_BOOTSTRAP  = 1000
CI_LEVEL     = 0.95
RANDOM_SEED  = 42

EXPERIMENTS = {
    "classical_only":  ["score_vina", "score_deltalinf9xgb"],

    "dl_only":         ["score_cnnaffinity", "score_dense_cnnaffinity"],

    "one_per_family":  ["score_vina", "score_cnnaffinity", "score_deltalinf9xgb"],

    "dense_variant":   ["score_vina", "score_dense_cnnaffinity", "score_deltalinf9xgb"],

    "vinardo_variant": ["score_vinardo", "score_cnnaffinity", "score_deltalinf9xgb"],

    "classical_pair":  ["score_vina", "score_vinardo"],

    "four_way":        ["score_vina", "score_vinardo",
                        "score_cnnaffinity", "score_deltalinf9xgb"],
}

EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "one_per_family")

ANALYSIS_DIR = ANALYSIS_DIR_BASE / EXPERIMENT_NAME

# Dritte Spalte = higher_is_better. Vina/Vinardo/dLinF9XGB sind Energien
# (niedriger ist besser), CNNaffinity/CNNscore sind gnina-Vorhersagen
# (hoeher ist besser). Legt die Pipeline CNN-Werte negiert ab, muss hier
# wieder False stehen — check_score_directions() meldet diesen Fall.
SCORES_COMPONENTS_ALL = [
    ("score_vina",              "Vina Score",    False, True),
    ("score_vinardo",           "Vinardo",       False, True),
    ("score_cnnaffinity",       "CNNaffinity",   True,  True),
    ("score_cnnscore",          "CNNscore",      True,  True),
    ("score_deltalinf9xgb",     "ΔLinF9XGB",     False, True),
    ("score_dense_cnnaffinity", "DenseAffinity", True,  False),
    ("score_dense_cnnscore",    "DenseCNNscore", True,  False),
]

if EXPERIMENT_NAME not in EXPERIMENTS:
    raise ValueError(
        f"Unknown EXPERIMENT_NAME '{EXPERIMENT_NAME}'. "
        f"Choose from: {list(EXPERIMENTS.keys())}"
    )
_active_keys = set(EXPERIMENTS[EXPERIMENT_NAME])
_score_by_key = {s[0]: s for s in SCORES_COMPONENTS_ALL}
_unknown = [k for k in _active_keys if k not in _score_by_key]
if _unknown:
    raise ValueError(
        f"EXPERIMENTS['{EXPERIMENT_NAME}'] references unknown score keys: "
        f"{_unknown}. Known keys: {list(_score_by_key.keys())}"
    )

SCORES_COMPONENTS = [
    (col, label, hib, (col in _active_keys))
    for (col, label, hib, _) in SCORES_COMPONENTS_ALL
]

print(f"[analyze] Experiment: {EXPERIMENT_NAME}")
print(f"[analyze] Active scores: "
      f"{[s[1] for s in SCORES_COMPONENTS if s[3]]}")
print(f"[analyze] Output dir: {ANALYSIS_DIR}")

ECR_SIGMA_FRACTION = 4

SCORES = SCORES_COMPONENTS + [
    ("ecr_total",               "ECR Score",     True,  True),
]

SCORE_COLORS = {
    "Vina Score":      "#E74C3C",
    "Vinardo":         "#D35400",
    "CNNaffinity":     "#3498DB",
    "CNNscore":        "#2ECC71",
    "ΔLinF9XGB":       "#E67E22",
    "DenseAffinity":   "#F39C12",
    "DenseCNNscore":   "#1ABC9C",
    "ECR Score":       "#9B59B6",
}

TARGET_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
                 "#1ABC9C", "#E67E22", "#8E44AD"]

STYLE = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
}

ACTIVE_COLOR = "#2e7d32"
DECOY_COLOR  = "#1a73e8"

PAIR_COLORS = ["#1a73e8", "#7c4dff", "#2e7d32", "#e65100", "#c62828", "#00838f",
               "#4527a0", "#00695c", "#bf360c", "#1b5e20", "#880e4f", "#01579b",
               "#4e342e", "#263238", "#f57f17"]

SCORES_ACTIVE = [s for s in SCORES if s[3]]

# recompute_ecr braucht die Richtung je Spalte; aus demselben Katalog
# abgeleitet, damit beide nicht auseinanderlaufen koennen.
HIB_BY_COL = {col: hib for col, _lbl, hib, _en in SCORES}

# Paare nur aus den Komponenten: die ECR gegen ihre eigenen Bestandteile
# zu korrelieren waere zirkulaer.
SCORE_PAIRS = [
    (SCORES_ACTIVE[i], SCORES_ACTIVE[j])
    for i in range(len(SCORES_ACTIVE))
    for j in range(i + 1, len(SCORES_ACTIVE))
    if SCORES_ACTIVE[i][1] != "ECR Score"
    and SCORES_ACTIVE[j][1] != "ECR Score"
]


def recompute_ecr(df: pd.DataFrame, sigma_fraction: int = ECR_SIGMA_FRACTION) -> pd.DataFrame:
    """
    Recompute ecr_total from the currently enabled component scores.
    Uses equal weights across all enabled scores.

    Rangrichtung: Rang 1 muss der BESTE Wert sein, sonst dreht exp(-rank/sigma)
    den Score um. Frueher war hier ascending=True fest verdrahtet, unabhaengig
    von higher_is_better — fuer CNNaffinity und CNNscore also genau verkehrt.
    """
    enabled_cols = [col for col, _, _, en in SCORES_COMPONENTS if en]
    if not enabled_cols:
        print("  WARNING: No component scores enabled for ECR!")
        df["ecr_total"] = 0.0
        return df

    N = len(df)
    if N == 0:
        df["ecr_total"] = 0.0
        return df

    # N sind hier LIGANDEN (Pose-1-Filter lief schon). In
    # ecr_cross_validation.py sind es Posen — dieselbe sigma_fraction bedeutet
    # dort also ein anderes sigma.
    sigma = max(N / sigma_fraction, 1.0)
    weight = 1.0 / len(enabled_cols)

    ecr_total = np.zeros(N)
    for col in enabled_cols:
        if col not in df.columns:
            print(f"  WARNING: Column '{col}' not found, skipping for ECR.")
            continue
        valid_mask = df[col].notna()
        if valid_mask.sum() == 0:
            continue
        # Rang 1 muss der BESTE Wert sein, sonst dreht exp(-rank/sigma)
        # den Score um.
        hib = HIB_BY_COL[col]
        ranks = df.loc[valid_mask, col].rank(method="min", ascending=not hib)
        # exp(-rank/sigma) ist immer > 0, ein NaN traegt 0 bei und liegt
        # damit unter dem letzten Rang: fehlende Scores werden bestraft.
        ecr_col = np.zeros(N)
        ecr_col[valid_mask.values] = np.exp(-ranks.values / sigma)
        ecr_total += weight * ecr_col

    df["ecr_total"] = ecr_total
    labels = [lbl for _, lbl, _, en in SCORES_COMPONENTS if en]
    print(f"  ECR recomputed from {len(enabled_cols)} scores: {', '.join(labels)}")
    print(f"  (sigma_fraction={sigma_fraction}, sigma={sigma:.1f}, equal weights)")
    return df


def check_score_directions(all_aucs: dict) -> list[str]:
    """
    Selbsttest auf falsch gesetzte higher_is_better-Flags.

    Ein Score, der die Aktiven erkennt, hat eine AUC ueber 0.5. Liegt der
    Median ueber alle Targets systematisch DARUNTER, ist mit hoher
    Wahrscheinlichkeit das Vorzeichen im Katalog verkehrt — dann liefert
    derselbe Score gespiegelt (1 - AUC) ein brauchbares Ergebnis.

    Der Test ersetzt keine inhaltliche Pruefung: ein wirklich nutzloser
    Score kann auch bei korrekter Richtung um 0.5 streuen. Er faengt aber
    den Fall ab, der sonst still durch die gesamte Auswertung laeuft.
    """
    warnings: list[str] = []
    if not all_aucs:
        return warnings

    labels = [lbl for _c, lbl, _h, en in SCORES_ACTIVE if en]
    for label in labels:
        vals = [a[label] for a in all_aucs.values() if label in a]
        if len(vals) < 3:
            continue
        med = float(np.median(vals))
        if med < 0.5:
            n_below = sum(1 for v in vals if v < 0.5)
            warnings.append(
                f"{label}: Median-AUC {med:.3f} ueber {len(vals)} Targets "
                f"({n_below} davon unter 0.5). Gespiegelt waere sie "
                f"{1 - med:.3f} — higher_is_better im Katalog pruefen."
            )
    return warnings


def load_poses_csv(csv_path: Path, target_name: str) -> pd.DataFrame | None:
    """Load poses CSV, extract active/decoy labels, return pose 1 only."""
    if not csv_path.exists():
        print(f"  WARNING: CSV not found – {target_name}: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if "active" not in df.columns:
        if "ligand" in df.columns:
            df["active"] = df["ligand"].apply(
                lambda x: 1 if str(x).endswith("_a") else
                          (0 if str(x).endswith("_d") else -1))
            df = df[df["active"] >= 0].copy()
        else:
            print(f"  WARNING: Cannot determine active/decoy – {target_name}")
            return None

    if "pose" in df.columns:
        df = df[df["pose"] == 1].copy()
        print(f"  {target_name}: Pose 1 selected ({len(df)} ligands)")
    else:
        print(f"  {target_name}: No 'pose' column, using all rows")

    n_a = (df["active"] == 1).sum()
    n_d = (df["active"] == 0).sum()
    print(f"  {target_name}: {n_a} Actives, {n_d} Decoys (Ratio 1:{n_d // max(1, n_a)})")
    return df


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _signed_scores(df: pd.DataFrame, score_col: str,
                   higher_is_better: bool) -> np.ndarray:
    """Return scores so that higher is always better (negate if needed)."""
    return df[score_col].values if higher_is_better else -df[score_col].values


def compute_roc(df: pd.DataFrame, score_col: str,
                higher_is_better: bool):
    sub = df[["active", score_col]].dropna()
    if len(sub) == 0:
        return None, None, None

    scores = _signed_scores(sub, score_col, higher_is_better)
    labels = sub["active"].values
    order  = np.argsort(-scores)
    labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None

    tpr = np.concatenate([[0], np.cumsum(labels == 1) / n_pos])
    fpr = np.concatenate([[0], np.cumsum(labels == 0) / n_neg])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def compute_ef(df: pd.DataFrame, score_col: str,
               higher_is_better: bool, fraction: float) -> float | None:
    sub = df[["active", score_col]].dropna()
    if len(sub) == 0:
        return None

    scores   = _signed_scores(sub, score_col, higher_is_better)
    labels   = sub["active"].values
    n_total  = len(labels)
    n_active = labels.sum()
    n_select = max(1, int(np.ceil(fraction * n_total)))

    order    = np.argsort(-scores)
    hits     = labels[order][:n_select].sum()
    expected = fraction * n_active
    return float(hits / expected) if expected > 0 else None


def compute_pr_curve(df: pd.DataFrame, score_col: str,
                     higher_is_better: bool):
    """
    Precision-Recall curve and AUPRC via sklearn.

    sklearn.metrics.precision_recall_curve handles ties and edge cases
    correctly. average_precision_score computes AUPRC as the weighted
    mean of precisions (standard in the literature, more robust than
    trapezoid integration).

    Returns: (recall, precision, auprc)  – all np.ndarray / float
    """
    sub = df[["active", score_col]].dropna()
    if len(sub) == 0:
        return None, None, None

    scores = _signed_scores(sub, score_col, higher_is_better)
    labels = sub["active"].values

    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return None, None, None

    precision, recall, _ = precision_recall_curve(labels, scores)
    recall    = recall[::-1]
    precision = precision[::-1]

    auprc = float(average_precision_score(labels, scores))
    return recall, precision, auprc


def compute_cohens_d(active_scores: np.ndarray,
                     decoy_scores: np.ndarray) -> float | None:
    if len(active_scores) < 2 or len(decoy_scores) < 2:
        return None
    mean_a, mean_d = active_scores.mean(), decoy_scores.mean()
    n_a, n_d       = len(active_scores), len(decoy_scores)
    var_a, var_d   = active_scores.var(ddof=1), decoy_scores.var(ddof=1)
    pooled_std     = np.sqrt(((n_a - 1) * var_a + (n_d - 1) * var_d) /
                             (n_a + n_d - 2))
    return float((mean_a - mean_d) / pooled_std) if pooled_std != 0 else None


def compute_ks_test(active_scores: np.ndarray, decoy_scores: np.ndarray):
    if len(active_scores) < 2 or len(decoy_scores) < 2:
        return None, None
    stat, pval = sp_stats.ks_2samp(active_scores, decoy_scores)
    return float(stat), float(pval)


def _stratified_boot_sample(df: pd.DataFrame,
                             actives_idx: np.ndarray,
                             decoys_idx: np.ndarray,
                             rng: np.random.Generator) -> pd.DataFrame:
    a = rng.choice(actives_idx, size=len(actives_idx), replace=True)
    d = rng.choice(decoys_idx,  size=len(decoys_idx),  replace=True)
    return df.loc[np.concatenate([a, d])]


def bootstrap_roc_curves(df: pd.DataFrame, score_col: str,
                         higher_is_better: bool,
                         n_boot: int = N_BOOTSTRAP,
                         seed: int = RANDOM_SEED) -> list:
    rng         = np.random.default_rng(seed)
    actives_idx = df.index[df["active"] == 1].values
    decoys_idx  = df.index[df["active"] == 0].values
    results     = []
    for _ in range(n_boot):
        boot = _stratified_boot_sample(df, actives_idx, decoys_idx, rng)
        fpr, tpr, auc = compute_roc(boot, score_col, higher_is_better)
        if fpr is not None:
            results.append((fpr, tpr, auc))
    return results


def bootstrap_pr_curves(df: pd.DataFrame, score_col: str,
                        higher_is_better: bool,
                        n_boot: int = N_BOOTSTRAP,
                        seed: int = RANDOM_SEED) -> list:
    rng         = np.random.default_rng(seed)
    actives_idx = df.index[df["active"] == 1].values
    decoys_idx  = df.index[df["active"] == 0].values
    results     = []
    for _ in range(n_boot):
        boot = _stratified_boot_sample(df, actives_idx, decoys_idx, rng)
        rec, prec, auprc = compute_pr_curve(boot, score_col, higher_is_better)
        if rec is not None:
            results.append((rec, prec, auprc))
    return results


def interpolate_curve_band(curves_xy, n_points: int = 200):
    """
    Interpolate bootstrap curves onto a common x-grid.

    Guarantees monotone x arrays via np.unique + np.interp.
    Returns: x_grid, y_median, y_lower, y_upper
    """
    alpha  = 1 - CI_LEVEL
    x_grid = np.linspace(0, 1, n_points)
    y_interp = []
    for x_arr, y_arr in curves_xy:
        x_u, idx = np.unique(x_arr, return_index=True)
        y_u = y_arr[idx]
        y_interp.append(np.interp(x_grid, x_u, y_u))

    y_matrix = np.array(y_interp)
    y_lower  = np.percentile(y_matrix, 100 * alpha / 2,       axis=0)
    y_upper  = np.percentile(y_matrix, 100 * (1 - alpha / 2), axis=0)
    y_median = np.median(y_matrix, axis=0)
    return x_grid, y_median, y_lower, y_upper


def compute_bootstrap_cis(df: pd.DataFrame) -> dict:
    """Bootstrap CIs for AUC, AUPRC, and all EF fractions."""
    rng         = np.random.default_rng(RANDOM_SEED)
    actives_idx = df.index[df["active"] == 1].values
    decoys_idx  = df.index[df["active"] == 0].values
    alpha       = 1 - CI_LEVEL

    results = {}
    for col, label, hib, _en in SCORES_ACTIVE:
        if col not in df.columns:
            continue

        boot_aucs, boot_auprcs = [], []
        boot_efs = {f"EF{int(f*100)}%": [] for f in EF_FRACTIONS}

        for _ in range(N_BOOTSTRAP):
            boot = _stratified_boot_sample(df, actives_idx, decoys_idx, rng)

            _, _, auc = compute_roc(boot, col, hib)
            if auc is not None:
                boot_aucs.append(auc)

            _, _, auprc = compute_pr_curve(boot, col, hib)
            if auprc is not None:
                boot_auprcs.append(auprc)

            for fraction in EF_FRACTIONS:
                ef = compute_ef(boot, col, hib, fraction)
                if ef is not None:
                    boot_efs[f"EF{int(fraction*100)}%"].append(ef)

        score_cis = {}
        for name, vals in [("auc_ci", boot_aucs), ("auprc_ci", boot_auprcs)]:
            if len(vals) > 10:
                score_cis[name] = (
                    float(np.percentile(vals, 100 * alpha / 2)),
                    float(np.percentile(vals, 100 * (1 - alpha / 2))),
                )
            else:
                score_cis[name] = None

        for ef_key, vals in boot_efs.items():
            ci_key = f"{ef_key}_ci"
            if len(vals) > 10:
                score_cis[ci_key] = (
                    float(np.percentile(vals, 100 * alpha / 2)),
                    float(np.percentile(vals, 100 * (1 - alpha / 2))),
                )
            else:
                score_cis[ci_key] = None

        results[label] = score_cis
    return results


def plot_roc_curves(df: pd.DataFrame, out_dir: Path,
                    target_name: str) -> dict:
    ci_pct = int(CI_LEVEL * 100)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.50)")

        aucs = {}
        for col, label, hib, _en in SCORES_ACTIVE:
            if col not in df.columns:
                continue
            fpr, tpr, auc = compute_roc(df, col, hib)
            if fpr is None:
                continue

            boot = bootstrap_roc_curves(df, col, hib)
            if len(boot) > 10:
                boot_aucs = [b[2] for b in boot]
                ci_lo = np.percentile(boot_aucs, (100 - ci_pct) / 2)
                ci_hi = np.percentile(boot_aucs, 100 - (100 - ci_pct) / 2)
                x_grid, _, y_lo, y_hi = interpolate_curve_band(
                    [(b[0], b[1]) for b in boot])
                ax.fill_between(x_grid, y_lo, y_hi,
                                color=SCORE_COLORS[label], alpha=0.12)
                ci_str = f" [{ci_lo:.3f}–{ci_hi:.3f}]"
            else:
                ci_str = ""

            ax.plot(fpr, tpr, lw=2.5, color=SCORE_COLORS[label],
                    label=f"{label}  (AUC={auc:.3f}{ci_str})")
            aucs[label] = auc

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curves – {target_name} (Pose 1)",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        _save(fig, out_dir / "roc_curves.png")

    return aucs


def plot_enrichment_factors(df: pd.DataFrame, out_dir: Path,
                            target_name: str) -> dict:
    valid     = [(col, label, hib, en)
                 for col, label, hib, en in SCORES_ACTIVE if col in df.columns]
    x         = np.arange(len(valid))
    width     = 0.22
    bar_colors = ["#2C3E50", "#7F8C8D", "#BDC3C7"]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        ef_data = {}

        for fi, (fraction, color) in enumerate(zip(EF_FRACTIONS, bar_colors)):
            vals = []
            for col, label, hib, _en in valid:
                ef = compute_ef(df, col, hib, fraction)
                vals.append(ef if ef is not None else 0.0)
                ef_data.setdefault(label, {})[f"EF{int(fraction*100)}%"] = ef

            bars = ax.bar(x + fi * width, vals, width,
                          label=f"EF{int(fraction*100)}%",
                          color=color, alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0.1:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f"{val:.1f}×",
                            ha="center", va="bottom",
                            fontsize=8.5, fontweight="bold")

        ax.axhline(y=1.0, color="#E74C3C", linestyle="--", lw=1.5,
                   alpha=0.8, label="Random (EF=1.0×)")
        ax.set_xticks(x + width)
        ax.set_xticklabels([label for _, label, _, _en in valid], fontsize=11)
        ax.set_ylabel("Enrichment Factor", fontsize=12)
        ax.set_title(f"Enrichment Factors – {target_name} (Pose 1)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        _save(fig, out_dir / "enrichment_factors.png")

    return ef_data


def plot_score_distributions(df: pd.DataFrame, out_dir: Path,
                             target_name: str) -> None:
    valid   = [(col, label, hib, en)
               for col, label, hib, en in SCORES_ACTIVE if col in df.columns]
    actives = df[df["active"] == 1]
    decoys  = df[df["active"] == 0]

    n_plots = len(valid)
    ncols   = min(2, n_plots)
    nrows   = (n_plots + 1) // 2

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
        axes = [axes] if n_plots == 1 else axes.flatten()

        for i, (col, label, _, _en) in enumerate(valid):
            ax      = axes[i]
            color   = SCORE_COLORS[label]
            a_vals  = actives[col].dropna()
            d_vals  = decoys[col].dropna()

            if len(a_vals) == 0 and len(d_vals) == 0:
                ax.set_visible(False)
                continue

            all_vals = pd.concat([a_vals, d_vals])
            lo, hi   = all_vals.quantile(0.01), all_vals.quantile(0.99)
            bins     = np.linspace(lo, hi, 45)

            ax.hist(d_vals, bins=bins, alpha=0.45, color="#95A5A6",
                    label=f"Decoys (n={len(d_vals)})",
                    density=True, edgecolor="none")
            ax.hist(a_vals, bins=bins, alpha=0.80, color=color,
                    label=f"Actives (n={len(a_vals)})",
                    density=True, edgecolor="none")

            if len(a_vals) > 0:
                ax.axvline(a_vals.median(), color=color, lw=2,
                           label=f"Median Active: {a_vals.median():.2f}")
            if len(d_vals) > 0:
                ax.axvline(d_vals.median(), color="#7F8C8D", lw=2,
                           linestyle="--",
                           label=f"Median Decoy: {d_vals.median():.2f}")

            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)

        for j in range(len(valid), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Score Distributions – {target_name} (Pose 1)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save(fig, out_dir / "score_distributions.png")


def plot_pr_curves(df: pd.DataFrame, out_dir: Path,
                   target_name: str) -> dict:
    """
    Precision-Recall curves using sklearn.

    - precision_recall_curve for the curve shape
    - average_precision_score for AUPRC (weighted mean of precision
      increments, not trapezoid — standard in virtual screening lit.)
    - Bootstrap CI bands with guaranteed monotone recall interpolation
    """
    ci_pct   = int(CI_LEVEL * 100)
    baseline = (df["active"] == 1).sum() / len(df)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.axhline(y=baseline, color="black", linestyle="--", lw=1, alpha=0.4,
                   label=f"Random (AUPRC≈{baseline:.3f})")

        auprcs = {}
        for col, label, hib, _en in SCORES_ACTIVE:
            if col not in df.columns:
                continue
            recall, precision, auprc = compute_pr_curve(df, col, hib)
            if recall is None:
                continue

            boot = bootstrap_pr_curves(df, col, hib)
            if len(boot) > 10:
                boot_auprcs = [b[2] for b in boot]
                ci_lo = np.percentile(boot_auprcs, (100 - ci_pct) / 2)
                ci_hi = np.percentile(boot_auprcs, 100 - (100 - ci_pct) / 2)
                x_grid, _, y_lo, y_hi = interpolate_curve_band(
                    [(b[0], b[1]) for b in boot])
                ax.fill_between(x_grid, y_lo, y_hi,
                                color=SCORE_COLORS[label], alpha=0.12)
                ci_str = f" [{ci_lo:.3f}–{ci_hi:.3f}]"
            else:
                ci_str = ""

            ax.plot(recall, precision, lw=2.5, color=SCORE_COLORS[label],
                    label=f"{label}  (AUPRC={auprc:.3f}{ci_str})")
            auprcs[label] = auprc

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curves – {target_name} (Pose 1)",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        _save(fig, out_dir / "pr_curves.png")

    return auprcs


def plot_score_separation(df: pd.DataFrame, out_dir: Path,
                          target_name: str) -> dict:
    valid   = [(col, label, hib, en)
               for col, label, hib, en in SCORES_ACTIVE if col in df.columns]
    actives = df[df["active"] == 1]
    decoys  = df[df["active"] == 0]

    sep_data = {}
    labels_list, cohens_d_vals, ks_stat_vals, ks_pval_vals = [], [], [], []

    for col, label, hib, _en in valid:
        a_vals = actives[col].dropna().values
        d_vals = decoys[col].dropna().values

        cd              = compute_cohens_d(a_vals, d_vals)
        ks_stat, ks_pval = compute_ks_test(a_vals, d_vals)

        if not hib and cd is not None:
            cd = -cd

        sep_data[label] = {"cohens_d": cd, "ks_stat": ks_stat, "ks_pval": ks_pval}
        labels_list.append(label)
        cohens_d_vals.append(cd if cd is not None else 0)
        ks_stat_vals.append(ks_stat if ks_stat is not None else 0)
        ks_pval_vals.append(ks_pval if ks_pval is not None else 1)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        x      = np.arange(len(labels_list))
        colors = [SCORE_COLORS[l] for l in labels_list]

        ax = axes[0]
        bars = ax.bar(x, cohens_d_vals, color=colors,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        max_abs = max((abs(v) for v in cohens_d_vals), default=1)
        for bar, val in zip(bars, cohens_d_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * max_abs,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        for threshold, desc in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
            ax.axhline(y=threshold, color="gray", linestyle=":", lw=0.8, alpha=0.5)
            ax.text(len(labels_list) - 0.5, threshold + 0.02, desc,
                    fontsize=7, color="gray", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, fontsize=10)
        ax.set_ylabel("Cohen's d  (positive = Actives better)", fontsize=11)
        ax.set_title("Cohen's d – Effect Size", fontweight="bold")

        ax = axes[1]
        bars = ax.bar(x, ks_stat_vals, color=colors,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val, pval in zip(bars, ks_stat_vals, ks_pval_vals):
            label_str = (f"D={val:.3f}\np={pval:.1e}"
                         if pval >= 1e-300 else f"D={val:.3f}\np≈0")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    label_str, ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, fontsize=10)
        ax.set_ylabel("KS-Statistik D", fontsize=11)
        ax.set_title("Kolmogorov-Smirnov Test", fontweight="bold")

        fig.suptitle(f"Score Separation – {target_name} (Pose 1)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save(fig, out_dir / "score_separation.png")

    return sep_data


def plot_comparison_auc(all_aucs: dict, out_dir: Path) -> None:
    targets      = list(all_aucs.keys())
    score_labels = [label for _, label, _, _en in SCORES_ACTIVE]
    x     = np.arange(len(targets))
    width = 0.18

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 6))
        for si, label in enumerate(score_labels):
            vals = [all_aucs[t].get(label) for t in targets]
            bars = ax.bar(x + si * width,
                          [v if v is not None else 0 for v in vals],
                          width, label=label,
                          color=SCORE_COLORS[label],
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val is not None and val > 0.1:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}",
                            ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold")
        ax.axhline(y=0.5, color="black", linestyle="--", lw=1.2,
                   alpha=0.5, label="Random (AUC=0.50)")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(targets, fontsize=12)
        ax.set_ylabel("AUC", fontsize=12)
        ax.set_title("AUC Comparison – All Targets (Pose 1)",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        fig.tight_layout()
        _save(fig, out_dir / "comparison_auc.png")
    print("  Comparison plot AUC saved")


def plot_comparison_ef(all_efs: dict, out_dir: Path,
                       fraction: float = 0.01) -> None:
    targets      = list(all_efs.keys())
    score_labels = [label for _, label, _, _en in SCORES_ACTIVE]
    pct_label    = f"EF{int(fraction*100)}%"
    x     = np.arange(len(targets))
    width = 0.18

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 6))
        for si, label in enumerate(score_labels):
            vals = [all_efs[t].get(label, {}).get(pct_label) for t in targets]
            bars = ax.bar(x + si * width,
                          [v if v is not None else 0 for v in vals],
                          width, label=label,
                          color=SCORE_COLORS[label],
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val is not None and val > 0.1:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f"{val:.1f}×",
                            ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold")
        ax.axhline(y=1.0, color="#E74C3C", linestyle="--", lw=1.5,
                   alpha=0.8, label="Random (EF=1.0×)")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(targets, fontsize=12)
        ax.set_ylabel(f"Enrichment Factor ({pct_label})", fontsize=12)
        ax.set_title(f"{pct_label} Comparison – All Targets (Pose 1)",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        _save(fig, out_dir / f"comparison_{pct_label.lower()}.png")
    print(f"  Comparison plot {pct_label} saved")


def _collect_spearman_matrix(all_corrs: dict) -> dict:
    """
    Collect per-target Spearman ρ for every score pair (excl. ECR).

    Returns a dict:  pair_key -> list of (target, rho)  with rho=None if missing.
    Pair keys are (label_x, label_y) strings for readability.
    """
    data: dict[str, list] = {}

    for target, corr_dict in all_corrs.items():
        for pair_key, entry in corr_dict.items():
            lbl = entry["label"]
            if "ECR" in lbl:
                continue
            if lbl not in data:
                data[lbl] = []
            rho = entry.get("spearman_r")
            data[lbl].append((target, rho if (rho is not None and
                                               not np.isnan(rho)) else None))
    return data


def _score_labels_no_ecr() -> list[str]:
    """Return display labels of enabled scores, ECR excluded."""
    return [label for _, label, _, _ in SCORES_ACTIVE
            if label != "ECR Score"]


def _pair_to_scores(pair_label: str) -> tuple[str, str]:
    """Split 'A vs B' into ('A', 'B')."""
    parts = pair_label.split(" vs ")
    return parts[0].strip(), parts[1].strip()


def plot_orthogonality_heatmap(all_corrs: dict, out_dir: Path) -> None:
    """
    Heatmap of median Spearman ρ over all targets (ECR excluded).

    Upper triangle: all ligands.
    The colour map runs red (−1) → white (0) → blue (+1).
    """
    sp_data  = _collect_spearman_matrix(all_corrs)
    sc_lbls  = _score_labels_no_ecr()
    n        = len(sc_lbls)
    idx      = {s: i for i, s in enumerate(sc_lbls)}

    med_mat = np.full((n, n), np.nan)
    np.fill_diagonal(med_mat, 1.0)

    for pair_lbl, vals in sp_data.items():
        rhos = [v for _, v in vals if v is not None]
        if not rhos:
            continue
        med = float(np.median(rhos))
        a, b = _pair_to_scores(pair_lbl)
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            med_mat[i, j] = med
            med_mat[j, i] = med

    cmap = plt.cm.RdBu_r
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(med_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Median Spearman ρ", fraction=0.046, pad=0.04)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(sc_lbls, rotation=35, ha="right", fontsize=10)
        ax.set_yticklabels(sc_lbls, fontsize=10)

        for i in range(n):
            for j in range(n):
                v = med_mat[i, j]
                if np.isnan(v):
                    txt, col = "N/A", "grey"
                elif i == j:
                    txt, col = "1.00", "black"
                else:
                    txt = f"{v:+.3f}"
                    col = "white" if abs(v) > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9.5, color=col, fontweight="bold")

        ax.set_title(
            f"Score Orthogonality – Median Spearman ρ over {len(all_corrs)} Targets\n"
            "(All ligands, ECR excluded)",
            fontsize=12, fontweight="bold", pad=10)
        fig.tight_layout()
        _save(fig, out_dir / "orthogonality_heatmap.png")
    print("  Orthogonality heatmap saved")


def plot_orthogonality_strip(all_corrs: dict, out_dir: Path) -> None:
    """
    Strip chart: one column per score pair, each dot = one target.
    Median shown as a horizontal line (─) marker.
    Highlights spread / outlier targets.
    """
    sp_data = _collect_spearman_matrix(all_corrs)
    if not sp_data:
        return

    pair_lbls = list(sp_data.keys())
    n_pairs   = len(pair_lbls)
    rng_jit   = np.random.default_rng(42)

    with plt.style.context(STYLE):
        fig_w = max(9, n_pairs * 1.6)
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        for pi, pair_lbl in enumerate(pair_lbls):
            vals  = [v for _, v in sp_data[pair_lbl] if v is not None]
            if not vals:
                continue
            jitter  = rng_jit.uniform(-0.18, 0.18, size=len(vals))
            xs      = pi + jitter
            color   = PAIR_COLORS[pi % len(PAIR_COLORS)]
            ax.scatter(xs, vals, color=color, alpha=0.55, s=28,
                       linewidths=0, zorder=3)

            med = float(np.median(vals))
            ax.plot([pi - 0.32, pi + 0.32], [med, med],
                    color=color, lw=2.8, solid_capstyle="round",
                    zorder=4, label=f"{pair_lbl} (med={med:+.3f})")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels(pair_lbls, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f"Score Orthogonality – Spearman ρ per Target  "
            f"(n={len(all_corrs)} targets, ECR excluded)\n"
            "Each dot = one target · horizontal bar = median",
            fontsize=11, fontweight="bold")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=7.5, ncol=2,
                  loc="upper right", framealpha=0.92)
        fig.tight_layout()
        _save(fig, out_dir / "orthogonality_strip.png")
    print("  Orthogonality strip chart saved")


def plot_orthogonality_table(all_corrs: dict, out_dir: Path) -> None:
    """
    Raw-data heatmap table: rows = targets, columns = score pairs (ECR excl.).
    Cells coloured by Spearman ρ; bottom row shows the median.
    """
    sp_data   = _collect_spearman_matrix(all_corrs)
    if not sp_data:
        return

    pair_lbls  = list(sp_data.keys())
    all_targets = list(all_corrs.keys())
    n_t = len(all_targets)
    n_p = len(pair_lbls)

    mat   = np.full((n_t + 1, n_p), np.nan)
    t_idx = {t: i for i, t in enumerate(all_targets)}

    for pi, pair_lbl in enumerate(pair_lbls):
        vals_dict = {t: v for t, v in sp_data[pair_lbl]}
        for t, i in t_idx.items():
            v = vals_dict.get(t)
            mat[i, pi] = v if v is not None else np.nan
        valid = [v for v in vals_dict.values() if v is not None]
        mat[n_t, pi] = float(np.median(valid)) if valid else np.nan

    row_labels = all_targets + ["MEDIAN"]
    cmap = plt.cm.RdBu_r

    fig_h = max(8, (n_t + 2) * 0.38)
    fig_w = max(10, n_p * 1.9)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ",
                     fraction=0.025, pad=0.02)

        ax.set_xticks(range(n_p))
        ax.set_xticklabels(pair_lbls, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(n_t + 1))
        ax.set_yticklabels(row_labels, fontsize=8)

        ax.axhline(n_t - 0.5, color="black", lw=1.8)

        for i in range(n_t + 1):
            for j in range(n_p):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=6.5, color="grey")
                else:
                    col = "white" if abs(v) > 0.55 else "black"
                    fw  = "bold" if i == n_t else "normal"
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                            fontsize=7, color=col, fontweight=fw)

        ax.set_title(
            "Score Orthogonality – Raw Spearman ρ per Target  (ECR excluded)\n"
            "Bottom row = median over all targets",
            fontsize=11, fontweight="bold", pad=10)
        fig.tight_layout()
        _save(fig, out_dir / "orthogonality_table.png")
    print("  Orthogonality raw-data table saved")


def safe_corr(x: np.ndarray, y: np.ndarray,
              method: str = "pearson") -> tuple[float, float]:
    """
    Compute correlation coefficient and p-value with error handling.

    Handles scipy ≥1.7 where kendalltau returns a named tuple with
    .statistic and .pvalue instead of bare (stat, pval).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    xm, ym = x[mask], y[mask]
    if np.std(xm) == 0 or np.std(ym) == 0:
        return np.nan, np.nan
    try:
        if method == "pearson":
            res = sp_stats.pearsonr(xm, ym)
            return float(res.statistic), float(res.pvalue)
        elif method == "spearman":
            res = sp_stats.spearmanr(xm, ym)
            return float(res.statistic), float(res.pvalue)
        elif method == "kendall":
            res = sp_stats.kendalltau(xm, ym)
            return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan
    return np.nan, np.nan


def _corr_pval_fmt(p: float) -> str:
    if np.isnan(p): return "n/a"
    if p < 0.001:   return "p < 0.001"
    if p < 0.01:    return "p < 0.01"
    if p < 0.05:    return "p < 0.05"
    return f"p = {p:.3f}"


def compute_all_correlations(df: pd.DataFrame) -> dict:
    """
    Pearson, Spearman, Kendall for all enabled score pairs.

    FIX: correct 4-tuple unpacking of SCORE_PAIRS entries.
    """
    results = {}
    actives = df[df["active"] == 1] if "active" in df.columns else pd.DataFrame()

    for (col_x, label_x, _hib_x, _en_x), (col_y, label_y, _hib_y, _en_y) in SCORE_PAIRS:
        if col_x not in df.columns or col_y not in df.columns:
            continue
        if df[col_x].isna().all() or df[col_y].isna().all():
            continue

        pair_key   = (col_x, col_y)
        pair_label = f"{label_x} vs {label_y}"
        entry      = {"label": pair_label}

        x_all = df[col_x].values
        y_all = df[col_y].values

        for method in ("pearson", "spearman", "kendall"):
            r, p = safe_corr(x_all, y_all, method)
            entry[f"{method}_r"] = r
            entry[f"{method}_p"] = p

        if len(actives) >= 3 and col_x in actives.columns and col_y in actives.columns:
            x_act = actives[col_x].values
            y_act = actives[col_y].values
            n_valid = int((np.isfinite(x_act) & np.isfinite(y_act)).sum())
            entry["actives_n"] = n_valid
            for method in ("pearson", "spearman", "kendall"):
                r, p = safe_corr(x_act, y_act, method)
                entry[f"actives_{method}_r"] = r
                entry[f"actives_{method}_p"] = p
        else:
            entry["actives_n"] = 0

        results[pair_key] = entry

    return results


def plot_correlation_scatter(df: pd.DataFrame, out_dir: Path,
                             target_name: str) -> dict:
    """
    Scatter plots for all score pairs + Pearson/Spearman/Kendall.

    FIX: correct 4-tuple unpacking of SCORE_PAIRS entries.
    """
    corr_data = compute_all_correlations(df)

    act        = df["active"].values if "active" in df.columns else np.full(len(df), np.nan)
    has_labels = np.any(act == 1) or np.any(act == 0)

    for idx, ((col_x, label_x, _hib_x, _en_x), (col_y, label_y, _hib_y, _en_y)) \
            in enumerate(SCORE_PAIRS):
        if col_x not in df.columns or col_y not in df.columns:
            continue
        if df[col_x].isna().all() or df[col_y].isna().all():
            continue

        pair_key = (col_x, col_y)
        color    = PAIR_COLORS[idx % len(PAIR_COLORS)]

        x    = df[col_x].values
        y    = df[col_y].values
        mask = np.isfinite(x) & np.isfinite(y)
        xc, yc, ac = x[mask], y[mask], act[mask]

        if mask.sum() < 3:
            continue

        n_total = int(mask.sum())
        alpha   = max(0.10, min(0.50, 300 / max(n_total, 1)))

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.set_facecolor("#f8f9fa")

            md = ac != 1
            if md.sum() > 0:
                col_d = DECOY_COLOR if has_labels else "#5c6bc0"
                lbl   = (f"Decoys (n={md.sum():,})" if has_labels
                         else f"Ligands (n={md.sum():,})")
                ax.scatter(xc[md], yc[md], c=col_d, alpha=alpha, s=8,
                           linewidths=0, rasterized=True, zorder=2, label=lbl)

            ma = ac == 1
            if ma.sum() > 0:
                ax.scatter(xc[ma], yc[ma], c=ACTIVE_COLOR,
                           alpha=min(0.92, alpha * 4), s=18,
                           linewidths=0.3, edgecolors="white",
                           rasterized=True, zorder=3,
                           label=f"Actives (n={ma.sum()})")

            try:
                m, b, *_ = sp_stats.linregress(xc, yc)
                xr = np.array([np.nanmin(xc), np.nanmax(xc)])
                ax.plot(xr, m * xr + b, color=color,
                        lw=2.0, alpha=0.75, ls="--", zorder=4)
            except Exception:
                pass

            entry = corr_data.get(pair_key, {})
            r = entry.get("pearson_r", np.nan)
            p = entry.get("pearson_p", np.nan)
            if not np.isnan(r):
                sign = "+" if r >= 0 else ""
                ann  = f"r = {sign}{r:.3f}\n{_corr_pval_fmt(p)}\nn = {n_total:,}"
                ax.text(0.97, 0.97, ann,
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=10, color="#212529",
                        bbox=dict(boxstyle="round,pad=0.5",
                                  facecolor="white", edgecolor="#dee2e6",
                                  alpha=0.95, linewidth=0.8),
                        zorder=5)

            ax.set_xlabel(label_x, fontsize=11, labelpad=6)
            ax.set_ylabel(label_y, fontsize=11, labelpad=6)
            ax.set_title(f"{target_name} – {label_x} vs {label_y}",
                         fontsize=13, fontweight="bold", pad=10)
            ax.tick_params(labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#dee2e6")
                spine.set_linewidth(0.8)

            if has_labels:
                ax.legend(fontsize=9, framealpha=0.95,
                          facecolor="white", edgecolor="#dee2e6",
                          loc="upper left", handletextpad=0.4,
                          borderpad=0.6)

            safe_name = f"{col_x}_{col_y}".replace("score_", "")
            fig.tight_layout()
            _save(fig, out_dir / f"correlation_{safe_name}.png")

    print(f"  Correlation scatter plots saved ({len(corr_data)} pairs)")
    return corr_data


def _fmt_val_ci(val, ci, fmt=".4f") -> str:
    if val is None:
        return "N/A"
    s = f"{val:{fmt}}"
    if ci is not None:
        s += f" [{ci[0]:{fmt}}–{ci[1]:{fmt}}]"
    return s


def write_summary(all_aucs: dict, all_efs: dict, all_auprcs: dict,
                  all_seps: dict, all_dfs: dict, all_cis: dict,
                  all_corrs: dict, out_dir: Path) -> None:
    ci_pct = int(CI_LEVEL * 100)
    col_w  = 28

    lines = [
        "=" * 130,
        "  DUD-E Validation – Full Report",
        f"  Strategy: Pose 1 (Vina-best) + Re-Ranking | "
        f"Bootstrap: {N_BOOTSTRAP} iter, {ci_pct}% CI | "
        f"AUPRC: sklearn average_precision_score",
        "=" * 130, "",
    ]

    lines += ["─" * 130, "  DATASET STATISTICS", "─" * 130,
              f"  {'Target':<15} {'Ligands':>10} {'Actives':>10} "
              f"{'Decoys':>10} {'Ratio':>12} {'Active %':>10}",
              "  " + "-" * 75]
    for target, df in all_dfs.items():
        n_total  = len(df)
        n_active = (df["active"] == 1).sum()
        n_decoy  = (df["active"] == 0).sum()
        ratio    = f"1:{n_decoy // max(1, n_active)}"
        pct      = n_active / n_total * 100
        lines.append(f"  {target:<15} {n_total:>10} {n_active:>10} "
                     f"{n_decoy:>10} {ratio:>12} {pct:>9.2f}%")
    lines.append("")

    lines += ["─" * 130,
              f"  AUC-ROC VALUES  (with {ci_pct}% bootstrap CI)",
              "─" * 130,
              f"  {'Target':<15}" +
              "".join(f"  {label:<{col_w}}" for _, label, _, _en in SCORES_ACTIVE),
              "  " + "-" * 120]
    for target in all_aucs:
        row = f"  {target:<15}"
        for _, label, _, _en in SCORES_ACTIVE:
            val = all_aucs[target].get(label)
            ci  = all_cis.get(target, {}).get(label, {}).get("auc_ci")
            row += f"  {_fmt_val_ci(val, ci):<{col_w}}"
        lines.append(row)
    lines.append("")

    lines += ["─" * 130,
              f"  AUPRC VALUES  (sklearn average_precision_score, {ci_pct}% CI)",
              "─" * 130,
              f"  {'Target':<15}" +
              "".join(f"  {label:<{col_w}}" for _, label, _, _en in SCORES_ACTIVE),
              "  " + "-" * 120]
    for target in all_auprcs:
        row = f"  {target:<15}"
        for _, label, _, _en in SCORES_ACTIVE:
            val = all_auprcs[target].get(label)
            ci  = all_cis.get(target, {}).get(label, {}).get("auprc_ci")
            row += f"  {_fmt_val_ci(val, ci):<{col_w}}"
        lines.append(row)
    lines.append("")

    for fraction in EF_FRACTIONS:
        ef_label = f"EF{int(fraction*100)}%"
        ci_key   = f"{ef_label}_ci"
        lines += ["─" * 130,
                  f"  {ef_label} VALUES  (with {ci_pct}% bootstrap CI)",
                  "─" * 130,
                  f"  {'Target':<15}" +
                  "".join(f"  {label:<{col_w}}" for _, label, _, _en in SCORES_ACTIVE),
                  "  " + "-" * 120]
        for target in all_efs:
            row = f"  {target:<15}"
            for _, label, _, _en in SCORES_ACTIVE:
                val = all_efs[target].get(label, {}).get(ef_label)
                ci  = all_cis.get(target, {}).get(label, {}).get(ci_key)
                row += f"  {_fmt_val_ci(val, ci, fmt='.2f'):<{col_w}}"
            lines.append(row)
        lines.append("")

    lines += ["─" * 130,
              "  COHEN'S D  (positive = actives score better)",
              "  |d|<0.2 negligible · 0.2–0.5 small · 0.5–0.8 medium · >0.8 large",
              "─" * 130,
              f"  {'Target':<15}" +
              "".join(f"  {label:<{col_w}}" for _, label, _, _en in SCORES_ACTIVE),
              "  " + "-" * 120]
    for target in all_seps:
        row = f"  {target:<15}"
        for _, label, _, _en in SCORES_ACTIVE:
            cd = all_seps[target].get(label, {}).get("cohens_d")
            row += (f"  {cd:<+{col_w}.4f}" if cd is not None
                    else f"  {'N/A':<{col_w}}")
        lines.append(row)
    lines.append("")

    lines += ["─" * 130,
              "  KOLMOGOROV-SMIRNOV TEST",
              "─" * 130,
              f"  {'Target':<15}" +
              "".join(f"  {label:<{col_w}}" for _, label, _, _en in SCORES_ACTIVE),
              "  " + "-" * 120]
    for target in all_seps:
        row_d = f"  {target:<15}"
        row_p = f"  {'  (p-value)':<15}"
        for _, label, _, _en in SCORES_ACTIVE:
            sep = all_seps[target].get(label, {})
            ks  = sep.get("ks_stat")
            kp  = sep.get("ks_pval")
            row_d += (f"  {'D=' + f'{ks:.4f}':<{col_w}}"
                      if ks is not None else f"  {'N/A':<{col_w}}")
            if kp is not None:
                p_str = f"p={kp:.2e}" if kp >= 1e-300 else "p≈0"
                row_p += f"  {p_str:<{col_w}}"
            else:
                row_p += f"  {'N/A':<{col_w}}"
        lines += [row_d, row_p]
    lines.append("")

    for target, df in all_dfs.items():
        actives = df[df["active"] == 1]
        decoys  = df[df["active"] == 0]
        lines += ["─" * 130,
                  f"  SCORE STATISTICS – {target}",
                  "─" * 130,
                  f"  {'Score':<16} {'Class':<10} {'N':>6} {'Mean':>10} "
                  f"{'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10}",
                  "  " + "-" * 82]
        for col, label, _, _en in SCORES_ACTIVE:
            if col not in df.columns:
                continue
            for cls_name, subset in [("Actives", actives), ("Decoys", decoys)]:
                vals = subset[col].dropna()
                if len(vals) == 0:
                    continue
                lines.append(
                    f"  {label:<16} {cls_name:<10} {len(vals):>6} "
                    f"{vals.mean():>10.3f} {vals.std():>10.3f} "
                    f"{vals.median():>10.3f} {vals.min():>10.3f} {vals.max():>10.3f}"
                )
        lines.append("")

    corr_col_w  = 20
    corr_header = (f"  {'Target':<15}  {'Score Pair':<30}  │ "
                   f"{'Pearson r':>{corr_col_w}} │ "
                   f"{'Spearman ρ':>{corr_col_w}} │ "
                   f"{'Kendall τ':>{corr_col_w}}")

    def _fmt_corr(r, p):
        if np.isnan(r):
            return "n/a".center(corr_col_w)
        sign = "+" if r >= 0 else ""
        sig  = ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "")
        return f"{sign}{r:.4f} {sig}".rjust(corr_col_w)

    for section_title, r_prefix in [
        ("ALL LIGANDS", ""),
        ("ACTIVES ONLY\n  → Do scoring functions agree on RANKING of actives?\n"
         "  → Kendall τ most robust for small sample sizes", "actives_"),
    ]:
        lines += ["─" * 130,
                  f"  SCORE CORRELATIONS – {section_title}",
                  "─" * 130,
                  corr_header,
                  "  " + "-" * 120]
        for target in all_corrs:
            entry_dict = all_corrs[target]
            first = True
            for pair_key, entry in entry_dict.items():
                t_label    = target if first else ""
                first      = False
                pair_label = entry["label"]
                n_act      = entry.get("actives_n", 0)

                if r_prefix == "actives_" and n_act < 3:
                    lines.append(
                        f"  {t_label:<15}  {pair_label:<30}  │ "
                        f"too few actives (n={n_act})")
                    continue

                rp = _fmt_corr(entry.get(f"{r_prefix}pearson_r",  np.nan),
                               entry.get(f"{r_prefix}pearson_p",  np.nan))
                rs = _fmt_corr(entry.get(f"{r_prefix}spearman_r", np.nan),
                               entry.get(f"{r_prefix}spearman_p", np.nan))
                rk = _fmt_corr(entry.get(f"{r_prefix}kendall_r",  np.nan),
                               entry.get(f"{r_prefix}kendall_p",  np.nan))
                n_str = f"   (n={n_act})" if r_prefix == "actives_" else ""
                lines.append(
                    f"  {t_label:<15}  {pair_label:<30}  │ {rp} │ {rs} │ {rk}{n_str}")
            lines.append("")
        lines += ["  Significance: *** p<0.001, ** p<0.01, * p<0.05", ""]

    lines.append("=" * 130)

    summary_path = out_dir / "summary.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary written: {summary_path}")
    for line in lines:
        print(f"  {line}")


def main() -> None:
    print("=" * 60)
    print(" DUD-E Validation Analysis")
    print(" Strategy: Pose 1 (Vina) + Score Re-Ranking")
    print(f" {len(TARGETS)} Targets | DPI={DPI} | sklearn PR curves")
    print("=" * 60)

    all_aucs   = {}
    all_efs    = {}
    all_auprcs = {}
    all_seps   = {}
    all_dfs    = {}
    all_cis    = {}
    all_corrs  = {}

    for target_name, csv_path in TARGETS:
        print(f"\n── {target_name} ──────────────────────────────────")
        df = load_poses_csv(csv_path, target_name)
        if df is None:
            continue

        df = recompute_ecr(df)

        all_dfs[target_name] = df
        out_dir = ANALYSIS_DIR / target_name

        print("  [1/7] ROC Curves...")
        all_aucs[target_name] = plot_roc_curves(df, out_dir, target_name)

        print("  [2/7] Enrichment Factors...")
        all_efs[target_name] = plot_enrichment_factors(df, out_dir, target_name)

        print("  [3/7] Score Distributions...")
        plot_score_distributions(df, out_dir, target_name)

        print("  [4/7] Precision-Recall Curves (sklearn)...")
        all_auprcs[target_name] = plot_pr_curves(df, out_dir, target_name)

        print("  [5/7] Score Separation (Cohen's d / KS)...")
        all_seps[target_name] = plot_score_separation(df, out_dir, target_name)

        print(f"  [6/7] Bootstrap CIs ({N_BOOTSTRAP} iterations)...")
        all_cis[target_name] = compute_bootstrap_cis(df)

        print("  [7/7] Score Correlations...")
        all_corrs[target_name] = plot_correlation_scatter(df, out_dir, target_name)

    if len(all_aucs) > 1:
        print("\n── Comparison Plots ──────────────────────────────")
        plot_comparison_auc(all_aucs, ANALYSIS_DIR)
        for frac in [0.01, 0.05]:
            plot_comparison_ef(all_efs, ANALYSIS_DIR, fraction=frac)

    if len(all_corrs) > 1:
        print("\n── Orthogonality Plots ───────────────────────────")
        plot_orthogonality_heatmap(all_corrs, ANALYSIS_DIR)
        plot_orthogonality_strip(all_corrs, ANALYSIS_DIR)
        plot_orthogonality_table(all_corrs, ANALYSIS_DIR)

    write_summary(all_aucs, all_efs, all_auprcs, all_seps,
                  all_dfs, all_cis, all_corrs, ANALYSIS_DIR)

    direction_warnings = check_score_directions(all_aucs)
    if direction_warnings:
        print(f"\n{'='*60}")
        print(" WARNUNG: moegliche Vorzeichenfehler im Score-Katalog")
        print(f"{'='*60}")
        for w in direction_warnings:
            print(f"  ! {w}")
        print("  Betrifft auch die ECR: sie rangiert nach derselben Flagge.")

    print(f"\n{'='*60}")
    print(f" Done! All files in: {ANALYSIS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
