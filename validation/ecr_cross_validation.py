"""
ecr_cross_validation.py
===================
Leave-One-Target-Out-Kreuzvalidierung fuer die ECR-Parameter
(sigma_fraction und Gewichte).

Verglichene Verfahren je Fold: Einzelscores (Pose 1 und Best-Pose),
ECR mit Gleichgewichten, ECR mit auf den Trainingstargets optimiertem
sigma, LOO-CV (Gewichte + sigma auf Trainingstargets), sowie Oracle und
Global als obere Schranken — beide sehen das Testtarget und sind keine
CV-Ergebnisse.

Parallelisierung: die Folds laufen sequenziell, weil jeder Fold selbst
einen ProcessPool fuer die Gittersuche aufspannt.

Experiment ueber die Umgebungsvariable EXPERIMENT_NAME waehlen.

Usage:
    python3 ecr_cross_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
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

SIGMA_FRACTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]
WEIGHT_STEP = 0.1
PRIMARY_METRIC = "EF1%"

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

OUTPUT_DIR = Path(f"/home/roessel/gpu8.0/RESULTS/cross_validation/{EXPERIMENT_NAME}")

# Dritte Spalte = higher_is_better; muss identisch zu
# enrichment_analysis.py bleiben (run_validation.sh prueft das beim Start).
SCORES = [
    ("score_vina",              "Vina",          False, True),
    ("score_vinardo",           "Vinardo",       False, True),
    ("score_cnnaffinity",       "CNNAffinity",   True,  True),
    ("score_cnnscore",          "CNNscore",      True,  True),
    ("score_deltalinf9xgb",     "ΔLinF9XGB",     False, True),
    ("score_dense_cnnaffinity", "DenseAffinity", True,  False),
    ("score_dense_cnnscore",    "DenseCNNscore", True,  False),
]

# Modulkonstante, damit die Worker-Prozesse sie beim Import mitbekommen
# und die Signatur der Grid-Tasks unveraendert bleibt.
HIB_BY_COL = {col: hib for col, _lbl, hib, _en in SCORES}

if EXPERIMENT_NAME not in EXPERIMENTS:
    raise ValueError(
        f"Unknown EXPERIMENT_NAME '{EXPERIMENT_NAME}'. "
        f"Choose from: {list(EXPERIMENTS.keys())}"
    )
_active_keys = EXPERIMENTS[EXPERIMENT_NAME]
_score_by_key = {s[0]: s for s in SCORES}
_unknown = [k for k in _active_keys if k not in _score_by_key]
if _unknown:
    raise ValueError(
        f"EXPERIMENTS['{EXPERIMENT_NAME}'] references unknown score keys: "
        f"{_unknown}. Known keys: {list(_score_by_key.keys())}"
    )
SCORES_ACTIVE = [
    (s[0], s[1], s[2], True) for s in (_score_by_key[k] for k in _active_keys)
]
print(f"[cross_validate] Experiment: {EXPERIMENT_NAME}")
print(f"[cross_validate] Active scores: {[s[1] for s in SCORES_ACTIVE]}")
print(f"[cross_validate] Output dir: {OUTPUT_DIR}")

DPI = 800

TARGET_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
                 "#1ABC9C", "#E67E22", "#8E44AD", "#2C3E50", "#D35400",
                 "#16A085", "#C0392B", "#2980B9", "#27AE60", "#F1C40F",
                 "#7F8C8D", "#8E44AD", "#1ABC9C", "#E74C3C", "#3498DB"]

STYLE = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
}

METRIC_LABELS = {
    "AUC":   "AUC-ROC",
    "EF1%":  "Enrichment Factor (1%)",
    "EF5%":  "Enrichment Factor (5%)",
    "AUPRC": "AUPRC",
}

N_WORKERS: int | None = None


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_weight_grid(n_scores, step=0.1):
    if n_scores == 1:
        return [(1.0,)]
    n_steps = int(round(1.0 / step))
    grid = []

    def _recurse(depth, remaining_steps, current):
        if depth == n_scores - 1:
            w = round(remaining_steps * step, 4)
            grid.append(tuple(current + [max(0.0, w)]))
            return
        for i in range(remaining_steps + 1):
            w = round(i * step, 4)
            _recurse(depth + 1, remaining_steps - i, current + [max(0.0, w)])

    _recurse(0, n_steps, [])
    return grid


def recompute_ecr_weighted(df, sigma_fraction, score_cols, weights):
    """
    Rangrichtung: Rang 1 muss der BESTE Wert sein, sonst dreht
    exp(-rank/sigma) den Score um. Frueher war hier ascending=True fest
    verdrahtet, unabhaengig von higher_is_better — fuer CNNaffinity und
    CNNscore also genau verkehrt.
    """
    df = df.copy()
    N = len(df)
    if N == 0:
        df["ecr_total"] = 0.0
        return df
    sigma = max(N / sigma_fraction, 1.0)
    ecr_cols = []
    for col in score_cols:
        ecr_col = f"ecr_{col.replace('score_', '')}"
        ecr_cols.append(ecr_col)
        df[ecr_col] = 0.0
        valid_mask = df[col].notna()
        if valid_mask.sum() == 0:
            continue
        # Rang 1 muss der BESTE Wert sein, sonst dreht exp(-rank/sigma)
        # den Score um.
        hib = HIB_BY_COL[col]
        ranks = df.loc[valid_mask, col].rank(method="min", ascending=not hib)
        df.loc[valid_mask, ecr_col] = np.exp(-ranks.values / sigma)
    df["ecr_total"] = sum(w * df[c] for w, c in zip(weights, ecr_cols))
    return df


def aggregate_to_ligands(df):
    # Pose-Auswahl nach ECR. enrichment_analysis.py nimmt stattdessen
    # Pose 1 — die beiden Skripte bewerten also verschiedene Protokolle.
    idx = df.groupby("ligand")["ecr_total"].idxmax()
    return df.loc[idx].copy()


def compute_auc(labels, scores, higher_is_better):
    s = scores if higher_is_better else -scores
    mask = np.isfinite(s)
    s, labels = s[mask], labels[mask]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s)
    labels = labels[order]
    tpr = np.concatenate([[0], np.cumsum(labels == 1) / n_pos])
    fpr = np.concatenate([[0], np.cumsum(labels == 0) / n_neg])
    return float(np.trapezoid(tpr, fpr))


def compute_ef(labels, scores, higher_is_better, fraction):
    s = scores if higher_is_better else -scores
    mask = np.isfinite(s)
    s, labels = s[mask], labels[mask]
    n_total, n_active = len(labels), (labels == 1).sum()
    if n_active == 0:
        return None
    n_select = max(1, int(np.ceil(fraction * n_total)))
    order = np.argsort(-s)
    hits = labels[order][:n_select].sum()
    expected = fraction * n_active
    return float(hits / expected) if expected > 0 else None


def compute_auprc(labels, scores, higher_is_better):
    # average_precision_score statt trapezfoermiger Integration: beide
    # Programme sollen denselben Schaetzer verwenden, sonst ist AUPRC
    # zwischen ihnen nicht zitierfaehig. Die Trapezregel interpoliert
    # linear zwischen Stuetzstellen, AP summiert die tatsaechlichen
    # Praezisionswerte. Bei diesen Datenmengen liegen beide unter 0.003
    # auseinander — es geht um Vergleichbarkeit, nicht um einen groben
    # Fehler.
    s = scores if higher_is_better else -scores
    mask = np.isfinite(s)
    s, labels = s[mask], labels[mask]
    n_pos = (labels == 1).sum()
    if n_pos == 0 or (labels == 0).sum() == 0:
        return None
    return float(average_precision_score(labels, s))


def eval_ecr(df_lig):
    labels = df_lig["active"].values
    ecr = df_lig["ecr_total"].values
    return {
        "AUC":   compute_auc(labels, ecr, True),
        "EF1%":  compute_ef(labels, ecr, True, 0.01),
        "EF5%":  compute_ef(labels, ecr, True, 0.05),
        "AUPRC": compute_auprc(labels, ecr, True),
    }


def load_target_data(target_name, csv_path, score_defs):
    if not csv_path.exists():
        print(f"  WARNING: CSV not found — {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if "active" not in df.columns:
        if "ligand" in df.columns:
            df["active"] = df["ligand"].apply(
                lambda x: 1 if str(x).endswith("_a") else
                          (0 if str(x).endswith("_d") else -1))
            df = df[df["active"] >= 0].copy()
        else:
            return None

    available = [col for col, _, _, *_ in score_defs if col in df.columns]
    labels = [lbl for col, lbl, _, *_ in score_defs if col in df.columns]

    if not available:
        return None

    return df, available, labels


def evaluate_with_params(df_raw, available_scores, sigma_f, weights):
    df_ecr = recompute_ecr_weighted(df_raw, sigma_f, available_scores, weights)
    df_lig = aggregate_to_ligands(df_ecr)
    return eval_ecr(df_lig)


def _eval_single_target_grid(args):
    """Worker: evaluate one (sigma_f, weights) combo on one target DataFrame."""
    df_raw, available_scores, sigma_f, weights = args
    df_ecr = recompute_ecr_weighted(df_raw, sigma_f, available_scores, weights)
    df_lig = aggregate_to_ligands(df_ecr)
    metrics = eval_ecr(df_lig)
    val = metrics.get(PRIMARY_METRIC)
    return sigma_f, weights, val, metrics


def find_best_params(df_raw, available_scores, score_labels):
    """
    Grid search for best (sigma, weights) on a single target.
    The inner loop over (sigma_f, weights) is parallelised.
    """
    weight_grid = generate_weight_grid(len(available_scores), WEIGHT_STEP)
    tasks = [
        (df_raw, available_scores, sf, w)
        for sf in SIGMA_FRACTIONS
        for w in weight_grid
    ]

    best_val = -np.inf
    best_params = None

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for sigma_f, weights, val, metrics in ex.map(
                _eval_single_target_grid, tasks, chunksize=64):
            if val is not None and val > best_val:
                best_val = val
                best_params = (sigma_f, weights, metrics)

    return best_params


def _eval_multi_target_grid(args):
    """
    Worker: evaluate one (sigma_f, weights) combo across all training
    DataFrames; return the median PRIMARY_METRIC.

    Using median instead of mean prevents easy targets (e.g. wee1 with
    EF1%=46x) from dominating the optimisation.  Alternative: use
    mean-rank aggregation (see AGGREGATION_METHOD).
    """
    train_dfs, available_scores, sigma_f, weights, agg_method = args
    vals = []
    for df_raw in train_dfs:
        df_ecr = recompute_ecr_weighted(df_raw, sigma_f, available_scores, weights)
        df_lig = aggregate_to_ligands(df_ecr)
        val = eval_ecr(df_lig).get(PRIMARY_METRIC)
        if val is not None:
            vals.append(val)
    if not vals:
        return sigma_f, weights, -np.inf
    if agg_method == "median":
        agg_val = float(np.median(vals))
    elif agg_method == "mean":
        agg_val = float(np.mean(vals))
    else:
        agg_val = float(np.median(vals))
    return sigma_f, weights, agg_val


def find_best_params_multi(all_data, train_targets, available_scores, score_labels,
                           agg_method="median"):
    """
    Find best (sigma, weights) across multiple targets (aggregated PRIMARY_METRIC).
    The inner loop over (sigma_f, weights) is parallelised.

    agg_method: "median" (default, robust) or "mean" (legacy).
    """
    weight_grid = generate_weight_grid(len(available_scores), WEIGHT_STEP)
    train_dfs = [all_data[t]["df"] for t in train_targets]

    tasks = [
        (train_dfs, available_scores, sf, w, agg_method)
        for sf in SIGMA_FRACTIONS
        for w in weight_grid
    ]

    best_mean = -np.inf
    best_sigma = None
    best_weights = None

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for sigma_f, weights, mean_val in ex.map(
                _eval_multi_target_grid, tasks, chunksize=64):
            if mean_val > best_mean:
                best_mean = mean_val
                best_sigma = sigma_f
                best_weights = weights

    return best_sigma, best_weights


def _run_fold(args):
    """
    Worker function that executes a single LOO fold.
    Returns a list of result-dicts (one per method) plus log lines.
    """
    fold_idx, test_target, target_names, all_data, available_scores, score_labels, equal_w, precomputed_global = args

    train_targets = [t for t in target_names if t != test_target]
    logs = [f"\n  ── Fold {fold_idx+1}: test={test_target}, train={train_targets} ──"]

    df_test = all_data[test_target]["df"]
    results = []

    df_pose1 = df_test[df_test["pose"] == 1].copy() if "pose" in df_test.columns else df_test.copy()
    labels_p1 = df_pose1["active"].values
    for col, label, hib, _en in SCORES_ACTIVE:
        if col not in df_pose1.columns:
            continue
        ef  = compute_ef(labels_p1, df_pose1[col].values, hib, 0.01)
        auc = compute_auc(labels_p1, df_pose1[col].values, hib)
        auprc = compute_auprc(labels_p1, df_pose1[col].values, hib)
        ef5 = compute_ef(labels_p1, df_pose1[col].values, hib, 0.05)
        results.append({
            "fold": fold_idx + 1, "test_target": test_target,
            "method": f"{label} (Pose 1)",
            "sigma": None, "weights": None,
            "AUC": auc, "EF1%": ef, "EF5%": ef5, "AUPRC": auprc,
        })

    for col, label, hib, _en in SCORES_ACTIVE:
        if col not in df_test.columns:
            continue
        df_tmp = df_test.dropna(subset=[col]).copy()
        if df_tmp.empty:
            results.append({
                "fold": fold_idx + 1, "test_target": test_target,
                "method": f"{label} (Best Pose)",
                "sigma": None, "weights": None,
                "AUC": None, "EF1%": None, "EF5%": None, "AUPRC": None,
            })
            continue
        if hib:
            best_idx = df_tmp.groupby("ligand")[col].idxmax()
        else:
            best_idx = df_tmp.groupby("ligand")[col].idxmin()
        df_best = df_tmp.loc[best_idx]
        labels_best = df_best["active"].values
        scores_best = df_best[col].values
        ef  = compute_ef(labels_best, scores_best, hib, 0.01)
        auc = compute_auc(labels_best, scores_best, hib)
        auprc = compute_auprc(labels_best, scores_best, hib)
        ef5 = compute_ef(labels_best, scores_best, hib, 0.05)
        results.append({
            "fold": fold_idx + 1, "test_target": test_target,
            "method": f"{label} (Best Pose)",
            "sigma": None, "weights": None,
            "AUC": auc, "EF1%": ef, "EF5%": ef5, "AUPRC": auprc,
        })

    # sigma auf dem Testtarget zu scannen waere Leakage; fester Default.
    default_eq_sigma = 4
    eq_metrics = evaluate_with_params(df_test, available_scores, default_eq_sigma, equal_w)
    w_str = "/".join(f"{l}={w:.2f}" for l, w in zip(score_labels, equal_w))
    results.append({
        "fold": fold_idx + 1, "test_target": test_target,
        "method": "ECR Equal",
        "sigma": default_eq_sigma, "weights": w_str,
        **eq_metrics,
    })

    best_eq_cv_sigma = default_eq_sigma
    best_eq_cv_val = -np.inf
    for sf in SIGMA_FRACTIONS:
        train_vals = []
        for t_name in train_targets:
            m = evaluate_with_params(all_data[t_name]["df"], available_scores, sf, equal_w)
            v = m.get(PRIMARY_METRIC)
            if v is not None:
                train_vals.append(v)
        if train_vals:
            agg = float(np.median(train_vals))
            if agg > best_eq_cv_val:
                best_eq_cv_val = agg
                best_eq_cv_sigma = sf
    eq_cv_metrics = evaluate_with_params(df_test, available_scores, best_eq_cv_sigma, equal_w)
    results.append({
        "fold": fold_idx + 1, "test_target": test_target,
        "method": "ECR Equal (CV)",
        "sigma": best_eq_cv_sigma, "weights": w_str,
        **eq_cv_metrics,
    })

    # Oracle optimiert AUF dem Testtarget — obere Schranke, keine Methode.
    oracle = find_best_params(df_test, available_scores, score_labels)
    if oracle:
        o_sigma, o_weights, o_metrics = oracle
        w_str = "/".join(f"{l}={w:.2f}" for l, w in zip(score_labels, o_weights))
        results.append({
            "fold": fold_idx + 1, "test_target": test_target,
            "method": "ECR Oracle",
            "sigma": o_sigma, "weights": w_str,
            **o_metrics,
        })
        logs.append(f"    Oracle:  sigma={o_sigma}  {w_str}  "
                    f"{PRIMARY_METRIC}={o_metrics[PRIMARY_METRIC]:.1f}x")

    loo_sigma, loo_weights = find_best_params_multi(
        all_data, train_targets, available_scores, score_labels,
        agg_method="median"
    )
    if loo_sigma is not None:
        loo_metrics = evaluate_with_params(df_test, available_scores, loo_sigma, loo_weights)
        w_str = "/".join(f"{l}={w:.2f}" for l, w in zip(score_labels, loo_weights))
        results.append({
            "fold": fold_idx + 1, "test_target": test_target,
            "method": "ECR LOO-CV",
            "sigma": loo_sigma, "weights": w_str,
            **loo_metrics,
        })
        logs.append(f"    LOO-CV:  sigma={loo_sigma}  {w_str}  "
                    f"{PRIMARY_METRIC}={loo_metrics[PRIMARY_METRIC]:.1f}x")

    # Global ist auf ALLEN Targets gefittet, das Testtarget eingeschlossen.
    # Wie Oracle als Schranke zu lesen, nicht als CV-Ergebnis.
    if precomputed_global is not None:
        global_sigma, global_weights = precomputed_global
        global_metrics = evaluate_with_params(df_test, available_scores, global_sigma, global_weights)
        w_str = "/".join(f"{l}={w:.2f}" for l, w in zip(score_labels, global_weights))
        results.append({
            "fold": fold_idx + 1, "test_target": test_target,
            "method": "ECR Global",
            "sigma": global_sigma, "weights": w_str,
            **global_metrics,
        })

    return fold_idx, results, logs


def run_cross_validation():
    """Leave-one-target-out cross-validation — folds run in parallel."""

    all_data = {}
    for target_name, csv_path in TARGETS:
        result = load_target_data(target_name, csv_path, SCORES_ACTIVE)
        if result is None:
            continue
        df_raw, available, labels = result
        n_lig = df_raw["ligand"].nunique()
        n_act = int(df_raw[df_raw["pose"] == 1]["active"].sum()) if "pose" in df_raw.columns else 0
        all_data[target_name] = {
            "df": df_raw, "available": available, "labels": labels,
            "n_ligands": n_lig, "n_actives": n_act,
        }
        print(f"  Loaded {target_name}: {n_lig} ligands, {n_act} actives")

    if len(all_data) < 3:
        print(f"\n  ERROR: Need >= 3 targets for LOO-CV, have {len(all_data)}")
        sys.exit(1)

    target_names = list(all_data.keys())
    available_scores = all_data[target_names[0]]["available"]
    score_labels     = all_data[target_names[0]]["labels"]
    n_scores  = len(available_scores)
    equal_w   = tuple(round(1.0 / n_scores, 4) for _ in range(n_scores))

    print(f"\n  Leave-One-Out CV: {len(target_names)} folds")
    print(f"  Grid: {len(SIGMA_FRACTIONS)} sigma × "
          f"{len(generate_weight_grid(n_scores, WEIGHT_STEP))} weights")
    print(f"  Workers: {N_WORKERS or os.cpu_count()}")
    print(f"  Aggregation: median (robust to outlier targets)")

    print(f"\n  Pre-computing global optimum (all {len(target_names)} targets)...")
    global_sigma, global_weights = find_best_params_multi(
        all_data, target_names, available_scores, score_labels,
        agg_method="median"
    )
    precomputed_global = (global_sigma, global_weights) if global_sigma is not None else None
    if precomputed_global:
        w_str = "/".join(f"{l}={w:.2f}" for l, w in zip(score_labels, global_weights))
        print(f"  Global optimum: sigma={global_sigma}  {w_str}")

    fold_tasks = [
        (fold_idx, test_target, target_names, all_data, available_scores,
         score_labels, equal_w, precomputed_global)
        for fold_idx, test_target in enumerate(target_names)
    ]

    all_results = []
    for task in fold_tasks:
        fold_idx, fold_results, logs = _run_fold(task)
        for line in logs:
            print(line)
        all_results.append((fold_idx, fold_results))

    all_results.sort(key=lambda x: x[0])
    rows = [row for _, fold_rows in all_results for row in fold_rows]
    return pd.DataFrame(rows)


def _chunk_targets(targets, n=4):
    """Split target list into pages of at most n targets each."""
    targets = list(targets)
    return [targets[i:i + n] for i in range(0, len(targets), n)]


def plot_cv_comparison(results, out_dir):
    """Grouped bar chart: per test target, compare all methods.

    With > 4 targets, produces multiple figures (part1, part2, …),
    each showing up to 4 targets as subplots arranged in a 2×2 grid.
    """
    methods_order = ["Best Single", "Best Pose", "ECR Equal", "ECR Equal (CV)",
                     "ECR Global", "ECR LOO-CV", "ECR Oracle"]
    method_colors = {
        "Best Single":    "#95A5A6",
        "Best Pose":      "#F39C12",
        "ECR Equal":      "#9B59B6",
        "ECR Equal (CV)": "#8E44AD",
        "ECR Global":     "#3498DB",
        "ECR LOO-CV":     "#2ECC71",
        "ECR Oracle":     "#E74C3C",
    }

    all_targets = list(results["test_target"].unique())
    metrics     = ["AUC", "EF1%", "EF5%", "AUPRC"]
    pages       = _chunk_targets(all_targets, n=4)
    n_pages     = len(pages)

    for metric in metrics:
        safe = metric.replace("%", "pct")

        for page_idx, page_targets in enumerate(pages):
            n_t      = len(page_targets)
            part_tag = f"_part{page_idx + 1}" if n_pages > 1 else ""
            out_name = f"cv_comparison_{safe}{part_tag}.png"

            ncols = min(2, n_t)
            nrows = (n_t + 1) // 2
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 6, nrows * 5),
                                     squeeze=False)
            axes_flat = axes.flatten()

            for extra in range(n_t, nrows * ncols):
                axes_flat[extra].set_visible(False)

            with plt.style.context(STYLE):
                for ti, target in enumerate(page_targets):
                    ax = axes_flat[ti]
                    td = results[results["test_target"] == target]
                    x  = np.arange(len(methods_order))
                    width = 0.6 / len(methods_order)

                    for mi, method_key in enumerate(methods_order):
                        if method_key == "Best Single":
                            singles = td[td["method"].str.contains("Pose 1")]
                            v = singles[metric].max() if not singles.empty else 0
                        elif method_key == "Best Pose":
                            poses = td[td["method"].str.contains("Best Pose")]
                            v = poses[metric].max() if not poses.empty else 0
                        else:
                            row = td[td["method"] == method_key]
                            v = (row[metric].values[0]
                                 if not row.empty and pd.notna(row[metric].values[0])
                                 else 0)

                        bar = ax.bar(mi, v, width=0.7,
                                     label=method_key,
                                     color=method_colors.get(method_key, "#666"),
                                     alpha=0.85, edgecolor="white", linewidth=0.5)

                        if v > 0.01:
                            fmt = f"{v:.3f}" if metric in ("AUC", "AUPRC") else f"{v:.1f}×"
                            ax.text(mi, v + 0.01 * max(v, 1),
                                    fmt, ha="center", va="bottom",
                                    fontsize=8, fontweight="bold")

                    if metric == "AUC":
                        ax.axhline(y=0.5, color="black", linestyle="--", lw=1, alpha=0.4)
                    elif metric in ("EF1%", "EF5%"):
                        ax.axhline(y=1.0, color="black", linestyle="--", lw=1, alpha=0.4)

                    ax.set_xticks(range(len(methods_order)))
                    ax.set_xticklabels(
                        [m.replace(" ", "\n") for m in methods_order],
                        fontsize=9)
                    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
                    ax.set_title(target, fontsize=12, fontweight="bold")
                    ax.legend(fontsize=7, framealpha=0.9, loc="best")

                page_label = f" (Part {page_idx + 1}/{n_pages})" if n_pages > 1 else ""
                fig.suptitle(
                    f"Leave-One-Out CV — {METRIC_LABELS.get(metric, metric)}{page_label}",
                    fontsize=14, fontweight="bold")
                fig.tight_layout()
                _save(fig, out_dir / out_name)

    print(f"  CV comparison plots saved ({n_pages} page(s) per metric)")


def plot_generalization_gap(results, out_dir):
    all_targets = list(results["test_target"].unique())
    pages       = _chunk_targets(all_targets, n=4)
    n_pages     = len(pages)

    for page_idx, page_targets in enumerate(pages):
        n_t      = len(page_targets)
        part_tag = f"_part{page_idx + 1}" if n_pages > 1 else ""
        out_name = f"cv_generalization_gap{part_tag}.png"

        oracle_vals, loo_vals, equal_vals = [], [], []
        for t in page_targets:
            td     = results[results["test_target"] == t]
            oracle = td[td["method"] == "ECR Oracle"][PRIMARY_METRIC]
            loo    = td[td["method"] == "ECR LOO-CV"][PRIMARY_METRIC]
            equal  = td[td["method"] == "ECR Equal"][PRIMARY_METRIC]
            oracle_vals.append(oracle.values[0] if not oracle.empty else 0)
            loo_vals.append(loo.values[0]       if not loo.empty    else 0)
            equal_vals.append(equal.values[0]   if not equal.empty  else 0)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(5, n_t * 2.0 + 1.5), 5))

            x     = np.arange(n_t)
            width = 0.25
            ax.bar(x - width, equal_vals,  width, label="ECR Equal",
                   color="#9B59B6", alpha=0.85, edgecolor="white")
            ax.bar(x,         loo_vals,    width, label="ECR LOO-CV",
                   color="#2ECC71", alpha=0.85, edgecolor="white")
            ax.bar(x + width, oracle_vals, width, label="ECR Oracle",
                   color="#E74C3C", alpha=0.85, edgecolor="white")

            for i in range(n_t):
                gap     = oracle_vals[i] - loo_vals[i]
                gap_pct = (gap / oracle_vals[i] * 100) if oracle_vals[i] > 0 else 0
                y_pos   = max(oracle_vals[i], loo_vals[i]) + 0.3
                ax.annotate(f"gap: {gap_pct:.0f}%",
                            xy=(i + width / 2, y_pos),
                            fontsize=8, ha="center",
                            color="#E74C3C", fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(page_targets, fontsize=11)
            ax.set_ylabel(METRIC_LABELS.get(PRIMARY_METRIC, PRIMARY_METRIC), fontsize=12)
            page_label = f" (Part {page_idx + 1}/{n_pages})" if n_pages > 1 else ""
            ax.set_title(
                f"Generalization Gap — {METRIC_LABELS.get(PRIMARY_METRIC, PRIMARY_METRIC)}{page_label}",
                fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, framealpha=0.95)
            fig.tight_layout()
            _save(fig, out_dir / out_name)

    print(f"  Generalization gap plot saved ({n_pages} figure(s))")


def plot_parameter_stability(results, out_dir):
    """Visualise sigma and weight stability across LOO-CV folds."""
    loo = results[results["method"] == "ECR LOO-CV"].copy()
    oracle = results[results["method"] == "ECR Oracle"].copy()
    if loo.empty:
        return

    targets = list(loo["test_target"].values)
    n_t = len(targets)

    def _parse_weights(row):
        w_str = row.get("weights", "")
        if pd.isna(w_str) or not w_str:
            return {}
        d = {}
        for part in str(w_str).split("/"):
            if "=" in part:
                name, val = part.split("=", 1)
                d[name.strip()] = float(val)
        return d

    loo_w = [_parse_weights(row) for _, row in loo.iterrows()]
    score_names = list(loo_w[0].keys()) if loo_w and loo_w[0] else []
    if not score_names:
        return

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})

        ax = axes[0]
        loo_sigmas = loo["sigma"].dropna().astype(int).values
        all_sigma_vals = sorted(set(SIGMA_FRACTIONS))
        loo_counts = [np.sum(loo_sigmas == s) for s in all_sigma_vals]
        bars = ax.bar(range(len(all_sigma_vals)), loo_counts,
                      color="#2ECC71", alpha=0.85, edgecolor="white", label="LOO-CV")

        if not oracle.empty:
            oracle_sigmas = oracle["sigma"].dropna().astype(int).values
            oracle_counts = [np.sum(oracle_sigmas == s) for s in all_sigma_vals]
            ax.bar(range(len(all_sigma_vals)), oracle_counts,
                   color="#E74C3C", alpha=0.35, edgecolor="white", label="Oracle")

        ax.set_xticks(range(len(all_sigma_vals)))
        ax.set_xticklabels(all_sigma_vals, fontsize=9)
        ax.set_xlabel(r"$\sigma_{\mathrm{frac}}$", fontsize=11)
        ax.set_ylabel("Number of folds", fontsize=11)
        ax.set_title("Sigma Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        ax = axes[1]
        w_matrix = np.array([[wd.get(sn, 0.0) for sn in score_names] for wd in loo_w])
        im = ax.imshow(w_matrix.T, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1, interpolation="nearest")

        ax.set_xticks(range(n_t))
        ax.set_xticklabels(targets, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(score_names)))
        ax.set_yticklabels(score_names, fontsize=10)
        ax.set_xlabel("Test target (fold)", fontsize=11)
        ax.set_title("LOO-CV Weight per Fold", fontsize=12, fontweight="bold")

        for i in range(len(score_names)):
            for j in range(n_t):
                val = w_matrix[j, i]
                color = "white" if val > 0.55 else "#1a1a1a"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold" if val > 0 else "normal")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("Weight", fontsize=10)

        fig.suptitle("Parameter Stability Across LOO-CV Folds",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save(fig, out_dir / "cv_parameter_stability.png")

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        w_data = [np.array([wd.get(sn, 0.0) for wd in loo_w]) for sn in score_names]
        bp = ax.boxplot(w_data, labels=score_names, patch_artist=True,
                        widths=0.5, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="#E74C3C",
                                       markeredgecolor="white", markersize=7))
        colors_box = ["#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
                      "#1ABC9C", "#E67E22", "#E74C3C", "#8E44AD"]
        for patch, color in zip(bp["boxes"], colors_box[:len(score_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("Weight", fontsize=11)
        ax.set_title("LOO-CV Weight Distribution", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)

        ax = axes[1]
        x = np.arange(n_t)
        ax.bar(x - 0.2, loo_sigmas, 0.35,
               color="#2ECC71", alpha=0.85, edgecolor="white", label="LOO-CV")
        if not oracle.empty:
            o_sigmas = oracle["sigma"].dropna().astype(int).values
            if len(o_sigmas) == n_t:
                ax.bar(x + 0.2, o_sigmas, 0.35,
                       color="#E74C3C", alpha=0.55, edgecolor="white", label="Oracle")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel(r"$\sigma_{\mathrm{frac}}$", fontsize=11)
        ax.set_title("Sigma per Fold", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        fig.suptitle("Parameter Stability — Details",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save(fig, out_dir / "cv_parameter_details.png")

    print(f"  Parameter stability plots saved")


def write_cv_summary(results, out_dir):
    targets = results["test_target"].unique()
    metrics = ["AUC", "EF1%", "EF5%", "AUPRC"]

    lines = [
        "=" * 100,
        "  LEAVE-ONE-TARGET-OUT CROSS-VALIDATION — RESULTS",
        f"  Primary metric: {PRIMARY_METRIC}",
        f"  Targets: {list(targets)}",
        f"  Folds: {len(targets)} (one per target)",
        "=" * 100, "",
    ]

    for t in targets:
        td = results[results["test_target"] == t]
        lines.append("-" * 100)
        lines.append(f"  FOLD: test = {t}")
        lines.append("-" * 100)
        lines.append(f"  {'Method':<25} {'Params':<35} {'AUC':>8} {'EF1%':>8} {'EF5%':>8} {'AUPRC':>8}")
        lines.append("  " + "-" * 95)

        for _, row in td.iterrows():
            params = ""
            if pd.notna(row.get("sigma")):
                params = f"s={int(row['sigma'])}"
            if pd.notna(row.get("weights")):
                params += f" {row['weights']}" if params else row['weights']
            if not params:
                params = "—"

            auc   = f"{row['AUC']:.4f}"   if pd.notna(row['AUC'])   else "    N/A"
            ef1   = f"{row['EF1%']:>5.1f}x" if pd.notna(row['EF1%']) else "    N/A"
            ef5   = f"{row['EF5%']:>5.1f}x" if pd.notna(row['EF5%']) else "    N/A"
            auprc = f"{row['AUPRC']:.4f}" if pd.notna(row['AUPRC']) else "    N/A"

            lines.append(f"  {row['method']:<25} {params:<35} {auc:>8} {ef1:>8} {ef5:>8} {auprc:>8}")
        lines.append("")

    lines.append("=" * 100)
    lines.append(f"  SUMMARY — Mean {PRIMARY_METRIC} across folds")
    lines.append("=" * 100)

    method_means = {}
    for method in results["method"].unique():
        md   = results[results["method"] == method]
        vals = md[PRIMARY_METRIC].dropna()
        if not vals.empty:
            method_means[method] = (vals.mean(), vals.std(), len(vals))

    single_vals = []
    pose_vals = []
    for method, (mean, std, n) in sorted(method_means.items()):
        if "Pose 1" in method:
            single_vals.append(mean)
        if "Best Pose" in method:
            pose_vals.append(mean)

    best_single_mean = max(single_vals) if single_vals else 0
    best_pose_mean = max(pose_vals) if pose_vals else 0

    lines.append(f"  {'Method':<25} {'Mean':>10} {'Std':>10} {'N':>5}")
    lines.append("  " + "-" * 55)
    lines.append(f"  {'Best Single (Pose 1)':<25} {best_single_mean:>10.2f} {'—':>10} {len(targets):>5}")
    lines.append(f"  {'Best Single (Best Pose)':<25} {best_pose_mean:>10.2f} {'—':>10} {len(targets):>5}")

    for method in ["ECR Equal", "ECR Equal (CV)", "ECR Global", "ECR LOO-CV", "ECR Oracle"]:
        if method in method_means:
            mean, std, n = method_means[method]
            lines.append(f"  {method:<25} {mean:>10.2f} {std:>10.2f} {n:>5}")
    lines.append("")

    loo_vals    = results[results["method"] == "ECR LOO-CV"][PRIMARY_METRIC].dropna()
    oracle_vals = results[results["method"] == "ECR Oracle"][PRIMARY_METRIC].dropna()
    equal_vals  = results[results["method"] == "ECR Equal"][PRIMARY_METRIC].dropna()
    equal_cv_vals = results[results["method"] == "ECR Equal (CV)"][PRIMARY_METRIC].dropna()

    if not loo_vals.empty and not oracle_vals.empty:
        mean_gap     = oracle_vals.mean() - loo_vals.mean()
        mean_gap_pct = (mean_gap / oracle_vals.mean() * 100) if oracle_vals.mean() > 0 else 0
        loo_vs_equal = loo_vals.mean() - equal_vals.mean() if not equal_vals.empty else 0

        lines.append("-" * 100)
        lines.append("  GENERALIZATION ASSESSMENT")
        lines.append("-" * 100)
        lines.append(f"  Mean LOO-CV {PRIMARY_METRIC}:        {loo_vals.mean():.2f}")
        lines.append(f"  Mean Oracle {PRIMARY_METRIC}:        {oracle_vals.mean():.2f}")
        lines.append(f"  Mean generalization gap:   {mean_gap:.2f} ({mean_gap_pct:.1f}%)")
        lines.append(f"  LOO-CV vs Equal Weights:   {loo_vs_equal:+.2f}")

        if not equal_cv_vals.empty:
            sigma_effect = equal_cv_vals.mean() - equal_vals.mean()
            weight_effect = loo_vals.mean() - equal_cv_vals.mean()
            total_effect = loo_vals.mean() - equal_vals.mean()
            lines.append("")
            lines.append("  Effect decomposition (LOO-CV vs Equal):")
            lines.append(f"    Sigma optimization:    {sigma_effect:+.2f}  "
                         f"(Equal → Equal CV)")
            lines.append(f"    Weight optimization:   {weight_effect:+.2f}  "
                         f"(Equal CV → LOO-CV)")
            lines.append(f"    Total improvement:     {total_effect:+.2f}  "
                         f"(Equal → LOO-CV)")
        lines.append("")

        if mean_gap_pct < 10:
            lines.append("  >>> CONCLUSION: Parameters generalize well (gap < 10%).")
            lines.append("      The optimized weights are transferable to unseen targets.")
        elif mean_gap_pct < 25:
            lines.append("  >>> CONCLUSION: Moderate generalization gap (10-25%).")
            lines.append("      Parameters are partially transferable; consider target-specific tuning.")
        else:
            lines.append("  >>> CONCLUSION: Large generalization gap (> 25%).")
            lines.append("      Parameters may be overfit to the training targets.")
            lines.append("      Consider using equal weights or target-specific optimization.")

    lines.append("")
    lines.append("=" * 100)
    lines.append("  PARAMETER STABILITY ANALYSIS")
    lines.append("  → How consistent are the chosen parameters across LOO-CV folds?")
    lines.append("  → Stable parameters = robust transferability to new targets")
    lines.append("=" * 100)

    loo_rows = results[results["method"] == "ECR LOO-CV"].copy()
    oracle_rows = results[results["method"] == "ECR Oracle"].copy()

    if not loo_rows.empty and "sigma" in loo_rows.columns:
        loo_sigmas = loo_rows["sigma"].dropna().astype(int).values
        oracle_sigmas = oracle_rows["sigma"].dropna().astype(int).values if not oracle_rows.empty else np.array([])
        sigma_median = int(np.median(loo_sigmas)) if len(loo_sigmas) > 0 else 4

        lines.append("")
        lines.append("-" * 100)
        lines.append("  SIGMA STABILITY (LOO-CV)")
        lines.append("-" * 100)

        if len(loo_sigmas) > 0:
            sigma_median = int(np.median(loo_sigmas))
            sigma_mean = float(np.mean(loo_sigmas))
            sigma_std = float(np.std(loo_sigmas))
            sigma_min = int(np.min(loo_sigmas))
            sigma_max = int(np.max(loo_sigmas))
            sigma_unique, sigma_counts = np.unique(loo_sigmas, return_counts=True)

            lines.append(f"  LOO-CV sigma values:    {list(loo_sigmas)}")
            lines.append(f"  Median: {sigma_median}   Mean: {sigma_mean:.1f}   "
                         f"Std: {sigma_std:.1f}   Range: [{sigma_min}, {sigma_max}]")
            lines.append("")
            lines.append("  Distribution:")
            for s, c in sorted(zip(sigma_unique, sigma_counts), key=lambda x: -x[1]):
                pct = c / len(loo_sigmas) * 100
                bar = "█" * int(pct / 2)
                lines.append(f"    sigma={s:>2d}:  {c:>2d} folds ({pct:>5.1f}%)  {bar}")

            lines.append("")
            mode_pct = sigma_counts.max() / len(loo_sigmas) * 100
            if mode_pct >= 70:
                lines.append(f"  >>> SIGMA STABLE: {mode_pct:.0f}% of folds agree on "
                             f"sigma={sigma_unique[sigma_counts.argmax()]}.")
            elif mode_pct >= 40:
                lines.append(f"  >>> SIGMA MODERATELY STABLE: mode at "
                             f"sigma={sigma_unique[sigma_counts.argmax()]} ({mode_pct:.0f}%), "
                             f"but some spread.")
            else:
                lines.append(f"  >>> SIGMA UNSTABLE: no clear consensus "
                             f"(mode only {mode_pct:.0f}%). Consider fixing sigma manually.")

        if len(oracle_sigmas) > 0:
            lines.append("")
            lines.append(f"  Oracle sigma values:    {list(oracle_sigmas)}")
            lines.append(f"  Oracle median: {int(np.median(oracle_sigmas))}   "
                         f"Range: [{int(np.min(oracle_sigmas))}, {int(np.max(oracle_sigmas))}]")

        lines.append("")
        lines.append("-" * 100)
        lines.append("  WEIGHT STABILITY (LOO-CV)")
        lines.append("-" * 100)

        loo_weight_dicts = []
        score_names_ordered = []
        for _, row in loo_rows.iterrows():
            w_str = row.get("weights", "")
            if pd.isna(w_str) or not w_str:
                continue
            w_dict = {}
            for part in str(w_str).split("/"):
                part = part.strip()
                if "=" in part:
                    name, val = part.split("=", 1)
                    w_dict[name.strip()] = float(val)
            if w_dict:
                loo_weight_dicts.append(w_dict)
                if not score_names_ordered:
                    score_names_ordered = list(w_dict.keys())

        if loo_weight_dicts and score_names_ordered:
            lines.append(f"  {'Score':<18} {'Median':>8} {'Mean':>8} {'Std':>8} "
                         f"{'Min':>8} {'Max':>8} {'Zero%':>8}")
            lines.append("  " + "-" * 75)

            weight_stability = {}
            for score_name in score_names_ordered:
                vals = np.array([wd.get(score_name, 0.0) for wd in loo_weight_dicts])
                med = float(np.median(vals))
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                zero_pct = float((vals == 0).sum() / len(vals) * 100)
                weight_stability[score_name] = {
                    "median": med, "mean": mean, "std": std,
                    "min": float(np.min(vals)), "max": float(np.max(vals)),
                    "zero_pct": zero_pct, "values": vals,
                }
                lines.append(
                    f"  {score_name:<18} {med:>8.2f} {mean:>8.2f} {std:>8.2f} "
                    f"{np.min(vals):>8.2f} {np.max(vals):>8.2f} {zero_pct:>7.1f}%"
                )

            lines.append("")
            lines.append("  Per-fold weights:")
            header = f"  {'Target':<12}"
            for sn in score_names_ordered:
                header += f" {sn:>10}"
            header += f" {'sigma':>6}"
            lines.append(header)
            lines.append("  " + "-" * (14 + 11 * len(score_names_ordered) + 7))
            for _, row in loo_rows.iterrows():
                target = row["test_target"]
                sigma = int(row["sigma"]) if pd.notna(row.get("sigma")) else "?"
                w_str = row.get("weights", "")
                w_dict = {}
                if pd.notna(w_str) and w_str:
                    for part in str(w_str).split("/"):
                        if "=" in part:
                            name, val = part.split("=", 1)
                            w_dict[name.strip()] = float(val)
                line = f"  {target:<12}"
                for sn in score_names_ordered:
                    line += f" {w_dict.get(sn, 0.0):>10.2f}"
                line += f" {sigma:>6}"
                lines.append(line)

            lines.append("")
            avg_std = np.mean([ws["std"] for ws in weight_stability.values()])
            n_consistent = sum(1 for ws in weight_stability.values() if ws["std"] < 0.15)
            n_scores_total = len(score_names_ordered)
            n_always_zero = sum(1 for ws in weight_stability.values() if ws["zero_pct"] >= 80)

            if n_always_zero > 0:
                zero_scores = [sn for sn in score_names_ordered
                               if weight_stability[sn]["zero_pct"] >= 80]
                lines.append(f"  >>> SCORES WITH ZERO WEIGHT in ≥80% of folds: "
                             f"{', '.join(zero_scores)}")
                lines.append(f"      → Consider removing these from the ECR.")

            if avg_std < 0.10:
                lines.append(f"  >>> WEIGHTS STABLE: mean std={avg_std:.3f} across scores.")
                lines.append(f"      The optimized weights are consistent across folds.")
            elif avg_std < 0.20:
                lines.append(f"  >>> WEIGHTS MODERATELY STABLE: mean std={avg_std:.3f}.")
                lines.append(f"      {n_consistent}/{n_scores_total} scores have std < 0.15.")
            else:
                lines.append(f"  >>> WEIGHTS UNSTABLE: mean std={avg_std:.3f}.")
                lines.append(f"      Optimal weights vary strongly across targets.")
                lines.append(f"      Consider using equal weights or target-class-specific parameters.")

            global_rows_cmp = results[results["method"] == "ECR Global"]
            if not global_rows_cmp.empty:
                lines.append("")
                lines.append("-" * 100)
                lines.append("  GLOBAL vs LOO-CV PARAMETER AGREEMENT")
                lines.append("-" * 100)

                g_row_cmp = global_rows_cmp.iloc[0]
                g_sigma_cmp = int(g_row_cmp["sigma"]) if pd.notna(g_row_cmp.get("sigma")) else None
                g_weights_str_cmp = g_row_cmp.get("weights", "")

                if g_sigma_cmp is not None and len(loo_sigmas) > 0:
                    sigma_agreement = (loo_sigmas == g_sigma_cmp).sum()
                    lines.append(f"  Global sigma:    {g_sigma_cmp}")
                    lines.append(f"  LOO-CV median:   {sigma_median}")
                    lines.append(f"  LOO-CV folds that agree with Global: "
                                 f"{sigma_agreement}/{len(loo_sigmas)} "
                                 f"({sigma_agreement/len(loo_sigmas)*100:.0f}%)")

                if pd.notna(g_weights_str_cmp) and g_weights_str_cmp:
                    lines.append(f"  Global weights:  {g_weights_str_cmp}")
                    g_dict_cmp = {}
                    for part in str(g_weights_str_cmp).split("/"):
                        if "=" in part:
                            name, val = part.split("=", 1)
                            g_dict_cmp[name.strip()] = float(val)
                    if g_dict_cmp and score_names_ordered:
                        lines.append(f"  LOO-CV medians:  " +
                                     "/".join(f"{sn}={weight_stability[sn]['median']:.2f}"
                                              for sn in score_names_ordered))
                        g_vec = np.array([g_dict_cmp.get(sn, 0.0) for sn in score_names_ordered])
                        m_vec = np.array([weight_stability[sn]["median"] for sn in score_names_ordered])
                        dist = np.sqrt(np.sum((g_vec - m_vec) ** 2))
                        lines.append(f"  Euclidean distance (Global vs LOO-CV median): {dist:.3f}")
                        if dist < 0.10:
                            lines.append("  >>> GOOD AGREEMENT: Global and LOO-CV converge to similar weights.")
                        elif dist < 0.25:
                            lines.append("  >>> MODERATE AGREEMENT: some divergence between Global and LOO-CV.")
                        else:
                            lines.append("  >>> POOR AGREEMENT: Global and LOO-CV give different weights.")
                            lines.append("      The Global fit may be influenced by specific outlier targets.")

        lines.append("")
        lines.append("=" * 100)
        lines.append("  RECOMMENDED PARAMETERS FOR PROSPECTIVE SCREENING")
        lines.append("=" * 100)

        global_rows = results[results["method"] == "ECR Global"]
        if not global_rows.empty:
            g_row = global_rows.iloc[0]
            g_sigma = int(g_row["sigma"]) if pd.notna(g_row.get("sigma")) else sigma_median
            g_weights_str = g_row.get("weights", "")

            global_mean_ef = results[results["method"] == "ECR Global"][PRIMARY_METRIC].dropna().mean()
            loo_mean_ef = loo_vals.mean() if not loo_vals.empty else 0

            lines.append("")
            lines.append(f"  Global params:   sigma={g_sigma}  {g_weights_str}")
            lines.append(f"  Global mean {PRIMARY_METRIC}:  {global_mean_ef:.2f}  "
                         f"(optimistic — includes test in training)")
            lines.append(f"  LOO-CV mean {PRIMARY_METRIC}:  {loo_mean_ef:.2f}  "
                         f"(honest — test excluded from training)")
            lines.append("")
            lines.append(f"  >>> USE Global params as starting point for prospective screens.")
            lines.append(f"  >>> EXPECT performance closer to LOO-CV estimate ({loo_mean_ef:.2f}x)"
                         f" than Global estimate ({global_mean_ef:.2f}x).")

            if loo_weight_dicts and score_names_ordered:
                n_always_zero_final = sum(
                    1 for sn in score_names_ordered
                    if weight_stability.get(sn, {}).get("zero_pct", 0) >= 80
                )
                if n_always_zero_final > 0:
                    zero_scores_final = [
                        sn for sn in score_names_ordered
                        if weight_stability.get(sn, {}).get("zero_pct", 0) >= 80
                    ]
                    n_scores_total_final = len(score_names_ordered)
                    lines.append(f"  >>> CONSIDER removing {', '.join(zero_scores_final)} and re-running "
                                 f"with {n_scores_total_final - n_always_zero_final} scores.")

    lines.append("")
    lines.append("=" * 100)

    txt_path = out_dir / "cv_summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary: {txt_path}")
    for line in lines:
        print(f"  {line}")


def main():
    print("=" * 60)
    print(" Leave-One-Target-Out Cross-Validation  [parallel]")
    print(f" {len(TARGETS)} targets")
    print(f" Primary metric: {PRIMARY_METRIC}")
    print(f" Workers: {N_WORKERS or os.cpu_count()} (set N_WORKERS to override)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_cross_validation()

    if results.empty:
        print("\n  ERROR: No results — aborting.")
        sys.exit(1)

    csv_path = OUTPUT_DIR / "cv_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Results: {csv_path}")

    print("\n── Plots ──────────────────────────────────")
    plot_cv_comparison(results, OUTPUT_DIR)
    plot_generalization_gap(results, OUTPUT_DIR)
    plot_parameter_stability(results, OUTPUT_DIR)

    write_cv_summary(results, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f" Done! All files in: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
