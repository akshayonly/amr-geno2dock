#!/usr/bin/env python3
# coding: utf-8
# AUTHOR: Akshay Shirsath (2026)
# =============================================================================
# AMR-Geno2Dock — XGBoost Hyperparameter Tuning (Stage 05c)
# =============================================================================
#
# PURPOSE:
#   Tune XGBoost on the full dataset using a nested cross-validation scheme
#   with Bayesian hyperparameter search (Optuna), then sweep the classification
#   threshold to minimise VME toward the FDA ≤1.5% target.
#
#   Produces a fully trained, tuned model ready for SHAP analysis in Stage 06.
#
# VALIDATION SCHEME — Nested CV (literature standard for AMR-ML):
#
#   ┌─ Outer loop: 5-fold stratified CV ──────────────────────────────┐
#   │   Produces unbiased generalisation estimates (reported metrics)  │
#   │   ┌─ Inner loop: 3-fold stratified CV ───────────────────────┐  │
#   │   │   Optuna Bayesian search over hyperparameter space        │  │
#   │   │   Optimises: ROC-AUC (primary) + VME penalty             │  │
#   │   └────────────────────────────────────────────────────────────┘  │
#   └──────────────────────────────────────────────────────────────────┘
#   Final model: retrain best params on FULL dataset → saved to disk
#
# PHASES:
#   0  — Imports, constants & CLI
#   1  — Data loading & validation
#   2  — Optuna objective (inner CV)
#   3  — Nested CV outer loop
#   4  — Threshold optimisation (VME reduction)
#   5  — Final model training on full data
#   6  — Report & export
#
# INPUT:
#   --X   ml_data_*/X.csv
#   --y   ml_data_*/y.csv
#
# OUTPUT:
#   tuning/
#   ├── best_params.json           — best hyperparameters from Optuna
#   ├── nested_cv_scores.csv       — per-fold metrics (outer CV)
#   ├── nested_cv_summary.txt      — mean ± std across folds
#   ├── threshold_sweep.csv        — VME/ME/F1/AUC at each threshold
#   ├── tuned_xgb_model.json       — final trained XGBoost model
#   └── tuning.log
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0 — Imports, constants & CLI
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: pip install xgboost")
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna")
    sys.exit(1)

warnings.filterwarnings("ignore")

# FDA thresholds
FDA_VME = 1.5
FDA_ME  = 3.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0B — CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amr_geno2dock_pipeline_stage_05c.py",
        description=(
            "Stage 05c — XGBoost hyperparameter tuning via Optuna Bayesian "
            "search inside a nested 5×3-fold stratified CV, followed by "
            "classification threshold optimisation for VME reduction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--X", required=True, type=Path, metavar="CSV",
        help="Feature matrix from Stage 04 (index_col=0).",
    )
    parser.add_argument(
        "--y", required=True, type=Path, metavar="CSV",
        help="Binary label CSV from Stage 04 (index_col=0).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("tuning"), metavar="DIR",
        help="Output directory.",
    )

    # ── Nested CV ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--outer-folds", type=int, default=5, metavar="K",
        help="Number of folds in the outer CV loop (unbiased performance estimate).",
    )
    parser.add_argument(
        "--inner-folds", type=int, default=3, metavar="K",
        help="Number of folds in the inner CV loop (hyperparameter optimisation).",
    )

    # ── Optuna ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--n-trials", type=int, default=100, metavar="N",
        help=(
            "Optuna trials per outer fold. 100 is a good balance of "
            "search quality vs. runtime on 4 cores. "
            "Increase to 200 for more thorough search."
        ),
    )
    parser.add_argument(
        "--optuna-jobs", type=int, default=1, metavar="N",
        help=(
            "Parallel Optuna jobs per trial. Keep at 1 — XGBoost already "
            "uses n_jobs=-1 internally. Avoids CPU oversubscription."
        ),
    )
    parser.add_argument(
        "--vme-penalty", type=float, default=2.0, metavar="FLOAT",
        help=(
            "Weight applied to VME in the Optuna objective. "
            "Objective = AUC - (vme_penalty × VME_rate). "
            "Higher values push the search toward VME minimisation "
            "at the cost of some AUC. Default 2.0 is a moderate penalty."
        ),
    )

    # ── Threshold sweep ───────────────────────────────────────────────────────
    parser.add_argument(
        "--threshold-steps", type=int, default=100, metavar="N",
        help="Number of threshold values to sweep between 0.1 and 0.9.",
    )
    parser.add_argument(
        "--target-vme", type=float, default=FDA_VME, metavar="FLOAT",
        help=f"Target VME%% for threshold optimisation (default: FDA threshold {FDA_VME}%%).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--seed", type=int, default=42, metavar="INT",
        help="Global random seed.",
    )
    parser.add_argument(
        "--no-pruning", action="store_true",
        help="Disable Optuna MedianPruner (slower but more thorough).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging and Optuna INFO verbosity.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(x_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not x_path.exists():
        log.error(f"X not found: {x_path}"); sys.exit(1)
    if not y_path.exists():
        log.error(f"y not found: {y_path}"); sys.exit(1)

    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0).squeeze()

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    ratio = round(n_neg / n_pos, 4)

    log.info(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    log.info(f"Labels : y=1 (resistant): {n_pos} | y=0 (susceptible): {n_neg}")
    log.info(f"scale_pos_weight (n_neg/n_pos): {ratio}")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 2 — Optuna objective (inner CV)
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_folds: int,
    seed: int,
    vme_penalty: float,
    n_pos: int,
    n_neg: int,
):
    """
    Returns an Optuna objective function closed over the training data.

    Hyperparameter search space (covers all major XGBoost knobs):

        n_estimators      [100, 500]   — number of trees; more = slower but better
        max_depth         [3, 8]       — tree depth; lower = less overfit
        learning_rate     [0.01, 0.3]  — shrinkage; lower needs more trees
        min_child_weight  [1, 10]      — min sum of instance weight in a leaf
        subsample         [0.6, 1.0]   — row sampling per tree (stochastic GBM)
        colsample_bytree  [0.6, 1.0]   — column sampling per tree
        gamma             [0, 5]       — min loss reduction for a split
        reg_alpha         [0, 2]       — L1 regularisation
        reg_lambda        [0.5, 5]     — L2 regularisation
        scale_pos_weight  fixed        — n_neg/n_pos; not tuned (shown to
                                         have minimal effect at this ratio)

    Objective = mean(AUC across inner folds) - vme_penalty × mean(VME rate)

    Maximising this objective simultaneously rewards high AUC and low VME.
    vme_penalty=2.0 means a 1% VME increase costs as much as a 0.02 AUC drop.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5),
            "scale_pos_weight":  round(n_neg / n_pos, 4),
            "eval_metric":       "logloss",
            "verbosity":         0,
            "random_state":      seed,
            "n_jobs":            -1,
        }

        clf = XGBClassifier(**params)
        inner_cv = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed
        )

        fold_aucs: list[float] = []
        fold_vmes: list[float] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(
            inner_cv.split(X_train, y_train)
        ):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            fold_aucs.append(roc_auc_score(y_val, y_prob))

            tn, fp, fn, tp = confusion_matrix(
                y_val, y_pred, labels=[0, 1]
            ).ravel()
            n_r = tp + fn
            fold_vmes.append(fn / n_r if n_r > 0 else 0.0)

            # Optuna pruning — report intermediate value
            trial.report(np.mean(fold_aucs), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_auc = float(np.mean(fold_aucs))
        mean_vme = float(np.mean(fold_vmes))

        # Composite objective: maximise AUC, penalise VME
        return mean_auc - vme_penalty * mean_vme

    return objective


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 3 — Nested CV outer loop
# ─────────────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_r = tp + fn
    n_s = tn + fp
    vme = fn / n_r if n_r > 0 else 0.0
    me  = fp / n_s if n_s > 0 else 0.0
    return {
        "ROC-AUC":     round(roc_auc_score(y_true, y_prob), 4),
        "F1 (macro)":  round(f1_score(y_true, y_pred, average="macro"), 4),
        "MCC":         round(matthews_corrcoef(y_true, y_pred), 4),
        "Sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "VME (%)":     round(vme * 100, 2),
        "ME (%)":      round(me  * 100, 2),
        "Accuracy":    round((tp + tn) / (tp + tn + fp + fn), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    n_trials: int,
    seed: int,
    vme_penalty: float,
    n_pos: int,
    n_neg: int,
    use_pruning: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Run the full nested CV.

    Returns:
        fold_results  — per-outer-fold metric dicts
        fold_params   — best hyperparameters found in each outer fold's inner CV
    """
    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=seed
    )

    fold_results: list[dict] = []
    fold_params:  list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        log.info(f"Outer fold {fold_idx}/{outer_folds} — running Optuna ({n_trials} trials) …")

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # ── Inner: Optuna search ──────────────────────────────────────────────
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
            if use_pruning
            else optuna.pruners.NopPruner()
        )
        sampler = optuna.samplers.TPESampler(seed=seed + fold_idx)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(
            make_objective(X_tr, y_tr, inner_folds, seed, vme_penalty, n_pos, n_neg),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_params.update({
            "scale_pos_weight": round(n_neg / n_pos, 4),
            "eval_metric":      "logloss",
            "verbosity":        0,
            "random_state":     seed,
            "n_jobs":           -1,
        })
        fold_params.append(best_params)

        log.info(
            f"  Fold {fold_idx} best inner score: {study.best_value:.4f} "
            f"| best params: max_depth={best_params['max_depth']} "
            f"lr={best_params['learning_rate']:.3f} "
            f"n_est={best_params['n_estimators']}"
        )

        # ── Outer: evaluate on held-out fold ──────────────────────────────────
        clf = XGBClassifier(**best_params)
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_fold_metrics(y_te, y_pred, y_prob)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

        log.info(
            f"  Fold {fold_idx} outer results: "
            f"AUC={metrics['ROC-AUC']:.4f} | "
            f"F1={metrics['F1 (macro)']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"VME={metrics['VME (%)']:.1f}% | "
            f"ME={metrics['ME (%)']:.1f}%"
        )

    return fold_results, fold_params


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 4 — Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def sweep_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_steps: int,
    target_vme: float,
) -> tuple[pd.DataFrame, float]:
    """
    Sweep classification threshold from 0.1 to 0.9 on the full dataset
    (using the final model trained on all data).

    For each threshold, compute VME, ME, F1, AUC, MCC and flag
    whether FDA thresholds are met.

    Returns:
        sweep_df      — DataFrame with all threshold results
        optimal_threshold — threshold that meets target_vme with best F1

    Note: This is performed on the full dataset probabilities from the
    final model, not inside CV. It is used to select the operating
    threshold for deployment/reporting, not to estimate generalisation.
    The nested CV outer-fold metrics (at threshold=0.5) remain the
    unbiased performance estimate.
    """
    thresholds = np.linspace(0.1, 0.9, n_steps)
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n_r = tp + fn
        n_s = tn + fp
        vme = fn / n_r if n_r > 0 else 0.0
        me  = fp / n_s if n_s > 0 else 0.0
        rows.append({
            "threshold":   round(t, 4),
            "VME (%)":     round(vme * 100, 2),
            "ME (%)":      round(me  * 100, 2),
            "VME FDA":     "✅" if vme * 100 <= FDA_VME else "❌",
            "ME FDA":      "✅" if me  * 100 <= FDA_ME  else "❌",
            "F1 (macro)":  round(f1_score(y_true, y_pred, average="macro"), 4),
            "MCC":         round(matthews_corrcoef(y_true, y_pred), 4),
            "ROC-AUC":     round(roc_auc_score(y_true, y_prob), 4),   # constant
            "Sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
            "Accuracy":    round((tp + tn) / (tp + tn + fp + fn), 4),
        })

    sweep_df = pd.DataFrame(rows)

    # Find optimal threshold: meets target_vme with highest F1
    meets_vme = sweep_df[sweep_df["VME (%)"] <= target_vme]
    if meets_vme.empty:
        log.warning(
            f"No threshold achieves VME ≤ {target_vme}%. "
            f"Selecting threshold with lowest VME instead."
        )
        optimal_row = sweep_df.loc[sweep_df["VME (%)"].idxmin()]
    else:
        optimal_row = meets_vme.loc[meets_vme["F1 (macro)"].idxmax()]

    optimal_threshold = float(optimal_row["threshold"])
    log.info(
        f"Optimal threshold: {optimal_threshold:.4f} → "
        f"VME={optimal_row['VME (%)']:.1f}% | "
        f"ME={optimal_row['ME (%)']:.1f}% | "
        f"F1={optimal_row['F1 (macro)']:.4f}"
    )

    return sweep_df, optimal_threshold


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 5 — Final model on full data
# ─────────────────────────────────────────────────────────────────────────────

def select_final_params(fold_params: list[dict]) -> dict:
    """
    Aggregate hyperparameters across outer folds to select final params.

    Strategy: take the median of continuous parameters and the mode of
    integer parameters across all fold-best param sets. This is more robust
    than taking params from the single best fold.
    """
    keys = [k for k in fold_params[0] if k not in
            ("eval_metric", "verbosity", "random_state", "n_jobs",
             "scale_pos_weight")]

    final = {}
    for k in keys:
        vals = [p[k] for p in fold_params]
        if isinstance(vals[0], int):
            # Mode for integers
            final[k] = int(pd.Series(vals).mode()[0])
        else:
            # Median for floats
            final[k] = float(np.median(vals))

    # Restore fixed params
    final.update({
        "scale_pos_weight": fold_params[0]["scale_pos_weight"],
        "eval_metric":      "logloss",
        "verbosity":        0,
        "random_state":     fold_params[0]["random_state"],
        "n_jobs":           -1,
    })

    return final


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    final_params: dict,
    model_path: Path,
) -> XGBClassifier:
    """Train on the full dataset and save to disk."""
    log.info("Training final model on full dataset …")
    clf = XGBClassifier(**final_params)
    clf.fit(X, y)
    clf.save_model(str(model_path))
    log.info(f"Final model saved → {model_path}")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 6 — Report & export
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    fold_results: list[dict],
    fold_params:  list[dict],
    final_params: dict,
    sweep_df: pd.DataFrame,
    optimal_threshold: float,
    out_dir: Path,
    args: argparse.Namespace,
    X_shape: tuple,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-fold CSV ──────────────────────────────────────────────────────────
    cv_df = pd.DataFrame(fold_results).set_index("fold")
    cv_df.to_csv(out_dir / "nested_cv_scores.csv")

    # ── Best params per fold ──────────────────────────────────────────────────
    params_df = pd.DataFrame(fold_params)
    params_df.index = [f"fold_{i+1}" for i in range(len(fold_params))]
    params_df.to_csv(out_dir / "fold_best_params.csv")

    # ── Final params JSON ─────────────────────────────────────────────────────
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(final_params, f, indent=2)

    # ── Threshold sweep CSV ───────────────────────────────────────────────────
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # ── Summary statistics ────────────────────────────────────────────────────
    metric_cols = ["ROC-AUC", "F1 (macro)", "MCC", "Sensitivity",
                   "Specificity", "VME (%)", "ME (%)"]
    means  = cv_df[metric_cols].mean()
    stds   = cv_df[metric_cols].std()
    opt_row = sweep_df[sweep_df["threshold"] == round(optimal_threshold, 4)].iloc[0] \
              if round(optimal_threshold, 4) in sweep_df["threshold"].values \
              else sweep_df.iloc[(sweep_df["threshold"] - optimal_threshold).abs().argsort()[:1]].iloc[0]

    summary_lines = [
        "═" * 68,
        "  STAGE 05c — XGBOOST TUNING REPORT",
        "═" * 68,
        f"  Dataset      : {X_shape[0]} samples × {X_shape[1]} features",
        f"  Outer folds  : {args.outer_folds}  |  Inner folds: {args.inner_folds}",
        f"  Optuna trials: {args.n_trials} per fold  |  VME penalty: {args.vme_penalty}",
        f"  FDA targets  : VME ≤ {FDA_VME}%  |  ME ≤ {FDA_ME}%",
        "",
        "  NESTED CV RESULTS (threshold = 0.50, unbiased estimates)",
        "  ─" * 34,
        f"  {'Metric':<18} {'Mean':>8} {'± Std':>8}  {'FDA':>6}",
        "  ─" * 34,
    ]
    for m in metric_cols:
        fda = ""
        if m == "VME (%)":
            fda = "✅" if means[m] <= FDA_VME else "❌"
        elif m == "ME (%)":
            fda = "✅" if means[m] <= FDA_ME else "❌"
        summary_lines.append(
            f"  {m:<18} {means[m]:>8.4f} {f'± {stds[m]:.4f}':>8}  {fda:>6}"
        )

    summary_lines += [
        "",
        f"  THRESHOLD OPTIMISATION (target VME ≤ {args.target_vme}%)",
        "  ─" * 34,
        f"  Optimal threshold  : {optimal_threshold:.4f}",
        f"  VME at threshold   : {opt_row['VME (%)']:.1f}%  {opt_row['VME FDA']}",
        f"  ME  at threshold   : {opt_row['ME (%)']:.1f}%  {opt_row['ME FDA']}",
        f"  F1 at threshold    : {opt_row['F1 (macro)']:.4f}",
        f"  MCC at threshold   : {opt_row['MCC']:.4f}",
        f"  Sensitivity        : {opt_row['Sensitivity']:.4f}",
        f"  Specificity        : {opt_row['Specificity']:.4f}",
        "",
        "  FINAL MODEL PARAMETERS (median aggregated across folds)",
        "  ─" * 34,
    ]
    for k, v in final_params.items():
        if k not in ("eval_metric", "verbosity", "n_jobs"):
            summary_lines.append(f"  {k:<25}: {v}")

    summary_lines += [
        "",
        "  OUTPUT FILES",
        "  ─" * 34,
        f"  nested_cv_scores.csv    → {out_dir}/nested_cv_scores.csv",
        f"  fold_best_params.csv    → {out_dir}/fold_best_params.csv",
        f"  best_params.json        → {out_dir}/best_params.json",
        f"  threshold_sweep.csv     → {out_dir}/threshold_sweep.csv",
        f"  tuned_xgb_model.json    → {out_dir}/tuned_xgb_model.json",
        "═" * 68,
        "",
        "  NEXT STEP (Stage 06 — SHAP + Co-selection Filter):",
        "    from xgboost import XGBClassifier",
        "    import json, pandas as pd",
        "    params = json.load(open('tuning/best_params.json'))",
        "    model  = XGBClassifier(**params)",
        "    model.load_model('tuning/tuned_xgb_model.json')",
        "    # → shap.TreeExplainer(model) → SHAP values → target prioritisation",
        "═" * 68,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (out_dir / "nested_cv_summary.txt").write_text(summary_text)


# ─────────────────────────────────────────────────────────────────────────────
## MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        optuna.logging.set_verbosity(optuna.logging.INFO)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Load data ────────────────────────────────────────────────────
    X_df, y_s = load_data(args.X, args.y)
    X = X_df.values.astype(np.float32)
    y = y_s.values.astype(int)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    # ── Phase 3: Nested CV ────────────────────────────────────────────────────
    log.info(
        f"Starting nested CV: {args.outer_folds} outer × {args.inner_folds} inner folds "
        f"| {args.n_trials} Optuna trials per fold"
    )
    fold_results, fold_params = run_nested_cv(
        X, y,
        outer_folds  = args.outer_folds,
        inner_folds  = args.inner_folds,
        n_trials     = args.n_trials,
        seed         = args.seed,
        vme_penalty  = args.vme_penalty,
        n_pos        = n_pos,
        n_neg        = n_neg,
        use_pruning  = not args.no_pruning,
    )

    # ── Phase 5: Final model ──────────────────────────────────────────────────
    final_params = select_final_params(fold_params)
    log.info(f"Aggregated final params: {final_params}")

    model_path = args.out_dir / "tuned_xgb_model.json"
    final_model = train_final_model(X, y, final_params, model_path)

    # ── Phase 4: Threshold sweep on final model ───────────────────────────────
    log.info("Running threshold sweep on full dataset probabilities …")
    y_prob_full = final_model.predict_proba(X)[:, 1]
    sweep_df, optimal_threshold = sweep_threshold(
        y, y_prob_full, args.threshold_steps, args.target_vme
    )

    # ── Phase 6: Report ───────────────────────────────────────────────────────
    write_report(
        fold_results, fold_params, final_params,
        sweep_df, optimal_threshold,
        args.out_dir, args, X_df.shape,
    )


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE
# =============================================================================
#
#  Standard run on 69-gene set (recommended from Stage 05b):
#    python amr_geno2dock_pipeline_stage_05c.py \
#        --X ml_data/X.csv \
#        --y ml_data/y.csv \
#        --out-dir tuning
#
#  Stronger VME penalty (prioritise VME reduction over AUC):
#    python amr_geno2dock_pipeline_stage_05c.py \
#        --X ml_data/X.csv \
#        --y ml_data/y.csv \
#        --vme-penalty 5.0
#
#  More Optuna trials (better search, slower):
#    python amr_geno2dock_pipeline_stage_05c.py \
#        --X ml_data/X.csv \
#        --y ml_data/y.csv \
#        --n-trials 200
#
#  Any feature set works — just swap the path:
#    python amr_geno2dock_pipeline_stage_05c.py \
#        --X ml_data_5pct/X.csv \
#        --y ml_data_5pct/y.csv \
#        --out-dir tuning_5pct
#
# =============================================================================
# INSTALL
# =============================================================================
#   pip install xgboost optuna
# =============================================================================