#!/usr/bin/env python3
# coding: utf-8
# AUTHOR: Akshay Shirsath (2026)
# =============================================================================
# AMR-Geno2Dock — XGBoost Baseline Benchmark (Stage 05b)
# =============================================================================
#
# PURPOSE:
#   Benchmark XGBoost at default hyperparameters across all three feature
#   sets produced by Stage 04, on the same stratified subset used in
#   Stage 05a, to give a direct apples-to-apples comparison.
#
#   XGBoost is tested with three class-imbalance handling strategies:
#     1. No correction          — raw default, majority-class biased
#     2. scale_pos_weight       — built-in XGBoost imbalance correction
#     3. sample_weight          — per-sample weights passed at fit time
#
#   This isolates the effect of (a) feature set size and (b) imbalance
#   handling, independently, before any hyperparameter tuning in Stage 05c.
#
# INPUT:
#   --datasets  list of "label:X_path:y_path" triples, one per feature set
#               e.g.
#                 "28 genes (5%):ml_data_5pct/X.csv:ml_data_5pct/y.csv"
#                 "69 genes (1%):ml_data_1pct/X.csv:ml_data_1pct/y.csv"
#                 "206 genes (all):ml_data_all/X.csv:ml_data_all/y.csv"
#
# OUTPUT:
#   xgb_baseline/
#   ├── xgb_baseline_scores.csv   — full metric table
#   ├── xgb_baseline_scores.md    — markdown comparison table
#   └── xgb_baseline.log          — run log
#
# NOTES ON XGBoost IMBALANCE HANDLING:
#   scale_pos_weight = n_negative / n_positive
#     Tells XGBoost to weight the minority class (resistant, y=1) more
#     heavily during tree construction. Equivalent to class_weight in sklearn.
#     Your ratio: 228 / 315 = 0.724 → scale_pos_weight = 0.724
#     (susceptible is the minority in raw count but resistant is y=1;
#      XGBoost uses this to upweight the positive class if it is minority —
#      in your case resistant IS the majority so this barely moves things,
#      but it's included for methodological completeness and future datasets.)
#
#   sample_weight (compute_sample_weight)
#     Per-sample weights computed from class frequency and passed to
#     fit(). More flexible than scale_pos_weight — can be combined with
#     SMOTE or other resampling in later stages.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0 — Imports & config
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
import time
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:
    print(
        "ERROR: xgboost not installed.\n"
        "Install with:  pip install xgboost\n"
        "           or: conda install -c conda-forge xgboost"
    )
    sys.exit(1)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# FDA thresholds (software-based AST devices)
FDA_VME_THRESHOLD = 1.5
FDA_ME_THRESHOLD  = 3.0


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0B — CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amr_geno2dock_pipeline_stage_05b.py",
        description=(
            "Stage 05b — XGBoost baseline benchmark across multiple feature sets. "
            "Tests default hyperparameters with three imbalance-handling strategies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="LABEL:X_CSV:Y_CSV",
        help=(
            "One or more dataset triples in the format 'label:X_path:y_path'. "
            "Example: '28genes:ml_data_5pct/X.csv:ml_data_5pct/y.csv'"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("xgb_baseline"),
        metavar="DIR",
        help="Output directory.",
    )
    parser.add_argument(
        "--subset-size",
        type=float,
        default=0.60,
        metavar="FLOAT",
        help="Fraction of each dataset to use as benchmark subset.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.70,
        metavar="FLOAT",
        help="Train fraction within the subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed — must match Stage 05a for comparable subsets.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        metavar="N",
        help="Number of XGBoost trees (default baseline: 100).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 1 — Data loading & subset
# ─────────────────────────────────────────────────────────────────────────────

def parse_dataset_arg(triple: str) -> tuple[str, Path, Path]:
    """
    Parse a 'label:X_path:y_path' CLI argument.
    Allows colons in paths on Windows by splitting on first two colons only.
    """
    parts = triple.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Dataset argument must be 'label:X_path:y_path', got: '{triple}'"
        )
    label  = parts[0]
    x_path = Path(":".join(parts[1:-1]))   # handles Windows drive letters
    y_path = Path(parts[-1])
    return label, x_path, y_path


def load_dataset(x_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not x_path.exists():
        log.error(f"X file not found: {x_path}")
        sys.exit(1)
    if not y_path.exists():
        log.error(f"y file not found: {y_path}")
        sys.exit(1)
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0).squeeze()
    return X, y


def make_subset(
    X: pd.DataFrame,
    y: pd.Series,
    subset_size: float,
    train_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - subset_size, random_state=seed
    )
    subset_idx, _ = next(sss_outer.split(X, y))
    X_sub, y_sub = X.iloc[subset_idx], y.iloc[subset_idx]

    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - train_size, random_state=seed
    )
    train_idx, test_idx = next(sss_inner.split(X_sub, y_sub))
    return (
        X_sub.iloc[train_idx], X_sub.iloc[test_idx],
        y_sub.iloc[train_idx], y_sub.iloc[test_idx],
    )


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 2 — XGBoost variant registry
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_variants(
    n_estimators: int,
    seed: int,
    n_pos: int,
    n_neg: int,
) -> dict[str, dict]:
    """
    Three XGBoost variants differing only in how class imbalance is handled.

    Returns a dict of:
        variant_name → {
            "clf":            XGBClassifier instance,
            "use_sample_weight": bool  — whether to pass sample_weight at fit
        }

    XGBoost base hyperparameters are left at defaults:
        max_depth=6, learning_rate=0.3, min_child_weight=1,
        subsample=1.0, colsample_bytree=1.0, gamma=0
    These will be tuned in Stage 05c.

    eval_metric='logloss' suppresses the default XGBoost stdout spam.
    use_label_encoder deprecated/removed in recent XGBoost versions.
    """
    spw = round(n_neg / n_pos, 4)   # scale_pos_weight = neg / pos

    base_kwargs = dict(
        n_estimators   = n_estimators,
        random_state   = seed,
        eval_metric    = "logloss",
        verbosity      = 0,
    )

    return {
        "XGBoost (default)": {
            "clf": XGBClassifier(**base_kwargs),
            "use_sample_weight": False,
            "note": "No imbalance correction. Majority-class biased baseline.",
        },
        f"XGBoost (scale_pos_weight={spw})": {
            "clf": XGBClassifier(**base_kwargs, scale_pos_weight=spw),
            "use_sample_weight": False,
            "note": (
                f"Built-in XGBoost correction: upweights positive class (y=1) "
                f"by factor {spw} during tree construction."
            ),
        },
        "XGBoost (sample_weight)": {
            "clf": XGBClassifier(**base_kwargs),
            "use_sample_weight": True,
            "note": (
                "Per-sample weights from sklearn compute_sample_weight('balanced'). "
                "Passed to fit() — more flexible than scale_pos_weight."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 3 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    n_resistant   = int(tp + fn)
    n_susceptible = int(tn + fp)

    vme = fn / n_resistant   if n_resistant   > 0 else 0.0
    me  = fp / n_susceptible if n_susceptible > 0 else 0.0

    return {
        "ROC-AUC":       round(roc_auc_score(y_test, y_prob), 4),
        "F1 (macro)":    round(f1_score(y_test, y_pred, average="macro"), 4),
        "MCC":           round(matthews_corrcoef(y_test, y_pred), 4),
        "Sensitivity":   round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "Specificity":   round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "VME (%)":       round(vme * 100, 2),
        "ME (%)":        round(me  * 100, 2),
        "VME FDA":       "✅" if vme * 100 <= FDA_VME_THRESHOLD else "❌",
        "ME FDA":        "✅" if me  * 100 <= FDA_ME_THRESHOLD  else "❌",
        "Accuracy":      round((tp + tn) / (tp + tn + fp + fn), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def run_xgb_on_dataset(
    dataset_label: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    variants: dict,
) -> list[dict]:
    """Run all XGBoost variants on one feature set. Returns list of result dicts."""
    rows = []

    for variant_name, spec in variants.items():
        clf  = spec["clf"]
        note = spec["note"]

        log.info(f"  [{dataset_label}]  {variant_name}")

        try:
            t0 = time.perf_counter()

            fit_kwargs = {}
            if spec["use_sample_weight"]:
                fit_kwargs["sample_weight"] = compute_sample_weight(
                    "balanced", y_train
                )

            clf.fit(X_train, y_train, **fit_kwargs)
            fit_time = round(time.perf_counter() - t0, 3)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)

            rows.append({
                "Feature set":  dataset_label,
                "Variant":      variant_name,
                "n_features":   X_train.shape[1],
                "Note":         note,
                **metrics,
                "Fit time (s)": fit_time,
                "Error":        "",
            })

        except Exception as e:
            log.error(f"  ✗ {variant_name} on {dataset_label} failed: {e}")
            rows.append({
                "Feature set": dataset_label,
                "Variant":     variant_name,
                "n_features":  X_train.shape[1],
                "Error":       str(e)[:200],
                "ROC-AUC":     np.nan,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 4 — Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    results: pd.DataFrame,
    out_dir: Path,
    subset_size: float,
    seed: int,
    dataset_info: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "xgb_baseline_scores.csv"
    results.to_csv(csv_path, index=False)

    # ── Markdown ──────────────────────────────────────────────────────────────
    display_cols = [
        "Feature set", "Variant", "n_features",
        "ROC-AUC", "F1 (macro)", "MCC",
        "Sensitivity", "Specificity",
        "VME (%)", "VME FDA", "ME (%)", "ME FDA",
        "Accuracy", "Fit time (s)",
    ]
    display = results[[c for c in display_cols if c in results.columns]]

    md_lines = [
        "# Stage 05b — XGBoost Baseline Benchmark\n",
        f"**Subset:** {subset_size:.0%} of full dataset | **Seed:** {seed}\n",
        f"**FDA thresholds:** VME ≤ {FDA_VME_THRESHOLD}% | ME ≤ {FDA_ME_THRESHOLD}%\n",
        "\n## Dataset Summary\n",
    ]
    for info in dataset_info:
        md_lines.append(
            f"- **{info['label']}**: {info['n_samples']} samples × "
            f"{info['n_features']} features | "
            f"Train: {info['n_train']} | Test: {info['n_test']} | "
            f"y=1: {info['n_pos']} | y=0: {info['n_neg']}\n"
        )
    md_lines += ["\n## Results\n", display.to_markdown(index=False)]

    md_path = out_dir / "xgb_baseline_scores.md"
    md_path.write_text("\n".join(md_lines))

    # ── Console ───────────────────────────────────────────────────────────────
    valid = results.dropna(subset=["ROC-AUC"])

    print("\n" + "═" * 80)
    print("  STAGE 05b — XGBOOST BASELINE BENCHMARK")
    print(f"  Subset {subset_size:.0%} | Seed {seed}")
    print(f"  FDA thresholds: VME ≤ {FDA_VME_THRESHOLD}%  |  ME ≤ {FDA_ME_THRESHOLD}%")
    print("═" * 80)
    print(
        f"  {'Feature Set':<22} {'Variant':<38} "
        f"{'AUC':>7} {'F1':>7} {'MCC':>7} "
        f"{'VME%':>6} {'V✓':>3} {'ME%':>6} {'M✓':>3}"
    )
    print(f"  {'─'*22} {'─'*38} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*3} {'─'*6} {'─'*3}")

    for _, row in valid.iterrows():
        print(
            f"  {str(row['Feature set']):<22} {str(row['Variant']):<38} "
            f"{row['ROC-AUC']:>7.4f} {row['F1 (macro)']:>7.4f} {row['MCC']:>7.4f} "
            f"{row['VME (%)']:>6.1f} {row.get('VME FDA',''):>3} "
            f"{row['ME (%)']:>6.1f} {row.get('ME FDA',''):>3}"
        )

    print("═" * 80)

    # Best per feature set
    print("\n  Best per feature set (by ROC-AUC):\n")
    for fs in valid["Feature set"].unique():
        best = valid[valid["Feature set"] == fs].sort_values(
            "ROC-AUC", ascending=False
        ).iloc[0]
        print(
            f"  {fs:<22}  {best['Variant']:<38}\n"
            f"  {'':22}  AUC={best['ROC-AUC']:.4f} | F1={best['F1 (macro)']:.4f} | "
            f"MCC={best['MCC']:.4f} | VME={best['VME (%)']:.1f}% {best.get('VME FDA','')} | "
            f"ME={best['ME (%)']:.1f}% {best.get('ME FDA','')}\n"
        )

    overall_best = valid.sort_values("ROC-AUC", ascending=False).iloc[0]
    print("─" * 80)
    print(f"  ✅ Overall best: [{overall_best['Feature set']}]  {overall_best['Variant']}")
    print(
        f"     AUC={overall_best['ROC-AUC']:.4f} | F1={overall_best['F1 (macro)']:.4f} | "
        f"MCC={overall_best['MCC']:.4f} | "
        f"VME={overall_best['VME (%)']:.1f}% {overall_best.get('VME FDA','')} | "
        f"ME={overall_best['ME (%)']:.1f}% {overall_best.get('ME FDA','')}"
    )
    print(f"\n  Full results → {csv_path}")
    print(f"  Markdown    → {md_path}")
    print("═" * 80)
    print(
        "\n  Interpretation:\n"
        "  Compare the three imbalance strategies per feature set:\n"
        "    default          → raw majority bias, highest ME expected\n"
        "    scale_pos_weight → XGBoost-native correction, reduces VME\n"
        "    sample_weight    → sklearn-balanced, most aggressive correction\n"
        "\n  Take the best-performing feature set + strategy into Stage 05c\n"
        "  for hyperparameter tuning and full 5-fold CV on the complete dataset.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
## MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    all_rows: list[dict]  = []
    dataset_info: list[dict] = []

    for triple in args.datasets:
        label, x_path, y_path = parse_dataset_arg(triple)

        log.info(f"Loading dataset: {label}")
        X, y = load_dataset(x_path, y_path)

        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        log.info(
            f"  {label}: {X.shape[0]} samples × {X.shape[1]} features | "
            f"y=1: {n_pos} | y=0: {n_neg}"
        )

        X_train, X_test, y_train, y_test = make_subset(
            X, y, args.subset_size, args.train_size, args.seed
        )
        log.info(
            f"  Subset → Train: {len(X_train)} | Test: {len(X_test)} | "
            f"y_train R/S: {y_train.sum()}/{(y_train==0).sum()} | "
            f"y_test R/S: {y_test.sum()}/{(y_test==0).sum()}"
        )

        dataset_info.append({
            "label":      label,
            "n_samples":  X.shape[0],
            "n_features": X.shape[1],
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "n_pos":      int(y_test.sum()),
            "n_neg":      int((y_test == 0).sum()),
        })

        variants = build_xgb_variants(args.n_estimators, args.seed, n_pos, n_neg)
        rows = run_xgb_on_dataset(
            label, X_train, X_test, y_train, y_test, variants
        )
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    write_report(results, args.out_dir, args.subset_size, args.seed, dataset_info)


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE
# =============================================================================
#
#  All three feature sets:
#    python amr_geno2dock_pipeline_stage_05b.py \
#        --datasets \
#            "28 genes (5%):ml_data_5pct/X.csv:ml_data_5pct/y.csv" \
#            "69 genes (1%):ml_data/X.csv:ml_data/y.csv" \
#            "206 genes (all):ml_data_all/X.csv:ml_data_all/y.csv" \
#        --out-dir xgb_baseline
#
#  Single feature set only:
#    python amr_geno2dock_pipeline_stage_05b.py \
#        --datasets "28 genes (5%):ml_data_5pct/X.csv:ml_data_5pct/y.csv"
#
# =============================================================================
# SCALE_POS_WEIGHT EXPLAINED
# =============================================================================
#
#  XGBoost's scale_pos_weight = n_negative_class / n_positive_class
#
#  Your dataset: 228 susceptible (y=0) / 315 resistant (y=1) = 0.724
#
#  Since resistant is actually the majority class (315 > 228), scale_pos_weight
#  < 1 slightly downweights the resistant class. The bigger effect here is
#  that it makes XGBoost's internal gradient updates more sensitive to the
#  susceptible (minority) class, which tends to reduce ME.
#
#  For severely imbalanced datasets (e.g. 10:1 ratio), scale_pos_weight has
#  a much larger effect. At your 1.38:1 ratio the difference will be subtle —
#  this benchmark quantifies exactly how subtle.
#
# =============================================================================
# NEXT STEP — Stage 05c
# =============================================================================
#
#  Take the best feature set + imbalance strategy from this benchmark and
#  run full hyperparameter tuning with leakage-safe 5-fold stratified CV:
#
#    param_grid = {
#        "max_depth":        [3, 4, 5, 6],
#        "learning_rate":    [0.05, 0.1, 0.2, 0.3],
#        "n_estimators":     [100, 200, 300],
#        "min_child_weight": [1, 3, 5],
#        "subsample":        [0.7, 0.8, 1.0],
#        "colsample_bytree": [0.7, 0.8, 1.0],
#        "gamma":            [0, 0.1, 0.2],
#    }
#
#  Then compute SHAP values on the tuned model for Stage 06.
# =============================================================================