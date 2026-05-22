#!/usr/bin/env python3
# coding: utf-8
# AUTHOR: Akshay Shirsath (2026)
# =============================================================================
# AMR-Geno2Dock — Baseline Classifier Benchmark (Stage 05a)
# =============================================================================
#
# PURPOSE:
#   Run every major sklearn classifier at default hyperparameters on a
#   stratified subset of the Stage 04 feature matrix to get an honest
#   baseline before any tuning. This informs which model family to carry
#   forward into Stage 05b (tuned training + SHAP).
#
# DESIGN PRINCIPLES:
#   — Subset, not full data: baselines on the full set inflate apparent
#     performance and mask overfitting. A held-out stratified subset
#     gives a cleaner picture of raw model capacity.
#   — No tuning: default hyperparameters only. The goal is model family
#     selection, not optimisation.
#   — Stratified split: class ratio (315R / 228S) preserved in both
#     train and test portions of the subset.
#   — Class-imbalance aware metrics: accuracy alone is misleading at
#     ratio 0.724. Reports ROC-AUC, F1 (macro), MCC, and clinical
#     metrics VME / ME as primary ranking criteria.
#   — Leakage-free: no preprocessing fitted on the full dataset.
#     All scalers fitted inside the subset train split only.
#
# INPUT  (from Stage 04):
#   ml_data_5pct/X.csv   — feature matrix (543 × 28)
#   ml_data_5pct/y.csv   — binary labels
#
# OUTPUT:
#   baseline_results/
#   ├── baseline_scores.csv      — full metric table, all classifiers
#   ├── baseline_scores.md       — markdown-formatted ranked table
#   └── baseline_benchmark.log   — run log
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
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
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
        prog="amr_geno2dock_pipeline_stage_05a.py",
        description=(
            "Stage 05a — Benchmark all major sklearn classifiers at default "
            "hyperparameters on a stratified subset of the Stage 04 feature matrix."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--X", required=True, type=Path, metavar="CSV",
        help="Feature matrix CSV from Stage 04 (index_col=0).",
    )
    parser.add_argument(
        "--y", required=True, type=Path, metavar="CSV",
        help="Binary label CSV from Stage 04 (index_col=0).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("baseline_results"), metavar="DIR",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--subset-size", type=float, default=0.60, metavar="FLOAT",
        help=(
            "Fraction of the full dataset to use as the benchmark subset. "
            "This subset is then split 70/30 train/test internally. "
            "Default 0.60 = 60%% of 543 ≈ 326 samples."
        ),
    )
    parser.add_argument(
        "--train-size", type=float, default=0.70, metavar="FLOAT",
        help="Train fraction within the subset (remainder = test). Default 0.70.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="INT",
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, metavar="INT",
        help="Number of top classifiers to highlight in the summary.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 1 — Classifier registry
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier_registry(seed: int) -> dict[str, object]:
    """
    All major sklearn classifier families at default hyperparameters.

    Organised into groups:
      Linear          — LR, Ridge, SGD, Perceptron, PassiveAggressive
      SVM             — SVC (RBF), LinearSVC
      Tree            — DecisionTree, ExtraTree (single)
      Ensemble        — RandomForest, ExtraTrees, GradientBoosting,
                        AdaBoost, Bagging
      Neighbours      — KNN (k=5 default)
      Naive Bayes     — Gaussian, Bernoulli, Complement, Multinomial
      Discriminant    — LDA, QDA
      Neural net      — MLP (single hidden layer default)
      Gaussian Proc   — GaussianProcess (expensive; skipped for large n)
      Dummy           — majority-class baseline

    Classifiers that require non-negative features (MultinomialNB) work
    fine here because the feature matrix is binary 0/1.

    Classifiers that need scaling (SVM, LR, MLP, etc.) are wrapped in
    a Pipeline with StandardScaler fitted on the train split only —
    this is the correct leakage-free approach.
    """

    # Shorthand wrappers
    def scaled(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    registry = {

        # ── Dummy baseline ────────────────────────────────────────────────────
        "Dummy (majority)": DummyClassifier(
            strategy="most_frequent", random_state=seed
        ),
        "Dummy (stratified)": DummyClassifier(
            strategy="stratified", random_state=seed
        ),

        # ── Linear ────────────────────────────────────────────────────────────
        "Logistic Regression (L2)": scaled(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            )
        ),
        "Logistic Regression (L1)": scaled(
            LogisticRegression(
                penalty="l1", solver="liblinear",
                max_iter=1000, class_weight="balanced", random_state=seed
            )
        ),
        "Ridge Classifier": scaled(
            RidgeClassifier(class_weight="balanced")
        ),
        "SGD Classifier": scaled(
            SGDClassifier(
                loss="hinge", class_weight="balanced",
                max_iter=1000, random_state=seed
            )
        ),
        "Perceptron": scaled(
            Perceptron(class_weight="balanced", random_state=seed)
        ),
        "Passive Aggressive": scaled(
            PassiveAggressiveClassifier(
                class_weight="balanced", random_state=seed, max_iter=1000
            )
        ),

        # ── SVM ───────────────────────────────────────────────────────────────
        "SVM (RBF)": scaled(
            SVC(kernel="rbf", class_weight="balanced",
                probability=True, random_state=seed)
        ),
        "SVM (Linear)": scaled(
            SVC(kernel="linear", class_weight="balanced",
                probability=True, random_state=seed)
        ),
        "SVM (Poly)": scaled(
            SVC(kernel="poly", class_weight="balanced",
                probability=True, random_state=seed)
        ),
        "LinearSVC": scaled(
            LinearSVC(class_weight="balanced", max_iter=2000, random_state=seed)
        ),

        # ── Tree ──────────────────────────────────────────────────────────────
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=seed
        ),
        "Extra Tree (single)": ExtraTreeClassifier(
            class_weight="balanced", random_state=seed
        ),

        # ── Ensemble ──────────────────────────────────────────────────────────
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=seed
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=seed
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=seed
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=seed
        ),
        "Bagging": BaggingClassifier(
            n_estimators=100, n_jobs=-1, random_state=seed
        ),

        # ── Neighbours ────────────────────────────────────────────────────────
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "KNN (k=9)": KNeighborsClassifier(n_neighbors=9, n_jobs=-1),

        # ── Naive Bayes ───────────────────────────────────────────────────────
        "Gaussian NB": GaussianNB(),
        "Bernoulli NB": BernoulliNB(),
        "Complement NB": ComplementNB(),
        "Multinomial NB": MultinomialNB(),

        # ── Discriminant analysis ─────────────────────────────────────────────
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),

        # ── Neural network ────────────────────────────────────────────────────
        "MLP (100)": scaled(
            MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=500, random_state=seed
            )
        ),
        "MLP (100, 50)": scaled(
            MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed
            )
        ),
    }

    return registry


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 2 — Stratified subset split
# ─────────────────────────────────────────────────────────────────────────────

def make_subset(
    X: pd.DataFrame,
    y: pd.Series,
    subset_size: float,
    train_size: float,
    seed: int,
) -> tuple:
    """
    1. Draw a stratified subset of `subset_size` from the full dataset.
    2. Split that subset into train / test at `train_size`.

    Both splits preserve the class ratio (stratified).

    Returns: X_train, X_test, y_train, y_test, subset_indices
    """
    n_full   = len(X)
    n_subset = int(n_full * subset_size)

    # Step 1: sample the subset
    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - subset_size, random_state=seed
    )
    subset_idx, _ = next(sss_outer.split(X, y))
    X_sub = X.iloc[subset_idx]
    y_sub = y.iloc[subset_idx]

    # Step 2: train/test split within subset
    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - train_size, random_state=seed
    )
    train_idx, test_idx = next(sss_inner.split(X_sub, y_sub))
    X_train = X_sub.iloc[train_idx]
    X_test  = X_sub.iloc[test_idx]
    y_train = y_sub.iloc[train_idx]
    y_test  = y_sub.iloc[test_idx]

    log.info(
        f"Subset: {n_subset}/{n_full} samples "
        f"| Train: {len(X_train)} | Test: {len(X_test)} "
        f"| y_train R/S: {y_train.sum()}/{(y_train==0).sum()} "
        f"| y_test R/S: {y_test.sum()}/{(y_test==0).sum()}"
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 3 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict:
    """
    Compute the full metric set for one classifier.

    Metrics:
        ROC-AUC   — primary ranking metric; threshold-free, imbalance-robust
        F1 macro  — average F1 across both classes; penalises ignoring minority
        MCC       — Matthews correlation coefficient; gold standard for imbalanced binary
        Accuracy  — included for completeness but not used for ranking
        VME       — Very Major Error rate: predicted susceptible, actually resistant
                    (clinically most dangerous — missed resistance)
        ME        — Major Error rate: predicted resistant, actually susceptible
                    (clinically wasteful — unnecessary treatment escalation)
        Sensitivity — recall for resistant class (y=1)
        Specificity — recall for susceptible class (y=0)
        Fit time  — added by caller
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    n_resistant    = int(tp + fn)
    n_susceptible  = int(tn + fp)

    vme = fn / n_resistant   if n_resistant   > 0 else 0.0
    me  = fp / n_susceptible if n_susceptible > 0 else 0.0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc = (
        roc_auc_score(y_test, y_prob) if y_prob is not None
        else roc_auc_score(y_test, y_pred)
    )

    return {
        "ROC-AUC":     round(roc_auc, 4),
        "F1 (macro)":  round(f1_score(y_test, y_pred, average="macro"), 4),
        "MCC":         round(matthews_corrcoef(y_test, y_pred), 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "VME (%)":     round(vme * 100, 2),
        "ME (%)":      round(me  * 100, 2),
        "Accuracy":    round((tp + tn) / (tp + tn + fp + fn), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def benchmark(
    registry: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Fit every classifier in the registry and score on the test split.
    Returns a DataFrame of results sorted by ROC-AUC descending.
    """
    rows = []
    n = len(registry)

    for i, (name, clf) in enumerate(registry.items(), 1):
        log.info(f"[{i:02d}/{n}]  {name} …")
        row = {"Classifier": name}

        try:
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_time = round(time.perf_counter() - t0, 3)

            y_pred = clf.predict(X_test)

            # Get probability scores where available for ROC-AUC
            y_prob = None
            if hasattr(clf, "predict_proba"):
                try:
                    y_prob = clf.predict_proba(X_test)[:, 1]
                except Exception:
                    pass
            elif hasattr(clf, "decision_function"):
                try:
                    y_prob = clf.decision_function(X_test)
                except Exception:
                    pass

            metrics = compute_metrics(y_test, y_pred, y_prob)
            row.update(metrics)
            row["Fit time (s)"] = fit_time
            row["Error"] = ""

        except Exception as e:
            log.warning(f"  ✗ {name} failed: {e}")
            row["Error"] = str(e)[:120]
            row["ROC-AUC"] = np.nan

        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results.index += 1   # rank from 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 4 — Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    results: pd.DataFrame,
    out_dir: Path,
    top_n: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    subset_size: float,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / "baseline_scores.csv"
    results.to_csv(csv_path)

    # ── Markdown table ────────────────────────────────────────────────────────
    display_cols = [
        "Classifier", "ROC-AUC", "F1 (macro)", "MCC",
        "Sensitivity", "Specificity", "VME (%)", "ME (%)",
        "Accuracy", "Fit time (s)",
    ]
    display = results[[c for c in display_cols if c in results.columns]].copy()
    display.index.name = "Rank"

    md_lines = ["# Stage 05a — Baseline Classifier Benchmark\n"]
    md_lines.append(
        f"**Subset:** {subset_size:.0%} of full dataset  "
        f"| **Train:** {len(X_train)}  "
        f"| **Test:** {len(X_test)}  "
        f"| **Seed:** {seed}\n"
    )
    md_lines.append(
        f"**Test set:** y=1 (resistant): {y_test.sum()} "
        f"| y=0 (susceptible): {(y_test==0).sum()}\n"
    )
    md_lines.append(
        "> Ranked by ROC-AUC. VME = Very Major Error (missed resistance); "
        "ME = Major Error (false resistance call).\n"
    )
    md_lines.append(display.to_markdown())

    md_path = out_dir / "baseline_scores.md"
    md_path.write_text("\n".join(md_lines))

    # ── Console summary ───────────────────────────────────────────────────────
    valid = results.dropna(subset=["ROC-AUC"])
    top   = valid.head(top_n)

    print("\n" + "═" * 72)
    print("  STAGE 05a — BASELINE BENCHMARK RESULTS")
    print(f"  Subset {subset_size:.0%} | Train {len(X_train)} | Test {len(X_test)} | Seed {seed}")
    print("═" * 72)
    print(f"  {'Rank':<5} {'Classifier':<35} {'AUC':>7} {'F1':>7} {'MCC':>7} {'VME%':>6} {'ME%':>6}")
    print(f"  {'─'*5} {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")

    for rank, row in top.iterrows():
        print(
            f"  {rank:<5} {row['Classifier']:<35} "
            f"{row.get('ROC-AUC', float('nan')):>7.4f} "
            f"{row.get('F1 (macro)', float('nan')):>7.4f} "
            f"{row.get('MCC', float('nan')):>7.4f} "
            f"{row.get('VME (%)', float('nan')):>6.1f} "
            f"{row.get('ME (%)', float('nan')):>6.1f}"
        )

    print("═" * 72)
    best = valid.iloc[0]
    print(f"\n  ✅ Best baseline: {best['Classifier']}")
    print(f"     ROC-AUC={best['ROC-AUC']:.4f} | F1={best['F1 (macro)']:.4f} | MCC={best['MCC']:.4f}")
    print(f"     VME={best['VME (%)']:.1f}%  ME={best['ME (%)']:.1f}%")
    print(f"\n  Full results → {csv_path}")
    print(f"  Markdown    → {md_path}")
    print("═" * 72)
    print(
        "\n  Next step (Stage 05b):\n"
        "    Take the top 2–3 model families from this table and run\n"
        "    leakage-safe 5-fold stratified cross-validation with tuning\n"
        "    on the FULL dataset, then compute SHAP values for Stage 06.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
## MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading X from {args.X}")
    X = pd.read_csv(args.X, index_col=0)
    log.info(f"Loading y from {args.y}")
    y = pd.read_csv(args.y, index_col=0).squeeze()

    if len(X) != len(y):
        log.error(f"X rows ({len(X)}) ≠ y rows ({len(y)})")
        sys.exit(1)

    log.info(f"Full dataset: {X.shape[0]} samples × {X.shape[1]} features")
    log.info(f"Label balance: y=1: {y.sum()} | y=0: {(y==0).sum()}")

    # ── Phase 2: Stratified subset ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = make_subset(
        X, y, args.subset_size, args.train_size, args.seed
    )

    # ── Phase 1: Build classifier registry ───────────────────────────────────
    registry = build_classifier_registry(args.seed)
    log.info(f"Classifiers to benchmark: {len(registry)}")

    # ── Phase 3: Benchmark ────────────────────────────────────────────────────
    results = benchmark(registry, X_train, X_test, y_train, y_test)

    # ── Phase 4: Report ───────────────────────────────────────────────────────
    write_report(
        results, args.out_dir, args.top_n,
        X_train, X_test, y_train, y_test,
        args.subset_size, args.seed,
    )


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE
# =============================================================================
#
#  Standard run (60% subset, default seed):
#    python amr_geno2dock_pipeline_stage_05a.py \
#        --X ml_data_5pct/X.csv \
#        --y ml_data_5pct/y.csv \
#        --out-dir baseline_results
#
#  Larger subset (80%):
#    python amr_geno2dock_pipeline_stage_05a.py \
#        --X ml_data_5pct/X.csv \
#        --y ml_data_5pct/y.csv \
#        --subset-size 0.80
#
#  Different seed (reproducibility check):
#    python amr_geno2dock_pipeline_stage_05a.py \
#        --X ml_data_5pct/X.csv \
#        --y ml_data_5pct/y.csv \
#        --seed 123
#
# =============================================================================
# METRIC GUIDE
# =============================================================================
#
#  ROC-AUC   Primary ranking metric. Threshold-free; measures how well the
#            model separates resistant from susceptible across all cutoffs.
#            >0.90 = excellent | 0.80–0.90 = good | <0.70 = poor baseline
#
#  F1 macro  Average F1 across both classes. Penalises models that ignore
#            the minority class (susceptible). Use alongside AUC.
#
#  MCC       Matthews Correlation Coefficient. The single most reliable
#            binary metric under class imbalance. Range: -1 to +1.
#            >0.60 = strong | 0.40–0.60 = moderate | <0.20 = poor
#
#  VME (%)   Very Major Error. Predicted susceptible when actually resistant.
#            Clinically most dangerous outcome (missed resistance → treatment
#            failure). FDA threshold: ≤7.5% acceptable.
#
#  ME (%)    Major Error. Predicted resistant when actually susceptible.
#            Clinically wasteful (unnecessary escalation). FDA threshold: ≤3%.
#
#  Sensitivity  Recall for resistant class. Prioritise if VME reduction is goal.
#  Specificity  Recall for susceptible class. Prioritise if ME reduction is goal.
#
# =============================================================================