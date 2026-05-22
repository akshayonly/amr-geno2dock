#!/usr/bin/env python3
# coding: utf-8
# AUTHOR: Akshay Shirsath (2026)
# =============================================================================
# AMR-Geno2Dock — Baseline Classifier Benchmark (Stage 05a)
# Multi-Antibiotic Edition
# =============================================================================
#
# PURPOSE:
#   Run every major sklearn classifier at default hyperparameters on a
#   stratified subset of the Stage 04 feature matrix to get an honest
#   baseline before any tuning. This informs which model family to carry
#   forward into Stage 05b (tuned training + SHAP).
#
#   This version supports a multi-antibiotic model_data CSV (with columns
#   Assembly_Accession, Antibiotic, binary_class) merged against an AMR
#   gene presence/absence matrix (X.csv).
#
# DESIGN PRINCIPLES:
#   — Held-out test set carved FIRST (Stage 00 pattern): 20% of full
#     dataset is locked away before any modelling. The benchmark runs
#     only on the remaining 80% training pool, using an internal 60/40
#     stratified subset split for speed.
#   — No tuning: default hyperparameters only. Goal is model-family
#     selection, not optimisation.
#   — Stratified throughout: class ratio preserved at every split level
#     (full → held-out, pool → subset, subset → train/test).
#   — Antibiotic-aware: per-antibiotic breakdown reported alongside the
#     pooled result. Amikacin imbalance (ratio ~0.15) is flagged.
#   — Class-imbalance aware metrics: ROC-AUC, F1 macro, MCC, VME, ME
#     as primary ranking criteria. Accuracy intentionally de-emphasised.
#   — Leakage-free: scalers fitted on subset train split only.
#
# DATA STRUCTURE (multi-antibiotic):
#   model_data CSV  — columns: Assembly_Accession, Antibiotic, binary_class
#                     binary_class values: 'susceptible' | 'non-susceptible'
#   X CSV           — gene presence/absence matrix; index = Assembly_Accession
#                     (from Stage 04, e.g. ml_data/X.csv)
#
# LABEL ENCODING:
#   non-susceptible → 1  (positive class; "resistant" in clinical sense)
#   susceptible     → 0
#
# INPUT:
#   --X            path to AMR feature matrix CSV (Assembly_Accession as index)
#   --model-data   path to phenotype CSV (Assembly_Accession, Antibiotic,
#                  binary_class columns required)
#
# OUTPUT:
#   baseline_results/
#   ├── baseline_scores_pooled.csv       — full metric table, all classifiers
#   ├── baseline_scores_pooled.md        — markdown-formatted ranked table
#   ├── baseline_scores_per_antibiotic/
#   │   ├── baseline_scores_amikacin.csv
#   │   ├── baseline_scores_gentamicin.csv
#   │   └── baseline_scores_tobramycin.csv
#   ├── held_out_test_accessions.txt     — 20% held-out accessions (LOCKED)
#   ├── train_pool_accessions.txt        — 80% training pool accessions
#   └── baseline_benchmark.log          — run log
#
# USAGE:
#   python amr_geno2dock_pipeline_stage_05a.py \
#       --X ml_data/X.csv \
#       --model-data model_data_multiab.csv \
#       --out-dir baseline_results
#
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

# Label encoding — used throughout
LABEL_MAP   = {"non-susceptible": 1, "susceptible": 0}
LABEL_NAMES = {1: "non-susceptible", 0: "susceptible"}


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0B — CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amr_geno2dock_pipeline_stage_05a.py",
        description=(
            "Stage 05a (Multi-Antibiotic) — Benchmark all major sklearn classifiers "
            "at default hyperparameters on a stratified subset of the merged "
            "phenotype + AMR gene presence/absence matrix."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--X", required=True, type=Path, metavar="CSV",
        help=(
            "AMR gene presence/absence matrix CSV from Stage 04. "
            "Index column must be Assembly_Accession."
        ),
    )
    parser.add_argument(
        "--model-data", required=True, type=Path, metavar="CSV",
        help=(
            "Phenotype CSV with columns: Assembly_Accession, Antibiotic, binary_class. "
            "binary_class values: 'susceptible' | 'non-susceptible'."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("baseline_results"), metavar="DIR",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--held-out-frac", type=float, default=0.20, metavar="FLOAT",
        help=(
            "Fraction of the full dataset to lock as held-out test set. "
            "These samples are saved to disk and NEVER used for benchmarking. "
            "Default 0.20."
        ),
    )
    parser.add_argument(
        "--subset-size", type=float, default=0.60, metavar="FLOAT",
        help=(
            "Fraction of the TRAINING POOL (post held-out removal) to use as "
            "the benchmark subset. Subset is split 70/30 train/test internally. "
            "Default 0.60."
        ),
    )
    parser.add_argument(
        "--train-size", type=float, default=0.70, metavar="FLOAT",
        help="Train fraction within the benchmark subset. Default 0.70.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="INT",
        help="Random seed for all splits. Default 42.",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, metavar="INT",
        help="Number of top classifiers to highlight in the summary. Default 10.",
    )
    parser.add_argument(
        "--skip-per-antibiotic", action="store_true",
        help="Skip per-antibiotic breakdown (faster run for quick checks).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 1 — Data loading & merging
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(
    X_path: Path,
    model_data_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the AMR feature matrix and phenotype table; merge on Assembly_Accession.

    Multi-antibiotic handling:
      — model_data may have multiple rows per Assembly_Accession (one per
        antibiotic). For the POOLED model the accession must appear once in X.
      — Deduplication strategy: if one accession has the same binary_class
        across all its antibiotic rows → keep one row (consistent label).
        If conflicting binary_class values exist across antibiotics for the
        same accession → drop the accession (conservative; avoids label noise).
      — The Antibiotic column is retained in model_data_clean for use in
        per-antibiotic breakdowns after the split.

    Returns:
        X_merged    — feature matrix, one row per accession, index = accession
        y_merged    — integer label Series (1=non-susceptible, 0=susceptible)
        antibiotic  — Series of antibiotic name per accession (for stratified
                      per-antibiotic reporting; same index as X_merged)
    """
    log.info(f"Loading feature matrix from {X_path}")
    X = pd.read_csv(X_path, index_col=0)
    X.index = X.index.astype(str)
    log.info(f"  → {X.shape[0]} samples × {X.shape[1]} genes")

    log.info(f"Loading phenotype data from {model_data_path}")
    md = pd.read_csv(model_data_path)

    # Validate required columns
    required = {"Assembly_Accession", "Antibiotic", "binary_class"}
    missing  = required - set(md.columns)
    if missing:
        log.error(f"model_data missing columns: {missing}")
        sys.exit(1)

    md["Assembly_Accession"] = md["Assembly_Accession"].astype(str)
    md["y"] = md["binary_class"].map(LABEL_MAP)

    unmapped = md["y"].isna().sum()
    if unmapped > 0:
        log.warning(
            f"{unmapped} rows with unrecognised binary_class values "
            f"(expected 'susceptible' | 'non-susceptible') — dropped."
        )
        md = md.dropna(subset=["y"])
    md["y"] = md["y"].astype(int)

    # ── Resolve multi-antibiotic rows per accession ───────────────────────────
    # Group by accession; check label consistency across antibiotics
    label_check = md.groupby("Assembly_Accession")["y"].nunique()
    conflicting  = label_check[label_check > 1].index.tolist()

    if conflicting:
        log.warning(
            f"{len(conflicting)} accessions have conflicting binary_class across "
            f"antibiotics → dropped (conservative label integrity policy)."
        )
        md = md[~md["Assembly_Accession"].isin(conflicting)]

    # Keep one representative row per accession (label now guaranteed consistent)
    # Retain the Antibiotic column as the "primary antibiotic" for that genome
    # (first appearing antibiotic per accession after sorting for reproducibility)
    md_dedup = (
        md.sort_values("Antibiotic")
          .drop_duplicates(subset="Assembly_Accession", keep="first")
          .set_index("Assembly_Accession")
    )

    # ── Inner join with feature matrix ───────────────────────────────────────
    common = X.index.intersection(md_dedup.index)
    if len(common) == 0:
        log.error(
            "No Assembly_Accession values overlap between X and model_data. "
            "Check that index column in X matches Assembly_Accession in model_data."
        )
        sys.exit(1)

    n_x_only   = len(X.index) - len(common)
    n_md_only  = len(md_dedup.index) - len(common)
    if n_x_only > 0:
        log.warning(f"  {n_x_only} accessions in X but not in model_data → excluded")
    if n_md_only > 0:
        log.warning(f"  {n_md_only} accessions in model_data but not in X → excluded")

    X_merged   = X.loc[common]
    y_merged   = md_dedup.loc[common, "y"]
    antibiotic = md_dedup.loc[common, "Antibiotic"]

    log.info(
        f"Merged dataset: {len(X_merged)} samples × {X_merged.shape[1]} genes\n"
        f"  Label distribution:\n"
        f"    non-susceptible (1): {(y_merged==1).sum()}\n"
        f"    susceptible     (0): {(y_merged==0).sum()}\n"
        f"  Antibiotic counts:\n"
        + "\n".join(
            f"    {ab}: {n}" for ab, n in antibiotic.value_counts().items()
        )
    )

    ratio = min((y_merged==1).sum(), (y_merged==0).sum()) / \
            max((y_merged==1).sum(), (y_merged==0).sum())
    flag = "✅ balanced" if ratio >= 0.75 else "⚠ moderate" if ratio >= 0.50 else "⚠ imbalanced"
    log.info(f"  Binary ratio: {ratio:.3f}  {flag}")

    return X_merged, y_merged, antibiotic


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 2 — Held-out split + benchmark subset
# ─────────────────────────────────────────────────────────────────────────────

def carve_held_out(
    X: pd.DataFrame,
    y: pd.Series,
    held_out_frac: float,
    seed: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Carve a stratified held-out test set from the full dataset.
    Save accession lists to disk — these samples must not be used for
    any benchmarking or tuning.

    Returns: X_pool, y_pool, X_held, y_held
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=held_out_frac, random_state=seed
    )
    pool_idx, held_idx = next(sss.split(X, y))

    X_pool = X.iloc[pool_idx]
    y_pool = y.iloc[pool_idx]
    X_held = X.iloc[held_idx]
    y_held = y.iloc[held_idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "train_pool_accessions.txt"
    held_path = out_dir / "held_out_test_accessions.txt"

    pd.Series(X_pool.index).to_csv(pool_path, index=False, header=False)
    pd.Series(X_held.index).to_csv(held_path, index=False, header=False)

    log.info(
        f"Held-out split:\n"
        f"  Training pool  : {len(X_pool)} samples → {pool_path}\n"
        f"  Held-out test  : {len(X_held)} samples → {held_path}  ⚠ LOCKED\n"
        f"  Pool balance   : non-susc={( y_pool==1).sum()}  susc={(y_pool==0).sum()}\n"
        f"  Held balance   : non-susc={(y_held==1).sum()}  susc={(y_held==0).sum()}"
    )

    return X_pool, y_pool, X_held, y_held


def make_subset(
    X: pd.DataFrame,
    y: pd.Series,
    subset_size: float,
    train_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    From the training pool:
      1. Draw a stratified subset of `subset_size`.
      2. Split that subset 70/30 (default) train/test.
    Both steps preserve class ratio.
    """
    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - subset_size, random_state=seed
    )
    sub_idx, _ = next(sss_outer.split(X, y))
    X_sub = X.iloc[sub_idx]
    y_sub = y.iloc[sub_idx]

    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - train_size, random_state=seed
    )
    tr_idx, te_idx = next(sss_inner.split(X_sub, y_sub))

    X_train = X_sub.iloc[tr_idx]
    X_test  = X_sub.iloc[te_idx]
    y_train = y_sub.iloc[tr_idx]
    y_test  = y_sub.iloc[te_idx]

    log.info(
        f"Benchmark subset:\n"
        f"  Subset  : {len(X_sub)}/{len(X)} samples from training pool\n"
        f"  Train   : {len(X_train)}  "
        f"(non-susc={y_train.sum()} | susc={(y_train==0).sum()})\n"
        f"  Test    : {len(X_test)}  "
        f"(non-susc={y_test.sum()} | susc={(y_test==0).sum()})"
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 3 — Classifier registry
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier_registry(seed: int) -> dict:
    """
    All major sklearn classifier families at default hyperparameters.

    Organised into groups:
      Linear          — LR (L1/L2), Ridge, SGD, Perceptron, PassiveAggressive
      SVM             — SVC (RBF/Linear/Poly), LinearSVC
      Tree            — DecisionTree, ExtraTree (single)
      Ensemble        — RandomForest, ExtraTrees, GradientBoosting,
                        AdaBoost, Bagging
      Neighbours      — KNN (k=3/5/9)
      Naive Bayes     — Gaussian, Bernoulli, Complement, Multinomial
      Discriminant    — LDA, QDA
      Neural net      — MLP (single and dual hidden layer)
      Dummy           — majority-class and stratified baselines

    Classifiers needing scaling (SVM, LR, MLP, etc.) are wrapped in a
    Pipeline with StandardScaler fitted on the train split only.
    Feature matrix is binary 0/1 so MultinomialNB and BernoulliNB are valid.

    scale_pos_weight is NOT set here (default baselines only).
    Class imbalance is handled via class_weight='balanced' where supported.
    """

    def scaled(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    registry = {

        # ── Dummy baselines ───────────────────────────────────────────────────
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
## PHASE 4 — Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict:
    """
    Full metric set for one classifier on one test split.

    Positive class (y=1): non-susceptible
    Negative class (y=0): susceptible

    VME (Very Major Error):
        Predicted susceptible when actually non-susceptible.
        Clinically most dangerous — missed resistance → treatment failure.
        FDA software threshold: ≤1.5%.

    ME (Major Error):
        Predicted non-susceptible when actually susceptible.
        Clinically wasteful — unnecessary antibiotic escalation.
        FDA software threshold: ≤3.0%.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    n_nonsus  = int(tp + fn)   # true non-susceptible count
    n_sus     = int(tn + fp)   # true susceptible count

    vme = fn / n_nonsus if n_nonsus > 0 else 0.0
    me  = fp / n_sus    if n_sus    > 0 else 0.0

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


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 5 — Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(
    registry: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label: str = "pooled",
) -> pd.DataFrame:
    """
    Fit every classifier in the registry on X_train / y_train.
    Score on X_test / y_test.
    Returns a DataFrame of results sorted by ROC-AUC descending.

    `label` is used only for logging (e.g. 'pooled', 'gentamicin').
    """
    rows = []
    n    = len(registry)

    log.info(f"Benchmarking {n} classifiers [{label}] …")

    for i, (name, clf) in enumerate(registry.items(), 1):
        log.info(f"  [{i:02d}/{n}]  {name}")
        row = {"Classifier": name}

        try:
            t0       = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_time = round(time.perf_counter() - t0, 3)

            y_pred = clf.predict(X_test)

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
            row["Error"]        = ""

        except Exception as e:
            log.warning(f"  ✗ {name} failed: {e}")
            row["Error"]   = str(e)[:120]
            row["ROC-AUC"] = np.nan

        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results.index += 1  # rank from 1
    return results


def benchmark_per_antibiotic(
    registry: dict,
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    antibiotic_pool: pd.Series,
    subset_size: float,
    train_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """
    For each unique antibiotic in the training pool:
      — Subset the pool to that antibiotic's accessions
      — Run the full benchmark on a stratified subset

    Returns a dict: {antibiotic_name: results_DataFrame}

    ⚠ Amikacin warning: ratio ~0.15 (severely imbalanced).
    The per-antibiotic benchmark will flag this but still run —
    results should be interpreted with caution.
    """
    ab_results = {}
    antibiotics = antibiotic_pool.unique()

    for ab in sorted(antibiotics):
        ab_idx = antibiotic_pool[antibiotic_pool == ab].index
        X_ab   = X_pool.loc[ab_idx]
        y_ab   = y_pool.loc[ab_idx]

        n_nonsus = (y_ab == 1).sum()
        n_sus    = (y_ab == 0).sum()
        ratio    = min(n_nonsus, n_sus) / max(n_nonsus, n_sus) if max(n_nonsus, n_sus) > 0 else 0
        flag     = "✅" if ratio >= 0.75 else "⚠" if ratio >= 0.50 else "⚠⚠ SEVERE"

        log.info(
            f"\n{'─'*60}\n"
            f"  Per-antibiotic benchmark: {ab.upper()}\n"
            f"  Samples: {len(X_ab)}  "
            f"non-susc={n_nonsus}  susc={n_sus}  ratio={ratio:.3f}  {flag}\n"
            f"{'─'*60}"
        )

        if len(X_ab) < 30:
            log.warning(f"  {ab}: only {len(X_ab)} samples — skipping (too few for split).")
            continue

        if ratio < 0.20:
            log.warning(
                f"  {ab}: severe class imbalance (ratio={ratio:.3f}). "
                f"Results will be biased — interpret with caution."
            )

        try:
            X_tr, X_te, y_tr, y_te = make_subset(X_ab, y_ab, subset_size, train_size, seed)
            # Rebuild registry each time (clones unfitted estimators)
            reg_ab = build_classifier_registry(seed)
            ab_results[ab] = benchmark(reg_ab, X_tr, X_te, y_tr, y_te, label=ab)
        except Exception as e:
            log.warning(f"  {ab}: benchmark failed — {e}")

    return ab_results


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 6 — Reports
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_COLS = [
    "Classifier", "ROC-AUC", "F1 (macro)", "MCC",
    "Sensitivity", "Specificity", "VME (%)", "ME (%)",
    "Accuracy", "Fit time (s)",
]


def _make_markdown(
    results: pd.DataFrame,
    title: str,
    n_train: int,
    n_test: int,
    y_test: pd.Series,
    subset_size: float,
    seed: int,
    extra_note: str = "",
) -> str:
    display = results[[c for c in DISPLAY_COLS if c in results.columns]].copy()
    display.index.name = "Rank"

    lines = [f"# {title}\n"]
    lines.append(
        f"**Subset:** {subset_size:.0%} of training pool  "
        f"| **Train:** {n_train}  "
        f"| **Test:** {n_test}  "
        f"| **Seed:** {seed}\n"
    )
    lines.append(
        f"**Test set:** non-susceptible (1): {(y_test==1).sum()} "
        f"| susceptible (0): {(y_test==0).sum()}\n"
    )
    if extra_note:
        lines.append(f"> ⚠ {extra_note}\n")
    lines.append(
        "> Ranked by ROC-AUC. "
        "VME = Very Major Error (missed non-susceptibility); "
        "ME = Major Error (false non-susceptibility call). "
        "FDA thresholds: VME ≤1.5%, ME ≤3.0%.\n"
    )
    lines.append(display.to_markdown())
    return "\n".join(lines)


def write_report(
    results: pd.DataFrame,
    ab_results: dict[str, pd.DataFrame],
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

    # ── Pooled CSV ────────────────────────────────────────────────────────────
    csv_path = out_dir / "baseline_scores_pooled.csv"
    results.to_csv(csv_path)

    # ── Pooled Markdown ───────────────────────────────────────────────────────
    md_text = _make_markdown(
        results,
        title="Stage 05a — Baseline Classifier Benchmark (Pooled: All Antibiotics)",
        n_train=len(X_train), n_test=len(X_test),
        y_test=y_test,
        subset_size=subset_size, seed=seed,
    )
    md_path = out_dir / "baseline_scores_pooled.md"
    md_path.write_text(md_text)

    # ── Per-antibiotic CSVs + Markdowns ───────────────────────────────────────
    if ab_results:
        ab_dir = out_dir / "baseline_scores_per_antibiotic"
        ab_dir.mkdir(exist_ok=True)

        for ab, ab_res in ab_results.items():
            ab_csv = ab_dir / f"baseline_scores_{ab}.csv"
            ab_res.to_csv(ab_csv)

            # Reconstruct y_test for this antibiotic from stored results
            # (n_test inferred from TP+TN+FP+FN of top row)
            top_row = ab_res.dropna(subset=["ROC-AUC"]).iloc[0]
            n_test_ab  = int(top_row["TP"] + top_row["TN"] + top_row["FP"] + top_row["FN"])
            n_train_ab = int(len(X_train) * subset_size * (1 - subset_size) * 0.7)  # approx

            # Build a minimal y_test proxy for the markdown header counts
            y_test_proxy = pd.Series(
                [1] * int(top_row["TP"] + top_row["FN"]) +
                [0] * int(top_row["TN"] + top_row["FP"])
            )

            note = (
                "Amikacin is severely imbalanced (non-susc ratio ~0.15). "
                "AUC and F1 are unreliable — use with caution."
                if ab == "amikacin" else ""
            )

            ab_md = _make_markdown(
                ab_res,
                title=f"Stage 05a — Baseline Benchmark: {ab.capitalize()}",
                n_train=n_train_ab, n_test=n_test_ab,
                y_test=y_test_proxy,
                subset_size=subset_size, seed=seed,
                extra_note=note,
            )
            (ab_dir / f"baseline_scores_{ab}.md").write_text(ab_md)

    # ── Console summary ───────────────────────────────────────────────────────
    valid = results.dropna(subset=["ROC-AUC"])
    top   = valid.head(top_n)

    print("\n" + "═" * 78)
    print("  STAGE 05a — BASELINE BENCHMARK RESULTS  [POOLED: ALL ANTIBIOTICS]")
    print(f"  Subset {subset_size:.0%} of train pool | Train {len(X_train)} | Test {len(X_test)} | Seed {seed}")
    print("═" * 78)
    print(f"  {'Rank':<5} {'Classifier':<35} {'AUC':>7} {'F1':>7} {'MCC':>7} {'VME%':>6} {'ME%':>6}")
    print(f"  {'─'*5} {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")

    for rank, row in top.iterrows():
        print(
            f"  {rank:<5} {row['Classifier']:<35} "
            f"{row.get('ROC-AUC',   float('nan')):>7.4f} "
            f"{row.get('F1 (macro)',float('nan')):>7.4f} "
            f"{row.get('MCC',       float('nan')):>7.4f} "
            f"{row.get('VME (%)',   float('nan')):>6.1f} "
            f"{row.get('ME (%)',    float('nan')):>6.1f}"
        )

    print("═" * 78)
    best = valid.iloc[0]
    print(f"\n  ✅ Best baseline (pooled): {best['Classifier']}")
    print(f"     ROC-AUC={best['ROC-AUC']:.4f} | F1={best['F1 (macro)']:.4f} | MCC={best['MCC']:.4f}")
    print(f"     VME={best['VME (%)']:.1f}%  ME={best['ME (%)']:.1f}%")

    if ab_results:
        print(f"\n  Per-antibiotic best classifiers:")
        for ab, ab_res in ab_results.items():
            ab_valid = ab_res.dropna(subset=["ROC-AUC"])
            if len(ab_valid) == 0:
                continue
            ab_best = ab_valid.iloc[0]
            imbal_note = "  ⚠ severe imbalance" if ab == "amikacin" else ""
            print(
                f"    {ab:<15}  {ab_best['Classifier']:<35}  "
                f"AUC={ab_best['ROC-AUC']:.4f}  VME={ab_best['VME (%)']:.1f}%"
                f"{imbal_note}"
            )

    print(f"\n  Full results (pooled) → {csv_path}")
    print(f"  Markdown (pooled)     → {md_path}")
    if ab_results:
        print(f"  Per-antibiotic       → {out_dir / 'baseline_scores_per_antibiotic'}/")
    print("═" * 78)
    print(
        "\n  Next step (Stage 05b):\n"
        "    Take the top 2–3 model families from this table and run\n"
        "    leakage-safe 5-fold stratified cross-validation with tuning\n"
        "    on the FULL TRAINING POOL (not the held-out set), then\n"
        "    compute SHAP values for Stage 06.\n"
        "\n  ⚠ Held-out test set is LOCKED until final evaluation in Stage 05c.\n"
        "    Do not use held_out_test_accessions.txt until all tuning is done.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
## MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Phase 1: Load & merge ─────────────────────────────────────────────────
    X, y, antibiotic = load_and_merge(args.X, args.model_data)

    # ── Phase 2a: Carve held-out test set (LOCK IT) ───────────────────────────
    X_pool, y_pool, X_held, y_held = carve_held_out(
        X, y, args.held_out_frac, args.seed, args.out_dir
    )

    # Subset the antibiotic Series to the training pool accessions
    antibiotic_pool = antibiotic.loc[X_pool.index]

    # ── Phase 2b: Benchmark subset from training pool ─────────────────────────
    X_train, X_test, y_train, y_test = make_subset(
        X_pool, y_pool, args.subset_size, args.train_size, args.seed
    )

    # ── Phase 3: Build classifier registry ───────────────────────────────────
    registry = build_classifier_registry(args.seed)
    log.info(f"Classifiers to benchmark: {len(registry)}")

    # ── Phase 5a: Pooled benchmark ────────────────────────────────────────────
    results = benchmark(registry, X_train, X_test, y_train, y_test, label="pooled")

    # ── Phase 5b: Per-antibiotic benchmarks ──────────────────────────────────
    ab_results = {}
    if not args.skip_per_antibiotic:
        ab_results = benchmark_per_antibiotic(
            registry={},   # rebuilt inside per-antibiotic function
            X_pool=X_pool,
            y_pool=y_pool,
            antibiotic_pool=antibiotic_pool,
            subset_size=args.subset_size,
            train_size=args.train_size,
            seed=args.seed,
        )

    # ── Phase 6: Reports ──────────────────────────────────────────────────────
    write_report(
        results, ab_results, args.out_dir, args.top_n,
        X_train, X_test, y_train, y_test,
        args.subset_size, args.seed,
    )


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE
# =============================================================================
#
#  Standard run (pooled + per-antibiotic):
#    python amr_geno2dock_pipeline_stage_05a.py \
#        --X ml_data/X.csv \
#        --model-data model_data_multiab.csv \
#        --out-dir baseline_results
#
#  Skip per-antibiotic breakdown (faster):
#    python amr_geno2dock_pipeline_stage_05a.py \
#        --X ml_data/X.csv \
#        --model-data model_data_multiab.csv \
#        --skip-per-antibiotic
#
#  Larger subset (80% of pool):
#    python amr_geno2dock_pipeline_stage_05a.py \
#        --X ml_data/X.csv \
#        --model-data model_data_multiab.csv \
#        --subset-size 0.80
#
#  Different held-out fraction:
#    python amr_geno2dock_pipeline_stage_05a.py \
#        --X ml_data/X.csv \
#        --model-data model_data_multiab.csv \
#        --held-out-frac 0.15
#
# =============================================================================
# REQUIRED model_data CSV FORMAT
# =============================================================================
#
#  Must contain these columns (order does not matter):
#    Assembly_Accession   — GCA_*/GCF_* accession matching X.csv index
#    Antibiotic           — 'gentamicin' | 'amikacin' | 'tobramycin'
#    binary_class         — 'susceptible' | 'non-susceptible'
#
#  Example rows:
#    Assembly_Accession,Antibiotic,binary_class
#    GCA_000001405.28,gentamicin,non-susceptible
#    GCA_000001405.28,tobramycin,susceptible
#    GCA_000002035.3,amikacin,susceptible
#
#  Multi-antibiotic deduplication:
#    If one accession has the same binary_class across all its antibiotic
#    rows → kept (one row per accession used in the pooled model).
#    If an accession has conflicting binary_class across antibiotics →
#    dropped (conservative label integrity policy).
#
# =============================================================================
# METRIC GUIDE
# =============================================================================
#
#  ROC-AUC      Primary ranking metric. Threshold-free; measures how well the
#               model separates non-susceptible from susceptible across all
#               cutoffs. >0.90 = excellent | 0.80-0.90 = good | <0.70 = poor.
#
#  F1 macro     Average F1 across both classes. Penalises models that ignore
#               the minority class. Use alongside AUC for imbalanced data.
#
#  MCC          Matthews Correlation Coefficient. Most reliable single binary
#               metric under class imbalance. Range: -1 to +1.
#               >0.60 = strong | 0.40-0.60 = moderate | <0.20 = poor.
#
#  VME (%)      Very Major Error. Predicted susceptible when actually
#               non-susceptible. Clinically most dangerous (missed resistance
#               → treatment failure). FDA software threshold: ≤1.5%.
#
#  ME (%)       Major Error. Predicted non-susceptible when actually
#               susceptible. Clinically wasteful (unnecessary escalation).
#               FDA software threshold: ≤3.0%.
#
#  Sensitivity  Recall for non-susceptible class (y=1). Prioritise for VME.
#  Specificity  Recall for susceptible class (y=0). Prioritise for ME.
#
#  Amikacin note:
#    Amikacin has a severe class imbalance (non-susceptible ratio ~0.15,
#    60 non-susceptible vs 395 susceptible). Per-antibiotic benchmark
#    results for amikacin are flagged and should not be used for model
#    family selection. Consider separate handling or exclusion from the
#    pooled model if amikacin dominates the susceptible class counts.
#
# =============================================================================
