#!/usr/bin/env python3
# coding: utf-8
# AUTHOR: Akshay Shirsath (2026)
# =============================================================================
# AMR-Geno2Dock — Feature Matrix Construction (Stage 04)
# =============================================================================
#
# PURPOSE:
#   Pivot the per-sample AMRFinderPlus TSVs from Stage 03 into a binary
#   gene presence/absence feature matrix, then merge with the phenotype
#   labels exported by Stage 01 to produce the final ML-ready dataset.
#
# INPUT  (from Stage 03):
#   amrfinder/
#   └── <accession>.tsv          — AMRFinderPlus output per genome
#
# INPUT  (from Stage 01):
#   ab_tobramycin_accessions.txt — accession list (used to recover labels)
#   model_data.csv               — cleaned dataframe with Assembly_Accession
#                                  and Resistance phenotype columns
#
# OUTPUT:
#   feature_matrix.csv           — rows = accessions, cols = AMR genes (0/1)
#   X.csv                        — feature matrix (accessions with labels)
#   y.csv                        — binary labels (1 = resistant, 0 = susceptible)
#   feature_matrix_report.tsv    — per-gene prevalence & per-sample gene counts
#   matrix_build_report.txt      — summary of construction decisions
#
# PIPELINE PHASES:
#   0  — Imports, constants & CLI
#   1  — Load & validate AMRFinderPlus TSVs
#   2  — Parse and normalise gene names
#   3  — Pivot to binary presence/absence matrix
#   4  — Quality filters (low-prevalence gene pruning)
#   5  — Merge phenotype labels
#   6  — Export & report
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0 — Imports, constants & CLI
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ── AMRFinderPlus column schema (v3.x) ────────────────────────────────────────
# The column used to name genes. AMRFinderPlus outputs both:
#   "Gene symbol"       — short HGNC-style name  (e.g. aac(6')-Ib)
#   "Sequence name"     — full descriptive name
# We use "Gene symbol" as the primary feature name (compact, standardised).
# "Element type" filters out non-AMR hits when --plus is used:
#   AMR   — acquired resistance gene
#   POINT — point mutation conferring resistance
#   STRESS, VIRULENCE, METAL — non-AMR elements added by --plus
GENE_COL    = "Gene symbol"
TYPE_COL    = "Element type"
SUBTYPE_COL = "Element subtype"   # e.g. AMR, POINT, STRESS …

# Element types to include in the feature matrix.
# Default: AMR (acquired genes) + POINT (chromosomal mutations).
# Remove POINT if you want acquired-gene features only.
DEFAULT_ELEMENT_TYPES = {"AMR", "POINT"}

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
        prog="amr_geno2dock_pipeline_stage_04.py",
        description=(
            "Stage 04 — Build gene presence/absence feature matrix from "
            "AMRFinderPlus TSVs (Stage 03) and merge with phenotype labels "
            "(Stage 01) to produce the ML-ready dataset for Stage 05."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "-i", "--amrfinder-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory of AMRFinderPlus TSV files from Stage 03.",
    )
    parser.add_argument(
        "--phenotypes",
        required=True,
        type=Path,
        metavar="CSV",
        help=(
            "CSV/TSV file with at minimum two columns: "
            "'Assembly_Accession' and 'Resistance phenotype'. "
            "This is the model_data export from Stage 01."
        ),
    )
    parser.add_argument(
        "-o", "--out-dir",
        type=Path,
        default=Path("ml_data"),
        metavar="DIR",
        help="Output directory for matrix files and report.",
    )

    # ── Feature filtering ────────────────────────────────────────────────────
    parser.add_argument(
        "--element-types",
        nargs="+",
        default=list(DEFAULT_ELEMENT_TYPES),
        metavar="TYPE",
        help=(
            "AMRFinderPlus element types to include as features. "
            "Options: AMR POINT STRESS VIRULENCE METAL. "
            "Use 'AMR' alone to exclude point mutations."
        ),
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help=(
            "Minimum gene prevalence (fraction of samples) to retain a gene "
            "as a feature. Genes present in fewer samples are near-zero variance "
            "and add noise without predictive value. Default 0.01 = 1%%."
        ),
    )
    parser.add_argument(
        "--max-prevalence",
        type=float,
        default=0.99,
        metavar="FLOAT",
        help=(
            "Maximum gene prevalence to retain a gene. Genes present in nearly "
            "all samples are also near-zero variance. Default 0.99 = 99%%."
        ),
    )

    # ── Label options ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--resistant-label",
        default="resistant",
        metavar="STR",
        help="String value in 'Resistance phenotype' that maps to y=1.",
    )
    parser.add_argument(
        "--susceptible-label",
        default="susceptible",
        metavar="STR",
        help="String value in 'Resistance phenotype' that maps to y=0.",
    )

    # ── Run control ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help=(
            "Keep accessions present in the AMRFinder outputs but absent from "
            "the phenotype file. These rows will have NaN labels and will be "
            "exported to feature_matrix.csv but excluded from X.csv / y.csv."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 1 — Load & validate AMRFinderPlus TSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_amrfinder_tsv(path: Path) -> pd.DataFrame | None:
    """
    Load one AMRFinderPlus TSV. Returns None if empty or malformed.

    AMRFinderPlus writes a header row even when no genes are found,
    so a zero-row DataFrame is valid (genome has no AMR hits).
    """
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        return df
    except Exception as e:
        log.warning(f"Could not parse {path.name}: {e}")
        return None


def load_all_tsvs(
    amrfinder_dir: Path,
    element_types: set[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Read every *.tsv in amrfinder_dir, tag with accession, stack into one
    long DataFrame, then filter to the requested element types.

    Returns:
        long_df      — stacked DataFrame with 'accession' column added
        all_accessions — all accession stems found in the directory
        empty_accessions — accessions with zero AMR hits after type filter
    """
    tsv_files = sorted(amrfinder_dir.glob("*.tsv"))

    # Exclude the run report written by Stage 03
    tsv_files = [f for f in tsv_files if f.stem != "amrfinder_run_report"]

    if not tsv_files:
        log.error(f"No *.tsv files found in {amrfinder_dir}")
        sys.exit(1)

    log.info(f"Loading {len(tsv_files)} AMRFinderPlus TSV files …")

    frames: list[pd.DataFrame] = []
    all_accessions: list[str]   = []
    empty_accessions: list[str] = []

    for f in tsv_files:
        accession = f.stem
        all_accessions.append(accession)
        df = load_amrfinder_tsv(f)

        if df is None or df.empty:
            empty_accessions.append(accession)
            continue

        # Validate required columns are present
        if GENE_COL not in df.columns:
            log.warning(
                f"{f.name}: expected column '{GENE_COL}' not found. "
                f"Columns present: {list(df.columns)[:8]}. Skipping."
            )
            empty_accessions.append(accession)
            continue

        # Filter to requested element types
        if TYPE_COL in df.columns:
            df = df[df[TYPE_COL].isin(element_types)]

        if df.empty:
            empty_accessions.append(accession)
            continue

        df = df[[GENE_COL]].copy()
        df["accession"] = accession
        frames.append(df)

    if not frames:
        log.error(
            "No gene hits found across all TSVs after element-type filtering. "
            "Check --element-types or verify AMRFinderPlus ran correctly."
        )
        sys.exit(1)

    long_df = pd.concat(frames, ignore_index=True)
    log.info(
        f"Loaded {len(long_df):,} gene hits across "
        f"{len(all_accessions) - len(empty_accessions)} genomes "
        f"({len(empty_accessions)} with zero hits)."
    )

    return long_df, all_accessions, empty_accessions


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 2 — Parse and normalise gene names
# ─────────────────────────────────────────────────────────────────────────────

def normalise_gene_names(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise gene symbol strings for safe use as DataFrame column names.

    AMRFinderPlus gene symbols can contain characters problematic for
    downstream ML libraries (e.g. parentheses in "aac(6')-Ib", apostrophes,
    hyphens). We replace them with underscores and strip leading/trailing
    whitespace.

    The original symbol is preserved in 'gene_raw'; the normalised version
    used as matrix column name is in 'gene'.
    """
    long_df["gene_raw"] = long_df[GENE_COL].str.strip()
    long_df["gene"] = (
        long_df["gene_raw"]
        .str.replace(r"[^\w]", "_", regex=True)   # non-word chars → _
        .str.replace(r"_+", "_", regex=True)       # collapse repeated underscores
        .str.strip("_")
    )
    return long_df


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 3 — Pivot to binary presence/absence matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_presence_absence_matrix(
    long_df: pd.DataFrame,
    all_accessions: list[str],
) -> pd.DataFrame:
    """
    Pivot long gene-hit table → wide binary presence/absence matrix.

    Rows   : one per accession (all accessions, including zero-hit genomes)
    Columns: one per unique normalised gene symbol
    Values : 1 if the gene was detected in that genome, 0 otherwise

    Multiple hits of the same gene in one genome are collapsed to 1
    (presence/absence, not copy number).
    """
    # Deduplicate: one row per (accession, gene) pair
    hits = long_df[["accession", "gene"]].drop_duplicates()

    # Pivot
    matrix = (
        hits
        .assign(present=1)
        .pivot_table(
            index="accession",
            columns="gene",
            values="present",
            aggfunc="max",      # max() collapses duplicates to 1
            fill_value=0,
        )
    )
    matrix.columns.name = None   # remove "gene" label from column axis

    # Reindex to include zero-hit accessions as all-zero rows
    matrix = matrix.reindex(all_accessions, fill_value=0)

    log.info(
        f"Raw feature matrix: {matrix.shape[0]} samples × {matrix.shape[1]} genes"
    )
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 4 — Quality filters
# ─────────────────────────────────────────────────────────────────────────────

def filter_low_variance_genes(
    matrix: pd.DataFrame,
    min_prev: float,
    max_prev: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove near-zero-variance gene columns.

    A gene that is present in <min_prev or >max_prev of samples contributes
    almost no information to a classifier and inflates dimensionality.

    Returns:
        filtered_matrix  — matrix with low/high-prevalence columns removed
        prevalence_report — DataFrame with per-gene prevalence stats
    """
    n_samples = len(matrix)
    prevalence = matrix.mean()   # fraction of samples with gene present

    prev_report = pd.DataFrame({
        "gene":          prevalence.index,
        "n_present":     (matrix > 0).sum().values,
        "prevalence":    prevalence.values,
        "kept":          (
            (prevalence >= min_prev) & (prevalence <= max_prev)
        ).values,
    }).sort_values("prevalence", ascending=False).reset_index(drop=True)

    keep_mask = (prevalence >= min_prev) & (prevalence <= max_prev)
    n_before  = matrix.shape[1]
    matrix    = matrix.loc[:, keep_mask]
    n_after   = matrix.shape[1]
    n_dropped = n_before - n_after

    log.info(
        f"Prevalence filter [{min_prev:.0%} – {max_prev:.0%}]: "
        f"dropped {n_dropped} genes, retained {n_after} genes."
    )

    if n_after == 0:
        log.error(
            "All genes removed by prevalence filter. "
            "Try relaxing --min-prevalence and --max-prevalence."
        )
        sys.exit(1)

    return matrix, prev_report


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 5 — Merge phenotype labels
# ─────────────────────────────────────────────────────────────────────────────

def load_phenotypes(
    path: Path,
    resistant_label: str,
    susceptible_label: str,
) -> pd.DataFrame:
    """
    Load the Stage 01 model_data export.

    Expects at minimum:
        Assembly_Accession   — must match accession stems in AMRFinder outputs
        Resistance phenotype — raw label ('resistant' / 'susceptible')

    Maps phenotype labels to binary integers:
        resistant   → 1
        susceptible → 0
    """
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    pheno = pd.read_csv(path, sep=sep, low_memory=False)

    required = {"Assembly_Accession", "Resistance phenotype"}
    missing  = required - set(pheno.columns)
    if missing:
        log.error(
            f"Phenotype file missing required columns: {missing}\n"
            f"Columns found: {list(pheno.columns)}"
        )
        sys.exit(1)

    pheno = pheno[["Assembly_Accession", "Resistance phenotype"]].copy()
    pheno = pheno.drop_duplicates(subset="Assembly_Accession")

    label_map = {resistant_label: 1, susceptible_label: 0}
    pheno["y"] = pheno["Resistance phenotype"].map(label_map)

    unmapped = pheno["y"].isna().sum()
    if unmapped > 0:
        bad_labels = pheno[pheno["y"].isna()]["Resistance phenotype"].unique()
        log.warning(
            f"{unmapped} accession(s) have unrecognised phenotype labels "
            f"and will be excluded: {bad_labels}. "
            f"Check --resistant-label / --susceptible-label."
        )

    pheno = pheno.dropna(subset=["y"])
    pheno["y"] = pheno["y"].astype(int)

    n_r = (pheno["y"] == 1).sum()
    n_s = (pheno["y"] == 0).sum()
    ratio = round(min(n_r, n_s) / max(n_r, n_s), 3) if max(n_r, n_s) > 0 else 0
    balance = (
        "✅ balanced" if ratio >= 0.75
        else "⚠ moderate imbalance" if ratio >= 0.50
        else "⚠ imbalanced"
    )
    log.info(
        f"Phenotype labels: {n_r} resistant | {n_s} susceptible "
        f"(balance ratio {ratio} — {balance})"
    )

    return pheno


def merge_labels(
    matrix: pd.DataFrame,
    pheno: pd.DataFrame,
    keep_unmatched: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Inner-join the feature matrix with phenotype labels on Assembly_Accession.

    Returns:
        full_matrix — matrix for all accessions (with NaN y if keep_unmatched)
        X           — feature matrix for labelled accessions only
        y           — binary label Series aligned to X
    """
    matrix_reset = matrix.reset_index().rename(
        columns={"accession": "Assembly_Accession"}
    )

    if keep_unmatched:
        merged = matrix_reset.merge(
            pheno[["Assembly_Accession", "y"]],
            on="Assembly_Accession",
            how="left",
        )
    else:
        merged = matrix_reset.merge(
            pheno[["Assembly_Accession", "y"]],
            on="Assembly_Accession",
            how="inner",
        )

    # Samples in AMRFinder output but not in phenotype file
    n_unmatched = merged["y"].isna().sum()
    if n_unmatched > 0:
        log.warning(
            f"{n_unmatched} accession(s) from AMRFinder have no phenotype label "
            f"and will be {'kept with NaN y' if keep_unmatched else 'excluded'}."
        )

    # Samples in phenotype file but not in AMRFinder output
    amrfinder_accs = set(matrix.index)
    pheno_accs     = set(pheno["Assembly_Accession"])
    missing_from_amrfinder = pheno_accs - amrfinder_accs
    if missing_from_amrfinder:
        log.warning(
            f"{len(missing_from_amrfinder)} accession(s) in the phenotype file "
            f"have no AMRFinder TSV — they will be excluded from X/y. "
            f"This may indicate failed downloads or AMRFinder runs."
        )

    merged = merged.set_index("Assembly_Accession")
    full_matrix = merged

    # ML-ready split: labelled rows only
    labelled = merged.dropna(subset=["y"])
    X = labelled.drop(columns=["y"])
    y = labelled["y"].astype(int)

    log.info(
        f"Final ML dataset: {X.shape[0]} samples × {X.shape[1]} features | "
        f"y=1: {y.sum()} | y=0: {(y == 0).sum()}"
    )

    return full_matrix, X, y


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 6 — Export & report
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    full_matrix: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    prev_report: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """
    Write all output files and print the human-readable summary.

    Files written:
        feature_matrix.csv       — complete matrix (all accessions, including
                                   unmatched if --keep-unmatched)
        X.csv                    — ML feature matrix (labelled accessions only)
        y.csv                    — binary labels aligned to X
        feature_matrix_report.tsv — per-gene prevalence stats
        matrix_build_report.txt  — human-readable build summary
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data files ────────────────────────────────────────────────────────────
    fm_path = out_dir / "feature_matrix.csv"
    X_path  = out_dir / "X.csv"
    y_path  = out_dir / "y.csv"
    pr_path = out_dir / "feature_matrix_report.tsv"

    full_matrix.to_csv(fm_path)
    X.to_csv(X_path)
    y.to_csv(y_path, header=True)
    prev_report.to_csv(pr_path, sep="\t", index=False)

    # ── Build report ──────────────────────────────────────────────────────────
    n_samples  = X.shape[0]
    n_features = X.shape[1]
    n_r        = int(y.sum())
    n_s        = int((y == 0).sum())
    ratio      = round(min(n_r, n_s) / max(n_r, n_s), 3) if max(n_r, n_s) > 0 else 0
    balance    = (
        "✅ balanced" if ratio >= 0.75
        else "⚠ moderate imbalance" if ratio >= 0.50
        else "⚠ imbalanced"
    )

    gene_counts_per_sample = X.sum(axis=1)

    report_lines = [
        "═" * 60,
        "  STAGE 04 — FEATURE MATRIX BUILD REPORT",
        "═" * 60,
        f"  Samples (labelled)   : {n_samples}",
        f"  Features (genes)     : {n_features}",
        f"  y=1 (resistant)      : {n_r}",
        f"  y=0 (susceptible)    : {n_s}",
        f"  Class balance ratio  : {ratio}  ({balance})",
        "",
        "  Genes per sample:",
        f"    median  : {gene_counts_per_sample.median():.0f}",
        f"    min/max : {gene_counts_per_sample.min():.0f} / {gene_counts_per_sample.max():.0f}",
        "",
        f"  Element types included : {', '.join(sorted(args.element_types))}",
        f"  Prevalence filter      : [{args.min_prevalence:.0%} – {args.max_prevalence:.0%}]",
        "",
        "  Output files:",
        f"    feature_matrix.csv          → {fm_path}",
        f"    X.csv (ML features)         → {X_path}",
        f"    y.csv (binary labels)       → {y_path}",
        f"    feature_matrix_report.tsv   → {pr_path}",
        "═" * 60,
        "",
        "  Next step (Stage 05):",
        "    Train classifier on X / y:",
        "      from sklearn.ensemble import RandomForestClassifier",
        "      import pandas as pd",
        "      X = pd.read_csv('ml_data/X.csv', index_col=0)",
        "      y = pd.read_csv('ml_data/y.csv', index_col=0).squeeze()",
        "      # → leakage-safe cross-validation → SHAP feature importance",
        "═" * 60,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (out_dir / "matrix_build_report.txt").write_text(report_text)


# ─────────────────────────────────────────────────────────────────────────────
## MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    element_types = set(args.element_types)
    log.info(f"Element types included: {element_types}")

    # ── Phase 1: Load TSVs ────────────────────────────────────────────────────
    long_df, all_accessions, empty_accessions = load_all_tsvs(
        args.amrfinder_dir, element_types
    )

    if empty_accessions:
        log.info(
            f"{len(empty_accessions)} genome(s) had zero AMR hits — "
            "they will appear as all-zero rows in the matrix."
        )

    # ── Phase 2: Normalise gene names ─────────────────────────────────────────
    long_df = normalise_gene_names(long_df)

    # ── Phase 3: Build presence/absence matrix ────────────────────────────────
    matrix = build_presence_absence_matrix(long_df, all_accessions)

    # ── Phase 4: Filter low-variance genes ───────────────────────────────────
    matrix, prev_report = filter_low_variance_genes(
        matrix, args.min_prevalence, args.max_prevalence
    )

    # ── Phase 5: Merge phenotype labels ───────────────────────────────────────
    pheno = load_phenotypes(
        args.phenotypes,
        args.resistant_label,
        args.susceptible_label,
    )

    full_matrix, X, y = merge_labels(matrix, pheno, args.keep_unmatched)

    # ── Phase 6: Export ───────────────────────────────────────────────────────
    write_outputs(full_matrix, X, y, prev_report, args.out_dir, args)


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
#  Standard run (AMR + POINT mutations, 1% prevalence filter):
#    python amr_geno2dock_pipeline_stage_04.py \
#        --amrfinder-dir amrfinder \
#        --phenotypes model_data.csv \
#        --out-dir ml_data
#
#  Acquired genes only (exclude point mutations):
#    python amr_geno2dock_pipeline_stage_04.py \
#        --amrfinder-dir amrfinder \
#        --phenotypes model_data.csv \
#        --element-types AMR
#
#  Stricter prevalence filter (gene must appear in ≥5% of samples):
#    python amr_geno2dock_pipeline_stage_04.py \
#        --amrfinder-dir amrfinder \
#        --phenotypes model_data.csv \
#        --min-prevalence 0.05
#
#  Keep accessions without a phenotype label (for inspection):
#    python amr_geno2dock_pipeline_stage_04.py \
#        --amrfinder-dir amrfinder \
#        --phenotypes model_data.csv \
#        --keep-unmatched
#
# =============================================================================
# STAGE 05 HANDOFF — LEAKAGE-SAFE ML TRAINING
# =============================================================================
#
#   X = pd.read_csv('ml_data/X.csv', index_col=0)   # (543 × n_genes)
#   y = pd.read_csv('ml_data/y.csv', index_col=0).squeeze()
#
#   IMPORTANT — leakage-safe cross-validation:
#   Use StratifiedKFold or StratifiedShuffleSplit.
#   Do NOT apply any feature selection or scaling on the full dataset
#   before splitting — fit transformers inside the fold only.
#
#   Recommended models (literature precedent for this task):
#     - RandomForestClassifier     (robust to correlated gene features)
#     - XGBClassifier              (handles class imbalance via scale_pos_weight)
#     - LogisticRegression (L1)    (sparse, interpretable)
#
#   After training:
#     import shap
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     # → top SHAP features feed Stage 06 target prioritisation
# =============================================================================