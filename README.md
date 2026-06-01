# AMR-Geno2Dock

**End-to-end pipeline: bacterial genomic AMR surveillance → structure-based virtual screening**

Author: Akshay Shirsath | Development started: May 2026

---

## Overview

AMR-Geno2Dock bridges population-level bacterial genomic surveillance with structure-based virtual screening for drug discovery. Given a species × antibiotic combination, the pipeline:

1. Pulls clinical AMR phenotype data from NCBI Antibiograms
2. Downloads matched genome assemblies
3. Runs AMRFinderPlus to produce gene presence/absence feature matrices
4. Trains an interpretable XGBoost classifier and extracts SHAP-ranked resistance genes
5. Filters causal genes from co-selected passengers
6. Retrieves protein structures (UniProt / PDB / AlphaFoldDB) for the top-ranked targets
7. Mines known ligands (ChEMBL / DrugBank)
8. Prepares inputs for AutoDock Vina virtual screening

The pipeline is fully species- and antibiotic-agnostic; species selection and antibiotic ranking are handled programmatically in Stage 01.

---

## Validated Runs

| Species | Antibiotic | Dataset | Status |
|---|---|---|---|
| *Acinetobacter baumannii* | Tobramycin | 543 isolates (315R / 228S) | Stages 01–05c complete |
| *Escherichia coli* | Ampicillin | 3,186 isolates (1,830R / 1,356S) | Stages 01–04 complete; 05c pending |

---

## Pipeline Stages

```
Stage 00  — Train / test split (stratified, 80/20)
Stage 01  — Data preparation & species × antibiotic selection
Stage 02  — Genome download (NCBI Datasets CLI)
Stage 03  — AMRFinderPlus batch runner
Stage 04  — Gene presence/absence feature matrix construction
Stage 05a — Sklearn baseline benchmark (30 classifiers)
Stage 05b — XGBoost baseline (3 feature sets × 3 imbalance strategies)
Stage 05c — XGBoost hyperparameter tuning (Optuna nested CV)
Stage 06  — SHAP analysis + co-selection filter  [NOT YET WRITTEN]
Stage 07  — Protein structure retrieval           [NOT YET WRITTEN]
Stage 08  — Ligand mining (ChEMBL / DrugBank)    [NOT YET WRITTEN]
Stage 09  — AutoDock Vina preparation             [NOT YET WRITTEN]
```

---

## Key Results — *A. baumannii* × Tobramycin

Feature set selected: **69 genes** (1% prevalence filter)  
Model: **XGBoost tuned via Optuna nested 5-fold CV**

> ⚠️ **Note:** These are development-set CV estimates. No held-out test set was carved out before modelling began. Metrics are optimistically biased. The E. coli run corrects this with a proper Stage 00 split.

| Metric | Value |
|---|---|
| ROC-AUC | 0.9795 ± 0.0145 |
| F1 (macro) | 0.9570 ± 0.0292 |
| MCC | 0.9152 ± 0.0578 |
| Sensitivity | 0.9460 ± 0.0348 |
| Specificity | 0.9738 ± 0.0283 |
| VME | 5.40% ± 3.48% ❌ (FDA ≤1.5%) |
| ME | 2.62% ± 2.83% ✅ (FDA ≤3.0%) |

**Final hyperparameters:**

```
n_estimators: 200        learning_rate: 0.0434
max_depth: 4             min_child_weight: 1
subsample: 0.9554        colsample_bytree: 0.7891
gamma: 0.3008            reg_alpha: 0.2774
reg_lambda: 1.2428       scale_pos_weight: 0.7238
random_state: 42
```

---

## Key Results — *E. coli* × Ampicillin (Stage 05b, preliminary)

> ⚠️ **Note:** Stage 05b was run on the full dataset (same leakage issue). Must be re-run on `X_dev` / `y_dev` from Stage 00 before tuning.

Recommended configuration for Stage 05c: **59 genes (1% filter) + `sample_weight`** — the only sub-441 variant to pass the FDA ME threshold (ME = 2.37%).

---

## Directory Structure

```
.
├── amr_geno2dock_pipeline_stage_00.py   # Train/test split
├── amr_geno2dock_pipeline_stage_01.py   # Data preparation
├── amr_geno2dock_pipeline_stage_02.py   # Genome download
├── amr_geno2dock_pipeline_stage_03.py   # AMRFinderPlus batch runner
├── amr_geno2dock_pipeline_stage_04.py   # Feature matrix construction
├── amr_geno2dock_pipeline_stage_04v2.py # (alternate)
├── amr_geno2dock_pipeline_stage_05a.py  # Sklearn baseline
├── amr_geno2dock_pipeline_stage_05b.py  # XGBoost baseline
├── amr_geno2dock_pipeline_stage_05c.py  # XGBoost tuning (Optuna)
└── biosample_to_assembly.py             # BioSample → GCA/GCF accession mapping

# Runtime outputs (generated)
genomes/                        # Downloaded .fna FASTA files
amrfinder/                      # Per-sample AMRFinderPlus TSVs + run report
ml_data_5pct/                   # 28-gene feature matrix (5% prevalence filter)
ml_data/                        # 69-gene feature matrix (1% filter) ★ selected
ml_data_all/                    # 206-gene feature matrix (no filter)
data/split_*/                   # Train/test split outputs from Stage 00
baseline_results/               # Stage 05a sklearn benchmark outputs
xgb_baseline/                   # Stage 05b XGBoost baseline outputs
tuning/                         # Stage 05c tuning outputs (69-gene)
tuning_all/                     # Stage 05c outputs (206-gene)
tuning_5pct/                    # Stage 05c outputs (28-gene)
```

---

## Installation

```bash
bash 00_setup.sh
```

This installs:

- `python=3.10`, `pandas`, `numpy`, `biopython`, `requests`, `tqdm`
- `entrez-direct`, `ncbi-datasets-cli` (genome download)
- `taxonkit`, `pytaxonkit` (taxonomic lineage resolution)
- NCBI taxonomy database (downloaded to `~/.taxonkit/`)

Additional Python packages for ML stages:

```bash
pip install xgboost optuna scikit-learn umap-learn matplotlib
```

AMRFinderPlus (Stage 03):

```bash
conda install -c bioconda ncbi-amrfinderplus
amrfinder --update
```

MLST (optional, for phylogenetic splits):

```bash
conda install -c conda-forge -c bioconda mlst
```

---

## Usage

Run stages in order. All scripts use `argparse` — pass `--help` for full option reference.

```bash
# Stage 00 — create held-out test set FIRST (before any ML work)
python amr_geno2dock_pipeline_stage_00.py \
    --X ml_data/X.csv --y ml_data/y.csv --out-dir data/split_1pct

# Stage 01 — data preparation
python amr_geno2dock_pipeline_stage_01.py

# Stage 02 — genome download
python amr_geno2dock_pipeline_stage_02.py \
    --accessions ab_tobramycin_accessions.txt --out-dir genomes/

# Stage 03 — AMRFinderPlus (organism flag differs by species)
#   A. baumannii → Acinetobacter_baumannii
#   E. coli      → Escherichia
python amr_geno2dock_pipeline_stage_03.py \
    --genomes-dir genomes/ --out-dir amrfinder/ \
    --organism Acinetobacter_baumannii

# Stage 04 — feature matrix
python amr_geno2dock_pipeline_stage_04.py \
    --amrfinder-dir amrfinder/ --labels ab_tobramycin_model_data.csv

# Stage 05b — XGBoost baseline (run on dev set, not full data)
python amr_geno2dock_pipeline_stage_05b.py \
    --datasets "69genes:data/split_1pct/X_dev.csv:data/split_1pct/y_dev.csv"

# Stage 05c — Optuna tuning (recommended: 59-gene + sample_weight for E. coli)
python amr_geno2dock_pipeline_stage_05c.py \
    --X data/split_1pct/X_dev.csv --y data/split_1pct/y_dev.csv \
    --out-dir tuning_ecoli_1pct
```

---

## AMRFinderPlus Schema Note

AMRFinderPlus v4.x (2024+) renamed columns:

| v3.x | v4.x |
|---|---|
| `Gene symbol` | `Element symbol` |
| `Element type` | `Type` |
| `Element subtype` | `Subtype` |

Stage 04 auto-detects the schema version. Mixed directories (v3 + v4 TSVs) are handled correctly.

---

## Modelling Conventions

- **Random seed:** 42 (all stages)
- **Splits:** `StratifiedKFold` / `StratifiedShuffleSplit` (class ratio preserved)
- **Evaluation thresholds:** FDA software AST — VME ≤1.5%, ME ≤3.0%
- **Model format:** XGBoost native JSON (`tuned_xgb_model.json`)
- **Feature matrix naming:** `ml_data_5pct/` (5% filter), `ml_data/` (1% filter), `ml_data_all/` (unfiltered)

---

## Bacterial Splitting Strategies

The pipeline supports five train/test split strategies (see project context for full discussion):

1. **Random stratified** — default (Stage 00); preserves class ratio
2. **Phylogenetic / clonal** — `GroupShuffleSplit` on MLST sequence type; prevents clonal leakage
3. **Temporal** — train on earlier collection years, test on later
4. **Geographic** — cross-country generalisation
5. **Isolation source** — clinical vs. environmental

Recommended workflow: establish random stratified baseline first, then phylogenetic split once MLST results are available. The AUC gap between the two estimates the severity of clonal leakage.

---

## Data Source

Both species use **NCBI Antibiogram (AST) data** — BioSample records with experimentally measured MIC or disk diffusion phenotypes. This is a higher-quality source than aggregated metadata databases.

> The original Conversation 1 context file incorrectly states BV-BRC as the data source. The correct source is NCBI AST for both species.

---

## Pending Work

- [ ] Stage 00 — run on all 3 E. coli feature matrices to create proper train/test splits
- [ ] Stage 05b — re-run E. coli on `X_dev` (current results have leakage)
- [ ] Stage 05c — run E. coli tuning (59-gene + `sample_weight` recommended)
- [ ] Stage 06 — SHAP analysis + co-selection filter (not yet written)
- [ ] Stages 07–09 — structure retrieval, ligand mining, docking (not yet written)
- [ ] MLST results for *A. baumannii* (was running at end of Conversation 2)
- [ ] CheckM QC for *E. coli* assemblies (run locally — NCBI blocked in cloud)
- [ ] Investigate 6 *A. baumannii* assemblies that failed CheckM QC

---

## Literature

Key references informing pipeline design:

- **Feldgarden 2019** — AMRFinderPlus validation (6,242 isolates, 5 species)
- **Nguyen 2017/2019** — XGBoost k-mer MIC prediction for *K. pneumoniae* and *Salmonella*
- **Kim 2020** — VAMPr: XGBoost on protein variants (3,393 isolates, 9 species, 29 antibiotics)
- **Yang 2023** — Salmonella pan-genome XGBoost (7,249 strains, 15 antibiotics)
- **CRyPTIC 2024** — *M. tuberculosis* XGBoost k-mer MIC (10,859 isolates)
- **Kim 2022** — Critical review: ML-AMR best practices, VME/ME, class imbalance
- **Hua 2020** — Environmental→clinical AMR temporal lag (*E. coli*, *Salmonella*)
- **Munita 2016** — Comprehensive mechanisms of antibiotic resistance
- **Dickens 2025** — CABBAGE database: 1.7M genome-phenotype entries, 24 pathogens
