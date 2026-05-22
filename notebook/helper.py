import pandas as pd


def antibiotic_candidate_summary(
    df: pd.DataFrame,
    antibiotic_col: str = 'Antibiotic',
    phenotype_col: str = 'Resistance phenotype',
    accession_col: str = 'Assembly_Accession',
    isolation_col: str = 'Isolation type',
    clinical_label: str = 'clinical',
    min_total: int = 100,
    min_binary_ratio: float = 0.25,
    resistance_label: str = 'resistant',
    susceptible_label: str = 'susceptible',
    intermediate_label: str = 'intermediate',
) -> pd.DataFrame:
    """
    For a species-filtered dataframe, produces a ranked antibiotic candidate
    table showing class balance, data utility, and pipeline suitability.

    Steps applied per antibiotic:
      1. Restrict to clinical isolates
      2. Drop intermediate phenotype
      3. Drop conflicting accessions (same genome, different label)
      4. Deduplicate to one record per unique genome
      5. Compute balance metrics and suitability score

    Parameters
    ----------
    df              : Species-filtered records (all isolation types)
    min_total       : Minimum unique genomes after cleaning to be considered
    min_binary_ratio: Minimum minority/majority ratio to pass filter

    Returns
    -------
    DataFrame ranked by suitability score, with all intermediate stats shown
    """

    rows = []

    for antibiotic in df[antibiotic_col].unique():

        ab_df = df[df[antibiotic_col] == antibiotic].copy()

        # ── Raw counts (before any cleaning) ──────────────────────────────────
        raw_total    = ab_df[accession_col].nunique()
        raw_clinical = ab_df[
            ab_df[isolation_col] == clinical_label
        ][accession_col].nunique()

        # ── Step 1: Clinical only ──────────────────────────────────────────────
        ab_df = ab_df[ab_df[isolation_col] == clinical_label].copy()

        # Raw phenotype counts before cleaning
        raw_r = ab_df[ab_df[phenotype_col] == resistance_label][accession_col].nunique()
        raw_s = ab_df[ab_df[phenotype_col] == susceptible_label][accession_col].nunique()
        raw_i = ab_df[ab_df[phenotype_col] == intermediate_label][accession_col].nunique()
        raw_nd = ab_df[
            ~ab_df[phenotype_col].isin([resistance_label, susceptible_label, intermediate_label])
        ][accession_col].nunique()

        # ── Step 2: Drop intermediate ──────────────────────────────────────────
        ab_df = ab_df[
            ab_df[phenotype_col].isin([resistance_label, susceptible_label])
        ].copy()

        # ── Step 3: Drop conflicting accessions ────────────────────────────────
        dup = ab_df.groupby(accession_col)[phenotype_col].nunique()
        conflicts = dup[dup > 1].index.tolist()
        ab_df = ab_df[~ab_df[accession_col].isin(conflicts)].copy()

        # ── Step 4: Deduplicate ────────────────────────────────────────────────
        ab_df = ab_df.drop_duplicates(
            subset=accession_col, keep='first'
        ).reset_index(drop=True)

        # ── Step 5: Final counts ───────────────────────────────────────────────
        n_r = (ab_df[phenotype_col] == resistance_label).sum()
        n_s = (ab_df[phenotype_col] == susceptible_label).sum()
        n_total = n_r + n_s

        if n_total == 0:
            continue

        # ── Metrics ───────────────────────────────────────────────────────────
        binary_ratio = round(min(n_r, n_s) / max(n_r, n_s), 3) if max(n_r, n_s) > 0 else np.nan

        resistance_prevalence = round(n_r / n_total * 100, 1)

        # Data loss from cleaning
        data_retention = round(n_total / raw_clinical * 100, 1) if raw_clinical > 0 else np.nan
        n_conflicts    = len(conflicts)
        n_dropped_i    = raw_i

        # Balance label
        if pd.isna(binary_ratio):
            balance_label = 'N/A'
        elif binary_ratio >= 0.75:
            balance_label = 'balanced'
        elif binary_ratio >= 0.50:
            balance_label = 'moderate'
        elif binary_ratio >= 0.25:
            balance_label = 'imbalanced'
        else:
            balance_label = 'severe'

        # ── Suitability score (0–100) ──────────────────────────────────────────
        # Weighted combination of:
        #   40% — binary balance ratio (higher = better)
        #   30% — log-scaled total n (more data = better, diminishing returns)
        #   15% — data retention after cleaning (less loss = better)
        #   15% — resistance prevalence proximity to 50% (more central = better)

        score_balance   = (binary_ratio if not pd.isna(binary_ratio) else 0) * 40
        score_n         = min(np.log10(max(n_total, 1)) / np.log10(1000), 1.0) * 30
        score_retention = (data_retention / 100 if not pd.isna(data_retention) else 0) * 15
        score_prev      = (1 - abs(resistance_prevalence - 50) / 50) * 15
        suitability     = round(score_balance + score_n + score_retention + score_prev, 1)

        rows.append({
            'antibiotic':           antibiotic,
            # Raw
            'raw_total':            raw_total,
            'raw_clinical':         raw_clinical,
            'raw_resistant':        raw_r,
            'raw_susceptible':      raw_s,
            'raw_intermediate':     raw_i,
            'raw_not_defined':      raw_nd,
            # Cleaned
            'clean_resistant':      n_r,
            'clean_susceptible':    n_s,
            'clean_total':          n_total,
            # Quality
            'n_conflicts_dropped':  n_conflicts,
            'n_intermediate_dropped': n_dropped_i,
            'data_retention_%':     data_retention,
            # Balance
            'binary_ratio':         binary_ratio,
            'balance_label':        balance_label,
            'resistance_prev_%':    resistance_prevalence,
            # Score
            'suitability_score':    suitability,
        })

    summary = (
        pd.DataFrame(rows)
        .set_index('antibiotic')
        .sort_values('suitability_score', ascending=False)
    )

    # ── Apply filters ──────────────────────────────────────────────────────────
    viable = summary[
        (summary['clean_total'] >= min_total) &
        (summary['binary_ratio'] >= min_binary_ratio)
    ].copy()

    # print("=" * 70)
    # print(f"ANTIBIOTIC CANDIDATE SUMMARY")
    # print(f"Total antibiotics evaluated : {len(summary)}")
    # print(f"Passing filters             : {len(viable)}")
    # print(f"  min_total ≥ {min_total} clean genomes, binary_ratio ≥ {min_binary_ratio}")
    # print("=" * 70)

    display_cols = [
        'clean_resistant', 'clean_susceptible', 'clean_total',
        'binary_ratio', 'balance_label', 'resistance_prev_%',
        'data_retention_%', 'n_conflicts_dropped', 'n_intermediate_dropped',
        'suitability_score'
    ]

    # print("\n── VIABLE CANDIDATES (ranked by suitability) ──")
    # print(viable[display_cols].to_string())

    # print("\n── FULL TABLE (all antibiotics) ──")
    # print(summary[display_cols].to_string())

    return summary, viable

def summarize_category_counts(
    records,
    category_column,
    category_value,
    groupby_column='Antibiotic',
    biosample_column='#BioSample',
    phenotype_column='Resistance phenotype',
    undefined_label='not defined'
):
    """
    Summarize testing coverage for any categorical subset
    within an AMR dataframe.

    Parameters
    ----------
    records : pandas.DataFrame
        Input AMR dataframe.

    category_column : str
        Column used for filtering.

        Examples:
        - "Species"
        - "Genus"
        - "Phylum"
        - "Country"

    category_value : str
        Value within category_column to subset.

        Examples:
        - "Acinetobacter baumannii"
        - "Acinetobacter"
        - "Proteobacteria"

    groupby_column : str, default='Antibiotic'
        Column used for summarization.

        Examples:
        - "Antibiotic"
        - "Resistance phenotype"
        - "Class"

    biosample_column : str, default='#BioSample'
        Column containing genome/sample identifiers.

    phenotype_column : str, default='Resistance phenotype'
        Column containing phenotype labels.

    undefined_label : str, default='not defined'
        Entries to exclude from phenotype filtering.

    Returns
    -------
    species_records : pandas.DataFrame
        Filtered dataframe.

    summary_counts : pandas.DataFrame
        Unique BioSample counts grouped by groupby_column.
    """

    # ---------------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------------

    required_cols = [
        category_column,
        groupby_column,
        biosample_column,
        phenotype_column
    ]

    missing_cols = [c for c in required_cols if c not in records.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # ---------------------------------------------------------------
    # Filter records
    # ---------------------------------------------------------------

    species_records = records[
        (records[category_column] == category_value) &
        (records[phenotype_column] != undefined_label)
    ].copy()

    # ---------------------------------------------------------------
    # Count unique BioSamples
    # ---------------------------------------------------------------

    n_genomes = species_records[biosample_column].nunique()

    print(f"{category_column}: {category_value}")
    print(f"No. unique BioSamples: {n_genomes}")

    # ---------------------------------------------------------------
    # Summarize unique genome counts
    # ---------------------------------------------------------------

    summary_counts = (
        species_records
        .groupby(groupby_column)[biosample_column]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name='Genomes')
    )

    return species_records, summary_counts
