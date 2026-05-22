"""
amr_plot_helpers.py
═══════════════════
Helper script for AMR phenotype visualisation and antibiotic merge decisions.

Visualisation functions
-----------------------
    plot_resistance_rate(df, ...)
    plot_mic_breakpoint_heatmap(df, ...)
    plot_mic_pairwise_correlation(df, ...)

Data homogeneity / merge decision functions
-------------------------------------------
    can_merge(df, antibiotic_a, antibiotic_b, ...)
    can_merge_class(df, antibiotic_class, ...)

Typical usage
-------------
    from amr_plot_helpers import (
        plot_resistance_rate,
        plot_mic_breakpoint_heatmap,
        plot_mic_pairwise_correlation,
        can_merge,
        can_merge_class,
    )

    plot_resistance_rate(plot_data)
    plot_mic_breakpoint_heatmap(plot_data)
    plot_mic_pairwise_correlation(plot_data)

    result  = can_merge(records, "imipenem", "meropenem", isolation_filter="clinical")
    summary = can_merge_class(records, "Carbapenem", isolation_filter="clinical")
"""

from __future__ import annotations

import warnings
from itertools import combinations

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore", category=UserWarning)

# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _log2_mic_series(
    d: pd.DataFrame,
    mic_col: str = "MIC (mg/L)",
    sign_col: str = "Measurement sign",
) -> pd.Series:
    """Convert MIC (mg/L) → log₂ dilution steps, honouring censoring signs."""
    mic = d[mic_col].dropna().astype(float).clip(lower=1e-6)
    log2 = np.log2(mic)
    if sign_col in d.columns:
        signs = d.loc[mic.index, sign_col]
        log2[signs.isin({">", ">="})] += 1
        log2[signs.isin({"<", "<="})] -= 1
    return log2.round().astype(int)


def _log2_to_mic_label(b: int | float) -> str:
    """Convert a log₂ bin integer back to a human-readable MIC string."""
    val = 2.0 ** float(b)
    if val < 1:
        return f"{val:.3f}".rstrip("0").rstrip(".")
    elif val < 10:
        return f"{val:.2f}".rstrip("0").rstrip(".")
    else:
        return f"{int(round(val))}"


# ═════════════════════════════════════════════════════════════════════════════
# 1. plot_resistance_rate
# ═════════════════════════════════════════════════════════════════════════════

def plot_resistance_rate(
    df: pd.DataFrame,
    antibiotic_col: str   = "Antibiotic",
    phenotype_col: str    = "Resistance phenotype",
    isolation_col: str    = "Isolation type",
    resistant_label: str  = "resistant",
    min_n: int            = 10,
    title_prefix: str     = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: resistance rate per antibiotic, split by isolation type.

    Parameters
    ----------
    df            : input DataFrame (plot_data)
    min_n         : bars with fewer than this many total isolates are suppressed
    title_prefix  : prepended to the figure title (e.g. species name)
    save_path     : if provided, figure is saved here (PNG, 150 dpi)

    Returns
    -------
    matplotlib Figure
    """
    df = df.dropna(subset=[phenotype_col, isolation_col, antibiotic_col]).copy()

    total = (
        df.groupby([antibiotic_col, isolation_col])
        .size()
        .reset_index(name="total")
    )
    resistant = (
        df[df[phenotype_col] == resistant_label]
        .groupby([antibiotic_col, isolation_col])
        .size()
        .reset_index(name="resistant")
    )
    rates = total.merge(resistant, on=[antibiotic_col, isolation_col], how="left")
    rates["resistant"] = rates["resistant"].fillna(0)
    rates["resistance_rate"] = rates["resistant"] / rates["total"]
    rates = rates[rates["total"] >= min_n]

    isolation_types = sorted(rates[isolation_col].unique())
    antibiotics     = sorted(rates[antibiotic_col].unique())
    n_groups        = len(antibiotics)
    n_bars          = len(isolation_types)
    bar_width       = 0.35
    x               = np.arange(n_groups)

    COLORS = {
        "clinical":            "#2166ac",
        "environmental/other": "#d6604d",
    }
    DEFAULT_COLORS = ["#4dac26", "#7b3294", "#f1a340"]

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.4), 5))

    for i, iso in enumerate(isolation_types):
        subset  = rates[rates[isolation_col] == iso].set_index(antibiotic_col)
        heights = [subset.loc[ab, "resistance_rate"] if ab in subset.index else 0
                   for ab in antibiotics]
        ns      = [int(subset.loc[ab, "total"]) if ab in subset.index else 0
                   for ab in antibiotics]
        color   = COLORS.get(iso, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        offset  = (i - (n_bars - 1) / 2) * bar_width
        bars    = ax.bar(x + offset, heights, bar_width,
                         label=iso.capitalize(), color=color,
                         alpha=0.88, edgecolor="white", linewidth=0.6)

        for bar, n in zip(bars, ns):
            if n == 0:
                continue
            ypos      = bar.get_height()
            va        = "bottom"
            color_txt = "black"
            if ypos > 0.08:
                ypos      = bar.get_height() / 2
                va        = "center"
                color_txt = "white"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"n={n}", ha="center", va=va,
                    fontsize=7.5, color=color_txt, fontweight="bold")

    title = (f"{title_prefix}\n" if title_prefix else "") + \
            "Resistance Rate by Antibiotic and Isolation Type"
    ax.set_xticks(x)
    ax.set_xticklabels(antibiotics, fontsize=10)
    ax.set_xlabel("Antibiotic", fontsize=11, labelpad=8)
    ax.set_ylabel("Resistance rate", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Isolation type", frameon=False, fontsize=9, title_fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2. plot_mic_breakpoint_heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_mic_breakpoint_heatmap(
    df: pd.DataFrame,
    antibiotic_col: str   = "Antibiotic",
    mic_col: str          = "MIC (mg/L)",
    sign_col: str         = "Measurement sign",
    title_prefix: str     = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap of isolate counts across log₂ MIC bins per antibiotic.

    Uses √ colour normalisation to prevent dominant bins washing out minority
    ones. Marks the right-censoring ceiling with a dashed blue line.

    Parameters
    ----------
    title_prefix  : prepended to the figure title
    save_path     : if provided, figure is saved here (PNG, 150 dpi)

    Returns
    -------
    matplotlib Figure
    """
    df = df.dropna(subset=[mic_col, antibiotic_col]).copy()
    df["log2_bin"] = _log2_mic_series(df, mic_col=mic_col, sign_col=sign_col)

    all_bins    = sorted(df["log2_bin"].unique())
    antibiotics = sorted(df[antibiotic_col].unique())

    count_matrix = (
        df.groupby([antibiotic_col, "log2_bin"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=antibiotics, columns=all_bins, fill_value=0)
    )

    data = count_matrix.values.astype(float)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=data.max())

    fig, ax = plt.subplots(
        figsize=(max(10, len(all_bins) * 0.7), max(3, len(antibiotics) * 0.9))
    )

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", norm=norm)

    ax.set_xticks(range(len(all_bins)))
    ax.set_xticklabels(
        [_log2_to_mic_label(b) for b in all_bins],
        fontsize=9, rotation=45, ha="right",
    )
    ax.set_yticks(range(len(antibiotics)))
    ax.set_yticklabels(antibiotics, fontsize=10)
    ax.set_xlabel("MIC (mg/L)  —  log₂ two-fold dilution steps", fontsize=11, labelpad=8)
    ax.set_ylabel("Antibiotic", fontsize=11, labelpad=8)

    title = (f"{title_prefix}\n" if title_prefix else "") + \
            "MIC Breakpoint Heatmap\n(cell colour = isolate count; √ colour scale)"
    ax.set_title(title, fontsize=13, pad=12)

    for row_i in range(data.shape[0]):
        for col_j in range(data.shape[1]):
            n = int(data[row_i, col_j])
            if n == 0:
                continue
            tc = "white" if norm(n) > 0.55 else "black"
            ax.text(col_j, row_i, str(n), ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Isolate count  (√ scale)", fontsize=9)

    last_populated = np.where(data.sum(axis=0) > 0)[0]
    if len(last_populated) > 1:
        ax.axvline(last_populated[-1] + 0.5, color="#2166ac", linewidth=1.5,
                   linestyle="--", alpha=0.7, label="Right-censoring ceiling")
        ax.legend(fontsize=8, loc="upper left", frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3. plot_mic_pairwise_correlation
# ═════════════════════════════════════════════════════════════════════════════

def plot_mic_pairwise_correlation(
    df: pd.DataFrame,
    antibiotic_col: str   = "Antibiotic",
    mic_col: str          = "MIC (mg/L)",
    sign_col: str         = "Measurement sign",
    id_col: str | None    = None,
    min_pairs: int        = 10,
    alpha: float          = 0.05,
    title_prefix: str     = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Spearman ρ heatmap across antibiotics on log₂-transformed MIC values,
    using pairwise complete observations.

    Parameters
    ----------
    id_col        : isolate identifier column. If None, auto-detected from
                    ['BioSample', 'Assembly_Accession', 'Isolate', '#BioSample']
                    and falls back to the DataFrame index.
    min_pairs     : minimum shared isolates required to compute a correlation
    title_prefix  : prepended to the figure title
    save_path     : if provided, figure is saved here (PNG, 150 dpi)

    Returns
    -------
    matplotlib Figure
    """
    # ── Resolve isolate ID column ─────────────────────────────────────────────
    if id_col is None:
        id_col = next(
            (c for c in ["BioSample", "Assembly_Accession", "Isolate", "#BioSample"]
             if c in df.columns),
            None,
        )

    df_clean = df.dropna(subset=[mic_col]).copy()
    df_clean["_log2_mic"] = _log2_mic_series(df_clean, mic_col=mic_col, sign_col=sign_col)

    if id_col:
        mic_wide = (
            df_clean.groupby([id_col, antibiotic_col])["_log2_mic"]
            .mean()
            .unstack(antibiotic_col)
        )
    else:
        mic_wide = (
            df_clean.reset_index()
            .rename(columns={"index": "_idx"})
            .groupby(["_idx", antibiotic_col])["_log2_mic"]
            .mean()
            .unstack(antibiotic_col)
        )

    antibiotics = [
        ab for ab in mic_wide.columns
        if mic_wide[ab].notna().sum() >= min_pairs
    ]
    mic_wide = mic_wide[antibiotics]
    n        = len(antibiotics)

    corr_matrix = pd.DataFrame(np.nan, index=antibiotics, columns=antibiotics)
    pval_matrix = pd.DataFrame(np.nan, index=antibiotics, columns=antibiotics)

    for i, ab1 in enumerate(antibiotics):
        for j, ab2 in enumerate(antibiotics):
            if i == j:
                corr_matrix.loc[ab1, ab2] = 1.0
                pval_matrix.loc[ab1, ab2] = 0.0
                continue
            paired = mic_wide[[ab1, ab2]].dropna()
            if len(paired) < min_pairs:
                continue
            r, p = stats.spearmanr(paired[ab1], paired[ab2])
            corr_matrix.loc[ab1, ab2] = r
            pval_matrix.loc[ab1, ab2] = p

    corr_vals = corr_matrix.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4, n * 1.0)))
    im = ax.imshow(corr_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(antibiotics, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(antibiotics, fontsize=10)

    title = (f"{title_prefix}\n" if title_prefix else "") + \
            "Pairwise MIC Correlation (Spearman ρ)\n" \
            "Log₂-transformed, pairwise complete observations"
    ax.set_title(title, fontsize=12, pad=12)

    for i in range(n):
        for j in range(n):
            val = corr_vals[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=8, color="#999999")
                continue
            p   = pval_matrix.iloc[i, j]
            sig = ""
            if i != j:
                if p < 0.001:  sig = "***"
                elif p < 0.01: sig = "**"
                elif p < 0.05: sig = "*"
            tc = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=9)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax.text(1.18, 0.15, "* p<0.05\n** p<0.01\n*** p<0.001",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4. can_merge
# ═════════════════════════════════════════════════════════════════════════════

def can_merge(
    df: pd.DataFrame,
    antibiotic_a: str,
    antibiotic_b: str,
    antibiotic_col: str        = "Antibiotic",
    phenotype_col: str         = "Resistance phenotype",
    mic_col: str               = "MIC (mg/L)",
    sign_col: str              = "Measurement sign",
    isolation_col: str         = "Isolation type",
    isolation_filter: str | None = "clinical",
    resistant_label: str       = "resistant",
    susceptible_label: str     = "susceptible",
    min_n: int                 = 30,
    rate_threshold: float      = 0.15,
    mwu_effect_threshold: float = 0.3,
    alpha: float               = 0.05,
    save_path: str | None      = None,
) -> dict:
    """
    Determine whether two antibiotics from the same class can be pooled
    into a single model dataset.

    Runs five diagnostic blocks:
        Block 1 — Sample sizes & minimum-n gate
        Block 2 — Resistance rate comparison
        Block 3 — MIC distribution overlap (heatmap)
        Block 4 — Mann-Whitney U on log₂ MIC
        Block 5 — Pooled class balance simulation

    Parameters
    ----------
    isolation_filter : restrict to this isolation type before analysis.
                       Set None to keep all isolation types.
    rate_threshold   : maximum allowed absolute difference in resistance rates
    mwu_effect_threshold : |r| above which a significant MWU result is a flag
    save_path        : if provided, diagnostic figure is saved here

    Returns
    -------
    dict with keys:
        'verdict'      : 'MERGE' | 'MERGE WITH INDICATOR' | 'DO NOT MERGE'
        'antibiotic_a' : str
        'antibiotic_b' : str
        'flags'        : list[str]
        'report'       : str
        'fig'          : matplotlib Figure | None
    """
    lines = []
    flags = []
    add   = lines.append

    def sep(title=""):
        add(f"\n{'─' * 60}")
        if title:
            add(f"  {title}")
        add(f"{'─' * 60}")

    sep("CAN-MERGE DIAGNOSTIC")
    add(f"  Antibiotic A : {antibiotic_a}")
    add(f"  Antibiotic B : {antibiotic_b}")
    add(f"  Isolation    : {isolation_filter or 'all'}")

    # ── Subset ────────────────────────────────────────────────────────────────
    df_ab = df[df[antibiotic_col].isin([antibiotic_a, antibiotic_b])].copy()
    if isolation_filter and isolation_col in df_ab.columns:
        df_ab = df_ab[df_ab[isolation_col] == isolation_filter].copy()

    df_a = df_ab[df_ab[antibiotic_col] == antibiotic_a].copy()
    df_b = df_ab[df_ab[antibiotic_col] == antibiotic_b].copy()

    # ── BLOCK 1: Sample sizes ─────────────────────────────────────────────────
    sep("BLOCK 1 — Sample sizes")
    na, nb = len(df_a), len(df_b)
    add(f"  {antibiotic_a:<20}: n={na}")
    add(f"  {antibiotic_b:<20}: n={nb}")

    if na < min_n or nb < min_n:
        msg = (f"Insufficient samples — {antibiotic_a} n={na}, "
               f"{antibiotic_b} n={nb} (min={min_n})")
        flags.append(msg)
        add(f"\n  ⚠ {msg}")

    # ── BLOCK 2: Resistance rates ─────────────────────────────────────────────
    sep("BLOCK 2 — Resistance rate comparison")

    def resistance_rate(d):
        valid = d[d[phenotype_col].isin([resistant_label, susceptible_label])]
        if len(valid) == 0:
            return np.nan, 0
        return (valid[phenotype_col] == resistant_label).mean(), len(valid)

    rate_a, n_binary_a = resistance_rate(df_a)
    rate_b, n_binary_b = resistance_rate(df_b)
    rate_diff = abs(rate_a - rate_b)

    add(f"  {antibiotic_a:<20}: {rate_a:.1%}  resistant  (n={n_binary_a})")
    add(f"  {antibiotic_b:<20}: {rate_b:.1%}  resistant  (n={n_binary_b})")
    add(f"  Absolute difference  : {rate_diff:.1%}  (threshold={rate_threshold:.0%})")

    rate_ok = rate_diff <= rate_threshold
    add(f"  Result               : {'✅ Within threshold' if rate_ok else '❌ Exceeds threshold'}")
    if not rate_ok:
        flags.append(f"Resistance rates differ by {rate_diff:.1%} "
                     f"({antibiotic_a}={rate_a:.1%}, {antibiotic_b}={rate_b:.1%})")

    # ── BLOCK 3: MIC distribution overlap ────────────────────────────────────
    sep("BLOCK 3 — MIC distribution overlap")

    log2_a = _log2_mic_series(df_a, mic_col=mic_col, sign_col=sign_col)
    log2_b = _log2_mic_series(df_b, mic_col=mic_col, sign_col=sign_col)

    # Guard: empty MIC series
    if len(log2_a) == 0 or len(log2_b) == 0:
        empty = antibiotic_a if len(log2_a) == 0 else antibiotic_b
        msg   = f"No valid MIC values for '{empty}' after filtering."
        flags.append(msg)
        add(f"\n  ⚠ {msg}")
        add("  Skipping Blocks 3–5.  Verdict: DO NOT MERGE")
        report = "\n".join(lines)
        print(report)
        return {"verdict": "DO NOT MERGE", "antibiotic_a": antibiotic_a,
                "antibiotic_b": antibiotic_b, "flags": flags,
                "report": report, "fig": None}

    all_bins = sorted(set(log2_a.tolist() + log2_b.tolist()))
    add(f"  {antibiotic_a:<20}: log₂ MIC range [{min(log2_a)}, {max(log2_a)}]")
    add(f"  {antibiotic_b:<20}: log₂ MIC range [{min(log2_b)}, {max(log2_b)}]")

    overlap    = set(log2_a.unique()) & set(log2_b.unique())
    overlap_ok = len(overlap) >= 3
    add(f"  Shared log₂ bins     : {sorted(overlap)}")
    add(f"  Result               : {'✅ Sufficient overlap' if overlap_ok else '❌ Poor overlap (<3 shared bins)'}")
    if not overlap_ok:
        flags.append("MIC ranges share fewer than 3 log₂ bins — scales not equivalent")

    # ── BLOCK 4: Mann-Whitney U ───────────────────────────────────────────────
    sep("BLOCK 4 — Mann-Whitney U on log₂ MIC")

    mwu_ok = True
    if len(log2_a) >= 3 and len(log2_b) >= 3:
        u, p = stats.mannwhitneyu(log2_a, log2_b, alternative="two-sided")
        r    = abs(1 - (2 * u) / (len(log2_a) * len(log2_b)))
        mag  = ("negligible" if r < 0.1 else "small" if r < 0.3
                else "moderate" if r < 0.5 else "large")
        sig         = p < alpha
        large_effect = r >= mwu_effect_threshold
        mwu_ok      = not (sig and large_effect)
        add(f"  U={u:.1f},  p={p:.3e},  |r|={r:.4f} ({mag} effect)")
        add(f"  Significant          : {'Yes ⚠' if sig else 'No'}")
        add(f"  Result               : {'✅ Distributions compatible' if mwu_ok else '❌ Significantly different with moderate+ effect'}")
        if not mwu_ok:
            flags.append(f"MIC distributions significantly different "
                         f"(p={p:.2e}, |r|={r:.3f}, {mag} effect)")
    else:
        add("  Insufficient data for Mann-Whitney test.")
        r, p, mwu_ok = np.nan, np.nan, None

    # ── BLOCK 5: Pooled class balance ─────────────────────────────────────────
    sep("BLOCK 5 — Pooled class balance simulation")

    pooled_binary = df_ab[df_ab[phenotype_col].isin([resistant_label, susceptible_label])]
    pooled_rate   = (pooled_binary[phenotype_col] == resistant_label).mean()
    pooled_n      = len(pooled_binary)
    balance_ratio = (min(pooled_rate, 1 - pooled_rate) /
                     max(pooled_rate, 1 - pooled_rate)) if pooled_n > 0 else 0

    add(f"  Pooled n             : {pooled_n}")
    add(f"  Pooled resistance    : {pooled_rate:.1%}")
    add(f"  Balance ratio        : {balance_ratio:.3f}  (1.0=perfect, <0.25=severe)")

    balance_ok = balance_ratio >= 0.25
    add(f"  Result               : {'✅ Acceptable' if balance_ok else '❌ Severe class imbalance'}")
    if not balance_ok:
        flags.append(f"Pooled class imbalance severe (ratio={balance_ratio:.3f})")

    # ── BLOCK 6: Verdict ──────────────────────────────────────────────────────
    sep("BLOCK 6 — Verdict")

    hard_fail = not rate_ok or not overlap_ok or not balance_ok
    soft_fail = (mwu_ok is False)

    if hard_fail:
        verdict = "DO NOT MERGE"
    elif soft_fail:
        verdict = "MERGE WITH INDICATOR"
    else:
        verdict = "MERGE"

    GUIDANCE = {
        "MERGE":
            "Pool records directly. Optionally add a drug indicator column.",
        "MERGE WITH INDICATOR":
            "Pool records but add a binary 'drug' feature so the model\n"
            "    can learn drug-specific MIC offsets.",
        "DO NOT MERGE":
            "Build separate models per antibiotic. Report class-level\n"
            "    performance as the average of individual drug models.",
    }
    add(f"\n  {'★' * 3}  VERDICT: {verdict}  {'★' * 3}")
    add(f"\n  Flags ({len(flags)}):")
    for f in flags:
        add(f"    • {f}")
    if not flags:
        add("    None — all checks passed.")
    add(f"\n  Guidance: {GUIDANCE[verdict]}")

    report = "\n".join(lines)
    print(report)

    # ── Figure ────────────────────────────────────────────────────────────────
    COLORS_V = {"MERGE": "#4dac26",
                "MERGE WITH INDICATOR": "#f1a340",
                "DO NOT MERGE": "#d7191c"}

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Can-Merge Diagnostic: {antibiotic_a} vs {antibiotic_b}\n"
        f"Verdict: {verdict}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A — Resistance rates
    ax_rate = fig.add_subplot(gs[0, 0])
    bars = ax_rate.bar(
        [antibiotic_a, antibiotic_b], [rate_a, rate_b],
        color=["#2166ac", "#d6604d"], alpha=0.85, edgecolor="white",
    )
    ax_rate.axhline(rate_a + rate_threshold, color="grey",
                    linestyle="--", linewidth=1, label=f"±{rate_threshold:.0%} band")
    ax_rate.axhline(max(0, rate_a - rate_threshold), color="grey",
                    linestyle="--", linewidth=1)
    for bar, val in zip(bars, [rate_a, rate_b]):
        ax_rate.text(bar.get_x() + bar.get_width() / 2, val / 2,
                     f"{val:.1%}", ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")
    ax_rate.set_ylim(0, 1.1)
    ax_rate.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1, decimals=0))
    ax_rate.set_title("A  Resistance rates", fontsize=10, loc="left")
    ax_rate.spines[["top", "right"]].set_visible(False)
    ax_rate.legend(fontsize=7, frameon=False)

    # Panel B — MIC heatmap
    ax_heat = fig.add_subplot(gs[0, 1:])
    count_a   = pd.Series(log2_a).value_counts().reindex(all_bins, fill_value=0)
    count_b   = pd.Series(log2_b).value_counts().reindex(all_bins, fill_value=0)
    heat_data = np.array([count_a.values, count_b.values], dtype=float)
    norm      = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=heat_data.max())
    im        = ax_heat.imshow(heat_data, cmap="YlOrRd", norm=norm, aspect="auto")
    ax_heat.set_xticks(range(len(all_bins)))
    ax_heat.set_xticklabels([_log2_to_mic_label(b) for b in all_bins],
                             rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks([0, 1])
    ax_heat.set_yticklabels([antibiotic_a, antibiotic_b], fontsize=9)
    ax_heat.set_xlabel("MIC (mg/L) — log₂ bins", fontsize=9)
    ax_heat.set_title("B  MIC distribution overlap", fontsize=10, loc="left")
    for ri in range(2):
        for ci in range(len(all_bins)):
            n = int(heat_data[ri, ci])
            if n == 0:
                continue
            tc = "white" if norm(n) > 0.55 else "black"
            ax_heat.text(ci, ri, str(n), ha="center", va="center",
                         fontsize=7, color=tc, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label="Count (√ scale)")

    # Panel C — KDE
    ax_kde = fig.add_subplot(gs[1, 0])
    for log2_vals, label, color in [
        (log2_a, antibiotic_a, "#2166ac"),
        (log2_b, antibiotic_b, "#d6604d"),
    ]:
        vals = np.array(log2_vals, dtype=float)
        if len(vals) > 2:
            kde = stats.gaussian_kde(vals, bw_method=0.4)
            xs  = np.linspace(vals.min() - 1, vals.max() + 1, 300)
            ax_kde.plot(xs, kde(xs), label=label, color=color, linewidth=2)
            ax_kde.fill_between(xs, kde(xs), alpha=0.15, color=color)
    ax_kde.set_xlabel("log₂ MIC", fontsize=9)
    ax_kde.set_ylabel("Density", fontsize=9)
    ax_kde.set_title("C  log₂ MIC density", fontsize=10, loc="left")
    ax_kde.legend(fontsize=8, frameon=False)
    ax_kde.spines[["top", "right"]].set_visible(False)

    # Panel D — Phenotype breakdown
    ax_pheno  = fig.add_subplot(gs[1, 1])
    pheno_order  = [susceptible_label, "intermediate", resistant_label, "not defined"]
    pheno_colors = {"susceptible": "#4dac26", "intermediate": "#808080",
                    "resistant": "#d7191c", "not defined": "#cccccc"}
    x       = np.array([0, 1])
    bottoms = np.zeros(2)
    for ph in pheno_order:
        vals = np.array([
            (d[phenotype_col] == ph).sum() / len(d) if len(d) else 0
            for d in [df_a, df_b]
        ])
        ax_pheno.bar(x, vals, bottom=bottoms,
                     color=pheno_colors.get(ph, "#aaaaaa"),
                     label=ph, alpha=0.85, edgecolor="white")
        for xi, (v, bot) in enumerate(zip(vals, bottoms)):
            if v > 0.04:
                ax_pheno.text(xi, bot + v / 2, f"{v:.0%}",
                              ha="center", va="center", fontsize=8,
                              color="white", fontweight="bold")
        bottoms += vals
    ax_pheno.set_xticks([0, 1])
    ax_pheno.set_xticklabels([antibiotic_a, antibiotic_b], fontsize=9)
    ax_pheno.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_pheno.set_title("D  Phenotype breakdown", fontsize=10, loc="left")
    ax_pheno.legend(fontsize=7, frameon=False, loc="upper right")
    ax_pheno.spines[["top", "right"]].set_visible(False)

    # Panel E — Verdict card
    ax_card = fig.add_subplot(gs[1, 2])
    ax_card.axis("off")
    checks = [
        ("Resistance rate ≤15% diff", rate_ok),
        ("MIC ranges overlap ≥3 bins", overlap_ok),
        ("MWU: no large sig. diff.",   mwu_ok if mwu_ok is not None else True),
        ("Pooled balance ≥0.25",       balance_ok),
    ]
    card_lines = [f"  VERDICT: {verdict}\n"]
    for label, ok in checks:
        card_lines.append(f"  {'✅' if ok else '❌'}  {label}")
    card_lines.append(f"\n  Flags: {len(flags)}")
    ax_card.text(0.05, 0.95, "\n".join(card_lines),
                 transform=ax_card.transAxes, fontsize=9,
                 va="top", ha="left", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=COLORS_V[verdict],
                           alpha=0.15, edgecolor=COLORS_V[verdict]))
    ax_card.set_title("E  Summary", fontsize=10, loc="left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return {"verdict": verdict, "antibiotic_a": antibiotic_a,
            "antibiotic_b": antibiotic_b, "flags": flags,
            "report": report, "fig": fig}


# ═════════════════════════════════════════════════════════════════════════════
# 5. can_merge_class
# ═════════════════════════════════════════════════════════════════════════════

def can_merge_class(
    df: pd.DataFrame,
    antibiotic_class: str,
    antibiotic_class_col: str = "Antibiotic_Class",
    antibiotic_col: str       = "Antibiotic",
    save_dir: str | None      = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run can_merge() for every pairwise combination of antibiotics within
    `antibiotic_class` and return a summary DataFrame of verdicts.

    Parameters
    ----------
    antibiotic_class     : e.g. 'Carbapenem'
    antibiotic_class_col : column holding antibiotic class labels
    save_dir             : if provided, each diagnostic figure is saved here
                           as 'can_merge_<A>_vs_<B>.png'
    **kwargs             : passed through to can_merge()

    Returns
    -------
    pd.DataFrame with columns:
        Antibiotic A | Antibiotic B | Verdict | Flags | Flag detail
    """
    class_drugs = (
        df[df[antibiotic_class_col] == antibiotic_class][antibiotic_col]
        .dropna().unique().tolist()
    )

    if len(class_drugs) < 2:
        print(f"Only {len(class_drugs)} antibiotic(s) found in class "
              f"'{antibiotic_class}' — nothing to compare.")
        return pd.DataFrame()

    pairs = list(combinations(sorted(class_drugs), 2))
    print(f"\nClass '{antibiotic_class}': {len(class_drugs)} antibiotics, "
          f"{len(pairs)} pairwise comparisons.\n")

    rows = []
    for ab_a, ab_b in pairs:
        sp = None
        if save_dir:
            import os
            sp = os.path.join(save_dir, f"can_merge_{ab_a}_vs_{ab_b}.png")
        result = can_merge(
            df             = df,
            antibiotic_a   = ab_a,
            antibiotic_b   = ab_b,
            antibiotic_col = antibiotic_col,
            save_path      = sp,
            **kwargs,
        )
        rows.append({
            "Antibiotic A": ab_a,
            "Antibiotic B": ab_b,
            "Verdict":      result["verdict"],
            "Flags":        len(result["flags"]),
            "Flag detail":  " | ".join(result["flags"]),
        })

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    return summary

def plot_mic_violin(
    plot_data: pd.DataFrame,
    species_name: str = "",
    ab_class: str = "",
) -> plt.Figure:
    """
    Violin + strip plot of MIC distributions per antibiotic,
    coloured by resistance phenotype category.
    """
    df = plot_data.dropna(subset=["MIC (mg/L)"]).copy()
    df["log2_MIC"] = np.log2(df["MIC (mg/L)"])

    antibiotics = sorted(df["Antibiotic"].unique())
    n = len(antibiotics)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, abx in zip(axes, antibiotics):
        sub = df[df["Antibiotic"] == abx]
        present = [p for p in PHENO_ORDER if p in sub["Resistance phenotype"].values]

        sns.violinplot(
            data=sub, x="Resistance phenotype", y="log2_MIC",
            palette={p: PALETTE[p] for p in present},
            order=present, inner=None, linewidth=1.2, cut=0, ax=ax,
        )
        sns.stripplot(
            data=sub, x="Resistance phenotype", y="log2_MIC",
            order=present, color="white", edgecolor="0.3",
            linewidth=0.6, size=5, jitter=True, alpha=0.85, ax=ax,
        )

        ax.set_title(abx.capitalize(), fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_xticklabels(
            [p.replace("-", "-\n") if len(p) > 12 else p for p in present],
            fontsize=8.5, rotation=0,
        )

        y_bottom = ax.get_ylim()[0]
        for i, pheno in enumerate(present):
            n_pts = (sub["Resistance phenotype"] == pheno).sum()
            ax.text(i, y_bottom - 0.2, f"n={n_pts}",
                    ha="center", va="top", fontsize=8, color="0.4")

    axes[0].set_ylabel("MIC (mg/L)  —  log₂ scale", fontsize=11)
    yticks = axes[0].get_yticks()
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels([f"{2**t:.3g}" for t in yticks], fontsize=9)

    all_present = [p for p in PHENO_ORDER if p in df["Resistance phenotype"].values]
    fig.legend(
        handles=[mpatches.Patch(color=PALETTE[p], label=p.capitalize())
                 for p in all_present],
        title="Phenotype", loc="upper right",
        frameon=True, fontsize=9, title_fontsize=10,
        bbox_to_anchor=(1.01, 1),
    )
    fig.suptitle(
        f"{species_name} MIC Distribution by Antibiotic and Resistance Phenotype"
        f" for {ab_class}",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig