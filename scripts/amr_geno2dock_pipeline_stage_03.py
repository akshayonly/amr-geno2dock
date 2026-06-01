#!/usr/bin/env python3
# coding: utf-8
# AUTHOR: Akshay Shirsath (2026)
# =============================================================================
# AMR-Geno2Dock — AMRFinderPlus Runner (Stage 03)
# =============================================================================
#
# PURPOSE:
#   Run AMRFinderPlus on every genome FASTA produced by Stage 02 and
#   collect per-sample TSV outputs ready for feature matrix construction
#   in Stage 04.
#
# INPUT  (from Stage 02):
#   genomes/
#   └── <accession>.fna          — one FASTA per assembly
#
# OUTPUT:
#   amrfinder/
#   ├── <accession>.tsv          — raw AMRFinderPlus output per genome
#   ├── failed_samples.txt       — accessions that failed (if any)
#   ├── amrfinder.log            — full run log
#   └── amrfinder_run_report.tsv — per-sample status + gene count summary
#
# PIPELINE PHASES:
#   0  — Imports, constants & CLI
#   1  — Dependency & database validation
#   2  — Genome discovery
#   3  — Parallel AMRFinderPlus execution
#   4  — Output validation & run report
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0 — Imports, constants & CLI
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

# Valid AMRFinderPlus --organism values (as of AMRFinderPlus v3.x / db 2024+).
# Pass exactly one of these; capitalisation matters.
VALID_ORGANISMS = {
    "Acinetobacter_baumannii",
    "Burkholderia_cepacia",
    "Burkholderia_pseudomallei",
    "Campylobacter",
    "Clostridioides_difficile",
    "Enterobacter_cloacae",
    "Enterococcus_faecalis",
    "Enterococcus_faecium",
    "Escherichia",
    "Klebsiella_oxytoca",
    "Klebsiella_pneumoniae",
    "Neisseria_gonorrhoeae",
    "Neisseria_meningitidis",
    "Pseudomonas_aeruginosa",
    "Salmonella",
    "Staphylococcus_aureus",
    "Staphylococcus_pseudintermedius",
    "Streptococcus_agalactiae",
    "Streptococcus_pneumoniae",
    "Streptococcus_pyogenes",
    "Vibrio_cholerae",
    "Acinetobacter",  # legacy alias still accepted
}

# Default for this pipeline (Acinetobacter baumannii × tobramycin)
DEFAULT_ORGANISM = "Acinetobacter_baumannii"


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0B — CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amr_geno2dock_pipeline_stage_03.py",
        description=(
            "Stage 03 — Run AMRFinderPlus on genome FASTAs from Stage 02 "
            "and produce per-sample TSV outputs for Stage 04 feature matrix "
            "construction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "-i",
        "--genomes-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory of genome FASTA files produced by Stage 02.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("amrfinder"),
        metavar="DIR",
        help="Directory for AMRFinderPlus TSV outputs and logs.",
    )
    parser.add_argument(
        "--ext",
        default=".fna",
        metavar="EXT",
        help="Genome FASTA file extension to glob for.",
    )

    # ── AMRFinderPlus options ─────────────────────────────────────────────────
    parser.add_argument(
        "--organism",
        default=DEFAULT_ORGANISM,
        metavar="NAME",
        help=(
            "AMRFinderPlus --organism value. Must match the controlled list "
            f"(e.g. {DEFAULT_ORGANISM}). Run `amrfinder --list_organisms` "
            "for the full set."
        ),
    )
    parser.add_argument(
        "--no-plus",
        action="store_true",
        help=(
            "Disable the --plus flag. By default --plus is passed, which adds "
            "point mutation screening for the chosen organism."
        ),
    )
    parser.add_argument(
        "--update-db",
        action="store_true",
        help=(
            "Run `amrfinder --update` before the batch to pull the latest "
            "AMRFinderPlus database. Recommended if the local database is "
            "more than 4 weeks old."
        ),
    )
    parser.add_argument(
        "--ident-min",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Minimum identity fraction for BLAST hits (0.0–1.0). "
            "Passed as --ident_min to amrfinder. Leave unset to use the "
            "AMRFinderPlus default (0.9)."
        ),
    )
    parser.add_argument(
        "--coverage-min",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Minimum coverage fraction for BLAST hits (0.0–1.0). "
            "Passed as --coverage_min to amrfinder. Leave unset to use the "
            "AMRFinderPlus default (0.5)."
        ),
    )

    # ── Parallelism ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Number of parallel AMRFinderPlus jobs. Each job uses --threads "
            "CPU threads, so total CPU usage = workers × threads. "
            "Keep workers × threads ≤ available cores."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        metavar="N",
        help="CPU threads per AMRFinderPlus job.",
    )

    # ── Run control ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run AMRFinderPlus even if the output TSV already exists.",
    )
    parser.add_argument(
        "--no-progressbar",
        action="store_true",
        help="Suppress tqdm progress bar (useful in cluster log files).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 1 — Dependency & database validation
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(out_dir: Path, verbose: bool) -> logging.Logger:
    """
    Configure dual-sink logging: stderr stream + persistent file.
    The log file is written to <out_dir>/amrfinder.log.
    """
    log = logging.getLogger("stage_03")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(out_dir / "amrfinder.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def check_amrfinder(log: logging.Logger) -> str:
    """
    Confirm amrfinder is on PATH and return its version string.
    Exits with a clear install message if not found.
    """
    if shutil.which("amrfinder") is None:
        log.error(
            "amrfinder not found on PATH.\n"
            "Install AMRFinderPlus:\n"
            "  conda install -c bioconda ncbi-amrfinderplus\n"
            "  or follow: https://github.com/ncbi/amr/wiki/Installing-AMRFinder"
        )
        sys.exit(1)

    result = subprocess.run(
        ["amrfinder", "--version"],
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip() or result.stderr.strip()
    log.info(f"AMRFinderPlus version: {version}")
    return version


def validate_organism(organism: str, log: logging.Logger) -> None:
    """
    Warn if the organism string is not in the known-valid set.
    AMRFinderPlus itself will error on a bad value, but this surfaces
    the problem earlier with a more useful message.
    """
    if organism not in VALID_ORGANISMS:
        log.warning(
            f"'{organism}' is not in the known organism list. "
            "If AMRFinderPlus rejects it, run `amrfinder --list_organisms` "
            "to see all valid values and pass the exact string."
        )


def update_amrfinder_database(log: logging.Logger) -> None:
    log.info("Updating AMRFinderPlus database …")
    result = subprocess.run(
        ["amrfinder", "--update"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Known bug: update reports failure but db may be valid
        if "version.txt" in stderr and "Cannot open" in stderr:
            log.warning(
                "AMRFinderPlus update returned a version.txt error — "
                "this is a known bug. Checking if a valid database exists …"
            )
            # Verify a db dir with version.txt actually exists
            db_base = Path(
                "/home/zeus/miniconda3/envs/cloudspace/share/amrfinderplus/data"
            )
            valid_dbs = [d for d in db_base.glob("*/") if (d / "version.txt").exists()]
            if valid_dbs:
                latest = sorted(valid_dbs)[-1]
                log.info(f"Using existing database: {latest}")
                return
        log.error(f"Database update failed:\n{stderr}")
        sys.exit(1)
    log.info("Database update complete.")


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 2 — Genome discovery
# ─────────────────────────────────────────────────────────────────────────────


def discover_genomes(genomes_dir: Path, ext: str, log: logging.Logger) -> list[Path]:
    """
    Glob for FASTA files in genomes_dir. Exits if none are found.
    Filters out zero-byte files to prevent pipeline failures.
    """
    if not genomes_dir.exists():
        log.error(f"Genomes directory not found: {genomes_dir}")
        sys.exit(1)

    fastas = sorted(genomes_dir.glob(f"*{ext}"))

    if not fastas:
        log.error(f"No {ext} files found in {genomes_dir}")
        sys.exit(1)

    # Filter out empty files before passing to the worker pool
    empty = [f for f in fastas if f.stat().st_size == 0]
    if empty:
        log.warning(
            f"{len(empty)} empty FASTA file(s) found — these will be skipped:\n"
            + "\n".join(f"  {f.name}" for f in empty)
        )

    valid_fastas = [f for f in fastas if f.stat().st_size > 0]
    log.info(f"Genomes discovered (valid for processing): {len(valid_fastas)}")
    return valid_fastas


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 3 — Parallel AMRFinderPlus execution
# ─────────────────────────────────────────────────────────────────────────────


def build_amrfinder_cmd(
    fasta: Path,
    out_path: Path,
    organism: str,
    threads: int,
    plus: bool,
    ident_min: float | None,
    coverage_min: float | None,
) -> list[str]:
    """
    Assemble the amrfinder command for one genome.
    """
    cmd = [
        "amrfinder",
        "-n",
        str(fasta),
        "--organism",
        organism,
        "--threads",
        str(threads),
        "-o",
        str(out_path),
        "--print_node",  # adds Gene hierarchy node column
    ]
    if plus:
        cmd.append("--plus")
    if ident_min is not None:
        cmd += ["--ident_min", str(ident_min)]
    if coverage_min is not None:
        cmd += ["--coverage_min", str(coverage_min)]

    return cmd


def count_genes(filepath: Path) -> int:
    """
    Helper function to efficiently count lines in the TSV minus the header.
    Returns -1 if the file cannot be read/parsed, indicating corruption.
    """
    try:
        with open(filepath, "r") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return -1


def run_one(
    fasta: Path,
    out_dir: Path,
    organism: str,
    threads: int,
    plus: bool,
    ident_min: float | None,
    coverage_min: float | None,
    force: bool,
) -> dict:
    """
    Run AMRFinderPlus on a single genome FASTA.
    """
    accession = fasta.stem
    out_path = out_dir / f"{accession}.tsv"
    tmp_path = out_path.with_suffix(".tmp")

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        n_genes = count_genes(out_path)
        if n_genes >= 0:
            return {
                "accession": accession,
                "status": "cached",
                "n_genes": n_genes,
                "stderr": None,
            }
        # If n_genes is -1, the file is corrupted. We ignore the cache and run.

    # ── Run ───────────────────────────────────────────────────────────────────
    cmd = build_amrfinder_cmd(
        fasta, tmp_path, organism, threads, plus, ident_min, coverage_min
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    # ── Handle failure ────────────────────────────────────────────────────────
    if result.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "accession": accession,
            "status": "failed",
            "n_genes": 0,
            "stderr": result.stderr.strip(),
        }

    # ── Validate output ───────────────────────────────────────────────────────
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        return {
            "accession": accession,
            "status": "failed",
            "n_genes": 0,
            "stderr": "AMRFinderPlus produced no output.",
        }

    # Count gene hits (header row excluded)
    n_genes = count_genes(tmp_path)
    if n_genes == -1:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "accession": accession,
            "status": "failed",
            "n_genes": 0,
            "stderr": "AMRFinderPlus output was unreadable.",
        }

    # ── Atomic rename ─────────────────────────────────────────────────────────
    tmp_path.rename(out_path)

    return {
        "accession": accession,
        "status": "done",
        "n_genes": n_genes,
        "stderr": None,
    }


def run_parallel(
    fastas: list[Path],
    out_dir: Path,
    organism: str,
    workers: int,
    threads: int,
    plus: bool,
    ident_min: float | None,
    coverage_min: float | None,
    force: bool,
    show_progress: bool,
    log: logging.Logger,
) -> list[dict]:
    """
    Dispatch run_one() across a ThreadPoolExecutor.
    """
    cpu_total = os.cpu_count() or 1
    if workers * threads > cpu_total:
        log.warning(
            f"workers ({workers}) × threads ({threads}) = {workers * threads} "
            f"exceeds detected CPU count ({cpu_total}). "
            "Consider reducing --workers or --threads to avoid oversubscription."
        )

    log.info(
        f"Launching {workers} parallel job(s) × {threads} thread(s) each "
        f"| plus={'on' if plus else 'off'} | organism={organism}"
    )

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                run_one,
                fasta=fasta,
                out_dir=out_dir,
                organism=organism,
                threads=threads,
                plus=plus,
                ident_min=ident_min,
                coverage_min=coverage_min,
                force=force,
            ): fasta
            for fasta in fastas
        }

        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="AMRFinderPlus",
            unit="genome",
            disable=not show_progress,
        ):
            res = future.result()
            results.append(res)

            if res["status"] == "failed":
                log.error(f"FAILED  {res['accession']}\n  {res['stderr']}")
            elif res["status"] == "cached":
                log.debug(f"CACHED  {res['accession']}  ({res['n_genes']} genes)")
            else:
                log.debug(f"DONE    {res['accession']}  ({res['n_genes']} genes)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 4 — Output validation & run report
# ─────────────────────────────────────────────────────────────────────────────


def write_run_report(results: list[dict], out_dir: Path, log: logging.Logger) -> None:
    """
    Write a per-accession TSV report and print a human-readable summary.
    """
    report = pd.DataFrame(results)
    report["tsv_path"] = report.apply(
        lambda r: (
            str(out_dir / f"{r['accession']}.tsv") if r["status"] != "failed" else ""
        ),
        axis=1,
    )
    report = report[["accession", "status", "n_genes", "tsv_path", "stderr"]]
    report = report.sort_values("accession").reset_index(drop=True)

    report_path = out_dir / "amrfinder_run_report.tsv"
    report.to_csv(report_path, sep="\t", index=False)
    log.info(f"Run report written → {report_path}")

    # ── Failed samples file ───────────────────────────────────────────────────
    failed = report[report["status"] == "failed"]["accession"].tolist()
    if failed:
        failed_path = out_dir / "failed_samples.txt"
        failed_path.write_text("\n".join(sorted(failed)))
        log.warning(f"Failed samples ({len(failed)}) written → {failed_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    counts = report["status"].value_counts().to_dict()
    n_total = len(report)
    n_done = counts.get("done", 0)
    n_cached = counts.get("cached", 0)
    n_failed = counts.get("failed", 0)
    n_ok = n_done + n_cached
    pct_ok = 100 * n_ok / n_total if n_total else 0

    ok_genes = report[report["status"] != "failed"]["n_genes"]
    median_g = ok_genes.median() if len(ok_genes) else 0
    max_g = ok_genes.max() if len(ok_genes) else 0
    min_g = ok_genes.min() if len(ok_genes) else 0

    print("\n" + "═" * 60)
    print("  STAGE 03 — AMRFINDERPLUS RUN REPORT")
    print("═" * 60)
    print(f"  Total genomes       : {n_total}")
    print(f"  ✅ Successful        : {n_ok}  ({pct_ok:.1f}%)")
    print(f"       — newly run     : {n_done}")
    print(f"       — cached        : {n_cached}")
    if n_failed:
        print(f"  ⚠  Failed            : {n_failed}")
    print(f"\n  AMR genes detected (successful genomes):")
    print(f"       median          : {median_g:.0f}")
    print(f"       min / max       : {min_g} / {max_g}")
    print(f"\n  Outputs  → {out_dir}/")
    print(f"  Report   → {report_path}")
    print("═" * 60)
    if n_failed:
        print(
            f"\n  Re-run failed genomes:\n"
            f"    python amr_geno2dock_pipeline_stage_03.py \\\n"
            f"        --genomes-dir <genomes_dir_containing_failed> \\\n"
            f"        --out-dir {out_dir}\n"
            f"  (Only missing TSVs will be re-run; cached outputs are skipped.)"
        )
    print(
        f"\n  Next step (Stage 04):\n"
        f"    Build gene presence/absence matrix from {out_dir}/*.tsv\n"
        f"    rows = Assembly_Accession  |  columns = AMR gene names  |  values = 0/1\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
## MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(args.out_dir, args.verbose)

    # ── Phase 1: Validate dependencies ────────────────────────────────────────
    check_amrfinder(log)
    validate_organism(args.organism, log)

    if args.update_db:
        update_amrfinder_database(log)

    # ── Phase 2: Discover genomes ─────────────────────────────────────────────
    fastas = discover_genomes(args.genomes_dir, args.ext, log)

    # ── Phase 3: Run in parallel ──────────────────────────────────────────────
    results = run_parallel(
        fastas=fastas,
        out_dir=args.out_dir,
        organism=args.organism,
        workers=args.workers,
        threads=args.threads,
        plus=not args.no_plus,
        ident_min=args.ident_min,
        coverage_min=args.coverage_min,
        force=args.force,
        show_progress=not args.no_progressbar,
        log=log,
    )

    # ── Phase 4: Report ───────────────────────────────────────────────────────
    write_run_report(results, args.out_dir, log)


if __name__ == "__main__":
    main()
