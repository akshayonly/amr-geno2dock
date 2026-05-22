#!/usr/bin/env python3
# coding: utf-8
# AUTHOR: Akshay Shirsath (2026)
# =============================================================================
# AMR-Geno2Dock — Genome Download Pipeline (Stage 02)
# =============================================================================
#
# PURPOSE:
#   Download genome FASTA files from NCBI for all assembly accessions
#   exported by Stage 01. Organises outputs into a clean directory
#   structure ready for AMRFinderPlus (Stage 03).
#
# INPUT  (from Stage 01):
#   ab_<antibiotic>_accessions.txt  — one GCA_*/GCF_* accession per line
#
# OUTPUT:
#   genomes/
#   ├── <accession>.fna             — one FASTA per assembly
#   └── ...
#   download_report.tsv             — per-accession status log
#
# PIPELINE PHASES:
#   0  — Imports, config & CLI
#   1  — Validate NCBI Datasets CLI availability
#   2  — Load & deduplicate accession list
#   3  — Batch download via `datasets download genome accession`
#   4  — Unzip, rename & organise FASTA files
#   5  — Quality check & report
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0 — Imports, config & CLI
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASETS_CMD   = "datasets"          # NCBI Datasets CLI executable
BATCH_SIZE     = 200                 # accessions per download call
RETRY_LIMIT    = 3                   # retries on transient failures
RETRY_DELAY    = 10                  # seconds between retries


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 0B — CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amr_geno2dock_pipeline_stage_02.py",
        description=(
            "Stage 02 — Download genome FASTA files from NCBI for all assembly "
            "accessions produced by Stage 01."
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        metavar="ACCESSIONS_TXT",
        help="Path to accession list file (one GCA_*/GCF_* per line). "
             "Produced by Stage 01 as ab_<antibiotic>_accessions.txt",
    )
    parser.add_argument(
        "-o", "--outdir",
        type=Path,
        default=Path("genomes"),
        metavar="DIR",
        help="Output directory for FASTA files (default: ./genomes/)",
    )
    parser.add_argument(
        "--tmp",
        type=Path,
        default=Path("tmp_downloads"),
        metavar="DIR",
        help="Temporary directory for zip staging (default: ./tmp_downloads/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"Accessions per NCBI Datasets call (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--file-type",
        choices=["fasta", "gbff", "gff3", "gtf", "seq-report"],
        default="fasta",
        help="Genome file type to download (default: fasta)",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=["genome", "rna", "protein", "cds", "gff3", "gtf", "gbff", "seq-report"],
        default=["genome"],
        help="Data types to include in download (default: genome)",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Restrict to NCBI reference/representative assemblies only",
    )
    parser.add_argument(
        "--no-progressbar",
        action="store_true",
        help="Suppress tqdm progress bars (useful for log files)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY",
        help=(
            "NCBI API key (optional). Raises rate limit from 3 → 10 req/s. "
            "If not passed, read from NCBI_API_KEY environment variable. "
            "Obtain a key at: https://www.ncbi.nlm.nih.gov/account/"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 1 — Validate NCBI Datasets CLI availability
# ─────────────────────────────────────────────────────────────────────────────

def check_datasets_cli() -> str:
    """
    Confirm that the NCBI Datasets CLI (`datasets`) is on PATH and
    return its version string.

    Install instructions if missing:
        conda install -c conda-forge ncbi-datasets-cli
        # or
        pip install ncbi-datasets-pylib   # Python wrapper (alternative)
        # or download binary:
        # https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/
    """
    try:
        result = subprocess.run(
            [DATASETS_CMD, "version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version = result.stdout.strip()
        log.info(f"NCBI Datasets CLI found: {version}")
        return version
    except FileNotFoundError:
        log.error(
            "NCBI Datasets CLI not found on PATH.\n"
            "Install with:  conda install -c conda-forge ncbi-datasets-cli\n"
            "or download from: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        log.error(f"Datasets CLI error: {e.stderr}")
        sys.exit(1)


def resolve_api_key(cli_key: str | None) -> str | None:
    """
    Resolve NCBI API key with priority:
      1. --api-key CLI argument
      2. NCBI_API_KEY environment variable
      3. None  (anonymous, 3 req/s limit)
    """
    key = cli_key or os.environ.get("NCBI_API_KEY")
    if key:
        log.info("NCBI API key found — rate limit: 10 req/s")
    else:
        log.info("No NCBI API key — rate limit: 3 req/s (sufficient for small datasets)")
    return key



# ─────────────────────────────────────────────────────────────────────────────

def load_accessions(path: Path) -> list[str]:
    """
    Read one assembly accession per line from the Stage 01 export file.
    Strips whitespace, drops blanks and comment lines, deduplicates.
    """
    if not path.exists():
        log.error(f"Accession file not found: {path}")
        sys.exit(1)

    raw = path.read_text().splitlines()
    accessions = [
        line.strip()
        for line in raw
        if line.strip() and not line.startswith("#")
    ]

    before = len(accessions)
    accessions = list(dict.fromkeys(accessions))   # deduplicate, preserve order
    after  = len(accessions)

    if before != after:
        log.warning(f"Removed {before - after} duplicate accessions.")

    log.info(f"Loaded {after} unique accessions from {path}")
    return accessions


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 3 — Batch download via NCBI Datasets CLI
# ─────────────────────────────────────────────────────────────────────────────

def _run_datasets_download(
    accessions: list[str],
    zip_path: Path,
    include: list[str],
    reference_only: bool,
    api_key: str | None = None,
) -> bool:
    """
    Run `datasets download genome accession` for a batch of accessions.
    Returns True on success, False on failure.

    Credentials:
      - No API key required for anonymous downloads.
      - Providing --api-key raises the NCBI rate limit from 3 → 10 req/s.
      - Key priority: argument → NCBI_API_KEY env var → anonymous.

    CLI reference:
        datasets download genome accession \
            GCF_000001405.40 GCF_000750555.1 \
            --include genome \
            --filename batch.zip \
            --no-progressbar \
            --api-key <KEY>            # optional
    """
    cmd = [
        DATASETS_CMD, "download", "genome", "accession",
        *accessions,
        "--include", *include,
        "--filename", str(zip_path),
        "--no-progressbar",          # suppress nested CLI bars; we handle ours
    ]
    if reference_only:
        cmd.append("--reference")
    if api_key:
        cmd += ["--api-key", api_key]

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            log.debug(result.stdout.strip())
            return True
        except subprocess.CalledProcessError as e:
            log.warning(
                f"Download attempt {attempt}/{RETRY_LIMIT} failed "
                f"(exit {e.returncode}): {e.stderr.strip()[:200]}"
            )
            if attempt < RETRY_LIMIT:
                log.info(f"Retrying in {RETRY_DELAY}s …")
                time.sleep(RETRY_DELAY)
    return False


def download_batches(
    accessions: list[str],
    tmp_dir: Path,
    include: list[str],
    batch_size: int,
    reference_only: bool,
    show_progress: bool,
    api_key: str | None = None,
) -> dict[str, str]:
    """
    Download all accessions in batches. Returns a dict mapping
    zip_path → batch_id for subsequent extraction.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        accessions[i : i + batch_size]
        for i in range(0, len(accessions), batch_size)
    ]
    log.info(f"Downloading {len(accessions)} accessions in {len(batches)} batch(es) "
             f"(batch_size={batch_size})")

    batch_map: dict[str, list[str]] = {}   # zip_path_str → accession list

    iterator = tqdm(batches, desc="Batches", unit="batch", disable=not show_progress)
    for idx, batch in enumerate(iterator):
        zip_path = tmp_dir / f"batch_{idx:04d}.zip"
        success  = _run_datasets_download(batch, zip_path, include, reference_only, api_key)
        if success:
            batch_map[str(zip_path)] = batch
            log.debug(f"Batch {idx}: OK → {zip_path}")
        else:
            log.error(f"Batch {idx}: FAILED after {RETRY_LIMIT} attempts. "
                      f"Accessions: {batch[:5]}{'…' if len(batch) > 5 else ''}")
            # Still record so we can flag them in the report
            batch_map[str(zip_path)] = batch   # zip_path may not exist

    return batch_map


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 4 — Unzip, rename & organise FASTA files
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_organise(
    batch_map: dict[str, list[str]],
    out_dir: Path,
    show_progress: bool,
) -> list[dict]:
    """
    For each downloaded zip, extract genome FASTA files and rename them to
    <accession>.fna, placing them flat in out_dir.

    NCBI Datasets zip internal structure (v2 schema):
        ncbi_dataset/data/<accession>/<accession>_<asm-name>_genomic.fna

    Returns a list of per-accession status dicts for the report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict] = []

    for zip_path_str, batch_accessions in tqdm(
        batch_map.items(),
        desc="Extracting",
        unit="batch",
        disable=not show_progress,
    ):
        zip_path = Path(zip_path_str)

        if not zip_path.exists():
            # Whole batch failed to download
            for acc in batch_accessions:
                report_rows.append({"Assembly_Accession": acc,
                                    "Status": "DOWNLOAD_FAILED",
                                    "FASTA_path": ""})
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()

                # Build accession → fna-member mapping
                acc_to_member: dict[str, str] = {}
                for m in members:
                    # Match paths like: ncbi_dataset/data/GCF_xxx/GCF_xxx_genomic.fna
                    if m.endswith("_genomic.fna") or m.endswith(".fna"):
                        parts = Path(m).parts
                        # parts[2] is typically the accession directory
                        if len(parts) >= 3:
                            acc_dir = parts[2]                 # e.g. GCF_000001405.40
                            # Normalise to match accession list
                            acc_key = acc_dir.split("_")[0] + "_" + acc_dir.split("_")[1] \
                                      if "_" in acc_dir else acc_dir
                            acc_to_member.setdefault(acc_dir, m)

                for acc in batch_accessions:
                    # Flexible match: GCA_ vs GCF_ version suffix tolerance
                    member = None
                    for dir_key, mbr in acc_to_member.items():
                        if acc in dir_key or dir_key in acc:
                            member = mbr
                            break

                    if member is None:
                        log.warning(f"No FASTA found in zip for accession: {acc}")
                        report_rows.append({"Assembly_Accession": acc,
                                            "Status": "NOT_IN_ZIP",
                                            "FASTA_path": ""})
                        continue

                    dest = out_dir / f"{acc}.fna"
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    report_rows.append({"Assembly_Accession": acc,
                                        "Status": "OK",
                                        "FASTA_path": str(dest)})
                    log.debug(f"Extracted: {acc} → {dest}")

        except zipfile.BadZipFile:
            log.error(f"Corrupted zip file: {zip_path}")
            for acc in batch_accessions:
                report_rows.append({"Assembly_Accession": acc,
                                    "Status": "BAD_ZIP",
                                    "FASTA_path": ""})

    return report_rows


# ─────────────────────────────────────────────────────────────────────────────
## PHASE 5 — Quality check & report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(report_rows: list[dict], out_dir: Path) -> pd.DataFrame:
    """
    Write per-accession download status TSV and print a summary.
    """
    report = pd.DataFrame(report_rows)
    report_path = out_dir.parent / "download_report.tsv"
    report.to_csv(report_path, sep="\t", index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    status_counts = report["Status"].value_counts()
    n_total   = len(report)
    n_ok      = status_counts.get("OK", 0)
    n_failed  = n_total - n_ok
    pct_ok    = 100 * n_ok / n_total if n_total else 0

    print("\n" + "═" * 60)
    print("  STAGE 02 — DOWNLOAD REPORT")
    print("═" * 60)
    print(f"  Total accessions  : {n_total}")
    print(f"  Successfully downloaded: {n_ok}  ({pct_ok:.1f}%)")
    if n_failed:
        print(f"  ⚠  Failed          : {n_failed}")
        for status, count in status_counts.items():
            if status != "OK":
                print(f"       {status:<22}: {count}")
    else:
        print("  ✅ All accessions downloaded successfully.")
    print(f"\n  FASTA files  → {out_dir}/")
    print(f"  Report       → {report_path}")
    print("═" * 60)

    if n_failed:
        failed_accs = report[report["Status"] != "OK"]["Assembly_Accession"].tolist()
        failed_path = out_dir.parent / "download_failed.txt"
        Path(failed_path).write_text("\n".join(failed_accs))
        print(f"\n  Failed accessions saved to: {failed_path}")
        print("  Re-run with this file as --input to retry only failures.\n")

    return report


# ─────────────────────────────────────────────────────────────────────────────
## MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    show_progress = not args.no_progressbar

    # ── Phase 1: Validate CLI ─────────────────────────────────────────────
    check_datasets_cli()

    # ── Resolve API key (CLI > env > anonymous) ───────────────────────────
    api_key = resolve_api_key(args.api_key)

    # ── Phase 2: Load accessions ──────────────────────────────────────────
    accessions = load_accessions(args.input)

    # ── Phase 3: Download ─────────────────────────────────────────────────
    batch_map = download_batches(
        accessions   = accessions,
        tmp_dir      = args.tmp,
        include      = args.include,
        batch_size   = args.batch_size,
        reference_only = args.reference_only,
        show_progress  = show_progress,
        api_key        = api_key,
    )

    # ── Phase 4: Extract & organise ───────────────────────────────────────
    report_rows = extract_and_organise(
        batch_map     = batch_map,
        out_dir       = args.outdir,
        show_progress = show_progress,
    )

    # ── Phase 5: Report ───────────────────────────────────────────────────
    write_report(report_rows, args.outdir)

    # ── Cleanup tmp zips ──────────────────────────────────────────────────
    if args.tmp.exists():
        shutil.rmtree(args.tmp)
        log.info(f"Temporary download directory removed: {args.tmp}")


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
#  Basic — no credentials needed (3 req/s, fine for ~543 accessions):
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input ab_tobramycin_accessions.txt
#
#  With NCBI API key (10 req/s, recommended for large datasets):
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input ab_tobramycin_accessions.txt \
#        --api-key YOUR_KEY_HERE
#
#  Or export key to env (preferred — keeps key out of shell history):
#    export NCBI_API_KEY="YOUR_KEY_HERE"
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input ab_tobramycin_accessions.txt
#
#  Custom output directory:
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input ab_tobramycin_accessions.txt \
#        --outdir data/genomes/tobramycin
#
#  Include protein sequences alongside genome FASTA:
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input ab_tobramycin_accessions.txt \
#        --include genome protein
#
#  Smaller batches (useful if NCBI throttles large requests):
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input ab_tobramycin_accessions.txt \
#        --batch-size 50
#
#  Retry only previously failed accessions:
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input download_failed.txt
#
#  Verbose logging:
#    python amr_geno2dock_pipeline_stage_02.py \
#        --input ab_tobramycin_accessions.txt -v
#
# =============================================================================
# NEXT STEP (Stage 03):
#   Run AMRFinderPlus on each downloaded FASTA:
#
#     for fna in genomes/*.fna; do
#         acc=$(basename "$fna" .fna)
#         amrfinder --nucleotide "$fna" \
#                   --output amrfinder_out/${acc}.tsv \
#                   --threads 4
#     done
# =============================================================================