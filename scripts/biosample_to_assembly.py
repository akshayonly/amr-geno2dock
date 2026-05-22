#!/usr/bin/env python3
"""
biosample_to_assembly.py
------------------------
Map BioSample accessions to NCBI Assembly accessions
using batched Entrez API calls.

Credentials are read from environment variables:
    export NCBI_EMAIL="your_email@example.com"
    export NCBI_API_KEY="your_api_key"          # optional but recommended

Usage:
    # From a text file (one BioSample ID per line)
    python biosample_to_assembly.py --input biosample_ids.txt --output mapping.tsv

    # Single or multiple IDs directly on the command line
    python biosample_to_assembly.py --ids SAMN12345678 SAMN09436495

    # Control batch size and output format
    python biosample_to_assembly.py --input biosample_ids.txt --output mapping.tsv \\
        --batch-size 100 --include-missing

    # Verbose logging
    python biosample_to_assembly.py --input biosample_ids.txt -v

Example output (TSV):
    SAMN12345678    GCF_000001405.40
    SAMN09436495    GCF_000750555.1
    SAMN00000001    NA                  # only with --include-missing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from Bio import Entrez

# ──────────────────────────── CONFIG ─────────────────────────────────────────

Entrez.email   = os.environ.get("NCBI_EMAIL")
Entrez.api_key = os.environ.get("NCBI_API_KEY")  # optional but raises limit to 10 req/s

BATCH_SIZE = 200    # safe ceiling for esearch OR queries
RATE_LIMIT = 0.11   # 10 req/s with API key; use 0.34 without

# ──────────────────────────── CLI ────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="biosample_to_assembly.py",
        description="Map BioSample accessions → NCBI Assembly accessions (batched Entrez).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-i", "--input",
        type=Path,
        metavar="FILE",
        help="Text file with one BioSample accession per line.",
    )
    source.add_argument(
        "--ids",
        nargs="+",
        metavar="BIOSAMPLE",
        help="One or more BioSample accessions passed directly.",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output TSV file (default: print to stdout).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"BioSamples per Entrez request (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include BioSamples with no assembly found (written as NA).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()

# ──────────────────────────── LOGGING ────────────────────────────────────────

def setup_logging(verbose: bool) -> logging.Logger:
    log = logging.getLogger("biosample_to_assembly")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(handler)
    return log

# ──────────────────────────── CORE ───────────────────────────────────────────

def biosample_to_assembly_batch(
    biosamples: list[str],
    log: logging.Logger,
) -> dict[str, Optional[str]]:
    """
    Map a batch of BioSample accessions → Assembly accessions in two Entrez calls.
    Returns dict: {biosample_id: assembly_accession_or_None}
    """
    results = {bs: None for bs in biosamples}

    # ── Step 1: batch esearch ─────────────────────────────────────────────
    query = " OR ".join(f"{bs}[BioSample]" for bs in biosamples)

    try:
        handle = Entrez.esearch(db="assembly", term=query, retmax=len(biosamples) * 2)
        record = Entrez.read(handle)
        handle.close()
    except Exception as e:
        log.error(f"esearch failed: {e}")
        return results

    assembly_uids = record.get("IdList", [])
    log.debug(f"  esearch returned {len(assembly_uids)} assembly UIDs")

    if not assembly_uids:
        return results

    time.sleep(RATE_LIMIT)

    # ── Step 2: single esummary for all UIDs ──────────────────────────────
    try:
        handle = Entrez.esummary(db="assembly", id=",".join(assembly_uids), report="full")
        summary = Entrez.read(handle, validate=False)
        handle.close()
    except Exception as e:
        log.error(f"esummary failed: {e}")
        return results

    # ── Step 3: match summaries back to input BioSamples ──────────────────
    docs = summary.get("DocumentSummarySet", {}).get("DocumentSummary", [])
    for doc in docs:
        bs_id = doc.get("BioSampleAccn", "")
        if bs_id in results:
            acc = doc.get("AssemblyAccession", "") or None
            results[bs_id] = acc

    return results


def map_biosamples(
    biosamples: list[str],
    batch_size: int,
    log: logging.Logger,
) -> dict[str, Optional[str]]:
    """
    Process an arbitrary-length list in safe-sized batches with progress logging.
    """
    all_results: dict[str, Optional[str]] = {}

    for i in range(0, len(biosamples), batch_size):
        chunk = biosamples[i : i + batch_size]
        log.info(f"Batch {i // batch_size + 1} — {len(chunk)} samples")
        all_results.update(biosample_to_assembly_batch(chunk, log))
        time.sleep(RATE_LIMIT)

    found   = sum(v is not None for v in all_results.values())
    missing = len(all_results) - found
    log.info(f"Done — {found} mapped, {missing} not found")
    return all_results

# ──────────────────────────── I/O ────────────────────────────────────────────

def load_biosamples(args: argparse.Namespace) -> list[str]:
    if args.ids:
        return args.ids
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    ids = [l.strip() for l in args.input.read_text().splitlines() if l.strip()]
    if not ids:
        raise ValueError("Input file is empty.")
    return ids


def write_output(
    mapping: dict[str, Optional[str]],
    output: Optional[Path],
    include_missing: bool,
) -> None:
    lines = []
    for bs, acc in mapping.items():
        if acc is None and not include_missing:
            continue
        lines.append(f"{bs}\t{acc if acc else 'NA'}")

    text = "\n".join(lines) + "\n"

    if output:
        output.write_text(text)
    else:
        sys.stdout.write(text)

# ──────────────────────────── MAIN ───────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log  = setup_logging(args.verbose)

    # ── Credential checks ─────────────────────────────────────────────────
    if not Entrez.email:
        log.error(
            "NCBI_EMAIL environment variable not set.\n"
            "  export NCBI_EMAIL='your_email@example.com'"
        )
        sys.exit(1)

    if not Entrez.api_key:
        log.warning(
            "NCBI_API_KEY not set — rate limited to 3 req/s. "
            "Set export NCBI_API_KEY='...' for 10 req/s."
        )
        # slow down to respect unauthenticated limit
        global RATE_LIMIT
        RATE_LIMIT = 0.34

    # ── Run ───────────────────────────────────────────────────────────────
    biosamples = load_biosamples(args)
    log.info(f"Loaded {len(biosamples)} BioSample IDs")

    mapping = map_biosamples(biosamples, args.batch_size, log)
    write_output(mapping, args.output, args.include_missing)


if __name__ == "__main__":
    main()