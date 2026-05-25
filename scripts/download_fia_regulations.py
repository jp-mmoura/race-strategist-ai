"""
FIA Regulations Downloader.

Organises FIA F1 regulation PDFs into the expected folder structure:

    data/raw/regulations/{year}/section_{letter}_{name}.pdf

Usage
-----
    # Scan existing PDFs in data/raw/ and move/copy them into year sub-folders:
    python scripts/download_fia_regulations.py --organise

    # Show what would be done without moving any files:
    python scripts/download_fia_regulations.py --organise --dry-run

    # Print URLs that can be opened in a browser for manual download:
    python scripts/download_fia_regulations.py --list-urls

Notes
-----
The FIA does not offer a stable machine-readable API for regulation
documents — URLs change with each issue (the issue number and date are
embedded in the filename).  This script therefore:

  1. Organises *existing* PDFs already stored in data/raw/ into
     per-year sub-directories so the ingestor can find them by year.
  2. Prints canonical FIA page URLs for manual download of additional
     year editions.

Sections (as they appear in the FIA 2026 documents)
----------------------------------------------------
  A  — General Provisions
  B  — Sporting Regulations
  C  — Technical Regulations
  D  — Financial Regulations (F1 Teams)
  E  — Financial Regulations (PU Manufacturers)
  F  — Operational Regulations
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_REG_BASE_DIR = _RAW_DIR / "regulations"

# ---------------------------------------------------------------------------
# Section metadata
# ---------------------------------------------------------------------------
# Maps the canonical letter  →  (short slug used in folder name)
SECTION_MAP: dict[str, str] = {
    "A": "general_provisions",
    "B": "sporting",
    "C": "technical",
    "D": "financial_teams",
    "E": "financial_pu_manufacturers",
    "F": "operational",
}

# FIA website pages where regulation PDFs can be found (manual download)
FIA_URLS: dict[int, str] = {
    2026: "https://www.fia.com/regulation/category/110",
    2025: "https://www.fia.com/regulation/category/110",
    2024: "https://www.fia.com/regulation/category/110",
    2023: "https://www.fia.com/regulation/category/110",
    2022: "https://www.fia.com/regulation/category/110",
    2021: "https://www.fia.com/regulation/category/110",
    2020: "https://www.fia.com/regulation/category/110",
}

# ---------------------------------------------------------------------------
# Filename → (year, section_letter) parser
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(20\d{2})")
_SECTION_RE = re.compile(
    r"section[_\s-]([a-fA-F])[_\s\-\.]",
    re.IGNORECASE,
)
_SECTION_WORD_RE = re.compile(
    r"\b(sporting|technical|financial|operational|general)[_\s]",
    re.IGNORECASE,
)

_WORD_TO_LETTER = {
    "general":    "A",
    "sporting":   "B",
    "technical":  "C",
    "financial":  "D",  # D by default; E for PU manufacturers
    "operational": "F",
}


def _parse_pdf_name(pdf_path: Path) -> tuple[int | None, str | None]:
    """Extract (year, section_letter) from a PDF filename.

    Returns (None, None) when either piece cannot be determined.
    """
    name = pdf_path.stem.lower()

    # Year
    year_m = _YEAR_RE.search(name)
    year = int(year_m.group(1)) if year_m else None

    # Section letter — prefer explicit "section_X" pattern
    sec_m = _SECTION_RE.search(name)
    if sec_m:
        letter = sec_m.group(1).upper()
    else:
        word_m = _SECTION_WORD_RE.search(name)
        if word_m:
            word = word_m.group(1).lower()
            letter = _WORD_TO_LETTER.get(word)
            # Distinguish section D vs E for financial documents
            if letter == "D" and "pu" in name or "manufacturer" in name:
                letter = "E"
        else:
            letter = None

    return year, letter


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def discover_pdfs(search_dir: Path) -> list[Path]:
    """Return all PDFs directly under *search_dir* (non-recursive)."""
    return sorted(search_dir.glob("*.pdf"))


def organise_pdfs(dry_run: bool = False) -> dict[int, list[Path]]:
    """Move existing PDFs from data/raw/ into data/raw/regulations/{year}/.

    Returns a mapping of year → list of destination paths.
    """
    pdfs = discover_pdfs(_RAW_DIR)
    if not pdfs:
        logger.warning("No PDFs found in %s", _RAW_DIR)
        return {}

    organised: dict[int, list[Path]] = {}

    for pdf in pdfs:
        year, letter = _parse_pdf_name(pdf)

        if year is None:
            logger.warning("Could not detect year in '%s' — skipped", pdf.name)
            continue

        dest_dir = _REG_BASE_DIR / str(year)

        if letter:
            slug = SECTION_MAP.get(letter, f"section_{letter.lower()}")
            dest_name = f"section_{letter.lower()}_{slug}.pdf"
        else:
            logger.warning(
                "Could not detect section letter in '%s' — using original name",
                pdf.name,
            )
            dest_name = pdf.name

        dest_path = dest_dir / dest_name

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if dest_path.exists():
                logger.info("Already exists, skipping: %s", dest_path)
            else:
                shutil.copy2(pdf, dest_path)
                logger.info("Copied %s → %s", pdf.name, dest_path)
        else:
            logger.info("[DRY-RUN] Would copy %s → %s", pdf.name, dest_path)

        organised.setdefault(year, []).append(dest_path)

    return organised


def list_manual_download_urls() -> None:
    """Print FIA website URLs for manual PDF downloads."""
    print("\nFIA F1 Regulations — Manual Download URLs")
    print("=" * 60)
    print(
        "Visit each URL below, navigate to 'Formula 1 Regulations',\n"
        "and download each section PDF into:\n"
        "  data/raw/regulations/{year}/\n"
    )
    for year, url in sorted(FIA_URLS.items(), reverse=True):
        existing = list((_REG_BASE_DIR / str(year)).glob("*.pdf")) if (
            _REG_BASE_DIR / str(year)
        ).exists() else []
        status = f"({len(existing)} PDF(s) already present)" if existing else "(not yet downloaded)"
        print(f"  {year}: {url}  {status}")

    print("\nSection reference:")
    for letter, slug in SECTION_MAP.items():
        print(f"  Section {letter} — {slug.replace('_', ' ').title()}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organise FIA regulation PDFs into year sub-folders.",
    )
    parser.add_argument(
        "--organise",
        action="store_true",
        help="Copy PDFs from data/raw/ into data/raw/regulations/{year}/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making any changes (use with --organise)",
    )
    parser.add_argument(
        "--list-urls",
        action="store_true",
        help="Print FIA website URLs for manual download",
    )
    args = parser.parse_args()

    if args.list_urls:
        list_manual_download_urls()

    if args.organise:
        result = organise_pdfs(dry_run=args.dry_run)
        total = sum(len(v) for v in result.values())
        tag = "[DRY-RUN] " if args.dry_run else ""
        print(f"\n{tag}Organised {total} PDFs across {len(result)} year(s):")
        for year, paths in sorted(result.items()):
            print(f"  {year}: {len(paths)} file(s)")
            for p in paths:
                print(f"    — {p.name}")

    if not args.organise and not args.list_urls:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
