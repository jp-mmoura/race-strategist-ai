"""
RAG Ingestor — populates ChromaDB with F1 reference data.

Usage:
    python -m rag.ingestor                  # ingest everything
    python -m rag.ingestor --circuits       # ingest only circuits.csv
    python -m rag.ingestor --regulations    # ingest only PDF regulations

PDF discovery order
-------------------
1. ``data/raw/regulations/{year}/`` — organised per-year sub-directories
   (created by ``scripts/download_fia_regulations.py --organise``).
2. ``data/raw/`` — legacy flat layout (existing 2026 PDFs, year inferred
   from filename).

The script reads raw data from  data/raw/  and stores embeddings in
the ChromaDB persist directory configured via .env (CHROMA_PERSIST_DIRECTORY).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_DEFAULT_CHROMA = str(_PROJECT_ROOT / "data" / "chroma_db")
_env_chroma = os.getenv("CHROMA_PERSIST_DIRECTORY", "")
if _env_chroma:
    _p = Path(_env_chroma)
    CHROMA_DIR = str(_p if _p.is_absolute() else _PROJECT_ROOT / _p)
else:
    CHROMA_DIR = _DEFAULT_CHROMA


# ---------------------------------------------------------------------------
# ChromaDB client (singleton)
# ---------------------------------------------------------------------------
def _get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    logger.info("ChromaDB client initialised — persist dir: %s", CHROMA_DIR)
    return client


# ===================================================================
# Circuits ingestion
# ===================================================================

def _row_to_document(row: pd.Series) -> str:
    """Convert a single circuits.csv row into a natural-language document.

    This rich text representation gives the embedding model enough
    semantic signal to retrieve relevant circuits by name, country,
    location, or geographic characteristics.
    """
    parts = [
        f"Circuit: {row['name']}.",
        f"Location: {row['location']}, {row['country']}.",
        f"Reference key: {row['circuitRef']}.",
        f"Coordinates: latitude {row['lat']}, longitude {row['lng']}.",
        f"Altitude: {row['alt']} metres above sea level.",
    ]
    if pd.notna(row.get("url")):
        parts.append(f"More info: {row['url']}")
    return " ".join(parts)


def ingest_circuits(client: chromadb.ClientAPI | None = None) -> int:
    """Read data/raw/circuits.csv and upsert into ChromaDB.

    Parameters
    ----------
    client : chromadb.ClientAPI | None
        An existing client; one is created if *None*.

    Returns
    -------
    int
        Number of documents upserted.
    """
    csv_path = _RAW_DIR / "circuits.csv"
    if not csv_path.exists():
        logger.error("circuits.csv not found at %s", csv_path)
        return 0

    client = client or _get_chroma_client()
    collection = client.get_or_create_collection(
        name="f1_circuits",
        metadata={"description": "F1 circuit reference data from circuits.csv"},
    )

    df = pd.read_csv(csv_path)
    logger.info("Read %d circuits from %s", len(df), csv_path)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for _, row in df.iterrows():
        doc_id = f"circuit_{row['circuitId']}"
        doc_text = _row_to_document(row)
        meta = {
            "circuitId": int(row["circuitId"]),
            "circuitRef": str(row["circuitRef"]),
            "name": str(row["name"]),
            "location": str(row["location"]),
            "country": str(row["country"]),
            "lat": float(row["lat"]),
            "lng": float(row["lng"]),
            "alt": int(row["alt"]),
            "source": "circuits.csv",
        }

        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(meta)

    # Upsert in a single batch (< 5 000 docs → safe)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(
        "Upserted %d documents into collection '%s'",
        len(ids),
        collection.name,
    )
    return len(ids)


# ===================================================================
# Regulations ingestion
# ===================================================================

# Chunk settings
_CHUNK_SIZE = 1500       # characters per chunk
_CHUNK_OVERLAP = 300     # characters overlapping between consecutive chunks

# Maps section letter → descriptive name stored in metadata
_SECTION_NAMES: dict[str, str] = {
    "A": "General Provisions",
    "B": "Sporting Regulations",
    "C": "Technical Regulations",
    "D": "Financial Regulations (Teams)",
    "E": "Financial Regulations (PU Manufacturers)",
    "F": "Operational Regulations",
}

# Patterns to detect year and section letter from a PDF filename
# Note: do NOT use \b — underscores are \w so "fia_2024_f1" has no word
# boundary before/after 2024.  We simply match the first 4-digit year ≥ 2000.
_YEAR_RE = re.compile(r"(20\d{2})")
# Note: the section letter is followed by _ (word char) not a word boundary.
# Pattern matches: section_a_, section-b_, section_c_ etc.
_SECTION_LETTER_RE = re.compile(r"section[_\s-]([a-fA-F])[_\s\-\.]", re.IGNORECASE)
_SECTION_WORD_RE = re.compile(
    r"\b(sporting|technical|financial|operational|general)\b",
    re.IGNORECASE,
)
_WORD_TO_LETTER: dict[str, str] = {
    "general":    "A",
    "sporting":   "B",
    "technical":  "C",
    "financial":  "D",
    "operational": "F",
}


def _parse_pdf_metadata(pdf_path: Path) -> tuple[int, str]:
    """Return ``(year, section_letter)`` derived from *pdf_path*.

    Falls back to ``(2026, "?")`` when detection fails so ingestion
    can still proceed.
    """
    name = pdf_path.stem.lower()

    # Year — prefer the sub-directory name, then filename
    try:
        year = int(pdf_path.parent.name)
    except ValueError:
        m = _YEAR_RE.search(name)
        year = int(m.group(1)) if m else 2026

    # Section letter — explicit "section_X" pattern wins
    sec_m = _SECTION_LETTER_RE.search(name)
    if sec_m:
        letter = sec_m.group(1).upper()
    else:
        word_m = _SECTION_WORD_RE.search(name)
        if word_m:
            word = word_m.group(1).lower()
            letter = _WORD_TO_LETTER.get(word, "?")
            if letter == "D" and ("pu" in name or "manufacturer" in name):
                letter = "E"
        else:
            letter = "?"

    return year, letter


def _extract_pdf_text(pdf_path: Path) -> str:
    """Return the full text content of a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF ingestion. "
            "Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping fixed-size character chunks.

    Chunks respect word boundaries when possible — they never cut in the
    middle of a word within the last ``overlap`` characters.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # If not at the end, try to break on whitespace
        if end < length:
            # Search backwards for the nearest space within the last 'overlap' chars
            boundary = text.rfind(" ", start + overlap, end)
            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance by (chunk_size - overlap) so successive chunks share 'overlap' chars
        start += max(1, chunk_size - overlap)

    return chunks


def _discover_regulation_pdfs() -> list[Path]:
    """Collect all regulation PDFs in discovery order.

    1. ``data/raw/regulations/{year}/`` sub-directories (organised layout).
    2. ``data/raw/`` root (legacy flat layout — existing 2026 PDFs).

    Returns a deduplicated list sorted by year (ascending) then filename.
    """
    found: dict[str, Path] = {}  # stem.lower() → Path (dedup)

    # 1. Per-year sub-directories
    reg_base = _RAW_DIR / "regulations"
    if reg_base.exists():
        for year_dir in sorted(reg_base.iterdir()):
            if year_dir.is_dir():
                for pdf in sorted(year_dir.glob("*.pdf")):
                    found[pdf.stem.lower()] = pdf

    # 2. Flat legacy layout
    for pdf in sorted(_RAW_DIR.glob("*.pdf")):
        key = pdf.stem.lower()
        if key not in found:
            found[key] = pdf

    return list(found.values())


def ingest_regulations(
    client: chromadb.ClientAPI | None = None,
    years: list[int] | None = None,
) -> int:
    """Ingest FIA regulation PDFs into the ``f1_regulations`` ChromaDB collection.

    Each PDF is chunked into ~1 500-character segments with 300-character
    overlap.  Every chunk carries rich metadata so retrieval can be filtered
    by ``year`` and ``section``.

    Parameters
    ----------
    client : chromadb.ClientAPI | None
        Existing client; one is created if *None*.
    years : list[int] | None
        If provided, only PDFs whose detected year is in this list are
        ingested.  Pass ``None`` (default) to ingest all years found.

    Returns
    -------
    int
        Total number of chunks upserted.
    """
    pdfs = _discover_regulation_pdfs()
    if not pdfs:
        logger.warning(
            "No regulation PDFs found under %s. "
            "Place PDF files in data/raw/ or run: "
            "python scripts/download_fia_regulations.py --organise",
            _RAW_DIR,
        )
        return 0

    client = client or _get_chroma_client()
    collection = client.get_or_create_collection(
        name="f1_regulations",
        metadata={"description": "FIA F1 regulation documents (sporting, technical, financial, operational)"},
    )

    total_chunks = 0

    for pdf_path in pdfs:
        year, letter = _parse_pdf_metadata(pdf_path)

        if years is not None and year not in years:
            logger.debug("Skipping %s (year=%d, filter=%s)", pdf_path.name, year, years)
            continue

        section_name = _SECTION_NAMES.get(letter, f"Section {letter}")
        logger.info(
            "Processing PDF: %s (year=%d, section=%s — %s)",
            pdf_path.name, year, letter, section_name,
        )

        try:
            full_text = _extract_pdf_text(pdf_path)
        except Exception as exc:
            logger.error("Failed to extract text from %s: %s", pdf_path.name, exc)
            continue

        if not full_text.strip():
            logger.warning("No text extracted from %s — skipping", pdf_path.name)
            continue

        chunks = _chunk_text(full_text)
        total = len(chunks)
        logger.info("  → %d chunk(s) from %s", total, pdf_path.name)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        # Stable ID: year_sectionLetter_chunkIndex (survives re-ingestion)
        id_prefix = f"reg_{year}_sec{letter.lower()}"

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{id_prefix}_{idx:05d}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source":        pdf_path.name,
                "year":          year,
                "section":       letter,
                "section_name":  section_name,
                "doc_type":      "regulation",
                "chunk_index":   idx,
                "total_chunks":  total,
            })

        # Upsert in batches of 500 to stay within ChromaDB limits
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        total_chunks += total
        logger.info(
            "  ✓ Upserted %d chunks (year=%d, section=%s)",
            total, year, letter,
        )

    logger.info(
        "Regulation ingestion complete: %d total chunks across %d PDF(s)",
        total_chunks, len(pdfs),
    )
    return total_chunks


# ===================================================================
# Unified entry point
# ===================================================================

def ingest_all() -> dict[str, int]:
    """Run every available ingestor and return counts per source."""
    client = _get_chroma_client()
    counts: dict[str, int] = {}

    counts["circuits"] = ingest_circuits(client)
    counts["regulations"] = ingest_regulations(client)

    logger.info("Ingestion complete: %s", counts)
    return counts


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="F1 RAG Ingestor")
    parser.add_argument(
        "--circuits", action="store_true",
        help="Ingest only circuits.csv",
    )
    parser.add_argument(
        "--regulations", action="store_true",
        help="Ingest only FIA regulation PDFs",
    )
    parser.add_argument(
        "--years", type=int, nargs="+",
        help="Limit PDF ingestion to these years (e.g. --years 2024 2025 2026)",
    )
    args = parser.parse_args()

    if args.circuits:
        n = ingest_circuits()
        print(f"[OK] circuits: {n} documents")
    elif args.regulations:
        n = ingest_regulations(years=args.years)
        print(f"[OK] regulations: {n} chunks")
    else:
        counts = ingest_all()
        for source, n in counts.items():
            print(f"[OK] {source}: {n} documents/chunks")

    # ── Quick verification ────────────────────────────────────────
    client = _get_chroma_client()

    try:
        circuits_col = client.get_collection("f1_circuits")
        print(f"\nCollection 'f1_circuits': {circuits_col.count()} documents")
        print("Test query: 'street circuit in Monaco'")
        results = circuits_col.query(
            query_texts=["street circuit in Monaco"],
            n_results=3,
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            print(f"  [{dist:.4f}] {meta['name']} ({meta['country']})")
    except Exception as exc:
        print(f"WARNING: f1_circuits not available: {exc}")

    try:
        reg_col = client.get_collection("f1_regulations")
        print(f"\nCollection 'f1_regulations': {reg_col.count()} chunks")
        if reg_col.count() > 0:
            print("Test query: 'safety car procedure'")
            reg_results = reg_col.query(
                query_texts=["safety car procedure"],
                n_results=3,
            )
            for rdoc, rmeta, rdist in zip(
                reg_results["documents"][0],
                reg_results["metadatas"][0],
                reg_results["distances"][0],
            ):
                preview = rdoc[:100].replace("\n", " ")
                print(
                    f"  [{rdist:.4f}] {rmeta['year']} Section {rmeta['section']}"
                    f" ({rmeta['section_name']}) chunk {rmeta['chunk_index']}: "
                    f"\"{preview}...\""
                )
    except Exception:
        print("\nCollection 'f1_regulations' not yet populated (run with --regulations or without flags)")


if __name__ == "__main__":
    main()
