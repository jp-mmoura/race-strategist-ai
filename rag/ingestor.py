"""
RAG Ingestor — populates ChromaDB with F1 reference data.

Usage:
    python -m rag.ingestor              # ingest everything
    python -m rag.ingestor --circuits   # ingest only circuits.csv

The script reads raw data from  data/raw/  and stores embeddings in
the ChromaDB persist directory configured via .env (CHROMA_PERSIST_DIRECTORY).
"""

from __future__ import annotations

import argparse
import logging
import os
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
# Unified entry point
# ===================================================================

def ingest_all() -> dict[str, int]:
    """Run every available ingestor and return counts per source."""
    client = _get_chroma_client()
    counts: dict[str, int] = {}

    counts["circuits"] = ingest_circuits(client)
    # Future: counts["regulations"] = ingest_regulations(client)

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
    args = parser.parse_args()

    if args.circuits:
        n = ingest_circuits()
        print(f"✅ Ingested {n} circuits")
    else:
        counts = ingest_all()
        for source, n in counts.items():
            print(f"✅ {source}: {n} documents")

    # ── Quick verification ────────────────────────────────────────
    client = _get_chroma_client()
    collection = client.get_collection("f1_circuits")
    print(f"\n📊 Collection 'f1_circuits' now has {collection.count()} documents")

    print("\n🔍 Test query: 'street circuit in Monaco'")
    results = collection.query(
        query_texts=["street circuit in Monaco"],
        n_results=3,
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [{dist:.4f}] {meta['name']} ({meta['country']})")


if __name__ == "__main__":
    main()
