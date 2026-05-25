"""
RAG Embedder — manages the ChromaDB embedding function and collection access.

Centralises ChromaDB client creation and collection handles so that
both the ingestor and the retriever share the same configuration.
ChromaDB's default embedding function (all-MiniLM-L6-v2 via
Sentence Transformers) is used — no OpenAI key required for embeddings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import chromadb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHROMA = str(_PROJECT_ROOT / "data" / "chroma_db")
_env_chroma = os.getenv("CHROMA_PERSIST_DIRECTORY", "")
# Resolve relative paths against the project root so it works from any CWD
if _env_chroma:
    _p = Path(_env_chroma)
    CHROMA_DIR = str(_p if _p.is_absolute() else _PROJECT_ROOT / _p)
else:
    CHROMA_DIR = _DEFAULT_CHROMA

# ---------------------------------------------------------------------------
# Known collections
# ---------------------------------------------------------------------------
COLLECTIONS: dict[str, str] = {
    "f1_circuits": "F1 circuit reference data",
    "f1_regulations": "FIA regulation documents (sporting, technical, financial, operational)",
    # Future:
    # "f1_race_history": "Historical race strategy summaries",
}

# Type alias for collection names
CollectionName = Literal["f1_circuits", "f1_regulations"]


# ===================================================================
# ChromaDB client (module-level singleton)
# ===================================================================

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """Return (or create) the persistent ChromaDB client."""
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        logger.info("ChromaDB client ready — %s", CHROMA_DIR)
    return _client


# ===================================================================
# Collection access
# ===================================================================

def get_collection(
    name: str = "f1_circuits",
) -> chromadb.Collection:
    """Return an existing ChromaDB collection by name.

    Raises
    ------
    ValueError
        If the collection does not exist yet (run the ingestor first).
    """
    client = get_client()
    try:
        collection = client.get_collection(name)
    except Exception as exc:
        raise ValueError(
            f"Collection '{name}' not found. "
            f"Run `python -m rag.ingestor` first."
        ) from exc

    logger.info(
        "Opened collection '%s' (%d documents)",
        collection.name,
        collection.count(),
    )
    return collection


def list_collections() -> list[dict[str, str | int]]:
    """Return a summary of every collection in the database."""
    client = get_client()
    result = []
    for col in client.list_collections():
        result.append({
            "name": col.name,
            "count": col.count(),
        })
    return result


# ===================================================================
# Quick sanity-check
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Collections:", list_collections())
    col = get_collection("f1_circuits")
    print(f"f1_circuits: {col.count()} documents")
    try:
        reg_col = get_collection("f1_regulations")
        print(f"f1_regulations: {reg_col.count()} documents")
    except ValueError as e:
        print(f"f1_regulations not yet populated: {e}")
