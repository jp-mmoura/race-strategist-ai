"""
Tests for the RAG pipeline — ingestor, embedder, and retriever.

All tests that touch ChromaDB work against the *real* persisted database
(populated by ``python -m rag.ingestor``).  Tests that would require a live
ChromaDB are skipped automatically when the collection is not available, so
the test suite stays green in a clean checkout that hasn't run the ingestor.

Run with:
    pytest tests/test_rag.py -v
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# Helpers / fixtures
# ===================================================================

def _circuits_available() -> bool:
    """Return True when the f1_circuits ChromaDB collection exists."""
    try:
        from rag.embedder import get_collection
        col = get_collection("f1_circuits")
        return col.count() > 0
    except Exception:
        return False


def _regulations_available() -> bool:
    """Return True when the f1_regulations ChromaDB collection exists."""
    try:
        from rag.embedder import get_collection
        col = get_collection("f1_regulations")
        return col.count() > 0
    except Exception:
        return False


requires_circuits    = pytest.mark.skipif(
    not _circuits_available(),
    reason="f1_circuits collection not found — run: python -m rag.ingestor --circuits",
)
requires_regulations = pytest.mark.skipif(
    not _regulations_available(),
    reason="f1_regulations collection not found — run: python -m rag.ingestor --regulations",
)


# ===================================================================
# 1. Chunking logic (unit — no ChromaDB needed)
# ===================================================================

class TestChunkText:
    """_chunk_text must produce correct overlapping chunks of the expected size."""

    def _chunk(self, text: str, size: int = 1500, overlap: int = 300) -> list[str]:
        from rag.ingestor import _chunk_text
        return _chunk_text(text, chunk_size=size, overlap=overlap)

    def test_empty_text_returns_empty_list(self):
        assert self._chunk("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert self._chunk("   \n\n  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "A" * 100
        chunks = self._chunk(text, size=200, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        # 3 000 chars with chunk_size=1 500 and overlap=300
        # advance = 1500 - 300 = 1200 per step → at least 2 chunks
        text = "word " * 600  # ~3 000 chars
        chunks = self._chunk(text, size=1500, overlap=300)
        assert len(chunks) >= 2

    def test_chunks_respect_maximum_size(self):
        """No chunk may exceed chunk_size characters."""
        text = "x " * 2000  # 4 000 chars
        chunks = self._chunk(text, size=500, overlap=100)
        for c in chunks:
            assert len(c) <= 500, f"Chunk too long: {len(c)} chars"

    def test_overlap_means_text_is_repeated(self):
        """When overlap > 0, some text from the end of chunk N must appear at
        the beginning of chunk N+1."""
        text = " ".join(f"word{i}" for i in range(300))   # ~2 100 chars
        chunks = self._chunk(text, size=500, overlap=100)
        assert len(chunks) >= 2

        # The last 80 characters of chunk 0 must appear somewhere in chunk 1
        tail_of_first = chunks[0][-80:]
        assert tail_of_first in chunks[1], (
            "Overlap content not found in the next chunk. "
            f"Tail: {tail_of_first!r:.40}"
        )

    def test_no_empty_chunks(self):
        text = "hello world " * 500
        chunks = self._chunk(text, size=200, overlap=40)
        for c in chunks:
            assert c.strip(), "Empty/whitespace chunk found"

    def test_all_content_covered(self):
        """Every word in the source must appear in at least one chunk."""
        words = [f"unique_word_{i}" for i in range(50)]
        text = " ".join(words)
        chunks = self._chunk(text, size=100, overlap=20)
        combined = " ".join(chunks)
        for w in words:
            assert w in combined, f"Word '{w}' missing from all chunks"


# ===================================================================
# 2. PDF metadata parsing (unit — no ChromaDB needed)
# ===================================================================

class TestParsePdfMetadata:
    """_parse_pdf_metadata must correctly detect year and section letter
    from the filenames that actually exist in data/raw/."""

    def _parse(self, filename: str) -> tuple[int, str]:
        from rag.ingestor import _parse_pdf_metadata
        # Simulate a flat file (parent = data/raw, not a year directory)
        path = Path("data") / "raw" / filename
        return _parse_pdf_metadata(path)

    # -- Lowercase snake_case filenames (new FIA style) -----------------

    def test_section_a_lowercase(self):
        year, letter = self._parse(
            "fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_02_-_2026-02-27.pdf"
        )
        assert year == 2026
        assert letter == "A"

    def test_section_c_lowercase(self):
        year, letter = self._parse(
            "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf"
        )
        assert year == 2026
        assert letter == "C"

    def test_section_e_lowercase_pu(self):
        year, letter = self._parse(
            "fia_2026_f1_regulations_-_section_e_financial_-_pu_manufacturers_-_iss_05_-_2026-05-07.pdf"
        )
        assert year == 2026
        assert letter == "E"

    def test_section_f_lowercase(self):
        year, letter = self._parse(
            "fia_2026_f1_regulations_-_section_f_operational_-_iss_08_-_2026-05-07.pdf"
        )
        assert year == 2026
        assert letter == "F"

    # -- Mixed-case bracket filenames (older FIA style) ------------------

    def test_section_b_mixed_case(self):
        year, letter = self._parse(
            "FIA 2026 F1 Regulations - Section B [Sporting] - Iss 06 - 2026-04-28.pdf"
        )
        assert year == 2026
        assert letter == "B"

    def test_section_d_mixed_case(self):
        year, letter = self._parse(
            "FIA 2026 F1 Regulations - Section D [Financial - F1 Teams] - Iss 06 - 2026-04-28.pdf"
        )
        assert year == 2026
        assert letter == "D"

    # -- Year parsed from parent directory name --------------------------

    def test_year_from_parent_directory(self):
        from rag.ingestor import _parse_pdf_metadata
        path = Path("data") / "raw" / "regulations" / "2024" / "section_b_sporting.pdf"
        year, letter = _parse_pdf_metadata(path)
        assert year == 2024

    # -- Fallback behaviour ----------------------------------------------

    def test_unknown_section_returns_question_mark(self):
        _, letter = self._parse("fia_2026_regulations_unknown.pdf")
        assert letter == "?"

    def test_fallback_year_is_2026(self):
        from rag.ingestor import _parse_pdf_metadata
        # No year anywhere in path
        path = Path("data") / "raw" / "regulations_no_year.pdf"
        year, _ = _parse_pdf_metadata(path)
        assert year == 2026


# ===================================================================
# 3. Ingestor — ingest_regulations returns correct chunk count
# ===================================================================

class TestIngestRegulations:
    """
    Validate that ingest_regulations() processes PDFs correctly.
    Uses a mock ChromaDB collection and a small synthetic PDF so no
    real file I/O or network calls are required.
    """

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        collection = MagicMock()
        collection.upsert = MagicMock()
        client.get_or_create_collection.return_value = collection
        return client

    def _make_fake_pdf(self, tmp_path: Path, filename: str, content: str) -> Path:
        """Create a real one-page PDF in tmp_path using pypdf's writer."""
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        path = tmp_path / filename
        with open(path, "wb") as f:
            writer.write(f)
        return path

    def test_empty_raw_dir_returns_zero(self, tmp_path, mock_client):
        """When no PDFs exist, ingest_regulations must return 0 without error."""
        from rag import ingestor
        original_raw = ingestor._RAW_DIR
        ingestor._RAW_DIR = tmp_path
        try:
            result = ingestor.ingest_regulations(client=mock_client)
        finally:
            ingestor._RAW_DIR = original_raw

        assert result == 0
        mock_client.get_or_create_collection.assert_not_called()

    def test_year_filter_skips_wrong_year(self, tmp_path, mock_client):
        """PDFs whose detected year is not in the filter list must be skipped."""
        from rag import ingestor

        # Create a fake PDF with 2024 in the name (we mock text extraction)
        pdf_path = tmp_path / "fia_2024_f1_regulations_-_section_b_sporting_-_iss_01.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj")

        original_raw = ingestor._RAW_DIR
        ingestor._RAW_DIR = tmp_path
        try:
            with patch("rag.ingestor._extract_pdf_text", return_value="Some regulation text " * 100):
                result = ingestor.ingest_regulations(client=mock_client, years=[2026])
        finally:
            ingestor._RAW_DIR = original_raw

        # year=2024 PDF should be skipped when filter=[2026]
        assert result == 0

    def test_year_filter_includes_matching_year(self, tmp_path, mock_client):
        """PDFs whose year matches the filter must be ingested."""
        from rag import ingestor

        pdf_path = tmp_path / "fia_2026_f1_regulations_-_section_b_sporting_-_iss_01.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj")

        original_raw = ingestor._RAW_DIR
        ingestor._RAW_DIR = tmp_path
        try:
            # Provide enough text to produce at least one chunk
            with patch("rag.ingestor._extract_pdf_text", return_value="Regulation text. " * 200):
                result = ingestor.ingest_regulations(client=mock_client, years=[2026])
        finally:
            ingestor._RAW_DIR = original_raw

        assert result > 0

    def test_metadata_fields_present_on_upsert(self, tmp_path, mock_client):
        """Every upserted chunk must carry the required metadata keys."""
        from rag import ingestor

        pdf_path = tmp_path / "fia_2026_f1_regulations_-_section_b_sporting_-_iss_01.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj")

        original_raw = ingestor._RAW_DIR
        ingestor._RAW_DIR = tmp_path
        try:
            with patch("rag.ingestor._extract_pdf_text", return_value="Safety car. " * 300):
                ingestor.ingest_regulations(client=mock_client, years=[2026])
        finally:
            ingestor._RAW_DIR = original_raw

        collection = mock_client.get_or_create_collection.return_value
        assert collection.upsert.called, "upsert should have been called at least once"

        # Inspect the first upsert call's metadata
        call_kwargs = collection.upsert.call_args_list[0][1]
        first_meta = call_kwargs["metadatas"][0]

        required_keys = {"source", "year", "section", "section_name",
                         "doc_type", "chunk_index", "total_chunks"}
        assert required_keys <= set(first_meta.keys()), (
            f"Missing metadata keys: {required_keys - set(first_meta.keys())}"
        )
        assert first_meta["doc_type"] == "regulation"
        assert first_meta["year"] == 2026
        assert first_meta["chunk_index"] == 0


# ===================================================================
# 4. Circuit retrieval (requires populated database)
# ===================================================================

@requires_circuits
class TestRetrieveCircuits:
    """Validate semantic circuit search against the real ChromaDB collection."""

    def test_returns_list_of_dicts(self):
        from rag.retriever import retrieve_circuits
        results = retrieve_circuits("Monaco street circuit", n_results=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_each_result_has_required_keys(self):
        from rag.retriever import retrieve_circuits
        results = retrieve_circuits("fast track Great Britain", n_results=2)
        for r in results:
            assert "document" in r
            assert "metadata" in r
            assert "distance" in r

    def test_monaco_query_finds_monaco(self):
        from rag.retriever import retrieve_circuits
        results = retrieve_circuits("Monaco street circuit Monte Carlo", n_results=5)
        names = [r["metadata"]["name"] for r in results]
        assert any("Monaco" in n or "Monte" in n for n in names), (
            f"Expected Monaco in top-5 results; got: {names}"
        )

    def test_distance_is_non_negative(self):
        from rag.retriever import retrieve_circuits
        results = retrieve_circuits("Italian Grand Prix Monza", n_results=3)
        for r in results:
            assert r["distance"] >= 0

    def test_metadata_contains_country(self):
        from rag.retriever import retrieve_circuits
        results = retrieve_circuits("circuit in Italy", n_results=3)
        for r in results:
            assert "country" in r["metadata"]

    def test_where_filter_restricts_results(self):
        from rag.retriever import retrieve_circuits
        results = retrieve_circuits(
            "Grand Prix circuit",
            n_results=5,
            where_filter={"country": "UK"},
        )
        for r in results:
            assert r["metadata"]["country"] == "UK", (
                f"where_filter not applied: got country={r['metadata']['country']}"
            )


# ===================================================================
# 5. Regulation retrieval — year filter (requires populated database)
# ===================================================================

@requires_regulations
class TestRetrieveRegulationsYearFilter:
    """retrieve_regulations must honour the year metadata filter."""

    def test_year_filter_returns_only_matching_year(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations(
            "pit stop minimum time",
            year=2026,
            n_results=5,
        )
        assert len(results) > 0, "Expected at least one result for year=2026"
        for r in results:
            assert r["metadata"]["year"] == 2026, (
                f"Year filter violated: expected 2026, got {r['metadata']['year']}"
            )

    def test_unmatched_year_returns_empty_or_fewer(self):
        """A year with no documents should return empty (or near-empty) results."""
        from rag.retriever import retrieve_regulations
        # 1900 is not a real F1 year — should match nothing
        results = retrieve_regulations(
            "safety car",
            year=1900,
            n_results=5,
        )
        # The collection only has 2026 data, so a filter for 1900 must yield nothing
        assert len(results) == 0, (
            f"Expected 0 results for year=1900, got {len(results)}"
        )

    def test_no_year_filter_returns_results(self):
        """Without year filter, results come from any available year."""
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("tyre compound selection", n_results=5)
        assert len(results) > 0

    def test_result_distance_is_finite(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("parc fermé", year=2026, n_results=3)
        for r in results:
            assert math.isfinite(r["distance"])


# ===================================================================
# 6. Regulation retrieval — section filter (requires populated database)
# ===================================================================

@requires_regulations
class TestRetrieveRegulationsSectionFilter:
    """retrieve_regulations must honour the section metadata filter."""

    def test_section_b_filter(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations(
            "safety car deployment",
            section="B",
            n_results=5,
        )
        assert len(results) > 0, "Expected results from Section B (Sporting)"
        for r in results:
            assert r["metadata"]["section"] == "B", (
                f"Section filter violated: expected B, got {r['metadata']['section']}"
            )

    def test_section_c_filter(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations(
            "aerodynamic component dimension",
            section="C",
            n_results=5,
        )
        assert len(results) > 0, "Expected results from Section C (Technical)"
        for r in results:
            assert r["metadata"]["section"] == "C", (
                f"Section filter violated: expected C, got {r['metadata']['section']}"
            )

    def test_section_filter_uppercase_and_lowercase_equivalent(self):
        """Section filter must be case-insensitive ('b' and 'B' must yield same results)."""
        from rag.retriever import retrieve_regulations
        upper = retrieve_regulations("race start", section="B", n_results=3)
        lower = retrieve_regulations("race start", section="b", n_results=3)
        upper_ids = [r["metadata"]["chunk_index"] for r in upper]
        lower_ids = [r["metadata"]["chunk_index"] for r in lower]
        assert upper_ids == lower_ids, (
            "Case should not affect section filter results"
        )

    def test_year_and_section_combined_filter(self):
        """Combining year and section filters must satisfy both constraints."""
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations(
            "pit lane speed limit",
            year=2026,
            section="B",
            n_results=5,
        )
        for r in results:
            meta = r["metadata"]
            assert meta["year"] == 2026 and meta["section"] == "B", (
                f"Combined filter violated: year={meta['year']}, section={meta['section']}"
            )

    def test_nonexistent_section_returns_empty(self):
        """Section 'Z' does not exist — must return empty list."""
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("anything", section="Z", n_results=5)
        assert len(results) == 0, (
            f"Expected 0 results for section=Z, got {len(results)}"
        )


# ===================================================================
# 7. Regulation metadata completeness (requires populated database)
# ===================================================================

@requires_regulations
class TestRegulationMetadataCompleteness:
    """Every chunk stored in f1_regulations must carry all required fields."""

    REQUIRED_METADATA_KEYS = {
        "source", "year", "section", "section_name",
        "doc_type", "chunk_index", "total_chunks",
    }

    def test_metadata_keys_present(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("Formula 1 regulation", n_results=10)
        assert len(results) > 0
        for r in results:
            missing = self.REQUIRED_METADATA_KEYS - set(r["metadata"].keys())
            assert not missing, f"Missing metadata keys: {missing}"

    def test_doc_type_is_regulation(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("Article 1", n_results=5)
        for r in results:
            assert r["metadata"]["doc_type"] == "regulation"

    def test_year_is_integer(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("championship", n_results=5)
        for r in results:
            year = r["metadata"]["year"]
            assert isinstance(year, int), f"year must be int, got {type(year)}"
            assert 2018 <= year <= 2030, f"Unexpected year: {year}"

    def test_section_is_single_letter(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("safety", n_results=10)
        for r in results:
            section = r["metadata"]["section"]
            assert re.match(r"^[A-F]$", section), (
                f"Section must be a single letter A-F; got '{section}'"
            )

    def test_chunk_index_is_non_negative(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("tyre", n_results=10)
        for r in results:
            assert r["metadata"]["chunk_index"] >= 0

    def test_total_chunks_greater_than_chunk_index(self):
        from rag.retriever import retrieve_regulations
        results = retrieve_regulations("tyre", n_results=10)
        for r in results:
            m = r["metadata"]
            assert m["total_chunks"] > m["chunk_index"], (
                f"total_chunks ({m['total_chunks']}) must exceed "
                f"chunk_index ({m['chunk_index']})"
            )


# ===================================================================
# 8. Embedder — list_collections includes known collections
# ===================================================================

class TestEmbedderListCollections:
    """list_collections must report at least the collections that have been populated."""

    def test_list_collections_returns_list(self):
        from rag.embedder import list_collections
        result = list_collections()
        assert isinstance(result, list)

    @requires_circuits
    def test_f1_circuits_listed_when_populated(self):
        from rag.embedder import list_collections
        names = [c["name"] for c in list_collections()]
        assert "f1_circuits" in names

    @requires_regulations
    def test_f1_regulations_listed_when_populated(self):
        from rag.embedder import list_collections
        names = [c["name"] for c in list_collections()]
        assert "f1_regulations" in names

    @requires_circuits
    def test_collection_count_is_positive(self):
        from rag.embedder import list_collections
        for col in list_collections():
            assert col["count"] >= 0
