import os

import pytest

from filings_rag.chunking import Chunk
from filings_rag.pgvector_index import PgVectorIndex, connect, vector_literal

PG_URL = os.environ.get("FILINGS_RAG_PG_URL")


def test_a_vector_is_written_as_a_pgvector_literal():
    assert vector_literal([1.0, 0.5, -0.25]) == "[1,0.5,-0.25]"


@pytest.mark.parametrize("name", ["chunks; DROP TABLE x", "Chunks", "1chunks", "", "a-b"])
def test_unsafe_table_names_are_rejected(name):
    with pytest.raises(ValueError):
        PgVectorIndex(None, name)


@pytest.mark.skipif(not PG_URL, reason="set FILINGS_RAG_PG_URL to a pgvector database to run this")
class TestAgainstARealDatabase:
    """Needs `docker compose -f docker/pgvector.yml up -d` and the URL that file prints."""

    def build(self):
        chunks = [
            Chunk("A#0", "A", "Item 1", 1, 2, "supplier text"),
            Chunk("A#1", "A", None, 3, 3, "lease text"),
            Chunk("B#0", "B", "Item 7", 5, 6, "supplier text again"),
        ]
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]
        connection = connect(PG_URL)
        return PgVectorIndex.build(connection, "test_chunks", chunks, vectors), chunks

    def test_the_nearest_chunk_comes_first_with_its_pages(self):
        index, chunks = self.build()

        hits = index.search([1.0, 0.0, 0.0], k=3)

        assert [h.chunk.chunk_id for h in hits] == ["A#0", "B#0", "A#1"]
        assert hits[0].chunk == chunks[0]
        assert hits[0].score == pytest.approx(1.0)

    def test_search_can_be_limited_to_one_filing(self):
        index, _ = self.build()

        hits = index.search([1.0, 0.0, 0.0], k=3, document="B")

        assert [h.chunk.chunk_id for h in hits] == ["B#0"]

    def test_k_limits_the_hits_and_building_again_replaces_the_rows(self):
        index, _ = self.build()
        assert len(index.search([1.0, 0.0, 0.0], k=1)) == 1

        index, _ = self.build()

        assert len(index.search([1.0, 0.0, 0.0], k=10)) == 3
