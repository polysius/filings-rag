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


CHUNKS = [
    Chunk("A#0", "A", "Item 1", 1, 2, "supplier text"),
    Chunk("A#1", "A", None, 3, 3, "lease text"),
    Chunk("B#0", "B", "Item 7", 5, 6, "supplier text again"),
]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]


@pytest.mark.skipif(not PG_URL, reason="set FILINGS_RAG_PG_URL to a pgvector database to run this")
class TestAgainstARealDatabase:
    """Needs `docker compose -f docker/pgvector.yml up -d` and the URL that file prints."""

    @pytest.fixture
    def connection(self):
        connection = connect(PG_URL)
        yield connection
        connection.close()

    def build(self, connection, chunks=CHUNKS, vectors=VECTORS) -> PgVectorIndex:
        return PgVectorIndex.build(connection, "test_chunks", chunks, vectors)

    def test_the_nearest_chunk_comes_first_with_its_pages(self, connection):
        hits = self.build(connection).search([1.0, 0.0, 0.0], k=3)

        assert [h.chunk.chunk_id for h in hits] == ["A#0", "B#0", "A#1"]
        assert hits[0].chunk == CHUNKS[0]
        assert hits[0].score == pytest.approx(1.0)

    def test_search_can_be_limited_to_one_filing(self, connection):
        hits = self.build(connection).search([1.0, 0.0, 0.0], k=3, document="B")

        assert [h.chunk.chunk_id for h in hits] == ["B#0"]

    def test_k_limits_the_hits_and_building_again_replaces_the_rows(self, connection):
        index = self.build(connection)
        assert len(index.search([1.0, 0.0, 0.0], k=1)) == 1

        index = self.build(connection)  # on the same connection, after a search

        assert len(index.search([1.0, 0.0, 0.0], k=10)) == 3

    def test_equal_scores_are_ordered_by_chunk_id_whatever_k_is(self, connection):
        chunks = [Chunk(f"D#{n}", "D", None, 1, 1, "same") for n in range(12)]
        index = self.build(connection, chunks, [[1.0, 0.0, 0.0]] * 12)

        wanted = sorted(chunk.chunk_id for chunk in chunks)  # text order, so D#10 before D#2
        assert [h.chunk.chunk_id for h in index.search([1.0, 0.0, 0.0], k=12)] == wanted
        assert [h.chunk.chunk_id for h in index.search([1.0, 0.0, 0.0], k=3)] == wanted[:3]

    def test_a_failed_search_leaves_the_connection_usable(self, connection):
        index = self.build(connection)

        with pytest.raises(Exception, match="dimensions"):
            index.search([1.0, 0.0, 0.0, 0.0], k=3)  # the table holds 3 dimensional vectors

        assert len(index.search([1.0, 0.0, 0.0], k=3)) == 3

    def test_a_failed_build_leaves_the_connection_usable(self, connection):
        index = self.build(connection)
        bad_chunks = [Chunk("X#0", "X", None, 1, 1, "bad \x00 text")]  # Postgres rejects NUL

        with pytest.raises(Exception, match="0x00"):
            self.build(connection, bad_chunks, [[1.0, 0.0, 0.0]])

        assert self.build(connection).search([1.0, 0.0, 0.0], k=3)
        assert index is not None

    def test_quotes_and_percent_signs_in_text_are_stored_as_they_are(self, connection):
        chunk = Chunk("Q#0", "it's", "Item 1", 1, 1, "100% of 'sales'; DROP TABLE x; --")
        index = self.build(connection, [chunk], [[1.0, 0.0, 0.0]])

        hit = index.search([1.0, 0.0, 0.0], k=1, document="it's")[0]

        assert hit.chunk == chunk
