"""Dense search in Postgres with the pgvector extension.

`PgVectorIndex` does the same job as `InMemoryDenseIndex`, but the vectors live in a table. Start
a local database with `docker compose -f docker/pgvector.yml up -d`. The driver is `pg8000`, which
is written in Python and needs no native Postgres library.
"""

import re
from collections.abc import Sequence
from urllib.parse import unquote, urlparse

from filings_rag.bm25 import Hit
from filings_rag.chunking import Chunk

TABLE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def connect(url: str):
    """Open a connection from a URL such as postgresql://postgres@127.0.0.1:5433/postgres."""
    import pg8000.dbapi

    parts = urlparse(url)
    return pg8000.dbapi.connect(
        user=unquote(parts.username or "postgres"),
        password=unquote(parts.password) if parts.password else None,
        host=parts.hostname or "127.0.0.1",
        port=parts.port or 5432,
        database=(parts.path or "/postgres").lstrip("/"),
    )


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"


class PgVectorIndex:
    def __init__(self, connection, table: str):
        self._connection = connection
        self._table = _checked_table_name(table)

    @classmethod
    def build(
        cls,
        connection,
        table: str,
        chunks: list[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> "PgVectorIndex":
        """Create the table if needed, replace its rows with `chunks` and return an index."""
        index = cls(connection, table)
        index._replace_rows(chunks, vectors)
        return index

    def search(self, vector: Sequence[float], k: int, document: str | None = None) -> list[Hit]:
        where = "WHERE document = %s" if document is not None else ""
        query = (
            "SELECT chunk_id, document, section, first_page, last_page, text, "
            f"1 - (embedding <=> %s::vector) FROM {self._table} {where} "
            "ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"  # chunk_id settles ties
        )
        literal = vector_literal(vector)
        arguments = [literal, *([document] if document is not None else []), literal, k]
        cursor = self._connection.cursor()
        try:
            cursor.execute(query, arguments)
            rows = cursor.fetchall()
        finally:
            # The driver opens a transaction on every query. End it even after an error, because
            # an open transaction keeps a lock on the table and blocks the next build that drops
            # it, and a failed one rejects every later statement.
            self._connection.rollback()
        return [
            Hit(Chunk(chunk_id, doc, section, first_page, last_page, text), float(score))
            for chunk_id, doc, section, first_page, last_page, text, score in rows
        ]

    def _replace_rows(self, chunks: list[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        try:
            self._write_rows(chunks, vectors)
        except Exception:
            self._connection.rollback()  # leave the connection usable for the caller
            raise

    def _write_rows(self, chunks: list[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        cursor = self._connection.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(f"DROP TABLE IF EXISTS {self._table}")
        dimensions = len(vectors[0]) if len(vectors) else 1
        cursor.execute(
            f"CREATE TABLE {self._table} (chunk_id text PRIMARY KEY, document text NOT NULL, "
            "section text, first_page integer NOT NULL, last_page integer NOT NULL, "
            f"text text NOT NULL, embedding vector({int(dimensions)}) NOT NULL)"
        )
        cursor.execute(f"CREATE INDEX ON {self._table} (document)")
        for chunk, vector in zip(chunks, vectors):
            cursor.execute(
                f"INSERT INTO {self._table} VALUES (%s, %s, %s, %s, %s, %s, %s::vector)",
                [
                    chunk.chunk_id,
                    chunk.document,
                    chunk.section,
                    chunk.first_page,
                    chunk.last_page,
                    chunk.text,
                    vector_literal(vector),
                ],
            )
        self._connection.commit()


def _checked_table_name(table: str) -> str:
    """Table names cannot be passed as query parameters, so only plain names are accepted."""
    if not TABLE_NAME_RE.match(table):
        raise ValueError(f"Not a valid table name: {table!r}")
    return table
