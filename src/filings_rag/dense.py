"""Dense (embedding) search over chunks.

An `Embedder` turns text into vectors, and a `DenseIndex` finds the chunks whose vectors are
closest to a query vector. Vectors are expected to have length 1, so the dot product is the
cosine similarity.
"""

import hashlib
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Protocol

import numpy as np

from filings_rag.bm25 import Hit
from filings_rag.chunking import Chunk


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        """Queries and documents may be embedded differently, so they have separate methods."""
        ...


class DenseIndex(Protocol):
    def search(self, vector: Sequence[float], k: int, document: str | None = None) -> list[Hit]:
        """Return the `k` nearest chunks. With `document`, only chunks of that filing compete."""
        ...


class InMemoryDenseIndex:
    def __init__(self, chunks: list[Chunk], vectors: Sequence[Sequence[float]]):
        self._chunks = chunks
        self._matrix = np.asarray(vectors, dtype=np.float32)
        self._documents = np.array([chunk.document for chunk in chunks])

    def search(self, vector: Sequence[float], k: int, document: str | None = None) -> list[Hit]:
        if not self._chunks:
            return []
        scores = self._matrix @ np.asarray(vector, dtype=np.float32)
        if document is None:
            candidates = np.arange(len(self._chunks))
        else:
            candidates = np.flatnonzero(self._documents == document)
        best_first = candidates[np.argsort(-scores[candidates], kind="stable")][:k]
        return [Hit(self._chunks[i], float(scores[i])) for i in best_first]


class CachedEmbedder:
    """Remembers document embeddings in a SQLite file, so a rerun does not embed them again.

    The key is a hash of the model name and the text. Change `model_name` when the model changes.
    """

    def __init__(self, inner: Embedder, path: Path, model_name: str):
        self._inner = inner
        self._path = path
        self._model_name = model_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )
            database.commit()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        with closing(sqlite3.connect(self._path)) as database:
            cached = self._load(database, keys)
            # The first position of each text that is not cached, so a repeated text is embedded once.
            first_position: dict[str, int] = {}
            for position, key in enumerate(keys):
                if key not in cached:
                    first_position.setdefault(key, position)
            missing = list(first_position.values())
            if missing:
                vectors = self._inner.embed_documents([texts[i] for i in missing])
                for i, vector in zip(missing, vectors):
                    cached[keys[i]] = np.asarray(vector, dtype=np.float32).tobytes()
                    database.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                        (keys[i], cached[keys[i]]),
                    )
                database.commit()
        return [np.frombuffer(cached[key], dtype=np.float32).tolist() for key in keys]

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._model_name}\n{text}".encode()).hexdigest()

    @staticmethod
    def _load(database: sqlite3.Connection, keys: list[str]) -> dict[str, bytes]:
        found = {}
        unique = list(set(keys))
        for start in range(0, len(unique), 500):  # SQLite limits the number of parameters
            batch = unique[start : start + 500]
            marks = ",".join("?" * len(batch))
            query = f"SELECT key, vector FROM embeddings WHERE key IN ({marks})"
            found.update(database.execute(query, batch).fetchall())
        return found
