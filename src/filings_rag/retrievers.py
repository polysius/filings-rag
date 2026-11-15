"""Ways to retrieve chunks: dense search, hybrid search with rank fusion, and reranking.

Every retriever has the same `search(query, k, document)` method, so one can wrap another.
"""

from typing import Protocol

from filings_rag.bm25 import Hit
from filings_rag.dense import DenseIndex, Embedder

RRF_CONSTANT = 60  # the usual damping constant of reciprocal rank fusion
CANDIDATES_PER_RETRIEVER = 50
RERANK_POOL = 30


class Retriever(Protocol):
    def search(self, query: str, k: int, document: str | None = None) -> list[Hit]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text. A higher score means a better match."""
        ...


class DenseRetriever:
    def __init__(self, embedder: Embedder, index: DenseIndex):
        self._embedder = embedder
        self._index = index

    def search(self, query: str, k: int, document: str | None = None) -> list[Hit]:
        return self._index.search(self._embedder.embed_query(query), k, document)


def reciprocal_rank_fusion(rankings: list[list[Hit]]) -> list[Hit]:
    """Merge ranked lists. A chunk scores 1 / (60 + rank) in each list that holds it.

    Only ranks are used, so keyword scores and cosine similarities need no common scale.
    """
    scores: dict[str, float] = {}
    chunks = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunks[hit.chunk.chunk_id] = hit.chunk
            scores[hit.chunk.chunk_id] = scores.get(hit.chunk.chunk_id, 0.0) + 1 / (
                RRF_CONSTANT + rank
            )
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [Hit(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered]


class HybridRetriever:
    """Asks several retrievers for candidates and fuses their rankings."""

    def __init__(self, retrievers: list[Retriever], candidates: int = CANDIDATES_PER_RETRIEVER):
        self._retrievers = retrievers
        self._candidates = candidates

    def search(self, query: str, k: int, document: str | None = None) -> list[Hit]:
        pool = max(self._candidates, k)
        rankings = [retriever.search(query, pool, document) for retriever in self._retrievers]
        return reciprocal_rank_fusion(rankings)[:k]


class RerankingRetriever:
    """Re-orders the top results of another retriever with a reranker."""

    def __init__(self, base: Retriever, reranker: Reranker, pool: int = RERANK_POOL):
        self._base = base
        self._reranker = reranker
        self._pool = pool

    def search(self, query: str, k: int, document: str | None = None) -> list[Hit]:
        hits = self._base.search(query, max(self._pool, k), document)
        top, rest = hits[: self._pool], hits[self._pool :]
        scores = self._reranker.score(query, [hit.chunk.text for hit in top])
        reranked = [Hit(hit.chunk, score) for hit, score in zip(top, scores)]
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return (reranked + rest)[:k]  # results past the pool keep their original order
