import pytest
from fakes import KeywordEmbedder, LengthReranker

from filings_rag.bm25 import Bm25Index, Hit
from filings_rag.chunking import Chunk
from filings_rag.dense import InMemoryDenseIndex
from filings_rag.retrievers import (
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    reciprocal_rank_fusion,
)


def chunk(number: int, text: str, document: str = "DOC") -> Chunk:
    return Chunk(f"{document}#{number}", document, None, 1, 1, text)


def hits(*chunks: Chunk) -> list[Hit]:
    return [Hit(c, 0.0) for c in chunks]


class StaticRetriever:
    """Returns a fixed ranking and remembers how it was asked."""

    def __init__(self, ranking: list[Hit]):
        self.ranking = ranking
        self.requests: list[tuple[str, int, str | None]] = []

    def search(self, query: str, k: int, document: str | None = None) -> list[Hit]:
        self.requests.append((query, k, document))
        return self.ranking[:k]


A, B, C, D = (chunk(n, f"text {n}") for n in range(4))


def test_a_chunk_in_both_rankings_beats_chunks_in_one():
    fused = reciprocal_rank_fusion([hits(A, B), hits(C, B)])

    assert fused[0].chunk == B


def test_fusion_scores_follow_the_reciprocal_rank_formula():
    fused = reciprocal_rank_fusion([hits(A, B), hits(B)])

    scores = {h.chunk.chunk_id: h.score for h in fused}
    assert scores[A.chunk_id] == pytest.approx(1 / 61)
    assert scores[B.chunk_id] == pytest.approx(1 / 62 + 1 / 61)


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_fusion_keeps_chunks_that_only_one_ranking_found():
    fused = reciprocal_rank_fusion([hits(A), hits(B)])

    assert {h.chunk.chunk_id for h in fused} == {A.chunk_id, B.chunk_id}


def test_hybrid_search_finds_a_chunk_that_only_the_dense_side_matches():
    chunks = [chunk(0, "our vendor is a single source"), chunk(1, "we lease office space")]
    keyword = Bm25Index(chunks)
    embedder = KeywordEmbedder()
    vectors = embedder.embed_documents([c.text for c in chunks])
    dense = DenseRetriever(embedder, InMemoryDenseIndex(chunks, vectors))

    query = "who is our supplier"  # shares no word with the first chunk except "our"
    assert keyword.search(query, 1)[0].chunk.chunk_id == "DOC#0"  # via "our" alone
    hybrid = HybridRetriever([keyword, dense]).search(query, 2)

    assert hybrid[0].chunk.chunk_id == "DOC#0"


def test_hybrid_asks_each_retriever_for_enough_candidates():
    first, second = StaticRetriever(hits(A, B)), StaticRetriever(hits(C))

    HybridRetriever([first, second], candidates=50).search("q", k=10, document="DOC")
    HybridRetriever([first], candidates=5).search("q", k=10)

    assert first.requests[0] == ("q", 50, "DOC")
    assert first.requests[1] == ("q", 10, None)  # never fewer than k


def test_hybrid_returns_at_most_k_chunks():
    ranking = StaticRetriever(hits(A, B, C, D))

    assert len(HybridRetriever([ranking]).search("q", k=2)) == 2


def test_reranking_reorders_the_pool_by_reranker_score():
    short, medium, long = chunk(0, "xx"), chunk(1, "xxxx"), chunk(2, "xxxxxxxx")
    base = StaticRetriever(hits(long, medium, short))
    reranker = LengthReranker()

    result = RerankingRetriever(base, reranker).search("q", k=3)

    assert [h.chunk for h in result] == [short, medium, long]
    assert reranker.calls == [("q", ["xxxxxxxx", "xxxx", "xx"])]


def test_results_beyond_the_pool_keep_their_order_after_the_pool():
    first, second, third = chunk(0, "xxxxxx"), chunk(1, "xx"), chunk(2, "x")
    base = StaticRetriever(hits(first, second, third))
    reranker = LengthReranker()

    result = RerankingRetriever(base, reranker, pool=2).search("q", k=3)

    assert [h.chunk for h in result] == [second, first, third]
    assert len(reranker.calls[0][1]) == 2


def test_reranking_asks_the_base_for_the_whole_pool_and_passes_the_filing():
    base = StaticRetriever(hits(A, B))

    RerankingRetriever(base, LengthReranker(), pool=30).search("q", k=5, document="DOC")

    assert base.requests == [("q", 30, "DOC")]


def test_reranking_nothing_does_not_call_the_model_wrongly():
    result = RerankingRetriever(StaticRetriever([]), LengthReranker()).search("q", k=5)

    assert result == []


def test_dense_retriever_embeds_the_query_and_passes_the_filing_on():
    chunks = [chunk(0, "supplier", "A"), chunk(1, "supplier", "B")]
    embedder = KeywordEmbedder()
    index = InMemoryDenseIndex(chunks, embedder.embed_documents([c.text for c in chunks]))

    result = DenseRetriever(embedder, index).search("our vendor", k=5, document="B")

    assert embedder.queries == ["our vendor"]
    assert [h.chunk.document for h in result] == ["B"]
