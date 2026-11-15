from fakes import KeywordEmbedder

from filings_rag.chunking import Chunk
from filings_rag.dense import CachedEmbedder, InMemoryDenseIndex


def chunk(number: int, text: str, document: str = "DOC") -> Chunk:
    return Chunk(f"{document}#{number}", document, None, 1, 1, text)


def build_index(chunks: list[Chunk]) -> InMemoryDenseIndex:
    return InMemoryDenseIndex(chunks, KeywordEmbedder().embed_documents([c.text for c in chunks]))


def test_the_nearest_chunk_comes_first():
    chunks = [chunk(0, "office lease"), chunk(1, "supplier risk"), chunk(2, "cyber attack")]
    index = build_index(chunks)

    hits = index.search(KeywordEmbedder().embed_query("our vendor"), k=3)

    assert hits[0].chunk.chunk_id == "DOC#1"
    assert hits[0].score > hits[1].score


def test_k_limits_the_hits_and_a_filing_can_be_chosen():
    chunks = [chunk(0, "supplier", "A"), chunk(1, "supplier", "B"), chunk(2, "supplier", "B")]
    index = build_index(chunks)
    query = KeywordEmbedder().embed_query("supplier")

    assert len(index.search(query, k=2)) == 2
    assert {h.chunk.document for h in index.search(query, k=5, document="B")} == {"B"}
    assert index.search(query, k=5, document="missing") == []


def test_an_empty_index_returns_nothing():
    assert InMemoryDenseIndex([], []).search([1.0, 0.0], k=3) == []


def test_cached_embeddings_are_not_computed_twice(tmp_path):
    inner = KeywordEmbedder()
    embedder = CachedEmbedder(inner, tmp_path / "cache.sqlite", "model-a")

    first = embedder.embed_documents(["supplier risk", "office lease"])
    second = embedder.embed_documents(["supplier risk", "office lease"])

    assert first == second
    assert inner.embedded_texts == ["supplier risk", "office lease"]


def test_only_the_missing_texts_are_embedded(tmp_path):
    inner = KeywordEmbedder()
    embedder = CachedEmbedder(inner, tmp_path / "cache.sqlite", "model-a")
    embedder.embed_documents(["supplier risk"])

    vectors = embedder.embed_documents(["supplier risk", "office lease", "office lease"])

    assert inner.embedded_texts == ["supplier risk", "office lease"]
    assert vectors[1] == vectors[2]


def test_the_cache_survives_a_new_embedder_and_is_keyed_by_model(tmp_path):
    path = tmp_path / "cache.sqlite"
    CachedEmbedder(KeywordEmbedder(), path, "model-a").embed_documents(["supplier"])

    same_model = KeywordEmbedder()
    CachedEmbedder(same_model, path, "model-a").embed_documents(["supplier"])
    other_model = KeywordEmbedder()
    CachedEmbedder(other_model, path, "model-b").embed_documents(["supplier"])

    assert same_model.embedded_texts == []
    assert other_model.embedded_texts == ["supplier"]


def test_a_large_batch_is_looked_up_in_pieces(tmp_path):
    inner = KeywordEmbedder()
    embedder = CachedEmbedder(inner, tmp_path / "cache.sqlite", "model-a")
    texts = [f"supplier {n}" for n in range(1200)]  # more than SQLite allows in one query

    embedder.embed_documents(texts)
    embedder.embed_documents(texts)

    assert len(inner.embedded_texts) == 1200


def test_queries_are_passed_through_uncached(tmp_path):
    inner = KeywordEmbedder()
    embedder = CachedEmbedder(inner, tmp_path / "cache.sqlite", "model-a")

    embedder.embed_query("supplier")
    embedder.embed_query("supplier")

    assert inner.queries == ["supplier", "supplier"]
