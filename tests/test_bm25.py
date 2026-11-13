from filings_rag.bm25 import Bm25Index
from filings_rag.chunking import Chunk


def chunk(number: int, text: str, document: str = "DOC") -> Chunk:
    return Chunk(f"{document}#{number}", document, None, 1, 1, text)


def test_the_chunk_with_the_query_terms_ranks_first():
    index = Bm25Index(
        [
            chunk(0, "interest rate risk on our loan portfolio"),
            chunk(1, "supply chain disruption risk"),
            chunk(2, "we lease office space"),
        ]
    )

    hits = index.search("supply chain", k=3)

    assert [h.chunk.chunk_id for h in hits] == ["DOC#1"]


def test_a_rarer_term_counts_for_more_than_a_common_one():
    index = Bm25Index(
        [chunk(0, "risk risk risk"), chunk(1, "risk litigation"), chunk(2, "risk of rain")]
    )

    hits = index.search("litigation risk", k=3)

    assert hits[0].chunk.chunk_id == "DOC#1"


def test_search_can_be_limited_to_one_filing():
    index = Bm25Index([chunk(0, "cyber attack", "A"), chunk(0, "cyber attack", "B")])

    hits = index.search("cyber attack", k=5, document="B")

    assert [h.chunk.document for h in hits] == ["B"]


def test_k_limits_the_number_of_hits():
    index = Bm25Index([chunk(n, "revenue growth") for n in range(5)])

    assert len(index.search("revenue", k=2)) == 2


def test_no_match_and_an_empty_index_return_nothing():
    assert Bm25Index([chunk(0, "revenue")]).search("zebra", k=3) == []
    assert Bm25Index([]).search("revenue", k=3) == []


def test_an_index_with_no_words_does_not_divide_by_zero():
    index = Bm25Index([chunk(0, "$ | %"), chunk(1, "* * *")])

    assert index.search("risk", k=3) == []


def test_matching_ignores_case_and_punctuation():
    index = Bm25Index([chunk(0, "Net Income, in millions")])

    assert len(index.search("net-income", k=1)) == 1
