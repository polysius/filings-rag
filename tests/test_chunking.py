from filings_rag.chunking import chunk_by_section, chunk_fixed
from filings_rag.parser import Block


def text(page: int, section: str, content: str) -> Block:
    return Block("text", page, section, text=content)


def test_fixed_chunks_run_across_section_boundaries():
    blocks = [text(1, "Item 1", "Business overview."), text(2, "Item 1A", "Risk overview.")]

    chunks = chunk_fixed("DOC", blocks)

    assert len(chunks) == 1
    assert chunks[0].text == "Business overview.\nRisk overview."
    assert (chunks[0].first_page, chunks[0].last_page) == (1, 2)


def test_section_chunks_stop_at_each_section_boundary():
    blocks = [text(1, "Item 1", "Business overview."), text(2, "Item 1A", "Risk overview.")]

    chunks = chunk_by_section("DOC", blocks)

    assert [c.section for c in chunks] == ["Item 1", "Item 1A"]
    assert [c.text for c in chunks] == ["Business overview.", "Risk overview."]


def test_section_chunks_keep_a_table_apart_from_the_text_around_it():
    blocks = [
        text(3, "Item 8", "Statement of operations"),
        Block("table", 3, "Item 8", rows=[["Total net sales", "100"]]),
        text(3, "Item 8", "Note 1"),
    ]

    chunks = chunk_by_section("DOC", blocks)

    assert [c.text for c in chunks] == [
        "Statement of operations",
        "Total net sales | 100",
        "Note 1",
    ]


def test_fixed_chunks_include_tables_as_text():
    blocks = [text(3, "Item 8", "Statement"), Block("table", 3, "Item 8", rows=[["Sales", "1"]])]

    assert chunk_fixed("DOC", blocks)[0].text == "Statement\nSales | 1"


def test_chunks_stay_within_the_target_and_keep_the_page_span():
    blocks = [text(page, "Item 7", "word " * 30) for page in (4, 5, 6)]  # 150 characters each

    chunks = chunk_by_section("DOC", blocks, target=200)

    assert all(len(c.text) <= 200 for c in chunks)
    assert [(c.first_page, c.last_page) for c in chunks] == [(4, 4), (5, 5), (6, 6)]


def test_an_oversized_block_is_split_without_cutting_words():
    block = text(7, "Item 7", " ".join(f"word{n}" for n in range(100)))

    chunks = chunk_fixed("DOC", [block], target=100)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.first_page == c.last_page == 7 for c in chunks)
    assert " ".join(c.text for c in chunks) == block.text.replace("\n", " ")


def test_a_long_unbroken_string_is_still_cut_to_the_target():
    chunks = chunk_fixed("DOC", [text(1, "Item 1", "x" * 250)], target=100)

    assert [len(c.text) for c in chunks] == [100, 100, 50]


def test_chunk_ids_are_numbered_per_document_and_spans_can_be_queried():
    chunks = chunk_by_section("DOC", [text(1, "A", "one"), text(2, "B", "two")])

    assert [c.chunk_id for c in chunks] == ["DOC#0", "DOC#1"]
    assert chunks[0].spans_page(1)
    assert not chunks[0].spans_page(2)


def test_a_long_table_is_cut_between_rows():
    rows = [[f"Line item {n}", f"${n},000"] for n in range(60)]

    chunks = chunk_by_section("DOC", [Block("table", 5, "Item 8", rows=rows)], target=200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert all(line.startswith("Line item ") for line in chunk.text.splitlines())


def test_empty_blocks_make_no_chunks():
    assert chunk_fixed("DOC", [text(1, "A", "   ")]) == []
