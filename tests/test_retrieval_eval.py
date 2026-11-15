import json

import pytest
from fakes import KeywordEmbedder, LengthReranker

from filings_rag.dense import InMemoryDenseIndex
from filings_rag.parser import Block
from filings_rag.retrieval_eval import (
    EvidenceNotFound,
    Question,
    QuestionFileError,
    evaluate_corpus,
    evaluate_strategies,
    gold_pages,
    load_questions,
    render_results,
)

DOC = "ACME_10-K_2024-12-31"


def blocks() -> list[Block]:
    return [
        Block("text", 1, "Item 1", text="Acme builds widgets for industrial customers."),
        Block("text", 2, "Item 1A", text="Our supply chain depends on a single   supplier."),
        Block("table", 3, "Item 8", rows=[["Total net sales", "1,000"]]),
    ]


def question(text: str, evidence: str, document: str = DOC) -> Question:
    return Question(text, document, "answer", evidence)


def test_gold_pages_match_evidence_ignoring_case_and_spacing():
    assert gold_pages(blocks(), "supply chain depends on a single supplier") == {2}


def test_gold_pages_find_evidence_inside_a_table():
    assert gold_pages(blocks(), "Total net sales | 1,000") == {3}


def test_gold_pages_fail_when_the_evidence_is_missing():
    with pytest.raises(EvidenceNotFound):
        gold_pages(blocks(), "something the filing never says")


def test_recall_counts_a_hit_when_the_chunk_spans_the_gold_page():
    questions = [question("what does the supply chain depend on", "single supplier")]

    results, scored, problems = evaluate_strategies({DOC: blocks()}, questions)

    assert scored == 1 and problems == []
    assert results[("section", "bm25")] == {1: 1.0, 5: 1.0, 10: 1.0}
    assert results[("fixed", "bm25")][10] == 1.0


def test_recall_is_a_miss_when_the_answer_page_is_not_retrieved():
    filler = [Block("text", n, "Item 7", text=f"quarterly discussion {n}") for n in range(4, 30)]
    questions = [question("quarterly discussion", "single supplier")]

    results, _, _ = evaluate_strategies({DOC: [*blocks(), *filler]}, questions)

    assert results[("section", "bm25")][1] == 0.0


def vendor_filing() -> list[Block]:
    """Page 2 answers the question with different words than the question uses."""
    filler = [Block("text", n, "Item 7", text=f"supplier supplier notes {n}") for n in range(3, 20)]
    return [
        Block("text", 1, "Item 1", text="office lease supplier supplier supplier supplier"),
        Block("text", 2, "Item 1A", text="we rely on one vendor for all of our parts"),
        *filler,
    ]


def test_dense_and_hybrid_methods_are_scored_when_an_embedder_is_given():
    questions = [question("supplier dependence", "one vendor")]

    results, scored, _ = evaluate_strategies({DOC: vendor_filing()}, questions, KeywordEmbedder())

    assert scored == 1
    methods = {method for _, method in results}
    assert methods == {"bm25", "dense", "hybrid"}
    assert {strategy for strategy, _ in results} == {"fixed", "section"}
    # The vendor page shares no query word with the question, but the embedder links the words.
    assert results[("section", "dense")][5] == 1.0
    assert results[("section", "bm25")][1] == 0.0


def test_a_reranker_adds_the_reranked_hybrid_method():
    questions = [question("which supplier do we depend on", "one vendor")]
    reranker = LengthReranker()

    results, _, _ = evaluate_strategies(
        {DOC: vendor_filing()}, questions, KeywordEmbedder(), reranker
    )

    assert ("section", "hybrid+rerank") in results
    assert reranker.calls


def test_only_filings_that_questions_ask_about_are_embedded():
    other = [Block("text", 1, "Item 1", text="cyber litigation other filing")]
    documents = {DOC: vendor_filing(), "OTHER_10-K_2024-12-31": other}
    embedder = KeywordEmbedder()

    evaluate_strategies(documents, [question("supplier", "one vendor")], embedder)

    assert not any("other filing" in text for text in embedder.embedded_texts)


def test_a_dense_index_is_built_per_strategy_through_the_given_factory():
    tables = []

    def make_index(table, chunks, vectors):
        tables.append(table)
        return InMemoryDenseIndex(chunks, vectors)

    evaluate_strategies(
        {DOC: vendor_filing()},
        [question("supplier", "one vendor")],
        KeywordEmbedder(),
        make_index=make_index,
    )

    assert tables == ["chunks_fixed", "chunks_section"]


def test_without_an_embedder_only_bm25_is_scored():
    results, _, _ = evaluate_strategies({DOC: blocks()}, [question("supply chain", "supplier")])

    assert {method for _, method in results} == {"bm25"}


def test_questions_that_cannot_be_scored_are_listed_not_counted():
    questions = [
        question("supply chain", "single supplier"),
        question("anything", "text that is not there"),
        question("anything", "single supplier", document="OTHER_10-K_2024-12-31"),
    ]

    _, scored, problems = evaluate_strategies({DOC: blocks()}, questions)

    assert scored == 1
    assert len(problems) == 2
    assert any("evidence not found" in p for p in problems)
    assert any("filing is not parsed" in p for p in problems)


def test_render_says_not_measured_without_questions():
    report = render_results({}, 0, [])

    assert "Not measured" in report
    assert "| fixed" not in report


def test_render_has_a_row_per_strategy():
    results = {
        ("fixed", "bm25"): {1: 0.5, 5: 0.75, 10: 1.0},
        ("section", "hybrid"): {1: 1.0, 5: 1.0, 10: 1.0},
    }

    report = render_results(results, 4, ["a problem"])

    assert "| fixed | bm25 | 50.0% | 75.0% | 100.0% |" in report
    assert "| section | hybrid | 100.0% | 100.0% | 100.0% |" in report
    assert "- a problem" in report


def write_corpus(data_dir, questions: list[dict] | None) -> None:
    (data_dir / "parsed").mkdir()
    (data_dir / "manifest.json").write_text(json.dumps([{"path": f"raw/{DOC}.htm"}]))
    parsed = {"page_count": 3, "blocks": [b.__dict__ for b in blocks()]}
    (data_dir / "parsed" / f"{DOC}.json").write_text(json.dumps(parsed))
    if questions is not None:
        lines = [json.dumps(q) for q in questions]
        (data_dir / "questions.jsonl").write_text("\n".join(lines) + "\n")


def test_evaluate_corpus_writes_the_report(tmp_path):
    write_corpus(
        tmp_path,
        [
            {
                "question": "what does the supply chain depend on",
                "document": DOC,
                "answer": "a single supplier",
                "evidence": "single supplier",
            }
        ],
    )

    evaluate_corpus(tmp_path, tmp_path / "questions.jsonl")

    assert "| section | bm25 | 100.0% |" in (tmp_path / "RETRIEVAL_RESULTS.md").read_text()


def test_evaluate_corpus_without_a_question_file_reports_not_measured(tmp_path):
    write_corpus(tmp_path, None)

    evaluate_corpus(tmp_path)

    assert "Not measured" in (tmp_path / "RETRIEVAL_RESULTS.md").read_text()


def test_blank_or_punctuation_only_evidence_is_never_found():
    for evidence in ("", "   ", "."):
        with pytest.raises(EvidenceNotFound):
            gold_pages(blocks(), evidence)


def test_evidence_copied_from_a_browser_table_matches():
    assert gold_pages(blocks(), "Total net sales\t1,000") == {3}
    assert gold_pages(blocks(), "total NET sales $1,000") == {3}


def test_zero_width_characters_in_evidence_are_ignored():
    assert gold_pages(blocks(), "single\u200b supplier") == {2}


def test_load_questions_reports_the_line_number_of_a_bad_line(tmp_path):
    path = tmp_path / "q.jsonl"
    good = json.dumps({"question": "q", "document": DOC, "answer": "a", "evidence": "e"})
    path.write_text(f"{good}\n{good}\nnot json\n")

    with pytest.raises(QuestionFileError, match=r"q\.jsonl:3: not valid JSON"):
        load_questions(path)


@pytest.mark.parametrize(
    "row, message",
    [
        ({"question": "q", "document": DOC, "answer": "a"}, "evidence"),
        ({"question": "q", "document": DOC, "answer": "a", "evidence": None}, "evidence"),
        ({"question": "q", "document": DOC, "answer": "a", "evidence": " "}, "evidence"),
        (["not", "an", "object"], "expected a JSON object"),
    ],
)
def test_load_questions_rejects_incomplete_rows(tmp_path, row, message):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(QuestionFileError, match=message):
        load_questions(path)


def test_load_questions_ignores_extra_keys_and_a_byte_order_mark(tmp_path):
    path = tmp_path / "q.jsonl"
    row = {"question": "q", "document": DOC, "answer": "a", "evidence": "e", "note": "x"}
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(row).encode() + b"\n")

    assert load_questions(path) == [Question("q", DOC, "a", "e")]


def test_a_given_questions_file_that_is_missing_is_an_error(tmp_path):
    write_corpus(tmp_path, None)

    with pytest.raises(FileNotFoundError):
        evaluate_corpus(tmp_path, tmp_path / "typo.jsonl")


def test_default_questions_file_is_read_from_the_data_directory(tmp_path):
    row = {"question": "supply chain", "document": DOC, "answer": "a", "evidence": "single"}
    write_corpus(tmp_path, [row])

    evaluate_corpus(tmp_path)

    assert "| section | bm25 | 100.0% |" in (tmp_path / "RETRIEVAL_RESULTS.md").read_text()


def test_load_questions_skips_blank_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    row = {"question": "q", "document": DOC, "answer": "a", "evidence": "e"}
    path.write_text(json.dumps(row) + "\n\n")

    assert load_questions(path) == [Question("q", DOC, "a", "e")]
