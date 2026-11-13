import json

import pytest

from filings_rag.parser import Block
from filings_rag.retrieval_eval import (
    EvidenceNotFound,
    Question,
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
    assert results["section"] == {1: 1.0, 5: 1.0, 10: 1.0}
    assert results["fixed"][10] == 1.0


def test_recall_is_a_miss_when_the_answer_page_is_not_retrieved():
    filler = [Block("text", n, "Item 7", text=f"quarterly discussion {n}") for n in range(4, 30)]
    questions = [question("quarterly discussion", "single supplier")]

    results, _, _ = evaluate_strategies({DOC: [*blocks(), *filler]}, questions)

    assert results["section"][1] == 0.0


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
    results = {"fixed": {1: 0.5, 5: 0.75, 10: 1.0}, "section": {1: 1.0, 5: 1.0, 10: 1.0}}

    report = render_results(results, 4, ["a problem"])

    assert "| fixed | 50.0% | 75.0% | 100.0% |" in report
    assert "| section | 100.0% | 100.0% | 100.0% |" in report
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

    assert "| section | 100.0% |" in (tmp_path / "RETRIEVAL_RESULTS.md").read_text()


def test_evaluate_corpus_without_a_question_file_reports_not_measured(tmp_path):
    write_corpus(tmp_path, None)

    evaluate_corpus(tmp_path, tmp_path / "questions.jsonl")

    assert "Not measured" in (tmp_path / "RETRIEVAL_RESULTS.md").read_text()


def test_load_questions_skips_blank_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    row = {"question": "q", "document": DOC, "answer": "a", "evidence": "e"}
    path.write_text(json.dumps(row) + "\n\n")

    assert load_questions(path) == [Question("q", DOC, "a", "e")]
