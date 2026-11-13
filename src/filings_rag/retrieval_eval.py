"""Measure retrieval recall for each chunking strategy on a hand-labeled question set.

A question file holds one JSON object per line:

    {"question": "...", "document": "AAPL_10-K_2024-09-28", "answer": "...", "evidence": "..."}

`evidence` is a sentence or table row copied from the filing, from within one paragraph or one
table. Words are compared without punctuation or case. Gold pages are the pages of the
parsed blocks that contain it, so labels follow the parser's physical page numbers and survive a
re-parse. A retrieved chunk is a hit when its page span contains a gold page.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from filings_rag.bm25 import Bm25Index
from filings_rag.chunking import STRATEGIES, block_text
from filings_rag.extract_corpus import load_blocks
from filings_rag.parser import Block

RECALL_CUTOFFS = (1, 5, 10)


@dataclass(frozen=True)
class Question:
    question: str
    document: str
    answer: str
    evidence: str


class EvidenceNotFound(Exception):
    """The evidence snippet of a question does not appear in its filing."""


class QuestionFileError(Exception):
    """A line of the question file is not a valid question."""


QUESTION_FIELDS = ("question", "document", "answer", "evidence")


def load_questions(path: Path) -> list[Question]:
    # utf-8-sig, because Windows tools often write a byte order mark at the start of the file.
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    questions = []
    for number, line in enumerate(lines, start=1):
        if line.strip():
            questions.append(_parse_question(line, f"{path}:{number}"))
    return questions


def _parse_question(line: str, where: str) -> Question:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise QuestionFileError(f"{where}: not valid JSON ({error.msg})") from error
    if not isinstance(row, dict):
        raise QuestionFileError(f"{where}: expected a JSON object")
    problems = [f for f in QUESTION_FIELDS if not isinstance(row.get(f), str) or not row[f].strip()]
    if problems:
        raise QuestionFileError(f"{where}: missing or empty text in {', '.join(problems)}")
    return Question(**{field: row[field] for field in QUESTION_FIELDS})


# Zero width space, non-joiner, joiner and byte order mark, which the parser also removes.
ZERO_WIDTH_CODE_POINTS = (0x200B, 0x200C, 0x200D, 0xFEFF)
ZERO_WIDTH_RE = re.compile("[" + "".join(chr(code) for code in ZERO_WIDTH_CODE_POINTS) + "]")
WORD_RE = re.compile(r"\w+")


def normalize(text: str) -> str:
    """Lowercase words separated by single spaces.

    Punctuation and symbols are dropped on both sides, so evidence copied from a browser still
    matches the parsed text, where a table row is joined with " | " and "$" has its own cell.
    """
    return " ".join(WORD_RE.findall(ZERO_WIDTH_RE.sub("", text).lower()))


def gold_pages(blocks: list[Block], evidence: str) -> set[int]:
    """Pages of every block that contains the evidence snippet."""
    needle = normalize(evidence)
    if not needle:
        raise EvidenceNotFound(evidence)
    pages = {b.page for b in blocks if needle in normalize(block_text(b))}
    if not pages:
        raise EvidenceNotFound(evidence)
    return pages


def recall_at(index: Bm25Index, question: Question, pages: set[int], cutoffs: tuple[int, ...]):
    """For each cutoff, whether a top-k chunk of the question's filing spans a gold page."""
    hits = index.search(question.question, max(cutoffs), document=question.document)
    return {
        k: any(chunk_hit.chunk.spans_page(page) for chunk_hit in hits[:k] for page in pages)
        for k in cutoffs
    }


def evaluate_strategies(
    documents: dict[str, list[Block]], questions: list[Question]
) -> tuple[dict[str, dict[int, float]], int, list[str]]:
    """Return recall@k per strategy, the number of scored questions and the unscorable ones."""
    scorable, problems = _find_gold_pages(documents, questions)

    results = {}
    for name, chunker in STRATEGIES.items():
        chunks = [c for stem, blocks in documents.items() for c in chunker(stem, blocks)]
        index = Bm25Index(chunks)
        found = {k: 0 for k in RECALL_CUTOFFS}
        for question, pages in scorable:
            for k, hit in recall_at(index, question, pages, RECALL_CUTOFFS).items():
                found[k] += hit
        results[name] = {k: found[k] / len(scorable) if scorable else 0.0 for k in RECALL_CUTOFFS}
    return results, len(scorable), problems


def _find_gold_pages(
    documents: dict[str, list[Block]], questions: list[Question]
) -> tuple[list[tuple[Question, set[int]]], list[str]]:
    """Pair each question with its gold pages. Questions that cannot be paired are listed."""
    scorable, problems = [], []
    for question in questions:
        label = f"{question.document} / {question.question[:60]}"
        if question.document not in documents:
            problems.append(f"{label}: filing is not parsed")
            continue
        try:
            pages = gold_pages(documents[question.document], question.evidence)
        except EvidenceNotFound:
            problems.append(f"{label}: evidence not found in the filing")
            continue
        scorable.append((question, pages))
    return scorable, problems


def render_results(
    results: dict[str, dict[int, float]], question_count: int, problems: list[str]
) -> str:
    lines = ["# Retrieval results", ""]
    if not results or question_count == 0:
        lines += ["Not measured: there are no scorable questions yet.", ""]
    else:
        introduction = (
            f"BM25 over {question_count} questions, searching only the filing each question is "
            "about. A hit means a top-k chunk spans a page that holds the evidence."
        )
        lines += [
            introduction,
            "",
            "| Chunking | " + " | ".join(f"recall@{k}" for k in RECALL_CUTOFFS) + " |",
            "|----------|" + "|".join("---------" for _ in RECALL_CUTOFFS) + "|",
        ]
        for name, recalls in results.items():
            cells = " | ".join(f"{recalls[k]:.1%}" for k in RECALL_CUTOFFS)
            lines.append(f"| {name} | {cells} |")
        lines.append("")

    lines += ["## Questions that could not be scored", ""]
    lines += [f"- {problem}" for problem in problems] or ["None."]
    return "\n".join(lines) + "\n"


def evaluate_corpus(data_dir: Path, questions_path: Path | None = None) -> str:
    """Score every strategy over the parsed corpus and write `RETRIEVAL_RESULTS.md`.

    Without `questions_path` the questions are read from `questions.jsonl` in `data_dir`, and a
    missing file means no questions yet. A path that is given but missing is an error.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    documents = {}
    for entry in manifest:
        stem = Path(entry["path"]).stem
        parsed_path = data_dir / "parsed" / f"{stem}.json"
        if parsed_path.exists():
            documents[stem] = load_blocks(parsed_path)

    if questions_path is None:
        default_path = data_dir / "questions.jsonl"
        questions = load_questions(default_path) if default_path.exists() else []
    else:
        questions = load_questions(questions_path)  # raises if the file is missing
    results, scored, problems = evaluate_strategies(documents, questions)
    report = render_results(results, scored, problems)
    (data_dir / "RETRIEVAL_RESULTS.md").write_text(report, encoding="utf-8")
    return report
