"""Run extraction over every parsed filing and score it against XBRL."""

import json
from dataclasses import dataclass
from pathlib import Path

from filings_rag.context import build_context
from filings_rag.evaluate import FieldScore, render_results, score_document
from filings_rag.extractor import Completer, extract_filing
from filings_rag.parser import Block
from filings_rag.schema import SCORED_FIELDS, FilingExtraction
from filings_rag.xbrl import reference_value


@dataclass
class CorpusRun:
    scores: list[FieldScore]
    failures: list[str]


def load_blocks(parsed_path: Path) -> list[Block]:
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    return [Block(**block) for block in parsed["blocks"]]


def read_cached(path: Path) -> FilingExtraction | None:
    """Return a saved extraction, or None if there is none or the file is unreadable."""
    try:
        return FilingExtraction.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers bad UTF-8 and pydantic's ValidationError
        return None


def write_atomically(path: Path, text: str) -> None:
    """Write via a temporary file, so a killed run never leaves a half-written result."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def extract_one(completer: Completer, data_dir: Path, entry: dict, stem: str) -> FilingExtraction:
    """Return the extraction for one filing, from the saved file when it is valid."""
    extracted_path = data_dir / "extracted" / f"{stem}.json"
    cached = read_cached(extracted_path)
    if cached:
        return cached

    blocks = load_blocks(data_dir / "parsed" / f"{stem}.json")
    description = (
        f"{entry['company']} {entry['form']} for the period ending {entry['period_of_report']}"
    )
    extraction = extract_filing(completer, build_context(blocks), description)
    write_atomically(extracted_path, extraction.model_dump_json(indent=2))
    return extraction


def score_one(data_dir: Path, entry: dict, stem: str, extraction: FilingExtraction):
    facts = json.loads((data_dir / "xbrl" / f"{entry['ticker']}.json").read_text("utf-8"))
    references = {
        field: reference_value(
            facts, field, entry["form"], entry["period_of_report"], entry["accession_number"]
        )
        for field in SCORED_FIELDS
    }
    return score_document(stem, extraction, references)


def extract_corpus(completer: Completer, data_dir: Path) -> CorpusRun:
    """Extract each parsed filing, then write `RESULTS.md`.

    Results already on disk are kept, so a rerun after a failure only repeats the failed filings.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    (data_dir / "extracted").mkdir(exist_ok=True)

    scores: list[FieldScore] = []
    failures: list[str] = []
    for entry in manifest:
        stem = Path(entry["path"]).stem
        try:
            extraction = extract_one(completer, data_dir, entry, stem)
            scores += score_one(data_dir, entry, stem, extraction)
        except Exception as error:  # noqa: BLE001 - one bad filing must not stop the batch
            failures.append(f"{stem}: {type(error).__name__}: {_first_line(error)}")

    (data_dir / "RESULTS.md").write_text(render_results(scores, failures), encoding="utf-8")
    return CorpusRun(scores, failures)


def _first_line(error: Exception) -> str:
    lines = str(error).splitlines()
    return lines[0] if lines else ""
