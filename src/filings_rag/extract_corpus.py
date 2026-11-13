"""Run extraction over every parsed filing and score it against XBRL."""

import json
from pathlib import Path

from filings_rag.context import build_context
from filings_rag.evaluate import FieldScore, render_results, score_document
from filings_rag.extractor import Completer, ExtractionError, extract_filing
from filings_rag.parser import Block
from filings_rag.schema import SCORED_FIELDS, FilingExtraction
from filings_rag.xbrl import reference_value


def load_blocks(parsed_path: Path) -> list[Block]:
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    return [Block(**block) for block in parsed["blocks"]]


def extract_corpus(completer: Completer, data_dir: Path) -> list[FieldScore]:
    """Extract each parsed filing, then write `RESULTS.md`.

    Results already on disk are kept, so a rerun after a failure only repeats the failed filings.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    extracted_dir = data_dir / "extracted"
    extracted_dir.mkdir(exist_ok=True)

    scores: list[FieldScore] = []
    failures: list[str] = []
    for entry in manifest:
        stem = Path(entry["path"]).stem
        parsed_path = data_dir / "parsed" / f"{stem}.json"
        extracted_path = extracted_dir / f"{stem}.json"

        if not extracted_path.exists():
            if not parsed_path.exists():
                failures.append(f"{stem}: not parsed")
                continue
            try:
                context = build_context(load_blocks(parsed_path))
                extraction = extract_filing(completer, context)
            except ExtractionError as error:
                failures.append(f"{stem}: {error}")
                continue
            extracted_path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")

        extraction = FilingExtraction.model_validate_json(extracted_path.read_text("utf-8"))
        facts = json.loads((data_dir / "xbrl" / f"{entry['ticker']}.json").read_text("utf-8"))
        references = {
            field: reference_value(facts, field, entry["form"], entry["period_of_report"])
            for field in SCORED_FIELDS
        }
        scores += score_document(stem, extraction, references)

    (data_dir / "RESULTS.md").write_text(render_results(scores, failures), encoding="utf-8")
    return scores
