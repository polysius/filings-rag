"""Parse every filing in the manifest and write a parsing report."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filings_rag.parser import ParsedDocument, parse_filing

# Filings are long, so a document that parses to one page almost certainly lost its page breaks.
LONG_DOCUMENT_CHARACTERS = 20_000


@dataclass
class ParseResult:
    path: str
    status: str  # "ok" or "failed"
    pages: int = 0
    headings: int = 0
    text_blocks: int = 0
    tables: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def parsed_to_json(document: ParsedDocument) -> dict:
    return {"page_count": document.page_count, "blocks": [asdict(b) for b in document.blocks]}


def check_document(document: ParsedDocument) -> list[str]:
    """Return warnings for parses that succeeded but look incomplete."""
    warnings = []
    characters = sum(len(b.text) for b in document.blocks)
    if document.page_count == 1 and characters > LONG_DOCUMENT_CHARACTERS:
        warnings.append("no page breaks found, so every block is on page 1")
    if not any(b.section for b in document.blocks):
        warnings.append("no Item or Part headings found")
    if not document.of_kind("table"):
        warnings.append("no tables found")
    return warnings


def parse_one(raw_path: Path, parsed_path: Path, name: str) -> ParseResult:
    # Remove output from an earlier run so a filing that now fails leaves nothing stale behind.
    parsed_path.unlink(missing_ok=True)
    try:
        document = parse_filing(raw_path.read_bytes())
    except Exception as error:  # noqa: BLE001 - one bad filing must not stop the batch
        return ParseResult(name, "failed", error=f"{type(error).__name__}: {error}")

    if not document.blocks:
        return ParseResult(name, "failed", error="no content extracted")

    parsed_path.write_text(json.dumps(parsed_to_json(document)), encoding="utf-8")
    return ParseResult(
        name,
        "ok",
        pages=document.page_count,
        headings=len(document.of_kind("heading")),
        text_blocks=len(document.of_kind("text")),
        tables=len(document.of_kind("table")),
        warnings=check_document(document),
    )


def parse_corpus(data_dir: Path) -> list[ParseResult]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    parsed_dir = data_dir / "parsed"
    parsed_dir.mkdir(exist_ok=True)

    # Drop output for filings that are no longer in the manifest.
    expected = {f"{Path(entry['path']).stem}.json" for entry in manifest}
    for stale in parsed_dir.glob("*.json"):
        if stale.name not in expected:
            stale.unlink()

    results = []
    for entry in manifest:
        raw_path = data_dir / entry["path"]
        parsed_path = parsed_dir / f"{raw_path.stem}.json"
        results.append(parse_one(raw_path, parsed_path, entry["path"]))

    (data_dir / "parsing_report.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    (data_dir / "PARSING_REPORT.md").write_text(render_report(results), encoding="utf-8")
    return results


def render_report(results: list[ParseResult]) -> str:
    failed = [r for r in results if r.status == "failed"]
    lines = [
        "# Parsing report",
        "",
        f"Documents: {len(results)}. Parsed: {len(results) - len(failed)}. Failed: {len(failed)}.",
        "",
        "| Document | Pages | Headings | Text blocks | Tables | Warnings |",
        "|----------|-------|----------|-------------|--------|----------|",
    ]
    for r in results:
        if r.status == "ok":
            warnings = "; ".join(r.warnings) or "none"
            lines.append(
                f"| {r.path} | {r.pages} | {r.headings} | {r.text_blocks} | {r.tables} "
                f"| {warnings} |"
            )

    lines += ["", "## Failures", ""]
    if failed:
        lines += [f"- {r.path}: {r.error}" for r in failed]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"
