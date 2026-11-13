"""Score extracted values against XBRL and render the field-level accuracy table."""

from dataclasses import dataclass

from filings_rag.schema import SCORED_FIELDS, FilingExtraction

# Filings round to thousands or millions, and XBRL carries the same rounded figure, so a correct
# value matches almost exactly. A small tolerance absorbs float noise from unit conversion.
RELATIVE_TOLERANCE = 0.001

CORRECT, WRONG, MISSING = "correct", "wrong", "missing"

INTRODUCTION = (
    "Extracted values compared with XBRL company facts. A value is correct when it is within "
    f"{RELATIVE_TOLERANCE:.1%} of the XBRL value. Missing means the model returned null. "
    "Documents with no XBRL value for a field are left out of that field's count. Segments "
    "and risk factors have no XBRL ground truth and are not scored."
)


@dataclass
class FieldScore:
    document: str
    field: str
    outcome: str  # CORRECT, WRONG or MISSING
    extracted: float | None
    reference: float


def score_field(extracted: float | None, reference: float) -> str:
    if extracted is None:
        return MISSING
    if abs(extracted - reference) <= RELATIVE_TOLERANCE * abs(reference):
        return CORRECT
    return WRONG


def score_document(
    document: str, extraction: FilingExtraction, references: dict[str, float | None]
) -> list[FieldScore]:
    """Score each field that has an XBRL value. Fields without one are left out of accuracy."""
    scores = []
    for field in SCORED_FIELDS:
        reference = references.get(field)
        if reference is None:
            continue
        amount = getattr(extraction, field)
        extracted = amount.value if amount else None
        scores.append(
            FieldScore(document, field, score_field(extracted, reference), extracted, reference)
        )
    return scores


def render_results(scores: list[FieldScore], failures: list[str]) -> str:
    lines = [
        "# Extraction results",
        "",
        INTRODUCTION,
        "",
        "| Field | Documents | Correct | Wrong | Missing | Accuracy |",
        "|-------|-----------|---------|-------|---------|----------|",
    ]
    for field in SCORED_FIELDS:
        field_scores = [s for s in scores if s.field == field]
        counts = {o: sum(s.outcome == o for s in field_scores) for o in (CORRECT, WRONG, MISSING)}
        total = len(field_scores)
        accuracy = f"{counts[CORRECT] / total:.1%}" if total else "n/a"
        lines.append(
            f"| {field} | {total} | {counts[CORRECT]} | {counts[WRONG]} | {counts[MISSING]} "
            f"| {accuracy} |"
        )

    lines += ["", "## Wrong values", ""]
    wrong = [s for s in scores if s.outcome == WRONG]
    if wrong:
        lines += [f"- {s.document} {s.field}: got {s.extracted:,.0f}, XBRL {s.reference:,.0f}"
                  for s in wrong]  # fmt: skip
    else:
        lines.append("None.")

    lines += ["", "## Extraction failures", ""]
    lines += [f"- {name}" for name in failures] or ["None."]
    return "\n".join(lines) + "\n"
