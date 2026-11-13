import pytest

from filings_rag.context import MAX_RISK_CHARACTERS, build_context
from filings_rag.extractor import ExtractionError, extract_filing
from filings_rag.parser import Block

VALID_REPLY = '{"revenue": {"value": 1000000, "page": 3}, "top_risk_factors": ["Competition"]}'


class FakeCompleter:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.user_prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.user_prompts.append(user)
        return self.replies[len(self.user_prompts) - 1]


def test_valid_reply_is_parsed():
    extraction = extract_filing(FakeCompleter([VALID_REPLY]), "context")

    assert extraction.revenue.value == 1_000_000
    assert extraction.revenue.page == 3
    assert extraction.net_income is None


def test_code_fenced_reply_is_accepted():
    extraction = extract_filing(FakeCompleter([f"```json\n{VALID_REPLY}\n```"]), "context")

    assert extraction.revenue.page == 3


def test_invalid_reply_is_retried_with_the_error():
    completer = FakeCompleter(['{"revenue": {"value": 5, "page": 0}}', VALID_REPLY])

    extraction = extract_filing(completer, "context")

    assert extraction.revenue.value == 1_000_000
    assert "previous answer was invalid" in completer.user_prompts[1]


def test_gives_up_after_max_attempts():
    completer = FakeCompleter(["not json"] * 3)

    with pytest.raises(ExtractionError):
        extract_filing(completer, "context", max_attempts=3)

    assert len(completer.user_prompts) == 3


def test_too_many_risk_factors_fails_validation():
    reply = '{"top_risk_factors": ["a", "b", "c", "d", "e", "f"]}'

    with pytest.raises(ExtractionError):
        extract_filing(FakeCompleter([reply]), "context", max_attempts=1)


def test_context_keeps_keyword_tables_and_risk_text_with_pages():
    blocks = [
        Block("text", 2, "Item 1A. Risk Factors", text="We face competition."),
        Block("text", 5, "Item 7. MD&A", text="Unrelated discussion."),
        Block("table", 7, "Item 8", rows=[["Total revenues", "$", "1,000"]]),
        Block("table", 8, "Item 8", rows=[["Employees", "10"]]),
    ]

    context = build_context(blocks)

    assert "[page 2] (risk factors)\nWe face competition." in context
    assert "[page 7] (table)\nTotal revenues | $ | 1,000" in context
    assert "Unrelated" not in context
    assert "Employees" not in context


def test_risk_text_is_cut_at_the_limit_even_for_one_long_block():
    blocks = [Block("text", 2, "Item 1A. Risk Factors", text="x" * 50_000)]

    context = build_context(blocks)

    assert len(context) < MAX_RISK_CHARACTERS + 100


def test_risk_headings_are_included():
    blocks = [Block("heading", 2, "Item 1A. Risk Factors", text="We face intense competition.")]

    assert "We face intense competition." in build_context(blocks)


def test_table_keeps_empty_cells_so_columns_line_up():
    table = Block("table", 4, "Item 8", rows=[["", "2024", "2023"], ["Total net sales", "", "5"]])

    context = build_context([table])

    # The empty top-left cell stays, so each header sits above its own value column.
    assert " | 2024 | 2023" in context
    assert "Total net sales |  | 5" in context


def test_table_context_includes_the_units_line_above_it():
    blocks = [
        Block("text", 4, "Item 8", text="(In millions, except per-share amounts)"),
        Block("table", 4, "Item 8", rows=[["Total net sales", "391,035"]]),
    ]

    assert "(In millions" in build_context(blocks)


def test_description_is_part_of_the_prompt():
    completer = FakeCompleter([VALID_REPLY])

    extract_filing(completer, "context", "Acme 10-Q ending 2024-09-30")

    assert "Acme 10-Q ending 2024-09-30" in completer.user_prompts[0]
