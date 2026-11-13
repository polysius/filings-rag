import pytest

from filings_rag.context import build_context
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
