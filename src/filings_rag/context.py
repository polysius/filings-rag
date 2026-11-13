"""Pick the parts of a parsed filing that the extraction prompt needs.

A full filing is far too long to send, so we keep tables that mention a target statement line,
plus the start of the risk factor section. Every piece is tagged with its page, and the model
copies that page into its answer.
"""

import re
from dataclasses import replace

from filings_rag.parser import Block

TABLE_KEYWORDS = re.compile(
    r"total revenues?|net sales|total net sales|revenues?\b|net income|net earnings"
    r"|total assets|segment",
    re.IGNORECASE,
)
STATEMENT_TITLE_RE = re.compile(
    r"statements? of (?:consolidated )?(?:operations|income|earnings|financial (?:position|condition))"
    r"|balance sheets?",
    re.IGNORECASE,
)
RISK_SECTION_RE = re.compile(r"^item\s+1a\b", re.IGNORECASE)

MAX_TABLES = 25
MAX_RISK_CHARACTERS = 6_000
CAPTION_BLOCKS = 3  # blocks above a table that may hold its title and units
MAX_CAPTION = 200


def table_to_text(block: Block) -> str:
    """Join cells with " | " and keep empty cells, so values stay under their column headers."""
    return "\n".join(" | ".join(row).rstrip(" |") for row in block.rows)


def build_context(blocks: list[Block]) -> str:
    """Return prompt text made of risk factor excerpts and keyword-matching tables."""
    parts = [f"[page {b.page}] (risk factors)\n{b.text}" for b in _risk_blocks(blocks)]
    parts += [_render_table(blocks[index], captions) for index, captions in _chosen_tables(blocks)]
    return "\n\n".join(parts)


def _chosen_tables(blocks: list[Block]) -> list[tuple[int, list[str]]]:
    """Up to MAX_TABLES keyword-matching tables, financial statements first.

    MD&A and note tables also mention revenue and segments and come before the statements in
    Item 8, so taking the first matches in document order could cut the statements. The chosen
    tables are returned in document order.
    """
    candidates = []
    for index, block in enumerate(blocks):
        text = table_to_text(block) if block.kind == "table" else ""
        if TABLE_KEYWORDS.search(text):
            captions = _captions_above(blocks, index)
            candidates.append((not _is_statement(captions, text), index, captions))
    chosen = sorted(candidates)[:MAX_TABLES]  # False sorts first, then document order
    return [(index, captions) for _, index, captions in sorted(chosen, key=lambda c: c[1])]


def _is_statement(captions: list[str], text: str) -> bool:
    """A statement has a statement title above it, or balance sheet or income statement rows."""
    if STATEMENT_TITLE_RE.search(" ".join(captions)):
        return True
    has_balance_sheet_row = bool(re.search(r"total assets", text, re.IGNORECASE))
    has_income_rows = bool(re.search(r"net income|net earnings", text, re.IGNORECASE)) and bool(
        re.search(r"total revenues?|net sales|revenues?\b", text, re.IGNORECASE)
    )
    return has_balance_sheet_row or has_income_rows


def _captions_above(blocks: list[Block], index: int) -> list[str]:
    """The short blocks just above a table on the same page.

    These usually hold the statement title and the "(in millions)" line that sets the scale.
    """
    table = blocks[index]
    captions = []
    for above in reversed(blocks[max(0, index - CAPTION_BLOCKS) : index]):
        if above.kind == "table" or above.page != table.page or len(above.text) > MAX_CAPTION:
            break
        captions.insert(0, above.text)
    return captions


def _render_table(table: Block, captions: list[str]) -> str:
    return "\n".join([f"[page {table.page}] (table)", *captions, table_to_text(table)])


def _risk_blocks(blocks: list[Block]) -> list[Block]:
    """Headings and text under Item 1A, cut so the total stays within MAX_RISK_CHARACTERS."""
    selected, remaining = [], MAX_RISK_CHARACTERS
    for block in blocks:
        in_risk_section = bool(block.section and RISK_SECTION_RE.match(block.section))
        if block.kind == "table" or not in_risk_section:
            continue
        if remaining <= 0:
            break
        selected.append(replace(block, text=block.text[:remaining]))
        remaining -= len(block.text)
    return selected
