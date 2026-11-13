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
    matched = 0
    for index, block in enumerate(blocks):
        if block.kind == "table" and TABLE_KEYWORDS.search(table_to_text(block)):
            parts.append(_table_with_caption(blocks, index))
            matched += 1
            if matched == MAX_TABLES:
                break
    return "\n\n".join(parts)


def _table_with_caption(blocks: list[Block], index: int) -> str:
    """A table with the short blocks just above it on the same page.

    These usually hold the statement title and the "(in millions)" line that sets the scale.
    """
    table = blocks[index]
    captions = []
    for above in reversed(blocks[max(0, index - CAPTION_BLOCKS) : index]):
        if above.kind == "table" or above.page != table.page or len(above.text) > MAX_CAPTION:
            break
        captions.insert(0, above.text)
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
