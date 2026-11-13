"""Pick the parts of a parsed filing that the extraction prompt needs.

A full filing is far too long to send, so we keep tables that mention a target statement line,
plus the start of the risk factor section. Every piece is tagged with its page, and the model
copies that page into its answer.
"""

import re

from filings_rag.parser import Block

TABLE_KEYWORDS = re.compile(
    r"total revenues?|net sales|total net sales|revenues?\b|net income|net earnings"
    r"|total assets|segment",
    re.IGNORECASE,
)
RISK_SECTION_RE = re.compile(r"^item\s+1a\b", re.IGNORECASE)

MAX_TABLES = 25
MAX_RISK_CHARACTERS = 6_000


def table_to_text(block: Block) -> str:
    return "\n".join(" | ".join(cell for cell in row if cell) for row in block.rows)


def build_context(blocks: list[Block]) -> str:
    """Return prompt text made of risk factor excerpts and keyword-matching tables."""
    parts = [f"[page {b.page}] (risk factors)\n{b.text}" for b in _risk_blocks(blocks)]
    tables = [b for b in blocks if b.kind == "table" and TABLE_KEYWORDS.search(table_to_text(b))]
    parts += [f"[page {b.page}] (table)\n{table_to_text(b)}" for b in tables[:MAX_TABLES]]
    return "\n\n".join(parts)


def _risk_blocks(blocks: list[Block]) -> list[Block]:
    """Text blocks under Item 1A, up to MAX_RISK_CHARACTERS in total."""
    selected, used = [], 0
    for block in blocks:
        in_risk_section = bool(block.section and RISK_SECTION_RE.match(block.section))
        if block.kind != "text" or not in_risk_section:
            continue
        if used >= MAX_RISK_CHARACTERS:
            break
        selected.append(block)
        used += len(block.text)
    return selected
