"""Split parsed filings into chunks for retrieval.

Two strategies share one size target, so a comparison between them is fair:

- `fixed`: walks the blocks in document order and cuts a chunk whenever the target size is
  reached. It ignores sections and treats tables like any other text.
- `section`: never lets a chunk cross an Item or Part boundary, and keeps each table in chunks of
  its own.

A chunk remembers the pages it spans, so an answer built from it can cite a page.
"""

from collections.abc import Callable
from dataclasses import dataclass

from filings_rag.context import table_to_text
from filings_rag.parser import Block

TARGET_CHARACTERS = 1_500


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str  # filing name, for example "AAPL_10-K_2024-09-28"
    section: str | None
    first_page: int
    last_page: int
    text: str

    def spans_page(self, page: int) -> bool:
        return self.first_page <= page <= self.last_page


@dataclass
class Piece:
    """A block, or a part of an oversized block, ready to be packed into a chunk."""

    text: str
    page: int
    section: str | None
    is_table: bool


def block_text(block: Block) -> str:
    return table_to_text(block) if block.kind == "table" else block.text


def chunk_fixed(document: str, blocks: list[Block], target: int = TARGET_CHARACTERS) -> list[Chunk]:
    pieces = _pieces(blocks, target)
    return _pack(document, pieces, target, starts_new_chunk=lambda previous, piece: False)


def chunk_by_section(
    document: str, blocks: list[Block], target: int = TARGET_CHARACTERS
) -> list[Chunk]:
    def starts_new_chunk(previous: Piece, piece: Piece) -> bool:
        return (
            piece.section != previous.section or piece.is_table or previous.is_table
        )  # a table is never joined to the text next to it

    return _pack(document, _pieces(blocks, target), target, starts_new_chunk)


STRATEGIES = {"fixed": chunk_fixed, "section": chunk_by_section}


def _pieces(blocks: list[Block], target: int) -> list[Piece]:
    pieces = []
    for block in blocks:
        text = block_text(block).strip()
        for part in _split(text, target):
            pieces.append(Piece(part, block.page, block.section, block.kind == "table"))
    return pieces


def _split(text: str, target: int) -> list[str]:
    """Cut text longer than `target` at a line break, else a space, so words stay whole.

    A line break comes first so that a long table is cut between rows and not inside one.
    """
    parts = []
    while len(text) > target:
        window = text[:target]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = target
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return [*parts, text] if text else parts


def _pack(
    document: str,
    pieces: list[Piece],
    target: int,
    starts_new_chunk: Callable[[Piece, Piece], bool],
) -> list[Chunk]:
    groups: list[list[Piece]] = []
    size = 0
    for piece in pieces:
        too_big = size + len(piece.text) > target
        if not groups or too_big or starts_new_chunk(groups[-1][-1], piece):
            groups.append([])
            size = 0
        groups[-1].append(piece)
        size += len(piece.text) + 1  # the joining newline

    return [
        Chunk(
            chunk_id=f"{document}#{number}",
            document=document,
            section=group[0].section,
            first_page=group[0].page,
            last_page=group[-1].page,
            text="\n".join(piece.text for piece in group),
        )
        for number, group in enumerate(groups)
    ]
