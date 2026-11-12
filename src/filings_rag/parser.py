"""Layout-aware parser for EDGAR HTML filings.

A filing is read in document order and split into three kinds of blocks: section headings,
narrative text and tables. Every block records the page it appears on, so later stages can cite
a page.

Pages are physical pages, counted from 1 by the page breaks the filer put in the HTML
(`page-break-after: always` and similar). They can differ from the page numbers printed in the
filing, which restart or skip around the cover and exhibits. A page with no text, table or image
is not counted.
"""

import re
import warnings
from dataclasses import dataclass, field

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
from bs4.element import PreformattedString

BLOCK_TAGS = {
    "html", "body", "div", "p", "li", "ul", "ol", "hr", "center", "blockquote", "section",
    "article", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
}  # fmt: skip
# Tags that separate words when their text is flattened, so "a<br>b" reads "a b".
SPACING_TAGS = BLOCK_TAGS | {"br", "table", "tr", "td", "th"}
SKIPPED_TAGS = {"head", "script", "style", "ix:header"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

MAX_HEADING_LENGTH = 100
MAX_SECTION_HEADING_LENGTH = 120
MAX_COLSPAN = 100

PAGE_BREAK_RE = re.compile(
    r"(page-break|break)-(?P<where>before|after)\s*:\s*(always|page)", re.IGNORECASE
)
BOLD_STYLE_RE = re.compile(r"font-weight\s*:\s*(bold|[6-9]00)", re.IGNORECASE)
HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)

# "Item 1A.", "Items 1 and 2 -", "PART II" and similar, followed by punctuation or the end of the
# line. A sentence such as "Part I, Item 2 of this report" does not match.
SECTION_HEADING_RE = re.compile(
    r"^(?:part\s+[ivx]+|items?\s+\d+[a-c]?(?:\s*(?:and|&|,)\s*\d+[a-c]?)*)\s*(?:[.:—–-]|$)",
    re.IGNORECASE,
)
# "Item 7 Management's Discussion" without punctuation counts only when it is also bold.
LOOSE_SECTION_RE = re.compile(r"^(?:part\s+[ivx]+|items?\s+\d+[a-c]?)\b", re.IGNORECASE)

# Page furniture: "12", "F-12", "- 2 -", "ii" and "Company | 2023 Form 10-K | 17".
PAGE_FURNITURE_RE = re.compile(
    r"^(?:[-–]\s*)?(?:[A-Z]-)?\d{1,3}(?:\s*[-–])?$"
    r"|^[ivxl]{1,6}$"
    r"|^.{0,60}\|.{0,40}form 10-[kq].{0,20}\|\s*\d{1,3}$",
    re.IGNORECASE,
)


@dataclass
class Block:
    kind: str  # "heading", "text" or "table"
    page: int
    section: str | None  # most recent "Item ..." or "Part ..." heading above this block
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)  # only for tables

    def to_dataframe(self) -> pd.DataFrame:
        """Return a table block as a dataframe (label column first, then value columns)."""
        if self.kind != "table":
            raise ValueError("Only table blocks have rows")
        return pd.DataFrame(self.rows)


@dataclass
class ParsedDocument:
    blocks: list[Block]
    page_count: int

    def of_kind(self, kind: str) -> list[Block]:
        return [block for block in self.blocks if block.kind == kind]


def parse_filing(content: bytes | str) -> ParsedDocument:
    with warnings.catch_warnings():
        # Inline XBRL files start with an XML declaration but are parsed as HTML on purpose.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(decode_html(content), "lxml")
    walker = _Walker()
    walker.walk(soup.body or soup)
    return ParsedDocument(blocks=walker.blocks, page_count=walker.page)


def decode_html(content: bytes | str) -> str:
    """Decode a filing ourselves, because BeautifulSoup's guess can silently cut a file short.

    Filings are UTF-8 or Windows-1252. A UTF-8 file with a stray bad byte stays UTF-8, and a
    file with no valid multi-byte sequences is read as Windows-1252.
    """
    if isinstance(content, str):
        return content
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    as_utf8 = content.decode("utf-8", errors="replace")
    bad_bytes = as_utf8.count("�")
    good_multibyte_chars = sum(1 for char in as_utf8 if char > "\x7f" and char != "�")
    if good_multibyte_chars >= bad_bytes:
        return as_utf8
    return content.decode("cp1252", errors="replace")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def clean_table_rows(grid: list[list[str]]) -> list[list[str]]:
    """Tidy a grid of table cells while keeping every value in its column.

    EDGAR tables put "$", ")" and "%" in cells of their own. Each symbol is merged into its
    number, so "$", "1,234", ")" becomes "$1,234)", placed in the column where the symbol
    started because that is where the column header sits. Rows and columns that are empty in
    the whole table are then dropped.
    """
    rows = [_merge_symbol_cells([normalize_text(cell) for cell in row]) for row in grid]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    used_columns = [i for i in range(width) if any(row[i] for row in rows)]
    return [[row[i] for i in used_columns] for row in rows]


def _merge_symbol_cells(cells: list[str]) -> list[str]:
    for i, cell in enumerate(cells):
        if cell == "$":
            following = next((j for j in range(i + 1, len(cells)) if cells[j]), None)
            if following is not None:
                cells[i], cells[following] = "$" + cells[following], ""
        elif cell in {")", "%", ")%"}:
            preceding = next((j for j in range(i - 1, -1, -1) if cells[j]), None)
            if preceding is not None:
                cells[preceding] += cell
                cells[i] = ""
    return cells


def _style(element: Tag) -> str:
    return element.get("style") or ""


def _is_bold(element: Tag) -> bool:
    return (
        element.name in {"b", "strong"} | HEADING_TAGS
        or BOLD_STYLE_RE.search(_style(element)) is not None
    )


def _is_hidden(element: Tag) -> bool:
    return element.name in SKIPPED_TAGS or HIDDEN_STYLE_RE.search(_style(element)) is not None


def _has_page_break(element: Tag, where: str) -> bool:
    return any(m.group("where") == where for m in PAGE_BREAK_RE.finditer(_style(element)))


def _needs_visit(element: Tag) -> bool:
    """True for elements the walker must descend into instead of treating as inline text."""
    return (
        element.name in BLOCK_TAGS
        or element.name == "table"
        or element.find([*BLOCK_TAGS, "table"]) is not None
        or PAGE_BREAK_RE.search(_style(element)) is not None
        or element.find(style=PAGE_BREAK_RE) is not None
    )


def _inline_parts(element: Tag, bold: bool) -> list[tuple[str, bool]]:
    """Flatten an element into (text, is_bold) pieces, skipping hidden text."""
    parts: list[tuple[str, bool]] = []
    for child in element.children:
        if isinstance(child, PreformattedString):  # comments, doctype, CDATA
            continue
        if isinstance(child, NavigableString):
            parts.append((str(child), bold))
        elif not _is_hidden(child):
            separator = [(" ", bold)] if child.name in SPACING_TAGS else []
            parts += separator + _inline_parts(child, bold or _is_bold(child)) + separator
    return parts


def _flat_text(element: Tag) -> str:
    return normalize_text("".join(text for text, _ in _inline_parts(element, False)))


def _grid_row(row: Tag) -> list[str]:
    """Cells of one table row, with empty slots after a cell that spans several columns."""
    cells: list[str] = []
    for cell in row.find_all(["td", "th"], recursive=False):
        try:
            span = min(max(int(cell.get("colspan") or 1), 1), MAX_COLSPAN)
        except ValueError:
            span = 1
        cells += [_flat_text(cell)] + [""] * (span - 1)
    return cells


class _Walker:
    """Walks the HTML tree once, collecting inline text until a block boundary is reached."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.page = 1
        self._section: str | None = None
        self._page_has_content = False
        self._break_pending = False
        self._pending: list[tuple[str, bool]] = []  # (text, is_bold) not yet emitted
        self._bold_depth = 0

    def walk(self, root: Tag) -> None:
        self._visit(root)
        self._flush()

    def _visit(self, element: Tag) -> None:
        if _has_page_break(element, "before"):
            self._start_new_page()
        if element.name in BLOCK_TAGS:
            self._flush()

        is_bold = _is_bold(element)
        self._bold_depth += is_bold
        if element.name == "table":
            self._flush()
            self._emit_table(element)
        else:
            for child in element.children:
                self._visit_child(child)
            if element.name in BLOCK_TAGS:
                self._flush()
        self._bold_depth -= is_bold

        if _has_page_break(element, "after"):
            self._start_new_page()

    def _visit_child(self, child) -> None:
        if isinstance(child, PreformattedString):
            return
        if isinstance(child, NavigableString):
            self._pending.append((str(child), self._bold_depth > 0))
            return
        if _is_hidden(child):
            return

        if child.name == "img" or child.find("img") is not None:
            self._mark_page_has_content()
        if _needs_visit(child):
            self._visit(child)
        else:
            is_bold = self._bold_depth > 0 or _is_bold(child)
            self._pending.append((" ", is_bold))
            self._pending += _inline_parts(child, is_bold)

    def _start_new_page(self) -> None:
        # The page number only advances when content arrives, so a break at the end of the
        # document or two breaks in a row do not create empty pages.
        self._flush()
        self._break_pending = True

    def _mark_page_has_content(self) -> None:
        self._current_page()
        self._page_has_content = True

    def _current_page(self) -> int:
        if self._break_pending:
            self._break_pending = False
            if self._page_has_content:
                self.page += 1
                self._page_has_content = False
        return self.page

    def _flush(self) -> None:
        text = normalize_text("".join(part for part, _ in self._pending))
        all_bold = all(bold for part, bold in self._pending if part.strip())
        self._pending = []
        if not text or PAGE_FURNITURE_RE.fullmatch(text):
            return

        is_heading = self._is_heading(text, all_bold)
        if is_heading and LOOSE_SECTION_RE.match(text):
            self._section = text
        self._add_block("heading" if is_heading else "text", text=text)

    @staticmethod
    def _is_heading(text: str, all_bold: bool) -> bool:
        if SECTION_HEADING_RE.match(text) and len(text) <= MAX_SECTION_HEADING_LENGTH:
            return True
        return all_bold and len(text) <= MAX_HEADING_LENGTH

    def _emit_table(self, table: Tag) -> None:
        caption = table.find("caption")
        if caption is not None:
            self._pending += _inline_parts(caption, self._bold_depth > 0)
            self._flush()

        rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
        if len(rows) == 1:
            self._emit_single_row_table(rows[0])
        elif rows:
            cleaned = clean_table_rows([_grid_row(row) for row in rows])
            if cleaned:
                self._add_block("table", rows=cleaned)

    def _emit_single_row_table(self, row: Tag) -> None:
        """A one-row table is page layout (a heading split in two cells, a bullet and its text).

        Its content is read as ordinary document content instead of a data table.
        """
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) == 1:
            self._visit(cells[0])
        else:
            for cell in cells:
                self._pending.append((" ", self._bold_depth > 0))
                self._pending += _inline_parts(cell, self._bold_depth > 0 or _is_bold(cell))
        self._flush()

    def _add_block(self, kind: str, text: str = "", rows: list[list[str]] | None = None) -> None:
        self.blocks.append(Block(kind, self._current_page(), self._section, text, rows or []))
        self._page_has_content = True
