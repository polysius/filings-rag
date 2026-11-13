# filings-rag

Document intelligence and citation-grounded question answering over SEC 10-K and 10-Q filings.
The pipeline downloads filings from EDGAR, parses narrative text and tables, extracts structured
financial fields, and validates them against XBRL ground truth. Answers cite document and page.

## Status

| Milestone | State |
|-----------|-------|
| M1 Corpus (EDGAR download, XBRL facts, manifest) | Implemented |
| M2 Parsing (headings, text, tables, page numbers, report) | Implemented, not yet run on real filings |
| M3 Structured extraction (Claude, schema validation, XBRL scoring) | Implemented, not yet run on real filings |
| M4 Retrieval and QA | Planned |
| M5 API | Planned |

## Pilot corpus

Twelve companies across four sectors, listed in `configs/companies.yaml`: technology (AAPL, MSFT,
NVDA), banks (JPM, BAC, GS), healthcare (JNJ, PFE, UNH) and energy (XOM, CVX, COP). Each company
contributes its two latest 10-Ks and three latest 10-Qs, plus its XBRL company facts.

## Usage

```
make install
export SEC_USER_AGENT="Your Name your.email@example.com"
make corpus
make test
```

The SEC requires a contact User-Agent and asks clients to stay under 10 requests per second. The
client sends the User-Agent from `SEC_USER_AGENT` and throttles itself to about 6 per second.

`make corpus` writes `data/manifest.json` with company, form, period, accession number, source URL
and SHA-256 per filing. Downloaded files are skipped on rerun, so an interrupted build resumes.

## Parsing

`make parse` reads every filing in the manifest and writes `data/parsed/<filing>.json`, a list of
blocks in document order. Each block is a heading, a narrative text paragraph or a table, and
carries its page number and the Item or Part section it sits under. Tables are stored as rows of
cleaned cells and convert to a dataframe with `Block.to_dataframe()`.

`data/PARSING_REPORT.md` lists page, heading, text and table counts per document, warnings for
parses that look incomplete (no page breaks, no Item headings, no tables), and every failure.

Known limits:

- Pages are physical pages counted from the filer's HTML page breaks. They can differ from the
  page numbers printed in the filing. Pages with no text, table or image are not counted.
- Headings are detected from Item and Part patterns and from short all-bold lines, so a bold
  caption can be mistaken for a heading.
- Table columns follow the HTML column spans. A table whose header and values use different
  spans can still misalign, so check the extracted tables before trusting a column.
- A nested table is flattened into the text of its parent cell. A table with one row is treated
  as page layout and read as text.
- Page breaks inside a table are ignored, so a break between table rows does not start a new
  page. A layout table with several rows is read as a data table.
- A document nested about 500 levels deep fails with a recursion error and is listed as failed in
  the report.
- Bare page numbers and common footers are dropped. An unusual footer can remain as text.
- Filings that are not HTML are not supported.

## Extraction

`make extract` needs `ANTHROPIC_API_KEY`. For each parsed filing it sends the tables that mention
revenue, net income, total assets or segments, plus the start of Item 1A, to Claude. Every excerpt
is tagged `[page N]`. The reply must be JSON that passes the `FilingExtraction` schema in
`src/filings_rag/schema.py`: revenue, net income, total assets, segments and up to five risk
factor titles, with a page for each amount. An invalid reply is sent back with the validation
error, up to three attempts. Results are saved in `data/extracted/` and a rerun skips them.

Revenue, net income and total assets are then compared with XBRL company facts and the accuracy
per field is written to `data/RESULTS.md`. The lookup tries several revenue concepts, because
companies tag revenue differently, and matches 10-Q flow values on period length so the quarter
is not confused with the year to date. Segments and risk factors have no XBRL value and are not
scored.

Known limits:

- Only the first 6,000 characters of Item 1A are sent, so the risk factors reflect the start of
  the section rather than a ranking of the whole section.
- Context is picked by keyword. A statement table that never uses these words is not sent, and
  only the first 25 matching tables in document order are sent.
- The reference value is the one the filing itself reported, matched by accession number. A later
  filing that recasts the same period is not used.
- Filings whose extraction failed are listed under failures and are not counted in the accuracy.
- Banks tag revenue in several ways, so a bank can show as missing when no tried concept exists.
