# filings-rag

Document intelligence and citation-grounded question answering over SEC 10-K and 10-Q filings.
The pipeline downloads filings from EDGAR, parses narrative text and tables, extracts structured
financial fields, and validates them against XBRL ground truth. Answers cite document and page.

## Status

| Milestone | State |
|-----------|-------|
| M1 Corpus (EDGAR download, XBRL facts, manifest) | Implemented |
| M2 Parsing | Planned |
| M3 Structured extraction | Planned |
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
