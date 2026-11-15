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
| M4 Retrieval and QA: chunking, BM25, retrieval evaluation | Implemented, not yet run on real filings |
| M4 Retrieval and QA: dense search, hybrid fusion, reranker, pgvector | Implemented, not yet run on real filings |
| M4 Retrieval and QA: cited answers and their evaluation | Planned |
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

`make extract` uses the local Claude Code bridge, so it needs no API key. Start the bridge in
another terminal with `make bridge`, or `python scripts/claude_cli_bridge.py` where `make` is not
installed. It runs the `claude` CLI that is signed in on this machine, so calls count against that
Claude Code plan, and it listens on 127.0.0.1 only. Extraction asks it for the `opus` model alias.
Set `LLM_MODEL` (for example `sonnet`) to change that, or `LLM_BASE_URL`, which must end in `/v1`,
to reach a bridge on another address. If the bridge is not running, the run lists every filing as
failed with a message that says so. For each parsed filing it sends the tables
that mention revenue, net income, total assets or segments, plus the start of Item 1A, to Claude. Every excerpt
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
- Context is picked by keyword. A statement table that never uses these words is not sent. At
  most 25 matching tables are sent, and tables that look like financial statements come first, so
  MD&A and note tables cannot push the statements out.
- The reference value is the one the filing itself reported, matched by accession number. A later
  filing that recasts the same period is not used.
- Filings whose extraction failed are listed under failures and are not counted in the accuracy.
- Banks tag revenue in several ways, so a bank can show as missing when no tried concept exists.

## Retrieval

`src/filings_rag/chunking.py` offers two strategies with the same size target of 1,500
characters. `fixed` fills each chunk from the blocks in document order and ignores structure.
`section` never crosses an Item or Part boundary and keeps each table in chunks of its own. Every
chunk records the pages it spans, so an answer built from it can cite a page. `bm25.py` ranks
chunks by keyword match.

`make eval-retrieval` compares the strategies on a question set you label by hand and writes
`data/RETRIEVAL_RESULTS.md`. Put one JSON object per line in `data/questions.jsonl`, or pass
another file with `--questions`:

```
{"question": "What does the company say about supplier concentration?", "document": "AAPL_10-K_2024-09-28", "answer": "...", "evidence": "..."}
```

Every line needs all four fields with text in them. A line that is not valid reports its line
number and stops the run. `document` is the file name in `data/raw/` without its extension.
`evidence` is a sentence or table row copied from the filing. It must sit inside one paragraph or
one table of the parsed text, so text that runs across a page break is not found. Case and
punctuation are ignored when matching. Gold pages are the pages of the parsed blocks that contain
the evidence, so labels follow the parser's physical page numbers and survive a re-parse. They are
not the page numbers printed in the filing.

Recall@k counts a question as found when one of the top k chunks of that filing spans a gold page.
Search is limited to the filing each question is about, because the filers share near identical
boilerplate and a search across all filings would mostly measure that. A chunk that spans several
pages counts as a hit for each of them, so a strategy with larger page spans scores higher for that
reason alone. Questions whose evidence cannot be found are listed in the report and left out of the
counts.

### Dense search, hybrid search and reranking

`make eval-retrieval` scores BM25 only. Add `--dense` to also score dense search and hybrid
search, and `--rerank` to add hybrid search followed by a reranker. The report then has one row
for each chunking strategy and method.

```
pip install -e ".[retrieval]"
filings-rag eval-retrieval --data data --dense --rerank
```

The models run locally through ONNX Runtime and are downloaded on first use into
`data/models/`, pinned to a fixed revision, about 220 MB in all. The embedding model is
`BAAI/bge-small-en-v1.5`. It uses the first token's vector, scaled to length 1, and queries
(but not chunks) start with the instruction sentence the model was trained with. The reranker is
the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`. Inputs are cut at 512 tokens.

Hybrid search takes the top 50 chunks from BM25 and the top 50 from dense search and merges
them by reciprocal rank fusion, which uses ranks only, so the two kinds of score need no common
scale. The reranker re-orders the top 30 of the hybrid result. Only the filings that questions
ask about are embedded, and chunk embeddings are cached in `data/embeddings.sqlite` under a key
made from the model revision and the text, so a rerun does not embed them again.

Dense vectors are held in memory unless you pass `--store pgvector`. That keeps them in a
Postgres table with the pgvector extension, one table per chunking strategy, rebuilt on each run.
Start a local database and point the command at it:

```
docker compose -f docker/pgvector.yml up -d
export FILINGS_RAG_PG_URL=postgresql://postgres@127.0.0.1:5433/postgres
filings-rag eval-retrieval --data data --dense --store pgvector
```

The compose file publishes the database on 127.0.0.1 only, on port 5433, with no password, so
use it for local experiments. The driver is `pg8000`, which needs no native Postgres library.

Tests use small stand-ins for the models. `tests/test_onnx_models.py` also has checks against the
real models, which are skipped until the models are downloaded; set `FILINGS_RAG_MODELS` to the
folder that holds them. `tests/test_pgvector.py` runs against a real database when
`FILINGS_RAG_PG_URL` is set. The models are small, so expect weaker results than from a large
embedding model, and a reranker trained on web search passages may not suit financial tables.
