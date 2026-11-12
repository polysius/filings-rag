from filings_rag.edgar import FilingRef, select_filings

RECENT = {
    "form": ["10-Q", "8-K", "10-K", "10-Q", "10-K"],
    "accessionNumber": [
        "0000320193-24-000081",
        "0000320193-24-000070",
        "0000320193-23-000106",
        "0000320193-23-000077",
        "0000320193-22-000108",
    ],
    "reportDate": ["2024-06-29", "2024-05-02", "2023-09-30", "2023-07-01", "2022-09-24"],
    "filingDate": ["2024-08-02", "2024-05-02", "2023-11-03", "2023-08-04", "2022-10-28"],
    "primaryDocument": ["q3.htm", "8k.htm", "k23.htm", "q2.htm", "k22.htm"],
}


def test_select_filings_returns_newest_matches_of_requested_form():
    filings = select_filings(320193, RECENT, "10-K", limit=1)

    assert [f.period_of_report for f in filings] == ["2023-09-30"]


def test_select_filings_skips_other_forms_and_respects_limit():
    filings = select_filings(320193, RECENT, "10-Q", limit=5)

    assert [f.form for f in filings] == ["10-Q", "10-Q"]


def test_filing_url_removes_dashes_from_accession_number():
    filing = FilingRef(
        cik=320193,
        accession_number="0000320193-23-000106",
        form="10-K",
        period_of_report="2023-09-30",
        filing_date="2023-11-03",
        primary_document="k23.htm",
    )

    assert filing.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/k23.htm"
    )
