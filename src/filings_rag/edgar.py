"""Thin client for the SEC EDGAR APIs.

The SEC requires a descriptive User-Agent and asks clients to stay under
10 requests per second, so every request is identified and throttled.
"""

import os
import time
from dataclasses import dataclass

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

MIN_SECONDS_BETWEEN_REQUESTS = 0.15  # about 6 requests per second, under the SEC limit


@dataclass(frozen=True)
class FilingRef:
    """Pointer to one filing, enough to download it and describe it in the manifest."""

    cik: int
    accession_number: str
    form: str
    period_of_report: str
    filing_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return DOCUMENT_URL.format(
            cik=self.cik,
            accession_nodash=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )


def user_agent_from_env() -> str:
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "Set SEC_USER_AGENT to 'Your Name your.email@example.com'. "
            "The SEC rejects requests without a contact User-Agent."
        )
    return user_agent


class EdgarClient:
    def __init__(self, user_agent: str, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent}, timeout=30, follow_redirects=True
        )
        self._last_request_at = 0.0

    def _get(self, url: str) -> httpx.Response:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        response = self._client.get(url)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def ticker_to_cik(self) -> dict[str, int]:
        entries = self._get(TICKERS_URL).json().values()
        return {entry["ticker"].upper(): int(entry["cik_str"]) for entry in entries}

    def company_name(self, cik: int) -> str:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()["name"]

    def list_filings(self, cik: int, form: str, limit: int) -> list[FilingRef]:
        """Return the most recent `limit` filings of one form type."""
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        return select_filings(cik, recent, form, limit)

    def company_facts(self, cik: int) -> bytes:
        return self._get(FACTS_URL.format(cik=cik)).content

    def download_filing(self, filing: FilingRef) -> bytes:
        return self._get(filing.url).content


def select_filings(cik: int, recent: dict, form: str, limit: int) -> list[FilingRef]:
    """Pick the newest filings of `form` from the columnar `recent` block of a submissions file.

    EDGAR lists filings newest first, so taking the first matches gives the latest ones.
    """
    rows = zip(
        recent["form"],
        recent["accessionNumber"],
        recent["reportDate"],
        recent["filingDate"],
        recent["primaryDocument"],
    )
    matches = [
        FilingRef(
            cik=cik,
            accession_number=accession,
            form=row_form,
            period_of_report=report_date,
            filing_date=filing_date,
            primary_document=document,
        )
        for row_form, accession, report_date, filing_date, document in rows
        if row_form == form
    ]
    return matches[:limit]
