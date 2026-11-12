"""Build the filings corpus and its manifest (company, form, period, file hash)."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from filings_rag.edgar import EdgarClient


@dataclass
class ManifestEntry:
    ticker: str
    sector: str
    company: str
    cik: int
    form: str
    period_of_report: str
    filing_date: str
    accession_number: str
    source_url: str
    path: str
    sha256: str


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_corpus(client: EdgarClient, config: dict, out_dir: Path) -> list[ManifestEntry]:
    """Download filings and XBRL facts listed in `config`, then write `manifest.json`.

    Files already on disk are not downloaded again, so an interrupted run can resume.
    """
    raw_dir = out_dir / "raw"
    xbrl_dir = out_dir / "xbrl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    xbrl_dir.mkdir(parents=True, exist_ok=True)

    cik_by_ticker = client.ticker_to_cik()
    entries: list[ManifestEntry] = []

    for sector, tickers in config["sectors"].items():
        for ticker in tickers:
            cik = cik_by_ticker[ticker]
            company = client.company_name(cik)

            facts_path = xbrl_dir / f"{ticker}.json"
            if not facts_path.exists():
                facts_path.write_bytes(client.company_facts(cik))

            for form, count in config["filings_per_company"].items():
                for filing in client.list_filings(cik, form, count):
                    suffix = Path(filing.primary_document).suffix
                    filing_path = raw_dir / f"{ticker}_{form}_{filing.period_of_report}{suffix}"
                    if not filing_path.exists():
                        filing_path.write_bytes(client.download_filing(filing))
                    entries.append(
                        ManifestEntry(
                            ticker=ticker,
                            sector=sector,
                            company=company,
                            cik=cik,
                            form=form,
                            period_of_report=filing.period_of_report,
                            filing_date=filing.filing_date,
                            accession_number=filing.accession_number,
                            source_url=filing.url,
                            path=filing_path.relative_to(out_dir).as_posix(),
                            sha256=sha256_of(filing_path.read_bytes()),
                        )
                    )

    write_manifest(entries, out_dir / "manifest.json")
    return entries


def write_manifest(entries: list[ManifestEntry], path: Path) -> None:
    path.write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")
