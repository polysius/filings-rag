import json

from filings_rag.corpus import build_corpus, sha256_of
from filings_rag.edgar import FilingRef

FILING_BYTES = b"<html>filing</html>"
CONFIG = {"filings_per_company": {"10-K": 1}, "sectors": {"technology": ["AAPL"]}}


class FakeEdgarClient:
    def __init__(self):
        self.downloads = 0

    def ticker_to_cik(self):
        return {"AAPL": 320193}

    def company_name(self, cik):
        return "Apple Inc."

    def company_facts(self, cik):
        return b'{"facts": {}}'

    def list_filings(self, cik, form, limit):
        return [
            FilingRef(cik, "0000320193-23-000106", form, "2023-09-30", "2023-11-03", "k23.htm")
        ]

    def download_filing(self, filing):
        self.downloads += 1
        return FILING_BYTES


def test_build_corpus_writes_manifest_with_hash(tmp_path):
    entries = build_corpus(FakeEdgarClient(), CONFIG, tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(entries) == 1
    assert manifest[0]["company"] == "Apple Inc."
    assert manifest[0]["form"] == "10-K"
    assert manifest[0]["period_of_report"] == "2023-09-30"
    assert manifest[0]["sha256"] == sha256_of(FILING_BYTES)
    assert (tmp_path / manifest[0]["path"]).exists()
    assert (tmp_path / "xbrl" / "AAPL.json").exists()


def test_build_corpus_does_not_redownload_existing_files(tmp_path):
    client = FakeEdgarClient()

    build_corpus(client, CONFIG, tmp_path)
    build_corpus(client, CONFIG, tmp_path)

    assert client.downloads == 1
