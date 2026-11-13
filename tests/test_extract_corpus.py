import json

from filings_rag.evaluate import CORRECT, MISSING, WRONG, score_field
from filings_rag.extract_corpus import extract_corpus

STEM = "ACME_10-Q_2024-09-30"
ACCN = "0000000001-24-000001"
REPLY = json.dumps(
    {
        "revenue": {"value": 300, "page": 1},
        "net_income": {"value": 999, "page": 1},
    }
)


class FakeCompleter:
    def __init__(self):
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return REPLY


def quarter_fact(value: int) -> dict:
    fact = {"val": value, "start": "2024-07-01", "end": "2024-09-30"}
    fact |= {"form": "10-Q", "accn": ACCN}
    return {"units": {"USD": [fact]}}


def write_corpus(data_dir) -> None:
    for folder in ("raw", "parsed", "xbrl"):
        (data_dir / folder).mkdir()

    manifest = [
        {
            "ticker": "ACME",
            "company": "Acme Corp",
            "accession_number": ACCN,
            "form": "10-Q",
            "period_of_report": "2024-09-30",
            "path": f"raw/{STEM}.htm",
        }
    ]
    (data_dir / "manifest.json").write_text(json.dumps(manifest))

    table = {"kind": "table", "page": 1, "section": None, "text": "", "rows": [["Revenues", "300"]]}
    parsed = {"page_count": 1, "blocks": [table]}
    (data_dir / "parsed" / f"{STEM}.json").write_text(json.dumps(parsed))

    concepts = {"Revenues": quarter_fact(300), "NetIncomeLoss": quarter_fact(50)}
    (data_dir / "xbrl" / "ACME.json").write_text(json.dumps({"facts": {"us-gaap": concepts}}))


def test_score_field_outcomes():
    assert score_field(1000.5, 1000) == CORRECT
    assert score_field(2000, 1000) == WRONG
    assert score_field(None, 1000) == MISSING


def test_extract_corpus_scores_against_xbrl_and_writes_results(tmp_path):
    write_corpus(tmp_path)

    scores = extract_corpus(FakeCompleter(), tmp_path)

    outcomes = {s.field: s.outcome for s in scores}
    # total_assets has no XBRL value in the fixture, so it is not scored.
    assert outcomes == {"revenue": CORRECT, "net_income": WRONG}
    results = (tmp_path / "RESULTS.md").read_text()
    assert "| revenue | 1 | 1 | 0 | 0 | 100.0% |" in results
    assert f"{STEM} net_income: got 999, XBRL 50" in results


def test_rerun_keeps_existing_extractions(tmp_path):
    write_corpus(tmp_path)
    completer = FakeCompleter()

    extract_corpus(completer, tmp_path)
    extract_corpus(completer, tmp_path)

    assert completer.calls == 1


def test_unparsed_filing_is_listed_as_a_failure(tmp_path):
    write_corpus(tmp_path)
    (tmp_path / "parsed" / f"{STEM}.json").unlink()

    extract_corpus(FakeCompleter(), tmp_path)

    assert f"- {STEM}: FileNotFoundError" in (tmp_path / "RESULTS.md").read_text()


def test_corrupt_saved_extraction_is_extracted_again(tmp_path):
    write_corpus(tmp_path)
    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted" / f"{STEM}.json").write_text('{"revenue": {"val')
    completer = FakeCompleter()

    scores = extract_corpus(completer, tmp_path)

    assert completer.calls == 1
    assert len(scores) == 2


def test_one_failing_filing_does_not_stop_the_others(tmp_path):
    write_corpus(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    broken = {**manifest[0], "path": "raw/ACME_10-K_2023-12-31.htm"}
    (tmp_path / "manifest.json").write_text(json.dumps([broken, *manifest]))
    (tmp_path / "parsed" / "ACME_10-K_2023-12-31.json").write_text("not json")

    scores = extract_corpus(FakeCompleter(), tmp_path)

    assert len(scores) == 2  # the good filing was still scored
    assert "- ACME_10-K_2023-12-31: JSONDecodeError" in (tmp_path / "RESULTS.md").read_text()


def test_nan_value_is_rejected_and_not_saved(tmp_path):
    write_corpus(tmp_path)

    class NanCompleter:
        def complete(self, system: str, user: str) -> str:
            return '{"revenue": {"value": NaN, "page": 1}}'

    extract_corpus(NanCompleter(), tmp_path)

    assert not (tmp_path / "extracted" / f"{STEM}.json").exists()
    assert f"- {STEM}: ExtractionError" in (tmp_path / "RESULTS.md").read_text()
