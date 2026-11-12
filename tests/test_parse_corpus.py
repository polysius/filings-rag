import json

from filings_rag.parse_corpus import parse_corpus

GOOD_FILING = (
    "<html><body><div><b>Item 1. Business</b></div><p>Some text.</p>"
    "<table><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table></body></html>"
)


def write_corpus(tmp_path, files: dict[str, str]) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = []
    for name, html in files.items():
        (raw / name).write_text(html, encoding="utf-8")
        manifest.append({"path": f"raw/{name}"})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_parse_corpus_reports_counts_and_lists_failures(tmp_path):
    write_corpus(tmp_path, {"good.htm": GOOD_FILING, "empty.htm": "<html><body></body></html>"})

    results = parse_corpus(tmp_path)

    good, empty = results
    assert (good.status, good.pages, good.headings, good.text_blocks, good.tables) == (
        "ok", 1, 1, 1, 1
    )  # fmt: skip
    assert empty.status == "failed"
    assert empty.error == "no content extracted"

    report = (tmp_path / "PARSING_REPORT.md").read_text()
    assert "raw/good.htm | 1 | 1 | 1 | 1" in report
    assert "- raw/empty.htm: no content extracted" in report


def test_parse_corpus_writes_parsed_blocks_with_pages(tmp_path):
    write_corpus(tmp_path, {"good.htm": GOOD_FILING})

    parse_corpus(tmp_path)

    parsed = json.loads((tmp_path / "parsed" / "good.json").read_text())
    assert parsed["page_count"] == 1
    assert {block["page"] for block in parsed["blocks"]} == {1}
    assert [block["kind"] for block in parsed["blocks"]] == ["heading", "text", "table"]


def test_parse_corpus_warns_when_document_has_no_tables(tmp_path):
    write_corpus(tmp_path, {"plain.htm": "<html><body><p>Only text.</p></body></html>"})

    result = parse_corpus(tmp_path)[0]

    assert "no tables found" in result.warnings
    assert "no Item or Part headings found" in result.warnings


def test_parse_corpus_removes_output_of_a_filing_that_now_fails(tmp_path):
    write_corpus(tmp_path, {"good.htm": GOOD_FILING})
    parse_corpus(tmp_path)
    (tmp_path / "raw" / "good.htm").write_text("<html><body></body></html>")

    results = parse_corpus(tmp_path)

    assert results[0].status == "failed"
    assert not (tmp_path / "parsed" / "good.json").exists()
