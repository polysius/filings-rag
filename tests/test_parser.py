from filings_rag.parser import clean_table_rows, parse_filing

FILING_HTML = """
<html><body>
<div style="display:none"><ix:header>hidden xbrl context</ix:header></div>
<div><span style="font-weight:700">PART I</span></div>
<div><span style="font-weight:700">Item 1. Business</span></div>
<div>The company designs phones.<br/>It sells them worldwide.</div>
<hr style="page-break-after:always"/>
<div>2</div>
<div><b>Item 1A. Risk Factors</b></div>
<p>Item 1A of this report is discussed below, and it is not a heading.</p>
<table>
  <tr><td></td><td>2023</td><td></td><td>2022</td></tr>
  <tr><td>Net sales</td><td>$</td><td>383,285</td><td>$</td><td>394,328</td></tr>
  <tr><td>Change</td><td>(4</td><td>)</td><td>%</td></tr>
</table>
<table><tr><td>Layout only cell</td></tr></table>
<div style="page-break-before:always"><span>Last page text</span></div>
</body></html>
"""


def test_parse_filing_separates_headings_text_and_tables():
    document = parse_filing(FILING_HTML)

    kinds = [(block.kind, block.text) for block in document.blocks if block.kind != "table"]
    assert ("heading", "Item 1. Business") in kinds
    assert ("text", "The company designs phones. It sells them worldwide.") in kinds
    assert len(document.of_kind("table")) == 1


def test_parse_filing_ignores_hidden_xbrl_header_and_bare_page_numbers():
    document = parse_filing(FILING_HTML)

    all_text = " ".join(block.text for block in document.blocks)
    assert "hidden xbrl context" not in all_text
    assert not any(block.text == "2" for block in document.blocks)


def test_parse_filing_records_page_numbers_from_page_breaks():
    document = parse_filing(FILING_HTML)

    page_of = {block.text: block.page for block in document.blocks if block.text}
    assert page_of["Item 1. Business"] == 1
    assert page_of["Item 1A. Risk Factors"] == 2
    assert page_of["Last page text"] == 3
    assert document.page_count == 3


def test_parse_filing_tracks_current_section():
    document = parse_filing(FILING_HTML)

    table = document.of_kind("table")[0]
    assert table.section == "Item 1A. Risk Factors"


def test_sentence_starting_with_item_is_not_a_heading():
    document = parse_filing(FILING_HTML)

    sentence = next(b for b in document.blocks if b.text.startswith("Item 1A of this"))
    assert sentence.kind == "text"


def test_one_cell_layout_table_becomes_text():
    document = parse_filing(FILING_HTML)

    assert any(b.kind == "text" and b.text == "Layout only cell" for b in document.blocks)


def test_clean_table_rows_merges_symbols_and_keeps_values_in_their_columns():
    rows = clean_table_rows(
        [
            ["", "2023", "", "2022"],
            ["Net sales", "$", "383,285", "$", "394,328"],
            ["Other", "", "", "$", "5"],
            ["", "", "", "", ""],
        ]
    )

    assert rows == [
        ["", "2023", "2022"],
        ["Net sales", "$383,285", "$394,328"],
        ["Other", "", "$5"],
    ]


def parse_blocks(html: str):
    return parse_filing(html).blocks


def test_invalid_byte_does_not_truncate_document():
    html = (
        b"<html><body><p>Revenue \xe2\x80\x94 grew.</p><p>Quote \xe2\x80\x9d end.</p>"
        b"<p>Company\x92s</p><p>after</p></body></html>"
    )

    texts = [block.text for block in parse_filing(html).blocks]

    assert texts[-1] == "after"
    assert len(texts) == 4


def test_windows_1252_file_decodes_curly_quotes():
    html = b"<html><body><p>The Company\x92s \x93core\x94 business</p></body></html>"

    text = parse_filing(html).blocks[0].text

    assert text == "The Company’s “core” business"


def test_table_values_stay_under_their_column_headers():
    html = (
        "<table><tr><td></td><td colspan='2'>2023</td><td colspan='2'>2022</td></tr>"
        "<tr><td>Net sales</td><td>$</td><td>383</td><td>$</td><td>394</td></tr>"
        "<tr><td>Other</td><td></td><td></td><td>$</td><td>5</td></tr></table>"
    )

    rows = parse_blocks(html)[0].rows

    assert rows == [["", "2023", "2022"], ["Net sales", "$383", "$394"], ["Other", "", "$5"]]


def test_nested_table_cells_are_not_repeated():
    html = (
        "<table><tr><td><table><tr><td>A</td><td>1</td></tr></table></td><td>X</td></tr>"
        "<tr><td>B</td><td>2</td></tr></table>"
    )

    rows = parse_blocks(html)[0].rows

    assert rows == [["A 1", "X"], ["B", "2"]]


def test_line_breaks_and_block_cells_separate_words():
    html = (
        "<p><font size=2>First line<br>Second line</font></p>"
        "<table><tr><td>Year Ended<br>December 31,</td><td>x</td></tr>"
        "<tr><td><div>Year Ended</div><div>Sept 30</div></td><td>y</td></tr></table>"
    )

    blocks = parse_blocks(html)

    assert blocks[0].text == "First line Second line"
    assert blocks[1].rows == [["Year Ended December 31,", "x"], ["Year Ended Sept 30", "y"]]


def test_nested_bold_text_is_a_heading_and_hidden_text_is_dropped():
    html = (
        "<p><font size=2><b>Liquidity and Capital Resources</b></font></p>"
        "<div><span><span style='font-weight:700'>Competition</span></span></div>"
        "<div><span>Visible <span style='display:none'>SECRET</span>end</span></div>"
    )

    blocks = parse_blocks(html)

    assert [b.kind for b in blocks] == ["heading", "heading", "text"]
    assert blocks[2].text == "Visible end"


def test_sentences_mentioning_parts_and_items_are_not_section_headings():
    html = (
        "<p>Part I, Item 2 of this Form 10-Q contains forward-looking statements.</p>"
        "<p>Part II of this report is incorporated by reference.</p>"
        "<p>Later text.</p>"
    )

    blocks = parse_blocks(html)

    assert {b.kind for b in blocks} == {"text"}
    assert {b.section for b in blocks} == {None}


def test_combined_and_unpunctuated_item_headings_set_the_section():
    html = (
        "<p><b>ITEMS 1 AND 2. BUSINESS AND PROPERTIES</b></p><p>Body one.</p>"
        "<p><b>Item 7 Management's Discussion</b></p><p>Body two.</p>"
    )

    blocks = parse_blocks(html)

    assert blocks[1].section == "ITEMS 1 AND 2. BUSINESS AND PROPERTIES"
    assert blocks[3].section == "Item 7 Management's Discussion"


def test_heading_split_across_a_one_row_layout_table_becomes_a_heading():
    html = (
        "<table><tr><td><b>Item 7.</b></td><td><b>Management's Discussion</b></td></tr></table>"
        "<p>Body.</p>"
    )

    blocks = parse_blocks(html)

    assert (blocks[0].kind, blocks[0].text) == ("heading", "Item 7. Management's Discussion")
    assert blocks[1].section == "Item 7. Management's Discussion"


def test_one_cell_table_keeps_its_paragraphs_separate():
    html = (
        "<table><tr><td><p><b>Item 1. Business</b></p><p>First para.</p>"
        "<p>Second para.</p></td></tr></table>"
    )

    blocks = parse_blocks(html)

    assert [(b.kind, b.text) for b in blocks] == [
        ("heading", "Item 1. Business"),
        ("text", "First para."),
        ("text", "Second para."),
    ]


def test_page_break_on_br_or_inline_element_starts_a_new_page():
    html = (
        "<p>Page one.</p><br clear=all style='page-break-before:always'><p>Page two.</p>"
        "<p><span><br style='page-break-before:always'></span>Page three.</p>"
    )

    assert [b.page for b in parse_blocks(html)] == [1, 2, 3]


def test_trailing_page_break_does_not_add_an_empty_page():
    document = parse_filing("<p>Only page text.</p><hr style='page-break-after:always'/>")

    assert document.page_count == 1


def test_image_only_page_still_counts_as_a_page():
    html = (
        "<div>P1</div><hr style='page-break-after:always'/><div><img src='g.jpg'></div>"
        "<hr style='page-break-after:always'/><div>P3 text</div>"
    )

    assert parse_blocks(html)[-1].page == 3


def test_page_footers_are_dropped():
    html = (
        "<p>Body text.</p><p>F-12</p><p>- 2 -</p><p>ii</p><p>Apple Inc. | 2023 Form 10-K | 17</p>"
    )

    assert [b.text for b in parse_blocks(html)] == ["Body text."]


def test_table_caption_is_kept_as_text():
    html = "<table><caption>Revenue by segment</caption><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>"

    blocks = parse_blocks(html)

    assert (blocks[0].kind, blocks[0].text) == ("text", "Revenue by segment")
    assert blocks[1].kind == "table"


def test_table_block_converts_to_dataframe():
    table = parse_filing(FILING_HTML).of_kind("table")[0]

    frame = table.to_dataframe()

    assert frame.iloc[1, 0] == "Net sales"
    assert frame.iloc[1, 1] == "$383,285"
    assert frame.iloc[2, 1] == "(4)%"


def test_header_over_the_number_cell_and_plain_number_rows_share_columns():
    html = (
        "<table><tr><td></td><td></td><td>2023</td><td></td><td>2022</td></tr>"
        "<tr><td>Net sales</td><td>$</td><td>383</td><td>$</td><td>394</td></tr>"
        "<tr><td>Cost</td><td></td><td>200</td><td></td><td>210</td></tr></table>"
    )

    rows = parse_blocks(html)[0].rows

    assert rows == [["", "2023", "2022"], ["Net sales", "$383", "$394"], ["Cost", "200", "210"]]


def test_symbols_next_to_symbols_are_not_merged():
    assert clean_table_rows([["Total", "$", "", "$", "5"]]) == [["Total", "$", "$5"]]


def test_inline_tags_do_not_add_spaces_inside_words_or_numbers():
    html = (
        "<div><span>Net sales increased </span><span>8</span><span>% to $</span>"
        "<span><ix:nonfraction>96.2</ix:nonfraction></span><span> billion.</span></div>"
        "<p>The Company<span>’s</span> non-<i>GAAP</i> results</p>"
    )

    texts = [b.text for b in parse_blocks(html)]

    assert texts == ["Net sales increased 8% to $96.2 billion.", "The Company’s non-GAAP results"]


def test_spacer_rows_hidden_rows_and_nested_captions_do_not_hide_a_layout_table():
    html = (
        "<table><tr><td></td><td></td></tr>"
        "<tr><td><b>Item 7.</b></td><td><b>Management's Discussion</b></td></tr>"
        "<tr style='display:none'><td>SECRET</td><td>9</td></tr></table><p>Body.</p>"
    )

    blocks = parse_blocks(html)

    assert (blocks[0].kind, blocks[0].text) == ("heading", "Item 7. Management's Discussion")
    assert "SECRET" not in " ".join(b.text for b in blocks)


def test_long_bold_item_heading_without_period_sets_the_section():
    title = "Item 5 Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases"
    blocks = parse_blocks(f"<p><b>{title}</b></p><p>Body.</p>")

    assert (blocks[0].kind, blocks[1].section) == ("heading", title)


def test_bold_roman_numeral_is_kept_but_lowercase_page_numeral_is_dropped():
    blocks = parse_blocks("<p><b>PART</b></p><p><b>I</b></p><p>ii</p>")

    assert [b.text for b in blocks] == ["PART", "I"]
