from filings_rag.xbrl import reference_value


def facts_with(concept: str, entries: list[dict]) -> dict:
    return {"facts": {"us-gaap": {concept: {"units": {"USD": entries}}}}}


def entry(val, end, form="10-Q", start=None, filed="2024-05-01"):
    result = {"val": val, "end": end, "form": form, "filed": filed}
    if start:
        result["start"] = start
    return result


def test_10q_revenue_picks_the_quarter_not_the_year_to_date():
    facts = facts_with(
        "Revenues",
        [
            entry(900, "2024-09-30", start="2024-01-01"),  # nine months to date
            entry(300, "2024-09-30", start="2024-07-01"),  # the quarter
        ],
    )

    assert reference_value(facts, "revenue", "10-Q", "2024-09-30") == 300


def test_revenue_falls_back_to_the_contract_revenue_concept():
    facts = facts_with(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        [entry(1200, "2023-12-31", form="10-K", start="2023-01-01")],
    )

    assert reference_value(facts, "revenue", "10-K", "2023-12-31") == 1200


def test_total_assets_uses_the_instant_value():
    facts = facts_with("Assets", [entry(5000, "2024-06-30")])

    assert reference_value(facts, "total_assets", "10-Q", "2024-06-30") == 5000


def test_restated_value_uses_the_latest_filing():
    facts = facts_with(
        "NetIncomeLoss",
        [
            entry(10, "2023-12-31", form="10-K", start="2023-01-01", filed="2024-02-01"),
            entry(12, "2023-12-31", form="10-K", start="2023-01-01", filed="2024-03-01"),
        ],
    )

    assert reference_value(facts, "net_income", "10-K", "2023-12-31") == 12


def test_returns_none_when_no_fact_matches():
    facts = facts_with("Revenues", [entry(300, "2024-09-30", start="2024-07-01")])

    assert reference_value(facts, "revenue", "10-K", "2024-09-30") is None
    assert reference_value(facts, "net_income", "10-Q", "2024-09-30") is None
