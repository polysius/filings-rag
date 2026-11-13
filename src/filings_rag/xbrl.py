"""Look up ground-truth values in SEC XBRL company facts.

Two details make a naive lookup wrong:

- Companies tag the same measure with different concepts. Revenue can be `Revenues` or
  `RevenueFromContractWithCustomer...`, so each field tries its concepts in order.
- A 10-Q reports several periods that end on the same date: the quarter and the year to date.
  Flow values (revenue, net income) are matched on period length as well as end date.
"""

from datetime import date

CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "total_assets": ["Assets"],
}
INSTANT_FIELDS = {"total_assets"}  # balance sheet values have no start date

# Allowed period length in days: a fiscal year (52 or 53 weeks) or one fiscal quarter.
DURATION_DAYS = {"10-K": (350, 380), "10-Q": (80, 100)}


def reference_value(facts: dict, field: str, form: str, period_end: str) -> float | None:
    """Return the XBRL value for `field` in the filing of `form` ending `period_end`, if any."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in CONCEPTS[field]:
        entries = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
        matches = [e for e in entries if _matches(e, field, form, period_end)]
        if matches:
            # A later filing can restate a value, so the most recently filed one wins.
            return max(matches, key=lambda e: e["filed"])["val"]
    return None


def _matches(entry: dict, field: str, form: str, period_end: str) -> bool:
    if entry.get("form") != form or entry.get("end") != period_end:
        return False
    if field in INSTANT_FIELDS:
        return "start" not in entry
    if "start" not in entry:
        return False
    low, high = DURATION_DAYS[form]
    days = (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days
    return low <= days <= high
