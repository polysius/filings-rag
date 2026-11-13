"""Pydantic schema for the fields extracted from one 10-K or 10-Q."""

from pydantic import BaseModel, Field

MAX_RISK_FACTORS = 5


class Amount(BaseModel):
    """A dollar amount and the page it was read from."""

    value: float = Field(description="Amount in US dollars, not thousands or millions")
    page: int = Field(ge=1, description="Page number shown in the [page N] marker")


class Segment(BaseModel):
    name: str
    revenue: Amount | None = None


class FilingExtraction(BaseModel):
    """Key fields for one filing. A field the filing does not state is null or empty."""

    revenue: Amount | None = Field(
        default=None, description="Total revenue for the current fiscal period"
    )
    net_income: Amount | None = Field(
        default=None, description="Net income for the current fiscal period"
    )
    total_assets: Amount | None = Field(
        default=None, description="Total assets at the end of the current period"
    )
    segments: list[Segment] = Field(default_factory=list, description="Reportable segments")
    top_risk_factors: list[str] = Field(
        default_factory=list,
        max_length=MAX_RISK_FACTORS,
        description="Short titles of the most prominent risk factors, most important first",
    )


# Fields that XBRL company facts can confirm. Segments and risk factors have no single XBRL
# concept, so they are extracted but not scored.
SCORED_FIELDS = ("revenue", "net_income", "total_assets")
