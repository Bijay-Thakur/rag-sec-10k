"""SEC XBRL fact normalization and conservative decimal arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN

from .models import DiscoveryError, normalize_cik, validate_accession


@dataclass(frozen=True)
class Fact:
    cik: str
    accession: str
    taxonomy: str
    concept: str
    unit: str
    value: Decimal
    start: date | None
    end: date
    filed: date
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    frame: str | None
    source_pointer: str = ""

    @property
    def identity(self) -> str:
        return f"{self.cik}:{self.accession}:{self.taxonomy}:{self.concept}:{self.unit}:{self.start}:{self.end}"


def normalize_companyfacts(payload: object, cik: str, accession: str) -> list[Fact]:
    cik, accession = normalize_cik(cik), validate_accession(accession)
    if not isinstance(payload, dict) or str(payload.get("cik", "")).zfill(10) != cik:
        raise DiscoveryError("xbrl_schema", "XBRL filer identity mismatch", 502)
    result: list[Fact] = []
    try:
        for taxonomy, concepts in payload["facts"].items():
            for concept, detail in concepts.items():
                for unit, observations in detail["units"].items():
                    for index, row in enumerate(observations):
                        if row.get("accn") != accession:
                            continue
                        value = Decimal(str(row["val"]))
                        start = date.fromisoformat(row["start"]) if row.get("start") else None
                        end, filed = date.fromisoformat(row["end"]), date.fromisoformat(row["filed"])
                        if (not value.is_finite() or row["form"] not in ("10-K", "10-Q", "10-K/A", "10-Q/A")
                                or (start and start > end) or filed < end):
                            raise ValueError("nonfinite")
                        result.append(Fact(cik, accession, taxonomy, concept, unit, value,
                                           start, end, filed,
                                           row["form"], row.get("fy"), row.get("fp"), row.get("frame"),
                                           "/facts/" + "/".join(str(v).replace("~", "~0").replace("/", "~1")
                                                                 for v in (taxonomy, concept, "units", unit, index))))
    except (KeyError, TypeError, AttributeError, ValueError, InvalidOperation) as exc:
        raise DiscoveryError("xbrl_schema", "Malformed XBRL fact response", 502) from exc
    return result


def select_as_filed(facts: list[Fact], *, cik: str, accession: str, taxonomy: str,
                    concept: str, unit: str, end: date, start: date | None,
                    as_of: date | None = None) -> Fact:
    """Never silently select a different accession, amendment, unit or duration."""
    cik, accession = normalize_cik(cik), validate_accession(accession)
    matching = [fact for fact in facts if fact.cik == cik and fact.accession == accession
                and fact.taxonomy == taxonomy and fact.concept == concept and fact.unit == unit
                and fact.start == start and fact.end == end
                and (as_of is None or fact.filed <= as_of)]
    if not matching:
        raise DiscoveryError("fact_unavailable", "Exact as-filed fact unavailable", 404)
    newest = max(fact.filed for fact in matching)
    candidates = [fact for fact in matching if fact.filed == newest]
    if len({fact.value for fact in candidates}) > 1:
        raise DiscoveryError("fact_ambiguous", "Conflicting facts for selected filing and period", 409)
    return candidates[0]


@dataclass(frozen=True)
class Calculation:
    formula: str
    value: Decimal
    unit: str
    operand_ids: tuple[str, ...]


def calculate(formula: str, current: Fact, other: Fact, *, years: int | None = None) -> Calculation:
    if (current.unit != other.unit or current.cik != other.cik or current.taxonomy != other.taxonomy
            or not current.value.is_finite() or not other.value.is_finite()):
        raise DiscoveryError("incomparable_facts", "Units or filer differ", 422)
    if (current.start is None) != (other.start is None):
        raise DiscoveryError("incomparable_facts", "Instant and duration facts cannot mix", 422)
    if current.start and other.start:
        lengths = ((current.end - current.start).days, (other.end - other.start).days)
        if abs(lengths[0] - lengths[1]) > 7:
            raise DiscoveryError("incomparable_facts", "Reporting durations differ", 422)
    if other.value == 0:
        raise DiscoveryError("division_by_zero", "Denominator is zero", 422)
    if formula == "yoy":
        if current.concept != other.concept or not 350 <= (current.end - other.end).days <= 380:
            raise DiscoveryError("incomparable_facts", "YoY needs the same concept one year apart", 422)
        value = (current.value - other.value) / abs(other.value) * 100
        unit = "percent"
    elif formula in ("margin", "ratio"):
        if formula == "margin" and current.start is None:
            raise DiscoveryError("incomparable_facts", "Margin needs duration operands", 422)
        if current.accession != other.accession or current.start != other.start or current.end != other.end:
            raise DiscoveryError("incomparable_facts", "Operands require one filing and period", 422)
        value = current.value / other.value * (100 if formula == "margin" else 1)
        unit = "percent" if formula == "margin" else "ratio"
    elif formula == "cagr":
        if current.concept != other.concept or not years or years < 1 or years > 20 or current.value <= 0 or other.value <= 0:
            raise DiscoveryError("incomparable_facts", "CAGR inputs invalid", 422)
        if abs((current.end - other.end).days / 365.2425 - years) > .1:
            raise DiscoveryError("incomparable_facts", "CAGR periods differ", 422)
        value = ((current.value / other.value) ** (Decimal(1) / Decimal(years)) - 1) * 100
        unit = "percent"
    else:
        raise DiscoveryError("unsupported_formula", "Formula unavailable", 422)
    return Calculation(formula, value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN),
                       unit, (current.identity, other.identity))
