"""Step 9.1 — rate, slab and threshold data, read only from the 2025 Act.

Every number the calculator may use lives in a per-tax-year data file, and every
entry quotes the Act verbatim beside its citation (ADR-100). The loader reads
each value back out of its own quote, so a value typed wrongly refuses to load;
`tests/test_rates.py` checks each quote against the cited chunk, so a quote
typed wrongly or pinned to the wrong provision fails the build.

What the Act does not print — old-regime slabs, age-based exemption limits,
surcharge, marginal relief on surcharge, cess — is declared under `outside_act`
with the Act's own deferral to "any Central Act", and asking for one raises
`OutsideActError`. Those values are never supplied from elsewhere: the corpus is
the Act, and a number the Act does not contain is not this system's to state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

RATES_STAGE_VERSION = 1

RATES_DIR = Path(__file__).parent
_FILENAME = re.compile(r"^tax_year_(\d{4})_(\d{2})\.json$")
_TAX_YEAR = re.compile(r"^\d{4}-\d{2}$")


class Unit(StrEnum):
    RUPEES = "rupees"
    PERCENT = "percent"
    YEARS = "years"


class RatesError(ValueError):
    """A data file that contradicts itself or its own quotes."""


class UnsupportedTaxYearError(LookupError):
    pass


class OutsideActError(LookupError):
    """The Act states no value for this component, so none may be computed."""

    def __init__(self, component: str, reason: str, citation: str) -> None:
        super().__init__(f"{component} is outside the Act (see {citation}): {reason}")
        self.component = component
        self.reason = reason
        self.citation = citation


@dataclass(frozen=True)
class Provenance:
    citation: str
    source_text: str


@dataclass(frozen=True)
class Slab:
    # A band covers income above `lower` up to and including `upper`, which is
    # how the Act's whole-rupee rows read: "From Rs. 400001 to Rs. 800000".
    lower: Decimal
    upper: Decimal | None
    rate_percent: Decimal
    provenance: Provenance


@dataclass(frozen=True)
class RateValue:
    value: Decimal
    unit: Unit
    provenance: Provenance


@dataclass(frozen=True)
class OutsideAct:
    reason: str
    provenance: Provenance


@dataclass(frozen=True)
class NotAllowed:
    """A deduction the section 202(1) computation excludes. `section` and
    `chapter` say which provision is excluded and why the quote reaches it."""

    section: str
    chapter: str
    provenance: Provenance


@dataclass(frozen=True)
class TaxYearRates:
    tax_year: str
    commencement: Provenance
    new_regime_slabs: tuple[Slab, ...]
    values: dict[str, RateValue]
    outside_act: dict[str, OutsideAct]
    not_allowed_under_202_1: dict[str, NotAllowed]
    rules: dict[str, Provenance]

    def value(self, name: str) -> RateValue:
        if name in self.outside_act:
            declared = self.outside_act[name]
            raise OutsideActError(name, declared.reason, declared.provenance.citation)
        if name not in self.values:
            raise KeyError(f"no rate named {name!r} for tax year {self.tax_year}")
        return self.values[name]

    def rule(self, name: str) -> Provenance:
        if name not in self.rules:
            raise KeyError(f"no rule named {name!r} for tax year {self.tax_year}")
        return self.rules[name]

    def provenances(self) -> tuple[Provenance, ...]:
        return (
            self.commencement,
            *(slab.provenance for slab in self.new_regime_slabs),
            *(entry.provenance for entry in self.values.values()),
            *(entry.provenance for entry in self.outside_act.values()),
            *(entry.provenance for entry in self.not_allowed_under_202_1.values()),
            *self.rules.values(),
        )


def supported_tax_years() -> tuple[str, ...]:
    years = []
    for path in sorted(RATES_DIR.glob("tax_year_*.json")):
        match = _FILENAME.match(path.name)
        if match:
            years.append(f"{match.group(1)}-{match.group(2)}")
    return tuple(years)


def load_rates(tax_year: str) -> TaxYearRates:
    if not _TAX_YEAR.match(tax_year):
        raise UnsupportedTaxYearError(f"{tax_year!r} does not read like 2026-27")
    path = RATES_DIR / f"tax_year_{tax_year.replace('-', '_')}.json"
    if not path.exists():
        raise UnsupportedTaxYearError(
            f"the Act carries no rate data for tax year {tax_year}; "
            f"supported: {', '.join(supported_tax_years())}"
        )
    return parse_rates(json.loads(path.read_text(encoding="utf-8")))


def parse_rates(payload: dict[str, Any]) -> TaxYearRates:
    tax_year = payload["tax_year"]
    values = {name: _value(name, entry) for name, entry in payload["values"].items()}
    outside = {
        name: OutsideAct(reason=entry["reason"], provenance=_provenance(entry))
        for name, entry in payload["outside_act"].items()
    }
    if overlap := sorted(values.keys() & outside.keys()):
        raise RatesError(f"declared both supported and outside the Act: {overlap}")
    for name, declared in outside.items():
        if _digits(declared.provenance.source_text):
            raise RatesError(f"{name} is outside the Act but its quote carries a number")
    return TaxYearRates(
        tax_year=tax_year,
        commencement=_provenance(payload["commencement"]),
        new_regime_slabs=_slabs(payload["new_regime_slabs"]),
        values=values,
        outside_act=outside,
        not_allowed_under_202_1={
            name: NotAllowed(section=entry["section"], chapter=entry["chapter"], provenance=_provenance(entry))
            for name, entry in payload["not_allowed_under_202_1"].items()
        },
        rules={name: _provenance(entry) for name, entry in payload["rules"].items()},
    )


def _provenance(entry: dict[str, Any]) -> Provenance:
    citation, source_text = entry.get("citation"), entry.get("source_text")
    if not citation or not source_text:
        raise RatesError(f"entry without citation or source text: {entry}")
    return Provenance(citation=citation, source_text=source_text)


def _decimal(raw: Any, what: str) -> Decimal:
    # Strings only: a JSON number is parsed as a float before anyone sees it,
    # which is the binary intermediate ADR-017 exists to keep out.
    if not isinstance(raw, str):
        raise RatesError(f"{what} must be a string, not {type(raw).__name__}")
    try:
        return Decimal(raw)
    except InvalidOperation as error:
        raise RatesError(f"{what} is not a number: {raw!r}") from error


def _value(name: str, entry: dict[str, Any]) -> RateValue:
    provenance = _provenance(entry)
    value = _decimal(entry["value"], name)
    unit = Unit(entry["unit"])
    if value not in _read_back(unit, provenance.source_text):
        raise RatesError(f"{name} = {value} is not what its quote says")
    return RateValue(value=value, unit=unit, provenance=provenance)


def _slabs(entries: list[dict[str, Any]]) -> tuple[Slab, ...]:
    slabs: list[Slab] = []
    lower = Decimal(0)
    for index, entry in enumerate(entries):
        provenance = _provenance(entry)
        rate = _decimal(entry["rate_percent"], "rate_percent")
        upper = None if entry["upper"] is None else _decimal(entry["upper"], "upper")
        last = index == len(entries) - 1
        if (upper is None) != last:
            raise RatesError("only the top slab is open-ended, and it must be")
        if upper is not None and upper <= lower:
            raise RatesError(f"slab {index} ends at {upper}, not above {lower}")
        if not Decimal(0) <= rate <= Decimal(100):
            raise RatesError(f"slab {index} rate {rate}% is not a percentage")
        if _read_slab(provenance.source_text) != (lower, upper, rate):
            raise RatesError(f"slab {index} is not what its quote says")
        slabs.append(Slab(lower, upper, rate, provenance))
        lower = upper if upper is not None else lower
    if not slabs:
        raise RatesError("no slabs")
    return tuple(slabs)


_RUPEES = re.compile(r"Rs\.\s*(\d+)")
_PERCENT = re.compile(r"(\d+)%")
_LAKH_WORDS = re.compile(r"\b([a-z-]+) lakh rupees\b")
_YEARS_WORDS = re.compile(r"\b([a-z-]+) years\b")
_UNITS = "zero one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _word_number(word: str) -> int | None:
    if word in _UNITS:
        return _UNITS.index(word)
    if word in _TEENS:
        return 10 + _TEENS.index(word)
    tens, _, units = word.partition("-")
    if tens in _TENS[2:] and (not units or units in _UNITS[1:]):
        return 10 * _TENS.index(tens) + (_UNITS.index(units) if units else 0)
    return None


def _read_back(unit: Unit, quote: str) -> set[Decimal]:
    """The numbers a quote states in `unit`. Figures and number words both, since
    the Act writes "Rs. 60000" and "twelve lakh rupees" in the same sentence."""
    text = " ".join(quote.split())
    found: set[int] = set()
    if unit is Unit.RUPEES:
        found |= {int(n) for n in _RUPEES.findall(text)}
        words = (_word_number(w) for w in _LAKH_WORDS.findall(text.lower()))
        found |= {n * 100_000 for n in words if n is not None}
    elif unit is Unit.PERCENT:
        found |= {int(n) for n in _PERCENT.findall(text)}
    else:
        words = (_word_number(w) for w in _YEARS_WORDS.findall(text.lower()))
        found |= {n for n in words if n is not None}
    return {Decimal(n) for n in found}


_SLAB_UPTO = re.compile(r"^Upto Rs\. (\d+) Nil$")
_SLAB_FROM = re.compile(r"^From Rs\. (\d+) to Rs\. (\d+) (\d+)%$")
_SLAB_ABOVE = re.compile(r"^Above Rs\. (\d+) (\d+)%$")


def _read_slab(quote: str) -> tuple[Decimal, Decimal | None, Decimal] | None:
    text = " ".join(quote.split())
    if match := _SLAB_UPTO.match(text):
        return Decimal(0), Decimal(match.group(1)), Decimal(0)
    if match := _SLAB_FROM.match(text):
        start, end, rate = (Decimal(g) for g in match.groups())
        return start - 1, end, rate
    if match := _SLAB_ABOVE.match(text):
        return Decimal(match.group(1)), None, Decimal(match.group(2))
    return None


def _digits(text: str) -> bool:
    return bool(re.search(r"\d", text))
