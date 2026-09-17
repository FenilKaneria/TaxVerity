"""R20 Step 20.5 — deterministic validation of the `reason` call's raw
output against the evidence pack and the thread's facts (rule 02's
recurring test: a deterministic check wherever one is possible, and this is
one).

The model's analysis is data, not truth (PLAN R20's "Reasoning validation"
section, and standing amendment 2 — `reason` "may not invent a condition,
draw on model memory, add a legal number the pack doesn't ground... or have
its own output served without going through the same claim verifier as
everything else"):

- Every marker must fall inside the pack; a marker outside it is dropped,
  and a rule or condition left with none is dropped outright (uncited).
- Every number a rule, limit, exception or definition states must be
  grounded in the units its own markers actually name
  (`generation.verifier.ground_numbers` — the exact check a claim's own
  numbers already pass at 20.7). A rule whose own statement carries a
  number its cited passages do not is dropped whole, not trimmed: a rule
  is one legal statement, and a fabricated figure inside it means the
  statement itself cannot be trusted.
- A `satisfied`/`not_satisfied` check must reference a fact this thread
  has actually stated or inferred; missing that, it is downgraded to
  `unknown` rather than dropped — the check (which condition, which rule)
  is still meaningful even once its verdict cannot be trusted.
- A `MissingFact` must point at a condition that survived the step above
  and is `unknown` after it; any legal number in its question must be
  grounded in that condition's own cited passages (amendment 3 — the
  person may be asked for their own figure, never handed a rate or
  threshold the model invented). Its length is capped, not refused.
- This module only ever drops or trims. It never rewrites a status, a
  number or a citation to make something pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from taxverity.facts import FactStatus
from taxverity.generation.verifier import ground_numbers, numbers_in
from taxverity.memory.fact_state import ThreadFactState
from taxverity.reasoning.models import (
    AnswerPlan,
    CheckStatus,
    Condition,
    ConditionCheck,
    LegalRule,
    MissingFact,
    ReasoningAnalysis,
)
from taxverity.retrieval.evidence import EvidencePack

VALIDATE_STAGE_VERSION = 1

# A clarify question this long is a paragraph, not a short targeted
# follow-up — capped the same way `CLARIFY_TEMPLATES` stays a fixed, short
# sentence, per PLAN R20's "its length is capped."
MAX_QUESTION_LENGTH = 240


@dataclass(frozen=True)
class ValidatedAnalysis:
    legal_rules: tuple[LegalRule, ...]
    applicability: tuple[ConditionCheck, ...]
    missing_facts: tuple[MissingFact, ...]
    answer_plan: AnswerPlan

    @property
    def has_governing_rule(self) -> bool:
        """False means nothing survived validation at all — the caller
        falls back to plain generation over the pack (PLAN R20: "Uncited
        rules are dropped. If nothing survives, fall back to today's
        path")."""
        return bool(self.legal_rules)


def validate(
    analysis: ReasoningAnalysis, pack: EvidencePack, *, fact_state: ThreadFactState
) -> ValidatedAnalysis:
    valid_markers = range(1, len(pack.units) + 1)
    known_facts = _known_fact_names(fact_state)

    rules: list[LegalRule] = []
    condition_markers: dict[str, tuple[int, ...]] = {}
    for raw_rule in analysis.legal_rules:
        rule = _validate_rule(raw_rule, pack, valid_markers)
        if rule is None:
            continue
        rules.append(rule)
        for condition in rule.conditions:
            condition_markers[condition.id] = condition.markers or rule.markers

    applicability: list[ConditionCheck] = []
    unknown_condition_ids: set[str] = set()
    for check in analysis.applicability:
        if check.condition_id not in condition_markers:
            continue
        validated = _validate_check(check, known_facts)
        applicability.append(validated)
        if validated.status is CheckStatus.UNKNOWN:
            unknown_condition_ids.add(check.condition_id)

    missing_facts: list[MissingFact] = []
    for missing in analysis.missing_facts:
        if missing.condition_id not in unknown_condition_ids:
            continue
        allowed = _grounded_numbers(pack, condition_markers[missing.condition_id])
        if not (numbers_in(missing.question) <= allowed):
            continue
        missing_facts.append(_capped(missing))

    plan_numbers = _grounded_numbers(pack, valid_markers) | _fact_numbers(fact_state)
    answer_plan = _validate_plan(analysis.answer_plan, plan_numbers)

    return ValidatedAnalysis(
        legal_rules=tuple(rules),
        applicability=tuple(applicability),
        missing_facts=tuple(missing_facts),
        answer_plan=answer_plan,
    )


def _validate_rule(
    rule: LegalRule, pack: EvidencePack, valid_markers: range
) -> LegalRule | None:
    markers = tuple(m for m in rule.markers if m in valid_markers)
    if not markers:
        return None
    allowed = _grounded_numbers(pack, markers)
    if not (numbers_in(rule.rule) <= allowed):
        return None
    conditions = tuple(
        validated
        for raw_condition in rule.conditions
        if (validated := _validate_condition(raw_condition, pack, valid_markers, markers))
        is not None
    )
    return rule.model_copy(
        update={
            "markers": markers,
            "limits": tuple(t for t in rule.limits if numbers_in(t) <= allowed),
            "exceptions": tuple(t for t in rule.exceptions if numbers_in(t) <= allowed),
            "definitions": tuple(t for t in rule.definitions if numbers_in(t) <= allowed),
            "conditions": conditions,
        }
    )


def _validate_condition(
    condition: Condition, pack: EvidencePack, valid_markers: range, rule_markers: tuple[int, ...]
) -> Condition | None:
    # An omitted `markers` list (the schema's own "leave empty to reuse the
    # rule's own markers") falls back to the rule's citation. A *non-empty*
    # list that filters to nothing is a different thing — the model tried to
    # cite something specific and got it wrong — and is dropped rather than
    # silently substituted, the same distrust a fabricated marker earns
    # everywhere else in this module.
    if condition.markers:
        own_markers = tuple(m for m in condition.markers if m in valid_markers)
        if not own_markers:
            return None
        grounding_markers = own_markers
    else:
        own_markers = ()
        grounding_markers = rule_markers
    if not grounding_markers:
        return None
    if not (numbers_in(condition.text) <= _grounded_numbers(pack, grounding_markers)):
        return None
    return condition.model_copy(update={"markers": own_markers})


def _validate_check(check: ConditionCheck, known_facts: set[str]) -> ConditionCheck:
    if check.status not in (CheckStatus.SATISFIED, CheckStatus.NOT_SATISFIED):
        return check
    if any(_normalise_ref(ref) in known_facts for ref in check.fact_refs):
        return check
    return check.model_copy(update={"status": CheckStatus.UNKNOWN})


def _validate_plan(plan: AnswerPlan, allowed: frozenset[Decimal]) -> AnswerPlan:
    steps = tuple(step for step in plan.steps if numbers_in(step) <= allowed)
    next_step = plan.next_step if numbers_in(plan.next_step) <= allowed else ""
    return plan.model_copy(update={"steps": steps, "next_step": next_step})


def _capped(missing: MissingFact) -> MissingFact:
    if len(missing.question) <= MAX_QUESTION_LENGTH:
        return missing
    truncated = missing.question[:MAX_QUESTION_LENGTH].rstrip() + "…"
    return missing.model_copy(update={"question": truncated})


def _grounded_numbers(pack: EvidencePack, markers: Iterable[int]) -> frozenset[Decimal]:
    numbers: set[Decimal] = set()
    for marker in markers:
        if 1 <= marker <= len(pack.units):
            numbers |= ground_numbers(pack.units[marker - 1])
    return frozenset(numbers)


def _normalise_ref(ref: str) -> str:
    return ref.strip().casefold()


def _known_fact_names(fact_state: ThreadFactState) -> set[str]:
    names: set[str] = set()
    for field_, fact in fact_state.facts.items():
        if fact.status in (FactStatus.STATED, FactStatus.INFERRED):
            names.add(_normalise_ref(field_.value))
    # `fact_state.situation` is already keyed by the normalised name.
    names |= {
        key
        for key, situation_fact in fact_state.situation.items()
        if situation_fact.status in (FactStatus.STATED, FactStatus.INFERRED)
    }
    return names


def _fact_numbers(fact_state: ThreadFactState) -> frozenset[Decimal]:
    numbers: set[Decimal] = set()
    for fact in fact_state.facts.values():
        if fact.status not in (FactStatus.STATED, FactStatus.INFERRED):
            continue
        if isinstance(fact.value, bool):
            continue
        if isinstance(fact.value, (Decimal, int)):
            numbers.add(Decimal(fact.value).normalize())
            numbers.add(abs(Decimal(fact.value)).normalize())
        elif isinstance(fact.value, str):
            numbers |= numbers_in(fact.value)
    for situation_fact in fact_state.situation.values():
        if situation_fact.status in (FactStatus.STATED, FactStatus.INFERRED):
            numbers |= numbers_in(situation_fact.raw_value)
    return frozenset(numbers)
