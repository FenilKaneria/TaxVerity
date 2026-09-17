"""Step 7.6 — the extraction node and its one repair retry.

No network: every test drives the real Step 7.2 client over a scripted
MockTransport, so what the node asks for — the strict schema, the temperature,
the repair prompt — is asserted on the bytes that actually left.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal

import httpx2
import pytest

from taxverity.config import Settings
from taxverity.facts import (
    FACTS_SCHEMA_NAME,
    FactField,
    FactIssue,
    FactStatus,
)
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import GROQ, LLMClient
from taxverity.llm.extract import (
    EXTRACTION_MAX_COMPLETION_TOKENS,
    EXTRACTION_STAGE_VERSION,
    EXTRACTION_TEMPERATURE,
    OBJECT_FORMAT,
    REPAIR_HINTS,
    REPAIRABLE,
    STRICT_FORMAT,
    SYSTEM_PROMPT,
    FactExtractor,
)
from taxverity.llm.tracing import NullTracer, TracedLLMClient
from taxverity.observability import PAN_MASK

LOGGER = "taxverity.llm.extract"
KEY = "test-key-not-real"
PAN = "ABCDE1234F"
TURN = "My salary is 14,00,000 and I put Rs. 1,50,000 into my PPF this year."


def payload(*entries: dict) -> str:
    return json.dumps({"fields": list(entries)})


def entry(
    name: str,
    value: str,
    status: str = "stated",
    span: str = "",
) -> dict:
    return {"name": name, "value": value, "status": status, "source_span": span}


SALARY = entry("salary_income", "1400000", span="My salary is 14,00,000")
PPF = entry("deduction_savings_insurance", "150000", span="Rs. 1,50,000")


def ok(text: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": GROQ.model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 60,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        },
    )


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return ok(payload())

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(record.getMessage() for record in self.records)


def build(*responses, **kwargs):
    """The node over a real client, so response_format reaches the wire."""
    recorder = Recorder(*responses)
    client = LLMClient(
        GROQ,
        KEY,
        http_client=httpx2.Client(transport=httpx2.MockTransport(recorder)),
    )
    return FactExtractor(client, **kwargs), recorder


@pytest.fixture
def capture():
    handler = CaptureHandler()
    logger = logging.getLogger(LOGGER)
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)


# --- the contract of the vocabulary the node asks against -------------------


def test_stage_version_is_declared():
    assert EXTRACTION_STAGE_VERSION == 4


def test_strict_format_names_the_facts_schema():
    assert STRICT_FORMAT["type"] == "json_schema"
    assert STRICT_FORMAT["json_schema"]["name"] == FACTS_SCHEMA_NAME
    assert STRICT_FORMAT["json_schema"]["strict"] is True


def test_object_format_is_the_measured_fallback_mode():
    assert OBJECT_FORMAT == {"type": "json_object"}


def test_every_repairable_issue_has_a_hint():
    assert set(REPAIR_HINTS) == set(REPAIRABLE)


def test_an_unknown_field_is_never_repaired():
    # 7.5 keeps it as a measurement of what the closed vocabulary misses;
    # asking the model to rename it into the enum destroys that.
    assert FactIssue.UNKNOWN_FIELD not in REPAIRABLE


def test_a_duplicate_field_is_never_repaired():
    assert FactIssue.DUPLICATE_FIELD not in REPAIRABLE


def test_the_prompt_names_every_field():
    for field in FactField:
        assert field.value in SYSTEM_PROMPT


def test_the_prompt_never_offers_a_status_the_model_may_not_assert():
    assert FactStatus.PROFILE_DEFAULT.value not in SYSTEM_PROMPT


# --- one clean turn ---------------------------------------------------------


def test_a_clean_turn_extracts_its_facts():
    node, _ = build(ok(payload(SALARY, PPF)))
    result = node.extract(TURN)
    salary = result.facts.get(FactField.SALARY_INCOME)
    assert salary is not None
    assert salary.value == Decimal("1400000")
    assert salary.status is FactStatus.STATED
    assert result.rejections == ()
    assert result.repaired is False


def test_one_clean_turn_costs_one_call():
    node, recorder = build(ok(payload(SALARY)))
    node.extract(TURN)
    assert len(recorder.requests) == 1


def test_unreported_fields_are_completed_as_missing():
    node, _ = build(ok(payload(SALARY, PPF)))
    result = node.extract(TURN)
    assert len(result.facts.facts) == len(FactField)
    assert FactField.AGE in result.facts.missing()
    assert FactField.SALARY_INCOME not in result.facts.missing()


def test_the_model_is_not_asked_to_enumerate_missing_fields():
    assert "a field you omit is recorded as missing" in SYSTEM_PROMPT


def test_a_model_emitted_missing_is_accepted_and_not_duplicated():
    node, _ = build(ok(payload(SALARY, entry("age", "", "missing"))))
    result = node.extract(TURN)
    assert len(result.facts.facts) == len(FactField)
    age = result.facts.get(FactField.AGE)
    assert age is not None and age.status is FactStatus.MISSING


def test_facts_come_back_in_declaration_order():
    node, _ = build(ok(payload(PPF, SALARY)))
    result = node.extract(TURN)
    order = [field for field in FactField]
    assert [fact.field for fact in result.facts.facts] == order


def test_tokens_and_degradation_are_reported():
    node, _ = build(ok(payload(SALARY)))
    result = node.extract(TURN)
    assert result.tokens == 260
    assert result.degraded is False


def test_an_empty_turn_is_refused_without_a_call():
    node, recorder = build()
    with pytest.raises(ValueError):
        node.extract("   ")
    assert recorder.requests == []


# --- what goes on the wire --------------------------------------------------


def test_the_strict_schema_is_requested_by_default():
    node, recorder = build(ok(payload(SALARY)))
    node.extract(TURN)
    assert recorder.bodies[0]["response_format"] == STRICT_FORMAT


def test_extraction_asks_for_no_creativity():
    node, recorder = build(ok(payload(SALARY)))
    node.extract(TURN)
    body = recorder.bodies[0]
    assert body["temperature"] == EXTRACTION_TEMPERATURE
    assert body["max_completion_tokens"] == EXTRACTION_MAX_COMPLETION_TOKENS


def test_the_turn_is_sent_as_a_user_message_under_the_system_prompt():
    node, recorder = build(ok(payload(SALARY)))
    node.extract(TURN)
    messages = recorder.bodies[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": TURN}


def test_a_pan_in_the_turn_never_reaches_the_wire():
    node, recorder = build(ok(payload()))
    node.extract(f"My PAN is {PAN} and I earn a salary.")
    sent = json.dumps(recorder.bodies[0])
    assert PAN not in sent
    assert PAN_MASK in sent


def test_the_turn_is_never_logged(capture):
    node, _ = build(ok(payload(SALARY)))
    node.extract(TURN)
    assert "14,00,000" not in capture.text


# --- the repair retry -------------------------------------------------------


def test_a_fabricated_span_is_repaired():
    bad = entry("salary_income", "1400000", span="I told you my salary")
    node, recorder = build(ok(payload(bad)), ok(payload(SALARY)))
    result = node.extract(TURN)
    assert result.repaired is True
    assert len(recorder.requests) == 2
    salary = result.facts.get(FactField.SALARY_INCOME)
    assert salary is not None and salary.value == Decimal("1400000")
    assert result.rejections == ()


def test_a_loss_written_positive_is_repaired_to_a_negative():
    turn = "I booked a business loss of 80,000."
    bad = entry("business_income", "80000", span="business loss of 80,000")
    good = entry("business_income", "-80000", span="business loss of 80,000")
    node, recorder = build(ok(payload(bad)), ok(payload(good)))
    result = node.extract(turn)
    repair = recorder.bodies[1]["messages"][-1]["content"]
    assert REPAIR_HINTS[FactIssue.SIGN_CONTRADICTS_SPAN] in repair
    assert result.facts.get(FactField.BUSINESS_INCOME).value == Decimal("-80000")
    assert result.rejections == ()


def test_the_prompt_asks_for_a_minus_sign_on_a_loss():
    assert "leading minus sign" in SYSTEM_PROMPT


def test_the_repair_quotes_the_rejected_entry_and_its_reason():
    bad = entry("age", "not a number")
    node, recorder = build(ok(payload(bad, SALARY)), ok(payload()))
    node.extract(TURN)
    repair = recorder.bodies[1]["messages"][-1]["content"]
    assert "not a number" in repair
    assert REPAIR_HINTS[FactIssue.MISSING_SPAN] in repair
    # The accepted entry is not re-asked: the model may not change what already
    # passed, and a second full answer costs a second full prompt.
    assert "salary_income" not in repair


def test_the_repair_carries_the_models_own_answer_back():
    bad = entry("age", "", "stated")
    first = payload(bad)
    node, recorder = build(ok(first), ok(payload()))
    node.extract(TURN)
    messages = recorder.bodies[1]["messages"]
    assert messages[2] == {"role": "assistant", "content": first}
    assert len(messages) == 4


def test_the_repair_is_bounded_at_one_attempt():
    bad = entry("age", "still not a number")
    node, recorder = build(ok(payload(bad)), ok(payload(bad)))
    result = node.extract(TURN)
    assert len(recorder.requests) == 2
    assert [r.issue for r in result.rejections] == [
        FactIssue.MISSING_SPAN,
        FactIssue.MISSING_SPAN,
    ]


def test_an_entry_the_repair_drops_is_still_reported():
    bad = entry("age", "not a number")
    node, _ = build(ok(payload(bad)), ok(payload()))
    result = node.extract(TURN)
    assert [r.issue for r in result.rejections] == [FactIssue.MISSING_SPAN]


def test_repair_can_be_turned_off():
    bad = entry("age", "not a number")
    node, recorder = build(ok(payload(bad)), repair=False)
    result = node.extract(TURN)
    assert len(recorder.requests) == 1
    assert result.repaired is False
    assert result.rejections[0].issue is FactIssue.MISSING_SPAN


def test_an_unknown_field_does_not_trigger_a_repair():
    node, recorder = build(ok(payload(entry("crypto_gains", "50000", span=TURN[:10]))))
    result = node.extract(TURN)
    assert len(recorder.requests) == 1
    assert result.repaired is False
    assert [u.name for u in result.facts.unmapped] == ["crypto_gains"]
    assert result.rejections[0].issue is FactIssue.UNKNOWN_FIELD


def test_a_duplicate_field_does_not_trigger_a_repair():
    node, recorder = build(ok(payload(SALARY, SALARY)))
    result = node.extract(TURN)
    assert len(recorder.requests) == 1
    assert result.rejections[0].issue is FactIssue.DUPLICATE_FIELD


def test_the_repair_may_not_change_an_accepted_field():
    bad = entry("age", "not a number")
    changed = entry("salary_income", "9", span="My salary is 14,00,000")
    node, _ = build(ok(payload(bad, SALARY)), ok(payload(changed)))
    result = node.extract(TURN)
    salary = result.facts.get(FactField.SALARY_INCOME)
    assert salary is not None and salary.value == Decimal("1400000")
    assert FactIssue.DUPLICATE_FIELD in {r.issue for r in result.rejections}


def test_facts_accepted_before_a_repair_survive_it():
    bad = entry("age", "not a number")
    node, _ = build(ok(payload(bad, SALARY)), ok(payload()))
    result = node.extract(TURN)
    salary = result.facts.get(FactField.SALARY_INCOME)
    assert salary is not None and salary.value == Decimal("1400000")


def test_both_calls_are_counted():
    bad = entry("age", "not a number")
    node, _ = build(ok(payload(bad)), ok(payload()))
    result = node.extract(TURN)
    assert len(result.completions) == 2
    assert result.tokens == 520


def test_a_repair_logs_what_it_is_repairing(capture):
    bad = entry("age", "not a number")
    node, _ = build(ok(payload(bad)), ok(payload()))
    node.extract(TURN)
    assert "repairing 1" in capture.text


# --- a payload that is not a payload ----------------------------------------


def test_a_completion_that_is_not_json_is_a_rejection_not_an_exception():
    node, _ = build(ok("I could not do that"), repair=False)
    result = node.extract(TURN)
    assert result.rejections[0].issue is FactIssue.MALFORMED_ENTRY
    assert result.facts.missing() == tuple(FactField)


def test_a_malformed_payload_is_repaired_whole():
    node, recorder = build(ok("not json at all"), ok(payload(SALARY)))
    result = node.extract(TURN)
    assert len(recorder.requests) == 2
    assert result.rejections == ()
    assert result.facts.get(FactField.SALARY_INCOME) is not None


def test_a_payload_that_stays_malformed_is_reported_once():
    node, _ = build(ok("not json"), ok("still not json"))
    result = node.extract(TURN)
    assert [r.issue for r in result.rejections] == [FactIssue.MALFORMED_ENTRY]


def test_a_payload_without_a_fields_list_is_refused():
    node, _ = build(ok(json.dumps({"facts": []})), repair=False)
    result = node.extract(TURN)
    assert result.rejections[0].issue is FactIssue.MALFORMED_ENTRY


# --- the strict-schema fallback ---------------------------------------------


def refusal() -> httpx2.Response:
    return httpx2.Response(400, text="response_format json_schema is not supported")


def test_a_provider_refusing_the_schema_falls_back_to_json_object():
    node, recorder = build(refusal(), ok(payload(SALARY)))
    result = node.extract(TURN)
    assert recorder.bodies[0]["response_format"] == STRICT_FORMAT
    assert recorder.bodies[1]["response_format"] == OBJECT_FORMAT
    assert result.facts.get(FactField.SALARY_INCOME) is not None


def test_the_refusal_is_paid_once_per_process():
    node, recorder = build(refusal(), ok(payload(SALARY)), ok(payload(SALARY)))
    node.extract(TURN)
    node.extract(TURN)
    assert node.schema_refused is True
    assert [b["response_format"] for b in recorder.bodies] == [
        STRICT_FORMAT,
        OBJECT_FORMAT,
        OBJECT_FORMAT,
    ]


def test_the_fallback_is_logged(capture):
    node, _ = build(refusal(), ok(payload(SALARY)))
    node.extract(TURN)
    assert "json_object" in capture.text


def test_the_fallback_still_polices_the_shape():
    # Nothing is enforced at the wire under json_object, so parse_facts() is the
    # only thing standing between an invented field and the calculator.
    node, _ = build(refusal(), ok(payload(entry("bitcoin", "1", span=TURN[:5]))))
    result = node.extract(TURN)
    assert [u.name for u in result.facts.unmapped] == ["bitcoin"]


# --- the stack ---------------------------------------------------------------


def test_from_settings_builds_the_phase_7_stack(tmp_path):
    settings = Settings(groq_api_key=KEY, data_dir=tmp_path)
    node = FactExtractor.from_settings(settings)
    traced = node._client
    assert isinstance(traced, TracedLLMClient)
    cached = traced._inner
    assert isinstance(cached, CachedLLMClient)
    assert isinstance(cached._inner, LLMClient)
    assert cached.directory == tmp_path / "llm"


def test_an_unconfigured_process_traces_nothing(tmp_path):
    settings = Settings(groq_api_key=KEY, data_dir=tmp_path)
    node = FactExtractor.from_settings(settings)
    assert isinstance(node._client._tracer, NullTracer)


def test_the_stack_is_optional(tmp_path):
    settings = Settings(groq_api_key=KEY, data_dir=tmp_path)
    node = FactExtractor.from_settings(settings, cache=False, trace=False)
    assert isinstance(node._client, LLMClient)


# --- redaction reaches what is stored, not only what is sent ----------------


def test_a_span_is_checked_against_the_turn_the_model_actually_saw():
    # redact() masks the PAN inside the client, so the only quotation the model
    # can make over it is of the masked text. Checking against the raw turn
    # would drop the fact as fabricated.
    turn = f"My PAN is {PAN} and my salary is 1400000."
    span = f"My PAN is {PAN_MASK} and my salary is 1400000."
    node, _ = build(ok(payload(entry("salary_income", "1400000", span=span))))
    result = node.extract(turn)
    salary = result.facts.get(FactField.SALARY_INCOME)
    assert salary is not None
    assert PAN not in salary.source_span


def test_a_stored_span_never_carries_a_pan():
    turn = f"My PAN is {PAN} and my salary is 1400000."
    quoting_the_raw_pan = entry("salary_income", "1400000", span=f"PAN is {PAN}")
    node, _ = build(ok(payload(quoting_the_raw_pan)), repair=False)
    result = node.extract(turn)
    assert PAN not in json.dumps(result.facts.model_dump(mode="json"))
    assert result.rejections[0].issue is FactIssue.SPAN_NOT_IN_TURN


def test_what_the_repair_was_asked_to_fix_is_reported():
    bad = entry("age", "not a number")
    node, _ = build(ok(payload(bad)), ok(payload()))
    result = node.extract(TURN)
    assert [r.issue for r in result.repairable] == [FactIssue.MISSING_SPAN]


# --- Step 7.8: the context guard through the node (ADR-109) -----------------


def test_a_loss_whose_span_omits_the_loss_word_is_repaired_to_a_negative():
    turn = "My business made a loss of 3,00,000 last year."
    bad = entry("business_income", "300000", span="3,00,000")
    good = entry("business_income", "-300000", span="3,00,000")
    node, recorder = build(ok(payload(bad)), ok(payload(good)))
    result = node.extract(turn)
    assert REPAIR_HINTS[FactIssue.SIGN_CONTRADICTS_SPAN] in recorder.bodies[1]["messages"][-1]["content"]
    assert result.facts.get(FactField.BUSINESS_INCOME).value == Decimal("-300000")
    assert result.rejections == ()


def test_a_positive_reaffirmed_in_a_loss_clause_is_dropped_never_accepted():
    turn = "My business made a loss of 3,00,000 last year."
    bad = entry("business_income", "300000", span="3,00,000")
    node, recorder = build(ok(payload(bad)), ok(payload(bad)))
    result = node.extract(turn)
    assert len(recorder.requests) == 2
    assert result.facts.get(FactField.BUSINESS_INCOME).status is FactStatus.MISSING
    assert {r.issue for r in result.rejections} == {FactIssue.SIGN_CONTRADICTS_SPAN}


# --- R20 Step 20.4: open-vocabulary situation facts --------------------------


def situation_entry(
    name: str,
    value: str,
    status: str = "stated",
    span: str = "",
) -> dict:
    return {"name": name, "value": value, "status": status, "source_span": span}


def payload_with_situation(*entries: dict, situation: tuple[dict, ...] = ()) -> str:
    return json.dumps({"fields": list(entries), "situation_facts": list(situation)})


def test_situation_facts_ride_alongside_the_closed_fields():
    turn = "My salary is 14,00,000. I pay rent to my mother for a flat."
    rent = situation_entry(
        "rent recipient", "my mother", span="pay rent to my mother"
    )
    node, recorder = build(ok(payload_with_situation(SALARY, situation=(rent,))))
    result = node.extract(turn)
    assert len(recorder.requests) == 1
    assert result.facts.get(FactField.SALARY_INCOME).value == Decimal("1400000")
    (fact,) = result.situation_facts
    assert fact.name == "rent recipient"
    assert fact.raw_value == "my mother"


def test_the_system_prompt_names_the_situation_facts_array():
    assert "situation_facts" in SYSTEM_PROMPT


def test_a_fabricated_situation_span_is_rejected_not_repaired():
    """A situation-fact rejection must never trigger the closed-field repair
    pass — that pass is keyed on `FactField`, and a repaired situation entry
    has no field to re-attach to."""
    turn = "My salary is 14,00,000."
    bad = situation_entry("rent recipient", "my mother", span="invented words")
    node, recorder = build(ok(payload_with_situation(SALARY, situation=(bad,))))
    result = node.extract(turn)
    assert len(recorder.requests) == 1  # no repair call fired
    assert result.situation_facts == ()
    assert result.situation_rejections != ()
    assert result.rejections == ()


def test_a_repair_does_not_touch_the_first_passs_situation_facts():
    """The repair prompt is built from `repairable` (closed fields only), so a
    repair triggered by a bad closed field must not drop the situation facts
    the first pass already accepted."""
    turn = "My salary is 14,00,000. I pay rent to my mother for a flat."
    rent = situation_entry(
        "rent recipient", "my mother", span="pay rent to my mother"
    )
    bad = entry("business_income", "not-a-number")
    fixed = entry("business_income", "50000")
    node, recorder = build(
        ok(payload_with_situation(SALARY, bad, situation=(rent,))),
        ok(payload_with_situation(fixed)),
    )
    result = node.extract(turn)
    assert len(recorder.requests) == 2
    assert result.repaired is True
    (fact,) = result.situation_facts
    assert fact.name == "rent recipient"

