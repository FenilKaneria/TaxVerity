"""Step 7.2 — the internal LLM client.

Every test but the last drives a scripted MockTransport: no network, no key.
The last one hits Groq and is skipped unless a key is configured.
"""

from __future__ import annotations

import json
import logging

import httpx2
import pytest

from taxverity.config import MissingSettingError, Settings
from taxverity.llm import client as llm
from taxverity.llm.client import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    GEMINI,
    GROQ,
    LLMClient,
    LLMError,
    LLMRequestError,
    LLMUnavailable,
    Message,
    Usage,
)
from taxverity.observability import PAN_MASK

LOGGER = "taxverity.llm.client"
KEY = "test-key-not-real"
FALLBACK_KEY = "test-fallback-key-not-real"
PAN = "ABCDE1234F"
ASK = [Message(role="user", content="What is the standard deduction?")]


def ok(
    text: str = "thirty per cent",
    *,
    model: str = GROQ.model,
    finish_reason: str = "stop",
    prompt: int = 120,
    completion: int = 40,
    reasoning: int = 12,
) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
        },
    )


class Recorder:
    """Scripts responses in order, then echoes a well-formed one."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.responses:
            scripted = self.responses.pop(0)
            if isinstance(scripted, Exception):
                raise scripted
            return scripted
        return ok()

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


def make(handler: Recorder, *, with_fallback: bool = False, **kwargs) -> LLMClient:
    kwargs.setdefault("backoff_base", 0.0)
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return LLMClient(
        GROQ,
        KEY,
        fallback=GEMINI if with_fallback else None,
        fallback_key=FALLBACK_KEY if with_fallback else None,
        http_client=http,
        **kwargs,
    )


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    return slept


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured():
    # Attached to the module's own logger, below both the `propagate = False`
    # root and RedactingFilter, so an assertion is about what the call site
    # passed rather than what the filter cleaned up (the Step 4.2 caplog trap).
    handler = CaptureHandler()
    logger = logging.getLogger(LOGGER)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


# --- the request ------------------------------------------------------------


def test_a_completion_carries_the_provider_that_answered():
    completion = make(Recorder()).complete(ASK)
    assert completion.text == "thirty per cent"
    assert completion.provider == "groq"
    assert completion.model == GROQ.model
    assert completion.finish_reason == "stop"
    assert completion.degraded is False


def test_the_request_goes_to_the_providers_own_chat_endpoint():
    handler = Recorder()
    make(handler).complete(ASK)
    assert handler.urls == [f"{GROQ.base_url}/chat/completions"]
    assert handler.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_provider_extras_ride_along_with_the_model():
    handler = Recorder()
    make(handler).complete(ASK)
    (body,) = handler.bodies
    assert body["model"] == GROQ.model
    # Step 7.1: reasoning cannot be disabled, so the default is the cheapest
    # setting that exists.
    assert body["reasoning_effort"] == "low"


def test_the_completion_cap_is_sized_for_reasoning_plus_answer():
    handler = Recorder()
    make(handler).complete(ASK)
    assert handler.bodies[0]["max_completion_tokens"] == DEFAULT_MAX_COMPLETION_TOKENS
    assert DEFAULT_MAX_COMPLETION_TOKENS > 400


def test_a_response_format_is_passed_through_untouched():
    handler = Recorder()
    schema = {"type": "json_schema", "json_schema": {"name": "facts", "strict": True}}
    make(handler).complete(ASK, response_format=schema)
    assert handler.bodies[0]["response_format"] == schema


def test_optional_fields_are_absent_rather_than_null():
    handler = Recorder()
    make(handler).complete(ASK)
    body = handler.bodies[0]
    assert "response_format" not in body
    assert "temperature" not in body


def test_an_empty_conversation_is_refused_before_any_call():
    handler = Recorder()
    with pytest.raises(ValueError):
        make(handler).complete([])
    assert handler.requests == []


# --- redaction (rule 03, egress path 5) -------------------------------------


def test_a_pan_never_leaves_the_process():
    handler = Recorder()
    make(handler).complete([Message(role="user", content=f"My PAN is {PAN}")])
    sent = handler.bodies[0]["messages"][0]["content"]
    assert PAN not in sent
    assert PAN_MASK in sent


def test_redaction_leaves_statutory_text_and_amounts_intact():
    handler = Recorder()
    evidence = "Under section 80C and 2(5)(b)(ii), the deduction is 150000."
    make(handler).complete([Message(role="system", content=evidence), *ASK])
    assert handler.bodies[0]["messages"][0]["content"] == evidence


def test_the_request_body_is_never_logged(captured, no_sleep):
    handler = Recorder(httpx2.Response(503), ok())
    make(handler).complete([Message(role="user", content=f"PAN {PAN} salary 1400000")])
    assert captured.records, "the capture must be live or this asserts nothing"
    for record in captured.records:
        line = record.getMessage()
        assert PAN not in line
        assert "1400000" not in line


# --- retry ------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_an_availability_failure_is_retried(status, no_sleep):
    handler = Recorder(httpx2.Response(status), ok())
    assert make(handler).complete(ASK).text == "thirty per cent"
    assert len(handler.requests) == 2


def test_a_transport_error_is_retried(no_sleep):
    handler = Recorder(httpx2.ConnectError("no route"), ok())
    assert make(handler).complete(ASK).provider == "groq"
    assert len(handler.requests) == 2


def test_backoff_doubles_between_attempts(no_sleep):
    handler = Recorder(httpx2.Response(503), httpx2.Response(503), ok())
    make(handler, backoff_base=0.5).complete(ASK)
    assert no_sleep == [0.5, 1.0]


def test_retry_after_overrides_the_backoff(no_sleep):
    handler = Recorder(httpx2.Response(429, headers={"retry-after": "3"}), ok())
    make(handler, backoff_base=0.5).complete(ASK)
    assert no_sleep == [3.0]


def test_an_unparseable_retry_after_falls_back_to_the_backoff(no_sleep):
    handler = Recorder(httpx2.Response(429, headers={"retry-after": "soon"}), ok())
    make(handler, backoff_base=0.5).complete(ASK)
    assert no_sleep == [0.5]


def test_exhausted_retries_raise_unavailable(no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 3)
    with pytest.raises(LLMUnavailable, match="after 3 attempts"):
        make(handler).complete(ASK)
    assert len(handler.requests) == 3


def test_a_bad_request_is_never_retried(no_sleep):
    handler = Recorder(httpx2.Response(400, text="tool_use_failed"))
    with pytest.raises(LLMRequestError, match="400"):
        make(handler).complete(ASK)
    assert len(handler.requests) == 1
    assert no_sleep == []


def test_max_attempts_must_be_at_least_one():
    with pytest.raises(ValueError):
        make(Recorder(), max_attempts=0)


# --- the single cross-vendor fallback ---------------------------------------


def test_the_fallback_answers_when_the_primary_is_exhausted(no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 3, ok(model=GEMINI.model))
    completion = make(handler, with_fallback=True).complete(ASK)
    assert completion.provider == "gemini"
    assert completion.degraded is True
    assert handler.urls[-1] == f"{GEMINI.base_url}/chat/completions"
    assert handler.requests[-1].headers["authorization"] == f"Bearer {FALLBACK_KEY}"


def test_the_fallback_is_not_sent_a_gpt_oss_only_control(no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 3, ok(model=GEMINI.model))
    make(handler, with_fallback=True).complete(ASK)
    assert "reasoning_effort" not in handler.bodies[-1]
    assert handler.bodies[-1]["model"] == GEMINI.model


def test_a_bad_request_does_not_fail_over(no_sleep):
    """The same request is refused by the other vendor too, so trying it twice
    turns one clear error into two and hides the cause."""
    handler = Recorder(httpx2.Response(400, text="schema rejected"))
    with pytest.raises(LLMRequestError):
        make(handler, with_fallback=True).complete(ASK)
    assert len(handler.requests) == 1


def test_both_failures_are_named_when_the_fallback_also_fails(no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 6)
    with pytest.raises(LLMUnavailable) as raised:
        make(handler, with_fallback=True).complete(ASK)
    assert "groq" in str(raised.value) and "gemini" in str(raised.value)


def test_without_a_fallback_the_primary_failure_surfaces_unchanged(no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 3)
    with pytest.raises(LLMUnavailable, match="groq"):
        make(handler).complete(ASK)
    assert len(handler.requests) == 3


def test_the_fallback_is_only_tried_once(no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 6)
    with pytest.raises(LLMUnavailable):
        make(handler, with_fallback=True, max_attempts=3).complete(ASK)
    gemini_calls = [u for u in handler.urls if u.startswith(GEMINI.base_url)]
    assert len(gemini_calls) == 3


def test_degradation_is_logged_as_a_warning(captured, no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 3, ok(model=GEMINI.model))
    make(handler, with_fallback=True).complete(ASK)
    warnings = [
        r.getMessage() for r in captured.records if r.levelno == logging.WARNING
    ]
    assert any("falling back to gemini" in line for line in warnings)


def test_a_fallback_provider_without_a_key_is_refused():
    with pytest.raises(ValueError, match="key"):
        LLMClient(GROQ, KEY, fallback=GEMINI)


def test_an_empty_primary_key_is_refused():
    with pytest.raises(ValueError):
        LLMClient(GROQ, "")


# --- accounting -------------------------------------------------------------


def test_usage_is_reported_per_call():
    completion = make(Recorder()).complete(ASK)
    assert completion.usage == Usage(
        prompt_tokens=120, completion_tokens=40, reasoning_tokens=12
    )
    assert completion.usage.total_tokens == 160


def test_tokens_accumulate_per_provider(no_sleep):
    handler = Recorder(ok(), *[httpx2.Response(503)] * 3, ok(model=GEMINI.model))
    client = make(handler, with_fallback=True)
    client.complete(ASK)
    client.complete(ASK)
    assert client.tokens_used == {"groq": 160, "gemini": 160}


def test_missing_usage_counts_as_zero_rather_than_raising():
    handler = Recorder(
        httpx2.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
        )
    )
    completion = make(handler).complete(ASK)
    assert completion.usage.total_tokens == 0


# --- malformed responses ----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [{"choices": []}, {}, {"choices": [{}]}, {"choices": "not a list"}],
)
def test_a_malformed_body_degrades_rather_than_raising_a_key_error(body, no_sleep):
    handler = Recorder(httpx2.Response(200, json=body))
    with pytest.raises(LLMError):
        make(handler).complete(ASK)


def test_a_non_json_body_is_an_availability_failure(no_sleep):
    handler = Recorder(httpx2.Response(200, text="<html>gateway</html>"))
    with pytest.raises(LLMUnavailable):
        make(handler).complete(ASK)


def test_a_message_without_a_content_key_is_empty_text_not_a_failure():
    """The Step 7.1 `plain` probe returned exactly this: reasoning consumed the
    whole cap and the answer came back empty. It is a real response."""
    handler = Recorder(
        httpx2.Response(200, json={"choices": [{"message": {"role": "assistant"}}]})
    )
    completion = make(handler).complete(ASK)
    assert completion.text == ""
    assert completion.finish_reason == "unknown"


def test_a_content_free_choice_reads_as_empty_text():
    handler = Recorder(
        httpx2.Response(
            200,
            json={
                "choices": [{"message": {"content": None}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )
    )
    completion = make(handler).complete(ASK)
    assert completion.text == ""
    assert completion.finish_reason == "length"


# --- construction from settings ---------------------------------------------


# These read LLMClient._PRIMARY/_FALLBACK rather than hardcoding GROQ/GEMINI
# so they stay correct across the temporary primary/fallback swap (2026-09-15,
# see client.py) without needing an edit themselves.
_PRIMARY = LLMClient._PRIMARY
_FALLBACK = LLMClient._FALLBACK


def test_from_settings_without_the_primary_key_names_the_variable(monkeypatch):
    monkeypatch.delenv(f"TAXVERITY_{_PRIMARY.settings_key.upper()}", raising=False)
    settings = Settings(_env_file=None)
    with pytest.raises(MissingSettingError, match=_PRIMARY.settings_key.upper()):
        LLMClient.from_settings(settings)


def test_from_settings_without_the_fallback_key_runs_with_no_fallback(captured, no_sleep):
    settings = Settings(
        _env_file=None, **{_PRIMARY.settings_key: KEY, _FALLBACK.settings_key: None}
    )
    handler = Recorder(*[httpx2.Response(503)] * 3)
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    client = LLMClient.from_settings(settings, http_client=http, backoff_base=0.0)
    warnings = [
        r.getMessage() for r in captured.records if r.levelno == logging.WARNING
    ]
    assert any(
        f"no TAXVERITY_{_FALLBACK.settings_key.upper()}" in line for line in warnings
    )
    with pytest.raises(LLMUnavailable):
        client.complete(ASK)
    assert all(u.startswith(_PRIMARY.base_url) for u in handler.urls)


def test_from_settings_wires_the_fallback_when_the_key_is_present(no_sleep):
    settings = Settings(
        _env_file=None,
        **{_PRIMARY.settings_key: KEY, _FALLBACK.settings_key: FALLBACK_KEY},
    )
    handler = Recorder(*[httpx2.Response(503)] * 3, ok(model=_FALLBACK.model))
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    client = LLMClient.from_settings(settings, http_client=http, backoff_base=0.0)
    assert client.complete(ASK).provider == _FALLBACK.name


def test_a_borrowed_http_client_is_left_open():
    http = httpx2.Client(transport=httpx2.MockTransport(Recorder()))
    LLMClient(GROQ, KEY, http_client=http).close()
    assert not http.is_closed


# --- the real provider ------------------------------------------------------


@pytest.mark.skipif(
    getattr(Settings(), LLMClient._PRIMARY.settings_key) is None,
    reason=f"TAXVERITY_{LLMClient._PRIMARY.settings_key.upper()} is not configured",
)
def test_the_real_provider_answers_and_bills_tokens():
    # Either provider is an acceptable answer here — a live vendor call can
    # legitimately fail over (both are free tiers with their own rate
    # limits), and `degraded` is exactly the signal that already exists to
    # say so. What this test actually guards is that *something* answers.
    with LLMClient.from_settings(Settings()) as client:
        completion = client.complete(
            [Message(role="user", content="Reply with exactly: ready.")],
            max_completion_tokens=256,
        )
    assert completion.provider in {LLMClient._PRIMARY.name, LLMClient._FALLBACK.name}
    assert completion.usage.total_tokens > 0
    assert "ready" in completion.text.lower()
    assert completion.usage.total_tokens > 0
