"""Step 7.4 — Langfuse tracing.

Every test drives a scripted MockTransport: no network, no Langfuse, no key.
The rule-03 assertions read the literal request body, which is the point of
hand-rolling the envelope — the bytes on the wire are the thing under test.
"""

from __future__ import annotations

import base64
import json
import logging

import httpx2
import pytest

from taxverity.config import Settings
from taxverity.llm.client import Completion, LLMUnavailable, Message, Usage
from taxverity.llm.tracing import (
    INGESTION_PATH,
    MAX_BATCH_EVENTS,
    TRACING_STAGE_VERSION,
    LangfuseTracer,
    NullTracer,
    TracedLLMClient,
    Tracer,
)
from taxverity.observability import PAN_MASK

LOGGER = "taxverity.llm.tracing"
PUBLIC = "pk-lf-test-not-real"
SECRET = "sk-lf-test-not-real"
HOST = "http://127.0.0.1:3000"
PAN = "ABCDE1234F"
ASK = [Message(role="user", content="What is the standard deduction?")]


def answer(
    text: str = "thirty per cent",
    *,
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
    degraded: bool = False,
) -> Completion:
    return Completion(
        text=text,
        provider=provider,
        model=model,
        finish_reason="stop",
        usage=Usage(prompt_tokens=120, completion_tokens=40, reasoning_tokens=12),
        degraded=degraded,
    )


class Recorder:
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
        return httpx2.Response(207, json={"successes": [], "errors": []})

    @property
    def batches(self) -> list[list[dict]]:
        return [json.loads(r.read())["batch"] for r in self.requests]

    @property
    def raw(self) -> str:
        return "".join(r.read().decode("utf-8") for r in self.requests)


def make(handler: Recorder, **kwargs) -> LangfuseTracer:
    return LangfuseTracer(
        PUBLIC,
        SECRET,
        HOST,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        **kwargs,
    )


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured():
    # Attached to the module's own logger, below the `propagate = False` root
    # and below RedactingFilter, so an assertion is about what the call site
    # passed rather than what the filter cleaned up (the Step 4.2 caplog trap).
    handler = CaptureHandler()
    logger = logging.getLogger(LOGGER)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


def warnings(captured: CaptureHandler) -> list[str]:
    return [r.getMessage() for r in captured.records if r.levelno == logging.WARNING]


class Failing:
    def complete(self, messages, **kwargs):
        raise LLMUnavailable("groq failed after 3 attempts")


class Fake:
    def __init__(self, completion: Completion | None = None) -> None:
        self.completion = completion or answer()
        self.calls: list[dict] = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        return self.completion


class Exploding:
    def generation(self, *args, **kwargs):
        raise RuntimeError("tracer is broken")

    def flush(self) -> None:
        raise RuntimeError("tracer is broken")

    def close(self) -> None:
        return None


# --- redaction: rule 03, egress path 1 -------------------------------------


def test_a_pan_in_a_prompt_never_reaches_the_wire():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", [Message(role="user", content=f"my PAN is {PAN}")])
    tracer.flush()

    assert PAN not in handler.raw
    assert PAN_MASK in handler.raw


def test_a_pan_echoed_back_in_a_completion_never_reaches_the_wire():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer(f"your PAN {PAN} is noted"))
    tracer.flush()

    assert PAN not in handler.raw
    assert PAN_MASK in handler.raw


def test_statutory_tokens_and_amounts_cross_a_trace_unchanged():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation(
        "llm",
        [Message(role="user", content="section 80C and 2(5)(b)(ii) on 1400000")],
        completion=answer("354A applies"),
    )
    tracer.flush()

    body = handler.raw
    for token in ("section 80C", "2(5)(b)(ii)", "1400000", "354A"):
        assert token in body


def test_redaction_survives_the_traced_client_wrapper():
    handler = Recorder()
    tracer = make(handler)
    client = TracedLLMClient(Fake(), tracer)
    client.complete([Message(role="user", content=f"PAN {PAN}")])
    tracer.flush()

    assert PAN not in handler.raw


# --- the envelope ----------------------------------------------------------


def test_a_generation_emits_a_trace_and_a_generation_sharing_a_trace_id():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("extract-facts", ASK, completion=answer())
    tracer.flush()

    batch = handler.batches[0]
    assert [event["type"] for event in batch] == [
        "trace-create",
        "generation-create",
    ]
    trace, generation = batch
    assert generation["body"]["traceId"] == trace["body"]["id"]
    assert trace["body"]["name"] == generation["body"]["name"] == "extract-facts"


def test_every_envelope_carries_its_own_unique_id_and_a_timestamp():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    batch = handler.batches[0]
    ids = [event["id"] for event in batch]
    assert len(set(ids)) == len(ids) == 4
    assert all(event["timestamp"].endswith("Z") for event in batch)
    # The event id deduplicates the envelope; the body id is the object.
    assert all(event["id"] != event["body"]["id"] for event in batch)


def test_usage_and_model_are_recorded_on_the_generation():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    body = handler.batches[0][1]["body"]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["usage"] == {
        "promptTokens": 120,
        "completionTokens": 40,
        "totalTokens": 160,
    }
    assert body["metadata"]["reasoning_tokens"] == 12
    assert body["metadata"]["stage_version"] == TRACING_STAGE_VERSION


def test_a_degraded_completion_is_traced_as_degraded():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer(provider="gemini", degraded=True))
    tracer.flush()

    metadata = handler.batches[0][1]["body"]["metadata"]
    assert metadata["degraded"] is True
    assert metadata["provider"] == "gemini"


def test_a_normal_completion_is_not_traced_as_degraded():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    assert handler.batches[0][1]["body"]["metadata"]["degraded"] is False


def test_a_failed_call_is_traced_at_error_level_with_no_output():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, error="LLMUnavailable: groq failed")
    tracer.flush()

    body = handler.batches[0][1]["body"]
    assert body["level"] == "ERROR"
    assert body["statusMessage"] == "LLMUnavailable: groq failed"
    assert body["output"] is None


def test_model_parameters_are_recorded_when_given():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation(
        "llm", ASK, completion=answer(), model_parameters={"temperature": 0.0}
    )
    tracer.flush()

    assert handler.batches[0][1]["body"]["modelParameters"] == {"temperature": 0.0}


def test_latency_puts_the_start_before_the_end():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer(), latency_s=2.5)
    tracer.flush()

    body = handler.batches[0][1]["body"]
    assert body["startTime"] < body["endTime"]


def test_release_and_environment_are_carried_when_configured():
    handler = Recorder()
    tracer = make(handler, release="v0.1.0", environment="dev")
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    for event in handler.batches[0]:
        assert event["body"]["release"] == "v0.1.0"
        assert event["body"]["environment"] == "dev"


def test_the_url_and_basic_auth_header_follow_the_documented_api():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    request = handler.requests[0]
    assert str(request.url) == HOST + INGESTION_PATH
    token = request.headers["authorization"].removeprefix("Basic ")
    assert base64.b64decode(token).decode() == f"{PUBLIC}:{SECRET}"


def test_a_trailing_slash_on_the_host_does_not_double_the_path():
    handler = Recorder()
    tracer = LangfuseTracer(
        PUBLIC,
        SECRET,
        HOST + "/",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    assert str(handler.requests[0].url) == HOST + INGESTION_PATH


# --- batching and flushing -------------------------------------------------


def test_nothing_is_posted_until_a_flush():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())

    assert handler.requests == []


def test_a_flush_with_an_empty_batch_posts_nothing():
    handler = Recorder()
    make(handler).flush()

    assert handler.requests == []


def test_a_full_batch_flushes_itself():
    handler = Recorder()
    tracer = make(handler)
    for _ in range(MAX_BATCH_EVENTS // 2):
        tracer.generation("llm", ASK, completion=answer())

    assert len(handler.requests) == 1
    assert len(handler.batches[0]) == MAX_BATCH_EVENTS


def test_closing_flushes_what_is_left():
    handler = Recorder()
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.close()

    assert len(handler.batches) == 1


def test_a_borrowed_client_survives_close():
    handler = Recorder()
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    tracer = LangfuseTracer(PUBLIC, SECRET, HOST, http_client=http)
    tracer.close()

    assert not http.is_closed


# --- failure is always a drop ----------------------------------------------


def test_an_unreachable_langfuse_drops_the_batch_and_warns(captured):
    handler = Recorder(httpx2.ConnectError("connection refused"))
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    assert tracer.dropped == 2
    assert any("dropped 2 trace events" in message for message in warnings(captured))


def test_a_rejected_batch_is_dropped_and_warned(captured):
    handler = Recorder(httpx2.Response(401, json={"message": "unauthorised"}))
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    assert tracer.dropped == 2
    assert any("returned 401" in message for message in warnings(captured))


def test_a_dropped_batch_is_not_requeued():
    handler = Recorder(httpx2.ConnectError("connection refused"))
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()
    tracer.flush()

    assert len(handler.requests) == 1


def test_partial_rejection_is_counted_per_event(captured):
    handler = Recorder(
        httpx2.Response(
            207,
            json={"successes": [], "errors": [{"id": "abc", "message": "bad"}]},
        )
    )
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    assert tracer.dropped == 1
    assert any("rejected trace event abc" in m for m in warnings(captured))


def test_a_body_that_is_not_json_is_not_a_failure(captured):
    handler = Recorder(httpx2.Response(200, text="ok"))
    tracer = make(handler)
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()

    assert tracer.dropped == 0
    assert warnings(captured) == []


def test_the_trace_payload_is_never_logged(captured):
    handler = Recorder(httpx2.ConnectError("connection refused"))
    tracer = make(handler)
    tracer.generation(
        "llm", [Message(role="user", content=f"my PAN is {PAN}")], completion=answer()
    )
    tracer.flush()

    assert captured.records, "the capture must be live or this asserts nothing"
    for record in captured.records:
        assert PAN not in record.getMessage()
        assert "standard deduction" not in record.getMessage()


# --- the unconfigured process ----------------------------------------------


def test_no_langfuse_configured_yields_a_null_tracer(captured):
    tracer = LangfuseTracer.from_settings(Settings(_env_file=None))

    assert isinstance(tracer, NullTracer)
    assert any("not traced" in message for message in warnings(captured))


def test_partial_langfuse_configuration_yields_a_null_tracer():
    settings = Settings(_env_file=None, langfuse_public_key=PUBLIC)

    assert isinstance(LangfuseTracer.from_settings(settings), NullTracer)


def test_full_configuration_yields_a_real_tracer():
    settings = Settings(
        _env_file=None,
        langfuse_public_key=PUBLIC,
        langfuse_secret_key=SECRET,
        langfuse_host=HOST,
    )
    tracer = LangfuseTracer.from_settings(settings)

    assert isinstance(tracer, LangfuseTracer)
    tracer.close()


def test_a_tracer_refuses_to_construct_without_a_host():
    with pytest.raises(ValueError, match="host"):
        LangfuseTracer(PUBLIC, SECRET, "")


def test_both_tracers_satisfy_the_protocol():
    assert isinstance(NullTracer(), Tracer)
    assert isinstance(make(Recorder()), Tracer)


def test_the_null_tracer_does_nothing_at_all():
    tracer = NullTracer()
    tracer.generation("llm", ASK, completion=answer())
    tracer.flush()
    tracer.close()


# --- the wrapper -----------------------------------------------------------


def test_the_wrapper_passes_every_argument_through():
    inner = Fake()
    client = TracedLLMClient(inner, NullTracer())
    client.complete(
        ASK,
        max_completion_tokens=64,
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    call = inner.calls[0]
    assert call["max_completion_tokens"] == 64
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0


def test_the_wrapper_returns_the_inner_completion_unchanged():
    inner = Fake(answer("thirty per cent"))
    client = TracedLLMClient(inner, NullTracer())

    assert client.complete(ASK) == inner.completion


def test_the_wrapper_traces_a_failure_and_re_raises_it():
    handler = Recorder()
    tracer = make(handler)
    client = TracedLLMClient(Failing(), tracer)

    with pytest.raises(LLMUnavailable):
        client.complete(ASK)
    tracer.flush()

    body = handler.batches[0][1]["body"]
    assert body["level"] == "ERROR"
    assert "LLMUnavailable" in body["statusMessage"]


def test_a_broken_tracer_never_breaks_a_call(captured):
    client = TracedLLMClient(Fake(), Exploding())

    assert client.complete(ASK).text == "thirty per cent"
    assert any("could not trace" in message for message in warnings(captured))


def test_a_broken_tracer_does_not_swallow_the_inner_error():
    client = TracedLLMClient(Failing(), Exploding())

    with pytest.raises(LLMUnavailable):
        client.complete(ASK)


def test_the_wrapper_records_the_parameters_it_was_called_with():
    handler = Recorder()
    tracer = make(handler)
    client = TracedLLMClient(Fake(), tracer)
    client.complete(ASK, max_completion_tokens=64, temperature=0.0)
    tracer.flush()

    parameters = handler.batches[0][1]["body"]["modelParameters"]
    assert parameters["max_completion_tokens"] == 64
    assert parameters["temperature"] == 0.0
