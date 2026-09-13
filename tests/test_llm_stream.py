"""Step 10.3 — the streaming LLM call, its cache replay and its trace.

All on a scripted MockTransport: no network, no key.
"""

from __future__ import annotations

import json

import httpx2
import pytest

from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import (
    GEMINI,
    GROQ,
    LLMClient,
    LLMRequestError,
    LLMUnavailable,
    Message,
)
from taxverity.llm.tracing import TracedLLMClient
from taxverity.observability import PAN_MASK

KEY = "test-key-not-real"
PAN = "ABCDE1234F"
ASK = [Message(role="user", content=f"My PAN is {PAN}. What does section 22(1)(a) allow?")]


def sse(*deltas: str, usage: bool = True, done: bool = True, model: str = GROQ.model) -> bytes:
    lines = []
    for delta in deltas:
        chunk = {"model": model, "choices": [{"delta": {"content": delta}, "finish_reason": None}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    final = {"model": model, "choices": [{"delta": {}, "finish_reason": "stop"}]}
    if usage:
        final["x_groq"] = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
            }
        }
    lines.append(f"data: {json.dumps(final)}\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        scripted = self.responses.pop(0)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


def client(recorder: Recorder, *, fallback: bool = False) -> LLMClient:
    return LLMClient(
        GROQ,
        KEY,
        fallback=GEMINI if fallback else None,
        fallback_key="fallback-key-not-real" if fallback else None,
        http_client=httpx2.Client(transport=httpx2.MockTransport(recorder)),
        backoff_base=0.0,
    )


def ok(*deltas: str) -> httpx2.Response:
    return httpx2.Response(200, content=sse(*deltas))


def test_deltas_arrive_in_order_and_the_completion_is_recorded():
    recorder = Recorder(ok('{"type": "sta', 'tute"}\n', '{"type": "computation"}'))
    llm = client(recorder)
    stream = llm.stream(ASK)
    assert stream.completion is None
    assert list(stream) == ['{"type": "sta', 'tute"}\n', '{"type": "computation"}']
    assert stream.completion.text == '{"type": "statute"}\n{"type": "computation"}'
    assert stream.completion.finish_reason == "stop"
    assert stream.completion.usage.reasoning_tokens == 5
    assert stream.completion.degraded is False
    assert llm.tokens_used == {"groq": 120}


def test_constructing_a_stream_sends_nothing():
    recorder = Recorder()
    client(recorder).stream(ASK)
    assert recorder.requests == []


def test_the_request_is_redacted_and_asks_for_a_stream():
    recorder = Recorder(ok("x"))
    list(client(recorder).stream(ASK, temperature=0.0))
    body = json.loads(recorder.requests[0].content)
    assert body["stream"] is True
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "low"
    sent = recorder.requests[0].content.decode()
    assert PAN not in sent
    assert PAN_MASK in sent
    assert "22(1)(a)" in sent


def test_a_retryable_status_before_the_first_token_is_retried():
    recorder = Recorder(httpx2.Response(503), ok("fine"))
    assert list(client(recorder).stream(ASK)) == ["fine"]
    assert len(recorder.requests) == 2


def test_a_malformed_chunk_before_the_first_token_is_retried():
    recorder = Recorder(httpx2.Response(200, content=b"data: {not json\n\n"), ok("fine"))
    assert list(client(recorder).stream(ASK)) == ["fine"]


def test_the_fallback_answers_when_the_primary_never_starts():
    recorder = Recorder(
        httpx2.Response(503),
        httpx2.Response(503),
        httpx2.Response(503),
        httpx2.Response(200, content=sse("from gemini", model=GEMINI.model)),
    )
    stream = client(recorder, fallback=True).stream(ASK)
    assert list(stream) == ["from gemini"]
    assert stream.completion.degraded is True
    assert stream.completion.provider == "gemini"
    assert GEMINI.base_url in str(recorder.requests[-1].url)


def test_a_refused_request_neither_retries_nor_fails_over():
    recorder = Recorder(httpx2.Response(400, text="bad request"))
    with pytest.raises(LLMRequestError):
        list(client(recorder, fallback=True).stream(ASK))
    assert len(recorder.requests) == 1


def test_every_attempt_failing_raises_unavailable():
    recorder = Recorder(*(httpx2.ConnectError("down") for _ in range(3)))
    with pytest.raises(LLMUnavailable):
        list(client(recorder).stream(ASK))


def test_a_failure_after_the_first_token_raises_without_a_retry():
    def body():
        yield sse("first claim\n", usage=False, done=False)
        raise httpx2.ReadError("connection reset")

    recorder = Recorder(httpx2.Response(200, content=body()), ok("never sent"))
    stream = client(recorder, fallback=True).stream(ASK)
    received = []
    with pytest.raises(LLMUnavailable, match="after the first token"):
        for delta in stream:
            received.append(delta)
    assert received == ["first claim\n"]
    assert len(recorder.requests) == 1
    assert stream.completion is None


def test_a_stream_that_ends_without_done_still_completes():
    recorder = Recorder(httpx2.Response(200, content=sse("a", "b", done=False)))
    stream = client(recorder).stream(ASK)
    assert "".join(stream) == "ab"
    assert stream.completion.text == "ab"


def test_a_stream_can_be_read_only_once():
    stream = client(Recorder(ok("a"))).stream(ASK)
    list(stream)
    with pytest.raises(RuntimeError):
        list(stream)


def test_a_cached_stream_replays_without_a_request(tmp_path):
    recorder = Recorder(ok('{"a": 1}\n', '{"b": 2}'))
    cached = CachedLLMClient(client(recorder), tmp_path)
    first = "".join(cached.stream(ASK))
    second = "".join(cached.stream(ASK))
    assert first == second == '{"a": 1}\n{"b": 2}'
    assert len(recorder.requests) == 1
    assert (cached.hits, cached.misses) == (1, 1)
    assert PAN not in "".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir())


def test_an_interrupted_stream_is_not_cached(tmp_path):
    def body():
        yield sse("partial\n", usage=False, done=False)
        raise httpx2.ReadError("reset")

    recorder = Recorder(httpx2.Response(200, content=body()))
    cached = CachedLLMClient(client(recorder), tmp_path)
    with pytest.raises(LLMUnavailable):
        list(cached.stream(ASK))
    assert list(tmp_path.iterdir()) == []


class SpyTracer:
    def __init__(self):
        self.calls = []

    def generation(self, name, messages, **kwargs):
        self.calls.append(kwargs)

    def flush(self):
        pass

    def close(self):
        pass


def test_a_stream_is_traced_once_when_it_ends():
    tracer = SpyTracer()
    traced = TracedLLMClient(client(Recorder(ok("a", "b"))), tracer)
    stream = traced.stream(ASK)
    assert tracer.calls == []
    assert "".join(stream) == "ab"
    assert len(tracer.calls) == 1
    assert tracer.calls[0]["completion"].text == "ab"
    assert tracer.calls[0]["model_parameters"]["stream"] is True


def test_a_failed_stream_is_traced_as_an_error():
    tracer = SpyTracer()
    traced = TracedLLMClient(client(Recorder(httpx2.Response(400))), tracer)
    with pytest.raises(LLMRequestError):
        list(traced.stream(ASK))
    assert "LLMRequestError" in tracer.calls[0]["error"]
