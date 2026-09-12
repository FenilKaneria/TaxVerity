"""Step 7.3 — the on-disk response cache.

No network: every test drives the Step 7.2 client over a scripted
MockTransport, so a hit is provably a hit — the handler records every request
that actually left.
"""

from __future__ import annotations

import json
import logging

import httpx2
import pytest

from taxverity.config import Settings
from taxverity.llm.cache import LLM_CACHE_VERSION, CachedLLMClient
from taxverity.llm.client import (
    GEMINI,
    GROQ,
    LLMClient,
    LLMUnavailable,
    Message,
    Provider,
)
from taxverity.observability import PAN_MASK

LOGGER = "taxverity.llm.cache"
KEY = "test-key-not-real"
PAN = "ABCDE1234F"
ASK = [Message(role="user", content="What is the standard deduction?")]


def ok(text: str = "thirty per cent", *, model: str = GROQ.model) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 12},
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
            scripted = self.responses.pop(0)
            if isinstance(scripted, Exception):
                raise scripted
            return scripted
        return ok()

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]


def make(handler: Recorder, directory, *, provider=GROQ, **kwargs) -> CachedLLMClient:
    kwargs.setdefault("backoff_base", 0.0)
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    inner = LLMClient(provider, KEY, http_client=http, **kwargs)
    return CachedLLMClient(inner, directory)


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured():
    # Below `propagate = False` and below RedactingFilter, so an assertion is
    # about what the call site passed (the Step 4.2 caplog trap).
    handler = CaptureHandler()
    logger = logging.getLogger(LOGGER)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "llm"


def messages(handler: CaptureHandler) -> list[str]:
    return [record.getMessage() for record in handler.records]


# --- the cache does its job ------------------------------------------------


def test_a_first_call_reaches_the_provider_and_is_stored(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)

    completion = client.complete(ASK)

    assert completion.text == "thirty per cent"
    assert len(handler.requests) == 1
    assert (client.hits, client.misses) == (0, 1)
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_a_repeated_call_never_reaches_the_provider(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)

    first = client.complete(ASK)
    second = client.complete(ASK)

    assert second == first
    assert len(handler.requests) == 1
    assert (client.hits, client.misses) == (1, 1)


def test_a_hit_bills_no_tokens(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)
    client.complete(ASK)
    billed = dict(client._inner.tokens_used)

    client.complete(ASK)

    assert client._inner.tokens_used == billed


def test_a_second_process_reads_what_the_first_stored(cache_dir):
    first_handler = Recorder()
    make(first_handler, cache_dir).complete(ASK)

    second_handler = Recorder()
    second = make(second_handler, cache_dir)
    completion = second.complete(ASK)

    assert completion.text == "thirty per cent"
    assert second_handler.requests == []
    assert (second.hits, second.misses) == (1, 0)


def test_usage_and_finish_reason_survive_the_round_trip(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)
    live = client.complete(ASK)

    replayed = make(Recorder(), cache_dir).complete(ASK)

    assert replayed.usage == live.usage
    assert replayed.usage.reasoning_tokens == 12
    assert replayed.finish_reason == live.finish_reason
    assert replayed.model == live.model


def test_a_degraded_completion_is_stored_with_its_flag(cache_dir):
    handler = Recorder(httpx2.Response(503), ok(model=GEMINI.model))
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    inner = LLMClient(
        GROQ,
        KEY,
        fallback=GEMINI,
        fallback_key=KEY,
        http_client=http,
        max_attempts=1,
        backoff_base=0.0,
    )
    client = CachedLLMClient(inner, cache_dir)

    live = client.complete(ASK)
    replayed = make(Recorder(), cache_dir).complete(ASK)

    # The flag is what tells a later reader the row was measured on the
    # fallback. Dropping the entry instead would hide that it ever happened.
    assert live.degraded is True
    assert replayed.degraded is True
    assert replayed.provider == GEMINI.name


# --- what the key covers ---------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_completion_tokens": 999},
        {"temperature": 0.2},
        {"response_format": {"type": "json_object"}},
    ],
)
def test_a_changed_request_field_misses(cache_dir, kwargs):
    handler = Recorder()
    client = make(handler, cache_dir)

    client.complete(ASK)
    client.complete(ASK, **kwargs)

    assert len(handler.requests) == 2
    assert client.misses == 2


def test_changed_message_text_misses(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)

    client.complete(ASK)
    client.complete([Message(role="user", content="What about HRA?")])

    assert len(handler.requests) == 2


def test_a_changed_role_misses(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)

    client.complete([Message(role="user", content="hello")])
    client.complete([Message(role="system", content="hello")])

    assert len(handler.requests) == 2


def test_message_order_is_part_of_the_key(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)
    one = Message(role="system", content="be brief")
    two = Message(role="user", content="hello")

    client.complete([one, two])
    client.complete([two, one])

    assert len(handler.requests) == 2


def test_another_provider_does_not_reuse_the_answer(cache_dir):
    make(Recorder(), cache_dir).complete(ASK)

    other = make(Recorder(), cache_dir, provider=GEMINI)
    other.complete(ASK)

    assert other.misses == 1


def test_a_changed_provider_extra_misses(cache_dir):
    make(Recorder(), cache_dir).complete(ASK)
    louder = Provider(
        name=GROQ.name,
        base_url=GROQ.base_url,
        model=GROQ.model,
        settings_key=GROQ.settings_key,
        extras={"reasoning_effort": "high"},
    )

    client = make(Recorder(), cache_dir, provider=louder)
    client.complete(ASK)

    assert client.misses == 1


def test_the_cache_version_is_in_the_key(cache_dir, monkeypatch):
    make(Recorder(), cache_dir).complete(ASK)
    monkeypatch.setattr("taxverity.llm.cache.LLM_CACHE_VERSION", LLM_CACHE_VERSION + 1)

    client = make(Recorder(), cache_dir)
    client.complete(ASK)

    assert client.misses == 1


# --- redaction (rule 03) ---------------------------------------------------


def stored_text(directory) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in directory.glob("*.json"))


def test_a_pan_never_reaches_the_disk(cache_dir):
    client = make(Recorder(), cache_dir)

    client.complete([Message(role="user", content=f"My PAN is {PAN}")])

    assert PAN not in stored_text(cache_dir)
    assert PAN_MASK in stored_text(cache_dir)


def test_two_pans_share_one_entry_because_both_are_masked(cache_dir):
    handler = Recorder()
    client = make(handler, cache_dir)

    client.complete([Message(role="user", content=f"My PAN is {PAN}")])
    client.complete([Message(role="user", content="My PAN is ZZZZZ9999Z")])

    # Not a collision: the wire carries the mask in both cases, so it genuinely
    # is one request. The key describes what is sent, not what was typed.
    assert len(handler.requests) == 1
    assert handler.bodies[0]["messages"][0]["content"].endswith(PAN_MASK)


def test_statutory_text_crosses_the_key_unchanged(cache_dir):
    client = make(Recorder(), cache_dir)

    client.complete([Message(role="user", content="section 80C and 2(5)(b)(ii)")])

    assert "section 80C" in stored_text(cache_dir)
    assert "2(5)(b)(ii)" in stored_text(cache_dir)


# --- failure, corruption, and never breaking a run -------------------------


def test_a_failure_is_not_stored(cache_dir):
    handler = Recorder(httpx2.Response(503))
    client = make(handler, cache_dir, max_attempts=1)

    with pytest.raises(LLMUnavailable):
        client.complete(ASK)

    assert list(cache_dir.glob("*.json")) == []

    # And the next call retries rather than replaying the failure.
    assert client.complete(ASK).text == "thirty per cent"


def test_a_corrupt_entry_is_a_miss_not_a_crash(cache_dir, captured):
    make(Recorder(), cache_dir).complete(ASK)
    entry = next(cache_dir.glob("*.json"))
    entry.write_text("{not json", encoding="utf-8")

    handler = Recorder()
    client = make(handler, cache_dir)
    completion = client.complete(ASK)

    assert completion.text == "thirty per cent"
    assert len(handler.requests) == 1
    assert any("unreadable llm cache entry" in m for m in messages(captured))


def test_a_corrupt_entry_is_replaced_by_the_fresh_answer(cache_dir):
    make(Recorder(), cache_dir).complete(ASK)
    entry = next(cache_dir.glob("*.json"))
    entry.write_text("{not json", encoding="utf-8")
    make(Recorder(), cache_dir).complete(ASK)

    handler = Recorder()
    make(handler, cache_dir).complete(ASK)

    assert handler.requests == []


def test_an_entry_that_does_not_match_its_key_is_refused(cache_dir, captured):
    make(Recorder(), cache_dir).complete(ASK)
    entry = next(cache_dir.glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["request"]["messages"][0]["content"] = "a different question"
    entry.write_text(json.dumps(payload), encoding="utf-8")

    handler = Recorder()
    client = make(handler, cache_dir)
    client.complete(ASK)

    assert len(handler.requests) == 1
    assert any("does not match its key" in m for m in messages(captured))


def test_a_malformed_completion_is_a_miss(cache_dir, captured):
    make(Recorder(), cache_dir).complete(ASK)
    entry = next(cache_dir.glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["completion"] = {"text": "only this"}
    entry.write_text(json.dumps(payload), encoding="utf-8")

    handler = Recorder()
    make(handler, cache_dir).complete(ASK)

    assert len(handler.requests) == 1
    assert any("malformed llm cache entry" in m for m in messages(captured))


def test_no_temporary_file_is_left_behind(cache_dir):
    make(Recorder(), cache_dir).complete(ASK)

    assert list(cache_dir.glob("*.tmp")) == []
    assert len(list(cache_dir.iterdir())) == 1


def test_the_directory_is_created_on_first_write(tmp_path):
    directory = tmp_path / "missing" / "llm"
    client = make(Recorder(), directory)

    assert not directory.exists()
    client.complete(ASK)
    assert directory.is_dir()


def test_a_missing_directory_is_a_miss_not_an_error(tmp_path):
    client = make(Recorder(), tmp_path / "absent")

    assert client.complete(ASK).text == "thirty per cent"
    assert (client.hits, client.misses) == (0, 1)


# --- wiring ----------------------------------------------------------------


def test_settings_puts_the_cache_under_the_data_directory(tmp_path):
    settings = Settings(data_dir=tmp_path)

    assert settings.llm_cache_dir == tmp_path / "llm"


def test_the_client_exposes_the_provider_the_key_covers():
    http = httpx2.Client(transport=httpx2.MockTransport(Recorder()))

    assert LLMClient(GROQ, KEY, http_client=http).primary is GROQ
