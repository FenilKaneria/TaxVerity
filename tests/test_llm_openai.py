"""R23 (ADR-129) — OpenAI as an LLM vendor: the request shape GPT-5 models
accept, and the generation switch in `graph/build.generation_providers`. Scripted
transport only; no network, no key."""

from __future__ import annotations

import httpx2

from taxverity.config import Settings
from taxverity.graph.build import generation_providers
from taxverity.llm.client import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    GEMINI,
    GROQ,
    OPENAI_MINI,
    LLMClient,
)
from test_llm_client import ASK, FALLBACK_KEY, KEY, PAN, Recorder, ok


def _openai(handler: Recorder, *, with_fallback: bool = False) -> LLMClient:
    return LLMClient(
        OPENAI_MINI,
        KEY,
        fallback=GROQ if with_fallback else None,
        fallback_key=FALLBACK_KEY if with_fallback else None,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        backoff_base=0.0,
    )


def test_gpt5_gets_no_temperature_and_its_own_reasoning_effort():
    handler = Recorder(ok(model=OPENAI_MINI.model))
    _openai(handler).complete(ASK, max_completion_tokens=700, temperature=0.0)
    (body,) = handler.bodies
    assert "temperature" not in body
    assert body["max_completion_tokens"] == 700
    assert body["model"] == "gpt-5-mini"
    # "low" spent whole caps on hidden reasoning (R23 smoke run).
    assert body["reasoning_effort"] == "minimal"
    assert handler.urls == ["https://api.openai.com/v1/chat/completions"]


def test_the_groq_fallback_still_gets_its_temperature():
    """`omit` is per provider: Groq answering for OpenAI keeps the caller's
    temperature and its own `reasoning_effort`."""
    handler = Recorder(*[httpx2.Response(503)] * 3, ok())
    completion = _openai(handler, with_fallback=True).complete(ASK, temperature=0.0)
    assert completion.provider == "groq"
    groq_body = handler.bodies[-1]
    assert groq_body["temperature"] == 0.0
    assert groq_body["max_completion_tokens"] == DEFAULT_MAX_COMPLETION_TOKENS
    assert groq_body["reasoning_effort"] == "low"


def test_a_pan_never_reaches_openai():
    handler = Recorder(ok(model=OPENAI_MINI.model))
    _openai(handler).complete(
        [ASK[0].model_copy(update={"content": f"My PAN is {PAN}."})]
    )
    assert PAN not in handler.requests[0].read().decode()


def test_generation_runs_on_openai_with_groq_behind_it_by_default():
    assert generation_providers(Settings(_env_file=None)) == (OPENAI_MINI, GROQ)


def test_generation_can_be_switched_back_to_groq():
    settings = Settings(_env_file=None, generation_llm="groq")
    assert generation_providers(settings) == (GROQ, GEMINI)
