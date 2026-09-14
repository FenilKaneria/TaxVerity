"""Advisor pivot, Step 5 — the conversational route's guarded LLM reply.

No network: the request assertions drive the real Step 7.2 client over a
scripted `MockTransport`, the same discipline `test_classifier.py` and
`test_extraction.py` use. `_looks_statutory` is tested directly since it is
the deterministic, externally verifiable check rule 01 requires alongside
any LLM step - the injection scenarios (system prompt untouched, a leaking
reply falls back) live in `test_injection.py`.
"""

from __future__ import annotations

import json

import httpx2

from taxverity.llm.client import GROQ, LLMClient, LLMUnavailable
from taxverity.llm.conversational import (
    CONVERSATIONAL_FALLBACK,
    CONVERSATIONAL_STAGE_VERSION,
    Conversationalist,
    _looks_statutory,
)

KEY = "test-key-not-real"


def ok(text: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": GROQ.model,
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    )


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]


def build(*responses):
    recorder = Recorder(*responses)
    client = LLMClient(
        GROQ, KEY, http_client=httpx2.Client(transport=httpx2.MockTransport(recorder))
    )
    return Conversationalist(client), recorder


def test_stage_version_is_declared():
    assert CONVERSATIONAL_STAGE_VERSION == 1


def test_reply_sends_the_fixed_system_prompt_and_a_delimited_user_message():
    node, recorder = build(ok("Hi! Ask me about the Act."))
    reply = node.reply("hi there")
    assert reply == "Hi! Ask me about the Act."
    body = recorder.bodies[0]
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"] == "<message>\nhi there\n</message>"
    assert body["temperature"] == 0.0


def test_a_provider_failure_falls_back_to_the_fixed_template():
    class Boom:
        def complete(self, messages, **kwargs):
            raise LLMUnavailable("no provider available")

    node = Conversationalist(Boom())
    assert node.reply("hi") == CONVERSATIONAL_FALLBACK


def test_a_statutory_looking_reply_falls_back_to_the_fixed_template():
    node, _ = build(ok("Section 19 gives you a standard deduction."))
    assert node.reply("what can you do?") == CONVERSATIONAL_FALLBACK


# --- _looks_statutory ---------------------------------------------------------


def test_a_plain_reply_passes():
    assert _looks_statutory("Hi! Ask me about the Income-tax Act.") is False


def test_empty_text_is_rejected():
    assert _looks_statutory("") is True


def test_a_number_is_rejected():
    assert _looks_statutory("You could save up to 50,000 this year.") is True


def test_a_statutory_word_is_rejected():
    assert _looks_statutory("I can tell you about a deduction.") is True


def test_a_citation_shaped_token_is_rejected():
    assert _looks_statutory("As set out in 22(2), see that instead.") is True


def test_an_overlong_reply_is_rejected():
    assert _looks_statutory(" ".join(["word"] * 61)) is True
