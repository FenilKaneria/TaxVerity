"""Step 11.7 — query contextualization, and Step 11.6's recent-turns window
feeding it.

No network: every LLM-calling test drives the real Step 7.2 client over a
scripted MockTransport, the same discipline `test_extraction.py` uses, so what
leaves the process — the system prompt, the temperature, the prior turns — is
asserted on the bytes that actually left.
"""

from __future__ import annotations

import json
import logging

import httpx2
import pytest

from taxverity.llm.client import GROQ, LLMClient
from taxverity.memory.contextualize import (
    CONTEXTUALIZE_MAX_COMPLETION_TOKENS,
    CONTEXTUALIZE_TEMPERATURE,
    SYSTEM_PROMPT,
    QueryContextualizer,
    needs_contextualization,
)
from taxverity.threads.store import append_message, create_thread, list_messages

KEY = "test-key-not-real"


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
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
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
        return ok("")

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]


def build(*responses):
    recorder = Recorder(*responses)
    client = LLMClient(
        GROQ,
        KEY,
        http_client=httpx2.Client(transport=httpx2.MockTransport(recorder)),
    )
    return QueryContextualizer(client), recorder


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(record.getMessage() for record in self.records)


# --- the deterministic skip -------------------------------------------------


def test_no_prior_turns_never_needs_contextualization():
    assert needs_contextualization("What if I'm 65 instead?", []) is False


@pytest.mark.parametrize(
    "query",
    [
        "What is section 80C?",
        "How much can I claim for medical insurance?",
    ],
)
def test_a_standalone_query_needs_no_rewrite(query):
    assert needs_contextualization(query, ["a prior turn"]) is False


@pytest.mark.parametrize(
    "query",
    [
        "What if it's higher instead?",
        "What about my mother?",
        "Can I claim that too?",
        "Does the same limit apply here?",
    ],
)
def test_a_follow_up_marker_with_history_needs_a_rewrite(query):
    assert needs_contextualization(query, ["a prior turn"]) is True


# --- the skip never calls the model -----------------------------------------


def test_a_standalone_query_never_reaches_the_model():
    contextualizer, recorder = build()
    result = contextualizer.contextualize("What is section 80C?", ["a prior turn"])
    assert result.query == "What is section 80C?"
    assert result.rewritten is False
    assert result.completion is None
    assert recorder.requests == []


def test_no_prior_turns_never_reaches_the_model():
    contextualizer, recorder = build()
    result = contextualizer.contextualize("What if it's higher instead?", [])
    assert result.rewritten is False
    assert recorder.requests == []


# --- the rewrite call --------------------------------------------------------


def test_a_follow_up_is_rewritten_from_the_prior_turn():
    contextualizer, recorder = build(
        ok("Under section 22(2), what if the property is not self-occupied?")
    )
    result = contextualizer.contextualize(
        "What if it's not self-occupied instead?",
        ["Under section 22(2), what is the maximum interest deduction for a self-occupied property?"],
    )
    assert result.rewritten is True
    assert result.query == "Under section 22(2), what if the property is not self-occupied?"
    assert result.completion is not None
    assert result.tokens == 100

    (body,) = recorder.bodies
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    user_message = body["messages"][1]["content"]
    assert "self-occupied property" in user_message
    assert "What if it's not self-occupied instead?" in user_message
    assert body["max_completion_tokens"] == CONTEXTUALIZE_MAX_COMPLETION_TOKENS
    assert body["temperature"] == CONTEXTUALIZE_TEMPERATURE


def test_an_empty_rewrite_falls_back_to_the_original_query(caplog):
    handler = CaptureHandler()
    logger = logging.getLogger("taxverity.memory.contextualize")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        contextualizer, _ = build(ok(""))
        result = contextualizer.contextualize(
            "What if it's higher instead?", ["a prior turn"]
        )
    finally:
        logger.removeHandler(handler)

    assert result.rewritten is False
    assert result.query == "What if it's higher instead?"
    assert result.completion is not None
    assert "no text" in handler.text
    assert "higher" not in handler.text


# --- Step 11.6: the recent-turns window feeds this, never fact state -------


def test_the_recent_turns_window_is_what_the_rewrite_prompt_carries(schema):
    from conftest import register_account

    user_id = register_account(schema, "carol@example.com", "correct horse battery")
    thread = create_thread(schema, user_id, "House property")
    append_message(
        schema, user_id, thread.thread_id, "user",
        "Under section 22(2), what is the maximum interest deduction for a "
        "self-occupied property?",
    )  # fmt: skip
    append_message(
        schema, user_id, thread.thread_id, "assistant",
        "Rs. 2,00,000, subject to conditions.",
    )  # fmt: skip

    recent = list_messages(schema, user_id, thread.thread_id, last=3)
    prior_turns = [message.content for message in recent]

    contextualizer, recorder = build(ok("What if the property is not self-occupied?"))
    result = contextualizer.contextualize(
        "What if it's not self-occupied instead?", prior_turns
    )

    assert result.rewritten is True
    (body,) = recorder.bodies
    user_message = body["messages"][1]["content"]
    assert "self-occupied property" in user_message
    assert "Rs. 2,00,000" in user_message


# --- R21: follow-ups that name no pronoun, and the previous answer ------------


@pytest.mark.parametrize(
    "query",
    [
        "Give some examples about the loss",
        "Explain in simple terms and not legally heavy language",
        "Can you elaborate on the carry forward part?",
    ],
)
def test_a_style_or_elaboration_follow_up_needs_a_rewrite(query):
    assert needs_contextualization(query, ["a prior turn"]) is True


def test_the_previous_answer_reaches_the_rewrite_prompt_bounded():
    contextualizer, recorder = build(ok("Give examples of setting off a house-property loss."))
    long_answer = "Loss from house property can be set off against salary. " + "x" * 2_000
    contextualizer.contextualize(
        "Give some examples about the loss",
        ["Set off house-property loss against salary"],
        previous_answer=long_answer,
    )
    (body,) = recorder.bodies
    user_message = body["messages"][1]["content"]
    assert "Previous answer:\nLoss from house property can be set off" in user_message
    assert len(user_message) < 1_000
    # The previous answer is data in the user role, never the system role.
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
