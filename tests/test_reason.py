"""R20 Step 20.5 — the `reason` call: strict schema first, `json_object`
fallback (Step 7.1's measured pattern, same as `IntentClassifier`/
`FactExtractor`), and a malformed completion degrades to `analysis=None`
rather than raising — a reasoning failure must never stop a turn from
answering.

No network: drives the real `LLMClient` over a scripted `MockTransport`,
same discipline as `test_extraction.py` and `test_classifier.py`.
"""

from __future__ import annotations

import json

import httpx2

from taxverity.facts import UserFacts
from taxverity.llm.client import GROQ, LLMClient
from taxverity.memory.fact_state import ThreadFactState
from taxverity.reasoning.models import ConclusionKind
from taxverity.reasoning.reason import (
    OBJECT_FORMAT,
    REASON_STAGE_VERSION,
    STRICT_FORMAT,
    Reasoner,
)
from test_verifier import PACK

KEY = "test-key-not-real"
QUESTION = "Can I deduct interest on my house property loan?"
EMPTY_STATE = ThreadFactState()


def good_payload() -> dict:
    return {
        "legal_rules": [
            {
                "id": "r1",
                "markers": [2],
                "rule": "The deduction is capped at Rs. 2,00,000.",
                "conditions": [],
                "limits": [],
                "exceptions": [],
                "definitions": [],
            }
        ],
        "applicability": [],
        "missing_facts": [],
        "answer_plan": {
            "conclusion_kind": "determined",
            "steps": ["The cap applies."],
            "next_step": "",
        },
    }


def ok(payload: dict) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": GROQ.model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 90,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        },
    )


def refusal() -> httpx2.Response:
    return httpx2.Response(400, text="response_format json_schema is not supported")


class Recorder:
    def __init__(self, *responses: httpx2.Response):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return ok(good_payload())

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]


def build(*responses: httpx2.Response) -> tuple[Reasoner, Recorder]:
    recorder = Recorder(*responses)
    client = LLMClient(
        GROQ, KEY, http_client=httpx2.Client(transport=httpx2.MockTransport(recorder))
    )
    return Reasoner(client), recorder


def test_stage_version_is_declared():
    assert REASON_STAGE_VERSION == 1


def test_a_clean_completion_parses_into_legal_rules():
    node, recorder = build(ok(good_payload()))
    result = node.reason(QUESTION, PACK, EMPTY_STATE, None)
    assert recorder.bodies[0]["response_format"] == STRICT_FORMAT
    assert result.analysis is not None
    (rule,) = result.analysis.legal_rules
    assert rule.id == "r1"
    assert result.analysis.answer_plan.conclusion_kind is ConclusionKind.DETERMINED


def test_a_malformed_completion_degrades_to_no_analysis_not_an_exception():
    node, recorder = build(
        httpx2.Response(
            200,
            json={
                "model": GROQ.model,
                "choices": [
                    {"message": {"role": "assistant", "content": "not json at all"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    )
    result = node.reason(QUESTION, PACK, EMPTY_STATE, None)
    assert result.analysis is None


def test_a_payload_missing_answer_plan_degrades_to_no_analysis():
    payload = good_payload()
    del payload["answer_plan"]
    node, recorder = build(ok(payload))
    result = node.reason(QUESTION, PACK, EMPTY_STATE, None)
    assert result.analysis is None


def test_a_provider_refusing_the_schema_falls_back_to_json_object():
    node, recorder = build(refusal(), ok(good_payload()))
    result = node.reason(QUESTION, PACK, EMPTY_STATE, None)
    assert recorder.bodies[0]["response_format"] == STRICT_FORMAT
    assert recorder.bodies[1]["response_format"] == OBJECT_FORMAT
    assert result.analysis is not None


def test_the_refusal_is_paid_once_per_process():
    node, recorder = build(refusal(), ok(good_payload()), ok(good_payload()))
    node.reason(QUESTION, PACK, EMPTY_STATE, None)
    node.reason(QUESTION, PACK, EMPTY_STATE, None)
    assert node.schema_refused is True
    assert [b["response_format"] for b in recorder.bodies] == [
        STRICT_FORMAT,
        OBJECT_FORMAT,
        OBJECT_FORMAT,
    ]


def test_the_prompt_carries_the_question_and_numbered_evidence():
    node, recorder = build(ok(good_payload()))
    node.reason(QUESTION, PACK, EMPTY_STATE, None)
    user_message = recorder.bodies[0]["messages"][1]["content"]
    assert QUESTION in user_message
    assert "[1]" in user_message
    assert "[2]" in user_message


def test_situation_facts_ride_in_the_prompt():
    from taxverity.facts import FactStatus, SituationFact
    from taxverity.memory.fact_state import merge_turn

    state = merge_turn(
        EMPTY_STATE,
        UserFacts(),
        turn=1,
        situation_facts=(
            SituationFact(
                name="rent recipient",
                status=FactStatus.STATED,
                raw_value="my mother",
                source_span="pay rent to my mother",
            ),
        ),
    )
    node, recorder = build(ok(good_payload()))
    node.reason(QUESTION, PACK, state, None)
    user_message = recorder.bodies[0]["messages"][1]["content"]
    assert "rent recipient" in user_message
    assert "my mother" in user_message
