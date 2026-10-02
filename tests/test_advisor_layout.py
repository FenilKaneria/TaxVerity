"""R22 Part B — the advisor layout's line grammar, simplify mode, and
release-only-what-is-served, at the unit level (no live LLM)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from conftest import register_account
from taxverity.chunking.models import Chunk
from taxverity.generation.claims import (
    ClaimEvent,
    ClaimType,
    WithheldEvent,
    classify_line,
    line_body,
    parse_claim,
    strip_list_number,
)
from taxverity.generation.generate import (
    STYLE_REQUEST_NOTE,
    SYSTEM_PROMPT,
    AnswerGenerator,
    layout_note,
    render_context,
)
from taxverity.generation.verifier import Verifier, Violation, numbers_in
from taxverity.graph.nodes import generate_verify, load_thread, retrieve, retrieve_retry
from taxverity.graph.state import StageEvent
from taxverity.memory.contextualize import is_style_followup
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.evidence import EvidencePacker
from taxverity.safety.evidence_gate import INSUFFICIENT_EVIDENCE_MESSAGE
from taxverity.threads.store import append_message, create_thread
from test_generation import GOOD, QUESTION, FakeLLM, answer
from test_graph_nodes import PASSWORD, Boom, Recorder, deps, thread_state
from test_verifier import CHUNKS, PACK, content, heading, violations

# --- line grammar -------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("### In short", ClaimType.HEADING),
        ("### What to do next", ClaimType.HEADING),
        ("You can deduct the interest you pay [1].", ClaimType.CONTENT),
        ("1. Keep the receipt [2].", ClaimType.CONTENT),
        ("2) Pay by cheque [2].", ClaimType.CONTENT),
        ("1. For example, you pay rent [1][eg].", ClaimType.EXAMPLE),
        ("3. The Act does not deal with this.", ClaimType.NO_BASIS),
        (
            "1. This can't yet be determined because the cap is unknown [2].",
            ClaimType.UNKNOWN,
        ),
    ],
)
def test_the_advisor_layout_lines_classify(line, expected):
    assert classify_line(line) is expected


def test_a_step_number_is_list_syntax_not_body():
    assert line_body("2. You can deduct it [1].") == "You can deduct it [1]."
    assert strip_list_number("12) Keep proof [1].") == "Keep proof [1]."
    # A figure that opens a sentence is not a step number.
    assert strip_list_number("1.5 lakh is the cap [2].") == "1.5 lakh is the cap [2]."
    assert strip_list_number("2,00,000 is the cap [2].") == "2,00,000 is the cap [2]."


def test_a_numbered_step_does_not_ground_its_own_number():
    """Mutation-style: the "2" must be read as list syntax, so the line
    passes; the same 2 stated inside the sentence is an unsupported figure
    (unit [2] grounds 24 and 2,00,000, never 2)."""
    step = "2. The deduction is capped at ₹2,00,000 [2]."
    assert Decimal(2) not in numbers_in("The deduction is capped at ₹2,00,000")
    assert violations(content(step)) == set()
    inline = "Rule 2: the deduction is capped at ₹2,00,000 [2]."
    assert Violation.UNSUPPORTED_NUMBER in violations(content(inline))


def test_a_figure_after_the_step_number_is_still_checked():
    assert Violation.UNSUPPORTED_NUMBER in violations(
        content("1. The cap is 5 lakh [2].")
    )


def test_a_numbered_step_keeps_its_number_when_released():
    llm = FakeLLM(answer("1. The deduction shall not exceed two lakh rupees [2]."))
    events = AnswerGenerator(llm, CHUNKS).generate(QUESTION, PACK)
    assert [type(e) for e in events] == [ClaimEvent]
    assert events[0].text.startswith("1. ")
    assert [c.path for c in events[0].citations] == ["24"]


def test_a_section_label_passes_and_a_numbered_label_is_malformed():
    assert violations(heading("### In short")) == set()
    assert Violation.MALFORMED_HEADING in violations(heading("### Step 2"))


def test_a_plain_paragraph_line_is_still_content_and_must_cite():
    claim = parse_claim("You can deduct thirty per cent of the annual value.")
    assert claim.type is ClaimType.CONTENT
    assert Violation.NO_CITATION in violations(claim)


# --- prompt v7 ------------------------------------------------------------------


def test_the_prompt_bans_legalese_and_section_numbers_in_the_text():
    assert '"assessee"' in SYSTEM_PROMPT
    assert '"tax liability"' in SYSTEM_PROMPT
    assert 'or the word "section", in a line' in SYSTEM_PROMPT
    assert "## " not in SYSTEM_PROMPT.replace("### ", "")


@pytest.mark.parametrize(
    ("intent", "label"),
    [
        ("calculation", "### Your tax"),
        ("eligibility", "### Conditions to check"),
        ("comparison", "### How they compare"),
        ("procedure", "### Steps"),
    ],
)
def test_the_layout_follows_the_kind_of_question(intent, label):
    assert label in layout_note(intent, follow_up=False)


def test_an_explanation_and_a_follow_up_get_no_fixed_sections():
    assert layout_note("explanation", follow_up=False).count("### ") == 0
    assert "no section label" in layout_note(None, follow_up=False)
    assert "follow-up" in layout_note("calculation", follow_up=True)


def test_the_style_sample_states_no_figure_of_its_own():
    sample = SYSTEM_PROMPT.split("Style sample")[1].split("Rules:")[0]
    # Only its citation markers and the "1." of its one numbered step carry
    # digits; nothing in it could be copied in as a rate, limit or date.
    body = "\n".join(strip_list_number(line) for line in sample.splitlines())
    assert numbers_in(body.replace("[1]", "").replace("[2]", "")) == frozenset()


# --- simplify mode ------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "explain in simply",
        "Can you explain that in simpler words?",
        "I don't understand",
        "still confused",
        "give me an example",
        "what does that mean?",
        "explain again please",
        "in plain english?",
    ],
)
def test_a_short_rephrase_request_is_a_style_followup(question):
    assert is_style_followup(question, "previous answer")


@pytest.mark.parametrize(
    "question",
    [
        "What is the HRA exemption?",
        "Can I claim interest on a second home loan?",
        # A style cue inside a longer new question is a new question.
        "Give me an example of how the HRA exemption works for a rented flat in Mumbai",
    ],
)
def test_a_new_question_is_not_a_style_followup(question):
    assert not is_style_followup(question, "previous answer")


def test_with_no_previous_answer_nothing_is_a_style_followup():
    assert not is_style_followup("explain simply", "")


def test_the_style_note_reaches_the_prompt_only_when_asked():
    plain = render_context(QUESTION, PACK, None, None, previous_answer="earlier")
    styled = render_context(
        QUESTION, PACK, None, None, previous_answer="earlier", style_request=True
    )
    assert STYLE_REQUEST_NOTE not in plain
    assert f"<style_request>\n{STYLE_REQUEST_NOTE}\n</style_request>" in styled


def test_the_generator_passes_the_style_request_through():
    llm = FakeLLM(answer(GOOD))
    AnswerGenerator(llm, CHUNKS).generate(
        QUESTION, PACK, previous_answer="earlier", style_request=True
    )
    assert STYLE_REQUEST_NOTE in llm.calls[0][0][1].content


def _lookup(path: str) -> ScoredChunk | None:
    chunk = CHUNKS.get(path)
    return None if chunk is None else ScoredChunk(chunk=chunk, score=1.0)


def test_a_style_followup_packs_the_previous_answers_passages_without_searching():
    state = {
        "query": "explain simply",
        "style_followup": True,
        "previous_citations": ("24", "22(1)", "24"),
    }
    recorder = Recorder()
    result = retrieve(
        state,
        deps(
            retriever=Boom(),  # a fresh search must not run
            packer=EvidencePacker(CHUNKS.values()),
            citation_lookup=_lookup,
        ),
        writer=recorder,
    )
    assert [unit.citation for unit in result["pack"].units] == ["24", "22(1)"]
    assert recorder.events == [
        StageEvent(stage="evidence", chunks=("24", "22(1)")).model_dump()
    ]


def test_a_style_followup_with_nothing_resolvable_searches_as_usual():
    calls: list[str] = []

    class FakeRetriever:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            calls.append(query)
            return [ScoredChunk(chunk=CHUNKS["23"], score=1.0)]

    state = {
        "query": "explain simply",
        "style_followup": True,
        "previous_citations": ("999",),
    }
    result = retrieve(
        state,
        deps(retriever=FakeRetriever(), citation_lookup=_lookup),
        writer=Recorder(),
    )
    assert calls == ["explain simply"]
    assert [unit.citation for unit in result["pack"].units] == ["23"]


def test_the_corrective_retry_searches_afresh_even_for_a_style_followup():
    calls: list[int] = []

    class FakeRetriever:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            calls.append(k)
            return [ScoredChunk(chunk=CHUNKS["23"], score=1.0)]

    state = {
        "query": "explain simply",
        "style_followup": True,
        "previous_citations": ("24",),
    }
    result = retrieve_retry(
        state,
        deps(retriever=FakeRetriever(), citation_lookup=_lookup),
        writer=Recorder(),
    )
    assert calls  # searched
    assert result["retried"] is True


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def thread_id(schema, alice):
    return create_thread(schema, alice, "House property").thread_id


def test_load_thread_reads_the_last_answers_citations(schema, alice, thread_id):
    append_message(schema, alice, thread_id, "user", "q1")
    append_message(
        schema,
        alice,
        thread_id,
        "assistant",
        "old",
        payload={"citations": [{"marker": 1, "path": "23", "quote": "x"}]},
    )
    append_message(schema, alice, thread_id, "user", "q2")
    append_message(
        schema,
        alice,
        thread_id,
        "assistant",
        "- Thirty per cent is deducted [1].\n- Capped [2].",
        payload={
            "citations": [
                {"marker": 1, "path": "22(1)", "quote": "x"},
                {"marker": 2, "path": "24", "quote": "y"},
            ]
        },
    )
    state = {"user_id": alice, "thread_id": thread_id, "question": "I don't understand"}
    result = load_thread(state, deps(conn=schema), writer=Recorder())
    assert result["previous_citations"] == ("22(1)", "24")
    assert result["style_followup"] is True


def test_load_thread_on_a_fresh_thread_has_no_style_followup(schema, alice, thread_id):
    state = {"user_id": alice, "thread_id": thread_id, "question": "explain simply"}
    result = load_thread(state, deps(conn=schema), writer=Recorder())
    assert result["previous_citations"] == ()
    assert result["style_followup"] is False


# --- release only what will be served -------------------------------------------


def _state(pack=PACK):
    return {
        "query": QUESTION,
        "pack": pack,
        "fact_state": thread_state(),
        "computation": None,
    }


def test_a_gated_answer_emits_none_of_its_events():
    """Zero grounded claims: the pass's withheld events and any ungrounded
    claim (here a no_basis line) are held back — the retry or the fixed
    template follows, never a half-answer."""
    llm = FakeLLM(
        answer("The Act does not deal with this.", "- Arrears are exempt [99]."),
        answer("- Arrears are exempt [98]."),
    )
    recorder = Recorder()
    result = generate_verify(
        _state(), deps(generator=AnswerGenerator(llm, CHUNKS)), writer=recorder
    )
    assert result["answer_text"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert {type(e) for e in result["events"]} == {ClaimEvent, WithheldEvent}
    assert all("stage" in event for event in recorder.events)


def test_a_served_answer_emits_every_event_in_order():
    llm = FakeLLM(answer(GOOD, "- Arrears are exempt [99]."), answer("- Still [98]."))
    recorder = Recorder()
    result = generate_verify(
        _state(), deps(generator=AnswerGenerator(llm, CHUNKS)), writer=recorder
    )
    released = [e for e in recorder.events if "stage" not in e]
    assert released == [event.model_dump() for event in result["events"]]
    assert all(e.get("verified", True) is True for e in released)


def test_a_section_label_with_nothing_served_under_it_is_left_out():
    llm = FakeLLM(
        answer(
            "### In short",
            GOOD,
            "### Example",
            "- Arrears are exempt [99].",
            "### What to do next",
        ),
        answer("- Still exempt [98]."),
    )
    events = AnswerGenerator(llm, CHUNKS).generate(QUESTION, PACK)
    served = [e.text for e in events if isinstance(e, ClaimEvent)]
    assert served == ["### In short", GOOD]
    # The withheld line is still reported; only the empty labels go.
    assert [e.id for e in events if isinstance(e, WithheldEvent)] == [4]


def test_the_prompt_keeps_next_steps_inside_the_passages():
    assert "Never describe how to use a website, portal, app or form" in SYSTEM_PROMPT
    assert "never pick one for them" in SYSTEM_PROMPT


# --- off-Act procedure guard -------------------------------------------------------

PORTAL_SECTION = Chunk.create(
    "v" * 64,
    "40",
    "40. The return shall be furnished electronically on the portal notified by the Board.",
    parent_id=None,
    doc_id="income-tax-act-2025",
    node_type="section",
    section_number="40",
    page_start=1,
    page_end=1,
)


@pytest.mark.parametrize(
    "line",
    [
        "1. Log in to the income\u2011tax e\u2011filing portal [1].",
        "- Click Verify on the website [2].",
        "Apply online for the certificate [1].",
    ],
)
def test_a_portal_step_no_cited_passage_mentions_is_withheld(line):
    assert Violation.OFF_ACT_PROCEDURE in violations(parse_claim(line))


def test_a_portal_step_its_cited_passage_mentions_passes():
    pack = EvidencePacker([PORTAL_SECTION]).pack(
        [ScoredChunk(chunk=PORTAL_SECTION, score=1.0)]
    )
    verdict = Verifier(pack).verify(parse_claim("File your return on the portal [1]."))
    assert verdict.passed


def test_an_uncited_no_basis_line_may_name_the_portal():
    claim = parse_claim("The Act does not describe the e-filing portal steps.")
    assert claim.type is ClaimType.NO_BASIS
    assert violations(claim) == set()


def test_an_example_naming_an_app_is_withheld():
    line = "- Suppose you pay by a mobile app; you can claim it [1][eg]."
    assert Violation.OFF_ACT_PROCEDURE in violations(parse_claim(line))
