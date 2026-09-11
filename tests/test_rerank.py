"""Step 5.6 — the reranker (ADR-051, ADR-052, ADR-084).

The API tests drive a scripted MockTransport: no network, no key. The corpus
tests read the gold rerank scores `scripts/measure_rerank.py` stored, and skip
without them.
"""

from __future__ import annotations

import json
import logging

import httpx2
import pytest

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.evals.baseline import measure
from taxverity.evals.gold import QuerySlice
from taxverity.evals.ladder import Verdict
from taxverity.evals.metrics import CreditMode, RunReport, Scores
from taxverity.evals.rerank import (
    RERANK_P95_BUDGET_MS,
    RERANK_SCORES_FILENAME,
    RerankScores,
    StaleRerankScoresError,
    StoredReranker,
    judge_rerank,
    load_rerank_scores,
    write_rerank_scores,
)
from taxverity.observability import PAN_MASK
from taxverity.retrieval import rerank
from taxverity.retrieval.base import ScoredChunk, as_ranked_citations
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.rerank import (
    API_URL,
    MODEL_ID,
    RERANK_DEPTH,
    CachedReranker,
    JinaReranker,
    RerankError,
    RerankRetriever,
)

LOGGER = "taxverity.retrieval.rerank"
KEY = "test-key-not-real"
PAN = "ABCDE1234F"
SALARY = "1400000"
CORPUS_VERSION = "r" * 64
# Below the measured 0.712 on the stored gold pool (ADR-084), not at it.
NDCG5_FLOOR = 0.70


def section(n: int, text: str | None = None) -> Chunk:
    return Chunk.create(
        CORPUS_VERSION,
        str(n),
        text or f"{n}. Provision number {n}.",
        parent_id=None,
        doc_id="income-tax-act-2025",
        node_type=NodeType.SECTION,
        section_number=str(n),
        page_start=1,
        page_end=1,
    )


CHUNKS = [section(n) for n in range(1, 31)]


def scored(chunks) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=c, score=float(len(chunks) - i)) for i, c in enumerate(chunks)]


def ok(scores: list[float], *, order=None, tokens=11) -> httpx2.Response:
    indices = order if order is not None else list(range(len(scores)))
    return httpx2.Response(
        200,
        json={
            "model": MODEL_ID,
            "object": "list",
            "usage": {"total_tokens": tokens},
            "results": [{"index": i, "relevance_score": scores[i]} for i in indices],
        },
    )


class Recorder:
    """Scripts responses in order, then scores documents by reverse position."""

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
        count = len(json.loads(request.read())["documents"])
        return ok([float(i) for i in range(count)])

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]


def make(handler: Recorder, **kwargs) -> JinaReranker:
    kwargs.setdefault("backoff_base", 0.0)
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return JinaReranker(KEY, http_client=client, **kwargs)


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(rerank.time, "sleep", slept.append)
    return slept


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured():
    """Attached below the `taxverity` root, which does not propagate to
    pytest's handler (Step 4.2's finding), and below `RedactingFilter`, so what
    is asserted is what the call site passed."""
    logger = logging.getLogger(LOGGER)
    handler = Capture()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler.records
    logger.removeHandler(handler)
    logger.setLevel(previous)


# --- the API client ---------------------------------------------------------------


def test_scores_come_back_by_chunk_id_whatever_order_the_api_answers_in():
    handler = Recorder(ok([0.1, 0.9, 0.5], order=[1, 2, 0]))
    scores = make(handler).score("rent", CHUNKS[:3])
    assert scores == {CHUNKS[0].chunk_id: 0.1, CHUNKS[1].chunk_id: 0.9, CHUNKS[2].chunk_id: 0.5}


def test_the_request_scores_every_document_as_it_is_indexed():
    handler = Recorder()
    make(handler).score("rent", CHUNKS[:3])
    (body,) = handler.bodies
    assert handler.requests[0].url == API_URL
    assert handler.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert body["model"] == MODEL_ID
    assert body["top_n"] == 3 and body["return_documents"] is False
    assert body["documents"] == [chunk.embed_text() for chunk in CHUNKS[:3]]


def test_the_query_is_redacted_before_it_leaves_and_the_documents_are_not():
    statute = section(99, f"99. A PAN such as {PAN} is quoted here.")
    handler = Recorder()
    make(handler).score(f"my PAN is {PAN}", [statute])
    (body,) = handler.bodies
    assert PAN not in body["query"] and PAN_MASK in body["query"]
    assert PAN in body["documents"][0]


def test_the_query_path_makes_one_attempt_and_raises():
    handler = Recorder(httpx2.Response(503))
    with pytest.raises(RerankError, match="after 1 attempt"):
        make(handler).score("rent", CHUNKS[:2])
    assert len(handler.requests) == 1


def test_a_timeout_is_a_rerank_error():
    handler = Recorder(httpx2.ReadTimeout("slow"))
    with pytest.raises(RerankError):
        make(handler).score("rent", CHUNKS[:2])


def test_with_attempts_a_rate_limit_is_ridden_out(no_sleep, captured):
    handler = Recorder(httpx2.Response(429, headers={"retry-after": "3"}))
    scores = make(handler, max_attempts=3).score("rent", CHUNKS[:2])
    assert len(scores) == 2 and len(handler.requests) == 2
    assert no_sleep == [3.0]
    assert [r.levelno for r in captured if "retrying" in r.getMessage()] == [logging.WARNING]


def test_a_client_error_is_not_retried(no_sleep):
    handler = Recorder(httpx2.Response(401, text="bad key"))
    with pytest.raises(RerankError, match="401"):
        make(handler, max_attempts=3).score("rent", CHUNKS[:2])
    assert len(handler.requests) == 1 and no_sleep == []


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, json={"usage": {}}),
        httpx2.Response(200, json={"results": [{"index": 0}]}),
        httpx2.Response(200, text="not json"),
        ok([0.1, 0.2], order=[0, 0]),
        ok([0.1, 0.2], order=[0]),
        ok([0.1, 0.2, 0.3], order=[0, 1, 2]),
    ],
)
def test_a_malformed_response_is_a_rerank_error(response):
    with pytest.raises(RerankError):
        make(Recorder(response)).score("rent", CHUNKS[:2])


def test_no_documents_means_no_request():
    handler = Recorder()
    assert make(handler).score("rent", []) == {}
    assert handler.requests == []


def test_tokens_and_latency_are_recorded():
    reranker = make(Recorder(ok([0.1], tokens=40), ok([0.1], tokens=2)))
    reranker.score("a", CHUNKS[:1])
    reranker.score("b", CHUNKS[:1])
    assert reranker.tokens_used == 42
    assert len(reranker.latencies_ms) == 2 and all(ms >= 0 for ms in reranker.latencies_ms)


def test_the_request_body_is_never_logged(no_sleep, captured):
    handler = Recorder(httpx2.Response(503), httpx2.Response(503))
    with pytest.raises(RerankError):
        make(handler, max_attempts=2).score(f"PAN {PAN}, salary {SALARY}", CHUNKS[:2])
    make(Recorder()).score(f"PAN {PAN}, salary {SALARY}", CHUNKS[:2])
    assert captured, "the capture must be live"
    for record in captured:
        assert PAN not in record.getMessage() and SALARY not in record.getMessage()


def test_bad_construction_and_duplicate_documents_are_refused():
    with pytest.raises(ValueError, match="api_key"):
        JinaReranker("")
    with pytest.raises(ValueError, match="max_attempts"):
        JinaReranker(KEY, max_attempts=0)
    with pytest.raises(ValueError, match="twice"):
        make(Recorder()).score("rent", [CHUNKS[0], CHUNKS[0]])


# --- the in-process cache ---------------------------------------------------------


class Counting:
    def __init__(self, fail_first: bool = False):
        self.calls = 0
        self.fail_first = fail_first

    def score(self, query, chunks):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RerankError("down")
        return {chunk.chunk_id: float(i) for i, chunk in enumerate(chunks)}


def test_the_same_question_over_the_same_pool_is_answered_from_the_cache():
    inner = Counting()
    cache = CachedReranker(inner)
    first = cache.score("rent paid", CHUNKS[:3])
    again = cache.score("rent paid", list(reversed(CHUNKS[:3])))
    assert inner.calls == 1 and again == first


def test_a_different_pool_or_question_misses():
    inner = Counting()
    cache = CachedReranker(inner)
    cache.score("rent", CHUNKS[:3])
    cache.score("rent", CHUNKS[:4])
    cache.score("interest", CHUNKS[:3])
    assert inner.calls == 3


def test_the_least_recently_used_entry_is_evicted():
    inner = Counting()
    cache = CachedReranker(inner, maxsize=2)
    cache.score("a", CHUNKS[:1])
    cache.score("b", CHUNKS[:1])
    cache.score("a", CHUNKS[:1])
    cache.score("c", CHUNKS[:1])
    assert len(cache) == 2
    cache.score("a", CHUNKS[:1])
    assert inner.calls == 3
    cache.score("b", CHUNKS[:1])
    assert inner.calls == 4


def test_a_failure_is_not_cached():
    inner = Counting(fail_first=True)
    cache = CachedReranker(inner)
    with pytest.raises(RerankError):
        cache.score("rent", CHUNKS[:2])
    assert len(cache.score("rent", CHUNKS[:2])) == 2 and inner.calls == 2


# --- the retriever ----------------------------------------------------------------


class Fixed:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k):
        return scored(self.chunks[:k])


class ByScore:
    def __init__(self, scores: dict[str, float] | None = None, error: Exception | None = None):
        self.scores = scores or {}
        self.error = error
        self.seen: list[list[str]] = []

    def score(self, query, chunks):
        self.seen.append([c.node_path for c in chunks])
        if self.error:
            raise self.error
        return {c.chunk_id: self.scores.get(c.node_path, 0.0) for c in chunks}


def paths(results):
    return [r.chunk.node_path for r in results]


def test_the_head_is_reordered_and_the_tail_keeps_its_place():
    reranker = ByScore({"3": 0.9, "1": 0.5})
    results = RerankRetriever(Fixed(CHUNKS[:25]), reranker, depth=5).search("q", 7)
    assert paths(results) == ["3", "1", "2", "4", "5", "6", "7"]
    assert reranker.seen == [["1", "2", "3", "4", "5"]]


def test_the_whole_depth_is_reranked_even_when_fewer_results_are_asked_for():
    reranker = ByScore({"20": 1.0})
    results = RerankRetriever(Fixed(CHUNKS), reranker).search("q", 3)
    assert len(reranker.seen[0]) == RERANK_DEPTH
    assert paths(results) == ["20", "1", "2"]


def test_the_scores_are_ordinal_and_rank_cleanly():
    results = RerankRetriever(Fixed(CHUNKS), ByScore({"9": 2.0, "4": 1.0})).search("q", 10)
    as_ranked_citations(results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_a_vendor_failure_keeps_the_fusion_order_and_warns_without_the_query(captured):
    """ADR-052's required degradation: never a failed query."""
    primary = Fixed(CHUNKS)
    retriever = RerankRetriever(primary, ByScore(error=RerankError("rate limited")))
    results = retriever.search(f"my PAN {PAN}", 10)
    assert paths(results) == paths(primary.search("", 10))
    warnings = [r for r in captured if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "fusion order" in warnings[0].getMessage()
    assert PAN not in warnings[0].getMessage()


def test_anything_else_is_a_bug_and_propagates():
    retriever = RerankRetriever(Fixed(CHUNKS), ByScore(error=KeyError("oops")))
    with pytest.raises(KeyError):
        retriever.search("q", 5)


def test_an_empty_ranking_never_calls_the_reranker():
    reranker = ByScore()
    assert RerankRetriever(Fixed([]), reranker).search("q", 5) == []
    assert reranker.seen == []


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="depth"):
        RerankRetriever(Fixed(CHUNKS), ByScore(), depth=0)
    with pytest.raises(ValueError, match="k must"):
        RerankRetriever(Fixed(CHUNKS), ByScore()).search("q", 0)


def test_an_exact_citation_stays_first_however_the_reranker_scores_it():
    """The shortcut sits outside the reranker (ADR-062)."""
    reranker = ByScore({"7": 5.0, "3": -1.0})
    composed = ShortcutRetriever(
        CitationRetriever(CHUNKS), RerankRetriever(Fixed(CHUNKS), reranker)
    )
    assert paths(composed.search("What does section 3 say?", 3)) == ["3", "7", "1"]


# --- the rule and the stored scores -------------------------------------------------


def scores(ndcg: float) -> dict[CreditMode, Scores]:
    return {mode: Scores(recall=0.8, mrr=0.7, ndcg=ndcg) for mode in CreditMode}


def report(ndcg: float, k: int = 5) -> RunReport:
    return RunReport(
        k=k,
        scored=(),
        overall=scores(ndcg),
        per_slice={member: scores(ndcg) for member in (QuerySlice.CITATION,)},
        negatives=0,
    )


def test_holding_ndcg_within_the_budget_is_adopted():
    assert judge_rerank(report(0.70), report(0.70), 900.0) == Verdict(adopted=True, reasons=())


def test_an_ndcg_fall_rejects():
    verdict = judge_rerank(report(0.65), report(0.70), 900.0)
    assert verdict.reasons == ("lenient nDCG@5 fell 0.700 -> 0.650",)


def test_a_p95_over_budget_rejects_even_with_a_gain():
    verdict = judge_rerank(report(0.80), report(0.70), RERANK_P95_BUDGET_MS + 1)
    assert verdict.reasons == ("rerank p95 1501 ms exceeds the 1500 ms budget",)


def test_the_guard_is_at_five():
    with pytest.raises(ValueError, match="nDCG@5"):
        judge_rerank(report(0.7, k=10), report(0.7, k=10), 900.0)


def stored(**overrides) -> RerankScores:
    fields = dict(
        model_id=MODEL_ID,
        corpus_version=CORPUS_VERSION,
        depth=RERANK_DEPTH,
        scores={"q1": {CHUNKS[0].chunk_id: 0.3}},
        latencies_ms=(410.0,),
        tokens_billed=120,
    )
    return RerankScores(**{**fields, **overrides})


def test_stored_scores_round_trip_and_refuse_a_mismatch(tmp_path):
    path = tmp_path / RERANK_SCORES_FILENAME
    write_rerank_scores(path, stored())
    want = dict(model_id=MODEL_ID, corpus_version=CORPUS_VERSION, depth=RERANK_DEPTH)
    assert load_rerank_scores(path, **want, questions=["q1"]) == stored()
    for field, value in (("model_id", "other"), ("corpus_version", "x" * 64), ("depth", 10)):
        with pytest.raises(StaleRerankScoresError, match=field):
            load_rerank_scores(path, **{**want, field: value}, questions=["q1"])
    with pytest.raises(StaleRerankScoresError, match="no scores"):
        load_rerank_scores(path, **want, questions=["q1", "q2"])


def test_the_stored_reranker_never_invents_a_score():
    reranker = StoredReranker(stored().scores)
    assert reranker.score("q1", CHUNKS[:1]) == {CHUNKS[0].chunk_id: 0.3}
    with pytest.raises(KeyError):
        reranker.score("q1", CHUNKS[:2])
    with pytest.raises(KeyError):
        reranker.score("q2", CHUNKS[:1])


# --- corpus: the real pool, scored once and stored, no network ----------------------


@pytest.fixture(scope="module")
def reranked_legs(gold, stored_chunks, retrieval_legs):
    from taxverity.config import Settings

    path = Settings().data_dir / "rerank" / RERANK_SCORES_FILENAME
    if not path.exists():
        pytest.skip("run scripts/measure_rerank.py to score the gold pool")
    corpus_version, _ = stored_chunks
    loaded = load_rerank_scores(
        path,
        model_id=MODEL_ID,
        corpus_version=corpus_version,
        depth=RERANK_DEPTH,
        questions=[q.question for q in gold],
    )
    index, dense, bm25, shortcut = retrieval_legs
    fusion = FusionRetriever([dense, bm25])
    base = ShortcutRetriever(shortcut, fusion)
    reranked = ShortcutRetriever(shortcut, RerankRetriever(fusion, StoredReranker(loaded.scores)))
    return index, base, reranked, loaded


def test_the_reranker_meets_its_rule_on_the_real_pool(gold, reranked_legs):
    """The rule registered for Step 5.6 (ADR-084), and a floor under it."""
    index, base, reranked, loaded = reranked_legs
    before = measure("fusion", base, gold, index, ordinal_scores=True)
    after = measure("reranked", reranked, gold, index, ordinal_scores=True)
    ordered = sorted(loaded.latencies_ms)
    p95 = ordered[-(-95 * len(ordered) // 100) - 1]
    verdict = judge_rerank(after.reports[5], before.reports[5], p95)
    assert verdict.adopted, verdict.reasons
    assert after.reports[5].overall[CreditMode.LENIENT].ndcg >= NDCG5_FLOOR
