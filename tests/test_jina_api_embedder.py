"""Step 4.4 (R15) — the Jina hosted-API embedder (ADR-075).

Every test but the last drives a scripted MockTransport: no network, no key.
The last one hits the real API and is skipped unless a key is configured.
"""

from __future__ import annotations

import json
import logging
import math

import httpx2
import pytest

from taxverity.config import MissingSettingError, Settings
from taxverity.embedding import jina_api
from taxverity.embedding.backends import Embedder, EmbedKind
from taxverity.embedding.jina_api import (
    API_URL,
    DIM,
    ENCODING,
    MODEL_ID,
    REVISION,
    RUNTIME,
    EmbeddingAPIError,
    JinaAPIEmbedder,
)
from taxverity.observability import PAN_MASK

LOGGER = "taxverity.embedding.jina_api"
KEY = "test-key-not-real"
PAN = "ABCDE1234F"
SALARY = "1400000"


def vector(i: int) -> list[float]:
    # Deliberately not unit length, so re-normalisation is observable.
    return [float(i + 1), 2.0] + [0.0] * (DIM - 2)


def ok(count: int, *, order=None, width=DIM, tokens=7) -> httpx2.Response:
    indices = order if order is not None else list(range(count))
    data = [
        {"object": "embedding", "index": i, "embedding": vector(i)[:width]}
        for i in indices
    ]
    return httpx2.Response(
        200,
        json={
            "model": MODEL_ID,
            "data": data,
            "usage": {"total_tokens": tokens, "prompt_tokens": tokens},
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
        return ok(len(json.loads(request.read())["input"]))

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.read()) for r in self.requests]


def make(handler: Recorder, **kwargs) -> JinaAPIEmbedder:
    kwargs.setdefault("backoff_base", 0.0)
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return JinaAPIEmbedder(KEY, http_client=client, **kwargs)


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(jina_api.time, "sleep", slept.append)
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
    # root and RedactingFilter — so the assertion is about what the call site
    # passed, not what the filter cleaned up (the Step 4.2 caplog trap).
    handler = CaptureHandler()
    logger = logging.getLogger(LOGGER)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


# --- identity & contract ---------------------------------------------------


def test_satisfies_the_embedder_protocol():
    assert isinstance(make(Recorder()), Embedder)


def test_info_reports_the_hosted_identity():
    info = make(Recorder()).info()
    assert (info.model_id, info.dim) == (MODEL_ID, 1024)
    assert info.revision == REVISION == "hosted-unpinned"
    assert info.runtime == RUNTIME == "jina-api"
    assert info.encoding == ENCODING


def test_a_document_is_sent_as_a_passage_with_the_full_recipe():
    handler = Recorder()
    make(handler).embed(["income from house property"], EmbedKind.DOCUMENT)
    (body,) = handler.bodies
    assert body == {
        "model": MODEL_ID,
        "task": "retrieval.passage",
        "input": ["income from house property"],
        "dimensions": 1024,
        "normalized": True,
        "embedding_type": "float",
        "truncate": False,
    }
    assert str(handler.requests[0].url) == API_URL


def test_a_query_is_sent_with_the_query_task():
    handler = Recorder()
    make(handler).embed(["what is the rebate limit?"], EmbedKind.QUERY)
    assert handler.bodies[0]["task"] == "retrieval.query"


def test_the_key_travels_as_a_bearer_token():
    handler = Recorder()
    make(handler).embed(["x"], EmbedKind.QUERY)
    assert handler.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_an_empty_batch_makes_no_request():
    handler = Recorder()
    assert make(handler).embed([], EmbedKind.DOCUMENT) == []
    assert handler.requests == []


# --- PII egress --------------------------------------------------------------


def test_query_text_is_redacted_before_it_leaves_the_process():
    handler = Recorder()
    make(handler).embed(
        [f"my PAN is {PAN}, salary {SALARY}, under section 2(5)(b)(ii)"],
        EmbedKind.QUERY,
    )
    (sent,) = handler.bodies[0]["input"]
    assert PAN not in sent
    assert PAN_MASK in sent
    # Amounts and statutory tokens must survive, or the question changes.
    assert SALARY in sent
    assert "2(5)(b)(ii)" in sent


def test_document_text_is_sent_verbatim():
    handler = Recorder()
    make(handler).embed([f"quoted identifier {PAN}"], EmbedKind.DOCUMENT)
    assert handler.bodies[0]["input"] == [f"quoted identifier {PAN}"]


def test_request_text_and_key_are_never_logged(captured, no_sleep):
    # A 503 first, so the retry path — the one that logs — runs too.
    handler = Recorder(httpx2.Response(503))
    make(handler).embed([f"my PAN is {PAN}, salary {SALARY}"], EmbedKind.QUERY)
    assert captured.records, "capture is not live — the assertion below would be vacuous"
    for record in captured.records:
        message = record.getMessage()
        assert PAN not in message
        assert SALARY not in message
        assert KEY not in message


# --- response checks ---------------------------------------------------------


def test_vectors_come_back_in_input_order_whatever_the_response_order():
    handler = Recorder(ok(3, order=[2, 0, 1]))
    vectors = make(handler).embed(["a", "b", "c"], EmbedKind.DOCUMENT)
    firsts = [v[0] / v[1] for v in vectors]
    assert firsts == pytest.approx([0.5, 1.0, 1.5])


def test_vectors_are_unit_length():
    vectors = make(Recorder()).embed(["a", "b"], EmbedKind.DOCUMENT)
    for v in vectors:
        assert len(v) == DIM
        assert math.isclose(math.sqrt(math.fsum(x * x for x in v)), 1.0, abs_tol=1e-12)


def test_a_missing_vector_is_refused():
    with pytest.raises(EmbeddingAPIError, match="indices"):
        make(Recorder(ok(1))).embed(["a", "b"], EmbedKind.DOCUMENT)


def test_a_duplicated_index_is_refused():
    with pytest.raises(EmbeddingAPIError, match="indices"):
        make(Recorder(ok(2, order=[0, 0]))).embed(["a", "b"], EmbedKind.DOCUMENT)


def test_a_wrong_width_is_refused():
    with pytest.raises(EmbeddingAPIError, match="width"):
        make(Recorder(ok(1, width=512))).embed(["a"], EmbedKind.DOCUMENT)


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, json={"usage": {}}),
        httpx2.Response(200, json={"data": [{"index": 0}]}),
        httpx2.Response(200, text="<html>gateway</html>"),
    ],
    ids=["no-data", "no-embedding", "not-json"],
)
def test_a_malformed_body_is_an_api_error(response):
    """Step 4.5 degrades on EmbeddingAPIError. A bare KeyError would escape the
    fallback and fail the query instead."""
    with pytest.raises(EmbeddingAPIError):
        make(Recorder(response)).embed(["a"], EmbedKind.DOCUMENT)


def test_billed_tokens_accumulate():
    embedder = make(Recorder(ok(1, tokens=5), ok(1, tokens=11)))
    embedder.embed(["a"], EmbedKind.DOCUMENT)
    embedder.embed(["b"], EmbedKind.DOCUMENT)
    assert embedder.tokens_used == 16


# --- retry ---------------------------------------------------------------------


def test_a_429_is_retried_after_the_server_named_delay(no_sleep):
    handler = Recorder(httpx2.Response(429, headers={"Retry-After": "3"}))
    make(handler).embed(["a"], EmbedKind.DOCUMENT)
    assert len(handler.requests) == 2
    assert no_sleep == [3.0]


def test_an_absurd_retry_after_is_capped(no_sleep):
    handler = Recorder(httpx2.Response(429, headers={"Retry-After": "86400"}))
    make(handler).embed(["a"], EmbedKind.DOCUMENT)
    assert no_sleep == [jina_api.MAX_RETRY_AFTER]


def test_a_transport_error_is_retried(no_sleep):
    handler = Recorder(httpx2.ConnectError("refused"))
    assert len(make(handler).embed(["a"], EmbedKind.DOCUMENT)) == 1
    assert len(handler.requests) == 2


def test_a_client_error_is_not_retried(no_sleep):
    handler = Recorder(httpx2.Response(400, json={"detail": "bad"}))
    with pytest.raises(EmbeddingAPIError, match="400"):
        make(handler).embed(["a"], EmbedKind.DOCUMENT)
    assert len(handler.requests) == 1
    assert no_sleep == []


def test_exhausted_retries_raise(no_sleep):
    handler = Recorder(*[httpx2.Response(503)] * 3)
    with pytest.raises(EmbeddingAPIError, match="after 3 attempts"):
        make(handler, max_attempts=3).embed(["a"], EmbedKind.DOCUMENT)
    assert len(handler.requests) == 3


# --- construction ------------------------------------------------------------


def test_a_missing_key_names_the_setting(monkeypatch):
    monkeypatch.delenv("TAXVERITY_JINA_API_KEY", raising=False)
    with pytest.raises(MissingSettingError, match="TAXVERITY_JINA_API_KEY"):
        JinaAPIEmbedder.from_settings(Settings(_env_file=None))


def test_an_empty_key_is_refused():
    with pytest.raises(ValueError, match="api_key"):
        JinaAPIEmbedder("")


# --- live: costs a handful of tokens, skipped without a key ------------------

needs_key = pytest.mark.skipif(
    Settings().jina_api_key is None, reason="TAXVERITY_JINA_API_KEY not configured"
)


@needs_key
def test_the_live_api_embeds_both_kinds_at_1024():
    with JinaAPIEmbedder.from_settings(Settings()) as embedder:
        text = ["income from house property"]
        document = embedder.embed(text, EmbedKind.DOCUMENT)[0]
        query = embedder.embed(text, EmbedKind.QUERY)[0]
    assert len(document) == len(query) == 1024
    assert document != query
